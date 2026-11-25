"""Per-wine flavor profiles and "wines like this" similarity.

Every wine gets one vector built from what the database already knows:

  * aromas     (wine_aromas)  - weighted by how many of the wine's reviews mention
                                them (intensity) and by rarity (IDF), so 'petrol'
                                says more than 'fruit', and a one-off mention less
                                than a note every critic makes
  * aroma categories          - 'Black fruit', 'Citrus', ... (lets 'blackberry' and
                                'cassis' count as partly similar)
  * sensory    (wine_sensory) - body, acidity, tannin, sweetness, finish

Most wines have no tasting note. Those borrow the average profile of similar wines
(the most specific group with enough data). Style (Still / Sparkling / Dessert /
Fortified) is part of every group, so a Blanc de Blancs borrows from sparkling
Chardonnay, not still Chardonnay:

    color+style+grape+country+region -> ...+grape+country -> ...+grape -> color+style

Group averages are pulled toward the next broader group when they rest on few
wines (shrinkage), so a region with 4 wines does not get an extreme profile.

similar() mixes two kinds of match so neither dominates (LOCAL_SHARE, default 40%):
wines from the same region area or sharing a grape, and taste matches from other
regions and grapes. Each result's ``match`` column says which ("Same region:
Provence", "Shares grape: Grenache", "Similar taste").

Nothing in the flavors says "bubbles", so similar() ranks wines of another style
than the query below same-style wines that are nearly as similar (STYLE_PENALTY):
a sparkling rosé can still appear for a still rosé, after the close still matches.

Build (done by ``python -m app.review_analytics``), then use from the app:

    from app.flavor_profiles import FlavorProfiles
    fp = FlavorProfiles.load()
    fp.similar(12099, k=10)                                   # wines like this one
    vec, level = fp.profile_for(color="White", style="Still", variety="Sauvignon Blanc",
                                country="France", region="Sancerre")
    fp.similar(vector=vec, color="White", style="Still",      # label not in the database
               region="Sancerre", grapes="Sauvignon Blanc")

Try it from the terminal:
    python -m app.flavor_profiles 12099
    python -m app.flavor_profiles --color White --variety "Sauvignon Blanc" --region Sancerre
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import normalize

PROFILE_DIR = Path("artifacts") / "flavor_profiles"
FORMAT_VERSION = 3

SENSORY_COLS = ["body", "acidity", "tannin", "sweetness", "finish"]
# Relative weight of each block in the final vector (each block has length ~1).
WEIGHTS = {"aroma": 1.0, "category": 0.5, "sensory": 0.7}
MIN_OWN_AROMAS = 2  # a wine needs this many aromas to use its own profile
SHRINK = 10.0  # pseudo-wines pulling a group's average toward its parent group
MIN_GROUP = 3  # wines a group needs before others may borrow its profile
# Subtracted from the similarity of a wine whose style differs from the query's,
# for ranking only (the similarity shown is not changed).
STYLE_PENALTY = 0.15
# Share of similar() results reserved for wines from the same region or sharing a
# grape with the query; the rest are taste matches from other regions and grapes.
LOCAL_SHARE = 0.4
GRAPE_OVERLAP = 0.5  # min share of the two wines' combined grapes they must have in common
STILL = "Still"  # the database stores still table wine as style NULL
STYLES = ("Still", "Sparkling", "Dessert", "Fortified")

# Coarse -> fine. Each level's key columns.
LEVELS = [
    ("color", ["color", "style_group"]),
    ("variety", ["color", "style_group", "variety"]),
    ("variety+country", ["color", "style_group", "variety", "country"]),
    ("variety+region", ["color", "style_group", "variety", "country", "region"]),
]
# Second chain, for labels that name no grape (Champagne, Provence rosé, Chablis):
# only used by profile_for().
PLACE_LEVELS = [
    ("color", ["color", "style_group"]),
    ("country", ["color", "style_group", "country"]),
    ("region", ["color", "style_group", "country", "region"]),
]

META_COLS = [
    "wine_id",
    "title",
    "winery",
    "region",
    "country",
    "color",
    "style_group",
    "variety",
    "grapes",
    "critic_score",
    "n_critic_reviews",
    "user_score",
    "n_user_ratings",
    "price_usd",
]
RESULT_COLS = [
    "wine_id",
    "title",
    "winery",
    "region",
    "country",
    "color",
    "style_group",
    "variety",
    "similarity",
    "match",
    "shared_flavors",
    "critic_score",
    "user_score",
    "price_usd",
    "profile_source",
]


# ---- helpers ------------------------------------------------------------------
def _norm_text(value) -> str:
    """Lowercase, accent-free, punctuation-free text (for keys, not display)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    t = unicodedata.normalize("NFKD", str(value))
    t = "".join(c for c in t if not unicodedata.combining(c)).lower()
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


def _split_grapes(text) -> list:
    """'Cabernet Sauvignon-Merlot' / 'Grenache, Syrah' / 'A|B' -> normalized grape names."""
    whole = _norm_text(text)
    if not whole:
        return []
    parts = re.split(r"\s*(?:\||,|/|&|\+|-|\band\b)\s*", str(text))
    return [whole] + [p for p in map(_norm_text, parts) if p and p != whole]


def _group_key(values: Iterable) -> Optional[str]:
    """Key for one group, or None when any part is unknown."""
    parts = [_norm_text(v) for v in values]
    return None if any(p == "" for p in parts) else "\x1f".join(parts)


def _level_codes(meta: pd.DataFrame, cols: list) -> tuple[np.ndarray, pd.Index]:
    """Group number of every wine at one level (-1 = a key part is missing)."""
    normed = [meta[c].map(_norm_text) for c in cols]
    keys = normed[0]
    for part in normed[1:]:
        keys = keys + "\x1f" + part
    missing = np.zeros(len(meta), bool)
    for part in normed:
        missing |= (part == "").to_numpy()
    codes, uniques = pd.factorize(keys.where(~missing))
    return codes.astype(np.int64), uniques


def _hierarchical_means(values, observed, level_codes, k=SHRINK, min_n=MIN_GROUP):
    """Fill unobserved cells with the average of the most specific group that has data.

    values, observed: (n_wines, d). level_codes: one code array per level, coarse->fine.
    Group averages are shrunk toward the parent group: (sum + k*parent) / (count + k).

    Returns (filled, level, group_means) where level is -1 for observed cells,
    0 for the global average, i for LEVELS[i-1]; group_means[i] = (means, counts).
    """
    n, d = values.shape
    obs_f = observed.astype(np.float32)
    vals = np.where(observed, values, 0).astype(np.float32)
    totals = obs_f.sum(axis=0)
    global_mean = np.divide(
        vals.sum(axis=0), totals, out=np.zeros(d, np.float32), where=totals > 0
    )

    parent = np.tile(global_mean, (n, 1))  # each wine's current group average
    filled = parent.copy()
    level = np.zeros((n, d), np.int8)
    group_means = []

    for li, codes in enumerate(level_codes, start=1):
        rows = np.flatnonzero(codes >= 0)
        member = codes[rows]
        n_groups = int(member.max()) + 1 if len(rows) else 0
        G = sparse.csr_matrix(
            (np.ones(len(rows), np.float32), (member, rows)), shape=(n_groups, n)
        )
        sums = np.asarray(G @ vals)
        counts = np.asarray(G @ obs_f)
        # All members of a group share a parent group: take it from the first member.
        first = pd.Series(rows).groupby(member).first().to_numpy()
        means = (sums + k * parent[first]) / (counts + k)
        group_means.append((means.astype(np.float32), counts))

        cand = means[member]
        use = ~observed[rows] & (counts[member] >= min_n)
        sub = filled[rows]
        sub[use] = cand[use]
        filled[rows] = sub
        sub_level = level[rows]
        sub_level[use] = li
        level[rows] = sub_level
        parent[rows] = cand

    filled[observed] = values[observed]
    level[observed] = -1
    return filled, level, (global_mean, group_means)


# ---- the profiles -------------------------------------------------------------
@dataclass
class FlavorProfiles:
    meta: pd.DataFrame  # one row per wine; row i <-> vectors[i]
    vectors: np.ndarray  # (n_wines, n_features) float32, rows L2-normalized
    feature_names: list
    groups: pd.DataFrame  # level, key, n_wines; row i <-> group_vectors[i]
    group_vectors: np.ndarray
    info: dict = field(default_factory=dict)
    # Only kept in memory after a build (used for the lift table), not saved.
    aroma_presence: Optional[sparse.csr_matrix] = None

    def __post_init__(self):
        self._row = pd.Series(np.arange(len(self.meta)), index=self.meta["wine_id"])
        self._n_aroma = sum(f.startswith("aroma:") for f in self.feature_names)
        self._aroma_names = np.array(
            [f.split(":", 1)[1] for f in self.feature_names[: self._n_aroma]]
        )
        self._group_row = {
            (lv, key): i for i, (lv, key) in enumerate(zip(self.groups.level, self.groups.key))
        }
        # Filter columns prepared once, so similar() only does array comparisons.
        for col in ("color", "country", "region"):  # normalized at build time
            if f"{col}_key" not in self.meta:
                self.meta[f"{col}_key"] = self.meta[col].map(_norm_text)
        self._color = self.meta["color_key"].to_numpy()
        self._country = self.meta["country_key"].to_numpy()
        self._style = self.meta["style_group"].map(_norm_text).to_numpy()
        self._own = (self.meta["profile_source"] == "own").to_numpy()
        self._price = self.meta["price_usd"].to_numpy(np.float64)
        self._n_ratings = self.meta["n_user_ratings"].to_numpy()
        self._key = self.meta["dedupe_key"].to_numpy()
        self._region = self.meta["region_key"].to_numpy()
        self._region_keys = sorted(set(self._region) - {""})
        self._region_cache: dict = {}
        # Wine x grape incidence matrix, for "shares a grape" in similar().
        grape_lists = self.meta["grapes"].fillna("").str.split("|")
        self._grape_id, self._grape_name = {}, []
        g_rows, g_cols = [], []
        for i, names in enumerate(grape_lists):
            for name in names:
                key = _norm_text(name)
                if not key:
                    continue
                if key not in self._grape_id:
                    self._grape_id[key] = len(self._grape_name)
                    self._grape_name.append(name)
                g_rows.append(i)
                g_cols.append(self._grape_id[key])
        self._grapes = sparse.csr_matrix(
            (np.ones(len(g_rows), np.int8), (g_rows, g_cols)),
            shape=(len(self.meta), len(self._grape_name)),
        )
        self._grapes_by_col = self._grapes.tocsc()
        self._n_grapes = np.diff(self._grapes.indptr)  # grapes per wine
        pairs = pd.DataFrame(
            {"region": self.meta["region_key"], "country": self._country}
        )
        pairs = pairs[(pairs.region != "") & (pairs.country != "")]
        most_common = pairs.value_counts().reset_index().drop_duplicates("region")
        self._region_country = dict(zip(most_common.region, most_common.country))

    # ---- lookups
    def row(self, wine_id: int) -> int:
        try:
            return int(self._row[wine_id])
        except KeyError:
            raise KeyError(f"wine_id {wine_id} has no flavor profile") from None

    def vector(self, wine_id: int) -> np.ndarray:
        return self.vectors[self.row(wine_id)]

    def profile_for(
        self,
        color: Optional[str] = None,
        variety: Optional[str] = None,
        country: Optional[str] = None,
        region: Optional[str] = None,
        style: Optional[str] = None,
    ) -> tuple[np.ndarray, str]:
        """Typical profile for a wine that is not in the database (e.g. a scanned label).

        ``style``: Still / Sparkling / Dessert / Fortified; unknown means Still.
        Uses the most specific group with at least MIN_GROUP profiled wines.
        Returns (vector, level name used).
        """
        if region and not country:
            # Labels often name the region but not the country: use the region's
            # most common country in the database.
            country = self._region_country.get(_norm_text(region))
        given = {
            "color": color,
            "style_group": style or STILL,
            "variety": variety,
            "country": country,
            "region": region,
        }
        cols_of = dict(LEVELS) | dict(PLACE_LEVELS)
        # Most specific first; the place-only groups cover labels without a grape
        # and grapes too rare in that region.
        order = ["variety+region", "region", "variety+country", "variety", "country", "color"]
        for name in order:
            key = _group_key(given[c] for c in cols_of[name])
            i = self._group_row.get((name, key)) if key else None
            if i is not None and self.groups.n_wines.iat[i] >= MIN_GROUP:
                return self.group_vectors[i], name
        return self.group_vectors[self._group_row[("global", "")]], "global"

    def explain(self, wine_id: int, top: int = 10) -> pd.DataFrame:
        """The features that weigh most in a wine's vector."""
        v = self.vector(wine_id)
        order = np.argsort(-np.abs(v))[:top]
        return pd.DataFrame(
            {"feature": np.array(self.feature_names)[order], "weight": v[order].round(3)}
        )

    # ---- similarity
    def similar(
        self,
        wine_id: Optional[int] = None,
        *,
        vector: Optional[np.ndarray] = None,
        k: int = 10,
        color: Optional[str] = None,
        same_color: bool = True,
        style: Optional[str] = None,
        style_penalty: float = STYLE_PENALTY,
        same_style_only: bool = False,
        own_profiles_only: bool = True,
        local_share: float = LOCAL_SHARE,
        region: Optional[str] = None,
        grapes: Optional[str] = None,
        country: Optional[str] = None,
        max_price: Optional[float] = None,
        min_user_ratings: int = 0,
        exclude_wine_ids: Iterable[int] = (),
    ) -> pd.DataFrame:
        """Most similar wines to ``wine_id`` (or to a ``vector`` from profile_for()).

        The list mixes two kinds of match, alternating from the top:
          * local: same region, or sharing a grape, as the query
            (``local_share`` of the slots, default 40%);
          * taste: the closest flavor/structure matches from other regions and grapes.
        Both are ranked by flavor similarity. The ``match`` column says which kind
        each result is ("Same region: Provence", "Shares grape: Grenache",
        "Similar taste"). If one kind runs short, the other fills its slots.
        local_share=0 ranks every wine by taste alone; 1 keeps only local wines.

        region, grapes      what "local" means for a ``vector`` query (a scanned
                            label): e.g. region="Provence", grapes="Grenache, Cinsault".
                            Taken from the wine itself for a ``wine_id`` query.
        same_color          only wines of the query wine's color (or ``color``)
        style               the query's style (taken from ``wine_id`` when given).
                            Wines of another style rank below same-style wines,
                            as if ``style_penalty`` less similar; the
                            ``similarity`` column still shows the real value.
        same_style_only     drop other styles entirely instead
        own_profiles_only   only wines with their own tasting data; borrowed
                            profiles are identical within a group, so they would
                            crowd the results
        country, max_price, min_user_ratings   optional filters
        Duplicate listings of the same wine (same winery + name) are collapsed.
        """
        meta = self.meta
        if (wine_id is None) == (vector is None):
            raise ValueError("pass exactly one of wine_id or vector")
        if wine_id is not None:
            qi = self.row(wine_id)
            q = self.vectors[qi]
            if same_color and color is None:
                color = meta["color"].iat[qi]
            if style is None:
                style = meta["style_group"].iat[qi]
            query_key = meta["dedupe_key"].iat[qi]
            region_key = _norm_text(region) or self._region[qi]
            region_country = self._country[qi] if not region else ""
            grape_ids = (
                self._grape_ids(grapes) if grapes else list(self._grapes[qi].indices)
            )
        else:
            q = np.asarray(vector, np.float32)
            query_key = None
            region_key = _norm_text(region)
            region_country = self._region_country.get(region_key, "") if region_key else ""
            grape_ids = self._grape_ids(grapes)

        mask = self._own.copy() if own_profiles_only else np.ones(len(meta), bool)
        color_key = _norm_text(color)
        if color_key:
            mask &= self._color == color_key
        if country:
            mask &= self._country == _norm_text(country)
        style_key = _norm_text(style)
        other_style = (self._style != style_key) if style_key else np.zeros(len(meta), bool)
        if same_style_only:
            mask &= ~other_style
        if max_price is not None:
            mask &= self._price <= max_price  # unknown price (NaN) is excluded
        if min_user_ratings:
            mask &= self._n_ratings >= min_user_ratings
        if query_key is not None:
            mask &= self._key != query_key
        excluded = list(exclude_wine_ids) + ([wine_id] if wine_id is not None else [])
        for wid in excluded:
            if wid in self._row.index:
                mask[self._row[wid]] = False

        if not mask.any():
            return pd.DataFrame(columns=RESULT_COLS)
        # Scoring every wine is one fast matrix-vector product; filter afterwards.
        sims = np.where(mask, self.vectors @ q, -np.inf)
        rank_score = sims - style_penalty * other_style  # ranking only

        # Local = same region area (and country), or shares at least one grape.
        n = len(meta)
        same_region = np.zeros(n, bool)
        if region_key:
            same_region = np.isin(self._region, self._related_regions(region_key))
            if region_country:
                same_region &= self._country == region_country
        # "Shares grapes" = at least half of the two wines' combined grapes in common
        # (Jaccard >= GRAPE_OVERLAP), so one shared component of a 5-grape blend
        # does not make every Cabernet "local" to Tignanello.
        shares_grape = np.zeros(n, bool)
        if grape_ids:
            common = np.asarray(self._grapes_by_col[:, grape_ids].sum(axis=1)).ravel()
            union = self._n_grapes + len(grape_ids) - common
            shares_grape = (common > 0) & (common >= GRAPE_OVERLAP * union)
        local = same_region | shares_grape

        def pool(keep):
            return np.where(keep, rank_score, -np.inf)

        if local_share <= 0 or not (mask & local).any():
            picked = self._ranked(rank_score, k)
        else:
            n_local = k if local_share >= 1 else min(k, max(1, round(k * local_share)))
            # Local slots alternate region and grape matches, so a common grape
            # (Grenache, Chardonnay) does not crowd out the region, or the reverse.
            local_list = self._interleave(
                self._ranked(pool(same_region), n_local),
                self._ranked(pool(shares_grape & ~same_region), n_local),
                -(-n_local // 2),
                n_local,
                rank_score,
            )
            taste_list = self._ranked(pool(~local), k) if n_local < k else []
            picked = self._interleave(local_list, taste_list, n_local, k, rank_score)
            # Interleaving can lift an other-style wine (a sparkling rosé among the
            # local picks) above a same-style wine that is nearly as similar:
            # move it down past every such wine.
            picked = self._style_order(picked, sims, other_style, style_penalty)

        rows = np.array(picked, dtype=np.int64)
        out = meta.iloc[rows].copy()
        out["similarity"] = sims[rows].round(3)
        out["match"] = [
            self._match_label(r, same_region[r], shares_grape[r], grape_ids) for r in rows
        ]
        # Aromas both wines share, strongest first (for "because it also has ...").
        qa = q[: self._n_aroma]
        shared = self.vectors[rows, : self._n_aroma] * qa
        out["shared_flavors"] = [
            ", ".join(self._aroma_names[np.argsort(-s)[: (s > 0).sum()][:3]]) for s in shared
        ]
        return out[RESULT_COLS].reset_index(drop=True)

    # ---- similarity helpers
    def _grape_ids(self, grapes: Optional[str]) -> list:
        """Known grape ids in a label's variety text ('Grenache-Cinsault')."""
        ids = [self._grape_id.get(g) for g in _split_grapes(grapes)]
        return sorted({i for i in ids if i is not None})

    def _related_regions(self, region_key: str) -> list:
        """The region and regions whose names contain it, or that it contains, as
        whole words: 'provence' <-> 'cotes de provence', 'sonoma' <-> 'sonoma coast'.
        (The database stores regions flat, without parent regions.)"""
        cached = self._region_cache.get(region_key)
        if cached is None:
            q = f" {region_key} "
            cached = [
                r for r in self._region_keys
                if r == region_key
                or q in f" {r} "
                or (len(r) >= 4 and f" {r} " in q)
            ]
            self._region_cache[region_key] = cached
        return cached

    def _ranked(self, scores: np.ndarray, n: int) -> list:
        """Rows with the n best finite scores, best first, one per dedupe key."""
        valid = int(np.isfinite(scores).sum())
        if valid == 0 or n <= 0:
            return []
        take = min(valid, n * 5)  # extra, to survive de-duplication
        top = np.argpartition(-scores, take - 1)[:take]
        top = top[np.argsort(-scores[top])]
        picked, seen = [], set()
        for t in top:
            if self._key[t] in seen:
                continue
            seen.add(self._key[t])
            picked.append(int(t))
            if len(picked) == n:
                break
        return picked

    def _interleave(self, local, taste, n_local, k, scores) -> list:
        """Fill k slots: n_local from ``local``, the rest from ``taste``, alternating
        so each kind keeps its share at every depth. A short list's unused slots go
        to the other list."""
        n_local = min(n_local, len(local))
        n_taste = min(k - n_local, len(taste))
        n_local = min(len(local), k - n_taste)  # give spare slots back to local
        queues = {"local": local[:n_local], "taste": taste[:n_taste]}
        quota = {"local": n_local, "taste": n_taste}
        used = {"local": 0, "taste": 0}
        out, seen = [], set()
        while any(used[p] < quota[p] for p in queues):
            # The kind furthest behind its share goes next; ties -> better score.
            open_kinds = [p for p in queues if used[p] < quota[p]]
            kind = min(
                open_kinds,
                key=lambda p: (used[p] / quota[p], -scores[queues[p][used[p]]]),
            )
            row = queues[kind][used[kind]]
            used[kind] += 1
            if self._key[row] not in seen:  # same wine listed in both kinds
                seen.add(self._key[row])
                out.append(row)
        return out

    @staticmethod
    def _style_order(picked, sims, other_style, penalty) -> list:
        """No other-style wine before a same-style wine within ``penalty`` of its similarity."""
        order = list(picked)
        i = 0
        while i < len(order):
            row = order[i]
            if other_style[row]:
                last = max(
                    (j for j in range(i + 1, len(order))
                     if not other_style[order[j]] and sims[order[j]] >= sims[row] - penalty),
                    default=None,
                )
                if last is not None:
                    order.insert(last, order.pop(i))  # lands right after that wine
                    continue  # re-check the wine that moved into position i
            i += 1
        return order

    def _match_label(self, row, same_region, shares_grape, grape_ids) -> str:
        shared = ""
        if shares_grape:
            hits = set(self._grapes[row].indices) & set(grape_ids)
            shared = ", ".join(sorted(self._grape_name[i] for i in hits)[:2])
        if same_region and shares_grape:
            return f"Same region & grape: {self.meta['region'].iat[row]}, {shared}"
        if same_region:
            return f"Same region: {self.meta['region'].iat[row]}"
        if shares_grape:
            return f"Shares grape: {shared}"
        return "Similar taste"

    # ---- persistence
    def save(self, out_dir: Path = PROFILE_DIR) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "vectors.npy", self.vectors)
        np.save(out_dir / "group_vectors.npy", self.group_vectors)
        self.meta.to_parquet(out_dir / "meta.parquet", index=False)
        self.groups.to_parquet(out_dir / "groups.parquet", index=False)
        (out_dir / "info.json").write_text(
            json.dumps({**self.info, "feature_names": self.feature_names}, indent=1),
            encoding="utf-8",
        )
        return out_dir

    @classmethod
    def load(cls, in_dir: Path = PROFILE_DIR) -> "FlavorProfiles":
        in_dir = Path(in_dir)
        if not (in_dir / "info.json").exists():
            raise FileNotFoundError(
                f"No flavor profiles in {in_dir}; build them with `python -m app.review_analytics`"
            )
        info = json.loads((in_dir / "info.json").read_text(encoding="utf-8"))
        if info.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"{in_dir} was built by another version; rebuild it")
        feature_names = info.pop("feature_names")
        return cls(
            meta=pd.read_parquet(in_dir / "meta.parquet"),
            vectors=np.load(in_dir / "vectors.npy"),
            feature_names=feature_names,
            groups=pd.read_parquet(in_dir / "groups.parquet"),
            group_vectors=np.load(in_dir / "group_vectors.npy"),
            info=info,
        )


# ---- build --------------------------------------------------------------------
def build_profiles(
    features: Optional[pd.DataFrame] = None,
    aromas: Optional[pd.DataFrame] = None,
    min_own_aromas: int = MIN_OWN_AROMAS,
) -> FlavorProfiles:
    """Build profiles for every wine from repository.get_wine_features() / get_aromas()."""
    if features is None or aromas is None:
        from database import repository as repo

        features = repo.get_wine_features() if features is None else features
        aromas = repo.get_aromas() if aromas is None else aromas

    meta = features.reset_index(drop=True)
    meta["style_group"] = meta["style"].fillna(STILL)  # NULL style = still table wine
    meta["grapes"] = meta["grapes"].fillna(meta["varietal"])  # 'A|B', for "shares a grape"
    n = len(meta)
    row_of = pd.Series(np.arange(n), index=meta["wine_id"])
    aromas = aromas[aromas["wine_id"].isin(row_of.index)]
    rows = row_of[aromas["wine_id"]].to_numpy()
    # Share of the wine's reviews mentioning the aroma; NULL (no reviews) = 1.
    weight = aromas["intensity"].fillna(1.0).clip(0, 1).to_numpy(np.float32)

    # Aroma presence and weights (wines x aroma names), category weights (wines x categories).
    a_codes, aroma_names = pd.factorize(aromas["name"], sort=True)
    A = sparse.csr_matrix(
        (np.ones(len(rows), np.float32), (rows, a_codes)), shape=(n, len(aroma_names))
    )
    A.data[:] = 1.0  # presence: for IDF, counts and the lift table
    W = sparse.csr_matrix((weight, (rows, a_codes)), shape=A.shape)
    has_cat = aromas["category"].notna().to_numpy()
    c_codes, cat_names = pd.factorize(aromas["category"][has_cat], sort=True)
    C = sparse.csr_matrix(
        (weight[has_cat], (rows[has_cat], c_codes)),
        shape=(n, len(cat_names)),
    )

    n_aromas = np.asarray(A.sum(axis=1)).ravel().astype(int)
    own = n_aromas >= min_own_aromas
    n_own = int(own.sum())

    # Rarity weight (IDF) from wines with their own profile.
    doc_freq = np.asarray(A[own].sum(axis=0)).ravel()
    idf = (np.log((1 + n_own) / (1 + doc_freq)) + 1).astype(np.float32)
    A_w = normalize(W.multiply(idf).tocsr())
    C_n = normalize(C)
    P = sparse.hstack(
        [WEIGHTS["aroma"] * A_w, WEIGHTS["category"] * C_n], format="csr"
    ).toarray()

    level_codes, level_keys = [], []
    for _, cols in LEVELS:
        codes, keys = _level_codes(meta, cols)
        level_codes.append(codes)
        level_keys.append(keys)

    # 1) aroma part: own aromas, else the group's average.
    P_obs = np.repeat(own[:, None], P.shape[1], axis=1)
    P, P_level, _ = _hierarchical_means(P, P_obs, level_codes)
    level_names = np.array(["global"] + [name for name, _ in LEVELS])
    profile_source = np.where(own, "own", level_names[P_level[:, 0].clip(min=0)])

    # 2) sensory part: each value the wine has, else its group's average; then z-scored.
    S = meta[SENSORY_COLS].to_numpy(np.float32)
    S_obs = ~np.isnan(S)
    S, _, _ = _hierarchical_means(np.nan_to_num(S), S_obs, level_codes)
    mu = np.array([np.nanmean(meta[c]) for c in SENSORY_COLS], np.float32)
    sd = np.array([np.nanstd(meta[c]) or 1.0 for c in SENSORY_COLS], np.float32)
    S = (S - mu) / sd * (WEIGHTS["sensory"] / np.sqrt(len(SENSORY_COLS)))

    vectors = normalize(np.hstack([P, S])).astype(np.float32)
    feature_names = (
        [f"aroma:{a}" for a in aroma_names]
        + [f"category:{c}" for c in cat_names]
        + [f"sensory:{s}" for s in SENSORY_COLS]
    )

    # Typical profile per group (from wines with their own data), for profile_for().
    X_obs = np.repeat(own[:, None], vectors.shape[1], axis=1)
    _, _, (global_vec, group_means) = _hierarchical_means(vectors, X_obs, level_codes)
    g_frames, g_vecs = [pd.DataFrame({"level": ["global"], "key": [""], "n_wines": [n_own]})], [
        global_vec[None, :]
    ]
    place_codes, place_keys = zip(*(_level_codes(meta, cols) for _, cols in PLACE_LEVELS))
    _, _, (_, place_means) = _hierarchical_means(vectors, X_obs, list(place_codes))
    chains = list(zip(LEVELS, level_keys, group_means))
    # PLACE_LEVELS[0] ("color") is the same grouping as LEVELS[0]: skip the copy.
    chains += list(zip(PLACE_LEVELS, place_keys, place_means))[1:]
    for (name, _), keys, (means, counts) in chains:
        n_g = counts[:, 0].astype(int)
        keep = n_g > 0
        g_frames.append(
            pd.DataFrame({"level": name, "key": np.asarray(keys)[keep], "n_wines": n_g[keep]})
        )
        g_vecs.append(means[keep])
    groups = pd.concat(g_frames, ignore_index=True)
    group_vectors = normalize(np.vstack(g_vecs)).astype(np.float32)

    # Top flavor words per wine: for own profiles the most mentioned (and rarest)
    # first, for borrowed profiles the most typical of the group.
    aroma_block = P[:, : len(aroma_names)]
    top = np.argsort(-aroma_block, axis=1)[:, :8]
    names = np.asarray(aroma_names)
    flavors = [
        names[t[aroma_block[i, t] > 1e-6]].tolist() for i, t in enumerate(top)
    ]

    out_meta = meta[META_COLS].copy()
    out_meta["profile_source"] = profile_source
    out_meta["n_aromas"] = n_aromas
    out_meta["flavors"] = flavors
    for col in ("color", "country", "region"):
        out_meta[f"{col}_key"] = out_meta[col].map(_norm_text)
    out_meta["dedupe_key"] = (
        meta["winery"].map(_norm_text) + "|" + meta["name"].map(_norm_text)
    )

    info = {
        "format_version": FORMAT_VERSION,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_wines": n,
        "n_own_profiles": n_own,
        "profile_sources": pd.Series(profile_source).value_counts().to_dict(),
        "weights": WEIGHTS,
        "min_own_aromas": min_own_aromas,
        "shrink": SHRINK,
        "min_group": MIN_GROUP,
        "style_penalty": STYLE_PENALTY,
    }
    return FlavorProfiles(
        meta=out_meta,
        vectors=vectors,
        feature_names=feature_names,
        groups=groups,
        group_vectors=group_vectors,
        info=info,
        aroma_presence=A,
    )


# ---- CLI ----------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description="Find wines with a similar flavor profile.")
    parser.add_argument("wine_id", nargs="?", type=int, help="wine to find look-alikes for")
    parser.add_argument("--color", help="for a wine not in the database, e.g. White")
    parser.add_argument("--variety", help='e.g. "Sauvignon Blanc"')
    parser.add_argument("--country")
    parser.add_argument("--region")
    parser.add_argument("--style", choices=STYLES, help="for a wine not in the database (default Still)")
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument(
        "--local-share", type=float, default=LOCAL_SHARE,
        help=f"share of results from the same region / grape (default {LOCAL_SHARE})",
    )
    parser.add_argument("--max-price", type=float)
    parser.add_argument("--profiles", type=Path, default=PROFILE_DIR)
    args = parser.parse_args(argv)

    fp = FlavorProfiles.load(args.profiles)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 45)

    if args.wine_id is not None:
        w = fp.meta.iloc[fp.row(args.wine_id)]
        print(f"{w.title} ({w.variety}, {w.region}, {w.style_group}) - profile: {w.profile_source}")
        print("Flavors:", ", ".join(w.flavors), "\n")
        result = fp.similar(
            args.wine_id, k=args.k, max_price=args.max_price, local_share=args.local_share
        )
    elif args.variety or args.color:
        style = args.style or STILL
        vec, level = fp.profile_for(args.color, args.variety, args.country, args.region, style)
        print(f"Typical profile from level: {level}\n")
        result = fp.similar(
            vector=vec, k=args.k, color=args.color, style=style, max_price=args.max_price,
            region=args.region, grapes=args.variety, local_share=args.local_share,
        )
    else:
        parser.error("give a wine_id, or --variety/--color (with optional --country/--region)")

    cols = ["title", "variety", "region", "similarity", "match", "shared_flavors", "price_usd"]
    print(result[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
