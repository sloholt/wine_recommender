"""Step 7b: do the wine vectors capture taste? Compare representations on the database.

The notebook's test, rebuilt on the wine database (one row per wine, not per review):
for each test wine, take its 10 nearest wines from other wineries and count how
often they have the same grape. Grape names are hidden from review text first, so
a method cannot just read "Chardonnay".

Two scores per method:
  same_grape_at_10                 neighbors from any other winery (the notebook's test)
  same_grape_at_10_other_regions   neighbors from other wineries AND other regions:
                                   finding the grape's taste, not its place ("Sancerre"
                                   in a review nearly gives away Sauvignon Blanc)

Fair comparison: every method searches the same pool - wines with review text and
their own tasting profile (borrowed profiles come from the grape group, so they
would leak the answer).

Run from the project folder (after python -m app.review_analytics):
    python -m app.evaluate_vectors
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, normalize

from app.flavor_profiles import PROFILE_DIR, FlavorProfiles
from app.review_analytics import STOP_WORDS
from database import repository as repo

ARTIFACT_DIR = Path("artifacts")
EVAL_N, K_EVAL, MIN_VARIETY, SEED = 2000, 10, 50, 42
GENERIC = {"red", "white", "blend", "rose", "sparkling", "de", "la", "du", "di", "del",
           "valley", "county", "coast", "hills", "river", "mountain", "district", "vino"}

# Chart colors (reference palette): method bars vs. baselines, ink, surface.
VECTOR_COLOR, BASELINE_COLOR = "#2a78d6", "#c3c2b7"
INK, INK_2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#898781", "#fcfcfb"


def _ascii(text: str) -> str:
    t = unicodedata.normalize("NFKD", text or "")
    return "".join(c for c in t if not unicodedata.combining(c)).lower()


def _words(values) -> set:
    """Distinctive words (> 3 letters, not generic) of names such as grapes or regions."""
    out = set()
    for v in values:
        if isinstance(v, str):
            out.update(w for w in re.findall(r"[a-z]+", _ascii(v)) if len(w) > 3 and w not in GENERIC)
    return out


def _masked(texts: pd.Series, hide: set) -> list:
    word = re.compile(r"[a-z]+")
    return [word.sub(lambda m: " " if m.group() in hide else m.group(), _ascii(t)) for t in texts]


def _tfidf(docs: list) -> sparse.csr_matrix:
    return TfidfVectorizer(
        stop_words=STOP_WORDS, ngram_range=(1, 2), min_df=3, max_df=0.5,
        max_features=30000, sublinear_tf=True, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
    ).fit_transform(docs)


def _hstack_normalized(*blocks) -> sparse.csr_matrix:
    """Equal-weight combination: each block L2-normalized, then the whole row."""
    parts = [normalize(sparse.csr_matrix(b)) for b in blocks]
    return normalize(sparse.hstack(parts).tocsr())


def _vectors_without_grape_fill(wine_ids: pd.Series) -> np.ndarray:
    """Flavor vectors whose only fallback level is color + style (no grape, no region)."""
    import app.flavor_profiles as fpm

    saved = fpm.LEVELS
    fpm.LEVELS = saved[:1]  # [("color", ["color", "style_group"])]
    try:
        built = fpm.build_profiles()
    finally:
        fpm.LEVELS = saved
    row_of = pd.Series(np.arange(len(built.meta)), index=built.meta["wine_id"])
    return built.vectors[row_of[wine_ids].to_numpy()]


def same_grape_at_k(X, eval_rows, variety, winery, region, k=K_EVAL, other_regions=False):
    hits = []
    for start in range(0, len(eval_rows), 250):
        batch = eval_rows[start:start + 250]
        S = X[batch] @ X.T
        S = S.toarray() if sparse.issparse(S) else np.asarray(S)
        for row, sims in zip(batch, S):
            sims = sims.astype(np.float64)
            sims[winery == winery[row]] = -np.inf
            if other_regions:
                sims[region == region[row]] = -np.inf
            nn = np.argpartition(-sims, k)[:k]
            hits.append(np.mean(variety[nn] == variety[row]))
    return float(np.mean(hits))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out-dir", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--profiles", type=Path, default=PROFILE_DIR)
    args = parser.parse_args(argv)

    fp = FlavorProfiles.load(args.profiles)
    reviews = repo.get_reviews()
    text = reviews.groupby("wine_id")["review"].apply(lambda s: " ".join(s.dropna()))

    # Pool: wines with review text and their own tasting profile.
    meta = fp.meta.reset_index().rename(columns={"index": "vec_row"})
    pool = meta[(meta["profile_source"] == "own") & meta["wine_id"].isin(text.index)].reset_index(drop=True)
    pool["review"] = pool["wine_id"].map(text)
    print(f"Pool: {len(pool):,} wines with review text and their own tasting profile")

    counts = pool["variety"].value_counts()
    common = counts[counts >= MIN_VARIETY].index
    rng = np.random.default_rng(SEED)
    candidates = np.flatnonzero(pool["variety"].isin(common).to_numpy())
    eval_rows = rng.choice(candidates, size=min(EVAL_N, len(candidates)), replace=False)
    print(f"Test wines: {len(eval_rows):,} from {len(common)} grapes with >= {MIN_VARIETY} wines")

    variety = pool["variety"].to_numpy()
    winery = pool["winery"].map(_ascii).to_numpy()
    region = pool["region_key"].replace("", np.nan).fillna(pool["wine_id"].astype(str)).to_numpy()

    # Words to hide from review text.
    grape_words = _words(pool["variety"]) | _words(
        g for gs in pool["grapes"].dropna() for g in gs.split("|")
    )
    flavor_words = _words(f.split(":", 1)[1] for f in fp.feature_names if f.startswith("aroma:"))
    place_words = (_words(meta["region"]) | _words(meta["country"])) - flavor_words - grape_words
    print("Fitting review TF-IDF (grape names hidden, then grape + place names hidden)...")
    X_rev = _tfidf(_masked(pool["review"], grape_words))
    X_rev_noplace = _tfidf(_masked(pool["review"], grape_words | place_words))

    # Metadata without the grape: country, region, price, critic score.
    num = pd.DataFrame({"price": np.log1p(pool["price_usd"]), "score": pool["critic_score"]})
    num = num.fillna(num.median())
    X_meta = _hstack_normalized(
        OneHotEncoder(handle_unknown="ignore", min_frequency=5).fit_transform(
            pool[["country", "region"]].fillna("unknown")),
        StandardScaler().fit_transform(num) * 0.5,
    )
    # The saved vectors fill a wine's missing sensory values (e.g. tannin) from its
    # grape group: fine for recommending, but it leaks the answer here. Rebuild
    # with missing values filled from the color/style average only.
    print("Rebuilding flavor vectors without grape-group fill (fair test)...")
    X_flavor = _vectors_without_grape_fill(pool["wine_id"])
    X_flavor_saved = fp.vectors[pool["vec_row"].to_numpy()]

    methods = [
        ("Random pick (baseline)", None, "baseline"),
        ("Metadata: country, region, price, score", X_meta, "baseline"),
        ("Review words (grape names hidden)", X_rev, "vector"),
        ("Review words (grape + place names hidden)", X_rev_noplace, "vector"),
        ("Review words + metadata", _hstack_normalized(X_rev, X_meta), "vector"),
        ("Flavor profiles (this project)", X_flavor, "vector"),
        ("Flavor profiles + review words", _hstack_normalized(X_flavor, X_rev_noplace), "vector"),
        # For reference only: the saved vectors, whose sensory fill knows the grape.
        ("Flavor profiles as saved (leaks grape)", X_flavor_saved, "reference"),
    ]

    share = pool["variety"].value_counts(normalize=True)
    rows = []
    for name, X, kind in methods:
        print(f"  scoring: {name}", flush=True)
        if X is None:  # chance of a random pool wine having the test wine's grape
            score = other = float(share[variety[eval_rows]].mean())
        else:
            score = same_grape_at_k(X, eval_rows, variety, winery, region)
            other = same_grape_at_k(X, eval_rows, variety, winery, region, other_regions=True)
        rows.append({"method": name, "kind": kind, "same_grape_at_10": round(score, 3),
                     "same_grape_at_10_other_regions": round(other, 3)})
    results = pd.DataFrame(rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results.drop(columns="kind").to_csv(args.out_dir / "eval_same_grape.csv", index=False)
    plot(results[results["kind"] != "reference"], args.out_dir / "eval_same_grape.png")
    pd.set_option("display.width", 160)
    print()
    print(results.drop(columns="kind").to_string(index=False))
    print(f"\nSaved {args.out_dir / 'eval_same_grape.csv'} and eval_same_grape.png", file=sys.stderr)
    return 0


def plot(results: pd.DataFrame, path: Path) -> None:
    """Two panels on one 0-100% scale: any other winery | other wineries and regions."""
    panels = [("same_grape_at_10", "Any other winery"),
              ("same_grape_at_10_other_regions", "Other wineries and other regions")]
    n = len(results)
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.42 * n + 1.6), sharey=True, facecolor=SURFACE)
    colors = [VECTOR_COLOR if k == "vector" else BASELINE_COLOR for k in results["kind"]]
    y = np.arange(n)
    for ax, (col, title) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        ax.barh(y, results[col], height=0.55, color=colors)
        for yi, v in zip(y, results[col]):
            ax.text(v + 0.012, yi, f"{v:.0%}", va="center", fontsize=9, color=INK_2)
        ax.set_xlim(0, 1)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.tick_params(colors=MUTED, labelsize=9, length=0)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(MUTED)
    axes[0].set_yticks(y, results["method"], color=INK)
    axes[0].invert_yaxis()
    fig.suptitle("How often a wine's 10 nearest neighbors share its grape (grape names hidden)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
