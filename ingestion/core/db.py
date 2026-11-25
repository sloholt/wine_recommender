"""Write normalized WineRecord objects into the wine SQLite database.

Design:
* get-or-create for countries, regions, wineries, wines and vintages, using
  in-memory lookup caches loaded once at startup (fast, no per-row SELECT storms).
* Wine identity = winery + name (accent/case/punctuation-insensitive), with
  fallbacks when the winery is unknown on one side:
    - record has a winery -> match winery + name; failing that, adopt an existing
      wine with the same name and NO winery (and set its winery);
    - record has no winery -> match the same name alone, but only if exactly one
      wine has it, and only if country / color do not conflict. If several wines
      share the name it is ambiguous, so a separate no-winery wine is created
      (a duplicate that can be merged later beats a wrong merge).
* Re-running an import is safe: existing rows are matched, NULL fields are filled
  in, and ratings / prices are updated instead of duplicated.
* If an insert fails, the current batch is rolled back and the caches reloaded,
  so the caches never point at rows that no longer exist.

* A record already loaded from the same source (same source + external_id) goes
  straight back to its wine.
* Fuzzy fallback (core/matching.py), tried only when nothing above matched:
  candidates are wines of any winery with the same name (any region; for a record
  with no winery, the longest known winery name its title starts with), from a
  different source, with no conflicting country / color / style / grapes.
  "Dog Point Sauvignon Blanc Marlborough Section 94" -> "Section 94 Sauvignon
  Blanc". Only a unique best candidate is accepted; each merge is recorded in
  ``self.fuzzy_matches`` for review.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from . import matching
from .colors import infer_colors
from .food import food_category
from .models import Price, Rating, Source, UserRating, WineRecord
from .naming import _KEEP, _PLACES
from .normalize import GRAPE_SYNONYMS, fold, normalize_user_rating, normalize_wine
from .tasting import _GRAPE_LIST

LOG = logging.getLogger(__name__)

REQUIRED_TABLES = {
    "countries",
    "regions",
    "region_aliases",
    "wineries",
    "wines",
    "wine_vintages",
    "grapes",
    "wine_grapes",
    "wine_sensory",
    "wine_aromas",
    "wine_ratings",
    "wine_prices",
    "wine_sources",
    "foods",
    "wine_pairings",
    "user_ratings",
}

SENSORY_FIELDS = (
    "body",
    "acidity",
    "structure",
    "finish",
    "balance",
    "tannin",
    "sweetness",
)


def _key(text: Optional[str]) -> str:
    """Dedup key; falls back to casefold when fold() would be empty."""
    return fold(text) or (text or "").casefold()


def _accent_count(text: str) -> int:
    return sum(ord(ch) > 127 for ch in text)


@dataclass
class IngestStats:
    created: int = 0  # new wines
    matched: int = 0  # wines that already existed (includes fuzzy)
    fuzzy: int = 0  # of those, matched by fuzzy name matching
    skipped: int = 0  # records dropped by the normalizer (no name)

    @property
    def total(self) -> int:
        return self.created + self.matched + self.skipped


@dataclass
class RatingStats:
    loaded: int = 0  # user ratings inserted or refreshed
    unknown_wine: int = 0  # wine id not loaded from this source yet
    invalid: int = 0  # dropped by the normalizer


class WineDB:
    def __init__(self, path, schema_path=None):
        """Open ``path``. If it has no tables yet, create them from ``schema_path``."""
        path = Path(path)
        if not path.exists() and not schema_path:
            raise FileNotFoundError(
                f"Database not found: {path} (pass schema_path to create it)"
            )
        path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")

        if schema_path and not self._tables():
            self.conn.executescript(Path(schema_path).read_text(encoding="utf-8"))
            self.conn.execute("PRAGMA foreign_keys = ON")

        missing = REQUIRED_TABLES - self._tables()
        if missing:
            raise ValueError(f"Database is missing tables: {sorted(missing)}")
        self._add_missing_columns()

        self.fuzzy_matches: List[dict] = []  # one entry per fuzzy merge, for review
        self._winery_created = False  # did the last get_winery() insert a row?

        self._load_caches()

    # ------------------------------------------------------------ lifecycle
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.conn.commit()
        else:
            self.conn.rollback()
        self.close()

    def commit(self):
        self.conn.commit()

    def close(self):
        self.conn.close()

    # Columns added to schema.sql after a database may have been created from an
    # older version: added in place so existing databases keep working.
    _LATER_COLUMNS = {
        "wines": {
            "blend_type": "TEXT",
            "color_inferred": "INTEGER NOT NULL DEFAULT 0",
        },
    }

    def _add_missing_columns(self):
        for table, columns in self._LATER_COLUMNS.items():
            have = {r["name"] for r in self.conn.execute(f"PRAGMA table_info({table})")}
            for column, decl in columns.items():
                if column not in have:
                    self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
                    LOG.info("Added column %s.%s", table, column)
        self.conn.execute("CREATE INDEX IF NOT EXISTS ix_wines_winery ON wines(winery_id)")
        self.conn.commit()

    def infer_colors(self, overrides_path=None) -> int:
        """Fill missing colors from grapes (see core/colors.py) and commit."""
        count = infer_colors(self.conn, overrides_path)
        self.conn.commit()
        return count

    def _tables(self) -> Set[str]:
        rows = self.conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        return {r["name"] for r in rows}

    # --------------------------------------------------------------- caches
    def _load_caches(self):
        c = self.conn
        self._countries: Dict[str, int] = {}
        self._country_by_iso: Dict[str, int] = {}
        self._country_has_iso: Set[int] = set()
        for r in c.execute("SELECT country_id, name, iso_code FROM countries"):
            self._countries[_key(r["name"])] = r["country_id"]
            if r["iso_code"]:
                self._country_by_iso[r["iso_code"].upper()] = r["country_id"]
                self._country_has_iso.add(r["country_id"])

        self._regions: Dict[Tuple[int, str], int] = {}
        self._region_country: Dict[int, int] = {}
        for r in c.execute(
            "SELECT region_id, country_id, parent_region_id, name FROM regions"
        ):
            self._region_country[r["region_id"]] = r["country_id"]
            if r["parent_region_id"] is None:
                self._regions.setdefault(
                    (r["country_id"], _key(r["name"])), r["region_id"]
                )

        self._aliases: Dict[str, List[int]] = {}
        for r in c.execute("SELECT region_id, alias FROM region_aliases"):
            self._aliases.setdefault(_key(r["alias"]), []).append(r["region_id"])

        # winery name key -> {region_id (0 = unknown): winery_id}
        self._wineries: Dict[str, Dict[int, int]] = {}
        # winery name key -> the spelling already stored ("Château d'Esclans"), so a
        # new row for another region of a known producer reuses it instead of the
        # record's spelling ("Chateau d'Esclans"). Prefers the accented spelling.
        self._winery_names: Dict[str, str] = {}
        for r in c.execute("SELECT winery_id, name, region_id FROM wineries"):
            key = _key(r["name"])
            self._wineries.setdefault(key, {})[r["region_id"] or 0] = r["winery_id"]
            known = self._winery_names.get(key)
            if known is None or _accent_count(r["name"]) > _accent_count(known):
                self._winery_names[key] = r["name"]

        self._grapes: Dict[str, int] = {}
        for r in c.execute("SELECT grape_id, name FROM grapes"):
            self._grapes[_key(r["name"])] = r["grape_id"]

        # (source_name, external_id) -> wine_id: a re-imported record goes straight
        # back to its wine, whatever the name matching would decide this time.
        self._source_wines: Dict[Tuple[str, str], int] = {}
        for r in c.execute(
            """SELECT source_name, external_id, wine_id FROM wine_sources
               WHERE external_id IS NOT NULL"""
        ):
            self._source_wines[(r["source_name"], r["external_id"])] = r["wine_id"]

        self._foods: Dict[str, int] = {}
        for r in c.execute("SELECT food_id, name FROM foods"):
            self._foods[_key(r["name"])] = r["food_id"]

        # title key -> {winery_id (0 = unknown): wine_id}; winery_id -> wine_ids
        self._wines: Dict[str, Dict[int, int]] = {}
        self._winery_wines: Dict[int, List[int]] = {}
        for r in c.execute("SELECT wine_id, winery_id, name FROM wines"):
            self._wines.setdefault(_key(r["name"]), {}).setdefault(
                r["winery_id"] or 0, r["wine_id"]
            )
            if r["winery_id"] is not None:
                self._winery_wines.setdefault(r["winery_id"], []).append(r["wine_id"])

        # wine_id -> names of the sources it came from (fuzzy matching never merges
        # two wines of the same source: a source lists its own wines separately)
        self._wine_source_names: Dict[int, Set[str]] = {}
        for r in c.execute("SELECT DISTINCT wine_id, source_name FROM wine_sources"):
            self._wine_source_names.setdefault(r["wine_id"], set()).add(r["source_name"])

        # Grapes are removed from titles before comparing; place words are kept
        # but do not count as distinctive.
        self._vocab = matching.Vocabulary()
        self._vocab.add_grapes(_GRAPE_LIST)
        self._vocab.add_grapes(GRAPE_SYNONYMS)
        self._vocab.add_grapes(r["name"] for r in c.execute("SELECT name FROM grapes"))
        self._vocab.add_places((_PLACES + "|" + _KEEP).replace("\n", "|").split("|"))
        self._vocab.add_places(r["name"] for r in c.execute("SELECT name FROM regions"))
        self._vocab.add_places(r["name"] for r in c.execute("SELECT name FROM countries"))

    # -------------------------------------------------------------- helpers
    def _insert(self, table: str, values: dict) -> int:
        values = {k: v for k, v in values.items() if v is not None}
        cols = ", ".join(values)
        marks = ", ".join("?" for _ in values)
        cur = self.conn.execute(
            f"INSERT INTO {table} ({cols}) VALUES ({marks})", list(values.values())
        )
        return cur.lastrowid

    def _fill(self, table: str, pk: str, row_id: int, values: dict):
        """Set columns that are currently NULL; never overwrite existing data."""
        values = {k: v for k, v in values.items() if v is not None}
        if not values:
            return
        sets = ", ".join(f"{k} = COALESCE({k}, ?)" for k in values)
        self.conn.execute(
            f"UPDATE {table} SET {sets} WHERE {pk} = ?", [*values.values(), row_id]
        )

    # ------------------------------------------------------- get-or-create
    def get_country(
        self, name: Optional[str], iso: Optional[str] = None
    ) -> Optional[int]:
        if not name:
            return None
        country_id = self._countries.get(_key(name))
        if country_id is None and iso:
            country_id = self._country_by_iso.get(iso)
        if country_id is None:
            country_id = self._insert("countries", {"name": name, "iso_code": iso})
            self._countries[_key(name)] = country_id
            self._vocab.add_place(name)
            if iso:
                self._country_by_iso[iso] = country_id
                self._country_has_iso.add(country_id)
        elif iso and country_id not in self._country_has_iso:
            self._fill("countries", "country_id", country_id, {"iso_code": iso})
            self._country_by_iso[iso] = country_id
            self._country_has_iso.add(country_id)
        return country_id

    def get_region(
        self, country_id: Optional[int], name: Optional[str]
    ) -> Optional[int]:
        if not name or country_id is None:
            return None
        key = _key(name)
        region_id = self._regions.get((country_id, key))
        if region_id is None:
            for candidate in self._aliases.get(key, []):
                if self._region_country.get(candidate) == country_id:
                    region_id = candidate
                    break
        if region_id is None:
            region_id = self._insert(
                "regions",
                {"country_id": country_id, "name": name, "region_type": "Wine Region"},
            )
            self._regions[(country_id, key)] = region_id
            self._region_country[region_id] = country_id
            self._vocab.add_place(name)
        return region_id

    def get_winery(
        self,
        name: Optional[str],
        region_id: Optional[int],
        website: Optional[str] = None,
    ) -> Optional[int]:
        if not name:
            return None
        by_region = self._wineries.setdefault(_key(name), {})
        region_key = region_id or 0
        winery_id = by_region.get(region_key)

        if winery_id is None and region_id is None and len(by_region) == 1:
            winery_id = next(iter(by_region.values()))  # unambiguous name match
        if winery_id is None and region_id is not None and 0 in by_region:
            winery_id = by_region.pop(0)  # adopt a winery that had no region yet
            by_region[region_id] = winery_id
            self.conn.execute(
                "UPDATE wineries SET region_id = ? WHERE winery_id = ?",
                (region_id, winery_id),
            )

        self._winery_created = winery_id is None
        if winery_id is None:
            spelling = self._winery_names.setdefault(_key(name), name)
            winery_id = self._insert(
                "wineries", {"name": spelling, "region_id": region_id, "website": website}
            )
            by_region[region_key] = winery_id
        else:
            self._fill("wineries", "winery_id", winery_id, {"website": website})
        return winery_id

    def _compatible(
        self, wine_id: int, country_id: Optional[int], color: Optional[str]
    ) -> bool:
        """False only if the candidate wine clearly conflicts on country or color."""
        row = self.conn.execute(
            """SELECT w.color, r.country_id FROM wines w
               LEFT JOIN regions r ON r.region_id = w.region_id WHERE w.wine_id = ?""",
            (wine_id,),
        ).fetchone()
        if row is None:
            return False
        if (
            country_id is not None
            and row["country_id"] is not None
            and row["country_id"] != country_id
        ):
            return False
        if color and row["color"] and fold(color) != fold(row["color"]):
            return False
        return True

    def _known_wine(self, rec: WineRecord) -> Optional[int]:
        """The wine this record was loaded into before (same source + external id)."""
        for s in rec.sources:
            if s.external_id:
                wine_id = self._source_wines.get((s.name, s.external_id))
                if wine_id is not None:
                    return wine_id
        return None

    def get_wine(
        self,
        rec: WineRecord,
        winery_id: Optional[int],
        region_id: Optional[int],
        country_id: Optional[int] = None,
        known_wine_id: Optional[int] = None,
    ) -> Tuple[int, bool]:
        title_key = _key(rec.name)
        by_winery = self._wines.setdefault(title_key, {})
        wine_id = known_wine_id if known_wine_id is not None else by_winery.get(winery_id or 0)

        if wine_id is not None:
            pass
        elif winery_id is not None and 0 in by_winery:
            # Same title already stored without a winery: adopt it and record the winery.
            candidate = by_winery[0]
            if self._compatible(candidate, country_id, rec.color):
                wine_id = by_winery.pop(0)
                by_winery[winery_id] = wine_id
                self._winery_wines.setdefault(winery_id, []).append(wine_id)
                self.conn.execute(
                    "UPDATE wines SET winery_id = ? WHERE wine_id = ?",
                    (winery_id, wine_id),
                )
        elif winery_id is None and by_winery:
            # No winery on this record: match on the title alone if it is unambiguous.
            if len(by_winery) == 1:
                ((candidate_winery, candidate),) = by_winery.items()
                if self._compatible(candidate, country_id, rec.color):
                    wine_id = candidate
            else:
                LOG.warning(
                    "Ambiguous wine name %r (%d wineries); not merging",
                    rec.name,
                    len(by_winery),
                )

        if wine_id is None:
            wine_id = self._fuzzy_wine(rec, country_id)
            if wine_id is not None and winery_id is not None:
                if self._winery_created:
                    self._drop_winery_if_empty(rec.winery, winery_id)
                else:  # remember the alias so a repeat of this title matches exactly
                    by_winery.setdefault(winery_id, wine_id)

        details = {
            "region_id": region_id,
            "full_title": rec.full_title,
            "style": rec.style,
            "color": rec.color,
            "varietal": rec.varietal,
            "blend_type": rec.blend_type,
            "abv": rec.abv,
            "closure": rec.closure,
        }
        if wine_id is not None:
            if rec.color:  # a stated color replaces one inferred from the grapes
                self.conn.execute(
                    """UPDATE wines SET color = ?, color_inferred = 0
                       WHERE wine_id = ? AND color_inferred = 1""",
                    (rec.color, wine_id),
                )
            self._fill("wines", "wine_id", wine_id, details)
            return wine_id, False
        wine_id = self._insert(
            "wines", {"winery_id": winery_id, "name": rec.name, **details}
        )
        by_winery[winery_id or 0] = wine_id
        if winery_id is not None:
            self._winery_wines.setdefault(winery_id, []).append(wine_id)
        return wine_id, True

    # ------------------------------------------------------- fuzzy matching
    def _drop_winery_if_empty(self, name: Optional[str], winery_id: int):
        """Remove a winery created for a record that then matched another wine."""
        cur = self.conn.execute(
            """DELETE FROM wineries WHERE winery_id = ?
               AND NOT EXISTS (SELECT 1 FROM wines WHERE winery_id = ?)""",
            (winery_id, winery_id),
        )
        if cur.rowcount:
            by_region = self._wineries.get(_key(name), {})
            for region_key, wid in list(by_region.items()):
                if wid == winery_id:
                    del by_region[region_key]
        self._winery_created = False

    def _winery_key_for(self, rec: WineRecord, title: str) -> Optional[str]:
        """The record's winery key; with no winery, the longest known winery name
        the title starts with ("Domaine Gramenon Côtes-du-Rhône ..." -> domaine gramenon)."""
        if rec.winery:
            return _key(rec.winery)
        tokens = fold(title).split()
        for size in range(min(6, len(tokens) - 1), 0, -1):
            key = " ".join(tokens[:size])
            if self._wineries.get(key):
                return key
        return None

    def _fuzzy_wine(self, rec: WineRecord, country_id: Optional[int]) -> Optional[int]:
        title = rec.full_title or rec.name
        winery_key = self._winery_key_for(rec, title)
        if not winery_key:
            return None
        rec_sources = {s.name for s in rec.sources}
        candidates = [
            wine_id
            for winery_id in set(self._wineries.get(winery_key, {}).values())
            for wine_id in self._winery_wines.get(winery_id, [])
            if not (self._wine_source_names.get(wine_id, set()) & rec_sources)
        ]
        if not candidates:
            return None

        marks = ",".join("?" * len(candidates))
        rows = self.conn.execute(
            f"""SELECT w.wine_id, w.name, w.full_title, w.style, w.varietal,
                       w.color, r.country_id
                FROM wines w LEFT JOIN regions r ON r.region_id = w.region_id
                WHERE w.wine_id IN ({marks})""",
            candidates,
        ).fetchall()
        grapes: Dict[int, List[str]] = {}
        for g in self.conn.execute(
            f"""SELECT wg.wine_id, g.name FROM wine_grapes wg
                JOIN grapes g USING (grape_id) WHERE wg.wine_id IN ({marks})""",
            candidates,
        ):
            grapes.setdefault(g["wine_id"], []).append(g["name"])

        winery_tokens = winery_key.split()
        rec_profile = matching.profile(title, winery_tokens, self._vocab).with_grapes(
            [g.name for g in rec.grapes] + ([rec.varietal] if rec.varietal else [])
        )
        scored = []
        for row in rows:
            if country_id and row["country_id"] and row["country_id"] != country_id:
                continue
            if rec.color and row["color"] and fold(rec.color) != fold(row["color"]):
                continue
            if rec.style and row["style"] and rec.style != row["style"]:
                continue
            cand = matching.profile(
                row["full_title"] or row["name"], winery_tokens, self._vocab
            ).with_grapes(
                grapes.get(row["wine_id"], []) + ([row["varietal"]] if row["varietal"] else [])
            )
            if matching.grapes_conflict(rec_profile, cand):
                continue
            score = matching.name_score(rec_profile, cand)
            if score is not None:
                scored.append((score, row["wine_id"]))

        best = matching.best_match(scored)
        if best is None:
            if len(scored) > 1:
                LOG.debug("Fuzzy match for %r is ambiguous; not merging", title)
            return None
        score, wine_id = best
        matched = next(r for r in rows if r["wine_id"] == wine_id)
        self.fuzzy_matches.append(
            {
                "record_title": title,
                "record_winery": rec.winery or "",
                "record_source": ", ".join(sorted(rec_sources)),
                "wine_id": wine_id,
                "matched_title": matched["full_title"] or matched["name"],
                "matched_sources": ", ".join(
                    sorted(self._wine_source_names.get(wine_id, set()))
                ),
                "score": round(score, 3),
            }
        )
        return wine_id

    def get_grape(self, name: str) -> int:
        """Get or create a grape in the shared ``grapes`` table."""
        grape_id = self._grapes.get(_key(name))
        if grape_id is None:
            grape_id = self._insert("grapes", {"name": name})
            self._grapes[_key(name)] = grape_id
            self._vocab.add_grape(name)
        return grape_id

    def get_food(self, name: str) -> int:
        """Get or create a food in the shared ``foods`` table (with its category)."""
        food_id = self._foods.get(_key(name))
        if food_id is None:
            food_id = self._insert(
                "foods", {"name": name, "category": food_category(name)}
            )
            self._foods[_key(name)] = food_id
        return food_id

    def get_vintage_id(self, wine_id: int, vintage: Optional[int]) -> int:
        """Get or create the vintage row. ``vintage=None`` is the non-vintage row."""
        row = self.conn.execute(
            # Written to match the ux_wine_vintages expression index exactly;
            # "vintage IS ?" made SQLite scan every row of that year instead.
            """SELECT vintage_id FROM wine_vintages
               WHERE wine_id = ? AND COALESCE(vintage, 0) = COALESCE(?, 0)""",
            (wine_id, vintage),
        ).fetchone()
        if row:
            return row["vintage_id"]
        return self._insert("wine_vintages", {"wine_id": wine_id, "vintage": vintage})

    # ---------------------------------------------------- child-table writes
    def _write_grapes(self, wine_id, rec):
        for g in rec.grapes:
            self.conn.execute(
                """INSERT INTO wine_grapes (wine_id, grape_id, blend_percentage)
                   VALUES (?, ?, ?)
                   ON CONFLICT(wine_id, grape_id) DO UPDATE
                   SET blend_percentage = COALESCE(blend_percentage, excluded.blend_percentage)""",
                (wine_id, self.get_grape(g.name), g.percentage),
            )

    def _write_sensory(self, wine_id, rec):
        if rec.sensory is None:
            return
        values = {f: getattr(rec.sensory, f) for f in SENSORY_FIELDS}
        cols = ", ".join(SENSORY_FIELDS)
        marks = ", ".join("?" for _ in SENSORY_FIELDS)
        updates = ", ".join(
            f"{f} = COALESCE({f}, excluded.{f})" for f in SENSORY_FIELDS
        )
        self.conn.execute(
            f"""INSERT INTO wine_sensory (wine_id, {cols}) VALUES (?, {marks})
                ON CONFLICT(wine_id) DO UPDATE SET {updates}""",
            (wine_id, *values.values()),
        )

    def _write_aromas(self, wine_id, rec):
        for a in rec.aromas:
            self.conn.execute(
                """INSERT INTO wine_aromas (wine_id, aroma_category, aroma_name, intensity)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(wine_id, aroma_name) DO UPDATE
                   SET aroma_category = COALESCE(aroma_category, excluded.aroma_category),
                       intensity = COALESCE(intensity, excluded.intensity)""",
                (wine_id, a.category, a.name, a.intensity),
            )

    def _write_pairings(self, wine_id, rec):
        source = rec.sources[0].name if rec.sources else None
        for food in rec.pairings:
            self.conn.execute(
                """INSERT INTO wine_pairings (wine_id, food_id, source) VALUES (?, ?, ?)
                   ON CONFLICT(wine_id, food_id) DO NOTHING""",
                (wine_id, self.get_food(food), source),
            )

    def _write_rating(self, wine_id: int, vintage_id: Optional[int], r: Rating):
        row = self.conn.execute(
            """SELECT rating_id FROM wine_ratings
               WHERE wine_id = ? AND vintage_id IS ? AND source IS ?
                 AND reviewer IS ? AND source_url IS ?""",
            (wine_id, vintage_id, r.source, r.reviewer, r.source_url),
        ).fetchone()
        if row:  # same rating slot re-imported: refresh the numbers
            self.conn.execute(
                "UPDATE wine_ratings SET score = ?, scale = ?, rating_count = ? WHERE rating_id = ?",
                (r.score, r.scale, r.rating_count, row["rating_id"]),
            )
        else:
            self._insert(
                "wine_ratings",
                {
                    "wine_id": wine_id,
                    "vintage_id": vintage_id,
                    "score": r.score,
                    "scale": r.scale,
                    "rating_count": r.rating_count,
                    "reviewer": r.reviewer,
                    "source": r.source,
                    "source_url": r.source_url,
                },
            )

    def _write_price(self, wine_id: int, vintage_id: Optional[int], p: Price):
        row = self.conn.execute(
            """SELECT price_id FROM wine_prices
               WHERE wine_id = ? AND vintage_id IS ? AND retailer IS ? AND currency = ?
                 AND source_url IS ? AND observed_at IS ?""",
            (wine_id, vintage_id, p.retailer, p.currency, p.source_url, p.observed_at),
        ).fetchone()
        if row:
            self.conn.execute(
                "UPDATE wine_prices SET price = ? WHERE price_id = ?",
                (p.price, row["price_id"]),
            )
        else:
            self._insert(
                "wine_prices",
                {
                    "wine_id": wine_id,
                    "vintage_id": vintage_id,
                    "price": p.price,
                    "currency": p.currency,
                    "retailer": p.retailer,
                    "source_url": p.source_url,
                    "observed_at": p.observed_at,
                },
            )

    def _write_source(self, wine_id: int, vintage_id: Optional[int], s: Source):
        if s.external_id:
            row = self.conn.execute(
                "SELECT source_id FROM wine_sources WHERE source_name = ? AND external_id = ?",
                (s.name, s.external_id),
            ).fetchone()
        else:
            row = self.conn.execute(
                """SELECT source_id FROM wine_sources
                   WHERE wine_id = ? AND vintage_id IS ? AND source_name = ?
                     AND source_url IS ? AND external_id IS NULL""",
                (wine_id, vintage_id, s.name, s.url),
            ).fetchone()
        if row:
            self._fill(
                "wine_sources",
                "source_id",
                row["source_id"],
                {
                    "vintage_id": vintage_id,
                    "source_url": s.url,
                    "rank": s.rank,
                    "tasting_note": s.tasting_note,
                },
            )
        else:
            if s.external_id:
                self._source_wines[(s.name, s.external_id)] = wine_id
            self._wine_source_names.setdefault(wine_id, set()).add(s.name)
            self._insert(
                "wine_sources",
                {
                    "wine_id": wine_id,
                    "vintage_id": vintage_id,
                    "source_name": s.name,
                    "external_id": s.external_id,
                    "source_url": s.url,
                    "rank": s.rank,
                    "tasting_note": s.tasting_note,
                    "tasting_note_truncated": int(s.tasting_note_truncated),
                },
            )

    # ---------------------------------------------------------------- ingest
    def ingest(self, rec: WineRecord) -> Tuple[int, bool]:
        """Write one already-normalized record. Returns (wine_id, created)."""
        self._winery_created = False
        country_id = self.get_country(rec.country, rec.country_iso)
        region_id = self.get_region(country_id, rec.region)
        known = self._known_wine(rec)
        if known is not None:
            row = self.conn.execute(
                "SELECT winery_id FROM wines WHERE wine_id = ?", (known,)
            ).fetchone()
            winery_id = row["winery_id"] if row else None
            if winery_id is not None and rec.winery_website:
                self._fill("wineries", "winery_id", winery_id, {"website": rec.winery_website})
        else:
            winery_id = self.get_winery(rec.winery, region_id, rec.winery_website)
        wine_id, created = self.get_wine(rec, winery_id, region_id, country_id, known)

        vintage_ids: Dict[Optional[int], int] = {
            v: self.get_vintage_id(wine_id, v) for v in rec.vintages
        }

        def vid(year: Optional[int]) -> Optional[int]:
            """Year -> vintage_id (creating it if needed). None = whole wine."""
            if year is None:
                return None
            if year not in vintage_ids:
                vintage_ids[year] = self.get_vintage_id(wine_id, year)
            return vintage_ids[year]

        self._write_grapes(wine_id, rec)
        self._write_sensory(wine_id, rec)
        self._write_aromas(wine_id, rec)
        self._write_pairings(wine_id, rec)
        for r in rec.ratings:
            self._write_rating(wine_id, vid(r.vintage), r)
        for p in rec.prices:
            self._write_price(wine_id, vid(p.vintage), p)
        for s in rec.sources:
            self._write_source(wine_id, vid(s.vintage), s)
        return wine_id, created

    def ingest_many(
        self,
        records: Iterable[WineRecord],
        *,
        normalize: bool = True,
        commit_every: int = 2000,
    ) -> IngestStats:
        """Normalize (optionally) and write many records, committing in batches.

        On any error the uncommitted batch is rolled back, caches are reloaded and a
        RuntimeError naming the offending record is raised. Already-committed batches
        stay, and re-running is safe.
        """
        stats = IngestStats()
        pending = 0
        index = 0
        title = "?"
        try:
            for index, rec in enumerate(records, 1):
                title = getattr(rec, "name", "?")
                if normalize:
                    rec = normalize_wine(rec)
                if rec is None:
                    stats.skipped += 1
                    LOG.warning("Skipping record %d with no name", index)
                    continue
                fuzzy_before = len(self.fuzzy_matches)
                _, created = self.ingest(rec)
                stats.fuzzy += len(self.fuzzy_matches) - fuzzy_before
                stats.created += int(created)
                stats.matched += int(not created)
                pending += 1
                if pending >= commit_every:
                    self.conn.commit()
                    pending = 0
            self.conn.commit()
        except Exception as exc:
            self.conn.rollback()
            self._load_caches()
            raise RuntimeError(
                f"Ingest failed at record {index} ({title!r}): {exc}"
            ) from exc

        LOG.info(
            "Ingested %d records: %d new wines, %d existing wines (%d by fuzzy "
            "name match), %d skipped",
            stats.total,
            stats.created,
            stats.matched,
            stats.fuzzy,
            stats.skipped,
        )
        return stats

    # ---------------------------------------------------------- user ratings
    def ingest_user_ratings(
        self, ratings: Iterable[UserRating], *, commit_every: int = 20000
    ) -> RatingStats:
        """Write individual user ratings, then refresh each wine's average rating.

        The wines must already be loaded from the same source: a rating's
        ``wine_external_id`` is looked up in ``wine_sources``. Ratings for unknown
        wines are counted and skipped. Re-running is safe (keyed on the rating id).
        """
        stats = RatingStats()
        wine_ids: Dict[Tuple[str, str], int] = {}
        vintage_ids: Dict[Tuple[int, Optional[int]], int] = {}
        sources: Set[str] = set()
        pending = 0
        try:
            for raw in ratings:
                r = normalize_user_rating(raw)
                if r is None:
                    stats.invalid += 1
                    continue
                if r.source not in sources:
                    sources.add(r.source)
                    for row in self.conn.execute(
                        """SELECT external_id, wine_id FROM wine_sources
                           WHERE source_name = ? AND external_id IS NOT NULL""",
                        (r.source,),
                    ):
                        wine_ids[(r.source, row["external_id"])] = row["wine_id"]
                wine_id = wine_ids.get((r.source, r.wine_external_id))
                if wine_id is None:
                    stats.unknown_wine += 1
                    continue

                vintage_id = None
                if r.non_vintage or r.vintage is not None:
                    key = (wine_id, r.vintage)  # vintage None here = the N.V. row
                    if key not in vintage_ids:
                        vintage_ids[key] = self.get_vintage_id(wine_id, r.vintage)
                    vintage_id = vintage_ids[key]

                self.conn.execute(
                    """INSERT INTO user_ratings
                         (source, user_id, wine_id, vintage_id, score, scale, rated_at, external_id)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(source, external_id) WHERE external_id IS NOT NULL
                       DO UPDATE SET score = excluded.score, scale = excluded.scale,
                                     vintage_id = excluded.vintage_id,
                                     rated_at = excluded.rated_at""",
                    (
                        r.source,
                        r.user_id,
                        wine_id,
                        vintage_id,
                        r.score,
                        r.scale,
                        r.rated_at,
                        r.external_id,
                    ),
                )
                stats.loaded += 1
                pending += 1
                if pending >= commit_every:
                    self.conn.commit()
                    pending = 0
            for source in sources:
                self.refresh_user_rating_averages(source)
            self.conn.commit()
        except Exception as exc:
            self.conn.rollback()
            self._load_caches()
            raise RuntimeError(f"User-rating ingest failed: {exc}") from exc

        LOG.info(
            "User ratings: %d loaded, %d for unknown wines, %d invalid",
            stats.loaded,
            stats.unknown_wine,
            stats.invalid,
        )
        return stats

    def refresh_user_rating_averages(self, source: str):
        """Rebuild the per-wine average of ``source``'s user ratings in wine_ratings.

        Stored with ``source = '<source> users'``, ``vintage_id`` NULL (all years),
        ``rating_count`` = number of ratings, so it sits beside critic scores.
        """
        label = f"{source} users"
        self.conn.execute("DELETE FROM wine_ratings WHERE source = ?", (label,))
        self.conn.execute(
            """INSERT INTO wine_ratings (wine_id, vintage_id, score, scale, rating_count, source)
               SELECT wine_id, NULL, ROUND(AVG(score * 1.0 / scale) * MAX(scale), 3),
                      MAX(scale), COUNT(*), ?
               FROM user_ratings WHERE source = ? GROUP BY wine_id""",
            (label, source),
        )
