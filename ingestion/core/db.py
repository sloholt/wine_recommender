"""Write normalized WineRecord objects into the wine SQLite database.

Design:
* get-or-create for countries, regions, wineries, wines and vintages, using
  in-memory lookup caches loaded once at startup (fast, no per-row SELECT storms).
* Wine identity = winery + title (accent/case/punctuation-insensitive).
* Re-running an import is safe: existing rows are matched, NULL fields are filled
  in, and ratings / prices are updated instead of duplicated.
* If an insert fails, the current batch is rolled back and the caches reloaded,
  so the caches never point at rows that no longer exist.

Known limitation: a record with no winery will not match an existing wine that
has one (the identity differs), so prefer adapters that supply the winery.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .models import Price, Rating, Source, WineRecord
from .normalize import fold, normalize_wine

LOG = logging.getLogger(__name__)

REQUIRED_TABLES = {
    "countries",
    "regions",
    "region_aliases",
    "wineries",
    "wines",
    "wine_vintages",
    "wine_grapes",
    "wine_sensory",
    "wine_aromas",
    "wine_pairings",
    "wine_ratings",
    "wine_prices",
    "wine_sources",
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


@dataclass
class IngestStats:
    created: int = 0  # new wines
    matched: int = 0  # wines that already existed
    skipped: int = 0  # records dropped by the normalizer (no title)

    @property
    def total(self) -> int:
        return self.created + self.matched + self.skipped


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
        for r in c.execute("SELECT winery_id, name, region_id FROM wineries"):
            self._wineries.setdefault(_key(r["name"]), {})[r["region_id"] or 0] = r[
                "winery_id"
            ]

        self._wines: Dict[Tuple[int, str], int] = {}
        for r in c.execute("SELECT wine_id, winery_id, title FROM wines"):
            self._wines.setdefault(
                (r["winery_id"] or 0, _key(r["title"])), r["wine_id"]
            )

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

        if winery_id is None:
            winery_id = self._insert(
                "wineries", {"name": name, "region_id": region_id, "website": website}
            )
            by_region[region_key] = winery_id
        else:
            self._fill("wineries", "winery_id", winery_id, {"website": website})
        return winery_id

    def get_wine(
        self, rec: WineRecord, winery_id: Optional[int], region_id: Optional[int]
    ) -> Tuple[int, bool]:
        key = (winery_id or 0, _key(rec.title))
        details = {
            "region_id": region_id,
            "color": rec.color,
            "varietal": rec.varietal,
            "abv": rec.abv,
            "closure": rec.closure,
        }
        wine_id = self._wines.get(key)
        if wine_id is not None:
            self._fill("wines", "wine_id", wine_id, details)
            return wine_id, False
        wine_id = self._insert(
            "wines", {"winery_id": winery_id, "title": rec.title, **details}
        )
        self._wines[key] = wine_id
        return wine_id, True

    def get_vintage_id(self, wine_id: int, vintage: Optional[int]) -> int:
        """Get or create the vintage row. ``vintage=None`` is the non-vintage row."""
        row = self.conn.execute(
            "SELECT vintage_id FROM wine_vintages WHERE wine_id = ? AND vintage IS ?",
            (wine_id, vintage),
        ).fetchone()
        if row:
            return row["vintage_id"]
        return self._insert("wine_vintages", {"wine_id": wine_id, "vintage": vintage})

    # ---------------------------------------------------- child-table writes
    def _write_grapes(self, wine_id, rec):
        for g in rec.grapes:
            self.conn.execute(
                """INSERT INTO wine_grapes (wine_id, grape_variety, blend_percentage)
                   VALUES (?, ?, ?)
                   ON CONFLICT(wine_id, grape_variety) DO UPDATE
                   SET blend_percentage = COALESCE(blend_percentage, excluded.blend_percentage)""",
                (wine_id, g.name, g.percentage),
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
        for food in rec.pairings:
            self.conn.execute(
                "INSERT OR IGNORE INTO wine_pairings (wine_id, food) VALUES (?, ?)",
                (wine_id, food),
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
        country_id = self.get_country(rec.country, rec.country_iso)
        region_id = self.get_region(country_id, rec.region)
        winery_id = self.get_winery(rec.winery, region_id, rec.winery_website)
        wine_id, created = self.get_wine(rec, winery_id, region_id)

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
                title = getattr(rec, "title", "?")
                if normalize:
                    rec = normalize_wine(rec)
                if rec is None:
                    stats.skipped += 1
                    LOG.warning("Skipping record %d with no title", index)
                    continue
                _, created = self.ingest(rec)
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
            "Ingested %d records: %d new wines, %d existing wines, %d skipped",
            stats.total,
            stats.created,
            stats.matched,
            stats.skipped,
        )
        return stats
