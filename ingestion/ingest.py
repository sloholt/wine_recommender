"""Load normalized scraper records into an existing SQLite wine database.

Run:
    python -m ingestion.ingest --input data/processed/2025_top100_everydayvalue_normalized.json

Optional fields are stored only if matching columns exist.
"""

import argparse
import json
import logging
import sqlite3
from pathlib import Path

from ingestion.normalize import normalize_record

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger(__name__)

ISO_CODES = {
    "Argentina": "AR",
    "Australia": "AU",
    "Austria": "AT",
    "Chile": "CL",
    "France": "FR",
    "Germany": "DE",
    "Greece": "GR",
    "Hungary": "HU",
    "Italy": "IT",
    "New Zealand": "NZ",
    "Portugal": "PT",
    "South Africa": "ZA",
    "Spain": "ES",
    "United States": "US",
}


def columns(conn, table):
    return {r["name"] for r in conn.execute(f'PRAGMA table_info("{table}")')}


def insert(conn, table, data):
    cols = columns(conn, table)

    values = {k: v for k, v in data.items() if k in cols and v is not None}

    if not values:
        raise ValueError(f"No compatible fields for table {table}")

    keys = list(values)
    names = ", ".join(f'"{k}"' for k in keys)
    marks = ", ".join("?" for _ in keys)

    return conn.execute(
        f'INSERT INTO "{table}" ({names}) VALUES ({marks})',
        [values[k] for k in keys],
    ).lastrowid


def fill_missing(conn, table, id_col, record_id, data):
    cols = columns(conn, table)

    for field, value in data.items():
        if field in cols and value is not None:
            conn.execute(
                f"""
                UPDATE "{table}"
                SET "{field}" = ?
                WHERE "{id_col}" = ?
                  AND "{field}" IS NULL
                """,
                (value, record_id),
            )


def get_country(conn, name):
    if not name:
        return None

    row = conn.execute(
        """
        SELECT country_id
        FROM countries
        WHERE name = ? COLLATE NOCASE
        """,
        (name,),
    ).fetchone()

    if row:
        return row["country_id"]

    return insert(
        conn,
        "countries",
        {
            "name": name,
            "iso_code": ISO_CODES.get(name),
        },
    )


def get_region(conn, country_id, name):
    if not name or country_id is None:
        return None

    if "parent_region_id" in columns(conn, "regions"):
        row = conn.execute(
            """
            SELECT region_id
            FROM regions
            WHERE country_id = ?
              AND name = ? COLLATE NOCASE
              AND parent_region_id IS NULL
            """,
            (country_id, name),
        ).fetchone()
    else:
        row = conn.execute(
            """
            SELECT region_id
            FROM regions
            WHERE country_id = ?
              AND name = ? COLLATE NOCASE
            """,
            (country_id, name),
        ).fetchone()

    if row:
        return row["region_id"]

    return insert(
        conn,
        "regions",
        {
            "name": name,
            "country_id": country_id,
            "region_type": "Wine Region",
        },
    )


def get_wine(conn, wine, region_id):
    # A full title + vintage is the current canonical identity.
    # A source rank is NOT a wine identifier.

    row = conn.execute(
        """
        SELECT wine_id
        FROM wines
        WHERE title = ? COLLATE NOCASE
          AND vintage IS ?
        """,
        (wine["title"], wine["vintage"]),
    ).fetchone()

    details = {
        "region_id": region_id,
        "winery": wine.get("winery"),
        "varietal": wine.get("varietal"),
        "color": wine.get("color"),
        "tasting_note": wine.get("tasting_note"),
        "tasting_note_truncated": int(wine.get("tasting_note_truncated", False)),
    }

    if row:
        fill_missing(
            conn,
            "wines",
            "wine_id",
            row["wine_id"],
            details,
        )
        return row["wine_id"], False

    wine_id = insert(
        conn,
        "wines",
        {
            "title": wine["title"],
            "vintage": wine["vintage"],
            **details,
        },
    )

    return wine_id, True


def exists(conn, table, criteria):
    cols = columns(conn, table)

    criteria = {k: v for k, v in criteria.items() if k in cols}

    if not criteria:
        raise ValueError(f"No deduplication keys available for {table}")

    where = " AND ".join(f'"{k}" IS ?' for k in criteria)

    return (
        conn.execute(
            f'SELECT 1 FROM "{table}" WHERE {where} LIMIT 1',
            list(criteria.values()),
        ).fetchone()
        is not None
    )


def ingest_one(conn, wine, report_id):
    country_id = get_country(
        conn,
        wine.get("country"),
    )

    region_id = get_region(
        conn,
        country_id,
        wine.get("region"),
    )

    wine_id, created = get_wine(
        conn,
        wine,
        region_id,
    )

    source = wine.get("source_name")
    url = wine.get("source_url")
    rank = wine.get("rank")

    external_id = f"{report_id}#{rank}" if rank is not None else None

    # --------------------------------------------------
    # Wine ratings
    # --------------------------------------------------

    if wine.get("score") is not None and columns(conn, "wine_ratings"):
        rating = {
            "wine_id": wine_id,
            "score": wine["score"],
            "reviewer": source,
            "source_name": source,
            "source": source,
            "source_url": url,
        }

        cols = columns(conn, "wine_ratings")

        key = {"wine_id": wine_id}

        for field in ("reviewer", "source_name", "source"):
            if field in cols and rating.get(field):
                key[field] = rating[field]
                break

        if "score" in cols:
            key["score"] = wine["score"]

        if "source_url" in cols:
            key["source_url"] = url

        if not exists(conn, "wine_ratings", key):
            insert(conn, "wine_ratings", rating)

    # --------------------------------------------------
    # Wine prices
    # --------------------------------------------------

    if (
        wine.get("price") is not None
        and wine.get("currency")
        and columns(conn, "wine_prices")
    ):
        price = {
            "wine_id": wine_id,
            "price": wine["price"],
            "currency": wine["currency"],
            "retailer": source,
            "source_name": source,
            "source_url": url,
        }

        cols = columns(conn, "wine_prices")

        key = {"wine_id": wine_id}

        for field in (
            "price",
            "currency",
            "retailer",
            "source_url",
        ):
            if field in cols:
                key[field] = price.get(field)

        if not exists(conn, "wine_prices", key):
            insert(conn, "wine_prices", price)

    # --------------------------------------------------
    # Wine sources and rankings
    # --------------------------------------------------

    if columns(conn, "wine_sources"):
        provenance = {
            "wine_id": wine_id,
            "source_name": source,
            "source": source,
            "source_url": url,
            "url": url,
            "external_id": external_id,
            "rank": rank,
            "tasting_note": wine.get("tasting_note"),
            "tasting_note_truncated": int(wine.get("tasting_note_truncated", False)),
            "color": wine.get("color"),
        }

        cols = columns(conn, "wine_sources")

        # Report rank is unique within a report.
        # The same wine can appear in multiple reports.

        if "external_id" in cols and external_id:
            key = {
                "external_id": external_id,
            }
        else:
            key = {
                "wine_id": wine_id,
            }

            if "source_url" in cols:
                key["source_url"] = url
            elif "url" in cols:
                key["url"] = url

        if not exists(conn, "wine_sources", key):
            insert(conn, "wine_sources", provenance)
        else:
            where = " AND ".join(f'"{k}" IS ?' for k in key)

            row = conn.execute(
                "SELECT rowid AS _rowid " "FROM wine_sources WHERE " + where,
                list(key.values()),
            ).fetchone()

            if row:
                fill_missing(
                    conn,
                    "wine_sources",
                    "rowid",
                    row["_rowid"],
                    provenance,
                )

    return created


def ingest_file(input_path, db_path=None):
    input_path = Path(input_path)

    db_path = Path(db_path) if db_path else ROOT / "data" / "wines.db"

    if not db_path.is_file():
        raise FileNotFoundError(
            f"Database not found: {db_path}. " "Initialize schema first."
        )

    records = json.loads(input_path.read_text(encoding="utf-8"))

    if not isinstance(records, list):
        raise ValueError("Input JSON must contain a list")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")

    new = 0
    existing = 0

    try:
        tables = {r["name"] for r in conn.execute("""
                SELECT name
                FROM sqlite_master
                WHERE type = 'table'
                """)}

        required = {
            "countries",
            "regions",
            "wines",
        }

        if not required.issubset(tables):
            raise ValueError(f"Missing tables: {sorted(required - tables)}")

        if "tasting_note" not in columns(
            conn, "wines"
        ) and "tasting_note" not in columns(conn, "wine_sources"):
            LOG.warning(
                "Schema has no tasting_note column in "
                "wines or wine_sources; notes will not be stored."
            )

        if "color" not in columns(conn, "wines") and "color" not in columns(
            conn, "wine_sources"
        ):
            LOG.warning(
                "Schema has no color column in "
                "wines or wine_sources; colors will not be stored."
            )

        with conn:
            for index, raw in enumerate(records, 1):
                if not isinstance(raw, dict):
                    raise ValueError(f"Record {index} is not an object")

                wine = normalize_record(raw)

                if wine is None:
                    LOG.warning(
                        "Skipping record %d with no title",
                        index,
                    )
                    continue

                try:
                    created = ingest_one(
                        conn,
                        wine,
                        input_path.stem.removesuffix("_normalized"),
                    )

                except Exception as exc:
                    raise RuntimeError(
                        f"Error at record {index} " f"({wine['title']}): {exc}"
                    ) from exc

                new += int(created)
                existing += int(not created)

    finally:
        conn.close()

    LOG.info(
        "Ingested %d records: %d new wines, " "%d existing wines",
        new + existing,
        new,
        existing,
    )

    return new, existing


def main():
    parser = argparse.ArgumentParser(
        description="Ingest normalized wine records into SQLite"
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to normalized wine JSON",
    )

    parser.add_argument(
        "--db",
        type=Path,
        default=ROOT / "data" / "wines.db",
        help="Path to existing SQLite database",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    ingest_file(args.input, args.db)


if __name__ == "__main__":
    main()
