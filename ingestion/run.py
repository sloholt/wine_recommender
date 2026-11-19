"""Load one source file into the wine database.

    python -m ingestion.run --source top100 --input data/raw/2023_top100_topworld.json
    python -m ingestion.run --source top100 --input FILE --db data/wines.db --schema schema.sql

If the database file does not exist yet (or has no tables), pass --schema to create it.
To add a source, write ingestion/adapters/<name>.py with ``read_records(path)`` that
yields WineRecord objects, then register it in ADAPTERS below.
"""

from __future__ import annotations

import argparse
import importlib
import logging
from pathlib import Path

from .core.db import WineDB

ROOT = Path(__file__).resolve().parents[1]

ADAPTERS = {
    "top100": "ingestion.adapters.top100_json",
}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Ingest a source file into the wine database"
    )
    parser.add_argument("--source", required=True, choices=sorted(ADAPTERS))
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--db", type=Path, default=ROOT / "data" / "wines.db")
    parser.add_argument(
        "--schema", type=Path, default=None, help="schema.sql, to create a new database"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    adapter = importlib.import_module(ADAPTERS[args.source])
    with WineDB(args.db, schema_path=args.schema) as db:
        stats = db.ingest_many(adapter.read_records(args.input))
    print(
        f"{stats.created} new wines, {stats.matched} already present, {stats.skipped} skipped"
    )
    return stats


if __name__ == "__main__":
    main()
