"""Load source files into the wine database.

    python -m ingestion.run --source top100 --input data/raw/top100json/2023_top100_topworld.json
    python -m ingestion.run --source top100 --input data/raw/top100json --schema database/schema.sql
    python -m ingestion.run --source xwines --input data/raw/XWines_Full_100K_wines.csv
    python -m ingestion.run --source xwines --ratings data/raw/XWines_Slim_150K_ratings.csv
    python -m ingestion.run --source wineinfo --input data/raw/wine_info/wine_info.csv
    python -m ingestion.run --source wineenthusiast --input data/wines.csv
    python -m ingestion.run --source vivino --input data/raw/vivino
    python -m ingestion.run --source famouswines --input data/raw/famous_wines
    python -m ingestion.run --source xwines --input FILE --match-report data/processed/matches.csv

If the database file does not exist yet (or has no tables), pass --schema to create it.
``--input`` may be a folder: every file in it matching the adapter's FILE_GLOB
(default "*") is loaded, in name order.
``--input`` loads wines; ``--ratings`` loads individual user ratings (only for sources
whose adapter has ``read_user_ratings``) and runs after --input when both are given,
because ratings are linked to wines already in the database.

To add a source, write ingestion/adapters/<name>.py with ``read_records(path)`` that
yields WineRecord objects, then register it in ADAPTERS below.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import logging
from pathlib import Path

from .core.db import WineDB

ROOT = Path(__file__).resolve().parents[1]

ADAPTERS = {
    "top100": "ingestion.adapters.top100_json",
    "xwines": "ingestion.adapters.xwines",
    "wineinfo": "ingestion.adapters.wine_info",
    "wineenthusiast": "ingestion.adapters.wine_enthusiast",
    "vivino": "ingestion.adapters.vivino",
    "famouswines": "ingestion.adapters.famous_wines",
}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Ingest a source file into the wine database"
    )
    parser.add_argument("--source", required=True, choices=sorted(ADAPTERS))
    parser.add_argument("--input", type=Path, help="wine file to load")
    parser.add_argument("--ratings", type=Path, help="user-ratings file to load")
    parser.add_argument("--db", type=Path, default=ROOT / "data" / "wines.db")
    parser.add_argument(
        "--schema", type=Path, default=None, help="schema.sql, to create a new database"
    )
    parser.add_argument(
        "--match-report",
        type=Path,
        help="write a CSV of the wines merged by fuzzy name matching, for review",
    )
    args = parser.parse_args(argv)
    if not args.input and not args.ratings:
        parser.error("give --input, --ratings, or both")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    adapter = importlib.import_module(ADAPTERS[args.source])
    if args.ratings and not hasattr(adapter, "read_user_ratings"):
        parser.error(f"source {args.source!r} has no user ratings")

    stats = rating_stats = None
    with WineDB(args.db, schema_path=args.schema) as db:
        if args.input:
            if args.input.is_dir():
                pattern = getattr(adapter, "FILE_GLOB", "*")
                files = sorted(p for p in args.input.glob(pattern) if p.is_file())
                if not files:
                    parser.error(f"no {pattern} files in {args.input}")
            else:
                files = [args.input]
            for path in files:
                stats = db.ingest_many(adapter.read_records(path))
                print(
                    f"{path.name}: {stats.created} new wines, {stats.matched} already "
                    f"present ({stats.fuzzy} by fuzzy match), {stats.skipped} skipped"
                )
            colored = db.infer_colors()
            print(f"{colored} wines have a color inferred from their grapes")
        if args.ratings:
            rating_stats = db.ingest_user_ratings(
                adapter.read_user_ratings(args.ratings)
            )
            print(
                f"{rating_stats.loaded} user ratings loaded, "
                f"{rating_stats.unknown_wine} for wines not in the database, "
                f"{rating_stats.invalid} invalid"
            )
        if args.match_report:
            write_match_report(args.match_report, db.fuzzy_matches)
    return stats, rating_stats


def write_match_report(path: Path, matches) -> None:
    """CSV of fuzzy merges (record title -> matched wine), sorted weakest first."""
    fields = [
        "score", "record_title", "record_winery", "record_source",
        "wine_id", "matched_title", "matched_sources",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(matches, key=lambda m: m["score"]))
    print(f"{len(matches)} fuzzy matches written to {path}")


if __name__ == "__main__":
    main()
