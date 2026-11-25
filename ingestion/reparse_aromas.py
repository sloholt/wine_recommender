"""Apply the current aroma vocabulary (core/tasting.py) to wines already in the database.

Run this after changing ``_AROMA_LEXICON`` instead of re-importing every source:

    python -m ingestion.reparse_aromas --dry-run   # report what would change, write nothing
    python -m ingestion.reparse_aromas             # apply it (one transaction)

It does four things:
  1. re-reads every stored tasting note and adds aromas the vocabulary now finds;
  2. removes stale aromas: ones the vocabulary no longer finds in any of the
     wine's notes (e.g. "rose" read from the word "rosé", or "pepper" from "bell
     pepper"). Only for wines whose aromas all come from notes: wines with a
     structured flavor list (the Vivino famous-wines source) keep theirs;
  3. maps uncategorized aromas (raw flavor notes the vocabulary did not know when
     they were loaded, e.g. "espresso") onto vocabulary names when it now knows
     them ("espresso" -> "coffee"), merging with an existing row for that wine;
  4. sets ``intensity`` = the share of the wine's reviews that mention the aroma
     (0-1), so a note that 8 of 10 critics mention outweighs a one-off. Aromas that
     came from a structured source (a flavor list, not a note) count as one
     mention by one extra "review". Wines without any tasting note keep NULL
     (read as 1.0 by the flavor profiles).

Re-run it after importing new sources so their aromas get intensities too.
Afterwards rebuild the flavor profiles: python -m app.review_analytics
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .core.tasting import _aromas, _prep

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "data" / "wines.db"
BACKUP_DIR = ROOT / "data" / "processed"
AROMA_COLUMNS = ["aroma_id", "wine_id", "aroma_category", "aroma_name", "intensity"]
# Sources whose aromas come from a flavor list rather than a tasting note
# (ingestion/adapters/famous_wines.py). Their wines' aromas are never pruned.
STRUCTURED_AROMA_SOURCES = ("Vivino",)

UPSERT = """
INSERT INTO wine_aromas (wine_id, aroma_category, aroma_name)
VALUES (?, ?, ?)
ON CONFLICT(wine_id, aroma_name) DO UPDATE
SET aroma_category = COALESCE(aroma_category, excluded.aroma_category)
"""


@dataclass
class Plan:
    additions: dict = field(default_factory=dict)  # (wine_id, name) -> category
    removals: list = field(default_factory=list)  # (wine_id, name)
    remaps: list = field(default_factory=list)  # (wine_id, old, new, category)
    intensities: dict = field(default_factory=dict)  # (wine_id, name) -> new value

    def __bool__(self):
        return bool(self.additions or self.removals or self.remaps or self.intensities)


def plan_changes(conn: sqlite3.Connection) -> Plan:
    """Work out every change without writing anything."""
    plan = Plan()
    existing = {
        (wine_id, name): intensity
        for wine_id, name, intensity in conn.execute(
            "SELECT wine_id, aroma_name, intensity FROM wine_aromas"
        )
    }

    # 1. parse every note once: mentions per (wine, aroma), notes per wine
    mentions: Counter = Counter()
    n_notes: Counter = Counter()
    notes = conn.execute(
        """SELECT wine_id, tasting_note FROM wine_sources
           WHERE TRIM(COALESCE(tasting_note, '')) <> ''"""
    )
    for wine_id, note in notes:
        n_notes[wine_id] += 1
        for aroma in _aromas(_prep(note)):  # each aroma at most once per note
            key = (wine_id, aroma.name)
            mentions[key] += 1
            if key not in existing and key not in plan.additions:
                plan.additions[key] = aroma.category

    # 2. stale aromas on wines whose aromas all come from notes
    marks = ", ".join("?" for _ in STRUCTURED_AROMA_SOURCES)
    structured_wines = {
        wine_id
        for (wine_id,) in conn.execute(
            f"SELECT DISTINCT wine_id FROM wine_sources WHERE source_name IN ({marks})",
            STRUCTURED_AROMA_SOURCES,
        )
    }
    plan.removals = [
        (wine_id, name)
        for wine_id, name in existing
        if n_notes[wine_id]
        and wine_id not in structured_wines
        and mentions[(wine_id, name)] == 0
    ]

    # 3. uncategorized names the vocabulary now recognises
    for wine_id, name in conn.execute(
        "SELECT wine_id, aroma_name FROM wine_aromas WHERE aroma_category IS NULL"
    ):
        known = _aromas(_prep(name))
        if known:
            plan.remaps.append((wine_id, name, known[0].name, known[0].category))

    # 4. intensities over the aromas each wine will have after 1-3
    final = (set(existing) - set(plan.removals)) | set(plan.additions)
    for wine_id, old, new, _ in plan.remaps:
        final.discard((wine_id, old))
        final.add((wine_id, new))
    by_wine: dict = {}
    for wine_id, name in final:
        if n_notes[wine_id]:
            by_wine.setdefault(wine_id, []).append(name)
    for wine_id, names in by_wine.items():
        structured = any(mentions[(wine_id, n)] == 0 for n in names)
        reviews = n_notes[wine_id] + (1 if structured else 0)
        for name in names:
            key = (wine_id, name)
            value = round((mentions[key] or 1) / reviews, 3)
            if existing.get(key) != value:
                plan.intensities[key] = value
    return plan


def apply_changes(conn: sqlite3.Connection, plan: Plan) -> None:
    with conn:  # one transaction: all or nothing
        conn.executemany(
            UPSERT, [(w, cat, name) for (w, name), cat in plan.additions.items()]
        )
        conn.executemany(
            "DELETE FROM wine_aromas WHERE wine_id = ? AND aroma_name = ?", plan.removals
        )
        for wine_id, old, new, category in plan.remaps:
            conn.execute(UPSERT, (wine_id, category, new))
            if new != old:
                conn.execute(
                    "DELETE FROM wine_aromas WHERE wine_id = ? AND aroma_name = ?",
                    (wine_id, old),
                )
        conn.executemany(
            "UPDATE wine_aromas SET intensity = ? WHERE wine_id = ? AND aroma_name = ?",
            [(v, w, name) for (w, name), v in plan.intensities.items()],
        )


def backup_aromas(conn: sqlite3.Connection) -> Path:
    """Save the whole wine_aromas table to a CSV before changing it."""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    path = BACKUP_DIR / f"wine_aromas_backup_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(AROMA_COLUMNS)
        writer.writerows(
            conn.execute(f"SELECT {', '.join(AROMA_COLUMNS)} FROM wine_aromas")
        )
    return path


def restore_aromas(conn: sqlite3.Connection, path: Path) -> int:
    """Replace wine_aromas with the contents of a backup CSV."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        if next(reader) != AROMA_COLUMNS:
            raise ValueError(f"{path} is not a wine_aromas backup")
        rows = [[None if v == "" else v for v in row] for row in reader]
    with conn:
        conn.execute("DELETE FROM wine_aromas")
        conn.executemany(
            f"INSERT INTO wine_aromas ({', '.join(AROMA_COLUMNS)}) VALUES (?, ?, ?, ?, ?)",
            rows,
        )
    return len(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    parser.add_argument(
        "--restore", metavar="BACKUP_CSV", help=f"undo: restore a backup from {BACKUP_DIR}"
    )
    args = parser.parse_args(argv)

    if not args.db.exists():
        parser.error(f"database not found: {args.db}")
    if args.restore:
        path = Path(args.restore)
        path = path if path.exists() else BACKUP_DIR / path
        conn = sqlite3.connect(args.db)
        try:
            print(f"Restored {restore_aromas(conn, path):,} aromas from {path}")
        finally:
            conn.close()
        return 0
    uri = f"file:{args.db.as_posix()}" + ("?mode=ro" if args.dry_run else "")
    conn = sqlite3.connect(uri, uri=True)
    try:
        plan = plan_changes(conn)
        print(f"Aromas to add from tasting notes: {len(plan.additions):,} "
              f"(across {len({w for w, _ in plan.additions}):,} wines)")
        for name, n in Counter(name for _, name in plan.additions).most_common(15):
            print(f"  {name:<18} {n:>7,}")
        print(f"Stale aromas to remove (no longer found in the wine's notes): "
              f"{len(plan.removals):,}")
        for name, n in Counter(name for _, name in plan.removals).most_common(15):
            print(f"  {name:<18} {n:>7,}")
        print(f"Uncategorized aromas to map onto the vocabulary: {len(plan.remaps):,}")
        for (old, new), n in Counter((o, nw) for _, o, nw, _ in plan.remaps).most_common(15):
            print(f"  {old:<18} -> {new:<16} {n:>4}")
        print(f"Intensities to set: {len(plan.intensities):,} "
              f"(across {len({w for w, _ in plan.intensities}):,} wines)")

        if args.dry_run:
            print("\nDry run: nothing written.")
        elif plan:
            backup = backup_aromas(conn)
            print(f"\nBacked up wine_aromas to {backup} (restore with --restore {backup.name})")
            apply_changes(conn, plan)
            print("\nApplied. Rebuild the profiles: python -m app.review_analytics")
        else:
            print("\nNothing to change.")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
