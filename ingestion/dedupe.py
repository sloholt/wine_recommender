"""Merge duplicate wineries and wines that differ only in spelling.

    python -m ingestion.dedupe --dry-run     # report the plan, write nothing
    python -m ingestion.dedupe               # back up the database, then apply

Wineries
  Rows whose names match once accents, case and punctuation are removed
  ("Chateau d'Esclans" / "Château d'Esclans"), in the same country, are one
  producer. They all get one spelling (the accented one, else the most used).
  Wineries are stored per region (the schema's identity is name + region), so a
  producer keeps one row per region; rows with the same region are merged.
  Same name in different countries (Tesco, Joseph Drouhin) is left alone.

Wines
  Within one producer, wines are duplicates when their names match after
  removing accents/case and a trailing color word that agrees with the wine's
  color ("Whispering Angel" / "Whispering Angel Rosé", "Sancerre" / "Sancerre
  Blanc"), and nothing conflicts:
    * color and style equal or unknown;
    * regions equal, unknown, or one contained in the other as whole words
      ("Provence" / "Côtes de Provence"). "Pinot Noir" from Sonoma Coast and from
      Napa Valley stay separate.
  The wine with the most sources survives; reviews, ratings, user ratings,
  prices, vintages, grapes, aromas, sensory values and pairings move to it.

Afterwards: user-rating averages and aroma intensities are refreshed. Rebuild
the flavor profiles with: python -m app.review_analytics
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .core.normalize import fold

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "data" / "wines.db"
REPORT_DIR = ROOT / "data" / "processed"

# Trailing words that only repeat the wine's color: dropped before comparing names.
COLOR_WORDS = {
    "Rosé": {"rose", "rosato", "rosado", "rosat"},
    "White": {"blanc", "bianco", "blanco", "white", "branco", "weiss"},
    "Red": {"rouge", "rosso", "tinto", "red", "rot"},
}
WINE_FILL_COLUMNS = [
    "full_title", "style", "color", "varietal", "blend_type", "abv", "closure", "region_id",
]
SENSORY_COLUMNS = ["body", "acidity", "structure", "finish", "balance", "tannin", "sweetness"]


# A color word after these is part of a grape name ("Sauvignon Blanc", "Pinot
# Bianco", "Grenache Blanc"), never a color label to drop.
GRAPE_STEMS = {
    "sauvignon", "chenin", "pinot", "grenache", "garnacha", "merlot", "muscat",
    "moscato", "malvasia", "trebbiano", "ugni", "cabernet", "tempranillo", "vino",
}


def base_name(name: str, color: Optional[str]) -> str:
    """Comparison key for a wine name: folded, without a trailing color word."""
    tokens = fold(name).split()
    words = COLOR_WORDS.get(color or "", set())
    while len(tokens) > 1 and tokens[-1] in words and tokens[-2] not in GRAPE_STEMS:
        tokens.pop()
    return " ".join(tokens)


def regions_related(a: str, b: str) -> bool:
    """Folded region names: equal, unknown, or one inside the other as whole words."""
    return not a or not b or a == b or f" {a} " in f" {b} " or f" {b} " in f" {a} "


def _accents(text: str) -> int:
    return sum(ord(ch) > 127 for ch in text)


@dataclass
class Plan:
    renames: Dict[int, str] = field(default_factory=dict)  # winery_id -> canonical name
    winery_merges: List[tuple] = field(default_factory=list)  # (keep_id, drop_id)
    wine_merges: List[tuple] = field(default_factory=list)  # (keep_id, drop_id)
    skipped: List[tuple] = field(default_factory=list)  # (reason, detail)
    report: List[dict] = field(default_factory=list)  # one row per wine merge


# ------------------------------------------------------------------- planning
def plan_changes(conn: sqlite3.Connection) -> Plan:
    plan = Plan()
    wineries = pd.read_sql(
        """SELECT wy.winery_id, wy.name, wy.region_id, r.country_id
           FROM wineries wy LEFT JOIN regions r ON r.region_id = wy.region_id""",
        conn,
    )
    wines = pd.read_sql(
        """SELECT w.wine_id, w.winery_id, w.name, w.full_title, w.color, w.style,
                  r.name AS region,
                  (SELECT COUNT(*) FROM wine_sources s WHERE s.wine_id = w.wine_id) AS n_sources
           FROM wines w LEFT JOIN regions r ON r.region_id = w.region_id
           WHERE w.winery_id IS NOT NULL""",
        conn,
    )
    wineries["key"] = wineries["name"].map(fold)
    wines_per_winery = wines.groupby("winery_id").size()
    wineries["n_wines"] = wineries["winery_id"].map(wines_per_winery).fillna(0).astype(int)

    # A row without a country joins its name's group if that name has one country.
    known = wineries[wineries["country_id"].notna()]
    countries = known.groupby("key")["country_id"].agg(["nunique", "first"])
    single = countries.loc[countries["nunique"] == 1, "first"]
    wineries["group_country"] = wineries["country_id"]
    no_country = wineries["group_country"].isna() & wineries["key"].isin(single.index)
    wineries.loc[no_country, "group_country"] = wineries.loc[no_country, "key"].map(single)
    grouped = wineries[wineries["group_country"].notna()]

    producer_of: Dict[int, int] = {}  # winery_id -> producer group number
    for g, (_, rows) in enumerate(grouped.groupby(["key", "group_country"])):
        if len(rows) < 2:
            continue
        rows = rows.assign(acc=rows["name"].map(_accents))
        canonical = rows.sort_values(["acc", "n_wines"], ascending=False)["name"].iloc[0]
        for r in rows.itertuples():
            producer_of[r.winery_id] = g
            if r.name != canonical:
                plan.renames[r.winery_id] = canonical
        # Rows with the same region become one row (they only differed in spelling).
        for _, same in rows.groupby(rows["region_id"].fillna(0)):
            if len(same) > 1:
                same = same.sort_values(["n_wines", "winery_id"], ascending=[False, True])
                keep = int(same["winery_id"].iloc[0])
                for drop in same["winery_id"].iloc[1:]:
                    plan.winery_merges.append((keep, int(drop)))

    # Wines: compare within each producer group (all of its rows).
    wines["producer"] = wines["winery_id"].map(producer_of)
    candidates = wines[wines["producer"].notna()].copy()
    candidates["base"] = [base_name(n, c) for n, c in zip(candidates["name"], candidates["color"])]
    candidates["region_key"] = candidates["region"].map(lambda s: fold(s) if s else "")

    merged_rows = set()
    for (_, _), bucket in candidates.groupby(["producer", "base"]):
        if len(bucket) < 2:
            continue
        bucket = bucket.sort_values(["n_sources", "wine_id"], ascending=[False, True])
        clusters: List[List[dict]] = []
        for wine in bucket.to_dict("records"):
            for cluster in clusters:
                if all(_compatible(wine, other) for other in cluster):
                    cluster.append(wine)
                    break
            else:
                clusters.append([wine])
        if len(clusters) > 1:
            plan.skipped.append(
                ("kept apart (color/style/region differ)",
                 " | ".join(f"{w['name']} [{w['color']}, {w['region']}]" for w in bucket.to_dict("records")))
            )
        for cluster in clusters:
            keep = cluster[0]
            for drop in cluster[1:]:
                plan.wine_merges.append((int(keep["wine_id"]), int(drop["wine_id"])))
                merged_rows.add(int(drop["wine_id"]))
                plan.report.append({
                    "keep_wine_id": keep["wine_id"], "keep_name": keep["name"],
                    "keep_region": keep["region"], "drop_wine_id": drop["wine_id"],
                    "drop_name": drop["name"], "drop_region": drop["region"],
                    "color": keep["color"] or drop["color"],
                })

    # Rows merged into one winery must not end up with two wines of the same name
    # (unique index on winery + name). Same-name wines there are merged above
    # unless they conflict; then keep that winery row separate.
    winery_merges = []
    final_winery = {drop: keep for keep, drop in plan.winery_merges}
    names = candidates[~candidates["wine_id"].isin(merged_rows)].assign(
        target=lambda d: d["winery_id"].map(lambda w: final_winery.get(w, w)),
        lname=lambda d: d["name"].str.lower(),
    )
    clashes = names.groupby(["target", "lname"])["winery_id"].nunique()
    clashing_rows = set()
    for (target, lname), n in clashes[clashes > 1].items():
        rows = names[(names["target"] == target) & (names["lname"] == lname)]
        clashing_rows.update(int(w) for w in rows["winery_id"] if w != target)
    for keep, drop in plan.winery_merges:
        if drop in clashing_rows:
            plan.skipped.append(("winery row kept: same-name wines conflict", f"{keep} <- {drop}"))
            plan.renames.pop(drop, None)  # its name would collide with the kept row
        else:
            winery_merges.append((keep, drop))
    plan.winery_merges = winery_merges
    return plan


def _compatible(a: dict, b: dict) -> bool:
    for col in ("color", "style"):
        if a[col] and b[col] and a[col] != b[col]:
            return False
    # Names equal only after dropping a color word: the colors must be known and
    # equal ("Extra Brut" with no color may be the white of "Extra Brut Rosé").
    if fold(a["name"]) != fold(b["name"]) and not (a["color"] and a["color"] == b["color"]):
        return False
    return regions_related(a["region_key"], b["region_key"])


# ------------------------------------------------------------------- applying
def merge_wine(conn: sqlite3.Connection, keep: int, drop: int) -> None:
    """Move everything of wine ``drop`` onto wine ``keep``, then delete ``drop``."""
    # Vintages: reuse keep's row for the same year, move the others.
    for vid, year in conn.execute(
        "SELECT vintage_id, vintage FROM wine_vintages WHERE wine_id = ?", (drop,)
    ).fetchall():
        same = conn.execute(
            "SELECT vintage_id FROM wine_vintages WHERE wine_id = ? AND vintage IS ?",
            (keep, year),
        ).fetchone()
        if same:
            # Rows pointing at drop's vintage belong to drop: filtering on wine_id
            # uses the wine_id indexes (user_ratings has ~21M rows, no vintage index).
            for table in ("wine_ratings", "wine_prices", "wine_sources", "user_ratings"):
                conn.execute(
                    f"UPDATE {table} SET vintage_id = ? WHERE wine_id = ? AND vintage_id = ?",
                    (same[0], drop, vid),
                )
            conn.execute("DELETE FROM wine_vintages WHERE vintage_id = ?", (vid,))
        else:
            conn.execute("UPDATE wine_vintages SET wine_id = ? WHERE vintage_id = ?", (keep, vid))

    for table in ("wine_sources", "wine_ratings", "wine_prices", "user_ratings"):
        conn.execute(f"UPDATE {table} SET wine_id = ? WHERE wine_id = ?", (keep, drop))

    conn.execute(
        """INSERT INTO wine_grapes (wine_id, grape_id, blend_percentage)
           SELECT ?, grape_id, blend_percentage FROM wine_grapes WHERE wine_id = ?
           ON CONFLICT(wine_id, grape_id) DO UPDATE
           SET blend_percentage = COALESCE(blend_percentage, excluded.blend_percentage)""",
        (keep, drop),
    )
    conn.execute(
        """INSERT INTO wine_aromas (wine_id, aroma_category, aroma_name, intensity)
           SELECT ?, aroma_category, aroma_name, intensity FROM wine_aromas WHERE wine_id = ?
           ON CONFLICT(wine_id, aroma_name) DO NOTHING""",
        (keep, drop),
    )
    conn.execute(
        """INSERT INTO wine_pairings (wine_id, food_id, source)
           SELECT ?, food_id, source FROM wine_pairings WHERE wine_id = ?
           ON CONFLICT(wine_id, food_id) DO NOTHING""",
        (keep, drop),
    )
    cols = ", ".join(SENSORY_COLUMNS)
    conn.execute(
        f"""INSERT INTO wine_sensory (wine_id, {cols})
            SELECT ?, {cols} FROM wine_sensory WHERE wine_id = ?
            ON CONFLICT(wine_id) DO UPDATE SET
            {", ".join(f"{c} = COALESCE({c}, excluded.{c})" for c in SENSORY_COLUMNS)}""",
        (keep, drop),
    )
    for table in ("wine_grapes", "wine_aromas", "wine_pairings", "wine_sensory"):
        conn.execute(f"DELETE FROM {table} WHERE wine_id = ?", (drop,))

    # Fill keep's missing details from drop, then remove drop.
    fills = ", ".join(
        f"{c} = COALESCE(wines.{c}, (SELECT {c} FROM wines WHERE wine_id = :drop))"
        for c in WINE_FILL_COLUMNS
    )
    conn.execute(
        f"""UPDATE wines SET
              color_inferred = CASE WHEN color IS NULL
                  THEN (SELECT color_inferred FROM wines WHERE wine_id = :drop)
                  ELSE color_inferred END,
              {fills}
            WHERE wine_id = :keep""",
        {"keep": keep, "drop": drop},
    )
    conn.execute("DELETE FROM wines WHERE wine_id = ?", (drop,))


def apply_changes(conn: sqlite3.Connection, plan: Plan) -> None:
    with conn:  # one transaction
        for keep, drop in plan.wine_merges:
            merge_wine(conn, keep, drop)
        for keep, drop in plan.winery_merges:
            conn.execute("UPDATE wines SET winery_id = ? WHERE winery_id = ?", (keep, drop))
            conn.execute(
                """UPDATE wineries SET website = COALESCE(website,
                       (SELECT website FROM wineries WHERE winery_id = ?))
                   WHERE winery_id = ?""",
                (drop, keep),
            )
            conn.execute("DELETE FROM wineries WHERE winery_id = ?", (drop,))
        merged_away = {drop for _, drop in plan.winery_merges}
        conn.executemany(
            "UPDATE wineries SET name = ? WHERE winery_id = ?",
            [(name, wid) for wid, name in plan.renames.items() if wid not in merged_away],
        )
        # Per-wine averages of individual user ratings now span the merged wines.
        for (source,) in conn.execute("SELECT DISTINCT source FROM user_ratings").fetchall():
            label = f"{source} users"
            conn.execute("DELETE FROM wine_ratings WHERE source = ?", (label,))
            conn.execute(
                """INSERT INTO wine_ratings (wine_id, vintage_id, score, scale, rating_count, source)
                   SELECT wine_id, NULL, ROUND(AVG(score * 1.0 / scale) * MAX(scale), 3),
                          MAX(scale), COUNT(*), ?
                   FROM user_ratings WHERE source = ? GROUP BY wine_id""",
                (label, source),
            )


def verify_references(conn: sqlite3.Connection, plan: Plan) -> dict:
    """Count rows that still reference a removed wine, vintage or winery (indexed lookups)."""
    dropped_wines = [drop for _, drop in plan.wine_merges]
    dropped_wineries = [drop for _, drop in plan.winery_merges]
    problems = {}
    for table in ("wine_sources", "wine_ratings", "wine_prices", "user_ratings",
                  "wine_vintages", "wine_grapes", "wine_aromas", "wine_pairings", "wine_sensory"):
        n = sum(
            conn.execute(f"SELECT COUNT(*) FROM {table} WHERE wine_id = ?", (w,)).fetchone()[0]
            for w in dropped_wines
        )
        if n:
            problems[table] = n
    # Vintage references must point at an existing vintage of the same wine.
    for table in ("wine_sources", "wine_ratings", "wine_prices"):
        n = conn.execute(
            f"""SELECT COUNT(*) FROM {table} t WHERE t.vintage_id IS NOT NULL AND NOT EXISTS
                (SELECT 1 FROM wine_vintages v WHERE v.vintage_id = t.vintage_id
                 AND v.wine_id = t.wine_id)"""
        ).fetchone()[0]
        if n:
            problems[f"{table}.vintage_id"] = n
    kept = {keep for keep, _ in plan.wine_merges}
    n = sum(
        conn.execute(
            """SELECT COUNT(*) FROM user_ratings u WHERE u.wine_id = ? AND u.vintage_id IS NOT NULL
               AND NOT EXISTS (SELECT 1 FROM wine_vintages v
                               WHERE v.vintage_id = u.vintage_id AND v.wine_id = u.wine_id)""",
            (w,),
        ).fetchone()[0]
        for w in kept
    )
    if n:
        problems["user_ratings.vintage_id"] = n
    n = sum(
        conn.execute("SELECT COUNT(*) FROM wines WHERE winery_id = ?", (w,)).fetchone()[0]
        for w in dropped_wineries
    )
    if n:
        problems["wines.winery_id"] = n
    return problems


def backup_database(db: Path) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f"wines_before_dedupe_{time.strftime('%Y%m%d_%H%M%S')}.db"
    src = sqlite3.connect(db)
    try:
        src.execute("VACUUM INTO ?", (str(path),))
    finally:
        src.close()
    return path


def write_report(plan: Plan) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / "dedupe_wine_merges.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["keep_wine_id", "keep_name", "keep_region", "drop_wine_id",
                        "drop_name", "drop_region", "color"],
        )
        writer.writeheader()
        writer.writerows(plan.report)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    parser.add_argument(
        "--skip-backup", action="store_true",
        help="do not copy the database first (only if you already have a copy)",
    )
    args = parser.parse_args(argv)
    if not args.db.exists():
        parser.error(f"database not found: {args.db}")

    ro = sqlite3.connect(f"file:{args.db.as_posix()}?mode=ro", uri=True)
    try:
        plan = plan_changes(ro)
    finally:
        ro.close()

    print(f"Winery rows to respell: {len(plan.renames):,}")
    print(f"Winery rows to merge (same producer, same region): {len(plan.winery_merges):,}")
    print(f"Wines to merge into another: {len(plan.wine_merges):,}")
    for reason, n in Counter(r for r, _ in plan.skipped).most_common():
        print(f"Skipped - {reason}: {n:,}")
    report = write_report(plan)
    print(f"Every wine merge is listed in {report}")

    if args.dry_run:
        print("\nDry run: nothing written.")
        return 0
    if not (plan.renames or plan.winery_merges or plan.wine_merges):
        print("\nNothing to change.")
        return 0

    if args.skip_backup:
        print("\nSkipping the backup (--skip-backup).")
    else:
        print("\nBacking up the database (a few minutes)...", flush=True)
        backup = backup_database(args.db)
        print(f"Backup: {backup}\n  (undo by copying it over {args.db})", flush=True)
    conn = sqlite3.connect(args.db)
    # Foreign-key enforcement is off during the merge: deleting a vintage would
    # otherwise scan all ~21M user_ratings (no index on vintage_id) per row.
    # merge_wine moves every reference first; verify_references() checks after.
    conn.execute("PRAGMA foreign_keys = OFF")
    started = time.time()
    try:
        apply_changes(conn, plan)
        problems = verify_references(conn, plan)
    finally:
        conn.close()
    print(f"Merged in {time.time() - started:.0f} s.")
    if problems:
        print("WARNING - rows still point at removed wines/wineries:", problems)
        print("Restore the backup if this is unexpected.")
    else:
        print("Checked: no rows point at a removed wine, vintage or winery.")
    print("Applied. Next: python -m ingestion.reparse_aromas   (aroma intensities)")
    print("      then: python -m app.review_analytics        (flavor profiles)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
