"""
Read and write helpers for the wine database (schema.sql).

Scripts and the app call these functions instead of writing SQL themselves:

    from database import repository as repo

    repo.search_wines(grape="Syrah", country="France", limit=10)
    repo.get_wine(1234)                  # one wine with grapes, aromas, ratings ...
    repo.get_wine_features()             # one row per wine, for recommendation models
    repo.get_reviews(source="Wine Enthusiast")

Read functions return pandas DataFrames (many rows) or plain dicts (one wine).
Run scripts from the project folder with ``python -m ...`` so ``database`` imports.

Adding wines goes through the ingestion pipeline (``ingestion/core/db.py``), which
owns the duplicate-matching rules; ``add_wine`` here is a thin wrapper around it.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Iterable, Optional

import pandas as pd

from .connection import DATABASE_PATH, get_connection


# --------------------------------------------------
# HELPERS
# --------------------------------------------------


@contextmanager
def _connect():
    """Open the database, commit on success / roll back on error, always close.

    (``with sqlite3.connect(...)`` alone commits but never closes the connection.)
    """
    if not DATABASE_PATH.exists():
        # sqlite3 would otherwise create an empty database file silently.
        raise FileNotFoundError(
            f"Database not found: {DATABASE_PATH} (build it with `python -m ingestion.run`)"
        )
    conn = get_connection()
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def _ids_param(wine_ids: Iterable[int]) -> str:
    """Pass any number of ids as one JSON parameter (no SQLite variable limit).

    Use in SQL as:  wine_id IN (SELECT value FROM json_each(?))
    """
    return json.dumps([int(i) for i in wine_ids])


def _like(text: str) -> str:
    return f"%{text.strip()}%"


def variety_label(varietal, blend_type, grapes, style, color) -> Optional[str]:
    """One grape/blend label per wine, in the style of wines.csv's 'variety'.

    The database splits that single text into varietal, blend_type, grapes
    (``'|'``-separated, sorted), style and color; this puts it back together so
    e.g. 'Red Blend' and 'Cabernet Sauvignon' stay separate groups.
    """
    if pd.notna(varietal):
        return varietal
    if pd.notna(blend_type):
        return blend_type
    names = grapes.split("|") if pd.notna(grapes) and grapes else []
    if len(names) > 1:
        return "-".join(names)  # e.g. 'Merlot-Tempranillo'
    if len(names) == 1:
        return f"{names[0]} Blend"  # e.g. 'Tempranillo Blend'
    if style == "Sparkling":
        return "Sparkling Blend"
    if color == "Rosé":
        return "Rosé"
    if pd.notna(color):
        return f"{color} Blend"  # 'Red Blend' / 'White Blend'
    return None


def _add_variety(df: pd.DataFrame) -> pd.DataFrame:
    df["variety"] = [
        variety_label(r.varietal, r.blend_type, r.grapes, r.style, r.color)
        for r in df[["varietal", "blend_type", "grapes", "style", "color"]].itertuples(
            index=False
        )
    ]
    return df


# Grapes of each wine as one sorted 'A|B|C' string (CTE, reused below).
_GRAPES_CTE = """
wine_grape_list AS (
    SELECT wine_id, group_concat(name, '|') AS grapes
    FROM (SELECT wg.wine_id, g.name
          FROM wine_grapes wg JOIN grapes g ON g.grape_id = wg.grape_id
          ORDER BY wg.wine_id, g.name)
    GROUP BY wine_id
)
"""

# Columns describing a wine, used by several queries (needs aliases w, wy, r, c).
_WINE_COLUMNS = """
    w.wine_id,
    COALESCE(w.full_title, w.name) AS title,
    w.name,
    wy.name  AS winery,
    r.name   AS region,
    c.name   AS country,
    w.color,
    w.style,
    w.varietal,
    w.blend_type,
    w.abv
"""

_WINE_JOINS = """
LEFT JOIN wineries wy ON wy.winery_id = w.winery_id
LEFT JOIN regions r   ON r.region_id = w.region_id
LEFT JOIN countries c ON c.country_id = r.country_id
"""


# --------------------------------------------------
# READ
# --------------------------------------------------


def count_wines() -> int:
    """Total number of wines (products, not vintages)."""
    with _connect() as conn:
        return conn.execute("SELECT COUNT(*) FROM wines").fetchone()[0]


def get_wine(wine_id: int) -> Optional[dict]:
    """One wine with everything known about it, or None if the id does not exist."""
    with _connect() as conn:
        row = conn.execute(
            f"SELECT {_WINE_COLUMNS}, w.closure, w.color_inferred "
            f"FROM wines w {_WINE_JOINS} WHERE w.wine_id = ?",
            (wine_id,),
        ).fetchone()
        if row is None:
            return None
        wine = dict(row)

        def rows(sql):
            return [dict(r) for r in conn.execute(sql, (wine_id,))]

        wine["vintages"] = [
            r["vintage"]
            for r in rows(
                "SELECT vintage FROM wine_vintages WHERE wine_id = ? ORDER BY vintage"
            )
        ]
        wine["grapes"] = rows(
            """SELECT g.name, wg.blend_percentage
               FROM wine_grapes wg JOIN grapes g ON g.grape_id = wg.grape_id
               WHERE wg.wine_id = ?
               ORDER BY wg.blend_percentage DESC, g.name"""
        )
        sensory = conn.execute(
            """SELECT body, acidity, structure, finish, balance, tannin, sweetness
               FROM wine_sensory WHERE wine_id = ?""",
            (wine_id,),
        ).fetchone()
        wine["sensory"] = dict(sensory) if sensory else None
        wine["aromas"] = rows(
            """SELECT aroma_category AS category, aroma_name AS name, intensity
               FROM wine_aromas WHERE wine_id = ?
               ORDER BY aroma_category, aroma_name"""
        )
        wine["pairings"] = rows(
            """SELECT f.name, f.category
               FROM wine_pairings p JOIN foods f ON f.food_id = p.food_id
               WHERE p.wine_id = ? ORDER BY f.category, f.name"""
        )
        wine["ratings"] = rows(
            """SELECT v.vintage, r.score, r.scale, r.score_pct, r.rating_count,
                      r.reviewer, r.source
               FROM wine_ratings r
               LEFT JOIN wine_vintages v ON v.vintage_id = r.vintage_id
               WHERE r.wine_id = ? ORDER BY v.vintage DESC, r.source"""
        )
        wine["prices"] = rows(
            """SELECT v.vintage, p.price, p.currency, p.retailer, p.observed_at
               FROM wine_prices p
               LEFT JOIN wine_vintages v ON v.vintage_id = p.vintage_id
               WHERE p.wine_id = ? ORDER BY v.vintage DESC"""
        )
        wine["reviews"] = rows(
            """SELECT s.source_name AS source, v.vintage, s.tasting_note AS review
               FROM wine_sources s
               LEFT JOIN wine_vintages v ON v.vintage_id = s.vintage_id
               WHERE s.wine_id = ? AND TRIM(COALESCE(s.tasting_note, '')) <> ''"""
        )
    return wine


def search_wines(
    name: Optional[str] = None,
    winery: Optional[str] = None,
    grape: Optional[str] = None,
    region: Optional[str] = None,
    country: Optional[str] = None,
    color: Optional[str] = None,
    style: Optional[str] = None,
    limit: int = 50,
) -> pd.DataFrame:
    """Find wines by any combination of filters (partial, case-insensitive text match).

    ``grape`` matches the single varietal or any grape in a blend.
    ``region`` matches the wine's region or its parent region
    (so 'Bordeaux' also finds wines from 'Pauillac' if Pauillac sits under Bordeaux).
    ``color`` / ``style`` must match exactly ('Red', 'Rosé'; 'Sparkling', ...).
    """
    where, params = [], []
    if name:
        where.append("(w.name LIKE ? OR w.full_title LIKE ?)")
        params += [_like(name), _like(name)]
    if winery:
        where.append("wy.name LIKE ?")
        params.append(_like(winery))
    if grape:
        where.append(
            """(w.varietal LIKE ? OR EXISTS (
                   SELECT 1 FROM wine_grapes wg JOIN grapes g ON g.grape_id = wg.grape_id
                   WHERE wg.wine_id = w.wine_id AND g.name LIKE ?))"""
        )
        params += [_like(grape), _like(grape)]
    if region:
        where.append(
            """(r.name LIKE ? OR EXISTS (
                   SELECT 1 FROM regions pr
                   WHERE pr.region_id = r.parent_region_id AND pr.name LIKE ?))"""
        )
        params += [_like(region), _like(region)]
    if country:
        where.append("c.name LIKE ?")
        params.append(_like(country))
    if color:
        where.append("w.color = ? COLLATE NOCASE")
        params.append(color)
    if style:
        where.append("w.style = ? COLLATE NOCASE")
        params.append(style)

    sql = f"SELECT {_WINE_COLUMNS} FROM wines w {_WINE_JOINS}"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY w.name LIMIT ?"
    params.append(int(limit))

    with _connect() as conn:
        return pd.read_sql_query(sql, conn, params=params)


def get_wine_features(wine_ids: Optional[Iterable[int]] = None) -> pd.DataFrame:
    """One row per wine: description, grapes, sensory profile, scores and price.

    Intended for similarity and recommendation models. Columns:
      wine_id, title, name, winery, region, country, color, style, varietal,
      blend_type, abv, grapes ('A|B'), variety,
      body, acidity, tannin, sweetness, finish        (NULL = unknown, not zero)
      critic_score (mean points /100), n_critic_reviews,
      user_score   (mean stars /5),    n_user_ratings,
      price_usd    (mean observed USD price), n_aromas
    """
    sql = f"""
    WITH {_GRAPES_CTE},
    critic AS (
        SELECT wine_id, AVG(score_pct) AS critic_score, COUNT(*) AS n_critic_reviews
        FROM wine_ratings WHERE scale = 100 GROUP BY wine_id
    ),
    users AS (
        -- per-wine averages; weight each by how many ratings it summarises
        SELECT wine_id,
               SUM(score * COALESCE(rating_count, 1)) / SUM(COALESCE(rating_count, 1))
                   AS user_score,
               SUM(COALESCE(rating_count, 1)) AS n_user_ratings
        FROM wine_ratings WHERE scale = 5 GROUP BY wine_id
    ),
    price AS (
        SELECT wine_id, AVG(price) AS price_usd
        FROM wine_prices WHERE currency = 'USD' GROUP BY wine_id
    ),
    aroma_n AS (
        SELECT wine_id, COUNT(*) AS n_aromas FROM wine_aromas GROUP BY wine_id
    )
    SELECT {_WINE_COLUMNS},
           gl.grapes,
           s.body, s.acidity, s.tannin, s.sweetness, s.finish,
           ROUND(cr.critic_score, 1)  AS critic_score,
           COALESCE(cr.n_critic_reviews, 0) AS n_critic_reviews,
           ROUND(u.user_score, 2)     AS user_score,
           COALESCE(u.n_user_ratings, 0) AS n_user_ratings,
           ROUND(p.price_usd, 2)      AS price_usd,
           COALESCE(a.n_aromas, 0)    AS n_aromas
    FROM wines w
    {_WINE_JOINS}
    LEFT JOIN wine_grape_list gl ON gl.wine_id = w.wine_id
    LEFT JOIN wine_sensory s     ON s.wine_id = w.wine_id
    LEFT JOIN critic cr          ON cr.wine_id = w.wine_id
    LEFT JOIN users u            ON u.wine_id = w.wine_id
    LEFT JOIN price p            ON p.wine_id = w.wine_id
    LEFT JOIN aroma_n a          ON a.wine_id = w.wine_id
    """
    params: list = []
    if wine_ids is not None:
        sql += " WHERE w.wine_id IN (SELECT value FROM json_each(?))"
        params.append(_ids_param(wine_ids))
    sql += " ORDER BY w.wine_id"

    with _connect() as conn:
        df = pd.read_sql_query(sql, conn, params=params)
    return _add_variety(df)


def get_aromas(wine_ids: Optional[Iterable[int]] = None) -> pd.DataFrame:
    """Aromas in long format: wine_id, category, name, intensity (one row per aroma)."""
    sql = """SELECT wine_id, aroma_category AS category, aroma_name AS name, intensity
             FROM wine_aromas"""
    params: list = []
    if wine_ids is not None:
        sql += " WHERE wine_id IN (SELECT value FROM json_each(?))"
        params.append(_ids_param(wine_ids))
    with _connect() as conn:
        return pd.read_sql_query(sql, conn, params=params)


def get_reviews(source: Optional[str] = None) -> pd.DataFrame:
    """One row per review (tasting note), with the wine's description and 'variety'.

    A wine reviewed by several sources (or for several vintages) appears once per
    review. ``source`` limits to one source, e.g. 'Wine Enthusiast'.
    """
    sql = f"""
    WITH {_GRAPES_CTE}
    SELECT s.wine_id,
           s.vintage_id,
           s.source_name AS source,
           v.vintage,
           {_WINE_COLUMNS.replace("w.wine_id,", "")},
           gl.grapes,
           s.tasting_note AS review
    FROM wine_sources s
    JOIN wines w ON w.wine_id = s.wine_id
    {_WINE_JOINS}
    LEFT JOIN wine_vintages v    ON v.vintage_id = s.vintage_id
    LEFT JOIN wine_grape_list gl ON gl.wine_id = w.wine_id
    WHERE TRIM(COALESCE(s.tasting_note, '')) <> ''
    """
    params: list = []
    if source:
        sql += " AND s.source_name = ? COLLATE NOCASE"
        params.append(source)

    with _connect() as conn:
        df = pd.read_sql_query(sql, conn, params=params)
    return _add_variety(df)


def get_wines_for_food(
    food: str, color: Optional[str] = None, min_ratings: int = 10, limit: int = 50
) -> pd.DataFrame:
    """Wines paired with a food, best rated first.

    ``food`` matches a food name ('Goat Cheese') or a whole category ('Cheese').
    Only wines with at least ``min_ratings`` user ratings or one critic score are
    ranked, so a single 5-star rating does not put a wine on top.
    """
    sql = f"""
    WITH matched AS (
        SELECT DISTINCT p.wine_id
        FROM wine_pairings p JOIN foods f ON f.food_id = p.food_id
        WHERE f.name LIKE ? OR f.category LIKE ?
    ),
    scores AS (
        SELECT wine_id,
               AVG(score_pct) AS score_pct,
               SUM(CASE WHEN scale = 5 THEN COALESCE(rating_count, 1) ELSE 0 END)
                   AS n_user_ratings,
               SUM(scale = 100) AS n_critic_reviews
        FROM wine_ratings
        WHERE wine_id IN (SELECT wine_id FROM matched)
        GROUP BY wine_id
    )
    SELECT {_WINE_COLUMNS},
           ROUND(sc.score_pct, 1) AS score_pct,
           sc.n_user_ratings,
           sc.n_critic_reviews
    FROM matched m
    JOIN wines w  ON w.wine_id = m.wine_id
    JOIN scores sc ON sc.wine_id = w.wine_id
    {_WINE_JOINS}
    WHERE (sc.n_user_ratings >= ? OR sc.n_critic_reviews > 0)
    """
    params: list = [_like(food), _like(food), int(min_ratings)]
    if color:
        sql += " AND w.color = ? COLLATE NOCASE"
        params.append(color)
    sql += " ORDER BY sc.score_pct DESC, sc.n_user_ratings DESC LIMIT ?"
    params.append(int(limit))

    with _connect() as conn:
        return pd.read_sql_query(sql, conn, params=params)


def get_user_ratings(
    source: Optional[str] = None, min_ratings_per_user: int = 1
) -> pd.DataFrame:
    """Individual user ratings (user_id, wine_id, score, scale) for collaborative filtering.

    The full table is ~21 million rows: filter with ``source`` and/or
    ``min_ratings_per_user`` (drops users with too few ratings to learn from).
    """
    where, params = [], []
    if source:
        where.append("source = ? COLLATE NOCASE")
        params.append(source)
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    sql = f"""
    SELECT source, user_id, wine_id, score, scale
    FROM user_ratings
    {where_sql}
    """
    if min_ratings_per_user > 1:
        sql = f"""
        WITH active AS (
            SELECT source, user_id FROM user_ratings {where_sql}
            GROUP BY source, user_id HAVING COUNT(*) >= ?
        )
        SELECT ur.source, ur.user_id, ur.wine_id, ur.score, ur.scale
        FROM user_ratings ur
        JOIN active a ON a.source = ur.source AND a.user_id = ur.user_id
        """
        params = params + [int(min_ratings_per_user)]

    with _connect() as conn:
        return pd.read_sql_query(sql, conn, params=params)


# --------------------------------------------------
# CREATE
# --------------------------------------------------


def add_wine(record) -> int:
    """Add (or merge) one wine, given an ``ingestion.core.models.WineRecord``.

    Uses the ingestion pipeline so the same duplicate-matching rules apply: if the
    wine already exists it is matched and missing fields are filled in. Opening
    the pipeline loads lookup caches (a few seconds), so for many wines use
    ``python -m ingestion.run`` or ``WineDB.ingest_many`` instead.

    Returns the wine_id.
    """
    from ingestion.core.db import WineDB  # imported here: only needed for writes

    with WineDB(DATABASE_PATH) as db:
        wine_id, _created = db.ingest(record)
    return wine_id


# --------------------------------------------------
# UPDATE
# --------------------------------------------------

_UPDATABLE_WINE_FIELDS = {
    "name",
    "full_title",
    "style",
    "color",
    "varietal",
    "blend_type",
    "abv",
    "closure",
}
_SENSORY_FIELDS = {"body", "acidity", "structure", "finish", "balance", "tannin", "sweetness"}


def update_wine(wine_id: int, **fields) -> bool:
    """Change a wine's own columns, e.g. ``update_wine(12, abv=13.5, closure='Screwcap')``.

    Returns False if the wine does not exist. Setting ``color`` by hand also
    clears ``color_inferred``.
    """
    unknown = set(fields) - _UPDATABLE_WINE_FIELDS
    if unknown:
        raise ValueError(
            f"Invalid wine fields: {sorted(unknown)}; allowed: {sorted(_UPDATABLE_WINE_FIELDS)}"
        )
    if not fields:
        return False
    if "color" in fields:
        fields["color_inferred"] = 0

    assignments = ", ".join(f"{k} = ?" for k in fields)  # keys are whitelisted above
    with _connect() as conn:
        cur = conn.execute(
            f"UPDATE wines SET {assignments} WHERE wine_id = ?",
            [*fields.values(), wine_id],
        )
        return cur.rowcount > 0


def update_sensory(wine_id: int, **fields) -> bool:
    """Add or change a wine's sensory values, e.g. ``update_sensory(12, body=4, tannin=3)``.

    Scales: body 1-5, acidity 1-3, tannin 1-5, finish 1-5, sweetness 1-4.
    Returns False if the wine does not exist.
    """
    unknown = set(fields) - _SENSORY_FIELDS
    if unknown:
        raise ValueError(
            f"Invalid sensory fields: {sorted(unknown)}; allowed: {sorted(_SENSORY_FIELDS)}"
        )
    if not fields:
        return False

    cols = ", ".join(fields)
    marks = ", ".join("?" for _ in fields)
    updates = ", ".join(f"{k} = excluded.{k}" for k in fields)
    with _connect() as conn:
        if conn.execute("SELECT 1 FROM wines WHERE wine_id = ?", (wine_id,)).fetchone() is None:
            return False
        conn.execute(
            f"""INSERT INTO wine_sensory (wine_id, {cols}) VALUES (?, {marks})
                ON CONFLICT(wine_id) DO UPDATE SET {updates}""",
            [wine_id, *fields.values()],
        )
        return True


# --------------------------------------------------
# DELETE
# --------------------------------------------------

# Every table that points at wines, children before parents
# (the schema has no ON DELETE CASCADE).
_WINE_CHILD_TABLES = [
    "user_ratings",
    "wine_ratings",
    "wine_prices",
    "wine_sources",
    "wine_pairings",
    "wine_aromas",
    "wine_sensory",
    "wine_grapes",
    "wine_vintages",
]


def delete_wine(wine_id: int) -> bool:
    """Delete a wine and every row linked to it, in one transaction.

    Returns False if the wine does not exist. Note: re-running an import of the
    source that contained this wine will add it back.
    """
    with _connect() as conn:
        if conn.execute("SELECT 1 FROM wines WHERE wine_id = ?", (wine_id,)).fetchone() is None:
            return False
        for table in _WINE_CHILD_TABLES:
            conn.execute(f"DELETE FROM {table} WHERE wine_id = ?", (wine_id,))
        conn.execute("DELETE FROM wines WHERE wine_id = ?", (wine_id,))
        return True
