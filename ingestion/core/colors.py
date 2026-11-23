"""Fill a missing wine color from its grapes.

Many sources say "Pinot Noir" but not "Red". The grape -> color table is learned
from the database itself: every single-grape still wine whose color a source
*stated* is a vote. A grape gets a color when the votes are clear:

* at least MIN_WINES wines;
* red vs white: at least PURITY of the red + white votes go one way;
* rosé votes at most MAX_ROSE_SHARE of all votes (Grenache, Cinsault and other
  grapes often made into rosé stay unknown: a bare "Grenache" may well be rosé).

Hand-made corrections in data/reference/grape_colors.csv (columns grape,color)
take precedence over the learned table (color "" = never infer for this grape).

A wine gets a color only when:
* it has no color, or only a previously inferred one (stated colors are never
  touched), and is a still or dessert wine (not sparkling / fortified);
* every one of its grapes has a color, and they all agree;
* its title does not hint otherwise ("rosé", "blanc de noirs", "white" on a red
  grape, "orange", ...) or at a sparkling wine its source did not mark as one
  ("Brut", "Champagne", "Crémant", ...).

Checked against Vivino: of 162 inferred colors Vivino later stated, 156 agreed;
the misses were unmarked Pinot Noir Champagnes (now excluded by the rule above)
and late-harvest wines Vivino calls "Dessert" (dessert wines now get a color).

Inferred colors are flagged ``wines.color_inferred = 1``, re-computed on every
run (so they follow the latest table), never used as votes, and replaced as soon
as a source states the color.
"""

from __future__ import annotations

import csv
import logging
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

from .normalize import fold

LOG = logging.getLogger(__name__)

MIN_WINES = 5
PURITY = 0.95
MAX_ROSE_SHARE = 0.20
DEFAULT_OVERRIDES = (
    Path(__file__).resolve().parents[2] / "data" / "reference" / "grape_colors.csv"
)

_NOT_PLAIN = re.compile(
    r"\b(rose|rosado|rosato|blush|blanc de noirs|vin gris|orange|ramato|skin contact"
    r"|clairet|oeil de perdrix"
    # sparkling wines not marked as such by their source (a Pinot Noir Champagne
    # is white): no color from grapes
    r"|brut|sparkling|champagne|cremant|cava|prosecco|spumante|franciacorta|sekt"
    r"|mousseux|methode traditionnelle|pet nat|petillant|extra dry)\b"
)
_WHITE_WORDS = re.compile(r"\b(white|blanc|bianco|blanco|branco|weiss|weisser)\b")
_RED_WORDS = re.compile(r"\b(red|rouge|rosso|tinto|rot|roter)\b")


def load_overrides(path=None) -> Dict[str, Optional[str]]:
    """fold(grape) -> 'Red' / 'White' / None (None = never infer for this grape)."""
    path = Path(path or DEFAULT_OVERRIDES)
    if not path.exists():
        return {}
    result: Dict[str, Optional[str]] = {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            grape = fold(row.get("grape"))
            color = (row.get("color") or "").strip().title() or None
            if grape and color in (None, "Red", "White"):
                result[grape] = color
    return result


def learn_grape_colors(conn: sqlite3.Connection) -> Dict[int, str]:
    """grape_id -> 'Red' / 'White', from single-grape still wines with a stated color."""
    votes: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    rows = conn.execute(
        """SELECT wg.grape_id, w.color, COUNT(*) AS n
           FROM wines w JOIN wine_grapes wg ON wg.wine_id = w.wine_id
           WHERE w.style IS NULL AND w.color_inferred = 0
             AND w.color IN ('Red', 'White', 'Rosé')
             AND (SELECT COUNT(*) FROM wine_grapes x WHERE x.wine_id = w.wine_id) = 1
           GROUP BY wg.grape_id, w.color"""
    )
    for grape_id, color, n in rows:
        votes[grape_id][color] += n

    learned: Dict[int, str] = {}
    for grape_id, v in votes.items():
        red, white, rose = v["Red"], v["White"], v["Rosé"]
        total = red + white + rose
        if total < MIN_WINES or not (red + white) or rose / total > MAX_ROSE_SHARE:
            continue
        if red / (red + white) >= PURITY:
            learned[grape_id] = "Red"
        elif white / (red + white) >= PURITY:
            learned[grape_id] = "White"
    return learned


def infer_colors(conn: sqlite3.Connection, overrides_path=None) -> int:
    """Re-compute every inferred color. Returns how many wines now have one."""
    table = learn_grape_colors(conn)
    overrides = load_overrides(overrides_path)
    if overrides:
        for grape_id, name in conn.execute("SELECT grape_id, name FROM grapes"):
            key = fold(name)
            if key in overrides:
                if overrides[key]:
                    table[grape_id] = overrides[key]
                else:
                    table.pop(grape_id, None)

    conn.execute(
        "UPDATE wines SET color = NULL, color_inferred = 0 WHERE color_inferred = 1"
    )
    updates = []
    rows = conn.execute(
        """SELECT w.wine_id, COALESCE(w.full_title, w.name) AS title,
                  GROUP_CONCAT(wg.grape_id) AS grape_ids
           FROM wines w JOIN wine_grapes wg ON wg.wine_id = w.wine_id
           WHERE w.color IS NULL AND (w.style IS NULL OR w.style = 'Dessert')
           GROUP BY w.wine_id"""
    )
    for wine_id, title, grape_ids in rows:
        colors = {table.get(int(g)) for g in grape_ids.split(",")}
        if len(colors) != 1 or None in colors:
            continue
        color = colors.pop()
        text = fold(title)
        if _NOT_PLAIN.search(text):
            continue
        if color == "Red" and _WHITE_WORDS.search(text):
            continue
        if color == "White" and _RED_WORDS.search(text):
            continue
        updates.append((color, wine_id))

    conn.executemany(
        "UPDATE wines SET color = ?, color_inferred = 1 WHERE wine_id = ?", updates
    )
    LOG.info(
        "Grape colors: %d grapes known; color inferred for %d wines",
        len(table),
        len(updates),
    )
    return len(updates)
