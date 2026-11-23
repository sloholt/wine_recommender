"""Adapter for data/wines.csv: the Wine Enthusiast reviews dataset (~130K reviews).

Columns: <index>, country, description, designation, points, price, province,
region_1, region_2, taster_name, taster_twitter_handle, title, variety, winery

One row is one critic review of one vintage. ``title`` is always
"<winery> [<year>] <name> (<region>)", e.g. "Nicosia 2013 Vulkà Bianco  (Etna)".

What it does that is specific to this source:
* vintage = the year directly after the winery in the title (so "San Pedro 2014
  1865 Single Vineyard" is 2014, and "Vignes de 1939" is not a vintage);
* wine name = the title without winery, vintage and the "(region)" suffix;
* ``variety`` is either grapes ("Pinot Noir", "Cabernet Sauvignon-Merlot",
  "Corvina, Rondinella, Molinara", "G-S-M") or a style label ("Bordeaux-style Red
  Blend", "Rosé", "Port", "Champagne Blend"). Labels become color / style /
  blend_type, named like X-Wines' blends ("Bordeaux Red Blend") so they line up;
  ambiguous half-names in blends ("Tannat-Cabernet") are not turned into grapes;
* points -> a 100-point Rating by the taster; price -> USD Price;
* the review text is the tasting note, parsed for aromas, body / acidity /
  tannin / finish / sweetness and grapes (core/tasting.py), like the critic lists;
* region = region_1 (e.g. Napa Valley), else province (e.g. Bordeaux);
* external_id = hash of title + review, so the ~10K exact duplicate rows in the
  file collapse into one review and re-imports are safe.
"""

from __future__ import annotations

import csv
import hashlib
import logging
import re
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from ..core.models import Grape, Price, Rating, Source, WineRecord
from ..core.naming import derive_color, is_non_vintage
from ..core.normalize import clean_note, clean_text, clean_vintage, fold, to_float
from ..core.tasting import parse_tasting_note

LOG = logging.getLogger(__name__)

SOURCE_NAME = "Wine Enthusiast"
FILE_GLOB = "wines.csv"
DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "wines.csv"

# fold(label) -> (style, color, blend_type)
VARIETY_LABELS = {
    "red blend": (None, "Red", None),
    "white blend": (None, "White", None),
    "cabernet blend": (None, "Red", None),
    "bordeaux style red blend": (None, "Red", "Bordeaux Red Blend"),
    "bordeaux style white blend": (None, "White", "Bordeaux White Blend"),
    "rhone style red blend": (None, "Red", "Rhône Red Blend"),
    "rhone style white blend": (None, "White", "Rhône White Blend"),
    "provence red blend": (None, "Red", "Provence Red Blend"),
    "provence white blend": (None, "White", "Provence White Blend"),
    "austrian red blend": (None, "Red", "Austrian Red Blend"),
    "austrian white blend": (None, "White", "Austrian White Blend"),
    "alsace white blend": (None, "White", "Alsace White Blend"),
    "portuguese red": (None, "Red", "Portuguese Red Blend"),
    "portuguese white": (None, "White", "Portuguese White Blend"),
    "portuguese rose": (None, "Rosé", None),
    "portuguese sparkling": ("Sparkling", None, None),
    "sparkling blend": ("Sparkling", None, None),
    "champagne blend": ("Sparkling", None, "Champagne Blend"),
    "rose": (None, "Rosé", None),
    "rosato": (None, "Rosé", None),
    "rosado": (None, "Rosé", None),
    "port": ("Fortified", None, "Port Blend"),
    "white port": ("Fortified", "White", "Port Blend"),
    "madeira blend": ("Fortified", None, "Madeira Blend"),
}
# Single grapes whose names contain a hyphen (never split).
HYPHENATED_GRAPES = {"muller thurgau", "xarel lo"}
# Half-names that appear in blends ("Tannat-Cabernet"): which grape is meant is
# not stated, so they are not stored as grapes.
AMBIGUOUS_PARTS = {"cabernet", "sauvignon", "pinot"}
# "G-S-M" and friends.
VARIETY_GRAPES = {
    "g s m": ["Grenache", "Syrah", "Mourvèdre"],
    "gsm": ["Grenache", "Syrah", "Mourvèdre"],
    "gros and petit manseng": ["Gros Manseng", "Petit Manseng"],
}
SPARKLING_VARIETIES = {"prosecco", "glera"}
# Title / region words that mark a sparkling wine whose variety does not say so
# ("Jean Josselin NV Cordon Royal Brut Pinot Noir (Champagne)").
_SPARKLING_CUES = re.compile(
    r"\b(champagne|brut|sparkling|cremant|cava|spumante|franciacorta|sekt"
    r"|mousseux|methode traditionnelle|extra dry|pet nat)\b"
)

_LEADING_YEAR = re.compile(r"^(?P<y>(?:18|19|20)\d{2})(?!\d)\s*")
_LEADING_NV = re.compile(r"^(?:NV|N\.V\.)(?=\s|$)\s*", re.IGNORECASE)
_REGION_SUFFIX = re.compile(r"\s*\([^()]*\)\s*$")


def _variety(raw) -> Tuple[List[str], Optional[str], Optional[str], Optional[str], bool]:
    """-> (grapes, style, color, blend_type, is_blend) for one ``variety`` value."""
    label = clean_text(raw)
    if not label:
        return [], None, None, None, False
    key = fold(label)
    if key in VARIETY_LABELS:
        style, color, blend = VARIETY_LABELS[key]
        return [], style, color, blend, True
    if key in VARIETY_GRAPES:
        return list(VARIETY_GRAPES[key]), None, None, None, True
    style = "Sparkling" if key in SPARKLING_VARIETIES else None
    if key.endswith(" blend"):  # "Tempranillo Blend": main grape, but a blend
        return [label[: -len(" blend")].strip()], style, None, None, True
    if key in HYPHENATED_GRAPES:
        return [label], style, None, None, False
    parts = [p.strip() for p in re.split(r"\s*,\s*|-", label) if p.strip()]
    grapes = [p for p in parts if fold(p) not in AMBIGUOUS_PARTS]
    return grapes, style, None, None, len(parts) > 1


def _split_title(title: str, winery: Optional[str]) -> Tuple[str, Optional[int], bool]:
    """-> (wine name, vintage, says NV) from "<winery> [<year>] <name> (<region>)"."""
    rest = title
    if winery and fold(title).startswith(fold(winery)):
        rest = title[len(winery):] if title.startswith(winery) else _drop_words(
            title, len(winery.split())
        )
    rest = rest.strip()
    vintage, nv = None, False
    match = _LEADING_YEAR.match(rest)
    if match:
        vintage = clean_vintage(match["y"])
        rest = rest[match.end():]
    elif _LEADING_NV.match(rest):
        nv = True
        rest = _LEADING_NV.sub("", rest)
    rest = _REGION_SUFFIX.sub("", rest)
    return clean_text(rest) or "", vintage, nv


def _drop_words(text: str, count: int) -> str:
    return " ".join(text.split()[count:])


def _review_id(title: str, note: Optional[str]) -> str:
    digest = hashlib.sha1(f"{title}\x1f{note or ''}".encode("utf-8")).hexdigest()
    return digest[:16]


def read_records(path=DEFAULT_PATH) -> Iterator[WineRecord]:
    with open(Path(path), newline="", encoding="utf-8-sig") as f:
        for line, row in enumerate(csv.DictReader(f), 2):
            title = clean_text(row.get("title"))
            winery = clean_text(row.get("winery"))
            if not title:
                LOG.warning("%s line %d: no title, skipped", path, line)
                continue

            name, vintage, says_nv = _split_title(title, winery)
            grapes, style, color, blend_type, is_blend = _variety(row.get("variety"))
            name = name or clean_text(row.get("designation")) or clean_text(
                row.get("variety")
            ) or title

            country = clean_text(row.get("country"))  # "US", "England": normalizer
            region = clean_text(row.get("region_1")) or clean_text(row.get("province"))
            if region and country and fold(region) == fold(country):
                region = None

            if style is None and _SPARKLING_CUES.search(fold(f"{title} {region or ''}")):
                style = "Sparkling"

            note = clean_note(row.get("description"))
            info = parse_tasting_note(note, title)
            if not grapes:  # generic blend label: use grapes the review names
                grapes = [g.name for g in info.grapes]
            color = derive_color(style, color, title, region)
            non_vintage = says_nv or is_non_vintage(vintage, title, style, note)

            taster = clean_text(row.get("taster_name"))
            ratings = []
            points = to_float(row.get("points"), 0, 100)
            if points is not None:
                ratings.append(
                    Rating(
                        score=points,
                        scale=100,
                        vintage=vintage,
                        reviewer=taster or SOURCE_NAME,
                        source=SOURCE_NAME,
                    )
                )
            prices = []
            price = to_float(row.get("price"), 0)
            if price:
                prices.append(Price(price=price, currency="USD", vintage=vintage))

            yield WineRecord(
                name=name,
                full_title=title,
                winery=winery,
                country=country,
                region=region,
                style=style,
                color=color,
                varietal=grapes[0] if len(grapes) == 1 and not is_blend else None,
                blend_type=blend_type,
                vintages=[vintage] if vintage else ([None] if non_vintage else []),
                grapes=[Grape(g) for g in grapes],
                aromas=info.aromas,
                sensory=info.sensory,
                ratings=ratings,
                prices=prices,
                sources=[
                    Source(
                        name=SOURCE_NAME,
                        external_id=_review_id(title, note),
                        tasting_note=note,
                        vintage=vintage,
                    )
                ],
            )
