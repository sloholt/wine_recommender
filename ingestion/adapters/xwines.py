"""Adapter for the X-Wines dataset (https://github.com/rogerioxavier/X-Wines).

Two kinds of file:

* wines   (XWines_Full_100K_wines.csv, XWines_Slim_1K_wines.csv, ...):
  WineID, WineName, Type, Elaborate, Grapes, Harmonize, ABV, Body, Acidity, Code,
  Country, RegionID, RegionName, WineryID, WineryName, Website, Vintages
  -> ``read_records(path)`` yields WineRecord objects.
* ratings (XWines_Slim_150K_ratings.csv, ...):
  RatingID, UserID, WineID, Vintage, Rating, Date
  -> ``read_user_ratings(path)`` yields UserRating objects. Load the wines first:
  ratings are linked through the X-Wines WineID.

What it does that is specific to this source:
* ``Type`` mixes color and style ("Red", "Sparkling", "Dessert/Port"); it is split
  into style + color, and a missing color is derived only by the shared stated rules
  (core/naming.derive_color, e.g. "Rosé" in the name);
* ``Elaborate`` = "Varietal/100%" | "Varietal/>75%" | "Assemblage/<blend>":
  a varietal sets ``varietal`` to the first (main) grape, with 100 % when it is the
  only grape; a named assemblage ("Bordeaux Red Blend") becomes ``blend_type``;
* ``Body`` / ``Acidity`` labels map to the shared numeric scales (body 1-5,
  acidity 1-3);
* ``Harmonize`` becomes food pairings (names / categories in core/food.py);
* ``Vintages`` lists the years, with 'N.V.' for the non-vintage bottling;
* ABV 0 means "unknown"; rows above MAX_WINE_ABV are spirits (grappa, marc,
  eau-de-vie) that the dataset files under "Dessert", and are skipped;
* the X-Wines WineID is the source's external_id, so re-imports are safe.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Iterator, List

from ..core.models import Grape, Sensory, Source, UserRating, WineRecord
from ..core.naming import derive_color
from ..core.normalize import (
    ACIDITY_SCALE,
    BODY_SCALE,
    clean_text,
    fold,
    ordinal,
    parse_list,
    split_style_color,
    to_float,
)

LOG = logging.getLogger(__name__)
FILE_GLOB = "*_wines.csv"  # for --input <folder>; ratings files are separate

SOURCE_NAME = "X-Wines"
RATING_SCALE = 5.0
MAX_WINE_ABV = 25.0  # fortified wines reach ~22 %; anything stronger is a spirit


def _open(path):
    return open(Path(path), newline="", encoding="utf-8-sig")


def _grapes(raw, elaborate: str) -> tuple:
    """-> (grapes, varietal, blend_type) from the Grapes + Elaborate columns."""
    names = parse_list(raw)
    kind, _, detail = (elaborate or "").partition("/")
    kind, detail = fold(kind), clean_text(detail)

    varietal = blend_type = None
    grapes: List[Grape] = [Grape(n) for n in names]
    if kind == "varietal" and names:
        varietal = names[0]
        if detail == "100%" and len(names) == 1:
            grapes = [Grape(names[0], 100.0)]
    elif kind == "assemblage" and detail and fold(detail) != "blend":
        blend_type = detail
    if varietal is None and len(names) == 1:
        varietal = names[0]
    return grapes, varietal, blend_type


def read_records(path) -> Iterator[WineRecord]:
    spirits = 0
    with _open(path) as f:
        for line, row in enumerate(csv.DictReader(f), 2):
            name = clean_text(row.get("WineName"))
            wine_id = clean_text(row.get("WineID"))
            if not name or not wine_id:
                LOG.warning("%s line %d: no WineName / WineID, skipped", path, line)
                continue

            abv = to_float(row.get("ABV"), 0, 100)
            if abv is not None and abv > MAX_WINE_ABV:
                spirits += 1
                LOG.debug("Skipping spirit %r (%.1f%% ABV)", name, abv)
                continue
            if abv == 0:
                abv = None

            region = clean_text(row.get("RegionName"))
            style, color = split_style_color(row.get("Type"))
            color = derive_color(style, color, name, region)
            grapes, varietal, blend_type = _grapes(
                row.get("Grapes"), row.get("Elaborate")
            )

            yield WineRecord(
                name=name,
                winery=clean_text(row.get("WineryName")),
                winery_website=clean_text(row.get("Website")),
                country=clean_text(row.get("Country")),
                country_iso=clean_text(row.get("Code")),
                region=region,
                style=style,
                color=color,
                varietal=varietal,
                blend_type=blend_type,
                abv=abv,
                vintages=parse_list(row.get("Vintages")),
                grapes=grapes,
                sensory=Sensory(
                    body=ordinal(row.get("Body"), BODY_SCALE),
                    acidity=ordinal(row.get("Acidity"), ACIDITY_SCALE),
                ),
                pairings=parse_list(row.get("Harmonize")),
                sources=[Source(name=SOURCE_NAME, external_id=wine_id)],
            )
    if spirits:
        LOG.info("Skipped %d spirits (ABV > %.0f%%)", spirits, MAX_WINE_ABV)


def read_user_ratings(path) -> Iterator[UserRating]:
    with _open(path) as f:
        for row in csv.DictReader(f):
            vintage = clean_text(row.get("Vintage"))
            non_vintage = fold(vintage) in {"n v", "nv"}
            yield UserRating(
                source=SOURCE_NAME,
                user_id=row.get("UserID"),
                wine_external_id=row.get("WineID"),
                score=row.get("Rating"),
                scale=RATING_SCALE,
                vintage=None if non_vintage else vintage,
                non_vintage=non_vintage,
                rated_at=row.get("Date"),
                external_id=row.get("RatingID"),
            )
