"""Adapter for the Kaggle "Vivino famous wines taste profiles 2026" dataset
(mrbridge/vivino-famous-wines-taste-profiles-2026).

File: data.csv (~45 famous wines). Get it with
``python -m ingestion.adapters.famous_wines`` (downloads into data/raw/famous_wines/).

Columns:
    name, winery, wine_type, region, country, appellation, average_rating,
    ratings_count, grape_varieties (JSON list), taste_profile (JSON: body,
    acidity, tannins, sweetness, fizziness on 1-5, flavor_notes list),
    food_pairings (JSON list), wine_description, vivino_url, wineId, scrapedAt,
    searchQuery, alcohol, vintage ("2016" / "N.V."), price, vintageId

What it does that is specific to this source:
* ``wineId`` is a Vivino wine id, so the source is recorded as "Vivino" with that
  id: these wines line up exactly with the same wines from the Vivino ratings
  file (ingestion/adapters/vivino.py), whichever is loaded first;
* taste profile -> the shared sensory scales: body and tannin 1-5 as-is,
  acidity 1-5 -> 1-3, sweetness 1-5 -> 1-4 (same mapping as wine_info.py);
* flavor notes come grouped by Vivino category, most-mentioned first, and a
  famous wine has 65-300 of them. Each is mapped to the shared aroma vocabulary
  (core/tasting.py: "cigar" -> cigar box, Woody & tobacco); a note it does not
  know ("oak", "espresso") is kept with no category (Vivino's groups do not line
  up with the shared categories, so none is guessed). Only the first
  AROMAS_PER_CATEGORY of each category (uncategorized counts as one), MAX_AROMAS
  in all, are kept, so the strongest notes describe the wine instead of every
  note anyone mentioned;
* food pairings use Vivino's labels ("Game (deer, venison)"), mapped to the
  shared food names in core/food.py;
* rating -> 5-point Rating by "Vivino users": for the vintage when the Vivino
  link names a year, otherwise for the whole wine;
* ``wine_description`` is kept as a tasting note only when it is a real
  description: Vivino's boilerplate ("X is a Red wine. Made from ...") and
  bare body labels ("Medium-bodied") are dropped;
* price is NOT loaded: the dataset does not say which currency it is in.
"""

from __future__ import annotations

import csv
import html
import json
import logging
import re
from pathlib import Path
from typing import Iterator, List, Optional

from ..core.models import Aroma, Grape, Rating, Sensory, Source, WineRecord
from ..core.naming import derive_color
from ..core.normalize import (
    clean_note,
    clean_text,
    clean_vintage,
    fold,
    split_style_color,
    to_float,
    to_int,
)
from ..core.tasting import _aromas, _prep

LOG = logging.getLogger(__name__)

SOURCE_NAME = "Vivino"  # same ids as the Vivino ratings file
REVIEWER = "Vivino users"
KAGGLE_DATASET = "mrbridge/vivino-famous-wines-taste-profiles-2026"
FILE_GLOB = "data.csv"
RATING_SCALE = 5.0
AROMAS_PER_CATEGORY = 3
MAX_AROMAS = 24

_BOILERPLATE = re.compile(
    r"\bis an? (?:red|white|rose|rosé|sparkling|dessert|fortified) wine\b"
    r"|check its rating and reviews",
    re.IGNORECASE,
)
_BODY_ONLY = re.compile(r"^(?:very )?(?:light|medium|full)[- ]bodied$", re.IGNORECASE)
_URL_YEAR = re.compile(r"[?&]year=(\d{4})\b")


def _json(value, default):
    text = clean_text(value)
    if not text:
        return default
    try:
        return json.loads(text)
    except ValueError:
        return default


def _rescale(value, top: float) -> Optional[float]:
    """1..5 -> 1..top, linearly (continuous Vivino scores)."""
    number = to_float(value, 1, 5)
    if number is None:
        return None
    return round(1 + (number - 1) * (top - 1) / 4, 2)


def _aromas_from_notes(notes: List[str]) -> List[Aroma]:
    kept: List[Aroma] = []
    per_category: dict = {}
    seen = set()
    for note in notes or []:
        text = clean_text(note)
        if not text:
            continue
        known = _aromas(_prep(text))
        if known:
            name, category = known[0].name, known[0].category
        else:  # not in the shared vocabulary ("oak", "espresso"): no category
            name, category = text.lower(), None
        if name in seen:
            continue
        if per_category.get(category, 0) >= AROMAS_PER_CATEGORY:
            continue
        seen.add(name)
        per_category[category] = per_category.get(category, 0) + 1
        kept.append(Aroma(name=name, category=category))
        if len(kept) >= MAX_AROMAS:
            break
    return kept


def _description(value) -> Optional[str]:
    text = clean_note(html.unescape(value or ""))  # some are escaped twice
    if not text or _BOILERPLATE.search(text) or _BODY_ONLY.match(text):
        return None
    return text


def read_records(path) -> Iterator[WineRecord]:
    with open(Path(path), newline="", encoding="utf-8-sig") as f:
        for line, row in enumerate(csv.DictReader(f), 2):
            wine_id = to_int(row.get("wineId"), 1)
            name = clean_text(html.unescape(row.get("name") or ""))
            if wine_id is None or not name:
                LOG.warning("%s line %d: no wineId / name, skipped", path, line)
                continue
            winery = clean_text(html.unescape(row.get("winery") or ""))
            region = clean_text(row.get("region"))
            country = clean_text(row.get("country"))
            if region and country and fold(region) == fold(country):
                region = None

            raw_vintage = clean_text(row.get("vintage"))
            non_vintage = fold(raw_vintage) in {"n v", "nv"}
            vintage = None if non_vintage else clean_vintage(raw_vintage)
            url = clean_text(row.get("vivino_url"))
            url_year = _URL_YEAR.search(url or "")
            rating_vintage = vintage if (url_year or clean_text(row.get("vintageId"))) else None

            style, color = split_style_color(row.get("wine_type"))
            color = derive_color(style, color, name, region)

            profile = _json(row.get("taste_profile"), {})
            sensory = Sensory(
                body=_rescale(profile.get("body"), 5),
                tannin=_rescale(profile.get("tannins"), 5),
                acidity=_rescale(profile.get("acidity"), 3),
                sweetness=_rescale(profile.get("sweetness"), 4),
            )
            grapes = [Grape(g) for g in _json(row.get("grape_varieties"), []) if g]

            ratings = []
            score = to_float(row.get("average_rating"), 0, RATING_SCALE)
            if score:
                ratings.append(
                    Rating(
                        score=score,
                        scale=RATING_SCALE,
                        vintage=rating_vintage,
                        rating_count=to_int(row.get("ratings_count"), 0),
                        reviewer=REVIEWER,
                        source=SOURCE_NAME,
                    )
                )

            yield WineRecord(
                name=name,
                full_title=" ".join(p for p in (winery, name) if p),
                winery=winery,
                country=country,
                region=region,
                style=style,
                color=color,
                varietal=grapes[0].name if len(grapes) == 1 else None,
                abv=to_float(row.get("alcohol"), 0, 25),
                vintages=[vintage] if vintage else ([None] if non_vintage else []),
                grapes=grapes,
                aromas=_aromas_from_notes(profile.get("flavor_notes") or []),
                sensory=sensory,
                pairings=_json(row.get("food_pairings"), []),
                ratings=ratings,
                sources=[
                    Source(
                        name=SOURCE_NAME,
                        external_id=str(wine_id),
                        url=url,
                        tasting_note=_description(row.get("wine_description")),
                    )
                ],
            )


def download(dest=None) -> Path:
    """Download the dataset with kagglehub and copy its CSV into data/raw/famous_wines/."""
    import shutil

    import kagglehub

    dest = Path(
        dest or Path(__file__).resolve().parents[2] / "data" / "raw" / "famous_wines"
    )
    dest.mkdir(parents=True, exist_ok=True)
    for csv_file in Path(kagglehub.dataset_download(KAGGLE_DATASET)).glob("*.csv"):
        shutil.copy2(csv_file, dest / csv_file.name)
    return dest


if __name__ == "__main__":
    print("Dataset files in:", download())
