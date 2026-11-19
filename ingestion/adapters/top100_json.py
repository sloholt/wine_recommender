"""Adapter for scraped critic "Top 100" JSON files.

Expected input: a JSON list of objects with (any may be null) rank, title, winery,
varietal, score, color, country, region, vintage, price, currency, tasting_note,
tasting_note_truncated, source_name, source_url.

What it does that is specific to this source:
* strips a trailing vintage from the title when it equals the vintage field, so
  every year of a wine unifies into one ``wines`` row;
* drops a region that only repeats the country ("Austria" / "Austria");
* parses the tasting note into aromas, body / acidity / tannin / finish /
  sweetness and grape blend (see core/tasting.py);
* sets ``varietal`` only when exactly one grape is known;
* uses "<file name>#<rank>" as the source's external_id so re-imports are safe.

* splits the published title into winery + wine name (core/naming.py, with the
  hand-made fixes in data/reference/title_overrides.csv) and keeps the original
  title as ``full_title``;
* separates style (Sparkling / Dessert) from color (Red / White / Rosé), deriving a
  missing color only by stated rules (Champagne -> White, "Rosé" in the title -> Rosé);
* marks multi-vintage blends (no year + "N.26" / "171ème Edition" / note says
  "blend of 2012, 2008 and 2007" / sparkling without a year) as non-vintage.

A winery is only set when the title gives one; an unsplittable title is kept whole.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator

from ..core.models import Price, Rating, Source, WineRecord
from ..core.naming import derive_color, is_non_vintage, load_overrides, split_wine_title
from ..core.normalize import (
    clean_note,
    clean_text,
    clean_vintage,
    fold,
    split_style_color,
    to_float,
)
from ..core.tasting import parse_tasting_note

LOG = logging.getLogger(__name__)


def _strip_vintage(title: str, vintage) -> str:
    year = str(vintage) if vintage else None
    if year and title.endswith(" " + year):
        return title[: -len(year)].rstrip(" ,-")
    return title


DEFAULT_OVERRIDES = (
    Path(__file__).resolve().parents[2] / "data" / "reference" / "title_overrides.csv"
)


def read_records(path, overrides_path=None) -> Iterator[WineRecord]:
    path = Path(path)
    overrides = load_overrides(overrides_path or DEFAULT_OVERRIDES)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path.name}: expected a JSON list of wine objects")
    report_id = path.stem.removesuffix("_normalized")

    for index, raw in enumerate(data, 1):
        if not isinstance(raw, dict):
            LOG.warning("Skipping non-object at index %d", index)
            continue
        title = clean_text(raw.get("title"))
        if not title:
            LOG.warning("Skipping record %d with no title", index)
            continue

        vintage = clean_vintage(raw.get("vintage"))
        title = _strip_vintage(title, vintage)

        country = clean_text(raw.get("country"))
        region = clean_text(raw.get("region"))
        if region and country and fold(region) == fold(country):
            region = None

        note = clean_note(raw.get("tasting_note"))
        parts = split_wine_title(title, region, overrides)
        style, color = split_style_color(raw.get("color"))
        color = derive_color(style, color, title, region)
        non_vintage = is_non_vintage(vintage, title, style, note)
        info = parse_tasting_note(note, title)
        varietal = (
            info.grapes[0].name
            if len(info.grapes) == 1
            else clean_text(raw.get("varietal"))
        )

        source_name = clean_text(raw.get("source_name")) or "Unknown source"
        source_url = clean_text(raw.get("source_url"))
        rank = raw.get("rank")

        ratings, prices = [], []
        score = to_float(raw.get("score"), 0, 100)
        if score is not None:
            ratings.append(
                Rating(
                    score=score,
                    scale=100,
                    vintage=vintage,
                    reviewer=source_name,
                    source=source_name,
                    source_url=source_url,
                )
            )
        price = to_float(raw.get("price"), 0)
        currency = clean_text(raw.get("currency"))
        if price is not None and currency:
            prices.append(
                Price(
                    price=price,
                    currency=currency,
                    vintage=vintage,
                    source_url=source_url,
                )
            )

        yield WineRecord(
            name=parts.name,
            full_title=clean_text(raw.get("title")),
            winery=clean_text(raw.get("winery")) or parts.winery,
            country=country,
            region=region,
            style=style,
            color=color,
            varietal=varietal,
            vintages=[vintage] if vintage else ([None] if non_vintage else []),
            grapes=info.grapes,
            aromas=info.aromas,
            sensory=info.sensory,
            ratings=ratings,
            prices=prices,
            sources=[
                Source(
                    name=source_name,
                    external_id=f"{report_id}#{rank}" if rank is not None else None,
                    url=source_url,
                    rank=rank,
                    tasting_note=note,
                    tasting_note_truncated=bool(
                        raw.get("tasting_note_truncated", False)
                    ),
                    vintage=vintage,
                )
            ],
        )
