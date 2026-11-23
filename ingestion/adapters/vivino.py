"""Adapter for the Kaggle "Vivino wine ratings 2026" dataset (mrbridge/vivino-wine-ratings-2026).

File: vivino_wines_2026.csv. Get it with ``python -m ingestion.adapters.vivino``
(downloads with kagglehub into data/raw/vivino/).

Columns:
    wine_id, vintage_id, full_name, wine_name, winery_name, vintage_year,
    rating (Vivino community average, 1-5), ratings_count, wine_type, region,
    country, is_natural, vivino_url, scraped_at

Two kinds of row share a ``wine_id``:
* wine rows (no vintage_id): the wine's overall average over all its vintages;
* vintage rows (vintage_id + vintage_year): one year's own average.

What it does that is specific to this source:
* ``wine_name`` is already the wine without the winery; ``full_name`` (winery +
  name + year) is kept as the display title;
* ``wine_type`` mixes color and style ("Red", "Sparkling", "Dessert",
  "Fortified"); it is split, and a missing color is derived by the shared rules
  (Champagne -> White, "Rosé" in the name -> Rosé);
* rating -> a 5-point Rating by "Vivino users" with its ratings_count: for the
  whole wine on wine rows, for that vintage on vintage rows;
* grapes are taken from the name when it names them ("... Malbec"); otherwise
  left empty (the grape-color step and other sources fill in what they can);
* a wine row is NOT a non-vintage wine (it averages all vintages), so it adds no
  vintage, unless its name says N.V.;
* external_id = wine_id for wine rows, "wine_id:vintage_id" for vintage rows, so
  re-imports are safe; the Vivino URL is kept on the source.

  Load into DB with: python -m ingestion.run --source vivino --input data/raw/vivino
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Iterator

from ..core.models import Rating, Source, WineRecord
from ..core.naming import derive_color, is_non_vintage
from ..core.normalize import (
    clean_text,
    clean_vintage,
    fold,
    split_style_color,
    to_float,
    to_int,
)
from ..core.tasting import parse_tasting_note

LOG = logging.getLogger(__name__)

SOURCE_NAME = "Vivino"
REVIEWER = "Vivino users"
KAGGLE_DATASET = "mrbridge/vivino-wine-ratings-2026"
FILE_GLOB = "vivino_wines*.csv"
RATING_SCALE = 5.0


def read_records(path) -> Iterator[WineRecord]:
    with open(Path(path), newline="", encoding="utf-8-sig") as f:
        for line, row in enumerate(csv.DictReader(f), 2):
            wine_id = clean_text(row.get("wine_id"))
            winery = clean_text(row.get("winery_name"))
            name = clean_text(row.get("wine_name"))
            if not wine_id or not name:
                LOG.warning("%s line %d: no wine_id / wine_name, skipped", path, line)
                continue
            title = clean_text(row.get("full_name")) or " ".join(
                p for p in (winery, name) if p
            )

            vintage_id = clean_text(row.get("vintage_id"))
            vintage = clean_vintage(row.get("vintage_year")) if vintage_id else None
            region = clean_text(row.get("region"))
            country = clean_text(row.get("country"))
            if region and country and fold(region) == fold(country):
                region = None

            style, color = split_style_color(row.get("wine_type"))
            color = derive_color(style, color, name, region)
            non_vintage = vintage is None and is_non_vintage(None, title, None, None)
            grapes = parse_tasting_note(None, name).grapes

            ratings = []
            score = to_float(row.get("rating"), 0, RATING_SCALE)
            if score:  # 0 = no rating
                ratings.append(
                    Rating(
                        score=score,
                        scale=RATING_SCALE,
                        vintage=vintage,
                        rating_count=to_int(row.get("ratings_count"), 0),
                        reviewer=REVIEWER,
                        source=SOURCE_NAME,
                    )
                )

            yield WineRecord(
                name=name,
                full_title=title,
                winery=winery,
                country=country,
                region=region,
                style=style,
                color=color,
                varietal=grapes[0].name if len(grapes) == 1 else None,
                vintages=[vintage] if vintage else ([None] if non_vintage else []),
                grapes=grapes,
                ratings=ratings,
                sources=[
                    Source(
                        name=SOURCE_NAME,
                        external_id=(
                            f"{wine_id}:{vintage_id}" if vintage_id else wine_id
                        ),
                        url=clean_text(row.get("vivino_url")),
                        vintage=vintage,
                    )
                ],
            )


def download(dest=None) -> Path:
    """Download the dataset with kagglehub and copy its CSV into data/raw/vivino/."""
    import shutil

    import kagglehub

    dest = Path(dest or Path(__file__).resolve().parents[2] / "data" / "raw" / "vivino")
    dest.mkdir(parents=True, exist_ok=True)
    for csv_file in Path(kagglehub.dataset_download(KAGGLE_DATASET)).glob("*.csv"):
        shutil.copy2(csv_file, dest / csv_file.name)
    return dest


if __name__ == "__main__":
    print("Dataset files in:", download())
