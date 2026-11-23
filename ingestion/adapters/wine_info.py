"""Adapter for the Kaggle "wine-information" dataset (dev7halo/wine-information).

A Korean wine-retail catalog, one row per wine + vintage. Two files:

* ``wine_info.csv``      raw: bilingual values ("프랑스 France"), blank = missing
* ``cleansingWine.csv``  English only, with a leading index column, but missing
  price / year were filled with 0

Both work; the raw file is preferred because its blanks are honest. Get it with
``python -m ingestion.adapters.wine_info`` (downloads into data/raw/wine_info/).

Columns:
    id, name, producer, nation, local1..local4 (region, broad -> specific),
    varieties1..varieties12 (grapes), type, use (Appetizer / Table / Dessert),
    abv ("13.5" or a range "13~14"), degree (serving temperature, °C),
    sweet / acidity / body / tannin ("SWEET1".."SWEET5": ranks 1-5),
    price (Korean won, KRW), year, ml (bottle volume)

What it does that is specific to this source:
* drops the Korean half of bilingual values ("깔베 Calvet" -> "Calvet");
* "Calvet, Beaujolais Villages" -> wine "Beaujolais Villages" by winery "Calvet",
  but only when the part before the comma is the producer itself (brand names
  such as "Carlo Rossi, Concord" by Gallo are kept whole);
* region = the most specific local1..4 value ("Bourgogne > Beaujolais" -> Beaujolais);
* skips what is not wine: types Hard (calvados, grappa), Sake/Rice, Korean
  (Korean liquors), Etc (sangria), and anything above MAX_WINE_ABV;
* an ABV range "13~14" is stored as its midpoint (13.5);
* rank scales are mapped to the shared ones (see core/tasting.py):
  body 1-5 and tannin 1-5 as-is; acidity 1-5 -> 1-3; sweetness 1-5 -> 1-4;
* "Blend" / "Etc" in the grape columns are not grapes and are dropped;
* "Appetizer" / "Dessert" in ``use`` become food pairings;
* price is kept only for 750 ml bottles, so prices stay comparable;
* a sparkling wine without a year is N.V. (shared rule, core/naming.py);
* the dataset id is the source's external_id, so re-imports are safe.
"""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from typing import Iterator, List, Optional

from ..core.models import Grape, Price, Sensory, Source, WineRecord
from ..core.naming import derive_color, is_non_vintage
from ..core.normalize import (
    clean_text,
    clean_vintage,
    fold,
    iso_for_country,
    normalize_country,
    split_style_color,
    to_float,
    to_int,
)

LOG = logging.getLogger(__name__)

SOURCE_NAME = "Wine Information (Kaggle)"
KAGGLE_DATASET = "dev7halo/wine-information"
FILE_GLOB = "wine_info.csv"  # for --input <folder>: the raw file only, not both
MAX_WINE_ABV = 25.0
NOT_WINE_TYPES = {"hard", "sake rice", "korean", "etc"}
NOT_GRAPES = {"blend", "etc"}
PRICE_BOTTLE_ML = 750

_HANGUL_PREFIX = re.compile(r"^(?:[ᄀ-ᇿ㄰-㆏가-힣]+[\s\-–]*)+")
_RANK = re.compile(r"(\d)\s*$")


def english(value) -> Optional[str]:
    """'칠레 Chile' -> 'Chile'. Values with no Latin part are kept as they are."""
    text = clean_text(value)
    if not text:
        return None
    stripped = _HANGUL_PREFIX.sub("", text).strip(" -–")
    return stripped or text


def _rank(value) -> Optional[int]:
    """'ACIDITY4' -> 4 (1..5), else None."""
    text = clean_text(value)
    match = _RANK.search(text) if text else None
    rank = int(match.group(1)) if match else None
    return rank if rank and 1 <= rank <= 5 else None


def _rescale(rank: Optional[int], top: float) -> Optional[float]:
    """Map a 1..5 rank linearly onto 1..top."""
    if rank is None:
        return None
    return round(1 + (rank - 1) * (top - 1) / 4, 2)


def _abv(value) -> Optional[float]:
    """'13.5' -> 13.5; '13~14' -> 13.5 (midpoint of a stated range); 0 -> None."""
    text = clean_text(value)
    if not text:
        return None
    parts = [to_float(p, 0, 100) for p in text.split("~")]
    if not parts or any(p is None for p in parts):
        return None
    abv = sum(parts) / len(parts)
    return round(abv, 2) if abv > 0 else None


def _split_name(name: str, producer: Optional[str]) -> str:
    """Drop a leading 'Producer, ' from the wine name when it really is the producer."""
    head, sep, rest = name.partition(",")
    rest = rest.strip()
    if not sep or not rest or not producer:
        return name
    head_key, producer_key = fold(head), fold(producer)
    if head_key and (
        head_key == producer_key
        or re.search(rf"\b{re.escape(head_key)}\b", producer_key)
    ):
        return rest
    return name


def _grapes(row) -> List[Grape]:
    grapes, seen = [], set()
    for i in range(1, 13):
        name = clean_text(row.get(f"varieties{i}"))
        key = fold(name)
        if not name or key in NOT_GRAPES or key in seen:
            continue
        seen.add(key)
        grapes.append(Grape(name))
    return grapes


def _country(value) -> Optional[str]:
    """Only real countries: 'Others' and the two rows with a grape here -> None."""
    country = normalize_country(english(value))
    if country and iso_for_country(country):
        return country
    if country:
        LOG.debug("Unknown country %r, left empty", country)
    return None


def read_records(path) -> Iterator[WineRecord]:
    skipped = 0
    with open(Path(path), newline="", encoding="utf-8-sig") as f:
        for line, row in enumerate(csv.DictReader(f), 2):
            wine_id = clean_text(row.get("id"))
            title = english(row.get("name"))
            if not wine_id or not title:
                LOG.warning("%s line %d: no id / name, skipped", path, line)
                continue

            raw_type = clean_text(row.get("type"))
            abv = _abv(row.get("abv"))
            if fold(raw_type) in NOT_WINE_TYPES or (abv and abv > MAX_WINE_ABV):
                skipped += 1
                continue

            producer = english(row.get("producer"))
            locals_ = [english(row.get(f"local{i}")) for i in range(1, 5)]
            locals_ = [x for x in locals_ if x]
            region = locals_[-1] if locals_ else None
            country = _country(row.get("nation"))
            if region and country and fold(region) == fold(country):
                region = None

            style, color = split_style_color(raw_type)
            color = derive_color(style, color, title, region)
            vintage = clean_vintage(row.get("year"))  # 0 / blank -> None
            non_vintage = is_non_vintage(vintage, title, style, None)
            grapes = _grapes(row)

            uses = {fold(u) for u in (clean_text(row.get("use")) or "").split(",")}
            pairings = [food for food in ("Appetizer", "Dessert") if fold(food) in uses]

            prices = []
            price = to_float(row.get("price"), 0)
            ml = to_int(row.get("ml"), 1)
            if price and (ml is None or ml == PRICE_BOTTLE_ML):
                prices.append(Price(price=price, currency="KRW", vintage=vintage))

            yield WineRecord(
                name=_split_name(title, producer),
                full_title=title,
                winery=producer,
                country=country,
                region=region,
                style=style,
                color=color,
                varietal=grapes[0].name if len(grapes) == 1 else None,
                abv=abv,
                vintages=[vintage] if vintage else ([None] if non_vintage else []),
                grapes=grapes,
                sensory=Sensory(
                    body=_rescale(_rank(row.get("body")), 5),
                    tannin=_rescale(_rank(row.get("tannin")), 5),
                    acidity=_rescale(_rank(row.get("acidity")), 3),
                    sweetness=_rescale(_rank(row.get("sweet")), 4),
                ),
                pairings=pairings,
                prices=prices,
                sources=[Source(name=SOURCE_NAME, external_id=wine_id, vintage=vintage)],
            )
    if skipped:
        LOG.info("Skipped %d non-wine rows (spirits, sake, sangria...)", skipped)


def download(dest=None) -> Path:
    """Download the dataset with kagglehub and copy its CSVs into data/raw/wine_info/."""
    import shutil

    import kagglehub

    dest = Path(dest or Path(__file__).resolve().parents[2] / "data" / "raw" / "wine_info")
    dest.mkdir(parents=True, exist_ok=True)
    for csv_file in Path(kagglehub.dataset_download(KAGGLE_DATASET)).glob("*.csv"):
        shutil.copy2(csv_file, dest / csv_file.name)
    return dest


if __name__ == "__main__":
    print("Dataset files in:", download())
