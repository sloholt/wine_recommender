"""Normalize scraper output without guessing missing wine attributes.

Run: python -m ingestion.normalize --input data/raw/2025_top100_everydayvalue.json
"""

import argparse
import html
import json
import logging
import re
import unicodedata
from pathlib import Path
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger(__name__)
COLORS = {
    "red": "Red",
    "white": "White",
    "rose": "Rosé",
    "rosé": "Rosé",
    "orange": "Orange",
    "sparkling": "Sparkling",
}


def clean(value):
    if value is None:
        return None
    text = unicodedata.normalize("NFC", html.unescape(str(value)))
    return re.sub(r"\s+", " ", text).strip() or None


def clean_note(value):
    text = clean(value)
    if not text:
        return None
    if re.search(r"<\s*/?\s*[a-z][^>]*>", text, re.I):
        text = BeautifulSoup(text, "html.parser").get_text(" ", strip=True)
    return clean(text)


def integer(value, minimum, maximum):
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(str(value).strip())
    except (ValueError, TypeError):
        return None
    return int(number) if number.is_integer() and minimum <= number <= maximum else None


def price_number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(str(value).replace(",", "").strip())
    except (ValueError, TypeError):
        return None
    return round(number, 2) if 0 <= number < 1_000_000 else None


def normalize_record(raw):
    title = clean(raw.get("title"))
    if not title:
        return None
    vintage = integer(raw.get("vintage"), 1800, 2100)
    if vintage is None:
        match = re.search(r"\b(18|19|20)\d{2}$", title)
        vintage = int(match.group()) if match else None
    color = clean(raw.get("color"))
    color = COLORS.get(color.casefold(), color) if color else None
    country = clean(raw.get("country"))
    if country and (country.islower() or country.isupper()):
        country = country.title()
    price = price_number(raw.get("price"))
    currency = clean(raw.get("currency"))
    currency = currency.upper() if currency else None
    if price is None:
        currency = None
    if currency and not re.fullmatch(r"[A-Z]{3}", currency):
        LOG.warning("Unrecognized currency %r for %s", currency, title)
        currency = None
    return {
        "rank": integer(raw.get("rank"), 1, 100000),
        "title": title,
        "winery": clean(raw.get("winery")),
        "varietal": clean(raw.get("varietal")),
        "score": integer(raw.get("score"), 0, 100),
        "color": color,
        "country": country,
        "region": clean(raw.get("region")),
        "vintage": vintage,
        "price": price,
        "currency": currency,
        "tasting_note": clean_note(raw.get("tasting_note")),
        "tasting_note_truncated": bool(raw.get("tasting_note_truncated", False)),
        "source_name": clean(raw.get("source_name")),
        "source_url": clean(raw.get("source_url")),
    }


def normalize_file(input_path, output_path=None):
    input_path = Path(input_path)
    records = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("Expected a JSON list of wine records")
    result, seen = [], set()
    for index, raw in enumerate(records, 1):
        if not isinstance(raw, dict):
            LOG.warning("Skipping non-object at index %d", index)
            continue
        wine = normalize_record(raw)
        if wine is None:
            LOG.warning("Skipping missing title at index %d", index)
            continue
        # One entry per report and rank; do not collapse different report entries.
        key = (
            (wine["source_url"], wine["rank"])
            if wine["source_url"] and wine["rank"] is not None
            else (wine["title"].casefold(), wine["vintage"], wine["source_url"])
        )
        if key in seen:
            LOG.warning(
                "Skipping duplicate entry at index %d: %s", index, wine["title"]
            )
            continue
        seen.add(key)
        result.append(wine)
    output_path = (
        Path(output_path)
        if output_path
        else ROOT / "data" / "processed" / f"{input_path.stem}_normalized.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    LOG.info("Normalized %d/%d wines -> %s", len(result), len(records), output_path)
    for field in ("winery", "varietal", "price", "vintage", "tasting_note"):
        LOG.info("Missing %s: %d", field, sum(r[field] is None for r in result))
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    normalize_file(args.input, args.output)


if __name__ == "__main__":
    main()
