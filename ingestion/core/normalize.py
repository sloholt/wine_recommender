"""Clean and standardize WineRecord objects before they reach the database.

Rules:
* Never guess a missing attribute. Unknown / invalid values become None.
* Fix text problems (mojibake, HTML entities, stray whitespace) the same way
  for every source.
* Provide small helpers adapters can reuse (parse_list, ordinal, to_float...).

Run ``normalize_wine(record)`` -> cleaned copy, or None if the record has no name.
"""

from __future__ import annotations

import ast
import html
import json
import logging
import math
import re
import unicodedata
from dataclasses import replace
from typing import Any, Iterable, List, Optional

from .models import (
    Aroma,
    Grape,
    Price,
    Rating,
    Sensory,
    Source,
    UserRating,
    WineRecord,
)

LOG = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Text helpers
# --------------------------------------------------------------------------

_NULL_STRINGS = {"nan", "none", "null", "n/a"}
_MOJIBAKE_HINTS = ("Ã", "Â", "â€")
_TAG_RE = re.compile(r"<[^>]+>")
_ZERO_WIDTH_RE = re.compile("[​‌‍⁠﻿]")


def fix_mojibake(text: str) -> str:
    """Repair UTF-8 text that was decoded as cp1252/latin-1 (e.g. 'RosÃ©' -> 'Rosé').

    Only changes the text if the round trip succeeds, so correct text is left alone.
    """
    if not any(hint in text for hint in _MOJIBAKE_HINTS):
        return text
    for encoding in ("cp1252", "latin-1"):
        try:
            return text.encode(encoding).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    return text


def clean_text(value: Any) -> Optional[str]:
    """Unescape HTML, fix mojibake, NFC-normalize, collapse whitespace. '' -> None."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = html.unescape(str(value))
    text = fix_mojibake(text)
    text = unicodedata.normalize("NFC", text)
    text = _ZERO_WIDTH_RE.sub("", text)  # 'Ugni​ blanc' -> 'Ugni blanc'
    text = re.sub(r"\s+", " ", text).strip()
    if not text or text.casefold() in _NULL_STRINGS:
        return None
    return text


def clean_note(value: Any) -> Optional[str]:
    """Like clean_text, but also strips HTML tags from tasting notes."""
    text = clean_text(value)
    if not text:
        return None
    if _TAG_RE.search(text):
        text = _TAG_RE.sub(" ", text)
    return clean_text(text)


def fold(text: Optional[str]) -> str:
    """Deduplication key: lowercase, accents/punctuation removed, spaces collapsed.

    'Château  Léoville-Barton' and 'chateau leoville barton' give the same key.
    Non-Latin letters are kept.
    """
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    stripped = stripped.casefold().replace("&", " and ")
    return re.sub(r"[\W_]+", " ", stripped).strip()


# --------------------------------------------------------------------------
# Numbers and lists
# --------------------------------------------------------------------------


def to_float(
    value: Any, minimum: Optional[float] = None, maximum: Optional[float] = None
) -> Optional[float]:
    """Parse a number from int/float/str. Out-of-range or invalid -> None.

    Handles '1,299.00' (thousands comma), '13,5' (decimal comma) and '13.5%'.
    """
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip().removesuffix("%").strip()
    if re.fullmatch(r"-?\d+,\d{1,2}", text):
        text = text.replace(",", ".")
    else:
        text = text.replace(",", "")
    try:
        number = float(text)
    except ValueError:
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    if minimum is not None and number < minimum:
        return None
    if maximum is not None and number > maximum:
        return None
    return number


def to_int(
    value: Any, minimum: Optional[int] = None, maximum: Optional[int] = None
) -> Optional[int]:
    number = to_float(value, minimum, maximum)
    if number is None or not number.is_integer():
        return None
    return int(number)


def parse_list(value: Any) -> List[str]:
    """Turn a list, JSON string, Python-list string or delimited string into clean items.

    "['Pork', 'Rich Fish']" -> ['Pork', 'Rich Fish']; 'a; b, c' -> ['a', 'b', 'c'].
    Duplicates (by fold key) are removed, order is kept.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, (list, tuple, set)):
        items: Iterable[Any] = value
    else:
        text = clean_text(value)
        if not text:
            return []
        items = None
        if text.startswith("["):
            for loader in (json.loads, ast.literal_eval):
                try:
                    parsed = loader(text)
                except (ValueError, SyntaxError):
                    continue
                if isinstance(parsed, (list, tuple)):
                    items = parsed
                    break
            if items is None:
                text = text.strip("[]")
        if items is None:
            items = re.split(r"[;|,]", text)
    seen, result = set(), []
    for item in items:
        cleaned = clean_text(item)
        if not cleaned:
            continue
        key = fold(cleaned) or cleaned.casefold()
        if key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result


# --------------------------------------------------------------------------
# Ordinal scales (labels -> numbers, so wines can be compared numerically)
# --------------------------------------------------------------------------

BODY_SCALE = {
    "very light bodied": 1.0,
    "light bodied": 2.0,
    "medium bodied": 3.0,
    "full bodied": 4.0,
    "very full bodied": 5.0,
}
ACIDITY_SCALE = {"low": 1.0, "medium": 2.0, "high": 3.0}


def ordinal(value: Any, scale: dict) -> Optional[float]:
    """Map a label ('Full-bodied') to its number via ``scale``; numbers pass through."""
    number = to_float(value)
    if number is not None:
        return number
    text = clean_text(value)
    return scale.get(fold(text)) if text else None


# --------------------------------------------------------------------------
# Colors / style, countries, currency
# --------------------------------------------------------------------------

COLORS = {
    "red": "Red",
    "white": "White",
    "rose": "Rosé",
    "rosado": "Rosé",
    "rosato": "Rosé",
    "orange": "Orange",
}
# Style is separate from color: a sparkling wine is still white / rosé / red.
STYLES = {
    "sparkling": "Sparkling",
    "dessert": "Dessert",
    "dessert port": "Dessert",
    "port": "Fortified",
    "fortified": "Fortified",
}


def normalize_color(value: Any) -> Optional[str]:
    """Color only (Red/White/Rosé/Orange). Style words such as 'Sparkling' return None."""
    text = clean_text(value)
    if not text:
        return None
    key = fold(text)
    if key in STYLES:
        return None
    return COLORS.get(key, text)


def normalize_style(value: Any) -> Optional[str]:
    text = clean_text(value)
    if not text:
        return None
    return STYLES.get(fold(text))


# Words on a label / in a title that mark a sparkling wine even when the grape
# does not ("Jean Josselin NV Cordon Royal Brut Pinot Noir (Champagne)").
# Match against fold()ed text. Shared by the importers and the label extractor.
SPARKLING_CUES = re.compile(
    r"\b(champagne|brut|sparkling|cremant|cava|spumante|franciacorta|sekt"
    r"|mousseux|methode traditionnelle|methode champenoise|metodo classico"
    r"|extra dry|pet nat|petillant naturel|frizzante|prosecco|perlwein|espumante)\b"
)


def is_sparkling_text(text: Optional[str]) -> bool:
    """True if the text names a sparkling wine ('Brut', 'Crémant', 'Prosecco', ...)."""
    return bool(SPARKLING_CUES.search(fold(text)))


def split_style_color(value: Any):
    """'Sparkling' -> ('Sparkling', None); 'Red' -> (None, 'Red'); 'Dessert/Port' -> ('Dessert', None)."""
    return normalize_style(value), normalize_color(value)


COUNTRY_ALIASES = {
    "usa": "United States",
    "us": "United States",
    "u s a": "United States",
    "united states of america": "United States",
    "uk": "United Kingdom",
    "u k": "United Kingdom",
    "great britain": "United Kingdom",
    "england": "United Kingdom",
    "scotland": "United Kingdom",
    "wales": "United Kingdom",
    "the republic of south africa": "South Africa",
    "republic of south africa": "South Africa",
    "korea": "South Korea",
    "republic of korea": "South Korea",
    "czechia": "Czech Republic",
    "north macedonia": "Macedonia",
    "republic of moldova": "Moldova",
}

COUNTRY_ISO = {
    "Albania": "AL",
    "Algeria": "DZ",
    "Argentina": "AR",
    "Armenia": "AM",
    "Australia": "AU",
    "Austria": "AT",
    "Azerbaijan": "AZ",
    "Belgium": "BE",
    "Bolivia": "BO",
    "Bosnia and Herzegovina": "BA",
    "Brazil": "BR",
    "Bulgaria": "BG",
    "Canada": "CA",
    "Chile": "CL",
    "China": "CN",
    "Colombia": "CO",
    "Costa Rica": "CR",
    "Croatia": "HR",
    "Cyprus": "CY",
    "Czech Republic": "CZ",
    "Denmark": "DK",
    "Ecuador": "EC",
    "Egypt": "EG",
    "France": "FR",
    "Georgia": "GE",
    "Germany": "DE",
    "Greece": "GR",
    "Hungary": "HU",
    "India": "IN",
    "Indonesia": "ID",
    "Israel": "IL",
    "Italy": "IT",
    "Japan": "JP",
    "Lebanon": "LB",
    "Luxembourg": "LU",
    "Macedonia": "MK",
    "Malta": "MT",
    "Mexico": "MX",
    "Moldova": "MD",
    "Montenegro": "ME",
    "Morocco": "MA",
    "Netherlands": "NL",
    "New Zealand": "NZ",
    "Paraguay": "PY",
    "Peru": "PE",
    "Portugal": "PT",
    "Romania": "RO",
    "Russia": "RU",
    "Serbia": "RS",
    "Slovakia": "SK",
    "Slovenia": "SI",
    "South Africa": "ZA",
    "South Korea": "KR",
    "Spain": "ES",
    "Sweden": "SE",
    "Switzerland": "CH",
    "Thailand": "TH",
    "Tunisia": "TN",
    "Turkey": "TR",
    "Ukraine": "UA",
    "United Kingdom": "UK",  # project convention (X-Wines also uses UK), not ISO's GB
    "United States": "US",
    "Uruguay": "UY",
    "Venezuela": "VE",
    "Vietnam": "VN",
}
_ISO_BY_FOLD = {fold(name): code for name, code in COUNTRY_ISO.items()}


def normalize_country(value: Any) -> Optional[str]:
    name = clean_text(value)
    if not name:
        return None
    alias = COUNTRY_ALIASES.get(fold(name))
    if alias:
        return alias
    if name.islower() or name.isupper():
        name = name.title()
    return name


def iso_for_country(name: Optional[str]) -> Optional[str]:
    return _ISO_BY_FOLD.get(fold(name)) if name else None


def normalize_iso(value: Any) -> Optional[str]:
    code = clean_text(value)
    if code and re.fullmatch(r"[A-Za-z]{2}", code):
        return code.upper()
    return None


def normalize_currency(value: Any) -> Optional[str]:
    code = clean_text(value)
    if code and re.fullmatch(r"[A-Za-z]{3}", code):
        return code.upper()
    return None


# --------------------------------------------------------------------------
# Vintage helpers
# --------------------------------------------------------------------------

_NON_VINTAGE = {"nv", "n v", "non vintage", "nonvintage"}
_YEAR_END = re.compile(r"^(?P<t>.*\S)\s+(?P<y>(?:19|20)\d{2})$")
_YEAR_START = re.compile(r"^(?P<y>(?:19|20)\d{2})\s+(?P<t>\S.*)$")


def clean_vintage(value: Any) -> Optional[int]:
    return to_int(value, 1800, 2100)


def split_title_vintage(title: str):
    """'2019 Château X' / 'Château X 2019' -> ('Château X', 2019); else (title, None)."""
    for pattern in (_YEAR_END, _YEAR_START):
        match = pattern.match(title)
        if match:
            return match["t"], int(match["y"])
    return title, None


def _clean_vintages(values: Iterable[Any]) -> List[Optional[int]]:
    result: List[Optional[int]] = []
    for value in values or []:
        if value is None or fold(str(value)) in _NON_VINTAGE:
            year = None
        else:
            year = clean_vintage(value)
            if year is None:
                continue  # invalid entry, not N.V.
        if year not in result:
            result.append(year)
    return result


# --------------------------------------------------------------------------
# Sub-record cleaners
# --------------------------------------------------------------------------


def _clean_rating(r: Rating, default_vintage: Optional[int]) -> Optional[Rating]:
    score = to_float(r.score, 0)
    scale = to_float(r.scale, 0)
    if score is None or scale is None or scale <= 0 or score > scale:
        return None
    return Rating(
        score=score,
        scale=scale,
        vintage=clean_vintage(r.vintage) if r.vintage is not None else default_vintage,
        rating_count=to_int(r.rating_count, 0),
        reviewer=clean_text(r.reviewer),
        source=clean_text(r.source),
        source_url=clean_text(r.source_url),
    )


def _clean_price(p: Price, default_vintage: Optional[int]) -> Optional[Price]:
    amount = to_float(p.price, 0, 1_000_000)
    currency = normalize_currency(p.currency)
    if amount is None or currency is None:
        return None
    return Price(
        price=round(amount, 2),
        currency=currency,
        vintage=clean_vintage(p.vintage) if p.vintage is not None else default_vintage,
        retailer=clean_text(p.retailer),
        source_url=clean_text(p.source_url),
        observed_at=clean_text(p.observed_at),
    )


def _clean_source(s: Source, default_vintage: Optional[int]) -> Optional[Source]:
    name = clean_text(s.name)
    if not name:
        return None
    return Source(
        name=name,
        external_id=clean_text(s.external_id),
        url=clean_text(s.url),
        rank=to_int(s.rank, 1, 1_000_000),
        tasting_note=clean_note(s.tasting_note),
        tasting_note_truncated=bool(s.tasting_note_truncated),
        vintage=clean_vintage(s.vintage) if s.vintage is not None else default_vintage,
    )


def _clean_sensory(s: Optional[Sensory]) -> Optional[Sensory]:
    if s is None:
        return None
    cleaned = Sensory(**{k: to_float(v, 0, 100) for k, v in s.__dict__.items()})
    return None if cleaned.is_empty() else cleaned


# Different names for the same grape -> one standard name, so every source
# shares one row in ``grapes``. Only true synonyms, not related varieties.
GRAPE_SYNONYMS = {
    "shiraz": "Syrah",
    "syrah shiraz": "Syrah",
    "syrah and shiraz": "Syrah",
    "grenache noir": "Grenache",
    "cot": "Malbec",
    "white riesling": "Riesling",
    "petite verdot": "Petit Verdot",
    "rousaanne": "Roussanne",  # misspelling in the Kaggle wine-information data
    "moscato": "Muscat",
    "muscatel": "Muscat",
    "muscat moscato": "Muscat",
    "muscat muscatel": "Muscat",
    "sylvaner": "Silvaner",
    "silvaner sylvaner": "Silvaner",
    "garnacha": "Grenache",
    "carinena": "Carignan",
    "mazuelo": "Carignan",
    "carignan carinena": "Carignan",
    "prosecco": "Glera",
    "glera prosecco": "Glera",
    "pinot grigio": "Pinot Gris",
    "pinot bianco": "Pinot Blanc",
    "weissburgunder": "Pinot Blanc",
    "grauburgunder": "Pinot Gris",
    "spatburgunder": "Pinot Noir",
    "mourvedre": "Mourvèdre",
    "monastrell": "Mourvèdre",
    "rolle rollo": "Rolle",
    "friulano sauvignonasse": "Friulano",
    "pais mission": "País",
}


def canonical_grape(name: Optional[str]) -> Optional[str]:
    """Standard grape name. 'Syrah/Shiraz' -> 'Syrah'; 'Muscat/Moscato Bianco' -> 'Moscato Bianco'.

    For other 'A/B' names, the specific name after a generic 'Muscat/' prefix is
    kept (it identifies the sub-variety); otherwise the first name is used.
    """
    name = clean_text(name)
    if not name:
        return None
    synonym = GRAPE_SYNONYMS.get(fold(name))
    if synonym:
        return synonym
    if "/" in name:
        first, _, rest = (part.strip() for part in name.partition("/"))
        if fold(first) == "muscat" and rest:
            return rest
        return GRAPE_SYNONYMS.get(fold(first)) or first or rest or None
    return name


def _clean_grapes(grapes: Iterable[Grape]) -> List[Grape]:
    seen, result = set(), []
    for g in grapes or []:
        name = canonical_grape(g.name)
        key = fold(name) or (name or "").casefold()
        if not name or key in seen:
            continue
        seen.add(key)
        result.append(Grape(name, to_float(g.percentage, 0, 100)))
    return result


def _clean_aromas(aromas: Iterable[Aroma]) -> List[Aroma]:
    seen, result = set(), []
    for a in aromas or []:
        name = clean_text(a.name)
        key = fold(name) or (name or "").casefold()
        if not name or key in seen:
            continue
        seen.add(key)
        result.append(
            Aroma(name, clean_text(a.category), to_float(a.intensity, 0, 100))
        )
    return result


def _clean_pairings(foods: Iterable[Any]) -> List[str]:
    from .food import normalize_food  # local import: food.py imports this module

    seen, result = set(), []
    for item in foods or []:
        name = normalize_food(item)
        key = fold(name) or (name or "").casefold()
        if name and key not in seen:
            seen.add(key)
            result.append(name)
    return result


def normalize_user_rating(r: UserRating) -> Optional[UserRating]:
    """Cleaned copy of a user rating, or None if it is unusable."""
    source = clean_text(r.source)
    user_id = clean_text(r.user_id)
    wine_id = clean_text(r.wine_external_id)
    scale = to_float(r.scale, 0)
    score = to_float(r.score, 0)
    if not (source and user_id and wine_id) or not scale or score is None:
        return None
    if score > scale:
        return None
    non_vintage = bool(r.non_vintage)
    vintage = None if non_vintage else clean_vintage(r.vintage)
    return UserRating(
        source=source,
        user_id=user_id,
        wine_external_id=wine_id,
        score=score,
        scale=scale,
        vintage=vintage,
        non_vintage=non_vintage,
        rated_at=clean_text(r.rated_at),
        external_id=clean_text(r.external_id),
    )


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------


def normalize_wine(rec: WineRecord) -> Optional[WineRecord]:
    """Return a cleaned copy of ``rec``, or None if it has no usable name.

    If the record lists no vintages but the name starts or ends with a year
    ('2019 Château X'), the year is stripped from the title and used as the
    vintage, including for any rating / price / source that has no vintage of
    its own. That way every year of a wine unifies into one ``wines`` row.
    """
    name = clean_text(rec.name)
    if not name:
        return None

    vintages = _clean_vintages(rec.vintages)
    inferred: Optional[int] = None
    if not vintages:
        stripped, year = split_title_vintage(name)
        if year is not None:
            name, inferred, vintages = stripped, year, [year]

    country = normalize_country(rec.country)
    iso = normalize_iso(rec.country_iso) or iso_for_country(country)

    ratings = [x for x in (_clean_rating(r, inferred) for r in rec.ratings) if x]
    prices = [x for x in (_clean_price(p, inferred) for p in rec.prices) if x]
    sources = [x for x in (_clean_source(s, inferred) for s in rec.sources) if x]

    return replace(
        rec,
        name=name,
        full_title=clean_text(rec.full_title),
        winery=clean_text(rec.winery),
        winery_website=clean_text(rec.winery_website),
        country=country,
        country_iso=iso,
        region=clean_text(rec.region),
        style=normalize_style(rec.style) or normalize_style(rec.color),
        color=normalize_color(rec.color),
        varietal=canonical_grape(rec.varietal),
        blend_type=clean_text(rec.blend_type),
        abv=to_float(rec.abv, 0, 100),
        closure=clean_text(rec.closure),
        vintages=vintages,
        grapes=_clean_grapes(rec.grapes),
        aromas=_clean_aromas(rec.aromas),
        sensory=_clean_sensory(rec.sensory),
        pairings=_clean_pairings(rec.pairings),
        ratings=ratings,
        prices=prices,
        sources=sources,
    )
