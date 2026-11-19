"""Canonical record format shared by every source adapter.

Each adapter (X-Wines, scraped Top 100 JSON, ...) reads its own file format and
yields ``WineRecord`` objects. Everything downstream (normalize.py, db.py) only
ever sees this one shape, so a new source never requires changes to the core.

Vintage semantics (important):

* ``WineRecord.vintages`` lists the years this wine exists in. A ``None`` entry
  means "non-vintage" (N.V.) and becomes a ``wine_vintages`` row with a NULL year.
* On ``Rating`` / ``Price`` / ``Source``, ``vintage=None`` means "applies to the
  whole wine" (``vintage_id`` stays NULL). A year attaches the row to that vintage.

Adapters may leave fields raw (mojibake, strings for numbers, list-strings); the
normalizer cleans them. They should not guess values the source does not give.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Grape:
    name: str
    percentage: Optional[float] = None  # 0-100, only if the source states it


@dataclass
class Aroma:
    name: str
    category: Optional[str] = None
    intensity: Optional[float] = None


@dataclass
class Sensory:
    """Numeric tasting structure. Use normalize.ordinal() to map labels to numbers."""

    body: Optional[float] = None
    acidity: Optional[float] = None
    structure: Optional[float] = None
    finish: Optional[float] = None
    balance: Optional[float] = None
    tannin: Optional[float] = None
    sweetness: Optional[float] = None

    def is_empty(self) -> bool:
        return all(v is None for v in self.__dict__.values())


@dataclass
class Rating:
    score: float
    scale: float  # 100 for critic points, 5 for X-Wines user ratings
    vintage: Optional[int] = None
    rating_count: Optional[int] = None  # number of ratings behind an average
    reviewer: Optional[str] = None
    source: Optional[str] = None
    source_url: Optional[str] = None


@dataclass
class Price:
    price: float
    currency: str  # ISO 4217 code, e.g. "USD"
    vintage: Optional[int] = None
    retailer: Optional[str] = None
    source_url: Optional[str] = None
    observed_at: Optional[str] = None  # ISO 8601 text


@dataclass
class Source:
    """Where a record came from. ``external_id`` makes re-imports idempotent."""

    name: str
    external_id: Optional[str] = None  # e.g. X-Wines WineID, or "<report>#<rank>"
    url: Optional[str] = None
    rank: Optional[int] = None
    tasting_note: Optional[str] = None
    tasting_note_truncated: bool = False
    vintage: Optional[int] = None


@dataclass
class WineRecord:
    title: str
    winery: Optional[str] = None
    winery_website: Optional[str] = None
    country: Optional[str] = None
    country_iso: Optional[str] = None  # ISO 3166-1 alpha-2, if the source has it
    region: Optional[str] = None
    color: Optional[str] = None
    varietal: Optional[str] = None  # only when a single grape; else use grapes
    abv: Optional[float] = None
    closure: Optional[str] = None

    vintages: List[Optional[int]] = field(default_factory=list)
    grapes: List[Grape] = field(default_factory=list)
    aromas: List[Aroma] = field(default_factory=list)
    pairings: List[str] = field(default_factory=list)
    sensory: Optional[Sensory] = None

    ratings: List[Rating] = field(default_factory=list)
    prices: List[Price] = field(default_factory=list)
    sources: List[Source] = field(default_factory=list)
