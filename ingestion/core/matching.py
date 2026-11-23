"""Fuzzy wine-name matching across sources.

Sources name the same wine differently:

    critic list:  "Dog Point Sauvignon Blanc Marlborough Section 94 2021"
    X-Wines:      "Section 94 Sauvignon Blanc"           (winery "Dog Point")
    critic list:  "Elderton Shiraz Barossa Valley Command 2022"
    X-Wines:      "Command Single Vineyard Shiraz"

Each title is reduced to a *profile*:
* ``core``: its words, minus the winery's words, years, filler words (STOP) and
  grape names; place names stay (they are part of a wine's identity);
* ``distinctive``: the core words that are neither places nor generic quality
  terms (GENERIC: reserve, gran, GG, old vines, ...). "Section 94" -> {section, 94};
* ``grapes``: grape names found in the title (canonical form) plus known grapes;
* ``places``: core words that are place names.

Two profiles match when (``name_score``):
1. their cores are equal                                                -> 1.0
   (cores must always agree on quality tiers: TIERS, e.g. reserva / crianza)
2. one core contains the other and the smaller one has a distinctive word
   ("command" in "barossa valley command")              -> 0.5 + 0.4 * overlap
   (so "Barolo" alone never absorbs "Barolo Cerequio")
3. neither has a distinctive word: same grapes and the same places
   ("Pinot Noir" from one source ~ "Pinot Noir" from another)          -> 0.3
   ("Pahlmeyer Chardonnay Napa Valley" ~ "Napa Valley Chardonnay" is rule 1)

The caller adds the hard checks (same winery name, country, color, grapes, not
the same source) and only accepts a unique best candidate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import FrozenSet, Iterable, List, Optional, Set, Tuple

from .normalize import canonical_grape, fold

# Words that carry no identity at all: removed before comparing.
STOP = {
    "the", "a", "an", "of", "and", "y", "e", "et", "und",
    "de", "di", "da", "do", "dos", "das", "del", "della", "dei", "degli",
    "la", "le", "les", "lo", "il", "el", "los", "las", "du", "des", "d", "l",
    "von", "van", "vom", "zu", "der", "die",
    "vineyard", "vineyards", "vinedo", "vinedos", "vigna", "vigneto", "single",
    "ried", "lage", "wine", "wines", "vino", "vin", "vinho",
}
# Words that describe a wine but do not identify it on their own.
GENERIC = {
    "reserve", "reserva", "riserva", "gran", "grand", "grande", "premier", "1er",
    "cru", "grands", "crus", "old", "vines", "vine", "alte", "reben", "vieilles",
    "vignes", "estate", "selection", "seleccion", "special", "especial",
    "classic", "classico", "superiore", "trocken", "dry", "brut", "nature",
    "extra", "gg", "smaragd", "federspiel", "steinfeder", "kabinett",
    "spatlese", "auslese", "crianza", "cuvee", "red", "white", "rose", "rosado",
    "blanc", "blanco", "bianco", "branco", "rouge", "rosso", "tinto", "noir",
    "limited", "edition", "family", "collection", "series", "barrel", "oak",
    "aged", "year", "years", "tawny", "ruby", "vintage", "late", "harvest",
    "port", "porto", "doc", "docg", "aoc", "igt", "valley", "hills", "coast",
    "county", "river", "mountain", "mount", "north", "south", "east", "west",
}
# Quality / ageing tiers: two names must agree on these ("Reserva" is not the
# plain wine, "Gran Reserva" is not "Reserva").
TIERS = {
    "reserve", "reserva", "riserva", "gran", "grand", "grande", "crianza",
    "superiore", "premier", "1er", "cru", "grands", "crus", "vecchie", "vieilles",
    "alte", "old", "selection", "seleccion", "special", "especial", "limited",
    # sparkling: dosage and vintage name different wines ("Brut" vs "Sec",
    # "Brut" vs "Brut Vintage")
    "vintage", "brut", "sec", "demi", "doux", "dosage", "zero", "nature", "dry",
    "extra",
}
_YEAR = re.compile(r"^(?:18|19|20)\d{2}$")
_MAX_PHRASE = 5


def _tokens(text: Optional[str]) -> List[str]:
    return fold(text).split() if text else []


class Vocabulary:
    """Grape names (removed from titles) and place words (kept, but not distinctive)."""

    def __init__(self):
        self.grapes: dict = {}  # token tuple -> canonical grape key
        self.places: Set[str] = set()

    def add_grape(self, name: Optional[str]):
        tokens = tuple(_tokens(name))
        if not tokens or len(tokens) > _MAX_PHRASE:
            return
        if len(tokens) == 1 and len(tokens[0]) < 4:  # 'cot', 'pais': too ambiguous
            return
        self.grapes[tokens] = fold(canonical_grape(name) or name)

    def add_place(self, name: Optional[str]):
        self.places.update(t for t in _tokens(name) if t not in STOP and not t.isdigit())

    def add_grapes(self, names: Iterable[str]):
        for n in names:
            self.add_grape(n)

    def add_places(self, names: Iterable[str]):
        for n in names:
            self.add_place(n)


@dataclass(frozen=True)
class Profile:
    core: FrozenSet[str]
    distinctive: FrozenSet[str]
    places: FrozenSet[str]
    grapes: FrozenSet[str] = field(default_factory=frozenset)  # title + known grapes
    title_grapes: FrozenSet[str] = field(default_factory=frozenset)  # named in title

    def with_grapes(self, names: Iterable[str]) -> "Profile":
        extra = {fold(canonical_grape(n) or n) for n in names if n}
        return Profile(
            self.core, self.distinctive, self.places, self.grapes | extra, self.title_grapes
        )


def profile(title: str, winery_tokens: Iterable[str], vocab: Vocabulary) -> Profile:
    winery = set(winery_tokens)
    tokens = [t for t in _tokens(title) if not _YEAR.match(t)]

    kept: List[str] = []
    grapes: Set[str] = set()
    i = 0
    while i < len(tokens):
        for size in range(min(_MAX_PHRASE, len(tokens) - i), 0, -1):
            grape = vocab.grapes.get(tuple(tokens[i : i + size]))
            if grape:
                grapes.add(grape)
                i += size
                break
        else:
            kept.append(tokens[i])
            i += 1

    core = frozenset(t for t in kept if t not in STOP and t not in winery)
    places = frozenset(t for t in core if t in vocab.places)
    distinctive = frozenset(t for t in core if t not in GENERIC and t not in vocab.places)
    return Profile(core, distinctive, places, frozenset(grapes), frozenset(grapes))


def name_score(a: Profile, b: Profile) -> Optional[float]:
    """How well two wine names agree (see module doc), or None if they do not."""
    if (a.core & TIERS) != (b.core & TIERS):
        return None
    if a.core and a.core == b.core:
        return 1.0
    if a.core and b.core:
        small, large = (a, b) if len(a.core) <= len(b.core) else (b, a)
        if small.core < large.core and small.distinctive:
            return 0.5 + 0.4 * len(small.core) / len(large.core)
        return None
    # Names made only of grapes ("Pinot Noir" ~ "Pinot Noir"): same grapes, and
    # the same places, so a "Barolo" never matches a plain "Nebbiolo".
    if (
        not a.distinctive
        and not b.distinctive
        and a.grapes
        and a.grapes == b.grapes
        and a.places == b.places
    ):
        return 0.3
    return None


def _disjoint(a: FrozenSet[str], b: FrozenSet[str]) -> bool:
    return bool(a and b and not (a & b))


def grapes_conflict(a: Profile, b: Profile) -> bool:
    """True when the two wines clearly have different grapes:
    * both titles name grapes, but not the same ones ("... Merlot" vs "... Cabernet
      Sauvignon", even when the Cabernet's blend lists some Merlot; a "Malbec" is
      not the "Malbec Bonarda Petit Verdot" blend), or
    * all known grapes (titles + grape lists) share none."""
    if a.title_grapes and b.title_grapes and a.title_grapes != b.title_grapes:
        return True
    return _disjoint(a.grapes, b.grapes)


def best_match(
    scored: List[Tuple[float, int]],
) -> Optional[Tuple[float, int]]:
    """The single best (score, id); None if empty or the top score is tied."""
    if not scored:
        return None
    scored = sorted(scored, reverse=True)
    if len(scored) > 1 and abs(scored[0][0] - scored[1][0]) < 1e-9:
        return None
    return scored[0]
