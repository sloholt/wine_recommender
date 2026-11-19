"""Split a published wine title into winery + wine name, and derive style facts.

Critic lists publish one string such as
    "Laurent-Perrier Champagne Grand Siècle Grande Cuvée N.26"
with the winery, grape, region and cuvée all run together. This module turns it into

    winery     = "Laurent-Perrier"
    name       = "Grand Siècle Grande Cuvée N.26"
    full_title = the original string (kept for display)

How: the winery is whatever comes before the first *marker* word, i.e. the first
grape ("Riesling"), place ("Champagne", "Wachau", "Valle de Uco") or appellation
("Brunello di Montalcino"). Place and grape markers are then dropped from the name
(they live in ``region`` / ``wine_grapes``); appellations that *are* the wine
(Barolo, Brunello di Montalcino, Châteauneuf-du-Pape ...) are kept in the name.

No heuristic is perfect, so a hand-maintained CSV of overrides
(``data/reference/title_overrides.csv``) wins over the rules. A title with no marker
and no override is NOT guessed: winery stays NULL and name is the whole title.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .normalize import fold
from .tasting import _GRAPE_LIST


def _k(token: str) -> str:
    """Comparison key for one word: lowercase, accents and punctuation removed."""
    return re.sub(r"[^a-z0-9]", "", fold(token))


def _seq(text: str) -> Tuple[str, ...]:
    return tuple(k for k in (_k(t) for t in text.split()) if k)


# Places and grapes: recognised, then removed from the wine name.
_PLACES = """
Champagne|Wachau|Pfalz|Nahe|Mosel|Rheingau|Rheinhessen|Alsace|Burgenland|Austria|Danube
Rioja|Priorat|Ribera del Duero|Sierra de Gredos|Alava|Álava|Toscana|Tuscany|Maremma|Maremma Toscana
Terre Siciliane|Bolgheri|Bolgheri Superiore|Valdarno di Sopra|Piedmont|Veneto|Savennières|Loire Valley
Bordeaux|Bordeaux Blanc|Pomerol|St.-Emilion|Saint-Emilion|Pauillac|Margaux|Pessac-Léognan|Sauternes|Médoc
Mendoza|Gualtallary|Valle de Uco|Valle de Aconcagua|Valle de Apalta|Valle de Colchagua|Valle de Maipo
Aconcagua|Puente Alto|Patagonia|Costa|Central Otago|Hawke's Bay|Hawkes Bay|Kumeu|Yarra Valley
Yakima Valley|Napa Valley|Carneros|Sonoma Coast|Sonoma County|Fort Ross-Seaview|Santa Rita Hills|Washington
Südsteiermark|Steiermark|Douro|Eger|Shangri-La|Yunnan|Rhone Valley|Rhône|Ermitage
Sierra de Gredos|Vinos Atlánticos|Spain|France|Germany|Italy|Chile|Argentina|Portugal|California
"""
_KEEP = """
Brunello di Montalcino|Barolo|Barbaresco|Amarone della Valpolicella|Châteauneuf-du-Pape|Moulin-à-Vent
Egri Bikavér|Chianti Classico|Valpolicella|Montepulciano d'Abruzzo|Gevrey-Chambertin|Chambolle-Musigny
"""
_GENERIC_ONLY = {
    "gran",
    "reserva",
    "riserva",
    "seleccion",
    "especial",
    "superiore",
    "classico",
    "crianza",
}
_TITLES_BEFORE = {
    "chateau",
    "domaine",
    "clos",
    "mas",
    "bodega",
    "bodegas",
    "tenuta",
    "castello",
    "weingut",
}


def _markers(raw: str) -> List[Tuple[str, ...]]:
    return [s for s in (_seq(x) for x in raw.replace("\n", "|").split("|")) if s]


_PLACE_SEQS = _markers(_PLACES)
_KEEP_SEQS = _markers(_KEEP)
_GRAPE_SEQS = [_seq(g) for g in _GRAPE_LIST]


@dataclass
class NameParts:
    winery: Optional[str]
    name: str
    full_title: str
    source: str  # "override" | "rule" | "unsplit"


def load_overrides(path) -> Dict[str, Tuple[Optional[str], str]]:
    """CSV columns: title, winery, name.  ``title`` is matched ignoring case / accents / vintage year."""
    path = Path(path)
    out: Dict[str, Tuple[Optional[str], str]] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            title = (row.get("title") or "").strip()
            if not title or title.startswith("#"):
                continue
            winery = (row.get("winery") or "").strip() or None
            name = (row.get("name") or "").strip() or title
            out[_override_key(title)] = (winery, name)
    return out


def _override_key(title: str) -> str:
    return " ".join(_seq(title))


def _match_at(keys: List[str], i: int, seqs: Iterable[Tuple[str, ...]]) -> int:
    """Length of the longest marker sequence starting at word ``i`` (0 if none)."""
    best = 0
    for seq in seqs:
        n = len(seq)
        if n > best and tuple(keys[i : i + n]) == seq:
            best = n
    return best


def split_wine_title(
    title: str,
    region: Optional[str] = None,
    overrides: Optional[Dict[str, Tuple[Optional[str], str]]] = None,
) -> NameParts:
    """Split ``title`` (vintage year already removed) into winery and wine name."""
    title = " ".join(title.split())
    if overrides:
        hit = overrides.get(_override_key(title))
        if hit:
            return NameParts(hit[0], hit[1], title, "override")

    words = title.split()
    keys = [_k(w) for w in words]
    region_seq = _seq(region) if region else ()
    places = _PLACE_SEQS + ([region_seq] if region_seq else [])

    # kind per word: None, "grape", "place", "keep"
    kind: List[Optional[str]] = [None] * len(words)
    i = 0
    while i < len(words):
        if not keys[i]:
            i += 1
            continue
        k_len = _match_at(keys, i, _KEEP_SEQS)
        p_len = _match_at(keys, i, places)
        g_len = _match_at(keys, i, _GRAPE_SEQS)
        best = max(k_len, p_len, g_len)
        # "Château Margaux": the word after Château/Domaine is part of the winery.
        if best and i > 0 and keys[i - 1] in _TITLES_BEFORE and i == 1:
            best = 0
        if best == 0:
            i += 1
            continue
        label = (
            "keep"
            if k_len == best
            else "grape" if g_len == best and p_len < best else "place"
        )
        for j in range(i, i + best):
            kind[j] = label
        i += best

    first = next((n for n, kd in enumerate(kind) if kd), None)
    if first is None or first == 0:
        return NameParts(None, title, title, "unsplit")

    winery = " ".join(words[:first])
    rest = [(w, kd) for w, kd in zip(words[first:], kind[first:])]
    removed = [w for w, kd in rest if kd == "place"]
    kept = [w for w, kd in rest if kd in (None, "keep")]
    name = " ".join(kept).strip()

    if not name:
        name = winery  # estate wine: the wine is known by its producer's name
    elif all(_k(w) in _GENERIC_ONLY for w in kept) and removed:
        name = f"{removed[0]} {name}"  # "Rioja Gran Reserva", not a bare "Gran Reserva"
    return NameParts(winery, name, title, "rule")


# ---------------------------------------------------------------- style / color
_SPARKLING_PLACES = (
    "champagne",
    "prosecco",
    "cava",
    "cremant",
    "franciacorta",
    "sekt",
    "spumante",
)
_WHITE_SWEET = (
    "sauternes",
    "barsac",
    "tokaji",
    "tokay",
    "monbazillac",
    "beerenauslese",
    "trockenbeerenauslese",
    "eiswein",
)


def derive_color(
    style: Optional[str], color: Optional[str], title: str, region: Optional[str]
) -> Optional[str]:
    """Fill a missing color from the wine itself; never overrides a stated color.

    Rules (stated, not guessed): "rosé" in the title -> Rosé; Champagne / Prosecco /
    Cava / Crémant (sparkling) -> White unless the title says rosé; Sauternes-type
    sweet wines -> White.
    """
    if color:
        return color
    text = fold(f"{title} {region or ''}")
    if re.search(r"\brose\b", text):
        return "Rosé"
    if style == "Sparkling" and any(
        re.search(rf"\b{p}\b", text) for p in _SPARKLING_PLACES
    ):
        return "White"
    if style == "Dessert" and any(re.search(rf"\b{p}\b", text) for p in _WHITE_SWEET):
        return "White"
    return None


# ------------------------------------------------------------ non-vintage cue
_NV_CUE = re.compile(
    r"(\bn\.?v\.?\b|non[- ]vintage|multi[- ]vintage|\bm\.?v\.?\b|\bn\.\s?\d+\b|\b\d+\s?(?:eme|th|st|nd|rd|e)?\s+edition\b)",
    re.IGNORECASE,
)
_YEARS = re.compile(r"\b(?:19|20)\d{2}\b")


def blend_years(note: Optional[str]) -> List[int]:
    """Years in a phrase like 'blend of 2012, 2008 and 2007' (needs 2+ years)."""
    if not note:
        return []
    m = re.search(
        r"(?:blend|assemblage|cuv[ée]e)\s+of\s+((?:(?:19|20)\d{2}(?:\s*(?:,|and|&)\s*)?){2,})",
        note,
        re.IGNORECASE,
    )
    return [int(y) for y in _YEARS.findall(m.group(1))] if m else []


def is_non_vintage(
    vintage: Optional[int], title: str, style: Optional[str], note: Optional[str]
) -> bool:
    """True when a record without a vintage year is really a multi-vintage blend.

    Evidence, any of: explicit cue in the title (NV, Multi-vintage, 'N.26', '171ème
    Edition'); a tasting note saying it blends several years; or a sparkling wine
    published without a year (a vintage Champagne always states its year).
    """
    if vintage:
        return False
    if _NV_CUE.search(fold_keep(title)) or len(blend_years(note)) >= 2:
        return True
    return style == "Sparkling"


def fold_keep(text: str) -> str:
    """Lowercase + strip accents, but keep dots/digits so 'N.26' stays recognisable."""
    import unicodedata

    return "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    ).lower()
