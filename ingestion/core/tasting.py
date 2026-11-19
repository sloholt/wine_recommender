"""Extract structured tasting data from free-text critic notes.

``parse_tasting_note(note, title)`` returns aromas, a Sensory profile and grape
blend found in the text. It is rule-based and conservative: if the note does not
say something clearly, the value stays None (never guessed).

Scales used for the numeric sensory values (same as the X-Wines mapping):
    body       1 very light, 2 light, 3 medium, 4 full, 5 very full
               ("medium to full" -> 3.5)
    acidity    1 low, 2 medium / balanced, 3 high
    tannin     1..5, starts at 3 when tannins are mentioned, +1/+2 for strong
               cues (powerful, muscular, firm...), -1/-2 for soft cues
    finish     1..5, 4 = long, 5 = very long / endless / lasts minutes
    sweetness  1 dry, 2 off-dry, 3 medium-sweet, 4 sweet

These are estimates from a critic's wording, not lab measurements.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .models import Aroma, Grape, Sensory


@dataclass
class TastingInfo:
    aromas: List[Aroma] = field(default_factory=list)
    sensory: Optional[Sensory] = None
    grapes: List[Grape] = field(default_factory=list)


def _prep(text: Optional[str]) -> str:
    """Lowercase, accent-free, straight quotes, single spaces."""
    if not text:
        return ""
    t = unicodedata.normalize("NFKD", text)
    t = "".join(c for c in t if not unicodedata.combining(c))
    for src, dst in (
        ("’", "'"),
        ("‘", "'"),
        ("“", '"'),
        ("”", '"'),
        ("–", "-"),
        ("—", "-"),
    ):
        t = t.replace(src, dst)
    return re.sub(r"\s+", " ", t.lower()).strip()


def _sentences(t: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?;])\s+", t) if s.strip()]


def _negated(t: str, start: int, window: int = 30) -> bool:
    before = t[max(0, start - window) : start]
    return bool(re.search(r"(?:\bnot\b|n't|\bnever\b|\bno\b)[^.;]*$", before))


# ---------------------------------------------------------------------- body
_LEVEL = r"(?:very\s+light|light|medium|very\s+full|full)"
_BODY_RE = re.compile(
    rf"(?<![a-z])(?P<a>{_LEVEL})(?:-|\s)*(?:(?:to|-)\s*(?P<b>{_LEVEL})(?:-|\s)*)?bod(?:y|ied|ed)(?![a-z])"
)
_BODY_VALUE = {
    "very light": 1.0,
    "light": 2.0,
    "medium": 3.0,
    "full": 4.0,
    "very full": 5.0,
}
_MID_WEIGHT_RE = re.compile(r"(?<![a-z])mid-?weight(?:ed)?(?![a-z])")
_BARE_FULL_RE = re.compile(
    r"(?:^|[.!?]\s+)(?:it's\s+|this is\s+)?full(?=\s+(?:and|yet|with|on)\b)|palate is full(?![a-z-])"
)


def _body(t: str) -> Optional[float]:
    for m in _BODY_RE.finditer(t):
        if _negated(t, m.start()):
            continue
        a = _BODY_VALUE[re.sub(r"\s+", " ", m["a"])]
        b = _BODY_VALUE[re.sub(r"\s+", " ", m["b"])] if m["b"] else a
        return (a + b) / 2
    if _MID_WEIGHT_RE.search(t):
        return 3.0
    if _BARE_FULL_RE.search(t):  # "Full and chewy", "It's full and so layered"
        return 4.0
    return None


# ------------------------------------------------------------------- acidity
_ACID_RE = re.compile(r"(?<![a-z])acid(?:ity)?(?![a-z])")
_ACID_HIGH = {
    "high",
    "great",
    "mind-blowing",
    "whiplash",
    "vibrant",
    "bright",
    "lively",
    "racy",
    "searing",
    "piercing",
    "super-mineral",
    "crisp",
    "electric",
    "brilliant",
    "zippy",
    "sharp",
    "mouthwatering",
}
_ACID_MED = {"medium", "moderate", "balanced", "integrated", "even"}
_ACID_LOW = {"low", "soft", "gentle", "round", "mellow"}
_ACID_CUES = re.compile(
    r"(?<![a-z])(?:racy|nervy|tangy|zesty|mouthwatering|crisp)(?![a-z])"
)


def _acidity(t: str) -> Optional[float]:
    scores = []
    for m in _ACID_RE.finditer(t):
        sent_start = max(
            t.rfind(". ", 0, m.start()),
            t.rfind("! ", 0, m.start()),
            t.rfind("? ", 0, m.start()),
            t.rfind("; ", 0, m.start()),
        )
        before = t[sent_start + 1 if sent_start >= 0 else 0 : m.start()].split()[-4:]
        score = None
        for word in reversed(before):  # nearest adjective wins
            w = word.strip(",.;:()\"'")
            if w in _ACID_HIGH:
                score = 3.0
            elif w in _ACID_MED:
                score = 2.0
            elif w in _ACID_LOW:
                score = 1.0
            if score is not None:
                break
        if score is None:
            after = re.split(r"[.!?;]", t[m.end() :], maxsplit=1)[0].split()[:4]
            if any(w.startswith("balanc") for w in after):
                score = 2.0
        if score is not None:
            scores.append(score)
    if scores:
        return max(scores)
    if _ACID_CUES.search(t):  # no explicit "acidity": strong freshness words only
        return 3.0
    return None


# -------------------------------------------------------------------- tannin
_TANNIN_RE = re.compile(r"(?<![a-z])tannin(?:s)?(?![a-z])|(?<![a-z])tannic(?![a-z])")
_TANNIN_HIGH = re.compile(
    r"(?<![a-z])(?:powerful|muscular|firm|structured|massive|monumental|chewy|tight(?:ly)?|grippy|"
    r"giant|enormous|toned|solid|imposing|broad-shouldered|robust|dense|intense|serious|"
    r"tannin[- ]driven|lots of|plenty of|plentiful|mouth-?coating)(?![a-z])"
)
_TANNIN_LOW = re.compile(
    r"(?<![a-z])(?:soft|supple|gentle|light|weightless|delicate|melt|dissolved|effortless|ethereal)(?![a-z])"
)


def _tannin(t: str) -> Optional[float]:
    high, low, mentioned = set(), set(), False
    for sentence in _sentences(t):
        if not _TANNIN_RE.search(sentence):
            continue
        mentioned = True
        for m in _TANNIN_HIGH.finditer(sentence):
            if not _negated(sentence, m.start(), 15):
                high.add(m.group())
        for m in _TANNIN_LOW.finditer(sentence):
            if not _negated(sentence, m.start(), 15):
                low.add(m.group())
    if not mentioned:
        return None
    return float(max(1, min(5, 3 + min(len(high), 2) - min(len(low), 2))))


# -------------------------------------------------------------------- finish
_FINISH_5 = [
    re.compile(p)
    for p in (
        r"(?<![a-z])(?:endless|never[- ]ending|infinite|eternal)(?![a-z])",
        r"goes on (?:and on|for (?:minutes|ages|ever))",
        r"go on and on",
        r"lasts? (?:for )?(?:almost |about |nearly )?(?:(?:a|one|two|three) )?minutes?",
        r"lasting (?:for )?(?:a|one|two|three) minutes?",
        r"(?:very|extremely|super|incredibly|really|so|remarkably|staggeringly)[- ]long",
        r"long,? long",
    )
]
_FINISH_4 = re.compile(
    r"(?<![a-z])(?<!a )long(?:er)?(?![a-z])(?!\s*(?:time|term|-term|way|years|life))"
    r"|(?<![a-z])(?:lengthy|persistent|lingering|lingers|length)(?![a-z])"
)
_FINISH_2 = re.compile(r"(?:short|brief)\s+(?:finish|length|aftertaste)")


def _finish(t: str) -> Optional[float]:
    if any(p.search(t) for p in _FINISH_5):
        return 5.0
    if _FINISH_4.search(t):
        return 4.0
    if _FINISH_2.search(t):
        return 2.0
    return None


# ----------------------------------------------------------------- sweetness
_SWEET_PATTERNS = [
    (re.compile(r"(?<![a-z])off[- ]dry"), 2.0),
    (re.compile(r"(?<![a-z])medium[- ]sweet"), 3.0),
    (
        re.compile(
            r"(?<![a-z])(?:bone[- ])?dry\s+(?:riesling|white|red|wine|chardonnay|sauvignon|chenin|pinot|"
            r"gewurztraminer|muscat|style)"
        ),
        1.0,
    ),
    (
        re.compile(r"(?:^|[.!?]\s+)sweet(?=\s+(?:and|yet|but)\b)|(?<![a-z])sweet wine"),
        4.0,
    ),
]


def _sweetness(t: str) -> Optional[float]:
    for pattern, value in _SWEET_PATTERNS:
        if pattern.search(t):
            return value
    return None


# -------------------------------------------------------------------- grapes
_GRAPE_LIST = [
    "cabernet sauvignon",
    "cabernet franc",
    "merlot",
    "petit verdot",
    "malbec",
    "carmenere",
    "syrah",
    "shiraz",
    "grenache",
    "garnacha peluda",
    "garnacha",
    "mourvedre",
    "monastrell",
    "tempranillo",
    "sangiovese",
    "nebbiolo",
    "barbera",
    "dolcetto",
    "pinot noir",
    "pinot meunier",
    "pinot gris",
    "pinot blanc",
    "pinot grigio",
    "chardonnay",
    "riesling",
    "sauvignon blanc",
    "semillon",
    "chenin blanc",
    "viognier",
    "gewurztraminer",
    "gruner veltliner",
    "muscat",
    "moscato",
    "furmint",
    "harslevelu",
    "kekfrankos",
    "kadarka",
    "blaufrankisch",
    "zweigelt",
    "saint laurent",
    "touriga nacional",
    "touriga franca",
    "tinta roriz",
    "tinta barroca",
    "tinto cao",
    "tinta amarela",
    "aglianico",
    "nero d'avola",
    "nerello mascalese",
    "primitivo",
    "zinfandel",
    "pinotage",
    "tannat",
    "carignan",
    "cinsault",
    "mencia",
    "graciano",
    "mazuelo",
    "godello",
    "albarino",
    "verdejo",
    "malvasia",
    "trebbiano",
    "garganega",
    "glera",
    "vermentino",
    "fiano",
    "greco",
    "falanghina",
    "assyrtiko",
    "xinomavro",
    "sylvaner",
    "silvaner",
    "muller-thurgau",
    "trousseau",
    "poulsard",
    "savagnin",
    "marsanne",
    "roussanne",
    "picpoul",
    "colombard",
    "ugni blanc",
    "petite sirah",
    "durif",
    "counoise",
    "sagrantino",
    "montepulciano",
    "corvina",
    "rondinella",
    "molinara",
    "refosco",
    "ribolla gialla",
    "friulano",
    "lagrein",
    "teroldego",
    "petit manseng",
    "gros manseng",
    "melon de bourgogne",
    "aligote",
    "gamay",
]
_GRAPE_DISPLAY = {
    "carmenere": "Carmenere",
    "gruner veltliner": "Grüner Veltliner",
    "blaufrankisch": "Blaufränkisch",
    "kekfrankos": "Kékfrankos",
    "semillon": "Semillon",
    "shiraz": "Syrah",
    "albarino": "Albariño",
    "mencia": "Mencía",
    "harslevelu": "Hárslevelű",
    "tinto cao": "Tinto Cão",
    "mourvedre": "Mourvèdre",
    "aligote": "Aligoté",
    "nero d'avola": "Nero d'Avola",
    "muller-thurgau": "Müller-Thurgau",
    "melon de bourgogne": "Melon de Bourgogne",
}


def _display(key: str) -> str:
    return _GRAPE_DISPLAY.get(key) or " ".join(w.capitalize() for w in key.split(" "))


_GRAPE = (
    "(?:"
    + "|".join(re.escape(k) for k in sorted(_GRAPE_LIST, key=len, reverse=True))
    + ")"
)
_SEP = r"(?:\s*,\s*(?:and\s+)?|\s+and\s+|\s*&\s*)"
_PCT_FIRST = re.compile(
    rf"(?<![\d.])(?P<pct>\d{{1,3}}(?:\.\d+)?)\s*%\s*(?:of\s+)?(?P<g>{_GRAPE})(?![a-z])"
)
_PCT_AFTER = re.compile(
    rf"(?<![a-z])(?P<g>{_GRAPE})\s*\(?\s*(?P<pct>\d{{1,3}}(?:\.\d+)?)\s*%\s*\)?"
)
_BLEND_OF = re.compile(
    rf"(?:blend|cuvee|powerhouse|assemblage)\s+of\s+(?P<lst>(?:{_GRAPE}{_SEP}?)+)"
)
_PURE = re.compile(rf"(?<![a-z])pure\s+(?P<g>{_GRAPE})(?![a-z])")
_ONLY_GRAPES = re.compile(rf"(?:an?\s+)?(?:{_GRAPE}{_SEP}?)+\.?")
_GRAPE_FIND = re.compile(rf"(?<![a-z])({_GRAPE})(?![a-z])")


def _grapes(note_t: str, title_t: str) -> List[Grape]:
    blend: Dict[str, float] = {}
    for m in _PCT_FIRST.finditer(note_t):
        blend.setdefault(m["g"], float(m["pct"]))
    for m in _PCT_AFTER.finditer(note_t):
        blend.setdefault(m["g"], float(m["pct"]))
    if blend and sum(blend.values()) <= 101.5:
        ranked = sorted(blend.items(), key=lambda kv: -kv[1])
        return [Grape(_display(k), v) for k, v in ranked]

    names: List[str] = []

    def add(key: str):
        if key not in names:
            names.append(key)

    for m in _BLEND_OF.finditer(note_t):
        for g in _GRAPE_FIND.findall(m["lst"]):
            add(g)
    for m in _PURE.finditer(note_t):
        add(m["g"])
    for sentence in _sentences(note_t):
        if _ONLY_GRAPES.fullmatch(sentence):
            for g in _GRAPE_FIND.findall(sentence):
                add(g)
    for g in _GRAPE_FIND.findall(title_t):  # the wine's own title names the grape
        add(g)
    return [Grape(_display(k)) for k in names]


# -------------------------------------------------------------------- aromas
_R, _K, _C, _S = "Red fruit", "Black fruit", "Citrus", "Stone fruit"
_O, _T, _D, _F = "Orchard fruit", "Tropical fruit", "Dried fruit", "Floral"
_H, _P, _M, _E = "Herbal", "Spice", "Mineral", "Earth & savory"
_W, _B, _N, _Z = "Woody & tobacco", "Toast & smoke", "Nutty", "Sweet & bakery"

# Order matters: more specific patterns first (they win at the same position).
_AROMA_LEXICON = [
    (r"orange blossom", "orange blossom", _F),
    (r"lemon ?grass", "lemongrass", _H),
    (r"stone fruits?", "stone fruit", _S),
    (r"yellow fruits?", "yellow fruit", _S),
    (r"red fruits?", "red fruit", _R),
    (r"red berr(?:y|ies)", "red berries", _R),
    (r"wild berr(?:y|ies)", "wild berries", _R),
    (r"forest (?:berr(?:y|ies)|fruits?)", "forest berries", _R),
    (r"(?:black|blue|dark) fruits?", "dark fruit", _K),
    (r"tropical fruits?", "tropical fruit", _T),
    (r"dried fruits?", "dried fruit", _D),
    (r"sour cherr(?:y|ies)", "sour cherry", _R),
    (r"black cherr(?:y|ies)", "black cherry", _K),
    (r"red ?currants?", "redcurrant", _R),
    (r"black ?currants?", "blackcurrant", _K),
    (r"(?:creme de )?cassis", "cassis", _K),
    (r"currants?", "currant", _K),
    (r"black ?berr(?:y|ies)", "blackberry", _K),
    (r"blue ?berr(?:y|ies)", "blueberry", _K),
    (r"boysenberr(?:y|ies)", "boysenberry", _K),
    (r"bramble ?berr(?:y|ies)", "brambleberry", _K),
    (r"plums?", "plum", _K),
    (r"strawberr(?:y|ies)", "strawberry", _R),
    (r"raspberr(?:y|ies)", "raspberry", _R),
    (r"cranberr(?:y|ies)", "cranberry", _R),
    (r"cherr(?:y|ies)", "cherry", _R),
    (r"pomegranates?", "pomegranate", _R),
    (r"grapefruit(?: zest)?", "grapefruit", _C),
    (r"mandarins?(?: orange)?", "mandarin", _C),
    (r"oranges?(?: zest| peel| curd)?", "orange", _C),
    (r"lemon(?: curd| drop)?s?", "lemon", _C),
    (r"limes?|limey", "lime", _C),
    (r"tangerines?", "tangerine", _C),
    (r"kumquats?", "kumquat", _C),
    (r"bergamot", "bergamot", _C),
    (r"citrus(?: skin| fruit)?", "citrus", _C),
    (r"peach(?:es|iness|y)?", "peach", _S),
    (r"apricots?", "apricot", _S),
    (r"nectarines?", "nectarine", _S),
    (r"mirabelles?", "mirabelle", _S),
    (r"apple pie", "apple pie", _Z),
    (r"apples?", "apple", _O),
    (r"pears?", "pear", _O),
    (r"quinces?", "quince", _O),
    (r"pineapples?", "pineapple", _T),
    (r"mango(?:es|s)?", "mango", _T),
    (r"lychees?", "lychee", _T),
    (r"papayas?", "papaya", _T),
    (r"passion ?fruit", "passion fruit", _T),
    (r"melons?", "melon", _T),
    (r"bananas?", "banana", _T),
    (r"coconut", "coconut", _T),
    (r"persimmons?", "persimmon", _T),
    (r"raisins?|prunes?|figs?", "dried fruit", _D),
    (r"honeysuckle", "honeysuckle", _F),
    (r"roses?(?: petals?| stems?)?", "rose", _F),
    (r"violets?", "violet", _F),
    (r"lavender", "lavender", _F),
    (r"hibiscus", "hibiscus", _F),
    (r"chamomile", "chamomile", _F),
    (r"jasmine|elderflower|acacia|lilac", "white flowers", _F),
    (r"flowers?|floral(?:ity)?|blossoms?", "flowers", _F),
    (r"thyme", "thyme", _H),
    (r"rosemary", "rosemary", _H),
    (r"sage", "sage", _H),
    (r"(?:spear|pepper)?mint", "mint", _H),
    (r"basil", "basil", _H),
    (r"tarragon|dill|fennel", "herbs", _H),
    (r"curry leaf|curry leaves", "curry leaf", _H),
    (r"herbs?|herbal|botanicals?", "herbs", _H),
    (r"eucalyptus", "eucalyptus", _H),
    (r"pine (?:needles?|cones?)", "pine", _H),
    (r"camphor", "camphor", _H),
    (r"nettles?", "nettle", _H),
    (r"scrub", "scrub", _H),
    (r"grass(?:y)?", "grass", _H),
    (r"tea|teas", "tea", _H),
    (
        r"white pepper|black pepper|cracked pepper|peppercorns?|peppery|pepper",
        "pepper",
        _P,
    ),
    (r"allspice", "allspice", _P),
    (r"nutmeg", "nutmeg", _P),
    (r"cinnamon", "cinnamon", _P),
    (r"cloves?", "clove", _P),
    (r"cardamom|cardamon", "cardamom", _P),
    (r"ginger", "ginger", _P),
    (r"licorice|liquorice|anise", "licorice", _P),
    (r"curry", "curry", _P),
    (r"chil(?:i|li) chocolate", "chocolate", _Z),
    (r"chil(?:i|li)", "chili", _P),
    (r"(?:sweet |fine |warm )?spic(?:e|es|y)", "spice", _P),
    (r"limestone", "limestone", _M),
    (r"minerals?|minerality|minerally", "minerals", _M),
    (r"stony|stones?", "stone", _M),
    (r"slate|slatey", "slate", _M),
    (r"granite", "granite", _M),
    (r"flint(?:y)?", "flint", _M),
    (r"graphite", "graphite", _M),
    (r"chalk(?!y)|chalky(?! (?:tannin|texture))", "chalk", _M),
    (r"iron|rust", "iron", _M),
    (r"iodine", "iodine", _M),
    (r"seaweed", "seaweed", _M),
    (r"oyster(?: shells?)?", "oyster shell", _M),
    (r"clam(?: shells?)?", "clam shell", _M),
    (r"mussel(?: shells?)?", "mussel shell", _M),
    (r"salt(?:y|iness|ed)?|saline", "salt", _M),
    (r"tile|pumice|cement|gravel", "stone", _M),
    (r"ink(?: pot)?", "ink", _M),
    (r"petrol|kerosene", "petrol", _M),
    (r"earthy|earthiness|earth|dirt|undergrowth", "earth", _E),
    (r"forest floor", "forest floor", _E),
    (r"mushrooms?", "mushroom", _E),
    (r"truffles?", "truffle", _E),
    (r"leather", "leather", _E),
    (r"meaty|meat", "meat", _E),
    (r"tapenade", "tapenade", _E),
    (r"olives?", "olive", _E),
    (r"balsamic", "balsamic", _E),
    (r"mezcal", "mezcal", _E),
    (r"tar(?:ry)?", "tar", _E),
    (r"cedar", "cedar", _W),
    (r"sandalwood", "sandalwood", _W),
    (r"cigar(?: box)?", "cigar box", _W),
    (r"tobacco", "tobacco", _W),
    (r"(?:lead )?pencil(?: shavings| lead)?", "pencil shavings", _W),
    (r"incense", "incense", _W),
    (r"(?:tree |perfumed )?bark", "bark", _W),
    (r"polished wood|wood", "wood", _W),
    (r"toast(?:y|ed)?", "toast", _B),
    (r"smok(?:e|y|iness)", "smoke", _B),
    (r"vanilla", "vanilla", _B),
    (r"coffee(?: beans?)?|mocha", "coffee", _B),
    (r"charcoal", "charcoal", _B),
    (r"hazelnuts?", "hazelnut", _N),
    (r"almonds?", "almond", _N),
    (r"walnuts?", "walnut", _N),
    (r"nutty|nuts?", "nuts", _N),
    (r"tarte tatin", "tarte tatin", _Z),
    (r"creme brulee", "creme brulee", _Z),
    (r"pie crust(?: dough)?", "pie crust", _Z),
    (r"biscuit|brioche|pastry|bread", "bread & pastry", _Z),
    (r"(?:salted )?caramel", "caramel", _Z),
    (r"candied", "candied fruit", _Z),
    (r"butter(?:y)?", "butter", _Z),
    (r"honey(?:ed)?", "honey", _Z),
    (r"chocolate", "chocolate", _Z),
    (r"cocoa(?: powder)?", "cocoa", _Z),
    (r"leesy|lees", "lees", _Z),
    (r"botrytis", "botrytis", _Z),
]
_AROMA_RE = re.compile(
    r"(?<![a-z])(?:"
    + "|".join(f"(?P<t{i}>{p})" for i, (p, _, _) in enumerate(_AROMA_LEXICON))
    + r")(?![a-z])"
)


def _aromas(t: str) -> List[Aroma]:
    seen, result = set(), []
    for m in _AROMA_RE.finditer(t):
        before = t[max(0, m.start() - 25) : m.start()]
        if re.search(
            r"(?:\bwithout|\bno|\bnot|\black of|\bnever)\s+(?:\w+\s+){0,2}$", before
        ):
            continue  # "without botrytis", "no oak" are absences, not aromas
        _, name, category = _AROMA_LEXICON[int(m.lastgroup[1:])]
        if name not in seen:
            seen.add(name)
            result.append(Aroma(name=name, category=category))
    return result


# ---------------------------------------------------------------- entry point
def parse_tasting_note(note: Optional[str], title: Optional[str] = None) -> TastingInfo:
    """Parse one tasting note. ``title`` helps identify the grape(s) the title names."""
    t = _prep(note)
    title_t = _prep(title)
    if not t and not title_t:
        return TastingInfo()
    sensory = Sensory(
        body=_body(t),
        acidity=_acidity(t),
        tannin=_tannin(t),
        finish=_finish(t),
        sweetness=_sweetness(t),
    )
    return TastingInfo(
        aromas=_aromas(t),
        sensory=None if sensory.is_empty() else sensory,
        grapes=_grapes(t, title_t),
    )
