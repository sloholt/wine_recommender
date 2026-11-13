import unicodedata

REGION_ALIASES = {
    "valle del limari": "limari",
    "limari valley": "limari",
}


def normalize(s):
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return s.lower().strip()


def match_key(producer, wine_name, variety, region, vintage):
    region = REGION_ALIASES.get(normalize(region), normalize(region))
    return "|".join(
        [
            normalize(producer),
            normalize(wine_name),
            normalize(variety),
            region,
            str(vintage or ""),
        ]
    )
