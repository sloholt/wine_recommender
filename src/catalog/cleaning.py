"""Clean wine reviews and build a wine-level catalog."""

import re
import unicodedata

import numpy as np
import pandas as pd

# Recognize four-digit vintages in wine titles.
YEAR_RE = r"\b(?:19|20)\d{2}\b"

REQUIRED_COLUMNS = {
    "title",
    "winery",
    "variety",
    "country",
    "province",
    "region",
    "price",
    "score",
    "review",
}

WHITE = {
    "chardonnay",
    "sauvignon blanc",
    "riesling",
    "pinot gris",
    "pinot grigio",
    "white blend",
    "gewurztraminer",
    "viognier",
    "chenin blanc",
    "gruner veltliner",
    "albarino",
    "alvarinho",
    "torrontes",
    "verdejo",
    "moscato",
    "muscat",
    "moscatel",
    "semillon",
    "vermentino",
    "garganega",
    "friulano",
    "pinot blanc",
    "pinot bianco",
    "marsanne",
    "roussanne",
    "melon",
    "assyrtiko",
    "fiano",
    "greco",
    "falanghina",
    "trebbiano",
    "viura",
    "macabeo",
    "godello",
    "verdicchio",
    "muscadet",
    "sylvaner",
    "silvaner",
    "muller thurgau",
    "furmint",
    "loureiro",
    "arinto",
    "encruzado",
    "grenache blanc",
    "picpoul",
    "aligote",
    "pecorino",
    "arneis",
    "cortese",
    "kerner",
    "chasselas",
    "petit manseng",
    "colombard",
    "ugni blanc",
    "malvasia",
    "moschofilero",
    "rhone style white blend",
    "bordeaux style white blend",
    "portuguese white",
    "austrian white blend",
    "white riesling",
    "vidal",
    "vidal blanc",
    "seyval blanc",
    "traminette",
    "catarratto",
    "grillo",
    "ribolla gialla",
    "welschriesling",
    "sauvignon",
    "sauvignon gris",
    "garnacha blanca",
    "xarel lo",
    "verdelho",
}

SPARKLING = {
    "champagne blend",
    "sparkling blend",
    "prosecco",
    "glera",
    "cava",
    "portuguese sparkling",
}

ROSE = {
    "rose",
    "rosado",
    "rosato",
    "rosado blend",
}


def simple_norm(text: object) -> str:
    """Lowercase text and remove accents and punctuation."""
    if pd.isna(text):
        return ""

    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(
        character for character in text if not unicodedata.combining(character)
    )
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def color_of(variety: object) -> str:
    """Estimate wine color/style from the grape or blend name.

    This preserves the notebook's heuristic. Unrecognized nonempty
    varieties default to red; grape alone cannot reliably identify
    every rosé or sparkling wine.
    """
    variety = simple_norm(variety)

    if not variety:
        return "unknown"

    if variety in SPARKLING or "sparkling" in variety:
        return "sparkling"

    if variety in ROSE:
        return "rose"

    white_words = {"white", "blanc", "bianco", "blanca", "branco", "weiss"}
    if variety in WHITE or white_words.intersection(variety.split()):
        return "white"

    return "red"


def clean_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    """Clean standardized review rows and extract wine identities.

    Expects the column names returned by loading.load_reviews().
    Returns a new DataFrame without changing the input.
    """
    missing = sorted(REQUIRED_COLUMNS - set(reviews.columns))
    if missing:
        raise ValueError(f"Missing required review columns: {missing}")

    cleaned = reviews.copy()

    # Treat empty strings as missing and trim surrounding whitespace.
    text_columns = [
        "title",
        "winery",
        "variety",
        "country",
        "province",
        "region",
        "review",
    ]
    for column in text_columns:
        cleaned[column] = (
            cleaned[column].astype("string").str.strip().replace("", pd.NA)
        )

    cleaned = (
        cleaned.dropna(subset=["review", "winery", "title"])
        .drop_duplicates(subset=["winery", "title", "review"])
        .copy()
    )

    for column in ["price", "score"]:
        cleaned[column] = pd.to_numeric(
            cleaned[column],
            errors="coerce",
        )

    # Nonpositive prices cannot be used as valid bottle prices.
    cleaned.loc[cleaned["price"] <= 0, "price"] = np.nan

    cleaned["vintage"] = pd.to_numeric(
        cleaned["title"].str.extract(f"({YEAR_RE})", expand=False),
        errors="coerce",
    ).astype("Int64")

    # Remove vintage and trailing "(Region)" from the wine identity.
    cleaned["wine"] = (
        cleaned["title"]
        .str.replace(YEAR_RE, "", regex=True)
        .str.replace(r"\s*\([^)]*\)\s*$", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    cleaned = cleaned.loc[cleaned["wine"].ne("")]

    return cleaned.reset_index(drop=True)


def build_catalog(
    cleaned_reviews: pd.DataFrame,
    min_reviews: int = 1,
) -> pd.DataFrame:
    """Aggregate cleaned reviews into one row per winery and wine.

    Prices are medians across available reviews/vintages, rather
    than current retail prices. Scores are averaged.
    """
    if min_reviews < 1:
        raise ValueError("min_reviews must be at least 1.")

    required = REQUIRED_COLUMNS | {"wine", "vintage"}
    missing = sorted(required - set(cleaned_reviews.columns))
    if missing:
        raise ValueError(
            f"Missing cleaned-review columns: {missing}. " "Run clean_reviews() first."
        )

    catalog = (
        cleaned_reviews.groupby(["winery", "wine"], sort=False)
        .agg(
            n_reviews=("review", "size"),
            review=("review", " ".join),
            variety=("variety", "first"),
            country=("country", "first"),
            province=("province", "first"),
            region=("region", "first"),
            price=("price", "median"),
            score=("score", "mean"),
            vintages=(
                "vintage",
                lambda values: sorted(int(value) for value in values.dropna().unique()),
            ),
        )
        .reset_index()
    )

    catalog = catalog.loc[catalog["n_reviews"] >= min_reviews].reset_index(drop=True)

    catalog["wine_id"] = np.arange(len(catalog))
    catalog["match_key"] = catalog["wine"].map(simple_norm)
    catalog["color"] = catalog["variety"].map(color_of)

    return catalog
