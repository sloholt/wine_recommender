"""Match structured labels, OCR text, or typed names to a wine catalog."""

import re

import pandas as pd
from rapidfuzz import fuzz, process

from .cleaning import simple_norm

MATCH_THRESHOLD = 75
MAX_APPELLATION_CANDIDATES = 500

GENERIC_PRODUCER = {
    "famille",
    "domaine",
    "domaines",
    "chateau",
    "maison",
    "bodega",
    "bodegas",
    "cantina",
    "cantine",
    "tenuta",
    "weingut",
    "winery",
    "vineyards",
    "vineyard",
    "cellars",
    "estate",
    "estates",
    "wines",
    "de",
    "du",
    "des",
    "la",
    "le",
    "les",
    "di",
    "del",
    "the",
}

CLASSIFICATION_TOKENS = {
    "do",
    "doc",
    "docg",
    "doca",
    "dop",
    "aoc",
    "aop",
    "ava",
    "igt",
    "igp",
    "vdp",
    "go",
    "wo",
}

NOISE = {
    "ml",
    "vol",
    "alc",
    "product",
    "of",
    "wine",
    "vino",
    "vin",
    "contains",
    "sulfites",
    "estate",
    "bottled",
}

REQUIRED_COLUMNS = {
    "wine_id",
    "wine",
    "winery",
    "variety",
    "region",
    "province",
    "country",
    "score",
}


def _no_match(reason: str) -> dict:
    """Return a consistent result when matching is unavailable."""
    return {
        "matched": False,
        "anchor_ids": [],
        "candidates": [],
        "fallback_anchor_ids": [],
        "fallback_basis": None,
        "reason": reason,
    }


def _producer_core(producer) -> str:
    """Remove generic producer words before comparing names."""
    return " ".join(
        word for word in simple_norm(producer).split() if word not in GENERIC_PRODUCER
    )


def strip_classification(appellation: str | None) -> str:
    """Remove classification tokens from an appellation's edges."""
    if not appellation:
        return ""

    words = appellation.split()

    def is_classification(word):
        token = re.sub(r"[^a-z]", "", word.lower())
        return token in CLASSIFICATION_TOKENS

    while words and is_classification(words[0]):
        words.pop(0)

    while words and is_classification(words[-1]):
        words.pop()

    return " ".join(words)


def _appellation_key(label: dict) -> str:
    return simple_norm(strip_classification(label.get("appellation")))


def _prepare_catalog(catalog: pd.DataFrame) -> pd.DataFrame:
    """Prepare matching keys without modifying the caller's catalog.

    Use wine_id as the lookup index, preserving the existing IDs.
    """
    missing = sorted(REQUIRED_COLUMNS - set(catalog.columns))
    if missing:
        raise ValueError(f"Missing catalog columns: {missing}")

    if catalog["wine_id"].isna().any() or catalog["wine_id"].duplicated().any():
        raise ValueError("Catalog wine_id values must be unique and nonmissing.")

    prepared = catalog.copy().set_index("wine_id", drop=False)
    prepared["match_key"] = prepared["wine"].map(simple_norm)
    prepared["winery_core"] = prepared["winery"].map(_producer_core)
    prepared["region_key"] = prepared["region"].map(simple_norm)

    return prepared


def _candidate_ids(
    catalog: pd.DataFrame,
    producer: str,
    name: str,
    variety: str,
    label: dict,
) -> set[int]:
    """Collect candidates using producer, appellation, and title."""
    ids = set()

    # 1. Search producer names, then collect their wines.
    core = _producer_core(producer)
    if core:
        wineries = pd.Series(catalog["winery_core"].unique())
        hits = process.extract(
            core,
            wineries,
            scorer=fuzz.token_set_ratio,
            limit=10,
        )
        good = {winery for winery, score, _ in hits if score >= 85 and winery}

        if good:
            ids.update(
                int(i) for i in catalog.index[catalog["winery_core"].isin(good)][:300]
            )

    # 2. Search the appellation, prioritizing the same grape.
    appellation = _appellation_key(label)
    if appellation:
        same_app = catalog.loc[catalog["region_key"] == appellation]

        if variety:
            same_variety = same_app["variety"].map(simple_norm) == simple_norm(variety)
            same_app = pd.concat(
                [
                    same_app.loc[same_variety],
                    same_app.loc[~same_variety],
                ]
            )

        ids.update(int(i) for i in same_app.index[:MAX_APPELLATION_CANDIDATES])

    # 3. Search catalog titles.
    query = simple_norm(f"{producer} {name} {variety}")
    if query:
        hits = process.extract(
            query,
            catalog["match_key"],
            scorer=fuzz.token_set_ratio,
            limit=50,
        )
        ids.update(int(i) for _, _, i in hits)

    return ids


def _fallback_pool(
    catalog: pd.DataFrame,
    label: dict,
    variety: str,
) -> tuple[pd.DataFrame, str]:
    """Select reference wines when no confident identity match exists."""
    appellation = _appellation_key(label)
    same_app = (
        catalog.loc[catalog["region_key"] == appellation]
        if appellation
        else catalog.iloc[0:0]
    )

    if variety and not same_app.empty:
        same_app_variety = same_app.loc[
            same_app["variety"].map(simple_norm) == simple_norm(variety)
        ]
        if len(same_app_variety) >= 5:
            return same_app_variety, "same appellation and grape"

    if len(same_app) >= 5:
        return same_app, "same appellation"

    pool = catalog
    basis = "well-rated wines"

    if variety:
        same_variety = pool.loc[
            pool["variety"].map(simple_norm) == simple_norm(variety)
        ]
        if not same_variety.empty:
            pool = same_variety
            basis = "same grape"

    if label.get("country") and not pool.empty:
        same_country = pool.loc[
            pool["country"].map(simple_norm) == simple_norm(label["country"])
        ]
        if not same_country.empty:
            pool = same_country
            basis += " and country"

    return pool, basis


def match_label_fields(
    label: dict,
    catalog: pd.DataFrame,
    top: int = 5,
    threshold: float = MATCH_THRESHOLD,
) -> dict:
    """Match extracted label fields to a catalog wine.

    Weighted score:
        45% producer
        35% wine name and vineyard
        10% grape
        10% place

    Fallback IDs represent similar reference wines, not identity matches.
    """
    if top < 1:
        raise ValueError("top must be at least 1.")

    catalog = _prepare_catalog(catalog)
    if catalog.empty:
        return _no_match("The wine catalog is empty.")

    producer = label.get("producer") or ""
    name = " ".join(
        value
        for value in (
            label.get("wine_name"),
            label.get("vineyard"),
        )
        if value
    )
    variety = label.get("variety") or ""
    place = simple_norm(
        " ".join(
            value
            for value in (
                label.get("appellation"),
                label.get("region"),
                label.get("country"),
            )
            if value
        )
    )

    if not simple_norm(f"{producer} {name} {variety}"):
        return _no_match("No name could be read from the label.")

    producer_core = _producer_core(producer) or simple_norm(producer)
    rows = []

    for wine_id in _candidate_ids(catalog, producer, name, variety, label):
        wine = catalog.loc[wine_id]
        wine_place = simple_norm(
            " ".join(
                value
                for value in (
                    wine["region"],
                    wine["province"],
                    wine["country"],
                )
                if isinstance(value, str)
            )
        )
        winery_core = wine["winery_core"] or simple_norm(wine["winery"])

        producer_score = (
            fuzz.token_set_ratio(producer_core, winery_core) if producer else 50
        )
        name_score = (
            fuzz.token_set_ratio(simple_norm(name), wine["match_key"]) if name else 50
        )
        variety_score = (
            100
            if variety and simple_norm(variety) == simple_norm(wine["variety"])
            else 0
        )
        place_score = fuzz.token_set_ratio(place, wine_place) if place else 50

        score = (
            0.45 * producer_score
            + 0.35 * name_score
            + 0.10 * variety_score
            + 0.10 * place_score
        )

        # Preserve the notebook's guard against a different grape,
        # unless the wine name is a near-exact fuzzy match.
        can_match = not (
            variety
            and simple_norm(wine["variety"])
            and variety_score == 0
            and name_score < 90
        )

        rows.append((score, wine_id, producer_score, can_match))

    rows.sort(reverse=True)

    candidates = [
        {
            "wine_id": wine_id,
            "wine": catalog.at[wine_id, "wine"],
            "winery": catalog.at[wine_id, "winery"],
            "same_producer": bool(producer) and producer_score >= 90,
            "match_score": round(score, 1),
        }
        for score, wine_id, producer_score, _ in rows[:top]
    ]

    if rows and rows[0][0] >= threshold and rows[0][3]:
        return {
            "matched": True,
            "anchor_ids": [rows[0][1]],
            "candidates": candidates,
            "fallback_anchor_ids": [],
            "fallback_basis": None,
            "reason": f"Matched '{candidates[0]['wine']}'.",
        }

    pool, basis = _fallback_pool(catalog, label, variety)
    fallback_ids = (
        pool.sort_values("score", ascending=False)["wine_id"]
        .head(10)
        .astype(int)
        .tolist()
    )

    return {
        "matched": False,
        "anchor_ids": [],
        "candidates": candidates,
        "fallback_anchor_ids": fallback_ids,
        "fallback_basis": basis,
        "reason": "No confident catalog match for this bottle.",
    }


def match_ocr_text(
    ocr_text: str,
    catalog: pd.DataFrame,
    top: int = 5,
    threshold: float = MATCH_THRESHOLD,
) -> dict:
    """Match raw OCR text without structured label extraction."""
    if top < 1:
        raise ValueError("top must be at least 1.")

    catalog = _prepare_catalog(catalog)
    if catalog.empty:
        return _no_match("The wine catalog is empty.")

    words = {
        word
        for word in simple_norm(ocr_text).split()
        if len(word) > 2 and not word.isdigit() and word not in NOISE
    }
    if not words:
        return _no_match("No useful wine-identifying text was found.")

    hits = process.extract(
        " ".join(sorted(words)),
        catalog["match_key"],
        scorer=fuzz.token_set_ratio,
        limit=200,
    )

    scored = [
        (
            score,
            len(words & set(catalog.at[wine_id, "match_key"].split())),
            int(wine_id),
        )
        for _, score, wine_id in hits
    ]

    best_overlap = max(overlap for _, overlap, _ in scored)
    scored.sort(
        key=lambda item: (
            item[1] == best_overlap,
            item[0],
            item[1],
        ),
        reverse=True,
    )

    candidates = [
        {
            "wine_id": wine_id,
            "wine": catalog.at[wine_id, "wine"],
            "score": round(score, 1),
            "words_matched": overlap,
        }
        for score, overlap, wine_id in scored[:top]
    ]

    best = candidates[0]
    matched = (best["score"] >= threshold and best["words_matched"] >= 2) or best[
        "words_matched"
    ] >= 4

    return {
        "matched": matched,
        "anchor_ids": [best["wine_id"]] if matched else [],
        "candidates": candidates,
        "fallback_anchor_ids": [],
        "fallback_basis": None,
        "reason": (f"Matched '{best['wine']}'." if matched else "No confident match."),
    }


def match_wine_name(
    text: str,
    catalog: pd.DataFrame,
    top: int = 5,
    threshold: float = MATCH_THRESHOLD,
) -> dict:
    """Match a typed wine name using the notebook's title search."""
    if top < 1:
        raise ValueError("top must be at least 1.")

    catalog = _prepare_catalog(catalog)
    if catalog.empty:
        return _no_match("The wine catalog is empty.")

    query = simple_norm(text)
    if not any(len(word) > 2 and not word.isdigit() for word in query.split()):
        return _no_match("Enter a wine name.")

    # Inspect at least two hits to detect ambiguous one-word queries.
    hits = process.extract(
        query,
        catalog["match_key"],
        scorer=fuzz.token_set_ratio,
        limit=max(top, 2),
    )

    best_score = hits[0][1]
    best_id = int(hits[0][2])
    ties = sum(score == best_score for _, score, _ in hits)
    ambiguous = ties > 1 and len(query.split()) < 2
    matched = best_score >= threshold and not ambiguous

    candidates = [
        {
            "wine_id": int(wine_id),
            "wine": catalog.at[wine_id, "wine"],
            "match_score": round(score, 1),
        }
        for _, score, wine_id in hits[:top]
    ]

    return {
        "matched": matched,
        "anchor_ids": [best_id] if matched else [],
        "candidates": candidates,
        "fallback_anchor_ids": [],
        "fallback_basis": None,
        "reason": (
            f"Matched '{catalog.at[best_id, 'wine']}'."
            if matched
            else "No confident match for that name."
        ),
    }
