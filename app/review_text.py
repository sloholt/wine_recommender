"""Review text -> TF-IDF vectors, shared by the flavor profiles, topics and evaluation.

Grape and place names are hidden before vectorizing, so the review block measures
how a wine is DESCRIBED (taste, texture, structure), not what it is or where it is
from: grape and region are handled separately by the recommender's "local" slots.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

# Words that appear in nearly every review and carry no flavor information.
WINE_STOP = {
    "wine", "wines", "flavors", "flavor", "aromas", "aroma", "palate", "finish",
    "nose", "notes", "drink", "drinks", "shows", "offers", "bottle", "glass",
    "vineyard", "vintage", "years", "year", "made", "now", "through", "well",
    "also", "just", "hint", "hints", "touch", "feel",
}
STOP_WORDS = list(ENGLISH_STOP_WORDS | WINE_STOP)

# Parts of grape / place names too common to hide.
GENERIC_NAME_WORDS = {
    "red", "white", "blend", "rose", "sparkling", "de", "la", "du", "di", "del",
    "valley", "county", "coast", "hills", "river", "mountain", "district", "vino",
}
TFIDF_PARAMS = dict(
    stop_words=STOP_WORDS, ngram_range=(1, 2), min_df=3, max_df=0.5,
    max_features=30000, sublinear_tf=True, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
)
_WORD = re.compile(r"[a-z]+")


def ascii_lower(text) -> str:
    t = unicodedata.normalize("NFKD", text if isinstance(text, str) else "")
    return "".join(c for c in t if not unicodedata.combining(c)).lower()


def name_words(values: Iterable) -> set:
    """Distinctive words (> 3 letters, not generic) of names such as grapes or regions."""
    out = set()
    for v in values:
        if isinstance(v, str):
            out.update(
                w for w in _WORD.findall(ascii_lower(v))
                if len(w) > 3 and w not in GENERIC_NAME_WORDS
            )
    return out


def hide_words(texts: Iterable, hidden: set) -> list:
    """Lowercase, accent-free texts with every word in ``hidden`` blanked out."""
    return [_WORD.sub(lambda m: " " if m.group() in hidden else m.group(), ascii_lower(t))
            for t in texts]


def words_to_hide(meta: pd.DataFrame, flavor_terms: Iterable[str]) -> set:
    """Grape and place name words in the catalog, minus words that are also flavors
    (the region 'Orange' must not hide the aroma 'orange')."""
    grapes = name_words(meta["variety"]) | name_words(
        g for gs in meta["grapes"].dropna() for g in str(gs).split("|")
    )
    places = name_words(meta["region"]) | name_words(meta["country"])
    return (grapes | places) - name_words(flavor_terms)


def review_matrix(
    meta: pd.DataFrame, reviews: pd.DataFrame, flavor_terms: Iterable[str]
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """TF-IDF of each wine's reviews (all joined), rows aligned with ``meta``.

    Returns (matrix, has_review); rows of wines without review text are empty.
    Rows are L2-normalized, so a dot product is a cosine similarity.
    """
    text = (
        reviews.dropna(subset=["review"])
        .groupby("wine_id")["review"].apply(" ".join)
    )
    has_review = meta["wine_id"].isin(text.index).to_numpy()
    rows = np.flatnonzero(has_review)
    docs = hide_words(text.loc[meta["wine_id"].to_numpy()[rows]], words_to_hide(meta, flavor_terms))
    X = TfidfVectorizer(**TFIDF_PARAMS).fit_transform(docs).tocsr().astype(np.float32)
    # One row per wine in meta: wines without reviews point at an appended empty row.
    padded = sparse.vstack([X, sparse.csr_matrix((1, X.shape[1]), dtype=np.float32)]).tocsr()
    index = np.full(len(meta), X.shape[0], dtype=np.int64)
    index[rows] = np.arange(len(rows))
    return padded[index], has_review
