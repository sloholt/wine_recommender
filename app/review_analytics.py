"""Step 5: review analytics -> flavor profile per wine, lift (flavor x grape), and topics.

Requires Step 4 (catalog). These outputs feed slides 4-5.

Run from the project folder:
    python -m app.review_analytics
    python -m app.review_analytics --catalog data/other_catalog.csv
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save charts to files without opening a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

ARTIFACT_DIR = Path("artifacts")
DEFAULT_CATALOG = Path("data") / "wines.csv"
SEED = 42

# Words that appear in nearly every review and carry no flavor information.
WINE_STOP = {
    "wine",
    "wines",
    "flavors",
    "flavor",
    "aromas",
    "aroma",
    "palate",
    "finish",
    "nose",
    "notes",
    "drink",
    "drinks",
    "shows",
    "offers",
    "bottle",
    "glass",
    "vineyard",
    "vintage",
    "years",
    "year",
    "made",
    "now",
    "through",
    "well",
    "also",
    "just",
    "hint",
    "hints",
    "touch",
    "feel",
}
STOP_WORDS = list(ENGLISH_STOP_WORDS | WINE_STOP)

# Flavor vocabulary: the descriptors tasters use. Edit freely.
FLAVOR_TERMS = [
    # fruit
    "cherry",
    "black cherry",
    "blackberry",
    "raspberry",
    "strawberry",
    "plum",
    "cassis",
    "black currant",
    "cranberry",
    "blueberry",
    "fig",
    "raisin",
    "prune",
    "citrus",
    "lemon",
    "lime",
    "grapefruit",
    "orange",
    "apple",
    "green apple",
    "pear",
    "peach",
    "apricot",
    "nectarine",
    "pineapple",
    "tropical",
    "melon",
    "passion fruit",
    "mango",
    "lychee",
    # oak and aging
    "oak",
    "vanilla",
    "toast",
    "toasty",
    "smoke",
    "smoky",
    "cedar",
    "tobacco",
    "leather",
    "chocolate",
    "cocoa",
    "coffee",
    "espresso",
    "mocha",
    "caramel",
    "butter",
    "buttery",
    "cream",
    "brioche",
    "almond",
    "hazelnut",
    "honey",
    # earth, herb, spice
    "earth",
    "earthy",
    "mineral",
    "minerality",
    "flint",
    "graphite",
    "slate",
    "mushroom",
    "forest floor",
    "herb",
    "herbal",
    "sage",
    "thyme",
    "mint",
    "eucalyptus",
    "grass",
    "grassy",
    "bell pepper",
    "olive",
    "pepper",
    "black pepper",
    "spice",
    "clove",
    "cinnamon",
    "licorice",
    "anise",
    "floral",
    "violet",
    "rose",
    "petrol",
    "gamey",
    "meaty",
    # structure
    "tannic",
    "firm tannins",
    "soft tannins",
    "acidity",
    "crisp",
    "zesty",
    "juicy",
    "jammy",
    "ripe",
    "bold",
    "rich",
    "lush",
    "light",
    "elegant",
    "smooth",
    "silky",
    "sweet",
    "dry",
    "tart",
]

N_TOP_VARIETIES = 8
MIN_SUPPORT = 20  # term must appear in at least this many wines of the grape
N_TOPICS = 8


def load_catalog(path: Path) -> pd.DataFrame:
    """Load the Step 4 catalog from .csv, .parquet or .pkl."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        catalog = pd.read_csv(path)
    elif suffix == ".parquet":
        catalog = pd.read_parquet(path)
    elif suffix in (".pkl", ".pickle"):
        catalog = pd.read_pickle(path)
    else:
        raise ValueError(f"Unsupported catalog format: {path.suffix}")

    # Datasets like WineEnthusiast name the review text "description".
    if "review" not in catalog.columns and "description" in catalog.columns:
        catalog = catalog.rename(columns={"description": "review"})

    missing = {"review", "variety"} - set(catalog.columns)
    if missing:
        raise ValueError(
            f"Catalog is missing required columns: {sorted(missing)}. "
            f"Columns found: {list(catalog.columns)}"
        )
    catalog = catalog.dropna(subset=["variety"]).reset_index(drop=True)
    catalog["review"] = catalog["review"].fillna("").astype(str)
    return catalog


# ---- 5a. Flavor presence per wine ------------------------------------------------
def add_flavor_profiles(catalog: pd.DataFrame):
    flavor_vec = CountVectorizer(
        vocabulary=FLAVOR_TERMS, ngram_range=(1, 2), binary=True, lowercase=True
    )
    F = flavor_vec.fit_transform(
        catalog["review"]
    ).tocsr()  # wines x flavor terms (0/1)
    flavor_names = np.array(flavor_vec.get_feature_names_out())

    # Keep each wine's flavor words, rarest first (rare words are more distinctive).
    term_share = np.asarray(F.mean(axis=0)).ravel()
    rarity_rank = np.argsort(np.argsort(term_share))  # 0 = rarest term
    catalog["flavors"] = [
        [
            flavor_names[j]
            for j in sorted(
                F.indices[F.indptr[i] : F.indptr[i + 1]], key=rarity_rank.__getitem__
            )
        ][:8]
        for i in range(F.shape[0])
    ]
    print(
        "Wines with at least one flavor term:",
        f"{(np.asarray(F.sum(axis=1)).ravel() > 0).mean():.0%}",
    )
    return F, flavor_names, term_share


# ---- 5b. Lift: which flavor words are characteristic of each grape? --------------
# lift = P(term | grape) / P(term). Lift 3 = the word shows up 3x more often for that grape.
def compute_lift(catalog, F, flavor_names, term_share, out_dir: Path) -> pd.DataFrame:
    top_varieties = (
        catalog["variety"].value_counts().head(N_TOP_VARIETIES).index.tolist()
    )

    lift_rows = []
    for var in top_varieties:
        mask = (catalog["variety"] == var).to_numpy()
        in_var = np.asarray(F[mask].sum(axis=0)).ravel()
        p_given = in_var / mask.sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            lift = np.where(term_share > 0, p_given / term_share, 0)
        for j in np.argsort(-lift):
            if in_var[j] >= MIN_SUPPORT:
                lift_rows.append(
                    {
                        "variety": var,
                        "term": flavor_names[j],
                        "lift": round(float(lift[j]), 2),
                        "share_of_grape": round(float(p_given[j]), 3),
                        "n_wines": int(in_var[j]),
                    }
                )
    lift_table = pd.DataFrame(
        lift_rows, columns=["variety", "term", "lift", "share_of_grape", "n_wines"]
    )

    if lift_table.empty:
        print(
            f"No flavor terms reached MIN_SUPPORT={MIN_SUPPORT}; skipping lift table and heatmap."
        )
        return lift_table

    print("\nTop flavor words per grape (by lift):")
    print(lift_table.groupby("variety").head(5).to_string(index=False))
    lift_table.to_csv(out_dir / "flavor_lift.csv", index=False)

    # Heatmap for the slides: top 3 lift terms per grape, shown across all grapes.
    show_terms = list(dict.fromkeys(lift_table.groupby("variety").head(3)["term"]))
    heat = lift_table.pivot_table(
        index="variety", columns="term", values="lift"
    ).reindex(index=top_varieties, columns=show_terms)
    fig, ax = plt.subplots(
        figsize=(min(1 + 0.55 * len(show_terms), 16), 0.5 * len(top_varieties) + 1.5)
    )
    im = ax.imshow(heat.fillna(0).values, cmap="Reds", aspect="auto")
    ax.set_xticks(range(len(show_terms)), show_terms, rotation=60, ha="right")
    ax.set_yticks(range(len(top_varieties)), top_varieties)
    for (r, c), v in np.ndenumerate(heat.values):
        if not np.isnan(v):
            ax.text(c, r, f"{v:.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="lift")
    ax.set_title("Which flavor words define each grape (lift)")
    fig.tight_layout()
    fig.savefig(out_dir / "flavor_lift_heatmap.png", dpi=200)
    plt.close(fig)
    return lift_table


# ---- 5c. Topics (LDA): flavor themes across all reviews --------------------------
def add_topics(catalog: pd.DataFrame) -> dict:
    count_vec = CountVectorizer(
        stop_words=STOP_WORDS,
        max_df=0.4,
        min_df=5,
        max_features=5000,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
    )
    C = count_vec.fit_transform(catalog["review"])
    lda = LatentDirichletAllocation(
        n_components=N_TOPICS,
        random_state=SEED,
        learning_method="online",
        batch_size=2048,
        max_iter=10,
    )
    doc_topics = lda.fit_transform(C)
    vocab = np.array(count_vec.get_feature_names_out())
    topic_words = {
        t: vocab[np.argsort(-lda.components_[t])[:10]].tolist() for t in range(N_TOPICS)
    }
    print("\nTopics:")
    for t, words in topic_words.items():
        print(f"Topic {t}: {', '.join(words)}")
    catalog["topic"] = doc_topics.argmax(axis=1)
    # Tip: after reading the word lists, give each topic a short name for the slides, e.g.
    # TOPIC_NAMES = {0: "dark fruit & oak", 1: "citrus & minerals", ...}
    return topic_words


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Step 5: flavor profiles, lift and topics from wine reviews."
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=DEFAULT_CATALOG,
        help=f"Step 4 catalog file (.csv/.parquet/.pkl). Default: {DEFAULT_CATALOG}",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ARTIFACT_DIR,
        help=f"Where outputs are saved (default: {ARTIFACT_DIR})",
    )
    args = parser.parse_args(argv)

    if not args.catalog.exists():
        parser.error(
            f"catalog not found: {args.catalog} (run Step 4 first, or pass --catalog)"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    catalog = load_catalog(args.catalog)
    print(f"Loaded {len(catalog):,} wines from {args.catalog}")

    F, flavor_names, term_share = add_flavor_profiles(catalog)
    compute_lift(catalog, F, flavor_names, term_share, args.out_dir)
    add_topics(catalog)

    # Save the enriched catalog so later steps can use the flavors and topics.
    # Pickle keeps the 'flavors' column as real Python lists.
    enriched_path = args.out_dir / "catalog_with_flavors.pkl"
    catalog.to_pickle(enriched_path)

    print(
        f"\nSaved to {args.out_dir}: flavor_lift.csv, flavor_lift_heatmap.png, {enriched_path.name}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
