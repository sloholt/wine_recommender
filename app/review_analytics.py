"""Step 5: review analytics -> per-wine flavor profiles, lift (flavor x grape), topics.

Builds everything from the wine database (Step 4, data/wines.db):

  5a. Flavor profiles: one vector per wine from its aromas (wine_aromas) and
      sensory scores (wine_sensory); wines without a tasting note borrow their
      grape + region group's profile. Saved to artifacts/flavor_profiles/ and
      used by the app through app.flavor_profiles.FlavorProfiles.
  5b. Lift: which aromas are characteristic of each grape (table + heatmap, slides 4-5).
  5c. Topics (optional, --topics): LDA themes over the review text, for the slides.

The aromas and sensory scores are extracted from tasting notes during ingestion
(ingestion/core/tasting.py), so the text is parsed once, with one vocabulary.

Run from the project folder:
    python -m app.review_analytics
    python -m app.review_analytics --topics                  # also the LDA topics
    python -m app.review_analytics --topics --source "Wine Enthusiast"
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

from app.flavor_profiles import PROFILE_DIR, build_profiles
from database import repository as repo
from database.connection import DATABASE_PATH

ARTIFACT_DIR = Path("artifacts")
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

N_TOP_VARIETIES = 8
MIN_SUPPORT = 20  # aroma must appear in at least this many wines of the grape
N_TOPICS = 8


# ---- 5b. Lift: which aromas are characteristic of each grape? ------------------
# lift = P(aroma | grape) / P(aroma), over wines with at least one aroma.
# Lift 3 = the aroma shows up 3x more often for that grape.
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
    ax.set_title("Which aromas define each grape (lift)")
    fig.tight_layout()
    fig.savefig(out_dir / "flavor_lift_heatmap.png", dpi=200)
    plt.close(fig)
    return lift_table


# ---- 5c. Topics (LDA): flavor themes across all reviews --------------------------
def add_topics(catalog: pd.DataFrame, out_dir: Path) -> dict:
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
    pd.DataFrame(
        {"topic": list(topic_words), "top_words": [", ".join(w) for w in topic_words.values()]}
    ).to_csv(out_dir / "topic_words.csv", index=False)
    # Tip: after reading the word lists, give each topic a short name for the slides, e.g.
    # TOPIC_NAMES = {0: "dark fruit & oak", 1: "citrus & minerals", ...}
    return topic_words


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Step 5: flavor profiles, lift and topics from the wine database."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ARTIFACT_DIR,
        help=f"Where lift/topic outputs are saved (default: {ARTIFACT_DIR})",
    )
    parser.add_argument(
        "--profiles-dir",
        type=Path,
        default=PROFILE_DIR,
        help=f"Where the flavor profiles are saved (default: {PROFILE_DIR})",
    )
    parser.add_argument(
        "--topics", action="store_true", help="Also fit LDA topics on the review text (slower)"
    )
    parser.add_argument(
        "--source",
        default=None,
        help="With --topics: only use reviews from this source, e.g. 'Wine Enthusiast'",
    )
    args = parser.parse_args(argv)

    if not DATABASE_PATH.exists():
        parser.error(
            f"database not found: {DATABASE_PATH} (build it with `python -m ingestion.run`)"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 5a. Flavor profiles
    profiles = build_profiles()
    info = profiles.info
    print(
        f"Built flavor profiles for {info['n_wines']:,} wines "
        f"({info['n_own_profiles']:,} from their own tasting data)."
    )
    for source, count in info["profile_sources"].items():
        print(f"  {source:<16} {count:>8,}")
    profiles.save(args.profiles_dir)

    # 5b. Lift, over wines with at least one aroma
    has_aroma = (profiles.meta["n_aromas"] > 0).to_numpy()
    catalog = profiles.meta[has_aroma].reset_index(drop=True)
    F = profiles.aroma_presence[has_aroma].tocsr()
    aroma_names = np.array(
        [f.split(":", 1)[1] for f in profiles.feature_names if f.startswith("aroma:")]
    )
    term_share = np.asarray(F.mean(axis=0)).ravel()
    compute_lift(catalog, F, aroma_names, term_share, args.out_dir)

    # Sanity check: look-alikes for the most-rated wine with its own profile.
    own = profiles.meta[profiles.meta["profile_source"] == "own"]
    example = own.loc[own["n_user_ratings"].idxmax()]
    print(f"\nExample - wines similar to {example.title} ({example.variety}):")
    similar = profiles.similar(int(example.wine_id), k=5)
    print(similar[["title", "variety", "similarity", "match"]].to_string(index=False))

    # 5c. Topics (optional)
    if args.topics:
        reviews = repo.get_reviews(source=args.source)
        reviews["review"] = reviews["review"].fillna("").astype(str)
        print(f"\nFitting topics on {len(reviews):,} reviews")
        add_topics(reviews, args.out_dir)

    saved = ["flavor_lift.csv", "flavor_lift_heatmap.png"] + (["topic_words.csv"] if args.topics else [])
    print(f"\nSaved to {args.out_dir}: {', '.join(saved)}", file=sys.stderr)
    print(f"Saved flavor profiles to {args.profiles_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
