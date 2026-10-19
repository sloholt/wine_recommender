"""Load wine-review data and standardize column names."""

from pathlib import Path
from typing import Mapping

import pandas as pd

# Original dataset column -> project column name.
COLUMN_MAP = {
    "title": "title",
    "winery": "winery",
    "variety": "variety",
    "country": "country",
    "province": "province",
    "region_1": "region",
    "price": "price",
    "points": "score",
    "description": "review",
}


def load_reviews(
    csv_path: str | Path,
    column_map: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Read a wine-review CSV and return standardized columns.

    Args:
        csv_path: Path to the source CSV.
        column_map: Optional source-to-project column mapping.
            Defaults to the notebook's Wine Enthusiast mapping.

    Returns:
        A DataFrame containing the selected, renamed columns.
        Cleaning and wine-level aggregation happen separately.

    Raises:
        FileNotFoundError: If the CSV does not exist.
        ValueError: If required source columns are missing.
    """
    csv_path = Path(csv_path)
    mapping = dict(COLUMN_MAP if column_map is None else column_map)

    if not csv_path.is_file():
        raise FileNotFoundError(f"Wine dataset not found: {csv_path}")

    raw = pd.read_csv(csv_path)

    missing = [column for column in mapping if column not in raw.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {csv_path.name}: {missing}. "
            "Check the dataset or update column_map."
        )

    return raw.loc[:, list(mapping)].rename(columns=mapping).copy()
