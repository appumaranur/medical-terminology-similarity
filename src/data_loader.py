"""Load and normalize the medical terminology dataset."""

from pathlib import Path

import pandas as pd


def load_medical_terms(file_path: str | Path) -> pd.DataFrame:
    """Read the CSV, normalize text, and keep one row per term."""
    dataset_path = Path(file_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset was not found: {dataset_path}")

    data = pd.read_csv(dataset_path)
    required_columns = {"term", "category"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"The dataset is missing required columns: {missing}")

    data = data.dropna(subset=["term"]).copy()
    text_columns = [
        "term",
        "category",
        "description",
        "related_terms",
        "relation_type",
    ]
    for column in text_columns:
        if column not in data.columns:
            data[column] = ""
        data[column] = data[column].fillna("").astype(str).str.strip()

    data["term_key"] = data["term"].str.casefold()
    data = data[data["term"] != ""].drop_duplicates("term_key")
    return data.drop(columns=["term_key"]).reset_index(drop=True)
