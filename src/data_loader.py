"""Utilities for downloading and loading the UCI Student Performance dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# UCI archive URL for the student-mat dataset (semicolon-separated CSV).
DATASET_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00320/student.zip"
)
DATASET_ARCHIVE_MEMBER = "student-mat.csv"


def download_dataset(destination_path: str | Path = "data/student_performance.csv") -> Path:
    """Download the dataset from UCI and save it locally.

    Args:
        destination_path: Path where the CSV should be persisted.

    Returns:
        The resolved path to the saved CSV file.
    """
    destination = Path(destination_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    # pandas can read zip files directly via URL. The member file is selected
    # using the archive syntax `zip_url!member_name`.
    source = f"{DATASET_URL}!{DATASET_ARCHIVE_MEMBER}"
    df = pd.read_csv(source, sep=";")
    df.to_csv(destination, index=False)
    return destination.resolve()


def load_local_dataset(path: str | Path = "data/student_performance.csv") -> pd.DataFrame:
    """Load the local student performance CSV into a DataFrame."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. Run download_dataset() first."
        )
    return pd.read_csv(csv_path)


def load_dataset(path: str | Path = "data/student_performance.csv") -> pd.DataFrame:
    """Load the dataset, downloading it first if a local copy is missing."""
    csv_path = Path(path)
    if not csv_path.exists():
        download_dataset(csv_path)
    return load_local_dataset(csv_path)


if __name__ == "__main__":
    saved_path = download_dataset()
    df = load_local_dataset(saved_path)
    print(f"Dataset saved to: {saved_path}")
    print(f"Shape: {df.shape}")
