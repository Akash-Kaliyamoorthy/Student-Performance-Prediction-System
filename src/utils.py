"""Shared utility functions for model loading and API input handling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

MODEL_PATH = Path("models/best_model.pkl")


def load_model_artifact(model_path: str | Path = MODEL_PATH) -> dict[str, Any]:
    """Load and return the serialized model artifact."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    return joblib.load(path)


def prepare_input_dataframe(payload: dict[str, Any], feature_columns: list[str]) -> pd.DataFrame:
    """Build a single-row DataFrame aligned to training feature columns.

    Missing columns are filled with None so the preprocessing pipeline can impute.
    Extra payload keys are ignored.
    """
    row = {col: payload.get(col, None) for col in feature_columns}
    return pd.DataFrame([row], columns=feature_columns)


def decode_prediction_label(prediction: int) -> str:
    """Convert binary model output to a human-readable label."""
    return "pass" if int(prediction) == 1 else "fail"
