"""Smoke test for the student performance ML pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure imports work when run from repo root.
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from data_loader import load_dataset
from train_model import MODEL_PATH
from utils import decode_prediction_label, load_model_artifact, prepare_input_dataframe


def main() -> None:
    df = load_dataset()
    assert not df.empty, "Dataset is empty"

    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model artifact missing. Run python src/train_model.py first.")

    artifact = load_model_artifact(MODEL_PATH)
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]

    sample_row = df.drop(columns=["G3"]).iloc[0].to_dict()
    sample_input = prepare_input_dataframe(sample_row, feature_columns)
    pred = int(model.predict(sample_input)[0])

    print("Smoke test prediction:", pred, decode_prediction_label(pred))


if __name__ == "__main__":
    main()
