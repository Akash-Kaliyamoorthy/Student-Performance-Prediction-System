"""Model training script for student performance prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from data_loader import load_dataset
from preprocessing import build_preprocessor, create_features_and_target, split_data


MODEL_PATH = Path("models/best_model.pkl")


def _candidate_models(random_state: int = 42) -> dict[str, tuple[Any, dict[str, list[Any]]]]:
    """Return candidate estimators and parameter grids."""
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=2000, random_state=random_state),
            {
                "model__C": [0.1, 1.0, 10.0],
                "model__solver": ["lbfgs", "liblinear"],
            },
        ),
        "random_forest": (
            RandomForestClassifier(random_state=random_state),
            {
                "model__n_estimators": [100, 300],
                "model__max_depth": [None, 10, 20],
                "model__min_samples_split": [2, 5],
            },
        ),
        "svm": (
            SVC(probability=True, random_state=random_state),
            {
                "model__C": [0.5, 1.0, 5.0],
                "model__kernel": ["rbf", "linear"],
                "model__gamma": ["scale", "auto"],
            },
        ),
        "gradient_boosting": (
            GradientBoostingClassifier(random_state=random_state),
            {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [2, 3],
            },
        ),
    }


def train_and_select_best_model(random_state: int = 42) -> dict[str, Any]:
    """Train multiple classifiers with CV and persist the best one."""
    df = load_dataset()
    X, y = create_features_and_target(df)
    split = split_data(X, y, random_state=random_state)

    preprocessor = build_preprocessor(split.X_train)
    candidates = _candidate_models(random_state=random_state)

    model_scores: dict[str, float] = {}
    best_name = ""
    best_search = None

    for name, (estimator, param_grid) in candidates.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="f1",
            cv=5,
            n_jobs=-1,
            verbose=0,
        )
        search.fit(split.X_train, split.y_train)
        y_pred = search.best_estimator_.predict(split.X_test)
        test_f1 = f1_score(split.y_test, y_pred)
        model_scores[name] = test_f1

        if best_search is None or test_f1 > model_scores.get(best_name, -1):
            best_name = name
            best_search = search

    assert best_search is not None

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": best_search.best_estimator_,
        "best_model_name": best_name,
        "best_cv_score": best_search.best_score_,
        "test_scores": model_scores,
        "feature_columns": split.X_train.columns.tolist(),
    }
    joblib.dump(artifact, MODEL_PATH)

    return {
        "saved_to": str(MODEL_PATH),
        "best_model_name": best_name,
        "best_cv_score": float(best_search.best_score_),
        "test_scores": model_scores,
    }


if __name__ == "__main__":
    results = train_and_select_best_model()
    print("Training complete.")
    print(results)
