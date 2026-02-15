"""Data preprocessing helpers for the student performance classification task."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class DatasetSplit:
    """Container for train/test splits."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def create_target(df: pd.DataFrame, pass_mark: int = 10) -> pd.Series:
    """Create a binary pass/fail target from the final grade column G3.

    The UCI data uses a 0-20 grading scale. This function creates labels:
    - 1 for pass (G3 >= pass_mark)
    - 0 for fail (G3 < pass_mark)
    """
    if "G3" not in df.columns:
        raise KeyError("Expected column 'G3' for target creation.")
    return (df["G3"] >= pass_mark).astype(int)


def create_features_and_target(df: pd.DataFrame, pass_mark: int = 10) -> tuple[pd.DataFrame, pd.Series]:
    """Create feature matrix X and label vector y.

    G3 is removed from the feature matrix to avoid data leakage.
    """
    y = create_target(df, pass_mark=pass_mark)
    X = df.drop(columns=["G3"])  # drop target source column
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build a column-wise preprocessing pipeline.

    Steps:
    - Numerical features: median imputation + standard scaling
    - Categorical features: most-frequent imputation + one-hot encoding
    """
    categorical_columns = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_columns = X.select_dtypes(exclude=["object", "category"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ]
    )
    return preprocessor


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> DatasetSplit:
    """Split data into train/test sets using stratification on target labels."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return DatasetSplit(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
