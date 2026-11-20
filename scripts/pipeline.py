import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from features import FeatureEngineer

NUMERIC_FEATURES = [
    "Year_of_Release",
    "is_rare_publisher",
    "publisher_avg_score"
]

CATEGORICAL_FEATURES = [
    "Platform",
    "Genre",
    "Publisher",
    "Developer",
    "Rating",
    "platform_generation",
    "genre_rating_combo"
]


def create_preprocessor():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(
            strategy="constant",
            fill_value="Unknown"
        )),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES)
    ])


def create_pipeline(model):
    return Pipeline([
        ("features", FeatureEngineer()),
        ("preprocessor", create_preprocessor()),
        ("model", model)
    ])
