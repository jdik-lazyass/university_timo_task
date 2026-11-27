
from pathlib import Path

import numpy as np
import pandas as pd


BASE_PATH = Path("../data/processed/base.csv")
INCOMING_DIR = Path("../data/incoming")
RESULTS_DIR = Path("../results")

CATEGORICAL_FEATURES = [
    "Platform",
    "Genre",
    "Publisher",
    "Developer",
    "Rating",
]

NUMERIC_FEATURES = ["Year_of_Release"]

EPSILON = 1e-6
MIN_CATEGORY_COUNT = 5


def prepare_categories(base, batch, column):
    base_values = base[column].fillna("Unknown").astype(str)
    batch_values = batch[column].fillna("Unknown").astype(str)
    counts = base_values.value_counts()
    rare = counts[counts < MIN_CATEGORY_COUNT].index

    base_values = base_values.where(~base_values.isin(rare), "Other")
    batch_values = batch_values.where(~batch_values.isin(rare), "Other")

    categories = sorted(
        set(base_values.unique()) | set(batch_values.unique())
    )

    return base_values, batch_values, categories


def calculate_categorical_psi(base, batch, column):
    base_values, batch_values, categories = prepare_categories(
        base, batch, column
    )

    base_dist = base_values.value_counts(normalize=True)
    batch_dist = batch_values.value_counts(normalize=True)

    psi = 0.0

    for category in categories:
        expected = max(base_dist.get(category, 0), EPSILON)
        actual = max(batch_dist.get(category, 0), EPSILON)

        psi += (actual - expected) * np.log(actual / expected)

    return psi


def calculate_numeric_psi(base, batch, column):
    base_values = pd.to_numeric(base[column], errors="coerce").dropna()
    batch_values = pd.to_numeric(batch[column], errors="coerce").dropna()
    bins = np.unique(
        np.quantile(base_values, np.linspace(0, 1, 11))
    )

    if len(bins) < 2:
        return np.nan

    bins[0] = -np.inf
    bins[-1] = np.inf

    base_counts = pd.cut(
        base_values, bins=bins
    ).value_counts(sort=False)

    batch_counts = pd.cut(
        batch_values, bins=bins
    ).value_counts(sort=False)

    base_dist = base_counts / base_counts.sum()
    batch_dist = batch_counts / batch_counts.sum()

    psi = 0.0

    for expected, actual in zip(base_dist, batch_dist):
        expected = max(expected, EPSILON)
        actual = max(actual, EPSILON)

        psi += (actual - expected) * np.log(actual / expected)

    return psi


def interpret_psi(value):
    if pd.isna(value):
        return "Не удалось рассчитать"
    if value < 0.10:
        return "Стабильность"
    if value <= 0.25:
        return "Умеренное изменение"
    return "Существенное изменение"


def main():
    base = pd.read_csv(BASE_PATH)
    results = []

    for path in sorted(INCOMING_DIR.glob("batch_*.csv")):
        batch = pd.read_csv(path)

        for column in CATEGORICAL_FEATURES:
            value = calculate_categorical_psi(base, batch, column)

            results.append({
                "Batch": path.stem,
                "Feature": column,
                "PSI": value,
                "Interpretation": interpret_psi(value),
            })

        for column in NUMERIC_FEATURES:
            value = calculate_numeric_psi(base, batch, column)

            results.append({
                "Batch": path.stem,
                "Feature": column,
                "PSI": value,
                "Interpretation": interpret_psi(value),
            })

    results_df = pd.DataFrame(results)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(
        RESULTS_DIR / "psi_results.csv",
        index=False
    )

    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
