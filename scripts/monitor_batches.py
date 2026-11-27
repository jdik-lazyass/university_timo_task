from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

ROOT_DIR = Path(__file__).resolve().parent.parent

BASE_PATH = ROOT_DIR / "data" / "processed" / "base.csv"
INCOMING_DIR = ROOT_DIR / "data" / "incoming"
CLEANED_DIR = ROOT_DIR / "results" / "quality_checks"
MODEL_PATH = ROOT_DIR / "models" / "tuned_HistGradientBoostingRegressor.joblib"

RESULTS_DIR = ROOT_DIR / "results" / "monitoring"

FEATURE_COLUMNS = [
    "Platform",
    "Year_of_Release",
    "Genre",
    "Publisher",
    "Developer",
    "Rating",
]

TARGET_COLUMN = "Critic_Score"

CATEGORICAL_FEATURES = [
    "Publisher",
    "Genre",
    "Rating",
    "Developer",
    "Platform",
]

NUMERIC_FEATURES = [
    "Year_of_Release",
]

MIN_CATEGORY_COUNT = 5

EPSILON = 1e-6

def calculate_metrics(y_true, y_pred):
    """Рассчитывает основные метрики регрессии."""

    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "R2": r2_score(y_true, y_pred) if len(y_true) >= 2 else np.nan,
    }


def evaluate_batch(model, batch_name):

    cleaned_path = CLEANED_DIR / f"{batch_name}_cleaned.csv"

    if not cleaned_path.exists():
        raise FileNotFoundError(
            f"Не найден очищенный файл: {cleaned_path}\n"
            "Сначала запусти quality_checks.py."
        )

    df = pd.read_csv(cleaned_path)

    df[TARGET_COLUMN] = pd.to_numeric(
        df[TARGET_COLUMN],
        errors="coerce"
    )

    test = df.dropna(subset=[TARGET_COLUMN]).copy()

    if test.empty:
        print(f"{batch_name}: нет записей с известным Critic_Score")
        return {
            "Batch": batch_name,
            "ValidRecords": len(df),
            "EvaluatedRecords": 0,
            "MAE": np.nan,
            "RMSE": np.nan,
            "R2": np.nan,
        }

    X_test = test[FEATURE_COLUMNS]
    y_test = test[TARGET_COLUMN]

    predictions = model.predict(X_test)

    metrics = calculate_metrics(y_test, predictions)

    return {
        "Batch": batch_name,
        "ValidRecords": len(df),
        "EvaluatedRecords": len(test),
        **metrics,
    }

def prepare_categorical_values(base, batch, column):
    base_values = (
        base[column]
        .fillna("__MISSING__")
        .astype(str)
    )

    batch_values = (
        batch[column]
        .fillna("__MISSING__")
        .astype(str)
    )

    counts = base_values.value_counts()

    rare_categories = set(
        counts[
            (counts < MIN_CATEGORY_COUNT)
            & (counts.index != "__MISSING__")
        ].index
    )

    known_categories = set(counts.index)

    def transform_base(value):
        if value == "__MISSING__":
            return "__MISSING__"
        if value in rare_categories:
            return "__OTHER__"
        return value

    def transform_batch(value):
        if value == "__MISSING__":
            return "__MISSING__"
        if value in rare_categories:
            return "__OTHER__"
        if value not in known_categories:
            return "__UNSEEN__"
        return value

    base_values = base_values.map(transform_base)
    batch_values = batch_values.map(transform_batch)

    return base_values, batch_values


def calculate_categorical_psi(base, batch, column):
    base_values, batch_values = prepare_categorical_values(
        base,
        batch,
        column
    )

    categories = sorted(
        set(base_values.unique()) |
        set(batch_values.unique())
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

    base_values = pd.to_numeric(
        base[column],
        errors="coerce"
    )

    batch_values = pd.to_numeric(
        batch[column],
        errors="coerce"
    )

    reference = base_values.dropna()

    if reference.empty:
        return np.nan

    edges = np.unique(
        np.quantile(reference, np.linspace(0, 1, 11))
    )

    if len(edges) < 2:
        edges = np.array([-np.inf, np.inf])
    else:
        edges[0] = -np.inf
        edges[-1] = np.inf

    def assign_bins(values):
        result = []

        for value in values:
            if pd.isna(value):
                result.append("__MISSING__")
            else:
                bin_number = np.digitize(
                    value,
                    edges[1:-1],
                    right=False
                )
                result.append(f"bin_{bin_number}")

        return pd.Series(result, index=values.index)

    base_bins = assign_bins(base_values)
    batch_bins = assign_bins(batch_values)

    categories = sorted(
        set(base_bins.unique()) |
        set(batch_bins.unique())
    )

    base_dist = base_bins.value_counts(normalize=True)
    batch_dist = batch_bins.value_counts(normalize=True)

    psi = 0.0

    for category in categories:
        expected = max(base_dist.get(category, 0), EPSILON)
        actual = max(batch_dist.get(category, 0), EPSILON)

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


def calculate_batch_psi(base, batch, batch_name):
    results = []

    for column in CATEGORICAL_FEATURES:
        value = calculate_categorical_psi(
            base,
            batch,
            column
        )

        results.append({
            "Batch": batch_name,
            "Feature": column,
            "PSI": value,
            "Interpretation": interpret_psi(value),
        })

    for column in NUMERIC_FEATURES:
        value = calculate_numeric_psi(
            base,
            batch,
            column
        )

        results.append({
            "Batch": batch_name,
            "Feature": column,
            "PSI": value,
            "Interpretation": interpret_psi(value),
        })

    return results

def create_quality_plot(metrics_df):
    data = metrics_df.dropna(subset=["MAE"]).copy()

    if data.empty:
        print("Нет данных для построения графика MAE.")
        return

    plt.figure(figsize=(9, 5))

    plt.plot(
        data["Batch"],
        data["MAE"],
        marker="o"
    )

    plt.axhline(
        y=10,
        linestyle="--",
        label="Порог MAE = 10"
    )

    plt.xlabel("Входящая порция")
    plt.ylabel("MAE")
    plt.title("Изменение MAE модели HGB по порциям")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "mae_by_batch.png",
        dpi=150
    )

    plt.close()


def create_psi_heatmap(psi_df):
    table = psi_df.pivot(
        index="Feature",
        columns="Batch",
        values="PSI"
    )

    table = table.reindex(columns=sorted(table.columns))

    plt.figure(figsize=(10, 6))

    image = plt.imshow(
        table.values,
        aspect="auto"
    )

    plt.colorbar(
        image,
        label="PSI"
    )

    plt.xticks(
        range(len(table.columns)),
        table.columns
    )

    plt.yticks(
        range(len(table.index)),
        table.index
    )

    plt.xlabel("Входящая порция")
    plt.ylabel("Признак")
    plt.title("Изменение распределений признаков (PSI)")

    for i in range(len(table.index)):
        for j in range(len(table.columns)):
            value = table.iloc[i, j]

            if pd.notna(value):
                plt.text(
                    j,
                    i,
                    f"{value:.3f}",
                    ha="center",
                    va="center"
                )

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "psi_heatmap.png",
        dpi=150
    )

    plt.close()


def main():
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    base = pd.read_csv(BASE_PATH)

    print("Загрузка модели HGB...")
    model = joblib.load(MODEL_PATH)

    metrics_results = []
    psi_results = []

    batch_paths = sorted(
        INCOMING_DIR.glob("batch_*.csv")
    )

    if not batch_paths:
        raise FileNotFoundError(
            f"В папке {INCOMING_DIR} не найдены batch-файлы."
        )

    for path in batch_paths:
        batch_name = path.stem

        print("\n" + "=" * 60)
        print(f"Обработка {batch_name}")

        metrics = evaluate_batch(
            model,
            batch_name
        )

        metrics_results.append(metrics)

        print(
            f"Валидных записей: {metrics['ValidRecords']}"
        )
        print(
            f"Записей с известной оценкой: "
            f"{metrics['EvaluatedRecords']}"
        )

        if metrics["EvaluatedRecords"] > 0:
            print(f"MAE:  {metrics['MAE']:.4f}")
            print(f"RMSE: {metrics['RMSE']:.4f}")
            print(f"R²:   {metrics['R2']:.4f}")

        raw_batch = pd.read_csv(path)

        batch_psi = calculate_batch_psi(
            base,
            raw_batch,
            batch_name
        )

        psi_results.extend(batch_psi)


    metrics_df = pd.DataFrame(metrics_results)
    psi_df = pd.DataFrame(psi_results)

    metrics_df.to_csv(
        RESULTS_DIR / "batch_metrics.csv",
        index=False
    )

    psi_df.to_csv(
        RESULTS_DIR / "psi_results.csv",
        index=False
    )

    psi_pivot = psi_df.pivot(
        index="Feature",
        columns="Batch",
        values="PSI"
    )

    psi_pivot.to_csv(
        RESULTS_DIR / "psi_summary.csv"
    )

    create_quality_plot(metrics_df)
    create_psi_heatmap(psi_df)

    print("\n" + "=" * 60)
    print("МЕТРИКИ КАЧЕСТВА HGB")

    print(metrics_df.to_string(index=False))

    print("\n" + "=" * 60)
    print("Результаты PSI")

    print(psi_df.to_string(index=False))


if __name__ == "__main__":
    main()