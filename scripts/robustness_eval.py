# для пункта 4.6
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

HGB_PATH = Path("../models/tuned_HistGradientBoostingRegressor.joblib")
INCOMING_DIR = Path("../data/incoming")
RESULTS_DIR = Path("../results")

FEATURE_COLUMNS = [
    "Platform",
    "Year_of_Release",
    "Genre",
    "Publisher",
    "Developer",
    "Rating",
]

TARGET_COLUMN = "Critic_Score"

RANDOM_STATE = 42
PERTURBATION_LEVELS = [0.05, 0.10]


def calculate_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "R2": r2_score(y_true, y_pred),
    }


def load_test_data():
    batches = []

    for path in sorted(INCOMING_DIR.glob("batch_*.csv")):
        batch = pd.read_csv(path)

        batch = batch.dropna(subset=[TARGET_COLUMN]).copy()

        batch["Batch"] = path.stem

        batches.append(batch)

    if not batches:
        raise FileNotFoundError(
            "Не найдены тестовые порции batch_*.csv"
        )

    return pd.concat(batches, ignore_index=True)


def introduce_noise(X, percentage, rng):

    X_noisy = X.copy()

    n_rows, n_columns = X.shape
    total_cells = n_rows * n_columns

    n_changed = max(1, round(total_cells * percentage))

    selected_cells = rng.choice(
        total_cells,
        size=min(n_changed, total_cells),
        replace=False
    )

    for cell in selected_cells:
        row_index = cell // n_columns
        column_index = cell % n_columns

        column = FEATURE_COLUMNS[column_index]

        if column == "Year_of_Release":
            value = pd.to_numeric(
                pd.Series([X_noisy.iloc[row_index][column]]),
                errors="coerce"
            ).iloc[0]

            if pd.notna(value):
                noise = rng.normal(0, 1)
                X_noisy.iat[
                    row_index,
                    column_index
                ] = int(round(value + noise))

        else:
            categories = (
                X_noisy[column]
                .dropna()
                .unique()
                .tolist()
            )

            current_value = X_noisy.iloc[row_index][column]

            alternatives = [
                value for value in categories
                if value != current_value
            ]

            if alternatives:
                new_value = rng.choice(alternatives)

                X_noisy.iat[
                    row_index,
                    column_index
                ] = new_value

    return X_noisy


def introduce_missing_values(X, percentage, rng):

    X_missing = X.copy()

    total_cells = X.shape[0] * X.shape[1]

    n_missing = max(
        1,
        round(total_cells * percentage)
    )

    selected_cells = rng.choice(
        total_cells,
        size=min(n_missing, total_cells),
        replace=False
    )

    n_columns = X.shape[1]

    for cell in selected_cells:
        row_index = cell // n_columns
        column_index = cell % n_columns

        X_missing.iat[
            row_index,
            column_index
        ] = np.nan

    return X_missing


def evaluate_scenario(
    model,
    X,
    y,
    baseline_predictions,
    scenario_name,
    batch_names
):
    predictions = model.predict(X)

    metrics = calculate_metrics(y, predictions)

    result = {
        "Scenario": scenario_name,
        "N": len(y),
        **metrics,
        "MAE_change": (
            metrics["MAE"] - mean_absolute_error(
                y,
                baseline_predictions
            )
        ),
        "Prediction_shift": np.mean(
            np.abs(predictions - baseline_predictions)
        ),
    }

    print(f"\n{scenario_name}")
    print(f"MAE: {metrics['MAE']:.4f}")
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"R2: {metrics['R2']:.4f}")
    print(
        f"Изменение MAE: {result['MAE_change']:+.4f}"
    )
    print(
        f"Изменение прогнозов: "
        f"{result['Prediction_shift']:.4f}"
    )

    results_by_batch = []

    for batch_name in sorted(batch_names.unique()):
        mask = batch_names == batch_name

        batch_metrics = calculate_metrics(
            y[mask],
            predictions[mask]
        )

        results_by_batch.append({
            "Scenario": scenario_name,
            "Batch": batch_name,
            "N": mask.sum(),
            **batch_metrics,
        })

    return result, results_by_batch


def create_plots(results):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))

    plt.bar(
        results["Scenario"],
        results["MAE"]
    )

    plt.axhline(
        y=10.1648,
        linestyle="--",
        label="Исходная MAE HGB"
    )

    plt.ylabel("MAE")
    plt.xlabel("Сценарий")
    plt.title("Устойчивость HGB к шуму и пропускам")
    plt.xticks(rotation=20)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "robustness_mae.png",
        dpi=150
    )

    plt.close()

    plt.figure(figsize=(10, 6))

    plt.bar(
        results["Scenario"],
        results["Prediction_shift"]
    )

    plt.ylabel("Среднее абсолютное изменение прогноза")
    plt.xlabel("Сценарий")
    plt.title("Чувствительность прогнозов HGB")
    plt.xticks(rotation=20)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "robustness_prediction_shift.png",
        dpi=150
    )

    plt.close()


def main():
    rng = np.random.default_rng(RANDOM_STATE)

    model = joblib.load(HGB_PATH)

    test = load_test_data()

    X = test[FEATURE_COLUMNS].copy()
    y = test[TARGET_COLUMN].to_numpy()
    batch_names = test["Batch"]

    print("Количество наблюдений:", len(test))

    # Исходные прогнозы
    baseline_predictions = model.predict(X)

    baseline_metrics = calculate_metrics(
        y,
        baseline_predictions
    )

    for name, value in baseline_metrics.items():
        print(f"{name}: {value:.4f}")

    results = []
    batch_results = []

    for percentage in PERTURBATION_LEVELS:
        X_noisy = introduce_noise(
            X,
            percentage,
            rng
        )

        scenario = f"Шум {int(percentage * 100)}%"

        result, batch_result = evaluate_scenario(
            model,
            X_noisy,
            y,
            baseline_predictions,
            scenario,
            batch_names
        )

        results.append(result)
        batch_results.extend(batch_result)

    for percentage in PERTURBATION_LEVELS:
        X_missing = introduce_missing_values(
            X,
            percentage,
            rng
        )

        scenario = f"Пропуски {int(percentage * 100)}%"

        try:
            result, batch_result = evaluate_scenario(
                model,
                X_missing,
                y,
                baseline_predictions,
                scenario,
                batch_names
            )

            results.append(result)
            batch_results.extend(batch_result)

        except Exception as error:
            print(
                f"\nОшибка при сценарии {scenario}: "
                f"{error}"
            )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        RESULTS_DIR / "robustness_results.csv",
        index=False
    )

    pd.DataFrame(batch_results).to_csv(
        RESULTS_DIR / "robustness_by_batch.csv",
        index=False
    )

    create_plots(results_df)

    print(results_df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
