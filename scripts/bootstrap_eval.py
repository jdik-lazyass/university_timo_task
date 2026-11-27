from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = Path("../results/final_test_predictions.csv")
RESULTS_DIR = Path("../results")

N_BOOTSTRAP = 2000
RANDOM_STATE = 42

MODELS = ["Ridge", "HGB", "Dummy"]


def calculate_metrics(data, model):
    y_true = data["Actual"]
    y_pred = data[model]

    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "R2": r2_score(y_true, y_pred),
    }

def main():
    data = pd.read_csv(DATA_PATH)

    rng = np.random.default_rng(RANDOM_STATE)

    groups = {
        name: group
        for name, group in data.groupby("Group")
    }

    group_names = list(groups.keys())

    bootstrap_results = {
        model: {"MAE": [], "RMSE": [], "R2": []}
        for model in MODELS
    }

    mae_differences = []

    for _ in range(N_BOOTSTRAP):
        sampled_names = rng.choice(
            group_names,
            size=len(group_names),
            replace=True
        )

        sample = pd.concat(
            [groups[name] for name in sampled_names],
            ignore_index=True
        )

        for model in MODELS:
            metrics = calculate_metrics(sample, model)

            for metric, value in metrics.items():
                bootstrap_results[model][metric].append(value)

        ridge_mae = calculate_metrics(sample, "Ridge")["MAE"]
        hgb_mae = calculate_metrics(sample, "HGB")["MAE"]

        mae_differences.append(ridge_mae - hgb_mae)

    rows = []

    for model in MODELS:
        original = calculate_metrics(data, model)

        for metric in ["MAE", "RMSE", "R2"]:
            values = bootstrap_results[model][metric]

            lower = np.percentile(values, 2.5)
            upper = np.percentile(values, 97.5)

            rows.append({
                "Model": model,
                "Metric": metric,
                "Value": original[metric],
                "CI_95_lower": lower,
                "CI_95_upper": upper,
            })

    results = pd.DataFrame(rows)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(
        RESULTS_DIR / "bootstrap_confidence_intervals.csv",
        index=False
    )

    diff_lower = np.percentile(mae_differences, 2.5)
    diff_upper = np.percentile(mae_differences, 97.5)
    diff_mean = np.mean(mae_differences)

    print("95% доверительные интервалы")
    print(results.to_string(index=False))

    print(f"Средняя разница MAE (Ridge - HGB): {diff_mean:.4f}")
    print(f"95% CI: [{diff_lower:.4f}; {diff_upper:.4f}]")

    if diff_lower > 0 or diff_upper < 0:
        print("Различие статистически значимо на уровне 5%")
    else:
        print("Статистически значимое различие не обнаружено")


if __name__ == "__main__":
    main()
