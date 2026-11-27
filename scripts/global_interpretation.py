from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error


BASE_PATH = Path("../data/processed/base.csv")
INCOMING_DIR = Path("../data/incoming")
RESULTS_DIR = Path("../results")

HGB_PATH = Path("../models/tuned_HistGradientBoostingRegressor.joblib")

FEATURE_COLUMNS = [
    "Platform",
    "Year_of_Release",
    "Genre",
    "Publisher",
    "Developer",
    "Rating",
]

TARGET_COLUMN = "Critic_Score"


def load_test_data():
    batches = []

    for path in sorted(INCOMING_DIR.glob("batch_*.csv")):
        batch = pd.read_csv(path)
        batch = batch.dropna(subset=[TARGET_COLUMN])

        batches.append(batch)

    if not batches:
        raise ValueError("Тестовые порции не найдены")

    test = pd.concat(batches, ignore_index=True)

    X_test = test[FEATURE_COLUMNS]
    y_test = test[TARGET_COLUMN]

    return X_test, y_test


def calculate_permutation_importance(model, X_test, y_test):

    baseline_mae = mean_absolute_error(
        y_test,
        model.predict(X_test)
    )

    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="neg_mean_absolute_error",
        n_repeats=30,
        random_state=42,
        n_jobs=-1
    )

    importance = pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "Importance_Mean": result.importances_mean,
        "Importance_STD": result.importances_std,
    })

    importance = importance.sort_values(
        "Importance_Mean",
        ascending=False
    )

    print(f"\nИсходная MAE: {baseline_mae:.4f}")

    print("\nПерестановочная важность:")
    print(importance.to_string(index=False))

    return importance


def create_importance_plot(importance):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    data = importance.sort_values("Importance_Mean")

    plt.figure(figsize=(10, 6))

    plt.barh(
        data["Feature"],
        data["Importance_Mean"],
        xerr=data["Importance_STD"],
        capsize=4
    )

    plt.xlabel("Увеличение MAE после перемешивания признака")
    plt.ylabel("Признак")
    plt.title("Перестановочная важность признаков HGB")
    plt.grid(axis="x", alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "permutation_importance.png",
        dpi=150
    )

    plt.close()


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model = joblib.load(HGB_PATH)

    X_test, y_test = load_test_data()

    print("Количество наблюдений:", len(X_test))

    importance = calculate_permutation_importance(
        model,
        X_test,
        y_test
    )

    importance.to_csv(
        RESULTS_DIR / "permutation_importance.csv",
        index=False
    )

    create_importance_plot(importance)


if __name__ == "__main__":
    main()
