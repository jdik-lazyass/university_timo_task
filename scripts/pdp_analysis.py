from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt

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

PDP_FEATURES = [
    "Publisher",
    "Genre",
    "Rating",
    "Developer",
]

TOP_CATEGORIES = 10
MIN_CATEGORY_SIZE = 30


def load_test_data():
    batches = []

    for path in sorted(INCOMING_DIR.glob("batch_*.csv")):
        batch = pd.read_csv(path)
        batch = batch.dropna(subset=[TARGET_COLUMN])
        batches.append(batch)

    if not batches:
        raise ValueError("Тестовые порции не найдены")

    test = pd.concat(batches, ignore_index=True)

    return test[FEATURE_COLUMNS]


def calculate_pdp(model, X, feature):
    counts = X[feature].value_counts()

    categories = counts[
        counts >= MIN_CATEGORY_SIZE
    ].index.tolist()

    if feature in ["Publisher", "Developer"]:
        categories = categories[:TOP_CATEGORIES]

    results = []

    print(f"\nАнализ признака: {feature}")

    for category in categories:
        X_modified = X.copy()
        X_modified[feature] = category

        predictions = model.predict(X_modified)

        results.append({
            "Feature": feature,
            "Category": category,
            "Mean_Prediction": predictions.mean(),
            "Count": int(counts[category])
        })

    return pd.DataFrame(results)


def create_plot(data, feature):
    data = data.sort_values(
        "Mean_Prediction",
        ascending=True
    )

    plt.figure(figsize=(11, 6))

    plt.barh(
        data["Category"].astype(str),
        data["Mean_Prediction"]
    )

    plt.xlabel("Средний прогноз Critic_Score")
    plt.ylabel(feature)
    plt.title(f"Partial Dependence Plot: {feature}")

    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / f"pdp_{feature.lower()}.png",
        dpi=150
    )

    plt.close()


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model = joblib.load(HGB_PATH)
    X_test = load_test_data()

    all_results = []

    for feature in PDP_FEATURES:
        result = calculate_pdp(
            model,
            X_test,
            feature
        )

        if result.empty:
            print(f"Нет подходящих категорий для {feature}")
            continue

        print(result.to_string(index=False))

        create_plot(result, feature)
        all_results.append(result)

    if all_results:
        combined = pd.concat(
            all_results,
            ignore_index=True
        )

        combined.to_csv(
            RESULTS_DIR / "pdp_analysis.csv",
            index=False
        )


if __name__ == "__main__":
    main()
