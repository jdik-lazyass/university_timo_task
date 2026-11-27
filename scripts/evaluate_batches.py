from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.dummy import DummyRegressor
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score)

BASE_PATH = Path("../data/processed/base.csv")
INCOMING_DIR = Path("../data/incoming")
RESULTS_DIR = Path("../results")

RIDGE_PATH = Path("../models/best_model.joblib")
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

MIN_GROUP_SIZE = 30


def calculate_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "R2": r2_score(y_true, y_pred),
    }


def calculate_slice_metrics(df, slice_column, slice_name):
    results = []

    for group_name, group in df.groupby(slice_column, observed=True):
        n = len(group)

        if n < MIN_GROUP_SIZE:
            continue

        actual = group["Actual"]
        predicted = group["HGB"]

        results.append({
            "Slice": slice_name,
            "Group": group_name,
            "N": n,
            "MAE": mean_absolute_error(actual, predicted),
            "RMSE": mean_squared_error(actual, predicted) ** 0.5,
            "Bias": (predicted - actual).mean(),
        })

    return results


def create_slice_analysis(predictions):
    df = predictions.copy()

    for column in ["Genre", "Platform", "Rating"]:
        df[column] = (
            df[column]
            .astype("string")
            .fillna("Не указан")
        )

    df["Year_of_Release"] = pd.to_numeric(
        df["Year_of_Release"],
        errors="coerce"
    )

    df["YearPeriod"] = pd.cut(
        df["Year_of_Release"],
        bins=[2008, 2011, 2014, 2017, 2020, 2023],
        labels=[
            "2009–2011",
            "2012–2014",
            "2015–2017",
            "2018–2020",
            "2021–2023",
        ]
    )

    slices = [
        ("Genre", "Жанр"),
        ("Platform", "Платформа"),
        ("Rating", "Возрастной рейтинг"),
        ("YearPeriod", "Период выпуска"),
        ("Batch", "Тестовая порция"),
    ]

    all_results = []

    for column, slice_name in slices:
        all_results.extend(
            calculate_slice_metrics(df, column, slice_name)
        )

    result = pd.DataFrame(all_results)

    result = result.sort_values(
        ["Slice", "MAE"],
        ascending=[True, False]
    )

    return result


def create_slice_plots(slice_results):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for slice_name, filename, title in [
        ("Жанр", "mae_by_genre.png", "MAE по жанрам"),
        (
            "Период выпуска",
            "mae_by_year.png",
            "MAE по периодам выпуска"
        ),
        (
            "Тестовая порция",
            "mae_by_batch.png",
            "MAE по тестовым порциям"
        ),
    ]:
        data = slice_results[
            slice_results["Slice"] == slice_name
        ].sort_values("MAE")

        if data.empty:
            continue

        plt.figure(figsize=(10, 6))

        plt.barh(
            data["Group"].astype(str),
            data["MAE"]
        )

        plt.xlabel("MAE")
        plt.ylabel(slice_name)
        plt.title(title)
        plt.grid(axis="x", alpha=0.3)
        plt.tight_layout()

        plt.savefig(
            RESULTS_DIR / filename,
            dpi=150
        )

        plt.close()


def main():
    base = pd.read_csv(BASE_PATH)
    train = base.dropna(subset=[TARGET_COLUMN])

    ridge = joblib.load(RIDGE_PATH)
    hgb = joblib.load(HGB_PATH)

    dummy = DummyRegressor(strategy="mean")
    dummy.fit([[0]] * len(train), train[TARGET_COLUMN])

    all_predictions = []

    for path in sorted(INCOMING_DIR.glob("batch_*.csv")):
        batch = pd.read_csv(path)
        test = batch.dropna(subset=[TARGET_COLUMN]).copy()

        X_test = test[FEATURE_COLUMNS]
        y_test = test[TARGET_COLUMN]

        ridge_pred = ridge.predict(X_test)
        hgb_pred = hgb.predict(X_test)
        dummy_pred = dummy.predict([[0]] * len(test))

        result = pd.DataFrame({
            "Batch": path.stem,
            "Name": test["Name"].values,
            "Platform": test["Platform"].values,
            "Genre": test["Genre"].values,
            "Year_of_Release": test["Year_of_Release"].values,
            "Rating": test["Rating"].values,
            "Actual": y_test.values,
            "Ridge": ridge_pred,
            "HGB": hgb_pred,
            "Dummy": dummy_pred,
        })

        result["Group"] = result["Name"].astype("object")

        missing = result["Group"].isna()
        result.loc[missing, "Group"] = [
            f"missing_{path.stem}_{i}"
            for i in result.index[missing]
        ]

        all_predictions.append(result)

    predictions = pd.concat(
        all_predictions,
        ignore_index=True
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    predictions.to_csv(
        RESULTS_DIR / "final_test_predictions.csv",
        index=False
    )

    print("Количество наблюдений:", len(predictions))
    print("Количество уникальных игр:", predictions["Group"].nunique())

    for model_name in ["Ridge", "HGB", "Dummy"]:
        metrics = calculate_metrics(
            predictions["Actual"],
            predictions[model_name]
        )

        print(f"\n{model_name}")

        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

    slice_results = create_slice_analysis(predictions)

    slice_results.to_csv(
        RESULTS_DIR / "slice_analysis.csv",
        index=False
    )

    for slice_name in slice_results["Slice"].unique():
        print(f"\n{slice_name}")

        data = slice_results[
            slice_results["Slice"] == slice_name
        ]

        print(
            data[
                ["Group", "N", "MAE", "RMSE", "Bias"]
            ].to_string(index=False)
        )

    create_slice_plots(slice_results)


if __name__ == "__main__":
    main()
