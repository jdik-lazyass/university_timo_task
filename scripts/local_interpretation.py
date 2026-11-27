import os
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

FEATURE_COLUMNS = [
    "Platform",
    "Year_of_Release",
    "Genre",
    "Publisher",
    "Developer",
    "Rating"
]

SELECTED_GAMES = [
    "Ride to Hell",
    "Fighter Within",
    "Borderlands",
    "Assassin's Creed III: Liberation"
]

SELECTED_OBSERVATIONS = {
    "Ride to Hell": ("batch_03", "PS3", 13.0),
    "Fighter Within": ("batch_03", "XOne", 23.0),
    "Borderlands": ("batch_01", "X360", 84.0),
    "Assassin's Creed III: Liberation": ("batch_03", "PSV", 70.0)
}

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "incoming"
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_DIR = RESULTS_DIR / "shap"

MODEL_PATH = BASE_DIR / "models" / "tuned_HistGradientBoostingRegressor.joblib"
PREDICTIONS_PATH = RESULTS_DIR / "final_test_predictions.csv"


def load_original_data():
    frames = []

    for file_path in sorted(DATA_DIR.glob("batch_*.csv")):
        batch = pd.read_csv(file_path)
        frames.append(batch)

    if not frames:
        raise FileNotFoundError(
            f"В папке {DATA_DIR} не найдены файлы batch_*.csv"
        )

    return pd.concat(frames, ignore_index=True)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = joblib.load(MODEL_PATH)

    predictions = pd.read_csv(PREDICTIONS_PATH)
    original_data = load_original_data()
    selected_rows = []

    for game_name in SELECTED_GAMES:
        batch, platform, actual = SELECTED_OBSERVATIONS[game_name]

        game_predictions = predictions[
            (predictions["Name"] == game_name) &
            (predictions["Batch"] == batch) &
            (predictions["Platform"] == platform) &
            (predictions["Actual"] == actual)
        ]

        if game_predictions.empty:
            raise ValueError(
                f"Не найдена нужная запись: {game_name}, "
                f"{batch}, {platform}, {actual}"
            )

        prediction = game_predictions.iloc[0]

        batch_path = DATA_DIR / f"{batch}.csv"
        batch_data = pd.read_csv(batch_path)

        mask = (
            (batch_data["Name"] == game_name) &
            (batch_data["Platform"] == platform) &
            (batch_data["Critic_Score"] == actual)
        )

        matches = batch_data.loc[mask]

        if matches.empty:
            raise ValueError(
                f"Не найдена исходная запись: {game_name}"
            )

        selected_rows.append(matches.iloc[[0]])

    selected = pd.concat(selected_rows, ignore_index=True)
    X_selected = selected[FEATURE_COLUMNS]


    background = original_data[FEATURE_COLUMNS].sample(
        n=min(50, len(original_data)),
        random_state=42
    )

    def predict_fn(data):
        if isinstance(data, pd.DataFrame):
            X = data[FEATURE_COLUMNS]
        else:
            X = pd.DataFrame(data, columns=FEATURE_COLUMNS)

        return model.predict(X)

    explainer = shap.KernelExplainer(
        predict_fn,
        background
    )

    shap_values = explainer.shap_values(
        X_selected,
        nsamples=100
    )

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_values = np.asarray(shap_values)

    base_value = np.asarray(explainer.expected_value).reshape(-1)[0]
    shap_table = pd.DataFrame(
        shap_values,
        columns=FEATURE_COLUMNS
    )

    shap_table.insert(0, "Name", selected["Name"].values)
    shap_table.insert(1, "Actual", selected["Critic_Score"].values)

    shap_table.to_csv(
        OUTPUT_DIR / "local_shap_values.csv",
        index=False
    )
    for i, game_name in enumerate(selected["Name"]):
        explanation = shap.Explanation(
            values=shap_values[i],
            base_values=base_value,
            data=X_selected.iloc[i].values,
            feature_names=FEATURE_COLUMNS
        )

        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(
            explanation,
            show=False
        )

        plt.title(game_name)
        plt.tight_layout()

        safe_name = "".join(
            c if c.isalnum() else "_"
            for c in game_name
        )

        plt.savefig(
            OUTPUT_DIR / f"{safe_name}_shap.png",
            dpi=150,
            bbox_inches="tight"
        )

        plt.close()


if __name__ == "__main__":
    main()
