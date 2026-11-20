import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from pipeline import create_pipeline
from sklearn.model_selection import RandomizedSearchCV
# init vals

RANDOM_STATE = 42
N_SPLITS = 5

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT_DIR / "data" / "processed" / "base.csv"
RESULTS_DIR = ROOT_DIR / "results"
MODELS_DIR = ROOT_DIR / "models"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLUMNS = [
    "Platform",
    "Year_of_Release",
    "Genre",
    "Publisher",
    "Developer",
    "Rating"
]

TARGET_COLUMN = "Critic_Score"
GROUP_COLUMN = "Name"
# data prepare and load

def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"файл с данными не найден: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    required_columns = (FEATURE_COLUMNS + [TARGET_COLUMN, GROUP_COLUMN])

    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns: raise ValueError(f"в датасете отсутствуют столбцы: {missing_columns}")

    # приведение целевой переменной к числовому типу
    df[TARGET_COLUMN] = pd.to_numeric(
        df[TARGET_COLUMN],
        errors="coerce"
    )

    # год выпуска также должен быть числовым
    df["Year_of_Release"] = pd.to_numeric(
        df["Year_of_Release"],
        errors="coerce"
    )

    # для обучения оставляем только строки с известной оценкой
    df = df.dropna(subset=[TARGET_COLUMN]).copy()

    # удаляем бесконечные значения целевой переменной
    df = df[np.isfinite(df[TARGET_COLUMN])].copy()

    # признаки
    X = df[FEATURE_COLUMNS].copy()

    # целевая переменная
    y = df[TARGET_COLUMN].copy()

    # группы для GroupKFold; если name отсутствует, каждой такой строке назначается отдельная группа, чтобы не объединять все пропуски.
    groups = df[GROUP_COLUMN].astype(object).copy()

    missing_name = groups.isna()
    groups.loc[missing_name] = [
        f"__missing_name_{i}"
        for i in df.index[missing_name]
    ]

    # сбрасываем индексы для согласованности X, y и groups
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    groups = groups.reset_index(drop=True)

    print(f"Всего строк в исходном файле: {len(pd.read_csv(DATA_PATH))}")
    print(f"Строк с известным Critic_Score: {len(X)}")
    print(f"Количество уникальных игр: {groups.nunique()}")

    return X, y, groups


# model defs
def get_models():
    return {
        "DummyRegressor": DummyRegressor(strategy="mean"),

        "Ridge": Ridge(alpha=1.0),

        "Lasso": Lasso(
            alpha=0.001,
            max_iter=10000,
            random_state=42
        ),

        "KNeighborsRegressor": KNeighborsRegressor(
            n_neighbors=5
        ),

        "DecisionTreeRegressor": DecisionTreeRegressor(
            random_state=42
        ),

        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),

        "HistGradientBoostingRegressor": HistGradientBoostingRegressor(
            max_iter=150,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=42
        )
    }


# cross val

def evaluate_models(X, y, groups):
    models = get_models()

    cv = GroupKFold(n_splits=N_SPLITS)

    fold_results = []
    summary_results = []

    for model_name, model in models.items():
        print(f"{model_name}")

        mae_scores = []
        rmse_scores = []
        r2_scores = []

        start_time = time.perf_counter()

        for fold, (train_idx, val_idx) in enumerate(
            cv.split(X, y, groups),
            start=1
        ):
            X_train = X.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_train = y.iloc[train_idx]
            y_val = y.iloc[val_idx]

            # создаём отдельный экземпляр модели для каждого фолда
            pipeline = create_pipeline(clone(model))

            # обучение
            pipeline.fit(X_train, y_train)

            # прогноз
            predictions = pipeline.predict(X_val)

            # метрики
            mae = mean_absolute_error(y_val,predictions)

            rmse = np.sqrt(mean_squared_error(y_val, predictions))

            r2 = r2_score(y_val, predictions)

            mae_scores.append(mae)
            rmse_scores.append(rmse)
            r2_scores.append(r2)

            fold_results.append({
                "model": model_name,
                "fold": fold,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2
            })

            print(
                f"  Фолд {fold}: "
                f"MAE={mae:.4f}, "
                f"RMSE={rmse:.4f}, "
                f"R2={r2:.4f}"
            )

        elapsed_time = time.perf_counter() - start_time

        # средние значения и стандартные отклонения
        result = {
            "model": model_name,
            "MAE_mean": np.mean(mae_scores),
            "MAE_std": np.std(mae_scores),
            "RMSE_mean": np.mean(rmse_scores),
            "RMSE_std": np.std(rmse_scores),
            "R2_mean": np.mean(r2_scores),
            "R2_std": np.std(r2_scores),
            "time_seconds": elapsed_time
        }

        summary_results.append(result)

        print(
            f"Итог {model_name}: "
            f"MAE={result['MAE_mean']:.4f} "
            f"+/- {result['MAE_std']:.4f}"
        )

    # таблицы с результатами
    fold_df = pd.DataFrame(fold_results)
    summary_df = pd.DataFrame(summary_results)

    # сортировка по основной метрике MAE
    summary_df = summary_df.sort_values(
        by="MAE_mean",
        ascending=True
    )
    # сохранение результатов
    fold_df.to_csv(
        RESULTS_DIR / "fold_metrics.csv",
        index=False
    )

    summary_df.to_csv(
        RESULTS_DIR / "experiments.csv",
        index=False
    )

    return summary_df


# learning and saveing best model

def evaluate_final_models(X, y, groups):

    models = {
        "DummyRegressor": DummyRegressor(strategy="mean"),

        "Ridge": Ridge(alpha=1.0),

        "Lasso": Lasso(
            alpha=0.001,
            max_iter=10000,
            random_state=42
        ),

        "RandomForestRegressor_tuned": RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=1,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1
        ),

        "HistGradientBoostingRegressor_tuned": HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.1,
            max_leaf_nodes=15,
            min_samples_leaf=10,
            l2_regularization=5.0,
            random_state=42
        ),

        "KNeighborsRegressor": KNeighborsRegressor(
            n_neighbors=5
        ),

        "DecisionTreeRegressor": DecisionTreeRegressor(
            random_state=42
        ),
    }

    cv = GroupKFold(n_splits=N_SPLITS)
    results = []

    for model_name, model in models.items():
        fold_metrics = []
        start_time = time.perf_counter()

        for train_idx, val_idx in cv.split(X, y, groups):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            pipeline = create_pipeline(clone(model))
            pipeline.fit(X_train, y_train)

            predictions = pipeline.predict(X_val)

            fold_metrics.append({
                "MAE": mean_absolute_error(y_val, predictions),
                "RMSE": np.sqrt(mean_squared_error(y_val, predictions)),
                "R2": r2_score(y_val, predictions)
            })

        elapsed_time = time.perf_counter() - start_time

        metrics_df = pd.DataFrame(fold_metrics)

        results.append({
            "Model": model_name,
            "MAE_mean": metrics_df["MAE"].mean(),
            "MAE_std": metrics_df["MAE"].std(),
            "RMSE_mean": metrics_df["RMSE"].mean(),
            "RMSE_std": metrics_df["RMSE"].std(),
            "R2_mean": metrics_df["R2"].mean(),
            "R2_std": metrics_df["R2"].std(),
            "time_seconds": elapsed_time
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("MAE_mean")

    print(results_df.to_string(index=False))

    results_df.to_csv(
        RESULTS_DIR / "final_model_comparison.csv",
        index=False
    )

    return results_df

def tune_ensemble_models(X, y, groups):
    cv = GroupKFold(n_splits=N_SPLITS)

    models_to_tune = {
        "RandomForestRegressor": {
            "model": RandomForestRegressor(
                random_state=42,
                n_jobs=-1
            ),
            "params": {
                "model__n_estimators": [100, 200, 300],
                "model__max_depth": [None, 10, 20, 30],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": [1.0, "sqrt", 0.5]
            },
            "n_iter": 15
        },

        "HistGradientBoostingRegressor": {
            "model": HistGradientBoostingRegressor(
                random_state=42
            ),
            "params": {
                "model__max_iter": [100, 150, 200, 300],
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__max_leaf_nodes": [15, 31, 63],
                "model__min_samples_leaf": [10, 20, 30],
                "model__l2_regularization": [0.0, 0.1, 1.0, 5.0]
            },
            "n_iter": 15
        }
    }

    all_results = []

    for model_name, config in models_to_tune.items():
        print(f"\nПодбор гиперпараметров: {model_name}")

        pipeline = create_pipeline(config["model"])

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=config["params"],
            n_iter=config["n_iter"],
            scoring="neg_mean_absolute_error",
            cv=cv,
            random_state=42,
            n_jobs=1,
            verbose=2,
            return_train_score=False
        )

        search.fit(X, y, groups=groups)

        best_mae = -search.best_score_

        print(f"Лучший MAE: {best_mae:.4f}")
        print(f"Лучшие параметры: {search.best_params_}")

        # сохраняем лучшую модель
        models_dir = Path("models")
        models_dir.mkdir(parents=True, exist_ok=True)

        model_filename = f"tuned_{model_name}.joblib"
        joblib.dump(
            search.best_estimator_,
            models_dir / model_filename
        )

        # сххраняем результаты всех проверенных комбинаций
        results = pd.DataFrame(search.cv_results_)
        results["model"] = model_name
        results["mean_test_MAE"] = -results["mean_test_score"]
        results["std_test_MAE"] = results["std_test_score"]

        all_results.append(
            results[
                [
                    "model",
                    "params",
                    "mean_test_MAE",
                    "std_test_MAE",
                    "rank_test_score"
                ]
            ]
        )

    tuning_results = pd.concat(all_results, ignore_index=True)

    tuning_results.to_csv(
        RESULTS_DIR /  "hyperparameter_search.csv",
        index=False
    )

def train_best_model(X, y, summary_df):
    best_model_name = summary_df.iloc[0]["model"]

    print(f"\nЛучшая модель по MAE: {best_model_name}")

    models = get_models()
    best_model = models[best_model_name]

    # создаём полный пайплайн
    best_pipeline = create_pipeline(
        clone(best_model)
    )

    # обучаем на всех доступных обучающих данных
    best_pipeline.fit(X, y)

    # сохраняем пайплайн
    model_path = MODELS_DIR / "best_model.joblib"

    joblib.dump(
        best_pipeline,
        model_path
    )

    # сохраняем информацию о модели
    model_info = {
        "model_name": best_model_name,
        "target": TARGET_COLUMN,
        "features": FEATURE_COLUMNS,
        "group_column": GROUP_COLUMN,
        "cv": "GroupKFold",
        "n_splits": N_SPLITS,
        "random_state": RANDOM_STATE,
        "selection_metric": "MAE",
        "MAE_mean": float(summary_df.iloc[0]["MAE_mean"]),
        "RMSE_mean": float(summary_df.iloc[0]["RMSE_mean"]),
        "R2_mean": float(summary_df.iloc[0]["R2_mean"])
    }

    with open(
        MODELS_DIR / "best_model_info.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            model_info,
            file,
            ensure_ascii=False,
            indent=4
        )

def train_final_model(X, y):
    model = Ridge(alpha=1.0)
    pipeline = create_pipeline(model)

    pipeline.fit(X, y)

    joblib.dump(pipeline, "../models/final_model.joblib")

    return pipeline

# main

def evaluate_regularization(X, y, groups):
    alphas = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

    cv = GroupKFold(n_splits=5)
    results = []

    models = {
        "Ridge": Ridge,
        "Lasso": Lasso
    }

    for model_name, model_class in models.items():
        for alpha in alphas:
            fold_mae = []
            fold_rmse = []
            fold_r2 = []
            fold_nonzero = []

            for train_idx, val_idx in cv.split(X, y, groups):
                X_train = X.iloc[train_idx]
                X_val = X.iloc[val_idx]

                y_train = y.iloc[train_idx]
                y_val = y.iloc[val_idx]

                model = model_class(
                    alpha=alpha,
                    max_iter=100000
                ) if model_name == "Lasso" else model_class(
                    alpha=alpha
                )

                pipeline = create_pipeline(model)
                pipeline.fit(X_train, y_train)

                predictions = pipeline.predict(X_val)

                fold_mae.append(
                    mean_absolute_error(y_val, predictions)
                )
                fold_rmse.append(
                    np.sqrt(mean_squared_error(y_val, predictions))
                )
                fold_r2.append(
                    r2_score(y_val, predictions)
                )

                coefficients = pipeline.named_steps[
                    "model"
                ].coef_

                fold_nonzero.append(
                    np.count_nonzero(np.abs(coefficients) > 1e-8)
                )

            results.append({
                "Model": model_name,
                "Alpha": alpha,
                "MAE_mean": np.mean(fold_mae),
                "MAE_std": np.std(fold_mae),
                "RMSE_mean": np.mean(fold_rmse),
                "R2_mean": np.mean(fold_r2),
                "Nonzero_coefficients": np.mean(fold_nonzero)
            })

    results_df = pd.DataFrame(results)

    output_path = RESULTS_DIR / "regularization.csv"

    results_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nРезультаты исследования регуляризации:")
    print(results_df.to_string(index=False))

    return results_df

def main():
    X, y, groups = load_data()

    if groups.nunique() < N_SPLITS: raise ValueError("Количество уникальных групп меньше числа фолдов.")

    summary_df = evaluate_models(X, y, groups)

    print("\nСводная таблица результатов:")
    print(summary_df.to_string(index=False,float_format=lambda value: f"{value:.4f}"))

    #regularization_results = evaluate_regularization(X, y, groups)
    #tune_ensemble_models(X, y, groups)
    #train_best_model(X, y, summary_df)
    evaluate_final_models(X, y, groups)
    #train_final_model(X, y)

if __name__ == "__main__":
    main()
