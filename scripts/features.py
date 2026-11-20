import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, min_publisher_count=5, min_developer_count=5):
        self.min_publisher_count = min_publisher_count
        self.min_developer_count = min_developer_count

    def fit(self, X, y):
        X = X.copy()
        y = pd.Series(y, index=X.index)

        publishers = X["Publisher"].fillna("Unknown").astype(str)
        developers = X["Developer"].fillna("Unknown").astype(str)

        # издатели и разработчики встречающиеся реже заданного порога
        self.rare_publishers_ = set(
            publishers.value_counts()[
                publishers.value_counts() < self.min_publisher_count
            ].index
        )

        self.rare_developers_ = set(
            developers.value_counts()[
                developers.value_counts() < self.min_developer_count
            ].index
        )

        # средняя оценка по обучающей выборке для неизвестных издателей
        self.global_score_ = float(y.mean())

        # средние оценки издателей для последующего predict
        self.publisher_scores_ = (
            pd.DataFrame({
                "Publisher": publishers,
                "Score": y
            })
            .groupby("Publisher")["Score"]
            .mean()
            .to_dict()
        )

        return self

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self._transform(X, y, training=True)

    def transform(self, X):
        return self._transform(X, training=False)

    def _transform(self, X, y=None, training=False):
        X = X.copy()

        publishers = X["Publisher"].fillna("Unknown").astype(str)
        developers = X["Developer"].fillna("Unknown").astype(str)

        # редкие издатели
        X["is_rare_publisher"] = publishers.isin(
            self.rare_publishers_
        ).astype(int)

        # историческая средняя оценка издателя
        if training:
            y = pd.Series(y, index=X.index)

            history = pd.DataFrame({
                "Publisher": publishers,
                "Year": pd.to_numeric(
                    X["Year_of_Release"], errors="coerce"
                ),
                "Score": y
            })

            # сортировка по дате выхода
            history["_original_index"] = history.index
            history = history.sort_values(
                "Year", kind="stable", na_position="last"
            )

            # для каждой строки учитываем только предыдущие игры
            grouped = history.groupby("Publisher")["Score"]

            history["publisher_avg_score"] = (
                grouped.transform(
                    lambda s: s.shift(1).expanding().mean()
                )
            )

            # для первого наблюдения издателя используем
            # среднее по предыдущим наблюдениям в датасете
            history["global_history"] = (
                history["Score"].shift(1).expanding().mean()
            )

            history["publisher_avg_score"] = (
                history["publisher_avg_score"].fillna(
                    history["global_history"]
                ).fillna(self.global_score_)
            )

            X["publisher_avg_score"] = (
                history["publisher_avg_score"]
                .reindex(X.index)
            )

        else:
            X["publisher_avg_score"] = (
                publishers.map(self.publisher_scores_)
                .fillna(self.global_score_)
            )

        # группировка платформ по семействам
        platform_groups = {
            "PS": "PlayStation",
            "PS2": "PlayStation",
            "PS3": "PlayStation",
            "PS4": "PlayStation",
            "PSP": "PlayStation",
            "PSV": "PlayStation",

            "XB": "Xbox",
            "X360": "Xbox",
            "XOne": "Xbox",

            "NES": "Nintendo",
            "SNES": "Nintendo",
            "N64": "Nintendo",
            "GC": "Nintendo",
            "Wii": "Nintendo",
            "WiiU": "Nintendo",
            "GB": "Nintendo",
            "GBA": "Nintendo",
            "DS": "Nintendo",
            "3DS": "Nintendo",

            "GEN": "Sega",
            "GG": "Sega",
            "SCD": "Sega",
            "SAT": "Sega",
            "DC": "Sega",

            "PC": "PC"
        }

        X["platform_generation"] = (
            X["Platform"]
            .fillna("Unknown")
            .map(platform_groups)
            .fillna("Other")
        )

        # комбинация жанра и возрастного рейтинга
        genre = X["Genre"].fillna("Unknown").astype(str)
        rating = X["Rating"].fillna("Unknown").astype(str)

        X["genre_rating_combo"] = genre + "_" + rating

        # объединяем редких издателей и разработчиков в Other
        X["Publisher"] = publishers.where(
            ~publishers.isin(self.rare_publishers_),
            "Other"
        )

        X["Developer"] = developers.where(
            ~developers.isin(self.rare_developers_),
            "Other"
        )

        # пропуски в категориальных признаках
        categorical_columns = [
            "Platform", "Genre", "Publisher",
            "Developer", "Rating"
        ]

        for column in categorical_columns:
            X[column] = X[column].fillna("Unknown").astype(str)

        return X
