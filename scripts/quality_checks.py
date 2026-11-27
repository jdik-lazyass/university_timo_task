# для пункта 4.7
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

BASE_FILE = BASE_DIR / "data" / "processed" / "base.csv"
INCOMING_DIR = BASE_DIR / "data" / "incoming"

RESULTS_DIR = BASE_DIR / "results" / "quality_checks"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MIN_YEAR = 1980
MAX_YEAR = datetime.now().year

MAX_MISSING_SHARE = 0.60
MAX_REJECTION_SHARE = 0.05

VALID_RATINGS = {
    "E",
    "E10+",
    "T",
    "M",
    "AO",
    "EC",
    "K-A",
    "RP",
}

REQUIRED_COLUMNS = [
    "Name",
    "Platform",
    "Year_of_Release",
    "Genre",
    "Publisher",
    "Developer",
    "Rating",
    "Critic_Score",
]

def load_reference_data():

    if not BASE_FILE.exists():
        raise FileNotFoundError(
            f"Не найден файл: {BASE_FILE}"
        )

    data_files = [BASE_FILE]
    data_files.extend(sorted(INCOMING_DIR.glob("batch_*.csv")))

    if len(data_files) == 1:
        print(
            "Предупреждение: входящие порции не найдены. "
            "Справочники будут построены только по base.csv."
        )

    frames = []

    for file in data_files:
        print(f"Загрузка справочника: {file.name}")
        frames.append(pd.read_csv(file))

    all_data = pd.concat(frames, ignore_index=True)

    valid_platforms = set(
        all_data["Platform"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
    )

    valid_genres = set(
        all_data["Genre"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
    )

    print("\nСправочники загружены:")
    print(f"Допустимых платформ: {len(valid_platforms)}")
    print(f"Допустимых жанров: {len(valid_genres)}")
    print(f"Всего строк для справочников: {len(all_data)}")

    return valid_platforms, valid_genres


VALID_PLATFORMS, VALID_GENRES = load_reference_data()

def is_empty(series):
    return (
        series.isna()
        | series.astype("string").str.strip().eq("").fillna(False)
    )


def add_check(report, name, violations, blocking, message):

    report["checks"][name] = {
        "violations": int(violations),
        "blocking": bool(blocking),
        "message": message,
    }

def validate_batch(df):

    df = df.copy().reset_index(drop=True)

    report = {
        "total": len(df),
        "checks": {},
        "warnings": [],
    }

    rejected = pd.Series(False, index=df.index)

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    add_check(
        report,
        "Q1. Наличие обязательных столбцов",
        len(missing_columns),
        True,
        (
            "Отсутствуют столбцы: "
            + ", ".join(missing_columns)
            if missing_columns
            else "Все обязательные столбцы присутствуют"
        ),
    )

    if missing_columns:
        report["valid"] = 0
        report["rejected"] = len(df)
        report["rejection_share"] = 1.0
        report["status"] = "ERROR"
        report["warnings_count"] = 0

        return None, report

    invalid_name = is_empty(df["Name"])

    rejected |= invalid_name

    add_check(
        report,
        "Q2. Заполнение Name",
        invalid_name.sum(),
        True,
        "Пустые названия игр",
    )

    platform = df["Platform"].astype("string").str.strip()

    invalid_platform = (
        platform.isna()
        | platform.eq("")
        | ~platform.isin(VALID_PLATFORMS)
    ).fillna(True)

    rejected |= invalid_platform

    add_check(
        report,
        "Q3. Допустимость Platform",
        invalid_platform.sum(),
        True,
        "Пустые или неизвестные платформы",
    )

    year = pd.to_numeric(
        df["Year_of_Release"],
        errors="coerce"
    )

    invalid_year = (
        year.isna()
        | (year < MIN_YEAR)
        | (year > MAX_YEAR)
    )

    rejected |= invalid_year

    add_check(
        report,
        "Q4. Диапазон Year_of_Release",
        invalid_year.sum(),
        True,
        f"Год должен находиться в диапазоне {MIN_YEAR}–{MAX_YEAR}",
    )

    genre = df["Genre"].astype("string").str.strip()

    invalid_genre = (
        genre.isna()
        | genre.eq("")
        | ~genre.isin(VALID_GENRES)
    ).fillna(True)

    rejected |= invalid_genre

    add_check(
        report,
        "Q5. Допустимость Genre",
        invalid_genre.sum(),
        True,
        "Пустые или неизвестные жанры",
    )

    rating = df["Rating"].astype("string").str.strip()

    invalid_rating = (
        rating.notna()
        & rating.ne("")
        & ~rating.isin(VALID_RATINGS)
    ).fillna(False)

    rating_violations = invalid_rating.sum()

    df.loc[invalid_rating, "Rating"] = "Unknown"
    df.loc[is_empty(df["Rating"]), "Rating"] = "Unknown"

    add_check(
        report,
        "Q6. Допустимость Rating",
        rating_violations,
        False,
        "Некорректные значения заменены на Unknown",
    )

    if rating_violations:
        report["warnings"].append(
            "Обнаружены некорректные значения Rating"
        )

    original_score = df["Critic_Score"]

    score = pd.to_numeric(
        original_score,
        errors="coerce"
    )

    invalid_score_range = (
        score.notna()
        & ((score <= 0) | (score >= 100))
    )

    invalid_score_text = (
        original_score.notna()
        & original_score.astype("string").str.strip().ne("")
        & score.isna()
    )

    score_violations = (
        invalid_score_range | invalid_score_text
    ).sum()


    df.loc[invalid_score_range | invalid_score_text, "Critic_Score"] = np.nan

    df["Critic_Score"] = pd.to_numeric(
        df["Critic_Score"],
        errors="coerce"
    )

    add_check(
        report,
        "Q7. Диапазон Critic_Score",
        score_violations,
        False,
        "Некорректные оценки заменены на пропуски",
    )

    if score_violations:
        report["warnings"].append(
            "Обнаружены некорректные значения Critic_Score"
        )

    for column in ["Publisher", "Developer"]:

        missing_share = df[column].isna().mean()

        violation = missing_share > MAX_MISSING_SHARE

        add_check(
            report,
            f"Q8. Пропуски {column}",
            int(violation),
            False,
            (
                f"Доля пропусков: {missing_share:.1%}; "
                f"допустимо не более {MAX_MISSING_SHARE:.0%}"
            ),
        )

        if violation:
            report["warnings"].append(
                f"Высокая доля пропусков в {column}"
            )

    rejected_count = int(rejected.sum())

    cleaned = df.loc[~rejected].copy().reset_index(drop=True)

    report["valid"] = len(cleaned)
    report["rejected"] = rejected_count

    report["rejection_share"] = (
        rejected_count / len(df)
        if len(df) > 0
        else 0
    )

    if report["rejection_share"] > MAX_REJECTION_SHARE:
        report["warnings"].append(
            "Доля отбракованных записей превышает 5%"
        )


    has_blocking_violations = any(
        item["blocking"] and item["violations"] > 0
        for item in report["checks"].values()
    )

    if has_blocking_violations:
        report["status"] = "ERROR"
    elif report["warnings"] or rejected_count > 0:
        report["status"] = "WARNING"
    else:
        report["status"] = "OK"

    report["warnings_count"] = len(report["warnings"])

    return cleaned, report


def create_corrupted_file(source_file):
    """
    Создает копию первой порции с намеренными нарушениями.
    Исходный batch_01.csv не изменяется.
    """

    df = pd.read_csv(source_file)

    if len(df) < 10:
        raise ValueError(
            "Недостаточно строк для создания испорченного файла"
        )

    # Q2: пустое название
    df.loc[0, "Name"] = None

    # Q3: неизвестная платформа
    df.loc[1, "Platform"] = "UNKNOWN_PLATFORM"

    # Q4: некорректный год
    df.loc[2, "Year_of_Release"] = 1700

    # Q5: неизвестный жанр
    df.loc[3, "Genre"] = "UNKNOWN_GENRE"

    # Q6: некорректный рейтинг
    df.loc[4, "Rating"] = "INVALID_RATING"

    # Q7: оценка вне диапазона
    df.loc[5, "Critic_Score"] = 150

    # Q8: искусственно увеличиваем долю пропусков
    publisher_count = int(len(df) * 0.65)
    developer_count = int(len(df) * 0.65)

    df.loc[:publisher_count - 1, "Publisher"] = None
    df.loc[:developer_count - 1, "Developer"] = None

    output_file = RESULTS_DIR / "corrupted_batch.csv"

    df.to_csv(output_file, index=False)

    print(
        f"\nИспорченный файл создан: {output_file}"
    )

    return output_file

def process_file(file_path):
    df = pd.read_csv(file_path)

    cleaned, report = validate_batch(df)

    file_name = file_path.stem

    print(f"\n{file_path.name}")
    print(f"Всего записей: {report['total']}")
    print(f"Валидных записей: {report['valid']}")
    print(f"Исключено записей: {report['rejected']}")
    print(f"Предупреждений: {report['warnings_count']}")
    print(f"Статус: {report['status']}")

    for name, result in report["checks"].items():
        if result["violations"] > 0:
            print(
                f"  {name}: {result['violations']} — "
                f"{result['message']}"
            )

    for warning in report["warnings"]:
        print(f"  Предупреждение: {warning}")

    if cleaned is not None:
        cleaned_file = (
            RESULTS_DIR / f"{file_name}_cleaned.csv"
        )
        cleaned.to_csv(cleaned_file, index=False)

    report_file = (
        RESULTS_DIR / f"{file_name}_quality.json"
    )

    with open(report_file, "w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=4
        )

    return {
        "Batch": file_path.name,
        "Total": report["total"],
        "Valid": report["valid"],
        "Rejected": report["rejected"],
        "Warnings": report["warnings_count"],
        "Status": report["status"],
    }

def main():

    print("=" * 60)
    print("ПРОВЕРКА КАЧЕСТВА ДАННЫХ")
    print("=" * 60)

    batch_files = sorted(
        INCOMING_DIR.glob("batch_*.csv")
    )

    if not batch_files:
        raise FileNotFoundError(
            f"В папке {INCOMING_DIR} не найдены batch_*.csv"
        )

    corrupted_file = create_corrupted_file(
        batch_files[0]
    )

    process_file(corrupted_file)

    results = []

    for batch_file in batch_files:
        result = process_file(batch_file)
        results.append(result)

    summary = pd.DataFrame(results)

    summary_file = RESULTS_DIR / "quality_summary.csv"

    summary.to_csv(
        summary_file,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 60)
    print("сводные результаты")
    print("=" * 60)

    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
