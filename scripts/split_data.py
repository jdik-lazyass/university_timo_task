from pathlib import Path

import pandas as pd

RAW_PATH = Path("../data/raw/dataset.csv")
BASE_PATH = Path("../data/processed/base.csv")
INCOMING_DIR = Path("../data/incoming")

DATE_COLUMN = "Year_of_Release"
BASE_FRACTION = 0.6
N_BATCHES = 4 
RANDOM_STATE = 42 

"""sorting by year, where early games are initial data and latter - imitation of new ones."""
def chronological_split(df: pd.DataFrame):
    df_sorted = df.sort_values(DATE_COLUMN).reset_index(drop=True)
    base_size = int(len(df_sorted) * BASE_FRACTION)
    base = df_sorted.iloc[:base_size]
    remainder = df_sorted.iloc[base_size:]
    return base, remainder


def random_split(df: pd.DataFrame):
    base = df.sample(frac=BASE_FRACTION, random_state=RANDOM_STATE)
    remainder = df.drop(base.index)
    return base, remainder

def main():
    df = pd.read_csv(RAW_PATH)

    if DATE_COLUMN in df.columns:
        base, remainder = chronological_split(df)
    else:
        base, remainder = random_split(df)

    BASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(BASE_PATH, index=False)
    print(f"Исходная часть: {len(base)} строк  {BASE_PATH}")

    INCOMING_DIR.mkdir(parents=True, exist_ok=True)
    batch_boundaries = pd.cut(range(len(remainder)), bins=N_BATCHES, labels=False)
    for i in range(N_BATCHES):
        batch = remainder[batch_boundaries == i]
        batch_path = INCOMING_DIR / f"batch_{i + 1:02d}.csv"
        batch.to_csv(batch_path, index=False)
        print(f"Порция {i + 1}: {len(batch)} строк  {batch_path}")


if __name__ == "__main__":
    main()