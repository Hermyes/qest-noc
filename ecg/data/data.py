import pandas as pd
from sklearn.model_selection import train_test_split

EXPECTED = [
    "Healthy","rr_interval","p_onset","p_end","qrs_onset","qrs_end",
    "t_end","p_axis","qrs_axis","t_axis"
]

def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [c for c in EXPECTED if c not in df.columns]
    if missing:
        raise ValueError(f"В CSV нет колонок: {missing}")
    return df[EXPECTED].copy()

def make_test_split(df: pd.DataFrame, test_size=0.2, random_state=42):
    X = df.drop(columns=["Healthy"])
    y = df["Healthy"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_test.reset_index(drop=True), y_test.reset_index(drop=True)
