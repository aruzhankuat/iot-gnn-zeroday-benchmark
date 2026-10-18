"""Data loading, validation, and zero-day (leave-one-attack-out) splitting."""

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

BENIGN_LABEL = "benign"


def load_dataset(config: dict) -> pd.DataFrame:
    """Load dataset from config path, verify required columns, sort chronologically."""
    path = config["data"]["path"]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at path: {path}")

    df = pd.read_csv(path)
    required_cols = (
        config["data"]["feature_cols"]
        + [config["data"]["label_col"]]
        + [config["data"]["attack_col"]]
    )
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' missing from dataset {path}")

    # Chronological order matters: benign split and graph edges both depend on it.
    if "ts" in df.columns:
        df = df.sort_values("ts", kind="stable")
    return df.reset_index(drop=True)


def list_attack_families(df: pd.DataFrame, config: dict) -> list:
    """All attack families present in the data (each one is held out in turn)."""
    attack_col = config["data"]["attack_col"]
    return sorted(df.loc[df[attack_col] != BENIGN_LABEL, attack_col].unique().tolist())


def split_leave_one_attack_out(df: pd.DataFrame, config: dict, held_out_attack: str):
    """
    Zero-day split for one fold:
    - Benign: first `benign_train_frac` (chronologically) -> train, rest -> test
    - Known attacks -> train
    - Held-out attack (unseen) -> test ONLY
    Both sets are re-sorted chronologically so graph edges do not follow class blocks.
    """
    feature_cols = config["data"]["feature_cols"]
    label_col = config["data"]["label_col"]
    attack_col = config["data"]["attack_col"]
    train_frac = config["data"].get("benign_train_frac", 0.6)

    benign_df = df[df[attack_col] == BENIGN_LABEL]
    n_benign_train = int(len(benign_df) * train_frac)
    benign_train = benign_df.iloc[:n_benign_train]
    benign_test = benign_df.iloc[n_benign_train:]

    attack_df = df[df[attack_col] != BENIGN_LABEL]
    known_attacks = attack_df[attack_df[attack_col] != held_out_attack]
    unseen_attacks = attack_df[attack_df[attack_col] == held_out_attack]

    # concat keeps original (chronological) index -> sort_index restores time order
    train_df = pd.concat([benign_train, known_attacks]).sort_index().reset_index(drop=True)
    test_df = pd.concat([benign_test, unseen_attacks]).sort_index().reset_index(drop=True)

    # Scaler fitted strictly on train to prevent leakage
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].values)
    X_test = scaler.transform(test_df[feature_cols].values)

    return {
        "train_df": train_df,
        "test_df": test_df,
        "X_train": X_train,
        "y_train": train_df[label_col].values,
        "X_test": X_test,
        "y_test": test_df[label_col].values,
        "test_attacks": test_df[attack_col].values,
        "held_out_attack": held_out_attack,
        "scaler": scaler,
    }
