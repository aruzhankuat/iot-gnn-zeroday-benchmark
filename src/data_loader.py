"""Data loading, validation, and zero-day (leave-one-attack-out) splitting."""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_dataset(config: dict) -> pd.DataFrame:
    """Load dataset from path specified in config, verifying required columns."""
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
    return df


def split_leave_one_attack_out(
    df: pd.DataFrame, config: dict, held_out_attack: str | None = None
):
    """
    Split data for zero-day evaluation:
    - Benign: chronological split (benign_train_frac for train, remainder for test)
    - Known attacks: included in train
    - Held-out attack (unseen zero-day): strictly withheld from train, appears ONLY in test
    """
    benign_label = "benign"
    feature_cols = config["data"]["feature_cols"]
    label_col = config["data"]["label_col"]
    attack_col = config["data"]["attack_col"]
    train_frac = config["data"].get("benign_train_frac", 0.6)

    # Benign flows chronological partition
    benign_df = df[df[attack_col] == benign_label].copy()
    n_benign_train = int(len(benign_df) * train_frac)
    benign_train = benign_df.iloc[:n_benign_train]
    benign_test = benign_df.iloc[n_benign_train:]

    # Attack flows partition
    attack_df = df[df[attack_col] != benign_label].copy()
    unique_attacks = attack_df[attack_col].unique().tolist()

    if held_out_attack is None and len(unique_attacks) > 0:
        held_out_attack = unique_attacks[0]

    # Known attacks go to train, held-out attack strictly goes to test
    known_attacks = attack_df[attack_df[attack_col] != held_out_attack]
    unseen_attacks = attack_df[attack_df[attack_col] == held_out_attack]

    train_df = pd.concat([benign_train, known_attacks], ignore_index=True)
    test_df = pd.concat([benign_test, unseen_attacks], ignore_index=True)

    # Fit scaler strictly on train set to prevent data leakage
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].values)
    y_train = train_df[label_col].values

    X_test = scaler.transform(test_df[feature_cols].values)
    y_test = test_df[label_col].values
    test_attacks = test_df[attack_col].values

    return {
        "train_df": train_df,
        "test_df": test_df,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "test_attacks": test_attacks,
        "held_out_attack": held_out_attack,
        "scaler": scaler,
    }
