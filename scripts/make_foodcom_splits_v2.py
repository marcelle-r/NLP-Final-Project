#!/usr/bin/env python3
"""
Create train/val splits from the UPDATED safe dataset produced by
filter_foodcom_by_keywords_v2.py.

Input:
    data/foodcom_filtered_v2.csv

Outputs:
    data/processed/foodcom_train_v2.csv
    data/processed/foodcom_val_v2.csv
"""

import pandas as pd
from pathlib import Path


def main():
    base_dir = Path(__file__).resolve().parents[1]

    # New filtered dataset from filter_foodcom_by_keywords_v2.py
    safe_path = base_dir / "data" / "foodcom_filtered_v2.csv"

    print(f"[splits_v2] Loading safe recipes from: {safe_path}")
    df = pd.read_csv(safe_path)
    print(f"[splits_v2] Loaded {len(df)} recipes.")

    # Shuffle for randomness (reproducible)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    train_frac = 0.8
    n_total = len(df)
    n_train = int(train_frac * n_total)

    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train:].copy()

    out_dir = base_dir / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "foodcom_train_v2.csv"
    val_path = out_dir / "foodcom_val_v2.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    print(f"[splits_v2] Train: {len(train_df)} -> {train_path}")
    print(f"[splits_v2] Val:   {len(val_df)} -> {val_path}")


if __name__ == "__main__":
    main()

