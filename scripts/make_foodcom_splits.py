#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

def main():
    base_dir = Path(__file__).resolve().parents[1]
    safe_path = base_dir / "data" / "processed" / "foodcom_safe_recipes.csv"

    print(f"Loading safe recipes from {safe_path}")
    df = pd.read_csv(safe_path)

    # Shuffle once for randomness
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    train_frac = 0.8
    n_total = len(df)
    n_train = int(train_frac * n_total)

    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train:].copy()

    out_dir = base_dir / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "foodcom_train.csv"
    val_path = out_dir / "foodcom_val.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    print(f"Train: {len(train_df)} -> {train_path}")
    print(f"Val:   {len(val_df)} -> {val_path}")

if __name__ == "__main__":
    main()
