#!/usr/bin/env python3
"""
Build T5-style (prompt, target) CSVs from Food.com-style splits.

Usage example:

    python scripts/build_t5_files_v2.py \
        --input_csv data/processed/foodcom_train_v2.csv \
        --output_csv data/processed/t5_train_safe_full_v2.csv

    python scripts/build_t5_files_v2.py \
        --input_csv data/processed/foodcom_val_v2.csv \
        --output_csv data/processed/t5_val_safe_full_v2.csv
"""

import argparse
import ast
from pathlib import Path

import pandas as pd


def _format_ingredients(raw):
    """
    RAW_recipes 'ingredients' is usually a Python list string.
    Convert to a clean bullet list if possible; otherwise, fall back to str(raw).
    """
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list):
            return "\n".join(f"- {item}" for item in parsed)
    except Exception:
        pass
    return str(raw)


def _format_steps(raw):
    """
    RAW_recipes 'steps' is usually a Python list string.
    Convert to numbered steps if possible; otherwise, fall back to str(raw).
    """
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list):
            return "\n".join(f"{i+1}. {step}" for i, step in enumerate(parsed))
    except Exception:
        pass
    return str(raw)


def build_t5_file(input_csv: str, output_csv: str):
    df = pd.read_csv(input_csv)

    # We assume the CSV has at least:
    # - 'name'       → recipe title
    # - 'ingredients'
    # - 'steps'
    # This matches Food.com official schema / RAW_recipes.
    missing_cols = [c for c in ["name", "ingredients", "steps"] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Input CSV is missing required columns: {missing_cols}")

    def make_prompt(row):
        # Prompt stays simple; data itself has already been filtered to be diabetes-safe.
        return (
            f"Generate a diabetes-friendly recipe for: {row['name']}\n"
            f"Include ingredients and step-by-step instructions.\n"
        )

    def make_target(row):
        ingredients_txt = _format_ingredients(row["ingredients"])
        steps_txt = _format_steps(row["steps"])
        return f"Ingredients:\n{ingredients_txt}\n\nSteps:\n{steps_txt}"

    out = pd.DataFrame(
        {
            "prompt": df.apply(make_prompt, axis=1),
            "target": df.apply(make_target, axis=1),
        }
    )

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    print(f"[t5_v2] Saved T5 training file to {output_path} with {len(out)} rows.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True, help="Input Food.com-style CSV")
    parser.add_argument("--output_csv", required=True, help="Output T5 (prompt,target) CSV")
    args = parser.parse_args()

    build_t5_file(args.input_csv, args.output_csv)


if __name__ == "__main__":
    main()

