#!/usr/bin/env python3
"""
Filter RAW_recipes.csv using UPDATED LCR criteria from evaluation/keywords.json
and save to data/foodcom_filtered_v2.csv.

This script builds a combined text field from ingredients + steps
because RAW_recipes.csv does not include a unified text column.
"""

import json
from pathlib import Path
import pandas as pd
import ast


BASE_DIR = Path(__file__).resolve().parents[1]

RAW_CSV = BASE_DIR / "data" / "RAW_recipes.csv"        # <-- Correct file
KEYWORDS_JSON = BASE_DIR / "evaluation" / "keywords.json"
OUT_CSV = BASE_DIR / "data" / "foodcom_filtered_v2.csv"


def load_keywords(path: Path):
    with open(path, "r") as f:
        data = json.load(f)
    return data.get("forbidden_groups", {}), data.get("required_groups", {})


def combine_text(row):
    """Turn raw ingredients + steps into a plain text string."""
    ingredients = row.get("ingredients", "")
    steps = row.get("steps", "")

    try:
        ingredients = ", ".join(ast.literal_eval(ingredients))
    except Exception:
        ingredients = str(ingredients)

    try:
        steps_list = ast.literal_eval(steps)
        if isinstance(steps_list, list):
            steps = " ".join(s for s in steps_list)
        else:
            steps = str(steps)
    except Exception:
        steps = str(steps)

    return f"Ingredients: {ingredients}. Steps: {steps}".lower()


def passes_lcr_filter(text: str, forbidden_groups, required_groups):
    """Apply updated LCR logic."""
    has_forbidden = any(kw.lower() in text for g in forbidden_groups.values() for kw in g)
    has_required = any(kw.lower() in text for g in required_groups.values() for kw in g)
    return has_required and not has_forbidden


def main():
    print(f"Loading RAW_recipes.csv from: {RAW_CSV}")
    df = pd.read_csv(RAW_CSV)

    print(f"Loaded {len(df)} recipes.")

    forbidden, required = load_keywords(KEYWORDS_JSON)
    print("Loaded updated forbidden/required keywords.")

    # Build combined text
    df["combined_text"] = df.apply(combine_text, axis=1)

    # Apply filter
    mask = df["combined_text"].apply(lambda t: passes_lcr_filter(t, forbidden, required))
    filtered = df[mask].copy()

    print(f"Filtered down to {len(filtered)} safe recipes.")
    
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(OUT_CSV, index=False)
    print(f"Saved updated filtered dataset to: {OUT_CSV}")


if __name__ == "__main__":
    main()

