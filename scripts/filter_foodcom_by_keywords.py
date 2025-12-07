#!/usr/bin/env python3

import ast
import json
import sys
import argparse
import pandas as pd
from pathlib import Path
from typing import List, Dict


# ------------------------- Keyword utils ------------------------- #

def load_keywords(path: Path) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def expand_groups(groups: Dict[str, List[str]]) -> List[str]:
    """
    Flatten {"group_name": ["term1", "term2", ...], ...}
    into a unique, lowercase list of terms.
    """
    all_terms = []
    for _, terms in groups.items():
        all_terms.extend(t.lower() for t in terms)

    seen = set()
    uniq = []
    for t in all_terms:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


# ------------------------- Ingredient helpers ------------------------- #

def list_from_string(s):
    """
    Ingredients in RAW_recipes look like "['sugar', 'flour']".
    This safely converts that to a Python list.
    If parsing fails, returns [].
    """
    if isinstance(s, list):
        return s
    try:
        return ast.literal_eval(s)
    except Exception:
        return []


def has_any(ingredients: List[str], terms: List[str]) -> bool:
    """
    Check whether any of the terms appear in the joined ingredient string.
    """
    joined = " ".join(ingredients).lower()
    return any(term in joined for term in terms)


# ------------------------- Core labeling logic ------------------------- #

def label_safe_series(df: pd.DataFrame,
                      ingredients_col: str,
                      forbidden_terms: List[str],
                      required_terms: List[str]) -> pd.Series:
    """
    Given a DataFrame + column name for ingredients, return a
    Series of 0/1 labels where 1 = diabetes-safe.
    """

    # Build ingredients_list column
    df = df.copy()
    df["ingredients_list"] = df[ingredients_col].apply(list_from_string)

    def label_row(row):
        ings = row["ingredients_list"]
        f_hit = has_any(ings, forbidden_terms)
        r_hit = has_any(ings, required_terms)
        # safe iff NO forbidden and at least one required
        return int((not f_hit) and r_hit)

    return df.apply(lambda row: label_row(row), axis=1)


# ------------------------- Mode 1: Original RAW_recipes split ------------------------- #

def run_raw_split_mode():
    """
    Original behavior:
      - Read RAW_recipes.csv
      - Label as diabetes_safe
      - Write foodcom_safe_recipes.csv and foodcom_unsafe_recipes.csv
    """
    base_dir = Path(__file__).resolve().parents[1]

    # 🔍 Try multiple possible locations for RAW_recipes.csv
    candidate_paths = [
        base_dir / "data" / "raw" / "RAW_recipes.csv",                    # teammates
        base_dir / "data" / "external" / "foodcom" / "RAW_recipes.csv",  # your local
        base_dir / "data" / "external" / "RAW_recipes.csv",
        base_dir / "RAW_recipes.csv",
    ]

    data_path = None
    for p in candidate_paths:
        if p.exists():
            data_path = p
            break

    if data_path is None:
        raise FileNotFoundError(
            "RAW_recipes.csv not found in expected locations.\n"
            "Checked:\n" + "\n".join(str(p) for p in candidate_paths)
        )

    keywords_path = base_dir / "evaluation" / "keywords.json"

    print(f"[raw_split] Loading data from {data_path}")
    df = pd.read_csv(data_path)

    print(f"[raw_split] Loading keywords from {keywords_path}")
    keywords = load_keywords(keywords_path)
    forbidden_terms = expand_groups(keywords["forbidden_groups"])
    required_terms = expand_groups(keywords["required_groups"])
    print(f"[raw_split] {len(forbidden_terms)} forbidden, "
          f"{len(required_terms)} required terms")

    # Label safe/unsafe
    df["diabetes_safe"] = label_safe_series(
        df,
        ingredients_col="ingredients",
        forbidden_terms=forbidden_terms,
        required_terms=required_terms,
    )

    safe = df[df["diabetes_safe"] == 1]
    unsafe = df[df["diabetes_safe"] == 0]

    out_dir = base_dir / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    safe.to_csv(out_dir / "foodcom_safe_recipes.csv", index=False)
    unsafe.to_csv(out_dir / "foodcom_unsafe_recipes.csv", index=False)

    print(f"[raw_split] Safe recipes:   {len(safe)}")
    print(f"[raw_split] Unsafe recipes: {len(unsafe)}")
    print(f"[raw_split] Wrote to {out_dir}")



# ------------------------- Mode 2: Generic filter (train/val) ------------------------- #

def run_filter_mode(args):
    """
    New behavior:
      - Read arbitrary CSV (e.g., foodcom_train.csv or foodcom_val.csv)
      - Label with diabetes_safe
      - Write ONLY the safe subset to output_csv
    """
    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    keywords_path = Path(args.keywords_json)
    ingredients_col = args.ingredients_col

    print(f"[filter] Loading data from {input_csv}")
    df = pd.read_csv(input_csv)

    print(f"[filter] Loading keywords from {keywords_path}")
    keywords = load_keywords(keywords_path)
    forbidden_terms = expand_groups(keywords["forbidden_groups"])
    required_terms = expand_groups(keywords["required_groups"])
    print(f"[filter] {len(forbidden_terms)} forbidden, "
          f"{len(required_terms)} required terms")

    df = df.copy()
    df["diabetes_safe"] = label_safe_series(
        df,
        ingredients_col=ingredients_col,
        forbidden_terms=forbidden_terms,
        required_terms=required_terms,
    )

    safe = df[df["diabetes_safe"] == 1].copy()
    n_total = len(df)
    n_safe = len(safe)
    frac = n_safe / max(1, n_total)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    safe.to_csv(output_csv, index=False)

    print(f"[filter] Safe recipes: {n_safe}/{n_total} ({frac:.1%})")
    print(f"[filter] Saved filtered (safe-only) data to {output_csv}")


# ------------------------- Main entrypoint ------------------------- #

def main():
    # If no extra CLI args -> preserve old behavior.
    if len(sys.argv) == 1:
        run_raw_split_mode()
        return

    # Otherwise use new filter mode with argparse.
    parser = argparse.ArgumentParser(
        description="Filter Food.com-style CSVs into diabetes-safe subsets "
                    "using grouped keywords.json."
    )
    parser.add_argument(
        "--input_csv",
        required=True,
        help="Path to input CSV (e.g., data/foodcom_train.csv)",
    )
    parser.add_argument(
        "--output_csv",
        required=True,
        help="Where to write filtered safe-only CSV "
             "(e.g., data/filtered_train.csv)",
    )
    parser.add_argument(
        "--keywords_json",
        required=True,
        help="Path to evaluation/keywords.json",
    )
    parser.add_argument(
        "--ingredients_col",
        default="ingredients",
        help="Name of the column containing the ingredient list string "
             "(default: 'ingredients')",
    )

    args = parser.parse_args()
    run_filter_mode(args)


if __name__ == "__main__":
    main()

