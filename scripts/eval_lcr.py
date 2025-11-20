#!/usr/bin/env python3
"""
Evaluate lexical compliance for generated recipes.

- Reads generations from a CSV file (default: outputs/baseline_generations.csv).
- Treats one column as the recipe text (default: 'generation').
- Loads keyword groups from evaluation/keywords.json:
    {
      "forbidden_groups": {
          "added_sugars": [ ... ],
          "refined_flours": [ ... ],
          ...
      },
      "required_groups": {
          "non_nutritive_sweeteners": [ ... ],
          "low_carb_flours": [ ... ],
          ...
      }
    }
- Flattens all groups into two keyword lists: forbidden and required.
- Computes Lexical Compliance Rate (LCR):
    compliant = (no forbidden keyword present) AND (at least one required keyword present)
- Writes per-example results to evaluation/lcr_baseline.csv
  and prints aggregate statistics.
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd


# ---------- Paths & config ----------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GENERATIONS_CSV = PROJECT_ROOT / "outputs" / "baseline_generations.csv"
DEFAULT_KEYWORDS_JSON = PROJECT_ROOT / "evaluation" / "keywords.json"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "evaluation" / "lcr_baseline.csv"


# ---------- Keyword loading ----------

def load_keyword_config(path: Path = DEFAULT_KEYWORDS_JSON):
    """
    Load keyword groups from evaluation/keywords.json and flatten them.

    Returns:
        forbidden_keywords (set[str])
        required_keywords (set[str])
    """
    if not path.exists():
        raise FileNotFoundError(f"Keyword config not found at {path}")

    with path.open("r") as f:
        cfg = json.load(f)

    forbidden_keywords = set()
    required_keywords = set()

    # Flatten forbidden groups
    for group_name, items in cfg.get("forbidden_groups", {}).items():
        for item in items:
            if item:
                forbidden_keywords.add(item.lower())

    # Flatten required groups
    for group_name, items in cfg.get("required_groups", {}).items():
        for item in items:
            if item:
                required_keywords.add(item.lower())

    print(
        "Loaded",
        len(forbidden_keywords),
        "forbidden keywords and",
        len(required_keywords),
        "required keywords.",
    )
    return forbidden_keywords, required_keywords


# ---------- Core evaluation ----------

def check_keywords(text: str, forbidden: set, required: set):
    """
    Check a single recipe text for forbidden/required keywords.

    Args:
        text: recipe text (ingredients + directions, etc.)
        forbidden: set of forbidden substrings (lowercased)
        required: set of required substrings (lowercased)

    Returns:
        has_forbidden (bool)
        has_required (bool)
    """
    if not isinstance(text, str):
        text_l = ""
    else:
        text_l = text.lower()

    has_forbidden = any(kw in text_l for kw in forbidden) if forbidden else False
    has_required = any(kw in text_l for kw in required) if required else False
    return has_forbidden, has_required


def evaluate_lcr(
    generations_csv: Path,
    text_column: str = "generation",
    output_csv: Path = DEFAULT_OUTPUT_CSV,
):
    """
    Compute lexical compliance metrics over a generations CSV.
    """
    if not generations_csv.exists():
        raise FileNotFoundError(f"Generations file not found at {generations_csv}")

    df = pd.read_csv(generations_csv)

    if text_column not in df.columns:
        raise ValueError(
            f"Column '{text_column}' not found in {generations_csv}. "
            f"Available columns: {list(df.columns)}"
        )

    print(f"Loading generations from: {generations_csv}")
    print(f"Using column '{text_column}' as the recipe text.")

    forbidden, required = load_keyword_config()

    # Evaluate per row
    has_forbidden_list = []
    has_required_list = []
    compliant_list = []

    for txt in df[text_column].tolist():
        hf, hr = check_keywords(txt, forbidden, required)
        has_forbidden_list.append(hf)
        has_required_list.append(hr)
        compliant_list.append((not hf) and hr)

    df_results = df.copy()
    df_results["has_forbidden"] = has_forbidden_list
    df_results["has_required"] = has_required_list
    df_results["compliant"] = compliant_list

    # Aggregate stats
    total = len(df_results)
    n_no_forbidden = sum(not x for x in has_forbidden_list)
    n_has_required = sum(has_required_list)
    n_compliant = sum(compliant_list)

    # Avoid division by zero
    def pct(n):
        return 0.0 if total == 0 else (100.0 * n / total)

    print("\n=== Lexical Compliance Results ===")
    print(f"Total generations:               {total}")
    print(
        f"Compliant (no forbidden AND required present): "
        f"{n_compliant} ({pct(n_compliant):.1f}%)"
    )
    print(
        f"No forbidden ingredients:        {n_no_forbidden} "
        f"({pct(n_no_forbidden):.1f}%)"
    )
    print(
        f"Has required ingredients:        {n_has_required} "
        f"({pct(n_has_required):.1f}%)"
    )

    # Ensure output directory exists
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(output_csv, index=False)
    print(f"\nPer-example results written to: {output_csv}")


# ---------- CLI ----------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate lexical compliance (LCR) of generated recipes."
    )
    parser.add_argument(
        "--generations",
        type=str,
        default=str(DEFAULT_GENERATIONS_CSV),
        help=f"Path to CSV with generations "
             f"(default: {DEFAULT_GENERATIONS_CSV})",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="generation",
        help="Column name containing recipe text (default: 'generation')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_CSV),
        help=f"Path for per-example LCR CSV "
             f"(default: {DEFAULT_OUTPUT_CSV})",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    generations_csv = Path(args.generations)
    output_csv = Path(args.output)

    evaluate_lcr(
        generations_csv=generations_csv,
        text_column=args.text_column,
        output_csv=output_csv,
    )


if __name__ == "__main__":
    main()

