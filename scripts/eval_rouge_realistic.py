#!/usr/bin/env python3
"""
Compute ROUGE on the *realistic* test set for:
  - baseline_realistic_generations.csv
  - finetuned_realistic.csv

Assumptions:
  - Reference recipes live in data/processed/foodcom_safe_recipes.csv
  - Generation CSVs have columns: dish, generation
  - Safe recipes CSV has at least: name, ingredients, steps
"""

from pathlib import Path
from typing import List, Tuple
import pandas as pd
from rouge_score import rouge_scorer


# ---- configuration ----

SAFE_RECIPES_REL = "data/processed/foodcom_safe_recipes.csv"

BASELINE_GEN_REL = "outputs/baseline_realistic_generations.csv"
FINETUNED_GEN_REL = "outputs/finetuned_realistic.csv"

# column names in CSVs
SAFE_NAME_COL = "name"          # in safe recipes
SAFE_INGR_COL = "ingredients"   # in safe recipes
SAFE_STEPS_COL = "steps"        # in safe recipes

GEN_DISH_COL = "dish"           # in generation files
GEN_TEXT_COL = "generation"     # in generation files

MAX_PAIRS = 100                 # number of (ref,gen) pairs to use


# ---- helper functions ----

def build_base_dir() -> Path:
    # repo root = parent of scripts/
    return Path(__file__).resolve().parents[1]


def load_reference_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # normalize for matching
    df["name_norm"] = df[SAFE_NAME_COL].str.strip().str.lower()

    # build reference text (ingredients + steps)
    for col in (SAFE_INGR_COL, SAFE_STEPS_COL):
        if col not in df.columns:
            df[col] = ""

    df["ref_text"] = (
        df[SAFE_INGR_COL].fillna("").astype(str)
        + " "
        + df[SAFE_STEPS_COL].fillna("").astype(str)
    ).str.strip()

    return df[[SAFE_NAME_COL, "name_norm", "ref_text"]]


def load_pairs(
    ref_df: pd.DataFrame, gen_path: Path, label: str
) -> Tuple[List[str], List[str]]:
    gen_df = pd.read_csv(gen_path)

    if GEN_DISH_COL not in gen_df.columns or GEN_TEXT_COL not in gen_df.columns:
        raise ValueError(
            f"{label}: expected columns '{GEN_DISH_COL}' and '{GEN_TEXT_COL}' "
            f"in {gen_path}"
        )

    gen_df["name_norm"] = gen_df[GEN_DISH_COL].str.strip().str.lower()
    merged = gen_df.merge(ref_df, on="name_norm", how="left", suffixes=("_gen", "_ref"))

    missing = merged["ref_text"].isna().sum()
    used = len(merged) - missing

    if missing > 0:
        print(
            f"{label}: WARNING – missing gold recipe for {missing} prompts, "
            f"using {used} pairs."
        )

    merged = merged[merged["ref_text"].notna()].copy()
    if MAX_PAIRS is not None:
        merged = merged.head(MAX_PAIRS)

    # refs = gold recipes, gens = generated recipes
    refs = merged["ref_text"].astype(str).tolist()
    gens = merged[GEN_TEXT_COL].astype(str).tolist()

    return refs, gens


def evaluate_rouge(name: str, refs: List[str], gens: List[str]) -> pd.DataFrame:
    assert len(refs) == len(gens)
    n = len(refs)
    print(f"\n=== Evaluating ROUGE for {name} ===")
    print(f"Used {n} pairs; missing gold for 0 prompts (after filtering).")

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    rows = []
    for i, (r, g) in enumerate(zip(refs, gens)):
        scores = scorer.score(r, g)
        rows.append(
            {
                "idx": i,
                "gold": r,
                "generation": g,
                "rouge1_recall": scores["rouge1"].recall,
                "rouge2_recall": scores["rouge2"].recall,
                "rougeL_recall": scores["rougeL"].recall,
            }
        )

    df = pd.DataFrame(rows)
    for col in ["rouge1_recall", "rouge2_recall", "rougeL_recall"]:
        print(f"{col.replace('_recall', '')}: {df[col].mean():.4f}")

    print(
        "rougeSum (mean of 1/2/L recall): "
        f"{df[['rouge1_recall','rouge2_recall','rougeL_recall']].mean(axis=1).mean():.4f}"
    )
    return df


def main():
    base_dir = build_base_dir()
    print(f"Current working directory (script base): {base_dir}")

    safe_path = base_dir / SAFE_RECIPES_REL
    if not safe_path.exists():
        raise FileNotFoundError(f"Safe recipes file not found: {safe_path}")

    ref_df = load_reference_table(safe_path)

    # Baseline
    baseline_path = base_dir / BASELINE_GEN_REL
    if baseline_path.exists():
        refs, gens = load_pairs(ref_df, baseline_path, "baseline_realistic")
        df_baseline = evaluate_rouge("baseline_realistic", refs, gens)
        out_path = base_dir / "evaluation" / "rouge_baseline_realistic.csv"
        out_path.parent.mkdir(exist_ok=True, parents=True)
        df_baseline.to_csv(out_path, index=False)
        print(f"Saved baseline ROUGE results to {out_path}")
    else:
        print(f"Skipping baseline_realistic – file not found: {baseline_path}")

    # Finetuned (filtered / safe LoRA)
    finetuned_path = base_dir / FINETUNED_GEN_REL
    if finetuned_path.exists():
        refs, gens = load_pairs(ref_df, finetuned_path, "finetuned_safe_lora")
        df_finetuned = evaluate_rouge("finetuned_safe_lora", refs, gens)
        out_path = base_dir / "evaluation" / "rouge_finetuned_safe_lora.csv"
        out_path.parent.mkdir(exist_ok=True, parents=True)
        df_finetuned.to_csv(out_path, index=False)
        print(f"Saved finetuned ROUGE results to {out_path}")
    else:
        print(f"Skipping finetuned_safe_lora – file not found: {finetuned_path}")


if __name__ == "__main__":
    main()

