#!/usr/bin/env python3
"""
Compute BERTScore on the *realistic* test set for:
  - outputs/baseline_realistic_generations.csv
  - outputs/finetuned_realistic.csv              (filtered / safe LoRA)
  - outputs/finetuned_unfiltered_realistic.csv   (unfiltered LoRA)
  - outputs/finetuned_rl_realistic.csv           (RL LoRA, LCR reward)
"""

from pathlib import Path
from typing import List, Tuple
import pandas as pd
from bert_score import score as bertscore

SAFE_RECIPES_REL = "data/processed/foodcom_safe_recipes.csv"

BASELINE_GEN_REL    = "outputs/baseline_realistic_generations.csv"
FILTERED_GEN_REL    = "outputs/finetuned_realistic.csv"
UNFILTERED_GEN_REL  = "outputs/finetuned_unfiltered_realistic.csv"
RL_GEN_REL          = "outputs/finetuned_rl_realistic.csv"   # <--- NEW

SAFE_NAME_COL = "name"
SAFE_INGR_COL = "ingredients"
SAFE_STEPS_COL = "steps"

GEN_DISH_COL = "dish"
GEN_TEXT_COL = "generation"

MAX_PAIRS = 100


def build_base_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def load_reference_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["name_norm"] = df[SAFE_NAME_COL].astype(str).str.strip().str.lower()

    for col in (SAFE_INGR_COL, SAFE_STEPS_COL):
        if col not in df.columns:
            df[col] = ""

    df["ref_text"] = (
        df[SAFE_INGR_COL].fillna("").astype(str)
        + " "
        + df[SAFE_STEPS_COL].fillna("").astype(str)
    ).str.strip()

    return df[["name_norm", "ref_text"]]


def load_pairs(
    ref_df: pd.DataFrame, gen_path: Path, label: str
) -> Tuple[List[str], List[str]]:
    gen_df = pd.read_csv(gen_path)

    if GEN_DISH_COL not in gen_df.columns or GEN_TEXT_COL not in gen_df.columns:
        raise ValueError(
            f"{label}: expected columns '{GEN_DISH_COL}' and '{GEN_TEXT_COL}' "
            f"in {gen_path}"
        )

    gen_df["name_norm"] = gen_df[GEN_DISH_COL].astype(str).str.strip().str.lower()
    merged = gen_df.merge(ref_df, on="name_norm", how="left")

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

    refs = merged["ref_text"].astype(str).tolist()
    gens = merged[GEN_TEXT_COL].astype(str).tolist()
    return refs, gens


def evaluate_bertscore(name: str, refs: List[str], gens: List[str]) -> pd.DataFrame:
    assert len(refs) == len(gens)
    n = len(refs)
    print(f"\n=== Evaluating BERTScore for {name} ===")
    print(f"Using {n} (ref, gen) pairs")

    # gens first, refs second (bert-score convention)
    P, R, F = bertscore(
        gens,
        refs,
        lang="en",
        model_type="roberta-large",
        rescale_with_baseline=False,
        verbose=True,
    )

    df = pd.DataFrame(
        {
            "idx": list(range(n)),
            "precision": P.tolist(),
            "recall": R.tolist(),
            "f1": F.tolist(),
        }
    )

    print(f"Mean BERTScore F1:    {df['f1'].mean():.4f}")
    print(f"Mean BERTScore Recall:{df['recall'].mean():.4f}")
    print(f"Mean BERTScore Prec.: {df['precision'].mean():.4f}")

    return df


def run_model(ref_df, rel_path, label, out_name):
    base_dir = build_base_dir()
    path = base_dir / rel_path
    if not path.exists():
        print(f"[WARNING] {path} not found. Skipping {label}.")
        return

    refs, gens = load_pairs(ref_df, path, label)
    df = evaluate_bertscore(label, refs, gens)
    out_path = base_dir / "evaluation" / out_name
    out_path.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(out_path, index=False)
    print(f"✅ Saved BERTScore results to {out_path}")


def main():
    base_dir = build_base_dir()
    safe_path = base_dir / SAFE_RECIPES_REL
    if not safe_path.exists():
        raise FileNotFoundError(f"Safe recipes file not found: {safe_path}")

    ref_df = load_reference_table(safe_path)

    run_model(ref_df, BASELINE_GEN_REL,   "baseline_realistic",        "bertscore_baseline_realistic.csv")
    run_model(ref_df, FILTERED_GEN_REL,   "finetuned_safe_lora",       "bertscore_finetuned_safe_lora.csv")
    run_model(ref_df, UNFILTERED_GEN_REL, "finetuned_unfiltered_lora", "bertscore_finetuned_unfiltered_lora.csv")
    # NEW: RL LoRA
    run_model(ref_df, RL_GEN_REL,         "finetuned_rl_lora",         "bertscore_finetuned_rl_lora.csv")


if __name__ == "__main__":
    main()

