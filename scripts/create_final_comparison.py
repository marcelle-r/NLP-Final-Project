#!/usr/bin/env python3
"""
Aggregate ROUGE, BERTScore, and LCR results into a single CSV:

evaluation/final_comparison.csv

Columns:
  model, rouge1, rouge2, rougeL,
  bertscore_precision, bertscore_recall, bertscore_f1,
  lcr
"""

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
EVAL_DIR = BASE_DIR / "evaluation"

# If your filenames differ slightly, tweak them here.
MODELS = [
    {
        "name": "Baseline",
        "rouge": "rouge_baseline_realistic.csv",
        "bertscore": "bertscore_baseline_realistic.csv",
        "lcr": "lcr_baseline.csv",
    },
    {
        "name": "Safe LoRA (filtered)",
        "rouge": "rouge_finetuned_safe_lora.csv",
        "bertscore": "bertscore_finetuned_safe_lora.csv",
        "lcr": "lcr_finetuned_safe_lora.csv",
    },
    {
        "name": "Unfiltered LoRA",
        "rouge": "rouge_finetuned_unfiltered_lora.csv",
        "bertscore": "bertscore_finetuned_unfiltered_lora.csv",
        "lcr": "lcr_finetuned_unfiltered_lora.csv",
    },
    {
        "name": "RL LoRA",
        "rouge": "rouge_finetuned_rl_lora.csv",
        "bertscore": "bertscore_finetuned_rl_lora.csv",
        "lcr": "lcr_finetuned_rl_lora.csv",
    },
]


def load_rouge(filename: str):
    df = pd.read_csv(EVAL_DIR / filename)
    return (
        df["rouge1_recall"].mean(),
        df["rouge2_recall"].mean(),
        df["rougeL_recall"].mean(),
    )


def load_bertscore(filename: str):
    df = pd.read_csv(EVAL_DIR / filename)
    return (
        df["precision"].mean(),
        df["recall"].mean(),
        df["f1"].mean(),
    )


def load_lcr(filename: str):
    df = pd.read_csv(EVAL_DIR / filename)
    # summary file has one row with a "compliant" count
    if "compliant" not in df.columns:
        raise ValueError(f"{filename} must contain a 'compliant' column")
    return float(df["compliant"].iloc[0])


def main():
    rows = []

    for cfg in MODELS:
        model_name = cfg["name"]
        print(f"Collecting metrics for: {model_name}")

        r1, r2, rL = load_rouge(cfg["rouge"])
        bp, br, bf = load_bertscore(cfg["bertscore"])
        lcr = load_lcr(cfg["lcr"])

        rows.append(
            {
                "model": model_name,
                "rouge1": r1,
                "rouge2": r2,
                "rougeL": rL,
                "bertscore_precision": bp,
                "bertscore_recall": br,
                "bertscore_f1": bf,
                "lcr": lcr,
            }
        )

    out_path = EVAL_DIR / "final_comparison.csv"
    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_path, index=False)
    print(f"\n✅ Saved final comparison to {out_path}")


if __name__ == "__main__":
    main()

