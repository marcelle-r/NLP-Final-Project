#!/usr/bin/env python3
"""
Compute BERTScore for realistic test set generations.

This script expects the ROUGE result CSVs produced by eval_rouge_realistic.py:
- evaluation/rouge_baseline_realistic.csv
- evaluation/rouge_finetuned_safe_lora.csv
- (optionally) evaluation/rouge_finetuned_unfiltered_lora.csv

Each CSV must contain at least the columns:
- 'generation': model output
- 'gold': reference recipe
"""

import pandas as pd
from pathlib import Path
from bert_score import score


def eval_model(run_name: str, rouge_csv_name: str, out_csv_name: str) -> None:
    base_dir = Path(__file__).resolve().parents[1]
    eval_dir = base_dir / "evaluation"

    in_path = eval_dir / rouge_csv_name
    if not in_path.exists():
        print(f"[WARNING] {in_path} not found. Skipping {run_name}.")
        return

    print(f"\n=== Evaluating BERTScore for {run_name} ===")
    print(f"Loading: {in_path}")

    df = pd.read_csv(in_path)

    if "generation" not in df.columns or "gold" not in df.columns:
        raise ValueError(
            f"{in_path} must contain 'generation' and 'gold' columns "
            f"(found: {list(df.columns)})"
        )

    candidates = df["generation"].astype(str).tolist()
    references = df["gold"].astype(str).tolist()

    # BERTScore returns tensors of precision, recall, and F1 for each pair
    P, R, F1 = score(
        candidates,
        references,
        lang="en",
        rescale_with_baseline=True,
    )

    df["bert_precision"] = P.tolist()
    df["bert_recall"] = R.tolist()
    df["bert_f1"] = F1.tolist()

    print(f"Mean BERTScore F1:     {df['bert_f1'].mean():.4f}")
    print(f"Mean BERTScore Recall: {df['bert_recall'].mean():.4f}")
    print(f"Mean BERTScore Prec.:  {df['bert_precision'].mean():.4f}")

    out_path = eval_dir / out_csv_name
    df.to_csv(out_path, index=False)
    print(f"✅ Saved BERTScore results to {out_path}")


def main() -> None:
    """
    Evaluate BERTScore for:
      - baseline_realistic
      - finetuned_safe_lora
      - (optionally) finetuned_unfiltered_lora
    """
    configs = [
        # name,                       input ROUGE csv,                        output BERTScore csv
        ("baseline_realistic",        "rouge_baseline_realistic.csv",         "bertscore_baseline_realistic.csv"),
        ("finetuned_safe_lora",      "rouge_finetuned_safe_lora.csv",        "bertscore_finetuned_safe_lora.csv"),
        ("finetuned_unfiltered_lora","rouge_finetuned_unfiltered_lora.csv",  "bertscore_finetuned_unfiltered_lora.csv"),
    ]

    for run_name, in_csv, out_csv in configs:
        eval_model(run_name, in_csv, out_csv)


if __name__ == "__main__":
    main()

