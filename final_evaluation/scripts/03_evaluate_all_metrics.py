#!/usr/bin/env python3
"""
Evaluate all generated recipes with LCR, ROUGE, and BERTScore.
Includes safety-stratified analysis (originally safe vs unsafe recipes).
"""

import pandas as pd
import json
import re
from pathlib import Path
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from tqdm import tqdm
import numpy as np

# ==================== PATH SETUP ====================
try:
    BASE_DIR = Path(__file__).resolve().parents[2]
except NameError:
    BASE_DIR = Path("/content/NLP-Final-Project")

EVAL_DIR = BASE_DIR / "final_evaluation"
OUTPUTS_DIR = EVAL_DIR / "outputs"
RESULTS_DIR = EVAL_DIR / "results"
KEYWORDS_PATH = BASE_DIR / "final_evaluation" / "keywords_final.json"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Models to evaluate
MODELS = ["baseline", "baseline_keywords", "safe_lora", "unfiltered_lora", "rl_lora"]
DATASETS = ["recipenlg", "kaggle", "adversarial"]

# ==================== KEYWORD MATCHING (Word-Boundary) ====================

def load_keywords(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data["forbidden_groups"], data["required_groups"]

def expand_groups(groups):
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

def has_keyword_match(text, keyword):
    """
    Word-boundary matching to avoid false positives.
    Examples:
    - 'sugar' matches 'brown sugar' but NOT 'sugar-free'
    - 'corn' matches 'corn syrup' but NOT 'popcorn'
    """
    text = text.lower()
    keyword = keyword.lower()
    pattern = r'\b' + re.escape(keyword).replace(r'\ ', r'\s*') + r'\b'
    return bool(re.search(pattern, text))

def check_lcr(text, forbidden_terms, required_terms):
    """
    Check Lexical Constraint Respect (LCR).
    Returns: (is_compliant, has_forbidden, has_required)
    """
    text = str(text).lower()
    
    has_forbidden = any(has_keyword_match(text, kw) for kw in forbidden_terms)
    has_required = any(has_keyword_match(text, kw) for kw in required_terms)
    
    is_compliant = (not has_forbidden) and has_required
    
    return is_compliant, has_forbidden, has_required

# ==================== EVALUATION FUNCTIONS ====================

def evaluate_lcr(generations_df, forbidden_terms, required_terms):
    """Evaluate LCR on generated recipes."""
    results = []
    
    for _, row in generations_df.iterrows():
        gen_text = row['generation']
        is_compliant, has_forbidden, has_required = check_lcr(gen_text, forbidden_terms, required_terms)
        
        results.append({
            'compliant': int(is_compliant),
            'has_forbidden': int(has_forbidden),
            'has_required': int(has_required)
        })
    
    return pd.DataFrame(results)

def evaluate_rouge(generations_df):
    """Evaluate ROUGE scores."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    rouge_scores = []
    
    for _, row in tqdm(generations_df.iterrows(), total=len(generations_df), desc="ROUGE"):
        # Reference: original ingredients + directions
        reference = str(row['ingredients']) + " " + str(row['directions'])
        hypothesis = str(row['generation'])
        
        scores = scorer.score(reference, hypothesis)
        
        rouge_scores.append({
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        })
    
    return pd.DataFrame(rouge_scores)

def evaluate_bertscore(generations_df):
    """Evaluate BERTScore."""
    references = []
    hypotheses = []
    
    for _, row in generations_df.iterrows():
        ref = str(row['ingredients']) + " " + str(row['directions'])
        hyp = str(row['generation'])
        references.append(ref)
        hypotheses.append(hyp)
    
    print("Computing BERTScore (this may take a few minutes)...")
    P, R, F1 = bert_score(hypotheses, references, lang='en', verbose=False, device='cuda' if Path('/dev/nvidia0').exists() else 'cpu')
    
    return pd.DataFrame({
        'bertscore_precision': P.numpy(),
        'bertscore_recall': R.numpy(),
        'bertscore_f1': F1.numpy()
    })

# ==================== MAIN EVALUATION ====================

def evaluate_model_dataset(model_name, dataset_name, forbidden_terms, required_terms):
    """Evaluate one model on one dataset."""
    
    print(f"\n{'='*70}")
    print(f"EVALUATING: {model_name.upper()} × {dataset_name.upper()}")
    print('='*70)
    
    # Load generations
    gen_path = OUTPUTS_DIR / dataset_name / f"{model_name}_generations.csv"
    
    if not gen_path.exists():
        print(f"⚠️ File not found: {gen_path}")
        return None
    
    df = pd.read_csv(gen_path)
    print(f"Loaded {len(df)} generations")
    
    # Evaluate LCR
    print("\n[1/3] Evaluating LCR...")
    lcr_results = evaluate_lcr(df, forbidden_terms, required_terms)
    
    overall_lcr = lcr_results['compliant'].mean() * 100
    print(f"  Overall LCR: {overall_lcr:.1f}%")
    
    # Safety-stratified LCR (only for non-adversarial)
    if 'original_safety' in df.columns:
        safe_mask = df['original_safety'] == 1
        unsafe_mask = df['original_safety'] == 0
        
        if safe_mask.sum() > 0:
            safe_lcr = lcr_results[safe_mask]['compliant'].mean() * 100
            print(f"  LCR (originally safe): {safe_lcr:.1f}% ({safe_mask.sum()} recipes)")
        
        if unsafe_mask.sum() > 0:
            unsafe_lcr = lcr_results[unsafe_mask]['compliant'].mean() * 100
            print(f"  LCR (originally unsafe): {unsafe_lcr:.1f}% ({unsafe_mask.sum()} recipes)")
    
    # Evaluate ROUGE (only for non-adversarial - need references)
    rouge_results = None
    if dataset_name != "adversarial":
        print("\n[2/3] Evaluating ROUGE...")
        rouge_results = evaluate_rouge(df)
        print(f"  ROUGE-1: {rouge_results['rouge1'].mean():.3f}")
        print(f"  ROUGE-2: {rouge_results['rouge2'].mean():.3f}")
        print(f"  ROUGE-L: {rouge_results['rougeL'].mean():.3f}")
    
    # Evaluate BERTScore (only for non-adversarial)
    bert_results = None
    if dataset_name != "adversarial":
        print("\n[3/3] Evaluating BERTScore...")
        bert_results = evaluate_bertscore(df)
        print(f"  BERTScore Precision: {bert_results['bertscore_precision'].mean():.3f}")
        print(f"  BERTScore Recall: {bert_results['bertscore_recall'].mean():.3f}")
        print(f"  BERTScore F1: {bert_results['bertscore_f1'].mean():.3f}")
    
    # Combine all results
    results_df = pd.concat([df, lcr_results], axis=1)
    
    if rouge_results is not None:
        results_df = pd.concat([results_df, rouge_results], axis=1)
    
    if bert_results is not None:
        results_df = pd.concat([results_df, bert_results], axis=1)
    
    # Save detailed results
    output_path = RESULTS_DIR / f"{model_name}_{dataset_name}_full_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved detailed results to: {output_path}")
    
    # Create summary
    summary = {
        'model': model_name,
        'dataset': dataset_name,
        'n_recipes': len(df),
        'lcr_overall': overall_lcr
    }
    
    if 'original_safety' in df.columns:
        safe_mask = df['original_safety'] == 1
        unsafe_mask = df['original_safety'] == 0
        
        if safe_mask.sum() > 0:
            summary['lcr_orig_safe'] = lcr_results[safe_mask]['compliant'].mean() * 100
            summary['n_orig_safe'] = safe_mask.sum()
        
        if unsafe_mask.sum() > 0:
            summary['lcr_orig_unsafe'] = lcr_results[unsafe_mask]['compliant'].mean() * 100
            summary['n_orig_unsafe'] = unsafe_mask.sum()
    
    if rouge_results is not None:
        summary['rouge1'] = rouge_results['rouge1'].mean()
        summary['rouge2'] = rouge_results['rouge2'].mean()
        summary['rougeL'] = rouge_results['rougeL'].mean()
    
    if bert_results is not None:
        summary['bertscore_precision'] = bert_results['bertscore_precision'].mean()
        summary['bertscore_recall'] = bert_results['bertscore_recall'].mean()
        summary['bertscore_f1'] = bert_results['bertscore_f1'].mean()
    
    return summary

# ==================== MAIN ====================

def main():
    print("="*70)
    print("EVALUATING ALL MODELS ON ALL DATASETS")
    print("="*70)
    
    # Load keywords
    print("\nLoading diabetes-safety keywords...")
    forbidden_groups, required_groups = load_keywords(KEYWORDS_PATH)
    forbidden_terms = expand_groups(forbidden_groups)
    required_terms = expand_groups(required_groups)
    print(f"✓ {len(forbidden_terms)} forbidden terms")
    print(f"✓ {len(required_terms)} required terms")
    
    # Evaluate all combinations
    all_summaries = []
    
    for model_name in MODELS:
        for dataset_name in DATASETS:
            try:
                summary = evaluate_model_dataset(
                    model_name=model_name,
                    dataset_name=dataset_name,
                    forbidden_terms=forbidden_terms,
                    required_terms=required_terms
                )
                
                if summary is not None:
                    all_summaries.append(summary)
                    
            except Exception as e:
                print(f"❌ Error evaluating {model_name} × {dataset_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Save summary table
    summary_df = pd.DataFrame(all_summaries)
    summary_path = RESULTS_DIR / "evaluation_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    
    print(f"\n{'='*70}")
    print("✅ EVALUATION COMPLETE")
    print("="*70)
    print(f"Summary saved to: {summary_path}")
    print(f"Detailed results in: {RESULTS_DIR}")
    print("\nNext step: Run 04_create_final_report.py")
    print("="*70)

if __name__ == "__main__":
    main()
