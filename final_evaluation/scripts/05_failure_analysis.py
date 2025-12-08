#!/usr/bin/env python3
"""
Detailed failure analysis: identify patterns in when models fail.
Categorizes failures and extracts examples for manual inspection.
"""

import pandas as pd
import json
import re
from pathlib import Path
from collections import defaultdict

# ==================== PATH SETUP ====================
try:
    BASE_DIR = Path(__file__).resolve().parents[2]
except NameError:
    BASE_DIR = Path("/content/NLP-Final-Project")

EVAL_DIR = BASE_DIR / "final_evaluation"
RESULTS_DIR = EVAL_DIR / "results"
ANALYSIS_DIR = RESULTS_DIR / "failure_analysis"
KEYWORDS_PATH = BASE_DIR / "evaluation" / "keywords.json"

ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["baseline", "baseline_keywords", "safe_lora", "unfiltered_lora", "rl_lora"]
DATASETS = ["recipenlg", "kaggle", "adversarial"]

# ==================== KEYWORD LOADING ====================

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
    text = text.lower()
    keyword = keyword.lower()
    pattern = r'\b' + re.escape(keyword).replace(r'\ ', r'\s*') + r'\b'
    return bool(re.search(pattern, text))

# ==================== FAILURE CATEGORIZATION ====================

def categorize_failure(row, forbidden_terms, required_terms):
    """
    Categorize why a recipe failed LCR.
    Returns: (category, details)
    """
    generation = str(row['generation']).lower()
    
    has_forbidden = row.get('has_forbidden', 0) == 1
    has_required = row.get('has_required', 0) == 1
    
    # Find which specific forbidden ingredients present
    forbidden_found = [kw for kw in forbidden_terms if has_keyword_match(generation, kw)]
    
    # Find which required ingredients present
    required_found = [kw for kw in required_terms if has_keyword_match(generation, kw)]
    
    if has_forbidden and not has_required:
        return "both_violations", {
            "forbidden": forbidden_found[:5],  # Top 5
            "required_missing": True
        }
    elif has_forbidden:
        return "has_forbidden", {
            "forbidden": forbidden_found[:5]
        }
    elif not has_required:
        return "missing_required", {
            "required_missing": True,
            "required_found": required_found  # Show what they DID include
        }
    else:
        return "unknown", {}

def extract_recipe_length(generation):
    """Categorize recipe by length."""
    gen_str = str(generation)
    word_count = len(gen_str.split())
    
    if word_count < 20:
        return "very_short"
    elif word_count < 50:
        return "short"
    elif word_count < 100:
        return "medium"
    else:
        return "long"

def extract_dish_type(name):
    """Categorize dish type from name."""
    name_lower = str(name).lower()
    
    # Desserts
    if any(word in name_lower for word in ['cake', 'cookie', 'brownie', 'pie', 'dessert', 'sweet', 'chocolate', 'candy']):
        return "dessert"
    
    # Breakfast
    if any(word in name_lower for word in ['pancake', 'waffle', 'breakfast', 'muffin', 'scone', 'bagel']):
        return "breakfast"
    
    # Pasta/Rice
    if any(word in name_lower for word in ['pasta', 'spaghetti', 'noodle', 'rice', 'risotto', 'mac', 'macaroni']):
        return "carb_heavy"
    
    # Fried
    if any(word in name_lower for word in ['fried', 'fry', 'crispy', 'crunchy']):
        return "fried"
    
    # Protein dishes
    if any(word in name_lower for word in ['chicken', 'fish', 'salmon', 'turkey', 'tofu', 'beef', 'pork']):
        return "protein"
    
    # Vegetable dishes
    if any(word in name_lower for word in ['salad', 'vegetable', 'veggie', 'broccoli', 'spinach', 'kale']):
        return "vegetable"
    
    # Soup
    if any(word in name_lower for word in ['soup', 'stew', 'chili', 'broth']):
        return "soup"
    
    return "other"

# ==================== ANALYSIS FUNCTIONS ====================

def analyze_model_failures(model_name, forbidden_terms, required_terms):
    """Analyze failure patterns for one model across all datasets."""
    
    print(f"\n{'='*70}")
    print(f"ANALYZING FAILURES: {model_name.upper()}")
    print('='*70)
    
    all_failures = []
    all_successes = []
    
    # Load results from all datasets
    for dataset_name in DATASETS:
        file_path = RESULTS_DIR / f"{model_name}_{dataset_name}_full_results.csv"
        
        if not file_path.exists():
            continue
        
        df = pd.read_csv(file_path)
        
        # Add failure categorization
        failures = df[df['compliant'] == 0].copy()
        successes = df[df['compliant'] == 1].copy()
        
        if len(failures) > 0:
            failures['dataset'] = dataset_name
            failures['failure_category'] = failures.apply(
                lambda row: categorize_failure(row, forbidden_terms, required_terms)[0], 
                axis=1
            )
            failures['recipe_length'] = failures['generation'].apply(extract_recipe_length)
            
            if 'name' in failures.columns:
                failures['dish_type'] = failures['name'].apply(extract_dish_type)
            elif 'dish' in failures.columns:
                failures['dish_type'] = failures['dish'].apply(extract_dish_type)
            
            all_failures.append(failures)
        
        if len(successes) > 0:
            successes['dataset'] = dataset_name
            all_successes.append(successes)
    
    if not all_failures:
        print(f"✓ No failures found for {model_name}!")
        return
    
    failures_df = pd.concat(all_failures, ignore_index=True)
    
    # Print statistics
    total_failures = len(failures_df)
    print(f"\nTotal failures: {total_failures}")
    
    print(f"\n📊 Failure by Category:")
    print(failures_df['failure_category'].value_counts())
    
    print(f"\n📊 Failure by Recipe Length:")
    print(failures_df['recipe_length'].value_counts())
    
    print(f"\n📊 Failure by Dish Type:")
    print(failures_df['dish_type'].value_counts())
    
    print(f"\n📊 Failure by Dataset:")
    print(failures_df['dataset'].value_counts())
    
    # Sample examples from each failure category
    print(f"\n📝 Example Failures by Category:")
    print("-"*70)
    
    examples = []
    
    for category in failures_df['failure_category'].unique():
        cat_failures = failures_df[failures_df['failure_category'] == category]
        
        print(f"\n{category.upper()} ({len(cat_failures)} failures):")
        
        # Get 3 examples
        samples = cat_failures.head(3)
        
        for idx, row in samples.iterrows():
            dish_name = row.get('name', row.get('dish', 'Unknown'))
            generation = str(row['generation'])[:200] + "..." if len(str(row['generation'])) > 200 else str(row['generation'])
            
            print(f"  - {dish_name}")
            print(f"    Generated: {generation}")
            print()
            
            examples.append({
                'model': model_name,
                'category': category,
                'dish': dish_name,
                'generation': row['generation'],
                'dataset': row['dataset']
            })
    
    # Save detailed failure data
    failures_df.to_csv(ANALYSIS_DIR / f"{model_name}_failures_detailed.csv", index=False)
    
    # Save examples
    examples_df = pd.DataFrame(examples)
    examples_df.to_csv(ANALYSIS_DIR / f"{model_name}_failure_examples.csv", index=False)
    
    print(f"\n✓ Saved detailed failures to: {ANALYSIS_DIR / f'{model_name}_failures_detailed.csv'}")
    print(f"✓ Saved examples to: {ANALYSIS_DIR / f'{model_name}_failure_examples.csv'}")
    
    return failures_df

def create_cross_model_comparison(forbidden_terms, required_terms):
    """Compare failure patterns across all models."""
    
    print(f"\n{'='*70}")
    print("CROSS-MODEL FAILURE COMPARISON")
    print('='*70)
    
    all_model_stats = []
    
    for model_name in MODELS:
        failures_path = ANALYSIS_DIR / f"{model_name}_failures_detailed.csv"
        
        if not failures_path.exists():
            continue
        
        df = pd.read_csv(failures_path)
        
        stats = {
            'model': model_name,
            'total_failures': len(df),
            'has_forbidden': (df['failure_category'].isin(['has_forbidden', 'both_violations'])).sum(),
            'missing_required': (df['failure_category'].isin(['missing_required', 'both_violations'])).sum(),
            'dessert_failures': (df['dish_type'] == 'dessert').sum(),
            'carb_heavy_failures': (df['dish_type'] == 'carb_heavy').sum(),
            'very_short_recipes': (df['recipe_length'] == 'very_short').sum(),
        }
        
        all_model_stats.append(stats)
    
    stats_df = pd.DataFrame(all_model_stats)
    
    print(f"\n📊 Failure Statistics by Model:")
    print(stats_df.to_string(index=False))
    
    stats_df.to_csv(ANALYSIS_DIR / "cross_model_failure_comparison.csv", index=False)
    
    print(f"\n✓ Saved cross-model comparison to: {ANALYSIS_DIR / 'cross_model_failure_comparison.csv'}")

# ==================== MAIN ====================

def main():
    print("="*70)
    print("FAILURE PATTERN ANALYSIS")
    print("="*70)
    
    # Load keywords
    print("\nLoading keywords...")
    forbidden_groups, required_groups = load_keywords(KEYWORDS_PATH)
    forbidden_terms = expand_groups(forbidden_groups)
    required_terms = expand_groups(required_groups)
    print(f"✓ {len(forbidden_terms)} forbidden terms")
    print(f"✓ {len(required_terms)} required terms")
    
    # Analyze each model
    for model_name in MODELS:
        try:
            analyze_model_failures(model_name, forbidden_terms, required_terms)
        except Exception as e:
            print(f"❌ Error analyzing {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Cross-model comparison
    create_cross_model_comparison(forbidden_terms, required_terms)
    
    print(f"\n{'='*70}")
    print("✅ FAILURE ANALYSIS COMPLETE")
    print("="*70)
    print(f"Results saved to: {ANALYSIS_DIR}")
    print("="*70)

if __name__ == "__main__":
    main()
