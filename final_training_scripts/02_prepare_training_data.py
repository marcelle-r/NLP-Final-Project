#!/usr/bin/env python3
"""
Prepare T5-formatted training data from filtered recipes
Creates prompt/target files for all dataset sizes
"""

import pandas as pd
from pathlib import Path
from tqdm import tqdm
import ast

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "final_training_data"

print("="*70)
print("PREPARING T5 TRAINING DATA")
print("="*70)

OUTPUT_DIR.mkdir(exist_ok=True)

def format_ingredients(ingredients_str):
    """Convert ['item1', 'item2'] to 'item1, item2'"""
    try:
        ing_list = ast.literal_eval(ingredients_str)
        return ", ".join(ing_list)
    except:
        return str(ingredients_str)

def format_steps(steps_str):
    """Convert ['step1', 'step2'] to '1. step1 2. step2'"""
    try:
        steps_list = ast.literal_eval(steps_str)
        formatted = " ".join([f"{i+1}. {step}" for i, step in enumerate(steps_list)])
        return formatted
    except:
        return str(steps_str)

def create_t5_format(df, output_prefix):
    """Create T5 input/target format."""
    
    prompts = []
    targets = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {output_prefix}"):
        # Prompt
        prompt = f"Generate a diabetes-friendly {row['name']} recipe."
        
        # Target - natural text format
        ingredients = format_ingredients(row['ingredients'])
        steps = format_steps(row['steps'])
        target = f"Ingredients: {ingredients} Instructions: {steps}"
        
        prompts.append(prompt)
        targets.append(target)
    
    # Save
    prompt_file = OUTPUT_DIR / f"{output_prefix}_prompts.txt"
    target_file = OUTPUT_DIR / f"{output_prefix}_targets.txt"
    
    with open(prompt_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(prompts))
    
    with open(target_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(targets))
    
    print(f"✓ Created {len(prompts):,} examples")

# Load data
print("\n1. Loading filtered datasets...")
train_df = pd.read_csv(DATA_DIR / "filtered_train_full.csv")
val_df = pd.read_csv(DATA_DIR / "filtered_val_full.csv")

print(f"✓ Training: {len(train_df):,}")
print(f"✓ Validation: {len(val_df):,}")

# Create full dataset
print("\n2. Creating FULL dataset (62K)...")
create_t5_format(train_df, "safe_full_train")
create_t5_format(val_df, "safe_full_val")

# Create subsets
print("\n3. Creating subset datasets...")
subsets = [(1000, "1k"), (5000, "5k"), (10000, "10k"), (20000, "20k")]

for size, name in subsets:
    if size <= len(train_df):
        print(f"\n   Creating {name} subset...")
        subset_train = train_df.sample(n=int(size * 0.9), random_state=42)
        subset_val = train_df.sample(n=int(size * 0.1), random_state=43)
        
        create_t5_format(subset_train, f"safe_{name}_train")
        create_t5_format(subset_val, f"safe_{name}_val")

# Sample output
print("\n4. Sample output:")
with open(OUTPUT_DIR / "safe_full_train_targets.txt", 'r') as f:
    print(f.readline()[:200])

print("\n✅ TRAINING DATA PREPARED")
