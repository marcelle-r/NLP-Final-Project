#!/usr/bin/env python3
"""
Filter Food.com dataset using keywords_final.json
Creates filtered_recipes_full.csv and 90/10 train/val splits
"""

import pandas as pd
import json
import re
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).parent.parent
RAW_DATA = BASE_DIR / "data" / "raw" / "RAW_recipes.csv"
KEYWORDS = BASE_DIR / "final_evaluation" / "keywords_final.json"

print("="*70)
print("FILTERING FOOD.COM DATASET")
print("="*70)

# Load data
print("\n1. Loading raw dataset...")
df = pd.read_csv(RAW_DATA)
print(f"✓ Loaded {len(df):,} recipes")

# Load keywords
print("\n2. Loading keywords...")
with open(KEYWORDS, 'r') as f:
    keywords_data = json.load(f)

def expand_groups(groups):
    all_terms = []
    for _, terms in groups.items():
        all_terms.extend(t.lower() for t in terms)
    return list(set(all_terms))

forbidden_terms = expand_groups(keywords_data["forbidden_groups"])
required_terms = expand_groups(keywords_data["required_groups"])

print(f"✓ {len(forbidden_terms)} forbidden terms")
print(f"✓ {len(required_terms)} required terms")

# Filter function
def has_keyword_match(text, keyword):
    text = str(text).lower()
    keyword = keyword.lower()
    pattern = r'\b' + re.escape(keyword).replace(r'\ ', r'\s*') + r'\b'
    return bool(re.search(pattern, text))

def check_recipe_safety(row):
    text = f"{row['name']} {row['ingredients']} {row['steps']}"
    text = str(text).lower()
    
    has_forbidden = any(has_keyword_match(text, kw) for kw in forbidden_terms)
    has_required = any(has_keyword_match(text, kw) for kw in required_terms)
    
    return (not has_forbidden) and has_required

# Filter
print("\n3. Filtering recipes...")
tqdm.pandas(desc="Checking recipes")
df['passes_lcr'] = df.progress_apply(check_recipe_safety, axis=1)
filtered_df = df[df['passes_lcr'] == True].copy()

print(f"\n4. Results:")
print(f"   Total: {len(df):,}")
print(f"   ✓ Passed: {len(filtered_df):,} ({len(filtered_df)/len(df)*100:.1f}%)")
print(f"   ✗ Failed: {len(df) - len(filtered_df):,}")

# Save
output_dir = BASE_DIR / "data"
filtered_df.to_csv(output_dir / "filtered_recipes_full.csv", index=False)

# Create splits
print("\n5. Creating 90/10 train/val split...")
train_df, val_df = train_test_split(filtered_df, test_size=0.10, random_state=42)

train_df.to_csv(output_dir / "filtered_train_full.csv", index=False)
val_df.to_csv(output_dir / "filtered_val_full.csv", index=False)

print(f"   Training: {len(train_df):,}")
print(f"   Validation: {len(val_df):,}")

print("\n✅ FILTERING COMPLETE")
