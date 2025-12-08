#!/usr/bin/env python3
"""
Generate recipes from all models on external test sets.
Models: Baseline, Keyword-Enhanced Baseline, Safe LoRA, Unfiltered LoRA, RL LoRA
WITH AUTO-PUSH TO GITHUB
"""

import pandas as pd
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from tqdm import tqdm
import time
import subprocess

# ==================== PATH SETUP ====================
try:
    BASE_DIR = Path(__file__).resolve().parents[2]
except NameError:
    BASE_DIR = Path("/content/NLP-Final-Project")

EVAL_DIR = BASE_DIR / "final_evaluation"
DATA_DIR = EVAL_DIR / "data"
OUTPUTS_DIR = EVAL_DIR / "outputs"

# Create output directories
for subdir in ["recipenlg", "kaggle", "adversarial"]:
    (OUTPUTS_DIR / subdir).mkdir(parents=True, exist_ok=True)

# Model paths
BASE_MODEL = "google/flan-t5-base"
MODELS = {
    "baseline": None,
    "baseline_keywords": None,
    "safe_lora": BASE_DIR / "models" / "lora_t5_base_safe_full",
    "unfiltered_lora": BASE_DIR / "models" / "lora_t5_base_5000_unfiltered",
    "rl_lora": BASE_DIR / "models" / "lora_t5_base_5000_unfiltered_rl_lcr",
}

# ==================== GITHUB AUTO-COMMIT ====================

def setup_git():
    try:
        subprocess.run(["git", "config", "--global", "user.email", "dbm2146@columbia.edu"], 
                      cwd=BASE_DIR, check=False, capture_output=True)
        subprocess.run(["git", "config", "--global", "user.name", "Danielle Maydan"], 
                      cwd=BASE_DIR, check=False, capture_output=True)
        print("✓ Git configured")
        return True
    except Exception as e:
        print(f"⚠️ Git setup failed: {e}")
        return False

def commit_and_push(file_path, message):
    try:
        rel_path = file_path.relative_to(BASE_DIR)
        subprocess.run(["git", "add", str(rel_path)], 
                      cwd=BASE_DIR, check=True, capture_output=True)
        
        result = subprocess.run(["git", "commit", "-m", message], 
                               cwd=BASE_DIR, 
                               capture_output=True, 
                               text=True)
        
        if result.returncode != 0:
            if "nothing to commit" in result.stdout:
                print("  ℹ️ No changes to commit")
                return True
        
        result = subprocess.run(["git", "push"], 
                               cwd=BASE_DIR, 
                               capture_output=True, 
                               text=True,
                               timeout=30)
        
        if result.returncode == 0:
            print(f"  📤 Pushed to GitHub: {rel_path.name}")
            return True
        else:
            print(f"  ⚠️ Push failed: {result.stderr[:100]}")
            return False
            
    except subprocess.TimeoutExpired:
        print("  ⚠️ Push timeout")
        return False
    except Exception as e:
        print(f"  ⚠️ Git error: {e}")
        return False

# ==================== GPU CHECK ====================

def check_gpu():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✓ GPU Available: {gpu_name}")
        print(f"✓ GPU Memory: {gpu_mem:.2f} GB")
        return True
    else:
        print("⚠️ WARNING: No GPU detected!")
        return False

# ==================== PROMPT TEMPLATES ====================

def create_prompt(dish_name, model_type="baseline"):
    if model_type == "baseline":
        return f"Generate a diabetes-friendly {dish_name} recipe."
    
    elif model_type == "baseline_keywords":
        avoid = "sugar, honey, white flour, pasta, white rice, corn syrup, high fructose corn syrup, refined carbs, processed desserts"
        include = "vegetables, lean proteins (chicken breast, fish, tofu), olive oil, whole grains (quinoa, brown rice), stevia or monk fruit sweetener, almond flour, coconut flour"
        return f"Generate a diabetes-friendly {dish_name} recipe.\n\nAVOID these ingredients: {avoid}.\n\nINCLUDE these ingredients: {include}.\n\nRecipe:"
    
    else:
        return f"Generate a diabetes-friendly {dish_name} recipe."

# ==================== GENERATION FUNCTIONS ====================

def load_model(model_name, model_path):
    print(f"\n{'='*70}")
    print(f"Loading model: {model_name}")
    print('-'*70)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    
    if model_path is None:
        model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
        print("✓ Loaded base FLAN-T5 model")
    else:
        print("Loading base model...")
        base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
        print(f"Loading LoRA adapters from: {model_path}")
        model = PeftModel.from_pretrained(base_model, model_path)
        print("Merging LoRA adapters...")
        model = model.merge_and_unload()
        print("✓ LoRA adapters merged")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Moving model to {device}...")
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model ready on: {device}")
    
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated(0) / 1e9
        print(f"✓ GPU Memory allocated: {memory_allocated:.2f} GB")
    
    return tokenizer, model, device

def generate_batch(prompts, tokenizer, model, device, batch_size=8):
    generations = []
    
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        
        inputs = tokenizer(
            batch, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=512
        ).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                num_beams=4,
                early_stopping=True,
                do_sample=False,
                use_cache=True
            )
        
        batch_generations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        generations.extend(batch_generations)
        
        if i % 50 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return generations

def generate_for_dataset(dataset_name, dataset_path, model_name, model_path, model_type, git_enabled=False):
    print(f"\n{'='*70}")
    print(f"GENERATING: {dataset_name.upper()} × {model_name.upper()}")
    print('='*70)
    
    df = pd.read_csv(dataset_path)
    print(f"Loaded {len(df)} prompts from {dataset_path.name}")
    
    if 'dish' in df.columns:
        prompts = [create_prompt(dish, model_type) for dish in df['dish']]
        df['prompt'] = prompts
    else:
        prompts = [create_prompt(name, model_type) for name in df['name']]
        df['prompt'] = prompts
    
    tokenizer, model, device = load_model(model_name, model_path)
    
    print(f"\nGenerating {len(prompts)} recipes...")
    start_time = time.time()
    
    batch_size = 8 if device == "cuda" else 1
    
    generations = []
    with tqdm(total=len(prompts), desc=f"{model_name}", unit="recipe") as pbar:
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]
            batch_gens = generate_batch(batch, tokenizer, model, device, batch_size=len(batch))
            generations.extend(batch_gens)
            pbar.update(len(batch))
            
            if len(generations) % 50 == 0:
                elapsed = time.time() - start_time
                speed = len(generations) / elapsed
                pbar.set_postfix({"recipes/sec": f"{speed:.2f}"})
    
    df['generation'] = generations
    
    elapsed = time.time() - start_time
    print(f"✓ Generated {len(prompts)} recipes in {elapsed:.1f}s ({len(prompts)/elapsed:.2f} recipes/sec)")
    
    output_path = OUTPUTS_DIR / dataset_name / f"{model_name}_generations.csv"
    df.to_csv(output_path, index=False)
    print(f"✓ Saved to: {output_path}")
    
    if git_enabled:
        commit_message = f"Auto-save: {model_name} on {dataset_name} ({len(df)} recipes)"
        commit_and_push(output_path, commit_message)
    
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ==================== MAIN ====================

def main():
    print("="*70)
    print("GENERATING RECIPES FROM ALL MODELS")
    print("="*70)
    
    gpu_available = check_gpu()
    if not gpu_available:
        print("\n⚠️ No GPU detected. This will take several hours!")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting.")
            return
    
    git_enabled = setup_git()
    if git_enabled:
        print("✓ Auto-commit to GitHub enabled")
    else:
        print("⚠️ Auto-commit disabled (will save locally only)")
    
    test_sets = [
        ("recipenlg", DATA_DIR / "recipenlg_test_set.csv"),
        ("kaggle", DATA_DIR / "kaggle_test_set.csv"),
        ("adversarial", DATA_DIR / "adversarial_prompts.csv"),
    ]
    
    total_start = time.time()
    
    for model_idx, (model_name, model_path) in enumerate(MODELS.items(), 1):
        print(f"\n{'#'*70}")
        print(f"MODEL {model_idx}/{len(MODELS)}: {model_name.upper()}")
        print('#'*70)
        
        model_type = "baseline_keywords" if model_name == "baseline_keywords" else model_name
        
        for dataset_name, dataset_path in test_sets:
            try:
                generate_for_dataset(
                    dataset_name=dataset_name,
                    dataset_path=dataset_path,
                    model_name=model_name,
                    model_path=model_path,
                    model_type=model_type,
                    git_enabled=git_enabled
                )
            except Exception as e:
                print(f"❌ Error generating {dataset_name} × {model_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print("✅ ALL GENERATIONS COMPLETE")
    print("="*70)
    print(f"Total time: {total_elapsed/60:.1f} minutes")
    print(f"Outputs saved to: {OUTPUTS_DIR}")
    print("\nNext step: Run 03_evaluate_all_metrics.py")
    print("="*70)

if __name__ == "__main__":
    main()
