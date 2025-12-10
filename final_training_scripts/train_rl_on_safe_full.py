#!/usr/bin/env python3
"""
RL training on Safe LoRA Full using REINFORCE with LCR rewards.
WITH AUTO-PUSH CHECKPOINTS
"""
import torch
import torch.nn.functional as F
import sys
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments, TrainerCallback
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import Dataset
import json
import re
import subprocess

BASE_DIR = Path(__file__).resolve().parents[1]

# ==================== AUTO-PUSH CALLBACK ====================
class GitHubPushCallback(TrainerCallback):
    def __init__(self, repo_path, model_name):
        self.repo_path = repo_path
        self.model_name = model_name
    
    def on_save(self, args, state, control, **kwargs):
        """Called after each checkpoint save"""
        print(f"\n📤 Auto-pushing RL checkpoint {state.global_step}...", flush=True)
        
        try:
            # Pull first
            result = subprocess.run(
                ["git", "pull", "--rebase"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0 and "CONFLICT" in result.stderr:
                print("  ⚠️ Conflict detected, resolving...", flush=True)
                subprocess.run(["git", "rebase", "--abort"], cwd=self.repo_path, capture_output=True)
                subprocess.run(["git", "pull", "--no-rebase"], cwd=self.repo_path, capture_output=True)
            
            # Add and commit
            subprocess.run(
                ["git", "add", f"final_models/{self.model_name}/"],
                cwd=self.repo_path,
                capture_output=True
            )
            
            result = subprocess.run(
                ["git", "commit", "-m", f"RL Checkpoint: {self.model_name} at step {state.global_step}"],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            
            if "nothing to commit" in result.stdout:
                print("  ℹ️ No new changes to commit", flush=True)
                return
            
            # Push with retry
            for attempt in range(3):
                result = subprocess.run(
                    ["git", "push"],
                    cwd=self.repo_path,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0:
                    print(f"  ✓ Pushed RL checkpoint {state.global_step}", flush=True)
                    return
                elif "rejected" in result.stderr:
                    print(f"  ⚠️ Push rejected (attempt {attempt+1}/3), pulling and retrying...", flush=True)
                    subprocess.run(["git", "pull", "--rebase"], cwd=self.repo_path, capture_output=True)
                else:
                    print(f"  ⚠️ Push failed: {result.stderr[:100]}", flush=True)
                    return
            
            print("  ⚠️ Push failed after 3 attempts", flush=True)
            
        except subprocess.TimeoutExpired:
            print("  ⚠️ Git operation timed out", flush=True)
        except Exception as e:
            print(f"  ⚠️ Error: {e}", flush=True)

# ==================== KEYWORD LOADING ====================
print("="*70, flush=True)
print("RL TRAINING: SAFE LORA FULL + RL", flush=True)
print("="*70, flush=True)

print("\n1. Loading keywords...", flush=True)
with open(BASE_DIR / "final_evaluation/keywords_final.json", 'r') as f:
    keywords = json.load(f)

forbidden_keywords = []
for group in keywords['forbidden_groups'].values():
    forbidden_keywords.extend(group)

required_keywords = []
for group in keywords['required_groups'].values():
    required_keywords.extend(group)

print(f"✓ {len(forbidden_keywords)} forbidden keywords", flush=True)
print(f"✓ {len(required_keywords)} required keywords", flush=True)

# ==================== REWARD FUNCTIONS ====================
def has_keyword_match(text, keyword):
    """Check if keyword exists in text with word boundaries."""
    text = text.lower()
    keyword = keyword.lower()
    # Handle multi-word keywords with flexible spacing
    pattern = r'\b' + re.escape(keyword).replace(r'\ ', r'\s+') + r'\b'
    return bool(re.search(pattern, text))
def check_lcr(text):
    text = str(text).lower()
    has_forbidden = any(has_keyword_match(text, kw) for kw in forbidden_keywords)
    has_required = any(has_keyword_match(text, kw) for kw in required_keywords)
    return (not has_forbidden) and has_required

def compute_rewards(generated_texts):
    rewards = []
    for text in generated_texts:
        if "Ingredients:" in text:
            ingredients = text.split("Ingredients:")[1]
            if "Instructions:" in ingredients:
                ingredients = ingredients.split("Instructions:")[0].strip()
            else:
                ingredients = ingredients.strip()
        else:
            ingredients = text
        is_safe = check_lcr(ingredients)
        reward = 1.0 if is_safe else 0.0
        rewards.append(reward)
    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
    return rewards, avg_reward

print("✓ Reward function ready", flush=True)

# ==================== RL TRAINER ====================
class RLTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generation_config = {
            "max_length": 512,
            "num_beams": 4,
            "early_stopping": True,
            "no_repeat_ngram_size": 2
        }
        # Override data collator to handle prompts
        self.data_collator = lambda data: {"prompt": [item["prompt"] for item in data]}
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        prompts = inputs["prompt"]
        
        prompt_inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(model.device)
        
        # Set model to eval for generation
        model.eval()
        with torch.no_grad():
            outputs = model.generate(
                **prompt_inputs,
                max_length=512,
                num_beams=4,
                early_stopping=True,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                output_scores=False,
                return_dict_in_generate=False
            )
        
        # Back to train mode
        model.train()
        
        generated_ids = outputs  # generate() returns tensor when return_dict_in_generate=False
        generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        # Debug: Print first generation at step 0
        if self.state.global_step == 0:
            print(f"\n[DEBUG] Sample prompt: {prompts[0][:80]}...", flush=True)
            print(f"[DEBUG] Sample generation: {generated_texts[0][:200]}...", flush=True)
            
            # Check full text like reward function does
            text = generated_texts[0]
            
            print(f"[DEBUG] Checking full recipe (ingredients + instructions)", flush=True)
            
            # Check what LCR finds in FULL text
            has_forbidden = any(has_keyword_match(text, kw) for kw in forbidden_keywords)
            has_required = any(has_keyword_match(text, kw) for kw in required_keywords)
            print(f"[DEBUG] LCR check: has_forbidden={has_forbidden}, has_required={has_required}", flush=True)
            print(f"[DEBUG] Should pass LCR: {not has_forbidden and has_required}\n", flush=True)
        
        rewards, avg_reward = compute_rewards(generated_texts)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=model.device)
        
        baseline = rewards_tensor.mean()
        advantages = rewards_tensor - baseline
        
        if advantages.std() > 0:
            advantages = advantages / (advantages.std() + 1e-8)
        
        # Re-encode for computing log probs
        with torch.enable_grad():
            encoder_outputs = model.get_encoder()(
                input_ids=prompt_inputs["input_ids"],
                attention_mask=prompt_inputs["attention_mask"]
            )
            
            decoder_outputs = model(
                encoder_outputs=encoder_outputs,
                decoder_input_ids=generated_ids[:, :-1],
                use_cache=False
            )
        
        logits = decoder_outputs.logits
        log_probs = F.log_softmax(logits, dim=-1)
        
        target_ids = generated_ids[:, 1:]
        gathered_log_probs = torch.gather(
            log_probs,
            dim=2,
            index=target_ids.unsqueeze(-1)
        ).squeeze(-1)
        
        padding_mask = (target_ids != self.tokenizer.pad_token_id).float()
        gathered_log_probs = gathered_log_probs * padding_mask
        
        sequence_log_probs = gathered_log_probs.sum(dim=1)
        loss = -(advantages * sequence_log_probs).mean()
        
        # Print progress
        print(f"Step {self.state.global_step}: loss={loss.item():.4f}, avg_reward={avg_reward:.3f}, baseline={baseline.item():.3f}", flush=True)
        
        self.log({
            "loss": loss.item(),
            "avg_reward": avg_reward,
            "baseline": baseline.item()
        })
        
        return (loss, {"loss": loss}) if return_outputs else loss

# ==================== MAIN ====================
def main():
    # Load Safe LoRA Full
    print("\n2. Loading Safe LoRA Full...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    base_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    
    safe_full_path = BASE_DIR / "final_models/safe_lora_full"
    if not safe_full_path.exists():
        print(f"❌ Safe LoRA Full not found: {safe_full_path}", flush=True)
        return
    
    model = PeftModel.from_pretrained(base_model, str(safe_full_path))
    
    print("   Merging LoRA weights...", flush=True)
    model = model.merge_and_unload()
    
    print("   Applying new LoRA for RL...", flush=True)
    rl_lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q", "v"]
    )
    model = get_peft_model(model, rl_lora_config)
    model.print_trainable_parameters()
    sys.stdout.flush()
    
    if torch.cuda.is_available():
        model = model.cuda()
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}", flush=True)
    
    # Load training data
    print("\n3. Loading training prompts...", flush=True)
    training_data_dir = BASE_DIR / "final_training_data"
    
    with open(training_data_dir / "safe_full_train_prompts.txt", 'r') as f:
        prompts = f.read().strip().split('\n')
    
    print(f"✓ Loaded {len(prompts)} prompts (will use all for training)", flush=True)
    
    rl_dataset = Dataset.from_dict({"prompt": prompts})
    print(f"✓ Loaded {len(prompts)} prompts", flush=True)
    
    # Training arguments
    print("\n4. Setting up RL training...", flush=True)
    training_args = TrainingArguments(
        output_dir=str(BASE_DIR / "final_models/safe_lora_full_rl"),
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=1e-5,
        logging_steps=5,
        logging_first_step=True,
        save_steps=50,
        save_total_limit=2,
        fp16=False,
        report_to="none",
        remove_unused_columns=False,
        disable_tqdm=False,
    )
    
    print(f"   Batch: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}", flush=True)
    print(f"   LR: {training_args.learning_rate}", flush=True)
    print(f"   Checkpoints every: {training_args.save_steps} steps", flush=True)
    
    # Initialize trainer with auto-push
    trainer = RLTrainer(
        model=model,
        args=training_args,
        train_dataset=rl_dataset,
        tokenizer=tokenizer,
        callbacks=[GitHubPushCallback(BASE_DIR, "safe_lora_full_rl")]
    )
    
    print("\n5. Training with RL...", flush=True)
    print("   Expected time: 45-60 minutes", flush=True)
    print("   Watch avg_reward increase from ~0.84!", flush=True)
    print("", flush=True)
    sys.stdout.flush()
    
    trainer.train()
    
    # Save final model
    print("\n6. Saving final model...", flush=True)
    model.save_pretrained(BASE_DIR / "final_models/safe_lora_full_rl")
    tokenizer.save_pretrained(BASE_DIR / "final_models/safe_lora_full_rl")
    
    # Final push
    print("\n7. Final push to GitHub...", flush=True)
    subprocess.run(["git", "pull", "--rebase"], cwd=BASE_DIR, capture_output=True)
    subprocess.run(["git", "add", "final_models/safe_lora_full_rl/"], cwd=BASE_DIR)
    subprocess.run(["git", "commit", "-m", "Final: Safe LoRA Full + RL complete"], cwd=BASE_DIR, capture_output=True)
    
    for attempt in range(3):
        result = subprocess.run(
            ["git", "push"],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print("✅ Final RL model pushed to GitHub!", flush=True)
            break
        elif "rejected" in result.stderr:
            print(f"⚠️ Push rejected (attempt {attempt+1}/3), pulling and retrying...", flush=True)
            subprocess.run(["git", "pull", "--rebase"], cwd=BASE_DIR, capture_output=True)
        else:
            print(f"⚠️ Push failed: {result.stderr[:200]}", flush=True)
            break
    
    print("\n" + "="*70, flush=True)
    print("✅ RL TRAINING COMPLETE", flush=True)
    print("="*70, flush=True)

if __name__ == "__main__":
    main()
