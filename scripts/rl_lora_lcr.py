#!/usr/bin/env python3
"""
RL fine-tuning of LoRA adapters using LCR as a reward.

- Base model: google/flan-t5-base
- Starting checkpoint: models/lora_t5_base_5000_unfiltered
- Reward: lexical constraint respect (forbidden/required keywords)
- Prompts: data/prompts_from_safe.csv (column 'dish')

This is a lightweight REINFORCE-style loop:
1) sample with generate() (no grad),
2) compute LCR reward,
3) run a second forward pass with labels = sampled tokens,
   and weight the NLL loss by -reward.

We only update LoRA parameters.
"""

import json
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# ----------------------- config -----------------------

BASE_MODEL = "google/flan-t5-base"

# your existing LoRA checkpoint
LORA_INIT_DIR = "models/lora_t5_base_5000_unfiltered"

# where to save the RL-tuned adapter
LORA_OUT_DIR = "models/lora_t5_base_5000_unfiltered_rl_lcr"

PROMPTS_CSV = "data/prompts_from_safe.csv"   # must contain column 'dish'
PROMPT_TEMPLATE = "Generate a diabetes-friendly {dish} recipe."

KEYWORDS_JSON = "evaluation/keywords.json"

MAX_NEW_TOKENS = 128
BATCH_SIZE = 4
EPOCHS = 1           # keep small so it finishes
LR = 1e-5

# ------------------------------------------------------


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_keywords(path: Path):
    with open(path, "r") as f:
        data = json.load(f)
    forbidden = data.get("forbidden_groups", {})
    required = data.get("required_groups", {})
    return forbidden, required


def compute_lcr_reward(text: str, forbidden_groups, required_groups) -> float:
    """
    Simple shaping:
      +1.0 if text has at least one required and no forbidden terms
      -1.0 if it contains any forbidden term and no required
      -0.5 if it has both
       0.0 if neither
    """
    t = text.lower()

    has_forbidden = any(
        kw.lower() in t for group in forbidden_groups.values() for kw in group
    )
    has_required = any(
        kw.lower() in t for group in required_groups.values() for kw in group
    )

    if has_required and not has_forbidden:
        return 1.0
    if has_forbidden and not has_required:
        return -1.0
    if has_forbidden and has_required:
        return -0.5
    return 0.0


def build_prompts(base_dir: Path):
    df = pd.read_csv(base_dir / PROMPTS_CSV)
    if "dish" not in df.columns:
        raise ValueError(f"{PROMPTS_CSV} must contain a 'dish' column")
    prompts = [
        PROMPT_TEMPLATE.format(dish=d) for d in df["dish"].astype(str).tolist()
    ]
    return prompts[:100]  # cap at 100 for speed


def load_model_and_tokenizer(base_dir: Path, device: torch.device):
    print("Loading tokenizer and base model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)

    lora_path = base_dir / LORA_INIT_DIR
    print(f"Loading LoRA adapters from: {lora_path}")
    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA init dir not found: {lora_path}")

    # Load adapters *on top* of base model (do NOT merge)
    model = PeftModel.from_pretrained(base_model, str(lora_path))

    model.to(device)

    # Freeze everything, then unfreeze LoRA parameters
    for name, param in model.named_parameters():
        param.requires_grad = False
        if "lora_" in name:
            param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.4f}%)")

    if trainable_params == 0:
        raise RuntimeError("No trainable parameters found (LoRA layers not enabled).")

    return tokenizer, model


def rl_finetune_lora():
    base_dir = Path(__file__).resolve().parents[1]
    device = get_device()
    print(f"Using device: {device}")

    # 1. data & keywords
    prompts = build_prompts(base_dir)
    print(f"Loaded {len(prompts)} prompts for RL fine-tuning.")

    forbidden_groups, required_groups = load_keywords(base_dir / KEYWORDS_JSON)
    print(f"Loaded {sum(len(v) for v in forbidden_groups.values())} forbidden "
          f"and {sum(len(v) for v in required_groups.values())} required keywords.")

    # 2. model
    tokenizer, model = load_model_and_tokenizer(base_dir, device)

    # optimizer over ONLY trainable (LoRA) params
    lora_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(lora_params, lr=LR)

    model.train()
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    global_step = 0
    for epoch in range(EPOCHS):
        print(f"\n=== Epoch {epoch + 1}/{EPOCHS} ===")
        for start in range(0, len(prompts), BATCH_SIZE):
            batch_prompts = prompts[start : start + BATCH_SIZE]
            if not batch_prompts:
                continue

            # --- encode prompts (this part will be used both for sampling and training) ---
            enc = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)

            # ================== 1) SAMPLE WITH GENERATE (NO GRAD) ==================
            with torch.no_grad():
                gen_out = model.generate(
                    **enc,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    num_return_sequences=1,
                )

            sequences = gen_out  # [B, T_gen] for T5

            # --- compute rewards from decoded texts ---
            texts = tokenizer.batch_decode(sequences, skip_special_tokens=True)
            rewards_list = [
                compute_lcr_reward(t, forbidden_groups, required_groups) for t in texts
            ]
            rewards = torch.tensor(
                rewards_list, dtype=torch.float32, device=device
            )  # [B]

            # baseline/centering for stability
            rewards = rewards - rewards.mean()

            # ================== 2) TRAIN WITH REWARD-WEIGHTED NLL ==================
            # We teach the model to increase the probability of its own samples.
            # For T5, we set labels = generated tokens and let the model handle
            # shifting internally.
            labels = sequences.clone()

            # ignore padding in loss
            labels[labels == tokenizer.pad_token_id] = -100

            # forward WITH grad
            outputs = model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                labels=labels,
            )

            logits = outputs.logits  # [B, T_dec, V]
            B, T_dec, V = logits.shape

            # Flatten for token-level CE
            token_loss = loss_fct(
                logits.view(-1, V),
                labels.view(-1),
            )  # [B*T]
            token_loss = token_loss.view(B, T_dec)

            # Mask out ignored tokens (-100)
            mask = labels.ne(-100)
            token_loss = token_loss * mask

            # Per-sequence NLL (average over tokens that are not ignored)
            denom = mask.sum(dim=1).clamp(min=1)
            seq_nll = token_loss.sum(dim=1) / denom  # [B]

            # RL-style objective: minimise (-reward * logprob) ≈ (-reward * -NLL)
            # => loss = seq_nll * (-rewards)
            loss = (seq_nll * (-rewards)).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            optimizer.step()

            global_step += 1

            if global_step % 5 == 0:
                avg_reward = rewards.mean().item()
                print(
                    f"step {global_step:03d} | loss {loss.item():.4f} | "
                    f"mean centered reward {avg_reward:.4f}"
                )

    # save RL-tuned adapters
    out_dir = base_dir / LORA_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving RL-tuned LoRA adapters to: {out_dir}")
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print("✅ RL fine-tuning complete.")


if __name__ == "__main__":
    rl_finetune_lora()

