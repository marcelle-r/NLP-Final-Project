#!/usr/bin/env python3

"""
Reinforcement Learning Fine-Tuning using PPO
Reward = LCR (1 if compliant, 0 else)
"""

import torch
import pandas as pd
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from trl import PPOTrainer, PPOConfig

import json

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_NAME = "google/flan-t5-base"
LORA_PATH = BASE_DIR / "models" / "lora_t5_base_5000"
PROMPTS_PATH = BASE_DIR / "data" / "prompts_from_safe.csv"
KEYWORDS_PATH = BASE_DIR / "evaluation" / "keywords.json"
SAVE_DIR = BASE_DIR / "models" / "lora_t5_base_5000_rl"


# ---------------- reward (LCR) ----------------

def load_keywords():
    with open(KEYWORDS_PATH, "r") as f:
        data = json.load(f)
    return data["forbidden_groups"], data["required_groups"]


forbidden_groups, required_groups = load_keywords()

def compute_lcr_reward(text: str):
    text = text.lower()

    has_forbidden = any(
        any(kw in text for kw in kws)
        for kws in forbidden_groups.values()
    )
    has_required = any(
        any(kw in text for kw in kws)
        for kws in required_groups.values()
    )
    compliant = (not has_forbidden) and has_required
    return 1.0 if compliant else 0.0


# ---------------- PPO training ----------------

def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    df = pd.read_csv(PROMPTS_PATH)
    prompts = [
        f"Generate a diabetes-friendly {d} recipe."
        for d in df["dish"].tolist()
    ]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    print(f"Loading LoRA adapter: {LORA_PATH}")
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.to(device)

    config = PPOConfig(
        batch_size=4,
        forward_batch_size=1,
        learning_rate=1e-5,
        mini_batch_size=4,
        optimize_cuda_cache=False,
        ppo_epochs=4
    )

    ppo_trainer = PPOTrainer(
        config=config,
        model=model,
        tokenizer=tokenizer
    )

    print("🔁 Starting PPO Reinforcement Learning (LCR Reward)...")

    for epoch in range(3):
       

