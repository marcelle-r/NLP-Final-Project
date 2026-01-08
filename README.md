## Project Overview
This project studies the reliability of large language models in a safety-critical setting. We treat **diabetes-safe recipe generation** as a **constrained conditional generation** task and evaluate how **data curation** and **training scale** affect safety compliance.

## Technical Highlights
- **Constraint framework:** Keyword-based system derived from ADA-guided rules with **267 forbidden** and **184 required** keywords.
- **Safety metric:** Implemented **Lexical Constraint Respect (LCR)** to measure strict dietary compliance.
- **PEFT fine-tuning:** Applied **LoRA/PEFT** to **FLAN-T5-base**, reducing trainable parameters to **0.36%** (rank **r=16**).
- **Ablation study:** Trained and evaluated multiple models across dataset sizes from **1K to ~195K** recipes.

## Key Results
- **Data quality > scale:** A **filtered 5K** training set improved LCR by **+24.9 percentage points** over an **unfiltered 5K** baseline.
- **Diminishing returns:** Scaling from **10K → ~195K** yielded only **+1.5 pp** additional LCR improvement.
- **RL instability (negative result):** Binary-reward **REINFORCE** caused a sharp compliance drop (**84.2% → 31.3% LCR**), highlighting instability in naive reward design for constraint satisfaction.

## Tech Stack
- **Model:** FLAN-T5-base
- **Frameworks:** PyTorch; Hugging Face (Transformers, PEFT)
- **Metrics:** LCR, ROUGE-1/2/L, BERTScore
- **Compute:** NVIDIA T4 GPU

