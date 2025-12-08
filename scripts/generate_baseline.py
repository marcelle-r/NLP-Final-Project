#!/usr/bin/env python3

import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# -------------------- Prompt builders for each baseline type -------------------- #

def make_prompt_prompt_only(dish: str) -> str:
    """
    Baseline 1: prompt_only
    Simple instruction to generate a diabetes-friendly recipe.
    """
    return (
        "You are a recipe generator.\n\n"
        f"Generate a diabetes-friendly recipe for: {dish}.\n"
        "Include a clear list of ingredients and numbered steps.\n"
    )


def make_prompt_recipe_examples(dish: str) -> str:
    """
    Baseline 2: recipe_examples_prompt
    Example-guided (few-shot) baseline with two short diabetes-friendly recipes.
    """

    example_1 = """Example 1:
Name: Low-Carb Lemon Cheesecake
Recipe:
Ingredients:
- almond flour
- butter
- cream cheese
- eggs
- stevia
Steps:
1. Mix almond flour with melted butter and press into a pan.
2. Beat cream cheese with stevia and eggs.
3. Pour filling over crust and bake until set.
"""

    example_2 = """Example 2:
Name: High-Protein Veggie Omelette
Recipe:
Ingredients:
- egg whites
- spinach
- mushrooms
- low-fat cottage cheese
- olive oil
Steps:
1. Sauté spinach and mushrooms in olive oil.
2. Add beaten egg whites and cook until almost set.
3. Add cottage cheese, fold, and finish cooking.
"""

    return (
        "You are a recipe generator that specializes in diabetes-friendly recipes.\n\n"
        "Here are example recipes:\n\n"
        f"{example_1}\n"
        f"{example_2}\n"
        f"Now generate a diabetes-friendly recipe for: {dish}.\n"
        "Include a clear list of ingredients and numbered steps.\n"
    )


def make_prompt_keyword_guided(dish: str) -> str:
    """
    Baseline 3: keyword_guided
    Explicitly instructs the model to avoid forbidden categories and prefer required ones.
    This ties directly to keywords.json and TA feedback about keyword-aware baselines.
    """
    return (
        f"Generate a diabetes-friendly recipe for: {dish}.\n\n"
        "You must:\n"
        "- Avoid ingredients in these categories: sugar, syrups, refined flours, "
        "sweetened dairy, sweet drinks, refined carbs, high-GI starches, processed desserts, "
        "fried foods, processed meats, dessert spreads, packaged snacks, prepackaged desserts, "
        "ultra-processed ingredients, cream-based sauces.\n"
        "- Prefer ingredients in these categories: non-nutritive sweeteners, low-carb flours, "
        "high-fiber additions, whole grains, fiber-rich veggies, lean proteins, healthy oils.\n\n"
        "Include a clear list of ingredients and numbered steps.\n"
    )


def build_prompt(dish: str, baseline_type: str) -> str:
    """
    Route to the correct prompt builder based on baseline_type.
    """
    if baseline_type == "prompt_only":
        return make_prompt_prompt_only(dish)
    elif baseline_type == "recipe_examples_prompt":
        return make_prompt_recipe_examples(dish)
    elif baseline_type == "keyword_guided":
        return make_prompt_keyword_guided(dish)
    else:
        raise ValueError(f"Unknown baseline_type: {baseline_type}")


# -------------------- Core generation logic -------------------- #

def load_prompts(path: Path) -> pd.DataFrame:
    """
    Expects a CSV with at least a 'dish' column.
    """
    return pd.read_csv(path)


def generate_recipes(
    df: pd.DataFrame,
    baseline_types=None,
    model_name: str = "google/flan-t5-large",
    max_new_tokens: int = 256,
    num_beams: int = 4,
) -> pd.DataFrame:
    """
    For each dish and each baseline_type, build a prompt and generate a recipe.

    Returns a long DataFrame with one row per (dish, baseline_type):
      columns: dish, baseline_type, prompt, generation
    """
    if baseline_types is None:
        baseline_types = [
            "prompt_only",
            "recipe_examples_prompt",
            "keyword_guided",
        ]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    rows = []

    for _, row in df.iterrows():
        dish = str(row["dish"])
        for baseline_type in baseline_types:
            prompt = build_prompt(dish, baseline_type)

            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=False,
            )
            text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            rows.append(
                {
                    "dish": dish,
                    "baseline_type": baseline_type,
                    "prompt": prompt,
                    "generation": text,
                }
            )

    return pd.DataFrame(rows)


def main():
    base_dir = Path(__file__).resolve().parents[1]
    prompts_path = base_dir / "data" / "prompts_dev.csv"
    out_path = base_dir / "outputs" / "baseline_generations.csv"

    df_prompts = load_prompts(prompts_path)

    df_gen = generate_recipes(df_prompts)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_gen.to_csv(out_path, index=False)
    print(
        f"Saved baseline generations to {out_path} "
        f"({len(df_gen)} rows, {df_gen['baseline_type'].nunique()} baselines)."
    )


if __name__ == "__main__":
    main()
