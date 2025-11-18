import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

PROMPT_TEMPLATE = "Generate a diabetes-friendly {dish} recipe."

def load_prompts(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def build_inputs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["prompt"] = df["dish"].apply(lambda d: PROMPT_TEMPLATE.format(dish=d))
    return df

def generate_recipes(
    df: pd.DataFrame,
    model_name: str = "google/flan-t5-large",
    max_new_tokens: int = 256,
    num_beams: int = 4,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    generations = []
    for prompt in df["prompt"]:
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
        )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generations.append(text)

    df_out = df.copy()
    df_out["generation"] = generations
    return df_out

def main():
    base_dir = Path(__file__).resolve().parents[1]
    prompts_path = base_dir / "data" / "prompts_dev.csv"
    out_path = base_dir / "outputs" / "baseline_generations.csv"

    df_prompts = load_prompts(prompts_path)
    df_inputs = build_inputs(df_prompts)

    df_gen = generate_recipes(df_inputs)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_gen.to_csv(out_path, index=False)
    print(f"Saved baseline generations to {out_path}")

if __name__ == "__main__":
    main()

