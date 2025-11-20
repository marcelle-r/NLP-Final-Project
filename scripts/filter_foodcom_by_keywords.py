import ast
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict

def load_keywords(path: Path) -> Dict:
    with open(path, "r") as f:
        return json.load(f)

def expand_groups(groups: Dict[str, List[str]]) -> List[str]:
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

def list_from_string(s):
    # ingredients in RAW_recipes look like "['sugar', 'flour']"
    if isinstance(s, list):
        return s
    try:
        return ast.literal_eval(s)
    except Exception:
        return []

def has_any(ingredients: List[str], terms: List[str]) -> bool:
    joined = " ".join(ingredients).lower()
    return any(term in joined for term in terms)

def main():
    base_dir = Path(__file__).resolve().parents[1]
    data_path = base_dir / "data" / "raw" / "RAW_recipes.csv"
    keywords_path = base_dir / "evaluation" / "keywords.json"

    df = pd.read_csv(data_path)
    keywords = load_keywords(keywords_path)

    forbidden_terms = expand_groups(keywords["forbidden_groups"])
    required_terms = expand_groups(keywords["required_groups"])

    df["ingredients_list"] = df["ingredients"].apply(list_from_string)

    def label_safe(row):
        ings = row["ingredients_list"]
        f_hit = has_any(ings, forbidden_terms)
        r_hit = has_any(ings, required_terms)
        return int((not f_hit) and r_hit)

    df["diabetes_safe"] = df.apply(label_safe, axis=1)

    safe = df[df["diabetes_safe"] == 1]
    unsafe = df[df["diabetes_safe"] == 0]

    out_dir = base_dir / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    safe.to_csv(out_dir / "foodcom_safe_recipes.csv", index=False)
    unsafe.to_csv(out_dir / "foodcom_unsafe_recipes.csv", index=False)

    print(f"Safe recipes: {len(safe)}")
    print(f"Unsafe recipes: {len(unsafe)}")

if __name__ == "__main__":
    main()

