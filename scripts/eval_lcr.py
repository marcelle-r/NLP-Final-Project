import pandas as pd
import json
from pathlib import Path

# === Paths ===
BASE_DIR = Path(__file__).resolve().parents[1]

GENERATIONS_PATH = BASE_DIR / "outputs" / "baseline_generations.csv"
KEYWORDS_PATH = BASE_DIR / "evaluation" / "keywords.json"

# OUTPUTS (we want both)
SCORED_OUT_PATH = BASE_DIR / "outputs" / "baseline_generations_scored.csv"
SUMMARY_OUT_PATH = BASE_DIR / "evaluation" / "lcr_baseline.csv"


def load_keywords(path):
    with open(path, "r") as f:
        data = json.load(f)
    forbidden = data.get("forbidden_groups", {})
    required = data.get("required_groups", {})
    return forbidden, required


def normalize(text: str) -> str:
    return text.lower()


def main():
    print(f"Loading generations from: {GENERATIONS_PATH}")
    df = pd.read_csv(GENERATIONS_PATH)

    if "generation" not in df.columns:
        raise ValueError("Expected a 'generation' column in baseline_generations.csv")

    forbidden_groups, required_groups = load_keywords(KEYWORDS_PATH)
    print(f"Loaded {sum(len(v) for v in forbidden_groups.values())} forbidden keywords "
          f"and {sum(len(v) for v in required_groups.values())} required keywords.")

    has_forbidden = []
    has_required = []
    compliant = []

    for _, row in df.iterrows():
        text = normalize(str(row["generation"]))

        f_flag = False
        r_flag = False

        # Check forbidden
        for group, kws in forbidden_groups.items():
            if any(kw.lower() in text for kw in kws):
                f_flag = True
                break

        # Check required
        for group, kws in required_groups.items():
            if any(kw.lower() in text for kw in kws):
                r_flag = True
                break

        has_forbidden.append(int(f_flag))
        has_required.append(int(r_flag))
        compliant.append(int((not f_flag) and r_flag))

    # === WRITE FULL SCORED OUTPUT ===
    df_scored = df.copy()
    df_scored["has_forbidden"] = has_forbidden
    df_scored["has_required"] = has_required
    df_scored["compliant"] = compliant

    df_scored.to_csv(SCORED_OUT_PATH, index=False)
    print(f"✔ Full scored file written to: {SCORED_OUT_PATH}")

    # === WRITE SUMMARY OUTPUT ===
    total = len(df_scored)
    summary = {
        "total": total,
        "no_forbidden": sum(1 - f for f in has_forbidden),
        "has_required": sum(has_required),
        "compliant": sum(compliant),
    }

    df_summary = pd.DataFrame([summary])
    df_summary.to_csv(SUMMARY_OUT_PATH, index=False)
    print(f"✔ Summary saved to: {SUMMARY_OUT_PATH}")

    print("\n=== Summary ===")
    print(summary)


if __name__ == "__main__":
    main()
