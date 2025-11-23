import pandas as pd
import json
from pathlib import Path

def evaluate_generations(generations_file, output_prefix):
    base_dir = Path(__file__).resolve().parents[1]
    gen_path = base_dir / "outputs" / generations_file
    keywords_path = base_dir / "evaluation" / "keywords.json"
    scored_path = base_dir / "outputs" / f"{output_prefix}_scored.csv"
    summary_path = base_dir / "evaluation" / f"lcr_{output_prefix}.csv"
    
    print(f"Evaluating: {gen_path}")
    df = pd.read_csv(gen_path)
    
    with open(keywords_path) as f:
        keywords = json.load(f)
    
    forbidden = keywords["forbidden_groups"]
    required = keywords["required_groups"]
    
    has_forbidden, has_required, compliant = [], [], []
    
    for _, row in df.iterrows():
        text = str(row["generation"]).lower()
        
        f_flag = any(any(kw.lower() in text for kw in kws) for kws in forbidden.values())
        r_flag = any(any(kw.lower() in text for kw in kws) for kws in required.values())
        
        has_forbidden.append(int(f_flag))
        has_required.append(int(r_flag))
        compliant.append(int((not f_flag) and r_flag))
    
    df["has_forbidden"] = has_forbidden
    df["has_required"] = has_required
    df["compliant"] = compliant
    df.to_csv(scored_path, index=False)
    
    summary = {
        "total": len(df),
        "no_forbidden": sum(1 - f for f in has_forbidden),
        "has_required": sum(has_required),
        "compliant": sum(compliant),
    }
    
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    
    print(f"✅ Compliant: {summary['compliant']}/{summary['total']} ({summary['compliant']/summary['total']*100:.1f}%)")
    return summary

if __name__ == "__main__":
    print("="*60)
    print("EVALUATING FINE-TUNED: ADVERSARIAL")
    print("="*60)
    evaluate_generations("finetuned_adversarial.csv", "finetuned_adversarial")
    
    print("\n" + "="*60)
    print("EVALUATING FINE-TUNED: REALISTIC")
    print("="*60)
    evaluate_generations("finetuned_realistic.csv", "finetuned_realistic")
