import pandas as pd
from pathlib import Path

def main():
    base_dir = Path(__file__).resolve().parents[1]

    baseline_adv = pd.read_csv(base_dir / "evaluation" / "lcr_baseline.csv")
    baseline_real = pd.read_csv(base_dir / "evaluation" / "lcr_baseline_realistic.csv")

    comparison = pd.DataFrame({
        'Test Set': ['Adversarial (prompts_dev)', 'Realistic (prompts_from_safe)'],
        'Total': [baseline_adv['total'].iloc[0], baseline_real['total'].iloc[0]],
        'No Forbidden': [baseline_adv['no_forbidden'].iloc[0], baseline_real['no_forbidden'].iloc[0]],
        'Has Required': [baseline_adv['has_required'].iloc[0], baseline_real['has_required'].iloc[0]],
        'Compliant': [baseline_adv['compliant'].iloc[0], baseline_real['compliant'].iloc[0]],
    })

    comparison['Compliance %'] = (comparison['Compliant'] / comparison['Total'] * 100).round(1)

    print("\n=== BASELINE COMPARISON ===")
    print(comparison.to_string(index=False))

    out_path = base_dir / "evaluation" / "baseline_comparison.csv"
    comparison.to_csv(out_path, index=False)
    print(f"\n✅ Saved to {out_path}")

if __name__ == "__main__":
    main()
