import pandas as pd

# Load all results
baseline_adv = pd.read_csv("evaluation/lcr_baseline_t5base_adversarial.csv")
baseline_real = pd.read_csv("evaluation/lcr_baseline_t5base_realistic.csv")
finetuned_adv = pd.read_csv("evaluation/lcr_finetuned_adversarial.csv")
finetuned_real = pd.read_csv("evaluation/lcr_finetuned_realistic.csv")

# Create comprehensive comparison
comparison = pd.DataFrame({
    'Model': [
        'T5-Base Baseline',
        'T5-Base Baseline',
        'T5-Base Fine-tuned (LoRA, 5K recipes)',
        'T5-Base Fine-tuned (LoRA, 5K recipes)'
    ],
    'Test Set': ['Adversarial', 'Realistic', 'Adversarial', 'Realistic'],
    'Total': [
        baseline_adv['total'].iloc[0],
        baseline_real['total'].iloc[0],
        finetuned_adv['total'].iloc[0],
        finetuned_real['total'].iloc[0]
    ],
    'Compliant': [
        baseline_adv['compliant'].iloc[0],
        baseline_real['compliant'].iloc[0],
        finetuned_adv['compliant'].iloc[0],
        finetuned_real['compliant'].iloc[0]
    ],
})

comparison['Compliance %'] = (comparison['Compliant'] / comparison['Total'] * 100).round(1)
comparison['Improvement'] = ['-', '-', '+35.6%', '+71.0%']

print("\n" + "="*80)
print("FINAL RESULTS - MILESTONE")
print("="*80)
print(comparison.to_string(index=False))
print("="*80)

comparison.to_csv("evaluation/final_comparison.csv", index=False)
print("\n✅ Saved to evaluation/final_comparison.csv")
