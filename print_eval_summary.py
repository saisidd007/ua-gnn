"""
Per-Horizon Evaluation Summary - DCRNN Protocol
Traffic Forecasting Model on PEMS-BAY Dataset
"""

import pandas as pd
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / 'results'

# Load existing per-horizon metrics
metrics_file = RESULTS / 'analysis_50epoch_per_horizon_metrics.csv'
df = pd.read_csv(metrics_file)

print("\n" + "=" * 80)
print("MULTI-HORIZON EVALUATION (DCRNN Protocol)")
print("PEMS-BAY Dataset | 50-Epoch Training | Full Test Set")
print("=" * 80)

# Key horizons: 3, 6, 12 steps (15, 30, 60 minutes)
# CSV is 1-indexed, so row 2 is horizon 3, row 5 is horizon 6, row 11 is horizon 12
key_horizons = [
    (3, "3-step (15 min)"),
    (6, "6-step (30 min)"),
    (12, "12-step (60 min)")
]

print(f"\n{'Horizon':<20} {'MAE':<15} {'RMSE':<15} {'MAPE':<15} {'Coverage95':<15}")
print("-" * 80)

results_summary = {}

for step, label in key_horizons:
    row = df[df['horizon'] == step].iloc[0]
    mae = row['mae']
    rmse = row['rmse']
    mape = row['mape']
    coverage = row['coverage95']
    
    results_summary[label] = {
        'step': step,
        'mae': float(mae),
        'rmse': float(rmse),
        'mape': float(mape),
        'coverage95': float(coverage)
    }
    
    print(f"{label:<20} {mae:<15.4f} {rmse:<15.4f} {mape:<15.2f} {coverage:<15.4f}")

print("-" * 80)

# Overall average across all 12 horizons
overall_mae = df['mae'].mean()
overall_rmse = df['rmse'].mean()
overall_mape = df['mape'].mean()
overall_coverage = df['coverage95'].mean()

print(f"{'OVERALL (All 12)':<20} {overall_mae:<15.4f} {overall_rmse:<15.4f} {overall_mape:<15.2f} {overall_coverage:<15.4f}")
print("=" * 80)

# Comparison with official baseline
print("\nCOMPARISON WITH OFFICIAL BASELINE:")
print("-" * 80)

official_baseline = {
    'mae': 0.4392,
    'rmse': 1.0327,
}

print(f"Official Baseline (50-epoch training):")
print(f"  MAE:  {official_baseline['mae']:.4f}")
print(f"  RMSE: {official_baseline['rmse']:.4f}")

print(f"\nCurrent Evaluation (All 12 horizons average):")
print(f"  MAE:  {overall_mae:.4f}")
print(f"  RMSE: {overall_rmse:.4f}")

mae_diff = overall_mae - official_baseline['mae']
rmse_diff = overall_rmse - official_baseline['rmse']
mae_pct = (mae_diff / official_baseline['mae']) * 100
rmse_pct = (rmse_diff / official_baseline['rmse']) * 100

print(f"\nDifference:")
print(f"  MAE:  {mae_diff:+.4f} ({mae_pct:+.2f}%)")
print(f"  RMSE: {rmse_diff:+.4f} ({rmse_pct:+.2f}%)")
print("-" * 80)

# Per-horizon comparison
print("\nPER-HORIZON COMPARISON WITH OFFICIAL BASELINE:")
print("-" * 80)
print(f"{'Horizon':<20} {'MAE Diff':<15} {'RMSE Diff':<15} {'Status':<20}")
print("-" * 80)

for step, label in key_horizons:
    mae = results_summary[label]['mae']
    rmse = results_summary[label]['rmse']
    
    mae_diff = mae - official_baseline['mae']
    rmse_diff = rmse - official_baseline['rmse']
    mae_pct = (mae_diff / official_baseline['mae']) * 100
    rmse_pct = (rmse_diff / official_baseline['rmse']) * 100
    
    status = "✓ Better" if mae < official_baseline['mae'] else "✗ Worse"
    
    mae_str = f"{mae_diff:+.4f} ({mae_pct:+.2f}%)"
    rmse_str = f"{rmse_diff:+.4f} ({rmse_pct:+.2f}%)"
    
    print(f"{label:<20} {mae_str:<15} {rmse_str:<15} {status:<20}")

print("-" * 80)

# Save summary as JSON
summary = {
    'official_baseline': official_baseline,
    'overall_evaluation': {
        'mae': float(overall_mae),
        'rmse': float(overall_rmse),
        'mape': float(overall_mape),
        'coverage95': float(overall_coverage)
    },
    'key_horizons': results_summary
}

summary_file = RESULTS / 'evaluation_summary.json'
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nSummary saved to: {summary_file}")
print("\n" + "=" * 80)

# Key findings
print("\nKEY FINDINGS:")
print("-" * 80)
print(f"✓ Model evaluated on {len(df)} prediction horizons (1-12 steps)")
print(f"✓ 3-step horizon (15 min):  MAE={results_summary['3-step (15 min)']['mae']:.4f}, RMSE={results_summary['3-step (15 min)']['rmse']:.4f}")
print(f"✓ 6-step horizon (30 min):  MAE={results_summary['6-step (30 min)']['mae']:.4f}, RMSE={results_summary['6-step (30 min)']['rmse']:.4f}")
print(f"✓ 12-step horizon (60 min): MAE={results_summary['12-step (60 min)']['mae']:.4f}, RMSE={results_summary['12-step (60 min)']['rmse']:.4f}")
print(f"✓ Overall average MAE:  {overall_mae:.4f}")
print(f"✓ Overall average RMSE: {overall_rmse:.4f}")
print(f"✓ 95% Prediction Interval Coverage: {overall_coverage:.4f}")
print("-" * 80)
print()
