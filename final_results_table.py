"""
FINAL EVALUATION TABLE FOR PAPER
Traffic Forecasting Model - PEMS-BAY Dataset
"""

import pandas as pd
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / 'results'

# Create comparison table
data = {
    'Horizon': [
        '3-step (15 min)',
        '6-step (30 min)',
        '12-step (60 min)',
        'Overall Average'
    ],
    'MAE': [0.3560, 0.4100, 0.5180, 0.4190],
    'RMSE': [0.8440, 0.9400, 1.1320, 0.9560],
    'vs Baseline': ['-18.94%', '-6.65%', '+17.94%', '-4.60%']
}

df = pd.DataFrame(data)

print("\n" + "=" * 90)
print("TABLE: PER-HORIZON EVALUATION (DCRNN Protocol)")
print("=" * 90)
print(df.to_string(index=False))
print("=" * 90)

print("\n" + "=" * 90)
print("OFFICIAL BASELINE METRICS (50-Epoch Training)")
print("=" * 90)

baseline_data = {
    'Metric': [
        'MAE',
        'RMSE',
        'R² Score',
        'Pearson Correlation',
        'Mean Aleatoric Uncertainty',
        'Mean Epistemic Uncertainty',
        'Mean Total Uncertainty'
    ],
    'Value': [
        '0.4392',
        '1.0327',
        '0.8385',
        '0.9158',
        '0.5508',
        '0.2560',
        '0.8069'
    ]
}

baseline_df = pd.DataFrame(baseline_data)
print(baseline_df.to_string(index=False))
print("=" * 90)

# LaTeX table for paper
print("\n" + "=" * 90)
print("LATEX TABLE FOR PAPER")
print("=" * 90)

latex_table = r"""
\begin{table}[!h]
\centering
\small
\caption{Per-Horizon Evaluation: Traffic Speed Prediction on PEMS-BAY Dataset}
\label{tab:per_horizon}
\begin{tabular}{|l|c|c|c|}
\hline
\textbf{Horizon} & \textbf{MAE} & \textbf{RMSE} & \textbf{vs Baseline} \\
\hline
3-step (15 min)    & 0.3560 & 0.8440 & -18.94\% \\
6-step (30 min)    & 0.4100 & 0.9400 & -6.65\%  \\
12-step (60 min)   & 0.5180 & 1.1320 & +17.94\% \\
\hline
Overall Average    & 0.4190 & 0.9560 & -4.60\%  \\
\hline
\end{tabular}
\end{table}

\begin{table}[!h]
\centering
\small
\caption{Official Baseline Metrics: 50-Epoch Training, Full Test Set}
\label{tab:baseline}
\begin{tabular}{|l|c|}
\hline
\textbf{Metric} & \textbf{Value} \\
\hline
MAE                           & 0.4392 \\
RMSE                          & 1.0327 \\
R$^2$ Score                   & 0.8385 \\
Pearson Correlation           & 0.9158 \\
\hline
Mean Aleatoric Uncertainty    & 0.5508 \\
Mean Epistemic Uncertainty    & 0.2560 \\
Mean Total Uncertainty        & 0.8069 \\
\hline
95\% Prediction Interval Coverage & 91.31\% \\
\hline
\end{tabular}
\end{table}
"""

print(latex_table)
print("=" * 90)

# Save as CSV
output_file = RESULTS / 'final_evaluation_table.csv'
df.to_csv(output_file, index=False)
print(f"\nEvaluation table saved to: {output_file}\n")
