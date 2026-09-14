#!/usr/bin/env python
"""Move METR-LA related files to metrla subfolder"""

import shutil
import os
from pathlib import Path

results_folder = Path(r'C:\Users\rockk\OneDrive\Desktop\traffic-flow-gnn\results')
metrla_folder = results_folder / 'metrla'

metrla_files = [
    'analysis_50epoch_metrics.png',
    'analysis_50epoch_per_horizon_metrics.csv',
    'analysis_50epoch_test_summary.csv',
    'analysis_50epoch_train_val_loss.png',
    'analysis_50epoch_train_val_loss.eps',
    'analysis_50epoch_train_val_loss(2).png',
    'analysis_50epoch_train_val_loss_no_title.eps',
    'analysis_50epoch_train_val_loss_no_title.png',
    'best_model_metrla.pt',
    'enhanced_best_model.pt',
    'enhanced_checkpoint_epoch_10.pt',
    'enhanced_checkpoint_epoch_20.pt',
    'enhanced_checkpoint_epoch_30.pt',
    'enhanced_checkpoint_epoch_40.pt',
    'enhanced_checkpoint_epoch_50.pt',
    'enhanced_training_history.png',
    'enhanced_training_results_20251115_000912.json',
    'enhanced_training_results_20251118_061800.json',
    'enhanced_training_results_20251119_054803.json',
    'enhanced_training_results_20251224_114956.json',
    'enhanced_training_results_metrla_20260502_125606.json',
    'evaluation_full_test_set.json',
    'evaluation_summary.json',
    'metrla_dropout_comparison.png',
    'metrla_mae_trend_50epoch.png',
    'metrla_r2_trend_50epoch.png',
    'metrla_rmse_trend_50epoch.png',
    'metrla_sensor_dropout_analysis.png',
    'metrla_sensor_dropout_results.json',
    'metrla_training_analysis.png',
    'final_evaluation_table.csv',
    'calibration_curve.png',
    'reliability_diagram.png',
    'sharpness_analysis.png'
]

print(f"Moving files to: {metrla_folder}")
print(f"Source folder: {results_folder}")

moved_count = 0
failed_count = 0

for file in metrla_files:
    src = results_folder / file
    dst = metrla_folder / file
    
    if src.exists():
        try:
            shutil.move(str(src), str(dst))
            moved_count += 1
            print(f"✓ Moved: {file}")
        except Exception as e:
            failed_count += 1
            print(f"✗ Failed to move {file}: {e}")
    else:
        print(f"- Not found: {file}")

print(f"\n{'='*50}")
print(f"Summary:")
print(f"  Moved: {moved_count} files")
print(f"  Failed: {failed_count} files")
print(f"{'='*50}")
