#!/usr/bin/env python
"""
Sensor Dropout Robustness Analysis
Applies progressive sensor dropout (0%, 5%, 10%, 20%, 30%, 40%, 50%)
and measures performance degradation across metrics
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from src.models.enhanced_gnn import create_improved_model
from src.utils.enhanced_dataset import create_enhanced_dataset


class MetricsCalculator:
    """Calculate metrics"""
    
    @staticmethod
    def calculate_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Calculate MAE, RMSE, MAPE, R²"""
        mae = np.mean(np.abs(predictions - targets))
        rmse = np.sqrt(np.mean((predictions - targets) ** 2))
        mape = np.mean(np.abs((targets - predictions) / (np.abs(targets) + 1e-8))) * 100
        
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape),
            'r2_score': float(r2_score)
        }


class SensorDropoutRobustness:
    """Test robustness with progressive sensor dropout"""
    
    def __init__(self, model, test_loader, device):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.metrics_calc = MetricsCalculator()
    
    def apply_sensor_dropout(self, features: torch.Tensor, missing_mask: torch.Tensor,
                             dropout_rate: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply progressive sensor dropout"""
        features_out = features.clone()
        mask_out = missing_mask.clone()
        
        if dropout_rate > 0:
            B, N, T, C = features.shape
            
            for b in range(B):
                # Find currently valid sensors
                valid_sensors = []
                for n in range(N):
                    if torch.sum(missing_mask[b, n, :]) > 0:
                        valid_sensors.append(n)
                
                if len(valid_sensors) > 0:
                    num_dropout = max(1, int(len(valid_sensors) * dropout_rate / 100))
                    dropout_indices = np.random.choice(
                        valid_sensors, 
                        size=min(num_dropout, len(valid_sensors)),
                        replace=False
                    )
                    
                    for n in dropout_indices:
                        features_out[b, n, :, :] = 0
                        mask_out[b, n, :] = 0
        
        return features_out, mask_out
    
    def test_dropout_rate(self, dropout_rate: float) -> Dict[str, float]:
        """Test model at specific dropout rate"""
        self.model.eval()
        all_preds = []
        all_targets = []
        
        print(f"\n  Testing {dropout_rate:.0f}% sensor dropout...")
        pbar = tqdm(self.test_loader, desc=f'Dropout {dropout_rate:.0f}%', disable=False)
        
        with torch.no_grad():
            for batch in pbar:
                batch = batch.to(self.device)
                
                # Apply dropout to input
                x_dropped, mask_dropped = self.apply_sensor_dropout(
                    batch.x, batch.missing_mask, dropout_rate
                )
                
                # MC forward pass (K=10 samples)
                mc_preds = []
                for k in range(10):
                    self.model.train()
                    preds, _, _ = self.model(
                        x_dropped,
                        batch.edge_index,
                        mask_dropped,
                        return_uncertainty=True
                    )
                    mc_preds.append(preds)
                
                mc_preds = torch.stack(mc_preds, dim=0)
                pred_mean = mc_preds.mean(dim=0)
                
                all_preds.append(pred_mean.cpu().numpy())
                all_targets.append(batch.y.cpu().numpy())
        
        preds = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        return self.metrics_calc.calculate_metrics(preds, targets)
    
    def run_robustness_analysis(self, dropout_rates: List[float]) -> pd.DataFrame:
        """Run robustness analysis for all dropout rates"""
        print("\n[ANALYSIS] Running sensor dropout robustness analysis...")
        
        results = []
        baseline_metrics = None
        
        for rate in dropout_rates:
            metrics = self.test_dropout_rate(rate)
            
            row = {
                'Dropout_Rate': f'{rate:.0f}%',
                'MAE': metrics['mae'],
                'RMSE': metrics['rmse'],
                'MAPE': metrics['mape'],
                'R2': metrics['r2_score']
            }
            
            # Calculate degradation
            if baseline_metrics is None:
                baseline_metrics = metrics
                row['MAE_Degradation'] = 0.0
                row['RMSE_Degradation'] = 0.0
                row['MAPE_Degradation'] = 0.0
            else:
                row['MAE_Degradation'] = 100 * (metrics['mae'] - baseline_metrics['mae']) / baseline_metrics['mae']
                row['RMSE_Degradation'] = 100 * (metrics['rmse'] - baseline_metrics['rmse']) / baseline_metrics['rmse']
                row['MAPE_Degradation'] = 100 * (metrics['mape'] - baseline_metrics['mape']) / baseline_metrics['mape']
            
            results.append(row)
        
        return pd.DataFrame(results)


def create_robustness_plots(results_df: pd.DataFrame):
    """Create visualization of robustness analysis"""
    print("\n[PLOTS] Creating robustness visualization...")
    
    dropout_rates = [float(x.rstrip('%')) for x in results_df['Dropout_Rate']]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Sensor Dropout Robustness Analysis', fontsize=16, fontweight='bold')
    
    # MAE vs Dropout
    axes[0, 0].plot(dropout_rates, results_df['MAE'], 'o-', linewidth=2, markersize=8, color='#2E86AB')
    axes[0, 0].set_xlabel('Sensor Dropout Rate (%)')
    axes[0, 0].set_ylabel('MAE')
    axes[0, 0].set_title('Mean Absolute Error')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xticks(dropout_rates)
    
    # RMSE vs Dropout
    axes[0, 1].plot(dropout_rates, results_df['RMSE'], 'o-', linewidth=2, markersize=8, color='#A23B72')
    axes[0, 1].set_xlabel('Sensor Dropout Rate (%)')
    axes[0, 1].set_ylabel('RMSE')
    axes[0, 1].set_title('Root Mean Squared Error')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xticks(dropout_rates)
    
    # MAPE vs Dropout
    axes[1, 0].plot(dropout_rates, results_df['MAPE'], 'o-', linewidth=2, markersize=8, color='#F18F01')
    axes[1, 0].set_xlabel('Sensor Dropout Rate (%)')
    axes[1, 0].set_ylabel('MAPE (%)')
    axes[1, 0].set_title('Mean Absolute Percentage Error')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xticks(dropout_rates)
    
    # R² vs Dropout
    axes[1, 1].plot(dropout_rates, results_df['R2'], 'o-', linewidth=2, markersize=8, color='#C73E1D')
    axes[1, 1].set_xlabel('Sensor Dropout Rate (%)')
    axes[1, 1].set_ylabel('R² Score')
    axes[1, 1].set_title('Coefficient of Determination')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xticks(dropout_rates)
    
    plt.tight_layout()
    plot_file = 'results/sensor_dropout_robustness.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"[SAVED] Plot: {plot_file}")
    plt.close()


def main():
    """Main robustness analysis"""
    print("\n" + "="*100)
    print("[SENSOR DROPOUT] Robustness Analysis with Progressive Sensor Failure")
    print("="*100)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[DEVICE] Using: {device}")
    
    # Load dataset
    print("[DATA] Loading test dataset...")
    dataset = create_enhanced_dataset(
        root_dir='data',
        sequence_length=12,
        prediction_length=12,
        preprocessing_method='robust'
    )
    
    test_data = dataset.get_test_data()
    test_loader = DataLoader(test_data, batch_size=8, shuffle=False, num_workers=0)
    print(f"[STATS] Test samples: {len(test_data)}")
    
    # Load best model
    print("[MODEL] Loading best trained model...")
    model = create_improved_model(
        in_channels=12,
        hidden_channels=64,
        out_channels=12,
        num_gnn_layers=4,
        num_temporal_layers=3,
        num_attention_heads=4,
    ).to(device)
    
    checkpoint = torch.load('results/enhanced_best_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"[LOADED] Best model from epoch {checkpoint['epoch'] + 1}")
    
    # Run robustness analysis
    analyzer = SensorDropoutRobustness(model, test_loader, device)
    dropout_rates = [0, 5, 10, 20, 30, 40, 50]
    results_df = analyzer.run_robustness_analysis(dropout_rates)
    
    # Print results
    print("\n" + "="*100)
    print("[RESULTS] SENSOR DROPOUT ROBUSTNESS")
    print("="*100)
    print(results_df.to_string(index=False))
    
    # Create plots
    create_robustness_plots(results_df)
    
    # Save results
    csv_file = 'results/sensor_dropout_robustness.csv'
    results_df.to_csv(csv_file, index=False)
    print(f"\n[SAVED] CSV file: {csv_file}")
    
    json_file = 'results/sensor_dropout_robustness.json'
    with open(json_file, 'w') as f:
        json.dump(results_df.to_dict(orient='records'), f, indent=2)
    print(f"[SAVED] JSON file: {json_file}")
    
    print("="*100 + "\n")
    
    return results_df


if __name__ == "__main__":
    main()
