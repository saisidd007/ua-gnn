#!/usr/bin/env python
"""Evaluate best Bi-LSTM model on test set and compute all metrics with uncertainties"""

import json
import torch
import numpy as np
from src.models.enhanced_gnn import create_improved_model
from src.utils.enhanced_dataset import create_enhanced_dataset
from torch_geometric.loader import DataLoader

# Load best model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load('BI-LSTM_PEMSBAY/enhanced_best_model.pt', map_location=device)
model = create_improved_model(in_channels=12, hidden_channels=64, out_channels=12, num_gnn_layers=4, num_temporal_layers=3, num_attention_heads=4).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load dataset
dataset = create_enhanced_dataset(root_dir='data', sequence_length=12, prediction_length=12, preprocessing_method='robust')
test_data = dataset.get_test_data()
test_loader = DataLoader(test_data, batch_size=8, shuffle=False, num_workers=0)

# Evaluate on test set
all_preds = []
all_targets = []
all_alea_vars = []
all_epi_vars = []

K = 10  # MC samples for uncertainty

for batch in test_loader:
    batch = batch.to(device)
    
    mc_preds = []
    mc_aleas = []
    
    for k in range(K):
        model.train()
        with torch.no_grad():
            preds_k, alea_k, _ = model(batch.x, batch.edge_index, batch.missing_mask, return_uncertainty=True)
        mc_preds.append(preds_k)
        mc_aleas.append(alea_k)
    
    mc_preds = torch.stack(mc_preds, dim=0)
    mc_aleas = torch.stack(mc_aleas, dim=0)
    
    mean_pred = mc_preds.mean(dim=0)
    epi_var = mc_preds.var(dim=0)
    alea_mean = mc_aleas.mean(dim=0)
    
    all_preds.append(mean_pred.cpu().numpy())
    all_targets.append(batch.y.cpu().numpy())
    all_alea_vars.append(alea_mean.cpu().numpy())
    all_epi_vars.append(epi_var.cpu().numpy())

preds_concat = np.concatenate(all_preds, axis=0).reshape(-1)
targets_concat = np.concatenate(all_targets, axis=0).reshape(-1)
alea_concat = np.concatenate(all_alea_vars, axis=0).reshape(-1)
epi_concat = np.concatenate(all_epi_vars, axis=0).reshape(-1)

# Compute metrics
mae = np.mean(np.abs(preds_concat - targets_concat))
rmse = np.sqrt(np.mean((preds_concat - targets_concat) ** 2))
mape = np.mean(np.abs((targets_concat - preds_concat) / (np.abs(targets_concat) + 1e-8))) * 100
ss_res = np.sum((targets_concat - preds_concat) ** 2)
ss_tot = np.sum((targets_concat - np.mean(targets_concat)) ** 2)
r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
corr = np.corrcoef(preds_concat, targets_concat)[0, 1]

mean_alea = np.mean(np.sqrt(alea_concat))
mean_epi = np.mean(np.sqrt(epi_concat))
mean_total = np.mean(np.sqrt(alea_concat + epi_concat))

print(f"Test Results:")
print(f"  MAE: {mae:.6f}")
print(f"  RMSE: {rmse:.6f}")
print(f"  MAPE: {mape:.2f}%")
print(f"  R²: {r2:.6f}")
print(f"  Correlation: {corr:.6f}")
print(f"  Mean Aleatoric Uncertainty: {mean_alea:.6f}")
print(f"  Mean Epistemic Uncertainty: {mean_epi:.6f}")
print(f"  Mean Total Uncertainty: {mean_total:.6f}")

# Save to file
results = {
    "mae": float(mae),
    "rmse": float(rmse),
    "mape": float(mape),
    "r2": float(r2),
    "correlation": float(corr),
    "mean_aleatoric_uncertainty": float(mean_alea),
    "mean_epistemic_uncertainty": float(mean_epi),
    "mean_total_uncertainty": float(mean_total)
}

with open('BI-LSTM_PEMSBAY/test_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n[SAVED] Results: BI-LSTM_PEMSBAY/test_results.json")
