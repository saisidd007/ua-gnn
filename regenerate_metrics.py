#!/usr/bin/env python
"""
Regenerate analysis_50epoch_per_horizon_metrics.csv
Runs inference on test set and computes per-horizon metrics
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.enhanced_dataset import create_enhanced_dataset
from src.models.enhanced_gnn import create_improved_model

RESULTS = PROJECT_ROOT / 'results'
DATA = PROJECT_ROOT / 'data'

def run_inference_and_per_horizon_metrics(
    checkpoint_path: str = str(RESULTS / 'enhanced_best_model.pt'),
    root_dir: str = str(DATA),
    sequence_length: int = 12,
    prediction_length: int = 12,
    mc_samples: int = 30,
):
    """Load checkpoint, run MC-dropout inference, compute per-horizon metrics."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        return None

    try:
        print("Building dataset...")
        dataset = create_enhanced_dataset(root_dir=root_dir, sequence_length=sequence_length, prediction_length=prediction_length)
        test_data = dataset.get_test_data()
        
        if len(test_data) == 0:
            print('No test sequences found; cannot generate metrics.')
            return None
            
        print(f"Test set size: {len(test_data)} sequences")
        
    except Exception as e:
        print(f'Failed to create dataset: {e}')
        return None

    from torch_geometric.loader import DataLoader
    test_loader = DataLoader(test_data, batch_size=8, shuffle=False, num_workers=0)

    print("Loading model...")
    model = create_improved_model(in_channels=sequence_length, out_channels=prediction_length).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
    else:
        state = ckpt
        
    model.load_state_dict(state)
    model.eval()

    print(f"Starting MC inference with {mc_samples} samples...")
    
    horizons = prediction_length
    all_preds_per_h = [[] for _ in range(horizons)]
    all_tgts_per_h = [[] for _ in range(horizons)]
    all_cov_flags_per_h = [[] for _ in range(horizons)]
    all_vars_per_h = [[] for _ in range(horizons)]
    all_piw_per_h = [[] for _ in range(horizons)]

    z95 = 1.96

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            batch = batch.to(device)
            
            if batch_idx % 10 == 0:
                print(f"  Processing batch {batch_idx}/{len(test_loader)}")

            mc_preds = []
            mc_aleas = []
            for k in range(mc_samples):
                model.train()
                preds_k, alea_k, _ = model(batch.x, batch.edge_index, batch.missing_mask if hasattr(batch, 'missing_mask') else None, return_uncertainty=True)
                mc_preds.append(preds_k.unsqueeze(0))
                mc_aleas.append(alea_k.unsqueeze(0))

            mc_preds = torch.cat(mc_preds, dim=0)
            mc_aleas = torch.cat(mc_aleas, dim=0)

            mean_pred = mc_preds.mean(dim=0).cpu().numpy()
            epi_var = mc_preds.var(dim=0).cpu().numpy()
            alea_var = mc_aleas.mean(dim=0).cpu().numpy()
            total_var = epi_var + alea_var

            targets = batch.y.cpu().numpy()

            for h in range(horizons):
                pred_h = mean_pred[:, :, h].reshape(-1)
                tgt_h = targets[:, :, h].reshape(-1)
                var_h = total_var[:, :, h].reshape(-1)

                ci_half = z95 * np.sqrt(np.maximum(var_h, 1e-6))
                covered = (tgt_h >= (pred_h - ci_half)) & (tgt_h <= (pred_h + ci_half))
                piw_h = (2.0 * ci_half)

                all_preds_per_h[h].append(pred_h)
                all_tgts_per_h[h].append(tgt_h)
                all_cov_flags_per_h[h].append(covered)
                all_vars_per_h[h].append(var_h)
                all_piw_per_h[h].append(piw_h)

    print("Computing per-horizon metrics...")
    
    rows = []
    for h in range(horizons):
        preds = np.concatenate(all_preds_per_h[h], axis=0)
        tgts = np.concatenate(all_tgts_per_h[h], axis=0)
        covs = np.concatenate(all_cov_flags_per_h[h], axis=0)
        piw_h_all = np.concatenate(all_piw_per_h[h], axis=0)

        mae = float(np.mean(np.abs(preds - tgts)))
        rmse = float(np.sqrt(np.mean((preds - tgts) ** 2)))
        mape = float(np.mean(np.abs((tgts - preds) / (np.abs(tgts) + 1e-6))) * 100)
        coverage95 = float(np.mean(covs))

        piw_mean = float(np.mean(piw_h_all)) if piw_h_all.size > 0 else float('nan')
        piw_median = float(np.median(piw_h_all)) if piw_h_all.size > 0 else float('nan')
        piw_q25 = float(np.percentile(piw_h_all, 25)) if piw_h_all.size > 0 else float('nan')
        piw_q75 = float(np.percentile(piw_h_all, 75)) if piw_h_all.size > 0 else float('nan')

        rows.append({
            'horizon': h+1,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'coverage95': coverage95,
            'piw95_mean': piw_mean,
            'piw95_median': piw_median,
            'piw95_q25': piw_q25,
            'piw95_q75': piw_q75
        })

    per_h_df = pd.DataFrame(rows)
    output_path = RESULTS / 'analysis_50epoch_per_horizon_metrics.csv'
    per_h_df.to_csv(output_path, index=False)
    
    print(f'\n✓ Saved per-horizon metrics to {output_path}')
    print(f'\n{per_h_df.to_string()}')

    return per_h_df


if __name__ == '__main__':
    try:
        run_inference_and_per_horizon_metrics()
        print('\n✓ Metrics regeneration completed successfully!')
    except Exception as e:
        print(f'✗ Error: {e}')
        import traceback
        traceback.print_exc()
