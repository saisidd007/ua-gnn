import argparse
import os
import yaml
import numpy as np
import torch

from model.ua_gnn import UAGNN
from utils.data_loader import load_traffic_data
from utils.graph_utils import load_adj_matrices
from utils.metrics import StandardScaler, masked_mae_np, masked_rmse_np, r2_score_np, pearson_correlation_np
from utils.calibration import compute_ece


def main():
    parser = argparse.ArgumentParser(description="Evaluate Deep Ensemble of UA-GNN models.")
    parser.add_argument("--config", type=str, default="configs/pems_bay.yaml", help="Path to config file.")
    parser.add_argument(
        "--model_paths",
        type=str,
        nargs="+",
        default=[
            "results/pems-bay/ensemble/model_seed1.pt",
            "results/pems-bay/ensemble/model_seed2.pt",
            "results/pems-bay/ensemble/model_seed3.pt"
        ],
        help="List of model checkpoint paths."
    )
    parser.add_argument("--output_dir", type=str, default="results/pems-bay/ensemble/eval/", help="Output directory.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    # Load Graph
    graph_info = load_adj_matrices(cfg['dataset']['adj_filename'], device=device)
    transition_matrices = graph_info['transition_matrices']

    # Load Test Data
    data_bundles = load_traffic_data(
        data_dir=cfg['dataset']['data_dir'],
        batch_size=cfg['training']['batch_size'],
        shuffle_train=False
    )
    test_loader = data_bundles['test_loader']

    ensemble_means = []
    ensemble_aleatorics = []
    y_ground_truth = []
    scaler = None

    for model_path in args.model_paths:
        if not os.path.exists(model_path):
            print(f"Warning: Checkpoint {model_path} not found. Skipping.")
            continue

        print(f"Loading Ensemble Member: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        scaler = StandardScaler.from_dict(checkpoint['scaler'])

        model = UAGNN(
            num_nodes=cfg['dataset']['num_nodes'],
            input_dim=cfg['dataset']['input_dim'],
            seq_len=cfg['dataset']['seq_len'],
            horizon=cfg['dataset']['horizon'],
            hidden_dim=cfg['model']['hidden_dim'],
            num_st_blocks=cfg['model']['num_st_blocks'],
            diffusion_steps=cfg['model']['diffusion_steps'],
            num_attention_heads=cfg['model']['num_attention_heads'],
            dilation_rates=cfg['model']['dilation_rates'],
            kernel_size=cfg['model']['kernel_size'],
            bilstm_hidden=cfg['model']['bilstm_hidden'],
            dropout=cfg['model']['dropout'],
            tod_embedding_dim=cfg['model'].get('tod_embedding_dim', 32),
            dow_embedding_dim=cfg['model'].get('dow_embedding_dim', 32)
        ).to(device)

        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        member_means = []
        member_aleat = []
        gt_list = []

        with torch.no_grad():
            for batch in test_loader:
                x = batch['x'].to(device)
                y = batch['y']
                tod = batch.get('tod', None)
                dow = batch.get('dow', None)
                if tod is not None:
                    tod = tod.to(device)
                if dow is not None:
                    dow = dow.to(device)

                mu, var_aleat = model(x, transition_matrices, tod=tod, dow=dow)

                # Denormalize
                mu_denorm = scaler.inverse_transform(mu.cpu().numpy())
                y_denorm = scaler.inverse_transform(y.numpy())
                var_denorm = var_aleat.cpu().numpy() * (scaler.std ** 2)

                member_means.append(mu_denorm)
                member_aleat.append(var_denorm)
                gt_list.append(y_denorm)

        ensemble_means.append(np.concatenate(member_means, axis=0))
        ensemble_aleatorics.append(np.concatenate(member_aleat, axis=0))
        if len(y_ground_truth) == 0:
            y_ground_truth = np.concatenate(gt_list, axis=0)

    if len(ensemble_means) == 0:
        print("No valid models were evaluated.")
        return

    # Stack ensemble predictions: shape (M, num_samples, horizon, num_nodes)
    ensemble_means_stack = np.stack(ensemble_means, axis=0)
    ensemble_aleat_stack = np.stack(ensemble_aleatorics, axis=0)

    # 1. Ensemble Mean: \bar{\mu}
    final_pred_mean = np.mean(ensemble_means_stack, axis=0)

    # 2. Epistemic Uncertainty: variance across the ensemble member predictions
    var_epistemic = np.var(ensemble_means_stack, axis=0)

    # 3. Expected Aleatoric Uncertainty: average predicted aleatoric variance
    var_aleatoric = np.mean(ensemble_aleat_stack, axis=0)

    # 4. Total Predictive Uncertainty
    var_total = var_aleatoric + var_epistemic
    std_total = np.sqrt(var_total)

    # Metrics
    mae = masked_mae_np(y_ground_truth, final_pred_mean)
    rmse = masked_rmse_np(y_ground_truth, final_pred_mean)
    r2 = r2_score_np(y_ground_truth, final_pred_mean)
    pearson = pearson_correlation_np(y_ground_truth, final_pred_mean)
    ece, _ = compute_ece(y_ground_truth, final_pred_mean, std_total)

    summary_text = (
        "===========================================================\n"
        f"Deep Ensemble Evaluation Summary ({len(ensemble_means)} Models)\n"
        "===========================================================\n"
        f"MAE:                  {mae:.4f}\n"
        f"RMSE:                 {rmse:.4f}\n"
        f"R2 Score:             {r2:.4f}\n"
        f"Pearson Correlation:  {pearson:.4f}\n"
        f"Mean Aleatoric Unc:   {np.mean(var_aleatoric):.4f}\n"
        f"Mean Epistemic Unc:   {np.mean(var_epistemic):.4f}\n"
        f"Mean Total Unc:       {np.mean(var_total):.4f}\n"
        f"Expected Calib Error: {ece:.4f}\n"
        "===========================================================\n"
    )
    print(summary_text)

    with open(os.path.join(args.output_dir, 'ensemble_summary.txt'), 'w') as f:
        f.write(summary_text)


if __name__ == "__main__":
    main()
