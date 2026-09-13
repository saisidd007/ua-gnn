import argparse
import os
import random
import yaml
import numpy as np
import torch

from model.ua_gnn import UAGNN
from utils.data_loader import load_traffic_data
from utils.graph_utils import load_adj_matrices
from utils.metrics import (
    StandardScaler,
    masked_mae_np,
    masked_rmse_np,
    r2_score_np,
    pearson_correlation_np,
    horizon_metrics
)
from utils.dropout import apply_sensor_dropout
from utils.calibration import compute_ece, plot_reliability_diagram


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def evaluate_model(
    model,
    test_loader,
    transition_matrices,
    device,
    scaler,
    mc_samples: int = 10,
    dropout_rate: float = 0.0,
    sensor_dropout_seed: int = 42
):
    """
    Runs MC Dropout evaluation over the test dataset with optional sensor failure simulation.
    """
    all_y_true = []
    all_y_pred_mean = []
    all_var_aleat = []
    all_var_epist = []
    all_var_total = []

    for batch in test_loader:
        x = batch['x']
        y = batch['y']
        tod = batch.get('tod', None)
        dow = batch.get('dow', None)

        # Apply sensor dropout if enabled
        if dropout_rate > 0.0:
            x = apply_sensor_dropout(x, dropout_rate=dropout_rate, seed=sensor_dropout_seed)

        x = x.to(device)
        if tod is not None:
            tod = tod.to(device)
        if dow is not None:
            dow = dow.to(device)

        # MC Dropout stochastic inference
        unc_outputs = model.predict_with_uncertainty(
            x=x,
            transition_matrices=transition_matrices,
            tod=tod,
            dow=dow,
            mc_samples=mc_samples
        )

        pred_mean = unc_outputs['pred_mean'].cpu().numpy()
        var_aleat = unc_outputs['var_aleatoric'].cpu().numpy()
        var_epist = unc_outputs['var_epistemic'].cpu().numpy()
        var_total = unc_outputs['var_total'].cpu().numpy()

        # Denormalize predictions and targets to original physical scale (mph)
        pred_mean_denorm = scaler.inverse_transform(pred_mean)
        y_denorm = scaler.inverse_transform(y.numpy())

        # Note: Variance in standardized scale -> multiply by (scaler.std ** 2) for original scale
        var_scale = (scaler.std ** 2)
        var_aleat_denorm = var_aleat * var_scale
        var_epist_denorm = var_epist * var_scale
        var_total_denorm = var_total * var_scale

        all_y_true.append(y_denorm)
        all_y_pred_mean.append(pred_mean_denorm)
        all_var_aleat.append(var_aleat_denorm)
        all_var_epist.append(var_epist_denorm)
        all_var_total.append(var_total_denorm)

    y_true = np.concatenate(all_y_true, axis=0)
    y_pred_mean = np.concatenate(all_y_pred_mean, axis=0)
    var_aleatoric = np.concatenate(all_var_aleat, axis=0)
    var_epistemic = np.concatenate(all_var_epist, axis=0)
    var_total = np.concatenate(all_var_total, axis=0)
    std_total = np.sqrt(var_total)

    # Compute Core Performance Metrics
    mae = masked_mae_np(y_true, y_pred_mean)
    rmse = masked_rmse_np(y_true, y_pred_mean)
    r2 = r2_score_np(y_true, y_pred_mean)
    pearson = pearson_correlation_np(y_true, y_pred_mean)
    mean_aleat = float(np.mean(var_aleatoric))
    mean_epist = float(np.mean(var_epistemic))
    mean_total = float(np.mean(var_total))

    # Horizon-specific metrics
    h_metrics = horizon_metrics(y_true, y_pred_mean, horizons=[3, 6, 9, 12])

    return {
        'y_true': y_true,
        'y_pred_mean': y_pred_mean,
        'y_pred_std': std_total,
        'var_aleatoric': var_aleatoric,
        'var_epistemic': var_epistemic,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'pearson': pearson,
        'mean_aleatoric': mean_aleat,
        'mean_epistemic': mean_epist,
        'mean_total': mean_total,
        'horizon_metrics': h_metrics
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate UA-GNN Model.")
    parser.add_argument("--config", type=str, default="configs/pems_bay.yaml", help="Path to YAML config file.")
    parser.add_argument("--model_path", type=str, default="results/pems-bay/best_model.pt", help="Path to model checkpoint.")
    parser.add_argument("--output_dir", type=str, default="results/pems-bay/eval/", help="Output directory for evaluation results.")
    parser.add_argument("--dropout_rate", type=float, default=0.0, help="Fraction of sensors to drop (0.0 to 0.5).")
    parser.add_argument("--num_seeds", type=int, default=1, help="Number of random seeds for sensor dropout evaluation.")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    # Load Graph
    graph_info = load_adj_matrices(cfg['dataset']['adj_filename'], device=device)
    transition_matrices = graph_info['transition_matrices']

    # Load Data
    data_bundles = load_traffic_data(
        data_dir=cfg['dataset']['data_dir'],
        batch_size=cfg['training']['batch_size'],
        shuffle_train=False
    )
    test_loader = data_bundles['test_loader']

    # Load Checkpoint
    print(f"Loading checkpoint from: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    scaler = StandardScaler.from_dict(checkpoint['scaler'])

    # Build Model
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
    print("Model state loaded successfully.")

    mc_samples = cfg.get('uncertainty', {}).get('mc_samples', 10)

    # If num_seeds > 1 (e.g., for sensor dropout robustness experiment)
    if args.num_seeds > 1:
        print(f"Running sensor dropout evaluation across {args.num_seeds} seeds at rate {args.dropout_rate}...")
        maes, rmses, total_uncs = [], [], []
        for s in range(args.num_seeds):
            seed_val = 42 + s * 100
            res = evaluate_model(
                model=model,
                test_loader=test_loader,
                transition_matrices=transition_matrices,
                device=device,
                scaler=scaler,
                mc_samples=mc_samples,
                dropout_rate=args.dropout_rate,
                sensor_dropout_seed=seed_val
            )
            maes.append(res['mae'])
            rmses.append(res['rmse'])
            total_uncs.append(res['mean_total'])

        mean_mae, std_mae = np.mean(maes), np.std(maes)
        mean_rmse, std_rmse = np.mean(rmses), np.std(rmses)
        mean_unc, std_unc = np.mean(total_uncs), np.std(total_uncs)

        summary_text = (
            f"Sensor Dropout Rate: {args.dropout_rate}\n"
            f"Num Seeds: {args.num_seeds}\n"
            f"MAE: {mean_mae:.4f} +/- {std_mae:.4f}\n"
            f"RMSE: {mean_rmse:.4f} +/- {std_rmse:.4f}\n"
            f"Total Uncertainty: {mean_unc:.4f} +/- {std_unc:.4f}\n"
        )
        print(summary_text)
        summary_path = os.path.join(args.output_dir, 'results_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(summary_text)
        return

    # Standard Single Evaluation Run
    print("Running evaluation on test set...")
    res = evaluate_model(
        model=model,
        test_loader=test_loader,
        transition_matrices=transition_matrices,
        device=device,
        scaler=scaler,
        mc_samples=mc_samples,
        dropout_rate=args.dropout_rate,
        sensor_dropout_seed=42
    )

    y_true = res['y_true']
    y_pred_mean = res['y_pred_mean']
    y_pred_std = res['y_pred_std']

    # Compute ECE
    ece, cal_data = compute_ece(y_true, y_pred_mean, y_pred_std)

    # Save arrays as requested in the instruction sheet
    np.save(os.path.join(args.output_dir, 'y_true.npy'), y_true)
    np.save(os.path.join(args.output_dir, 'y_pred_mean.npy'), y_pred_mean)
    np.save(os.path.join(args.output_dir, 'y_pred_std.npy'), y_pred_std)

    # Plot reliability diagram
    rel_diag_png = os.path.join(args.output_dir, 'reliability_diagram.png')
    rel_diag_pdf = os.path.join(args.output_dir, 'reliability_diagram.pdf')
    plot_reliability_diagram(cal_data, model_name='UA-GNN', dataset_name=cfg['dataset']['name'].upper(), save_path=rel_diag_png)
    plot_reliability_diagram(cal_data, model_name='UA-GNN', dataset_name=cfg['dataset']['name'].upper(), save_path=rel_diag_pdf)

    # Format output summary
    summary_lines = [
        "===========================================================",
        f"UA-GNN Evaluation Summary on {cfg['dataset']['name'].upper()}",
        "===========================================================",
        f"MAE:                  {res['mae']:.4f}",
        f"RMSE:                 {res['rmse']:.4f}",
        f"R2 Score:             {res['r2']:.4f}",
        f"Pearson Correlation:  {res['pearson']:.4f}",
        f"Aleatoric Uncertainty:{res['mean_aleatoric']:.4f}",
        f"Epistemic Uncertainty:{res['mean_epistemic']:.4f}",
        f"Total Uncertainty:    {res['mean_total']:.4f}",
        f"Expected Calib Error: {ece:.4f}",
        "-----------------------------------------------------------",
        "Horizon-Specific Metrics:",
    ]
    for k, v in res['horizon_metrics'].items():
        summary_lines.append(f"  {k}: MAE={v['MAE']:.4f}, RMSE={v['RMSE']:.4f}")
    summary_lines.append("===========================================================")

    summary_text = "\n".join(summary_lines)
    print(summary_text)

    summary_file = os.path.join(args.output_dir, 'results_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(summary_text)
    print(f"Results summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
