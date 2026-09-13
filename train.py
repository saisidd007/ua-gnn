import argparse
import os
import random
import json
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from model.ua_gnn import UAGNN
from utils.data_loader import load_traffic_data
from utils.graph_utils import load_adj_matrices
from utils.metrics import masked_mae_np, masked_rmse_np


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def compute_hybrid_loss(y_true, mu_pred, var_pred, lambda_weight=1.0):
    """
    Hybrid Uncertainty-Aware Loss:
    L_total = L_MSE + lambda * L_NLL
    where L_NLL = 0.5 * ( (y - mu)^2 / var + log(var) )
    """
    # Mean Squared Error term
    mse_loss = nn.functional.mse_loss(mu_pred, y_true)

    # Gaussian Negative Log-Likelihood term
    nll_loss = 0.5 * torch.mean(((y_true - mu_pred) ** 2) / (var_pred + 1e-6) + torch.log(var_pred + 1e-6))

    total_loss = mse_loss + (lambda_weight * nll_loss)
    return total_loss, mse_loss.item(), nll_loss.item()


def train_epoch(model, train_loader, optimizer, transition_matrices, device, lambda_weight, clip_grad=5.0):
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    total_nll = 0.0
    num_batches = 0

    for batch in train_loader:
        x = batch['x'].to(device)
        y = batch['y'].to(device)
        tod = batch.get('tod', None)
        dow = batch.get('dow', None)
        if tod is not None:
            tod = tod.to(device)
        if dow is not None:
            dow = dow.to(device)

        optimizer.zero_grad()
        mu, var_aleat = model(x, transition_matrices, tod=tod, dow=dow)

        loss, mse_val, nll_val = compute_hybrid_loss(y, mu, var_aleat, lambda_weight=lambda_weight)
        
        if torch.isnan(loss):
            raise ValueError("NaN loss encountered during training!")

        loss.backward()
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        total_loss += loss.item()
        total_mse += mse_val
        total_nll += nll_val
        num_batches += 1

    return total_loss / num_batches, total_mse / num_batches, total_nll / num_batches


def validate(model, val_loader, transition_matrices, device, scaler):
    model.eval()
    total_val_loss = 0.0
    all_preds = []
    all_targets = []
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            tod = batch.get('tod', None)
            dow = batch.get('dow', None)
            if tod is not None:
                tod = tod.to(device)
            if dow is not None:
                dow = dow.to(device)

            mu, var_aleat = model(x, transition_matrices, tod=tod, dow=dow)
            loss, _, _ = compute_hybrid_loss(y, mu, var_aleat, lambda_weight=1.0)
            total_val_loss += loss.item()
            num_batches += 1

            # Inverse transform to original scale
            mu_denorm = scaler.inverse_transform(mu.cpu().numpy())
            y_denorm = scaler.inverse_transform(y.cpu().numpy())
            all_preds.append(mu_denorm)
            all_targets.append(y_denorm)

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    val_mae = masked_mae_np(all_targets, all_preds)
    val_rmse = masked_rmse_np(all_targets, all_preds)

    return total_val_loss / num_batches, val_mae, val_rmse


def main():
    parser = argparse.ArgumentParser(description="Train UA-GNN on Traffic Forecasting Benchmark.")
    parser.add_argument("--config", type=str, default="configs/metr_la.yaml", help="Path to YAML config file.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--model_path", type=str, default=None, help="Optional override for saved model checkpoint.")
    args = parser.parse_args()

    set_seed(args.seed)

    # Load configuration
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} | Random Seed: {args.seed}")
    print(f"Dataset: {cfg['dataset']['name']} | Nodes: {cfg['dataset']['num_nodes']}")

    # Setup directories
    results_dir = cfg['output']['results_dir']
    os.makedirs(results_dir, exist_ok=True)
    model_save_path = args.model_path or cfg['output']['model_path']
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # Load Graph Structure
    graph_info = load_adj_matrices(cfg['dataset']['adj_filename'], device=device)
    transition_matrices = graph_info['transition_matrices']

    # Load Dataset
    data_bundles = load_traffic_data(
        data_dir=cfg['dataset']['data_dir'],
        batch_size=cfg['training']['batch_size']
    )
    train_loader = data_bundles['train_loader']
    val_loader = data_bundles['val_loader']
    scaler = data_bundles['scaler']

    # Initialize UA-GNN Model
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

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"UA-GNN Model Initialized. Total Trainable Parameters: {total_params:,}")

    # Optimization
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg['training']['learning_rate']),
        weight_decay=float(cfg['training'].get('weight_decay', 1e-4))
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=cfg['training']['lr_patience'],
        factor=cfg['training'].get('lr_decay_ratio', 0.5),
        verbose=True
    )

    epochs = cfg['training']['epochs']
    lambda_anneal = cfg['training'].get('lambda_anneal_epochs', 10)

    best_val_loss = float('inf')
    loss_history = {'train_loss': [], 'val_loss': [], 'val_mae': [], 'val_rmse': []}

    print("Starting Training Loop...")
    for epoch in range(1, epochs + 1):
        # Lambda annealing for hybrid loss
        lambda_w = min(1.0, epoch / float(lambda_anneal)) if lambda_anneal > 0 else 1.0

        train_loss, train_mse, train_nll = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            transition_matrices=transition_matrices,
            device=device,
            lambda_weight=lambda_w,
            clip_grad=cfg['training'].get('clip_grad_norm', 5.0)
        )

        val_loss, val_mae, val_rmse = validate(
            model=model,
            val_loader=val_loader,
            transition_matrices=transition_matrices,
            device=device,
            scaler=scaler
        )

        scheduler.step(val_loss)

        loss_history['train_loss'].append(train_loss)
        loss_history['val_loss'].append(val_loss)
        loss_history['val_mae'].append(val_mae)
        loss_history['val_rmse'].append(val_rmse)

        print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.4f} | Val RMSE: {val_rmse:.4f}")

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if cfg['output'].get('save_model', True):
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_mae': val_mae,
                    'val_rmse': val_rmse,
                    'scaler': scaler.to_dict(),
                    'config': cfg
                }, model_save_path)
                print(f"  --> Saved new best checkpoint to {model_save_path}")

    # Save loss history for Figure F1 plotting
    history_file = os.path.join(results_dir, 'loss_history.json')
    with open(history_file, 'w') as f:
        json.dump(loss_history, f, indent=2)
    print(f"Training completed! Loss history saved to {history_file}")


if __name__ == "__main__":
    main()
