import numpy as np
import torch


class StandardScaler:
    """
    Standard Z-score scaler for normalizing and denormalizing traffic flow/speed data.
    """
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def fit(self, data):
        self.mean = np.nanmean(data)
        self.std = np.nanstd(data)
        if self.std == 0:
            self.std = 1.0

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

    def to_dict(self):
        return {'mean': float(self.mean), 'std': float(self.std)}

    @classmethod
    def from_dict(cls, d):
        return cls(mean=d['mean'], std=d['std'])


def masked_mae_np(y_true, y_pred, null_val=0.0):
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = ~np.isnan(y_true) & (y_true != null_val)
        mask = mask.astype(float)
        mask /= np.mean(mask)
        mae = np.abs(y_true - y_pred)
        mae = np.nan_to_num(mae * mask)
        return np.mean(mae)


def masked_rmse_np(y_true, y_pred, null_val=0.0):
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = ~np.isnan(y_true) & (y_true != null_val)
        mask = mask.astype(float)
        mask /= np.mean(mask)
        mse = (y_true - y_pred) ** 2
        mse = np.nan_to_num(mse * mask)
        return np.sqrt(np.mean(mse))


def masked_mape_np(y_true, y_pred, null_val=0.0):
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = ~np.isnan(y_true) & (y_true != null_val)
        mask = mask.astype(float)
        mask /= np.mean(mask)
        mape = np.abs((y_true - y_pred) / y_true)
        mape = np.nan_to_num(mape * mask)
        return np.mean(mape) * 100.0


def r2_score_np(y_true, y_pred):
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_t = y_true[mask]
    y_p = y_pred[mask]
    ss_res = np.sum((y_t - y_p) ** 2)
    ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - (ss_res / ss_tot))


def pearson_correlation_np(y_true, y_pred):
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_t = y_true[mask]
    y_p = y_pred[mask]
    if len(y_t) < 2:
        return 0.0
    corr = np.corrcoef(y_t.flatten(), y_p.flatten())[0, 1]
    return float(corr)


def horizon_metrics(y_true, y_pred, horizons=(3, 6, 9, 12)):
    """
    Computes horizon-specific MAE and RMSE at steps corresponding to 15, 30, 45, 60 minutes.
    
    y_true: numpy array of shape (num_samples, 12, num_nodes)
    y_pred: numpy array of shape (num_samples, 12, num_nodes)
    horizons: list of step indices (1-indexed)
    """
    results = {}
    for h in horizons:
        idx = h - 1  # convert to 0-indexed
        true_h = y_true[:, idx, :]
        pred_h = y_pred[:, idx, :]
        mae = np.mean(np.abs(true_h - pred_h))
        rmse = np.sqrt(np.mean((true_h - pred_h) ** 2))
        results[f'{h * 5}min'] = {
            'MAE': round(float(mae), 4),
            'RMSE': round(float(rmse), 4)
        }
    return results
