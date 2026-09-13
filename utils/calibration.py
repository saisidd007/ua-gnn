import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os


def compute_ece(y_true, y_pred_mean, y_pred_std, confidence_levels=None, num_bins=10):
    """
    Computes Expected Calibration Error (ECE) for regression tasks.
    
    Parameters:
        y_true            : numpy array, shape (N,) -- ground truth values
        y_pred_mean       : numpy array, shape (N,) -- predicted means
        y_pred_std        : numpy array, shape (N,) -- predicted std deviations
                            (sqrt of total variance = aleatoric + epistemic)
        confidence_levels : list of floats, e.g. [0.5, 0.6, ..., 0.95]
        num_bins          : number of confidence bins for ECE

    Returns:
        ece               : float, Expected Calibration Error
        calibration_data  : dict with 'confidence_levels' and 'empirical_coverage'
    """
    y_true = np.asarray(y_true).flatten()
    y_pred_mean = np.asarray(y_pred_mean).flatten()
    y_pred_std = np.asarray(y_pred_std).flatten()

    # Avoid zero or nan std
    y_pred_std = np.clip(y_pred_std, 1e-6, None)

    if confidence_levels is None:
        confidence_levels = np.linspace(0.1, 0.95, num_bins)

    empirical_coverage = []
    for conf in confidence_levels:
        # Build symmetric Gaussian interval at nominal confidence level
        z = stats.norm.ppf(0.5 + conf / 2.0)
        lower = y_pred_mean - z * y_pred_std
        upper = y_pred_mean + z * y_pred_std
        covered = np.mean((y_true >= lower) & (y_true <= upper))
        empirical_coverage.append(covered)

    empirical_coverage = np.array(empirical_coverage)

    # ECE = mean absolute difference between nominal confidence and empirical coverage
    ece = float(np.mean(np.abs(confidence_levels - empirical_coverage)))

    calibration_data = {
        'confidence_levels': confidence_levels,
        'empirical_coverage': empirical_coverage
    }
    return ece, calibration_data


def plot_reliability_diagram(calibration_data, model_name, dataset_name, save_path):
    """
    Plots and saves a single reliability diagram.
    """
    conf = calibration_data['confidence_levels']
    emp = calibration_data['empirical_coverage']

    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    plt.plot(conf, emp, 'b-o', markersize=5, label=f'{model_name} ({dataset_name})')
    plt.fill_between(conf, conf, emp, alpha=0.15, color='blue')
    plt.xlabel('Nominal confidence level')
    plt.ylabel('Empirical coverage')
    plt.title(f'Reliability Diagram -- {model_name} on {dataset_name}')
    plt.legend(loc='upper left')
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved reliability diagram to {save_path}")


def plot_comparative_reliability_diagram(models_cal_data: dict, dataset_name: str, save_path: str):
    """
    Plots comparative reliability curves (e.g. UA-GNN vs Bayesian LSTM) on the same diagram.
    
    models_cal_data: dict of {model_label: calibration_data_dict}
    """
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfect Calibration')

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for idx, (label, data) in enumerate(models_cal_data.items()):
        conf = data['confidence_levels']
        emp = data['empirical_coverage']
        c = colors[idx % len(colors)]
        plt.plot(conf, emp, '-o', color=c, markersize=5, label=label)

    plt.xlabel('Nominal Confidence Level', fontsize=12)
    plt.ylabel('Empirical Coverage', fontsize=12)
    plt.title(f'Reliability Diagram -- {dataset_name}', fontsize=13, fontweight='bold')
    plt.legend(loc='upper left', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if save_path.endswith('.pdf'):
        png_path = save_path.replace('.pdf', '.png')
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparative reliability diagram to {save_path}")
