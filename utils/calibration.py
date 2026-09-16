import numpy as np
import matplotlib.pyplot as plt

def compute_ece(y_true, y_pred_mean, y_pred_std, confidence_levels=None, num_bins=10):
    """
    Computes Expected Calibration Error for regression.
    y_true: numpy array, shape (N,) -- ground truth values
    y_pred_mean: numpy array, shape (N,) -- predicted means
    y_pred_std: numpy array, shape (N,) -- predicted std deviations
    (sqrt of total variance = aleatoric + epistemic)
    confidence_levels: list of floats, e.g. [0.5, 0.6, ..., 0.95]
    num_bins: number of confidence bins for ECE
    Returns: ece (float), calibration_data (dict for plotting)
    """
    from scipy import stats
    if confidence_levels is None:
        confidence_levels = np.linspace(0.1, 0.95, num_bins)
    empirical_coverage = []
    for conf in confidence_levels:
        # Build a symmetric interval at this confidence level
        z = stats.norm.ppf(0.5 + conf / 2.0)
        lower = y_pred_mean - z * y_pred_std
        upper = y_pred_mean + z * y_pred_std
        covered = np.mean((y_true >= lower) & (y_true <= upper))
        empirical_coverage.append(covered)
    empirical_coverage = np.array(empirical_coverage)
    # ECE = mean absolute difference between nominal and empirical
    ece = np.mean(np.abs(confidence_levels - empirical_coverage))
    calibration_data = {
        'confidence_levels': confidence_levels,
        'empirical_coverage': empirical_coverage
    }
    return ece, calibration_data

def plot_reliability_diagram(calibration_data, model_name, dataset_name, save_path):
    """
    Plots and saves a reliability diagram.
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
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved reliability diagram to {save_path}")
