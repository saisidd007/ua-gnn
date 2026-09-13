import numpy as np
import torch


def apply_sensor_dropout(x, dropout_rate: float, seed: int = 42):
    """
    Randomly drops (zeroes out) a fraction of sensor input streams across time.
    
    Parameters:
        x: numpy array or torch.Tensor of shape (batch, time, nodes, features)
        dropout_rate: fraction of sensors to zero out (e.g., 0.10 = 10%)
        seed: random seed for reproducibility
        
    Returns:
        masked input array with same shape as x.
    """
    if dropout_rate <= 0.0:
        return x

    is_torch = isinstance(x, torch.Tensor)
    if is_torch:
        device = x.device
        x_np = x.detach().cpu().numpy()
    else:
        x_np = np.copy(x)

    rng = np.random.default_rng(seed)
    num_nodes = x_np.shape[2]
    num_drop = int(num_nodes * dropout_rate)
    
    mask = np.ones(num_nodes, dtype=x_np.dtype)
    if num_drop > 0:
        drop_idx = rng.choice(num_nodes, size=num_drop, replace=False)
        mask[drop_idx] = 0.0

    masked_x = x_np * mask[np.newaxis, np.newaxis, :, np.newaxis]

    if is_torch:
        return torch.from_numpy(masked_x).to(device)
    return masked_x
