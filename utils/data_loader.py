import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils.metrics import StandardScaler


class TrafficDataset(Dataset):
    """
    PyTorch Dataset for Traffic Flow Forecasting sequences.
    """
    def __init__(self, x, y, tod=None, dow=None):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.tod = torch.tensor(tod, dtype=torch.long) if tod is not None else None
        self.dow = torch.tensor(dow, dtype=torch.long) if dow is not None else None

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        item = {
            'x': self.x[idx],
            'y': self.y[idx]
        }
        if self.tod is not None:
            item['tod'] = self.tod[idx]
        if self.dow is not None:
            item['dow'] = self.dow[idx]
        return item


def load_traffic_data(data_dir: str, batch_size: int = 64, shuffle_train: bool = True):
    """
    Loads train.npz, val.npz, and test.npz, normalizes speed data, and returns DataLoaders + scaler.
    """
    train_file = os.path.join(data_dir, 'train.npz')
    val_file = os.path.join(data_dir, 'val.npz')
    test_file = os.path.join(data_dir, 'test.npz')

    train_data = np.load(train_file)
    val_data = np.load(val_file)
    test_data = np.load(test_file)

    x_train, y_train = train_data['x'], train_data['y']
    x_val, y_val = val_data['x'], val_data['y']
    x_test, y_test = test_data['x'], test_data['y']

    # Extract speed feature (channel 0)
    speed_train = x_train[..., 0]
    
    # Fit scaler strictly on training set to avoid data leakage
    scaler = StandardScaler()
    scaler.fit(speed_train)

    # Normalize speed channel
    x_train_scaled = np.copy(x_train)
    x_train_scaled[..., 0] = scaler.transform(x_train[..., 0])

    x_val_scaled = np.copy(x_val)
    x_val_scaled[..., 0] = scaler.transform(x_val[..., 0])

    x_test_scaled = np.copy(x_test)
    x_test_scaled[..., 0] = scaler.transform(x_test[..., 0])

    # Extract target speed (channel 0 of y)
    y_train_speed = y_train[..., 0]
    y_val_speed = y_val[..., 0]
    y_test_speed = y_test[..., 0]

    # Extract temporal indices if present
    def extract_temporal_indices(x_arr):
        tod, dow = None, None
        if x_arr.shape[-1] >= 2:
            # Channel 1 is normalized time of day [0, 1) -> map to slot [0..287]
            tod = np.clip(np.floor(x_arr[..., 1] * 288).astype(np.int64), 0, 287)
        if x_arr.shape[-1] >= 3:
            # Channel 2 is day of week [0..6]
            dow = np.clip(x_arr[..., 2].astype(np.int64), 0, 6)
        return tod, dow

    tod_train, dow_train = extract_temporal_indices(x_train)
    tod_val, dow_val = extract_temporal_indices(x_val)
    tod_test, dow_test = extract_temporal_indices(x_test)

    train_dataset = TrafficDataset(x_train_scaled[..., :1], y_train_speed, tod_train, dow_train)
    val_dataset = TrafficDataset(x_val_scaled[..., :1], y_val_speed, tod_val, dow_val)
    test_dataset = TrafficDataset(x_test_scaled[..., :1], y_test_speed, tod_test, dow_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'scaler': scaler,
        'y_test_true': y_test_speed
    }
