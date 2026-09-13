import argparse
import os
import numpy as np
import pandas as pd


def generate_graph_seq2seq_io_data(
    df: pd.DataFrame,
    x_offsets: np.ndarray,
    y_offsets: np.ndarray,
    add_time_in_day: bool = True,
    add_day_in_week: bool = True
):
    """
    Generates sequence-to-sequence input and target arrays with temporal features.
    
    Returns:
        x: (num_samples, seq_len, num_nodes, input_dim)
        y: (num_samples, horizon, num_nodes, output_dim)
    """
    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)
    data_list = [data]

    if add_time_in_day:
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        data_list.append(time_in_day)

    if add_day_in_week:
        day_of_week = df.index.dayofweek.values
        dow = np.tile(day_of_week, [1, num_nodes, 1]).transpose((2, 1, 0))
        data_list.append(dow)

    data = np.concatenate(data_list, axis=-1)

    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))

    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, ...]
        y_t = data[t + y_offsets, ...]
        x.append(x_t)
        y.append(y_t)

    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y


def main():
    parser = argparse.ArgumentParser(description="Preprocess traffic dataset with chronological 70/10/20 split.")
    parser.add_argument("--output_dir", type=str, default="data/metr-la/", help="Output directory path.")
    parser.add_argument("--traffic_df_filename", type=str, default="data/metr-la/metr-la.h5", help="Path to traffic h5 file.")
    parser.add_argument("--seq_len", type=int, default=12, help="Sequence length (default: 12 = 60 mins).")
    parser.add_argument("--horizon", type=int, default=12, help="Prediction horizon (default: 12 = 60 mins).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Loading raw traffic data from: {args.traffic_df_filename}")
    df = pd.read_hdf(args.traffic_df_filename)
    print(f"Data shape: {df.shape} (Time steps: {df.shape[0]}, Sensors: {df.shape[1]})")

    # Sequence offsets: 12 historical steps, 12 future steps
    x_offsets = np.sort(np.arange(-(args.seq_len - 1), 1, 1))
    y_offsets = np.sort(np.arange(1, args.horizon + 1, 1))

    print("Extracting sliding sequences...")
    x, y = generate_graph_seq2seq_io_data(
        df,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=True,
        add_day_in_week=True
    )
    print(f"Total sequences generated: x={x.shape}, y={y.shape}")

    # Standard 70/10/20 purely chronological split
    num_samples = len(x)
    num_train = round(num_samples * 0.7)
    num_val = round(num_samples * 0.1)
    num_test = num_samples - num_train - num_val

    # Train
    x_train, y_train = x[:num_train], y[:num_train]
    # Val
    x_val, y_val = x[num_train:num_train + num_val], y[num_train:num_train + num_val]
    # Test
    x_test, y_test = x[num_train + num_val:], y[num_train + num_val:]

    print(f"Splits: Train={len(x_train)}, Val={len(x_val)}, Test={len(x_test)}")

    for split_name, (x_s, y_s) in [('train', (x_train, y_train)),
                                  ('val', (x_val, y_val)),
                                  ('test', (x_test, y_test))]:
        out_path = os.path.join(args.output_dir, f"{split_name}.npz")
        np.savez_compressed(out_path, x=x_s, y=y_s)
        print(f"Saved: {out_path} (x: {x_s.shape}, y: {y_s.shape})")

    print("Data preprocessing completed successfully!")


if __name__ == "__main__":
    main()
