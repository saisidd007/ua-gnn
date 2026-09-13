import pickle
import numpy as np
import scipy.sparse as sp
import torch


def calculate_random_walk_matrix(adj_mx):
    """
    Computes random-walk transition matrix P = D^{-1} A.
    """
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.0
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx.toarray()


def calculate_reverse_random_walk_matrix(adj_mx):
    """
    Computes reverse random-walk transition matrix P_b = D_in^{-1} A^T.
    """
    return calculate_random_walk_matrix(adj_mx.T)


def load_adj_matrices(pkl_filename, device='cpu'):
    """
    Loads adjacency matrix from pickle file and returns forward and backward transition matrices as torch tensors.
    """
    with open(pkl_filename, 'rb') as f:
        content = f.read()
    # Normalize potential CRLF to LF for legacy protocol 0 pickles
    content = content.replace(b'\r\n', b'\n')
    try:
        sensor_ids, sensor_id_to_ind, adj_mx = pickle.loads(content, encoding='latin1')
    except Exception:
        sensor_ids, sensor_id_to_ind, adj_mx = pickle.loads(content)

    # Calculate transition matrices
    P_f = calculate_random_walk_matrix(adj_mx)
    P_b = calculate_reverse_random_walk_matrix(adj_mx)

    P_f_tensor = torch.tensor(P_f, dtype=torch.float32, device=device)
    P_b_tensor = torch.tensor(P_b, dtype=torch.float32, device=device)
    adj_tensor = torch.tensor(adj_mx, dtype=torch.float32, device=device)

    return {
        'sensor_ids': sensor_ids,
        'sensor_id_to_ind': sensor_id_to_ind,
        'adj_mx': adj_tensor,
        'transition_matrices': [P_f_tensor, P_b_tensor]
    }
