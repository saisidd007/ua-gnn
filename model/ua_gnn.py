import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers import (
    LinearEmbedding,
    SpatioTemporalBlock,
    BiLSTMAggregator,
    PredictionHeads
)


class UAGNN(nn.Module):
    """
    Uncertainty-Aware Spatio-Temporal Graph Neural Network (UA-GNN).
    
    Combines:
    1. Temporal context embeddings (Time-of-day & Day-of-week)
    2. Linear input feature projection (Conv 1x1)
    3. 4 Stacked Interleaved Spatio-Temporal blocks (WaveNet Dilated Conv + Diffusion Graph Conv + GAT)
    4. Bidirectional LSTM Aggregator
    5. Dual Prediction Heads (Predictive Mean & Aleatoric Variance)
    6. Monte Carlo Dropout for Epistemic Uncertainty Estimation
    """
    def __init__(
        self,
        num_nodes: int = 325,
        input_dim: int = 1,
        seq_len: int = 12,
        horizon: int = 12,
        hidden_dim: int = 64,
        num_st_blocks: int = 4,
        diffusion_steps: int = 2,
        num_attention_heads: int = 4,
        dilation_rates: list = (1, 2, 4, 8),
        kernel_size: int = 2,
        bilstm_hidden: int = 64,
        dropout: float = 0.3,
        tod_embedding_dim: int = 32,
        dow_embedding_dim: int = 32,
        use_diffusion: bool = True,
        use_bilstm: bool = True,
        use_gat: bool = True
    ):
        super(UAGNN, self).__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.num_st_blocks = num_st_blocks
        self.dropout_rate = dropout
        self.use_diffusion = use_diffusion
        self.use_bilstm = use_bilstm
        self.use_gat = use_gat

        # 1. Temporal context embeddings
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        
        if self.tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(288, tod_embedding_dim)
        else:
            self.tod_embedding = None

        if self.dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        else:
            self.dow_embedding = None

        total_in_dim = input_dim + (tod_embedding_dim if tod_embedding_dim > 0 else 0) + (dow_embedding_dim if dow_embedding_dim > 0 else 0)

        # 2. Linear feature projection layer
        self.embedding_layer = LinearEmbedding(in_features=total_in_dim, hidden_dim=hidden_dim)

        # 3. Stacked Spatio-Temporal blocks
        self.st_blocks = nn.ModuleList([
            SpatioTemporalBlock(
                hidden_dim=hidden_dim,
                diffusion_steps=diffusion_steps if use_diffusion else 0,
                num_heads=num_attention_heads if use_gat else 1,
                dilation_rates=dilation_rates,
                kernel_size=kernel_size,
                dropout=dropout
            )
            for _ in range(num_st_blocks)
        ])

        # 4. Bidirectional LSTM temporal aggregator
        if use_bilstm:
            self.temporal_aggregator = BiLSTMAggregator(
                input_dim=hidden_dim,
                hidden_dim=bilstm_hidden,
                dropout=dropout
            )
        else:
            self.temporal_aggregator = None

        # 5. Dual Prediction heads
        self.heads = PredictionHeads(
            hidden_dim=hidden_dim,
            horizon=horizon,
            dropout=dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        transition_matrices: list,
        tod: torch.Tensor = None,
        dow: torch.Tensor = None,
        adj_mask: torch.Tensor = None
    ):
        """
        x: (B, T, N, input_dim)
        tod: Optional (B, T, N) or (B, T) Time of day indices [0..287]
        dow: Optional (B, T, N) or (B, T) Day of week indices [0..6]
        transition_matrices: list of normalized diffusion matrices [P_f, P_b]
        """
        B, T, N, _ = x.shape
        features = [x]

        # Add time-of-day embedding if enabled
        if self.tod_embedding is not None and tod is not None:
            if tod.dim() == 2:  # (B, T)
                tod = tod.unsqueeze(-1).expand(B, T, N)
            tod_emb = self.tod_embedding(tod.long())  # (B, T, N, tod_dim)
            features.append(tod_emb)

        # Add day-of-week embedding if enabled
        if self.dow_embedding is not None and dow is not None:
            if dow.dim() == 2:  # (B, T)
                dow = dow.unsqueeze(-1).expand(B, T, N)
            dow_emb = self.dow_embedding(dow.long())  # (B, T, N, dow_dim)
            features.append(dow_emb)

        # Concatenate along feature dimension
        x_in = torch.cat(features, dim=-1)  # (B, T, N, total_in_dim)

        # 1. Feature embedding
        h = self.embedding_layer(x_in)  # (B, T, N, hidden_dim)

        # 2. Pass through stacked ST blocks
        for block in self.st_blocks:
            h = block(h, transition_matrices, adj_mask)

        # 3. Temporal aggregation (BiLSTM)
        if self.temporal_aggregator is not None:
            h = self.temporal_aggregator(h)

        # Extract representation at the latest time step (t = T-1)
        h_last = h[:, -1, :, :]  # (B, N, hidden_dim)

        # 4. Dual prediction heads
        mu, var_aleatoric = self.heads(h_last)
        return mu, var_aleatoric

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        transition_matrices: list,
        tod: torch.Tensor = None,
        dow: torch.Tensor = None,
        adj_mask: torch.Tensor = None,
        mc_samples: int = 10
    ):
        r"""
        Performs Monte Carlo Dropout inference over K stochastic forward passes.
        
        Returns:
            pred_mean: (B, horizon, N) - overall predictive mean \bar{\mu}
            var_aleatoric: (B, horizon, N) - expected aleatoric variance
            var_epistemic: (B, horizon, N) - variance across MC sample means
            var_total: (B, horizon, N) - total predictive variance
            std_total: (B, horizon, N) - total standard deviation
        """
        # Enable dropout during inference for MC sampling
        self.train()

        means = []
        aleatorics = []

        with torch.no_grad():
            for _ in range(mc_samples):
                mu_k, var_aleat_k = self.forward(
                    x=x,
                    transition_matrices=transition_matrices,
                    tod=tod,
                    dow=dow,
                    adj_mask=adj_mask
                )
                means.append(mu_k)
                aleatorics.append(var_aleat_k)

        # Stack samples: (K, B, horizon, N)
        means_stack = torch.stack(means, dim=0)
        aleat_stack = torch.stack(aleatorics, dim=0)

        # 1. Predictive Mean: \bar{\mu} = 1/K \sum \mu^{(k)}
        pred_mean = torch.mean(means_stack, dim=0)

        # 2. Aleatoric Uncertainty: \bar{\sigma}^2_aleatoric = 1/K \sum \sigma^{2(k)}
        var_aleatoric = torch.mean(aleat_stack, dim=0)

        # 3. Epistemic Uncertainty: \hat{\sigma}^2_epistemic = 1/K \sum (\mu^{(k)} - \bar{\mu})^2
        var_epistemic = torch.var(means_stack, dim=0, unbiased=False)

        # 4. Total Uncertainty
        var_total = var_aleatoric + var_epistemic
        std_total = torch.sqrt(var_total)

        return {
            'pred_mean': pred_mean,
            'var_aleatoric': var_aleatoric,
            'var_epistemic': var_epistemic,
            'var_total': var_total,
            'std_total': std_total
        }
