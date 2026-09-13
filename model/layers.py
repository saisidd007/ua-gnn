import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearEmbedding(nn.Module):
    """
    Stage 1: Linear feature embedding layer.
    Maps raw sensor input features (B, T, N, F) into latent representation (B, T, N, H).
    """
    def __init__(self, in_features: int, hidden_dim: int):
        super(LinearEmbedding, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_features,
            out_channels=hidden_dim,
            kernel_size=(1, 1),
            bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, T, N, F) -> permute to (B, F, T, N) for Conv2d
        x = x.permute(0, 3, 1, 2)
        out = self.conv(x)
        # Permute back to (B, T, N, H)
        out = out.permute(0, 2, 3, 1)
        return out


class TemporalDilatedConv(nn.Module):
    """
    Stage 2: WaveNet-inspired Dilated 1D Temporal Convolutions.
    Captures multi-scale temporal dependencies across time steps with exponential dilations.
    """
    def __init__(self, hidden_dim: int, dilation_rates=(1, 2, 4, 8), kernel_size=2, dropout=0.3):
        super(TemporalDilatedConv, self).__init__()
        self.dilation_rates = dilation_rates
        self.kernel_size = kernel_size
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for d in dilation_rates:
            pad = (kernel_size - 1) * d
            conv = nn.Conv1d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=kernel_size,
                dilation=d,
                padding=pad
            )
            self.layers.append(conv)
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, T, N, H)
        B, T, N, H = x.shape
        # Reshape to (B * N, H, T) for 1D convolution over time
        x_in = x.permute(0, 2, 3, 1).contiguous().view(B * N, H, T)
        
        out = x_in
        for conv, norm in zip(self.layers, self.norms):
            residual = out
            c_out = conv(out)
            # Slice off extra padding to maintain sequence length T
            c_out = c_out[:, :, :T]
            c_out = self.relu(c_out)
            c_out = self.dropout(c_out)
            # LayerNorm over channels (B*N, T, H)
            c_out = c_out.permute(0, 2, 1)
            c_out = norm(c_out)
            c_out = c_out.permute(0, 2, 1)
            out = self.relu(c_out + residual)

        # Reshape back to (B, T, N, H)
        out = out.view(B, N, H, T).permute(0, 3, 1, 2).contiguous()
        return out


class DiffusionGraphConv(nn.Module):
    """
    Stage 3: Diffusion-based Graph Convolution.
    Models directional multi-hop flow propagation using random-walk transition matrices P = D^{-1}A.
    """
    def __init__(self, hidden_dim: int, diffusion_steps=2):
        super(DiffusionGraphConv, self).__init__()
        self.hidden_dim = hidden_dim
        self.diffusion_steps = diffusion_steps
        self.num_matrices = 1 + 2 * diffusion_steps
        self.weight = nn.Parameter(torch.Tensor(self.num_matrices * hidden_dim, hidden_dim))
        self.bias = nn.Parameter(torch.Tensor(hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor, transition_matrices: list) -> torch.Tensor:
        """
        x: (B, T, N, H)
        transition_matrices: list of transition matrices [P_forward, P_backward]
        """
        B, T, N, H = x.shape
        x_flat = x.view(B * T, N, H)
        
        supports = [x_flat]
        for P in transition_matrices:
            x_k = x_flat
            for _ in range(self.diffusion_steps):
                x_k = torch.matmul(P, x_k)
                supports.append(x_k)

        x_cat = torch.cat(supports, dim=-1)
        out = torch.matmul(x_cat, self.weight) + self.bias
        out = F.relu(out)
        return out.view(B, T, N, H)


class GraphAttentionLayer(nn.Module):
    """
    Stage 4: Multi-Head Graph Attention (GAT) Mechanism.
    Refines local interactions and down-weights degraded/failed sensor streams.
    """
    def __init__(self, hidden_dim: int, num_heads=4, dropout=0.3, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.w_gat = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.a_src = nn.Parameter(torch.Tensor(num_heads, self.head_dim, 1))
        self.a_dst = nn.Parameter(torch.Tensor(num_heads, self.head_dim, 1))
        self.w_out = nn.Linear(hidden_dim, hidden_dim)

        self.leaky_relu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w_gat.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
        nn.init.xavier_uniform_(self.w_out.weight)
        if self.w_out.bias is not None:
            nn.init.zeros_(self.w_out.bias)

    def forward(self, x: torch.Tensor, adj_mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: (B, T, N, H)
        adj_mask: Optional (N, N) binary/attention connectivity mask
        """
        B, T, N, H = x.shape
        x_flat = x.view(B * T, N, H)

        h = self.w_gat(x_flat).view(B * T, N, self.num_heads, self.head_dim)
        h = h.permute(0, 2, 1, 3)  # (BT, heads, N, head_dim)

        attn_src = torch.matmul(h, self.a_src)  # (BT, heads, N, 1)
        attn_dst = torch.matmul(h, self.a_dst)  # (BT, heads, N, 1)
        attn = attn_src + attn_dst.transpose(-2, -1)  # (BT, heads, N, N)
        attn = self.leaky_relu(attn)

        if adj_mask is not None:
            mask = (adj_mask == 0).unsqueeze(0).unsqueeze(0)
            attn = attn.masked_fill(mask, -1e9)

        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, h)
        out = out.permute(0, 2, 1, 3).contiguous().view(B * T, N, H)
        out = self.w_out(out)
        out = F.relu(out)

        return (out + x_flat).view(B, T, N, H)


class SpatioTemporalBlock(nn.Module):
    """
    Interleaved Spatio-Temporal Block:
    Temporal Dilated Conv -> Diffusion Graph Conv -> Graph Attention Refinement.
    """
    def __init__(self, hidden_dim: int, diffusion_steps=2, num_heads=4,
                 dilation_rates=(1, 2, 4, 8), kernel_size=2, dropout=0.3):
        super(SpatioTemporalBlock, self).__init__()
        self.temporal_conv = TemporalDilatedConv(
            hidden_dim=hidden_dim,
            dilation_rates=dilation_rates,
            kernel_size=kernel_size,
            dropout=dropout
        )
        self.diffusion_conv = DiffusionGraphConv(
            hidden_dim=hidden_dim,
            diffusion_steps=diffusion_steps
        )
        self.graph_attention = GraphAttentionLayer(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, transition_matrices: list, adj_mask: torch.Tensor = None) -> torch.Tensor:
        res = x
        x_t = self.temporal_conv(x)
        x_s = self.diffusion_conv(x_t, transition_matrices)
        x_att = self.graph_attention(x_s, adj_mask)
        out = self.norm(x_att + res)
        out = self.dropout(out)
        return out


class BiLSTMAggregator(nn.Module):
    """
    Stage 5: Compact Bidirectional LSTM Temporal Aggregator.
    Aggregates long-range temporal dynamics, capturing congestion buildup, dissipation, and spillbacks.
    """
    def __init__(self, input_dim: int, hidden_dim: int, dropout=0.3):
        super(BiLSTMAggregator, self).__init__()
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, T, N, H)
        B, T, N, H = x.shape
        x_in = x.permute(0, 2, 1, 3).contiguous().view(B * N, T, H)
        lstm_out, _ = self.bilstm(x_in)  # (B * N, T, 2 * H)
        lstm_out = self.dropout(lstm_out)
        out = self.proj(lstm_out)        # (B * N, T, H)
        out = out.view(B, N, T, H).permute(0, 2, 1, 3).contiguous()
        return out


class PredictionHeads(nn.Module):
    r"""
    Stage 6: Dual Output Heads:
    - Mean Head: Point prediction of traffic speed \hat{\mu}_t
    - Aleatoric Head: Predicted variance \hat{\sigma}^2_t = softplus(\hat{s}_t)
    """
    def __init__(self, hidden_dim: int, horizon: int, dropout=0.3):
        super(PredictionHeads, self).__init__()
        self.horizon = horizon
        self.dropout = nn.Dropout(dropout)
        
        self.mean_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon)
        )
        
        self.variance_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon)
        )

    def forward(self, h: torch.Tensor):
        """
        h: (B, N, H) representation at the last time step
        Returns:
            mu: (B, horizon, N)
            var_aleatoric: (B, horizon, N)
        """
        h = self.dropout(h)
        mu_raw = self.mean_head(h)           # (B, N, horizon)
        var_raw = self.variance_head(h)     # (B, N, horizon)
        
        var_aleatoric = F.softplus(var_raw) + 1e-6
        
        mu = mu_raw.permute(0, 2, 1)
        var_aleatoric = var_aleatoric.permute(0, 2, 1)
        return mu, var_aleatoric
