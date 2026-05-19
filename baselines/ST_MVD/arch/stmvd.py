"""ST-MVD: Multi-View Decomposition for Spatio-Temporal Forecasting.

A graph-free MLP framework that applies multi-view decomposition along three axes:
  (1) Input axis:    parameter-free frequency-band decomposition
  (2) Relational axis: random orthogonal fingerprints with K views
  (3) Temporal axis: parallel multi-view branches with independent ToD/DoW

Each branch processes the same base representation through a different temporal
embedding, and outputs are linearly combined for the final prediction.
"""

import torch
import torch.nn as nn

from .mlp import MultiLayerPerceptron


def build_orthogonal_fingerprints(num_nodes: int, num_views: int, seed: int,
                                  fingerprint_dim: int | None = None) -> torch.Tensor:
    """Build K fingerprint matrices.

    fingerprint_dim is None: random orthogonal N x N (rows orthonormal in R^N).
    fingerprint_dim = d:     random Gaussian N x d, entries N(0, 1/d) -- JL random projection.
                             Pairwise row geometry preserved up to (1+/-eps) for d = O(log N / eps^2).

    Returns: (num_views, N, d) where d defaults to N.
    """
    generator = torch.Generator().manual_seed(seed)
    views = []
    for _ in range(num_views):
        if fingerprint_dim is None:
            q, _ = torch.linalg.qr(torch.randn(num_nodes, num_nodes, generator=generator))
            views.append(q)
        else:
            g = torch.randn(num_nodes, fingerprint_dim, generator=generator) / (fingerprint_dim ** 0.5)
            views.append(g)
    return torch.stack(views, dim=0)


class STMVD(nn.Module):
    """ST-MVD model.

    Args:
        num_nodes: number of sensor nodes N.
        input_len: input sequence length T.
        output_len: prediction horizon T_out.
        node_dim: per-view node embedding dimension (each of K views).
        num_views: number of orthogonal fingerprint views K.
        temp_dim_tid: time-of-day embedding dimension.
        temp_dim_diw: day-of-week embedding dimension.
        time_of_day_size: number of ToD slots (e.g., 288 for 5-min intervals).
        day_of_week_size: 7.
        if_D_i_W: whether to use DoW embedding (disable if DoW signal is noisy).
        num_layer: number of MLP layers per temporal branch.
        num_temporal_branches: number of parallel temporal branches.
        temporal_branch_dim: per-branch output dimension.
        fingerprint_seed: seed for random orthogonal fingerprint.
        fingerprint_dim: if None, use N x N orthogonal fingerprints (default).
            If an int d, use N x d random Gaussian fingerprints (JL random projection).
    """

    def __init__(self,
                 num_nodes: int,
                 input_len: int = 12,
                 output_len: int = 12,
                 node_dim: int = 42,
                 num_views: int = 4,
                 temp_dim_tid: int = 32,
                 temp_dim_diw: int = 32,
                 time_of_day_size: int = 288,
                 day_of_week_size: int = 7,
                 if_T_i_D: bool = True,
                 if_D_i_W: bool = True,
                 num_layer: int = 3,
                 num_temporal_branches: int = 2,
                 temporal_branch_dim: int = 16,
                 fingerprint_seed: int = 42,
                 fingerprint_dim: int | None = None,
                 mlp_dropout: float = 0.15):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_len = input_len
        self.output_len = output_len
        self.node_dim = node_dim
        self.num_views = num_views
        self.temp_dim_tid = temp_dim_tid
        self.temp_dim_diw = temp_dim_diw
        self.time_of_day_size = time_of_day_size
        self.day_of_week_size = day_of_week_size
        self.if_time_in_day = if_T_i_D
        self.if_day_in_week = if_D_i_W
        self.num_layer = num_layer
        self.num_temporal_branches = num_temporal_branches
        self.temporal_branch_dim = temporal_branch_dim
        self.fingerprint_dim = fingerprint_dim

        # ---- Relational axis: random orthogonal fingerprints (frozen) ----
        # fingerprint_dim is None -> N x N orthogonal; int -> N x d Gaussian (JL).
        fp_in_dim = num_nodes if fingerprint_dim is None else fingerprint_dim
        fingerprints = build_orthogonal_fingerprints(num_nodes, num_views, fingerprint_seed, fingerprint_dim)
        self.register_buffer('node_fingerprints', fingerprints)

        # Shared interpreter MLPs: fp_in_dim -> node_dim
        self.interpreter_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fp_in_dim, node_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(node_dim, node_dim),
            )
            for _ in range(num_views)
        ])
        self.relational_dim = node_dim * num_views

        # ---- Input axis: parameter-free FFT-based decomposition ----
        # 5 views: node_mean + DC + low + mid + high
        self.num_freq_views = 5
        self.ts_dim = self.num_freq_views * input_len

        # ---- Hidden dim ----
        self.hidden_dim = (self.ts_dim
                          + self.relational_dim
                          + temp_dim_tid * int(if_T_i_D)
                          + temp_dim_diw * int(if_D_i_W))

        # ---- Temporal axis: parallel temporal branches ----
        # Each branch: independent ToD/DoW embedding + 3-layer MLP + linear out
        self.temporal_branches = nn.ModuleList()
        self.temporal_tod_embeddings = nn.ParameterList()
        self.temporal_dow_embeddings = nn.ParameterList()
        for _ in range(num_temporal_branches):
            self.temporal_branches.append(
                self._build_branch(self.hidden_dim, num_layer, temporal_branch_dim, mlp_dropout)
            )
            if if_T_i_D:
                tod_emb = nn.Parameter(torch.empty(time_of_day_size, temp_dim_tid))
                nn.init.xavier_uniform_(tod_emb)
                self.temporal_tod_embeddings.append(tod_emb)
            if if_D_i_W:
                dow_emb = nn.Parameter(torch.empty(day_of_week_size, temp_dim_diw))
                nn.init.xavier_uniform_(dow_emb)
                self.temporal_dow_embeddings.append(dow_emb)

        # Final regression: concat of branch outputs -> output_len
        self.regression_layer = nn.Linear(num_temporal_branches * temporal_branch_dim, output_len)

    @staticmethod
    def _build_branch(in_dim: int, num_layer: int, out_dim: int, dropout: float) -> nn.ModuleDict:
        """Branch = num_layer MLP blocks + final linear projection to out_dim."""
        layers = nn.ModuleList([
            MultiLayerPerceptron(in_dim, in_dim, dropout=dropout) for _ in range(num_layer)
        ])
        out_proj = nn.Linear(in_dim, out_dim)
        return nn.ModuleDict({'layers': layers, 'out_proj': out_proj})

    # ---------- Input-axis decomposition ----------
    def _decompose_input(self, flow: torch.Tensor) -> list[torch.Tensor]:
        """Parameter-free frequency decomposition.

        Input: flow of shape (B, T, N).
        Returns 5 views, each (B, T, N):
          [node_mean, DC, low (bins 1-2), mid (bins 3-4), high (bins 5-F)]
        """
        T = flow.shape[1]
        # (1) Shared temporal rhythm across nodes
        node_mean = flow.mean(dim=2, keepdim=True).expand_as(flow)
        # (2) Per-node deviation from the shared rhythm
        deviation = flow - node_mean
        dev_fft = torch.fft.rfft(deviation, dim=1)  # (B, F, N)
        F_dim = dev_fft.shape[1]
        # (3) Split deviation by frequency band
        band_ranges = [(0, 1), (1, 3), (3, 5), (5, F_dim)]
        bands = []
        for lo, hi in band_ranges:
            mask = torch.zeros(F_dim, device=flow.device)
            mask[lo:min(hi, F_dim)] = 1.0
            band = torch.fft.irfft(dev_fft * mask.unsqueeze(-1), n=T, dim=1)
            bands.append(band)
        return [node_mean] + bands  # 5 views

    # ---------- Relational-axis decomposition ----------
    def _compute_node_embedding(self) -> torch.Tensor:
        """Apply interpreter MLPs to the K orthogonal fingerprints.

        Returns: (N, num_views * node_dim) relational embedding.
        """
        view_embs = []
        for v in range(self.num_views):
            fp = self.node_fingerprints[v]  # (N, N)
            view_embs.append(self.interpreter_mlps[v](fp))  # (N, node_dim)
        return torch.cat(view_embs, dim=-1)  # (N, V*node_dim)

    # ---------- Forward ----------
    def forward(self, history_data: torch.Tensor, future_data=None,
                batch_seen=None, epoch=None, **kwargs) -> torch.Tensor:
        """
        Args:
            history_data: (B, T, N, C). Channel 0 = flow, 1 = ToD index (0-1), 2 = DoW index (0-1).

        Returns:
            prediction: (B, T_out, N, 1).
        """
        # Flow in shape (B, T, N)
        flow = history_data[..., 0]
        batch_size = flow.shape[0]

        # === Input axis: FFT decomposition -> 5 views ===
        views = self._decompose_input(flow)  # list of (B, T, N)
        views_stacked = torch.stack(views, dim=-1)  # (B, T, N, 5)
        frequency_features = views_stacked.permute(0, 3, 1, 2).reshape(batch_size, self.ts_dim, self.num_nodes, 1)

        # === Relational axis: fingerprints + interpreter MLPs ===
        node_emb = self._compute_node_embedding()  # (N, relational_dim)
        relational_features = (node_emb.unsqueeze(0)
                               .expand(batch_size, -1, -1)
                               .transpose(1, 2)
                               .unsqueeze(-1))  # (B, relational_dim, N, 1)

        # Combine frequency history and relational identity before adding branch-specific temporal embeddings.
        node_base_features = torch.cat([frequency_features, relational_features], dim=1)
        node_base_features = node_base_features.squeeze(-1).transpose(1, 2)  # (B, N, D_base)

        # === Temporal axis: parallel temporal branches ===
        # Lookup ToD/DoW indices from the last input timestep
        if self.if_time_in_day:
            tod_idx = (history_data[:, -1, :, 1] * self.time_of_day_size).long()  # (B, N)
        if self.if_day_in_week:
            dow_idx = (history_data[:, -1, :, 2] * self.day_of_week_size).long()  # (B, N)

        branch_preds = []
        for v, branch in enumerate(self.temporal_branches):
            temporal_parts = []
            if self.if_time_in_day:
                temporal_parts.append(self.temporal_tod_embeddings[v][tod_idx])  # (B, N, temp_dim_tid)
            if self.if_day_in_week:
                temporal_parts.append(self.temporal_dow_embeddings[v][dow_idx])  # (B, N, temp_dim_diw)
            branch_input = torch.cat([node_base_features] + temporal_parts, dim=-1)  # (B, N, hidden_dim)

            # (B, N, D) -> (B, D, N, 1) for Conv2d-based MLPs
            h = branch_input.transpose(1, 2).unsqueeze(-1)
            for layer in branch['layers']:
                h = layer(h)
            h = h.squeeze(-1).transpose(1, 2)  # (B, N, D)
            pred = branch['out_proj'](h)  # (B, N, temporal_branch_dim)
            branch_preds.append(pred)

        branch_outputs = torch.cat(branch_preds, dim=-1)  # (B, N, V_t*temporal_branch_dim)
        prediction = self.regression_layer(branch_outputs)  # (B, N, T_out)
        return prediction.transpose(1, 2).unsqueeze(-1)  # (B, T_out, N, 1)
