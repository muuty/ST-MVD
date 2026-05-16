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


def build_orthogonal_fingerprints(num_nodes: int, num_views: int, seed: int) -> torch.Tensor:
    """Build K random orthogonal fingerprint matrices, each (N, N).

    Returns: (num_views, N, N), one row per node and view.
    """
    generator = torch.Generator().manual_seed(seed)
    views = []
    for _ in range(num_views):
        q, _ = torch.linalg.qr(torch.randn(num_nodes, num_nodes, generator=generator))
        views.append(q)
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
        num_layer: number of MLP layers per decoder branch.
        mvt_num_views: number of parallel temporal branches V.
        mvt_out_dim: per-branch output dimension.
        fingerprint_seed: seed for random orthogonal fingerprint.
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
                 mvt_num_views: int = 2,
                 mvt_out_dim: int = 16,
                 fingerprint_seed: int = 42,
                 mlp_dropout: float = 0.15,
                 **kwargs):
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
        self.mvt_num_views = mvt_num_views
        self.mvt_out_dim = mvt_out_dim

        # ---- Relational axis: random orthogonal fingerprints (frozen) ----
        fingerprints = build_orthogonal_fingerprints(num_nodes, num_views, fingerprint_seed)
        self.register_buffer('node_fingerprints', fingerprints)

        # Shared interpreter MLPs: N-dim code -> node_dim
        self.interpreter_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_nodes, node_dim),
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

        # ---- Temporal axis: parallel multi-view branches ----
        # Each branch: independent ToD/DoW embedding + 3-layer MLP + linear out
        self.mvt_branches = nn.ModuleList()
        self.mvt_tod_embs = nn.ParameterList()
        self.mvt_dow_embs = nn.ParameterList()
        for _ in range(mvt_num_views):
            self.mvt_branches.append(self._build_branch(self.hidden_dim, num_layer, mvt_out_dim, mlp_dropout))
            if if_T_i_D:
                tod_emb = nn.Parameter(torch.empty(time_of_day_size, temp_dim_tid))
                nn.init.xavier_uniform_(tod_emb)
                self.mvt_tod_embs.append(tod_emb)
            if if_D_i_W:
                dow_emb = nn.Parameter(torch.empty(day_of_week_size, temp_dim_diw))
                nn.init.xavier_uniform_(dow_emb)
                self.mvt_dow_embs.append(dow_emb)

        # Final regression: concat of branch outputs -> output_len
        self.regression_layer = nn.Linear(mvt_num_views * mvt_out_dim, output_len)

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
        time_series_emb = views_stacked.permute(0, 3, 1, 2).reshape(batch_size, self.ts_dim, self.num_nodes, 1)

        # === Relational axis: fingerprints + interpreter MLPs ===
        node_emb = self._compute_node_embedding()  # (N, relational_dim)
        node_emb_bct = (node_emb.unsqueeze(0)
                        .expand(batch_size, -1, -1)
                        .transpose(1, 2)
                        .unsqueeze(-1))  # (B, relational_dim, N, 1)

        # Base (without temporal) = concat(flow_views, node_emb)
        base_no_temporal = torch.cat([time_series_emb, node_emb_bct], dim=1)
        base_no_temporal = base_no_temporal.squeeze(-1).transpose(1, 2)  # (B, N, D_base)

        # === Temporal axis: parallel MV branches ===
        # Lookup ToD/DoW indices from the last input timestep
        if self.if_time_in_day:
            tod_idx = (history_data[:, -1, :, 1] * self.time_of_day_size).long()  # (B, N)
        if self.if_day_in_week:
            dow_idx = (history_data[:, -1, :, 2] * self.day_of_week_size).long()  # (B, N)

        branch_preds = []
        for v, branch in enumerate(self.mvt_branches):
            temporal_parts = []
            if self.if_time_in_day:
                temporal_parts.append(self.mvt_tod_embs[v][tod_idx])  # (B, N, temp_dim_tid)
            if self.if_day_in_week:
                temporal_parts.append(self.mvt_dow_embs[v][dow_idx])  # (B, N, temp_dim_diw)
            branch_input = torch.cat([base_no_temporal] + temporal_parts, dim=-1)  # (B, N, hidden_dim)

            # (B, N, D) -> (B, D, N, 1) for Conv2d-based MLPs
            h = branch_input.transpose(1, 2).unsqueeze(-1)
            for layer in branch['layers']:
                h = layer(h)
            h = h.squeeze(-1).transpose(1, 2)  # (B, N, D)
            pred = branch['out_proj'](h)  # (B, N, mvt_out_dim)
            branch_preds.append(pred)

        fused = torch.cat(branch_preds, dim=-1)  # (B, N, V*mvt_out_dim)
        prediction = self.regression_layer(fused)  # (B, N, T_out)
        return prediction.transpose(1, 2).unsqueeze(-1)  # (B, T_out, N, 1)
