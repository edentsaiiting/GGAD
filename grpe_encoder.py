"""
GRPE-style encoder for GGAD (full-graph, single-batch).

Phase 0 of the GRPE -> Diffusion -> Detector pipeline: trains a graph
Transformer with relative-positional biases (shortest-path hop + edge-type)
on raw node features, supervised by BCE at labeled nodes only. The learned
per-node embeddings replace raw features as the input to the downstream
diffusion augmentor and/or detector.

Differs from the original GRPE (grpe/model/graph_self_attention.py):
  * No TASK token -- GGAD is node-level, not graph-level. The TASK_DISTANCE
    and TASK_EDGE slots are still allocated in the embedding tables so
    vocab sizes stay compatible with GRPE checkpoints, but they are never
    indexed at forward time.
  * Linear node embedding (continuous raw features, not categorical atom
    types).
  * Edge-type table parameterizes the GGAD label-pair scheme:
        0: EDGE_GENERIC       -- adjacent, at least one endpoint unlabeled
        1: EDGE_LABELED_NN    -- both endpoints labeled normal
        2: EDGE_LABELED_AA    -- both endpoints labeled abnormal
        3: EDGE_LABELED_NA    -- labeled normal <-> labeled abnormal
    (SELF_EDGE / NO_EDGE / TASK_EDGE reserved slots appended as per GRPE
    convention.)

Topology: shortest-path hop distance, clamped at max_hop, with -1 ->
UNREACHABLE_DISTANCE.
Edge-type: distance-gated to hop=1 (any pair with hop != 1 is forced to
NO_EDGE regardless of the edge_type_matrix contents, mirroring GRPE
exactly).

Integration: run.py Phase 0 (and iter_cotrain.py per-iter encoder steps)
read hyperparams from grpe_config.json into GRPEConfig and call
train_grpe_encoder(); the returned [1, N, d_model] embeddings replace raw
features downstream.
"""

import os
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse.csgraph import shortest_path

from mha import EncoderLayer


# ---------------------------------------------------------------------------
# Encoder loss (ggad_subset_losses: GGAD three-term supervised loss)
# ---------------------------------------------------------------------------
def ggad_subset_losses(
    emb_l: torch.Tensor,
    logits_l: torch.Tensor,
    labels: torch.Tensor,
    adj_sub: torch.Tensor,
    deg_inv: torch.Tensor,
    n_normal: int,
    fc4: nn.Module,
    bce: nn.Module,
    *,
    confidence_margin: float = 0.7,
    noise_mean: float = 0.0,
    noise_var: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """GGAD's three supervised terms (BCE + affinity-margin + ego-centric
    reconstruction) on the encoder's LABELED SUBSET — a port of run.py's
    vanilla detector loss, with the [L, L] disclosed-subgraph ("known
    partial topo") standing in for the full graph the detector sees.

    Layout: subset rows are normals first (0..n_normal-1) then abnormals
    (n_normal..L-1), matching `labeled_list` order in `train_grpe_encoder`.

    Args:
        emb_l:    [L, d] final-LN embeddings of the labeled subset.
        logits_l: [L]    BCE-head logits for the same subset.
        labels:   [L]    {0=normal, 1=abnormal}.
        adj_sub:  [L, L] float subset adjacency (binary, self-loops stripped).
        deg_inv:  [L]    inverse subset degree (0 where the node is isolated).
        n_normal: number of leading normal rows.
        fc4:      learnable Linear(d, d) — the reconstruction "perturbation".
        bce:      BCEWithLogitsLoss instance (same object as the BCE term).

    Returns (loss, loss_bce, loss_margin, loss_rec).

    Nodes with no neighbor inside the subset are masked from the margin and
    reconstruction means: their affinity is structurally 0 (no gradient) and
    reconstructing them toward fc4(0) would be degenerate. BCE still covers
    every labeled node. On sparse-subgraph datasets few labeled pairs are
    adjacent, so the margin/rec terms are correspondingly weak — by design,
    since only the disclosed topology is available at encoder-train time.
    """
    n_labeled = emb_l.shape[0]
    has_nbr = deg_inv > 0

    # --- BCE: identical to the previous encoder loss ---
    loss_bce = bce(logits_l, labels)

    # --- Local affinity margin: normals cohere with neighbors, abnormals don't ---
    emb_norm = F.normalize(emb_l, dim=-1)                  # [S, d]
    sim = emb_norm @ emb_norm.t()                          # [S, S]
    affinity = (sim * adj_sub).sum(1) * deg_inv            # [S] mean nbr cos-sim
    normal_mask = torch.zeros_like(has_nbr)
    normal_mask[:n_normal] = True
    abn_mask = torch.zeros_like(has_nbr)
    abn_mask[n_normal:n_labeled] = True
    nm = normal_mask & has_nbr
    am = abn_mask & has_nbr
    aff_normal = affinity[nm].mean() if nm.any() else affinity.new_zeros(())
    aff_abn = affinity[am].mean() if am.any() else affinity.new_zeros(())
    loss_margin = (confidence_margin - (aff_normal - aff_abn)).clamp_min(0.0)

    # --- Ego-centric reconstruction: abnormal emb ~ fc4(mean of its neighbors) ---
    abn_idx = torch.arange(n_normal, n_labeled, device=emb_l.device)
    row_mean_adj = adj_sub * deg_inv.unsqueeze(1)          # [L, L] rows sum to 1 (or 0)
    neigh_mean = row_mean_adj[abn_idx] @ emb_l             # [n_abn, d] neighbor mean
    emb_con = F.relu(fc4(neigh_mean))                      # learnable perturbation
    emb_abnormal = emb_l[abn_idx]
    if noise_var != 0.0 or noise_mean != 0.0:
        emb_abnormal = emb_abnormal + (
            torch.randn_like(emb_abnormal) * noise_var + noise_mean
        )
    rec_mask = has_nbr[abn_idx]
    if rec_mask.any():
        diff = torch.pow(emb_con[rec_mask] - emb_abnormal[rec_mask], 2)
        loss_rec = torch.mean(torch.sqrt(torch.sum(diff, dim=1) + 1e-12))
    else:
        loss_rec = emb_l.new_zeros(())

    loss = loss_margin + loss_bce + loss_rec
    return loss, loss_bce, loss_margin, loss_rec


# ---------------------------------------------------------------------------
# Edge-type & distance matrix builders
# ---------------------------------------------------------------------------
# User-defined edge-type slot layout. Reserved GRPE specials (TASK/SELF/NO)
# are appended by GRPEEncoder at the end of the vocab.
# NB: kept at 6 (slots 4-5 historically held removed context edge types) so
# embedding-table sizes and weight-init RNG draws stay bit-identical to the
# campaign runs. Do not shrink without accepting a reproducibility break.
NUM_USER_EDGE_TYPES = 6
EDGE_GENERIC = 0
EDGE_LABELED_NN = 1
EDGE_LABELED_AA = 2
EDGE_LABELED_NA = 3


def build_edge_type_matrix(
    adj_binary, normal_label_idx, abnormal_label_idx
) -> torch.Tensor:
    """[N, N] long tensor of edge-type IDs at adjacent pairs, -1 elsewhere.

    -1 stands in for "no edge" (encoder remaps to NO_EDGE). Adjacent pairs
    default to EDGE_GENERIC; pairs whose BOTH endpoints fall in the supplied
    labeled index sets get an EDGE_LABELED_{NN,AA,NA} ID.
    """
    if isinstance(adj_binary, torch.Tensor):
        adj = adj_binary.detach().cpu()
    else:
        adj = torch.as_tensor(np.asarray(adj_binary))
    N = adj.shape[-1]
    adj = adj.reshape(N, N)

    ea = torch.full((N, N), -1, dtype=torch.long)
    rows, cols = (adj > 0).nonzero(as_tuple=True)
    ea[rows, cols] = EDGE_GENERIC

    is_n = torch.zeros(N, dtype=torch.bool)
    is_a = torch.zeros(N, dtype=torch.bool)
    if len(normal_label_idx) > 0:
        is_n[list(normal_label_idx)] = True
    if len(abnormal_label_idx) > 0:
        is_a[list(abnormal_label_idx)] = True

    r_n, r_a = is_n[rows], is_a[rows]
    c_n, c_a = is_n[cols], is_a[cols]
    nn_mask = r_n & c_n
    aa_mask = r_a & c_a
    na_mask = (r_n & c_a) | (r_a & c_n)
    ea[rows[nn_mask], cols[nn_mask]] = EDGE_LABELED_NN
    ea[rows[aa_mask], cols[aa_mask]] = EDGE_LABELED_AA
    ea[rows[na_mask], cols[na_mask]] = EDGE_LABELED_NA
    return ea


def build_distance_matrix(
    adj_binary,
    max_hop: int,
    indices: Optional[np.ndarray] = None,
    dst_indices: Optional[np.ndarray] = None,
) -> torch.Tensor:
    """Shortest-path hop distance over an undirected unweighted graph.

    Returns a long tensor where disconnected pairs are -1 (encoder remaps
    to UNREACHABLE_DISTANCE). Values > max_hop are NOT clamped here; the
    encoder clamps via .clamp(max=max_hop) to match GRPE.

    Index modes:
      * `indices=None, dst_indices=None`:  full [N, N] SPD.
      * `indices=src,   dst_indices=None`: square subset SPD [L, L] with
            rows = cols = `src`. BFS restricted to `src` sources.
      * `indices=src,   dst_indices=dst`:  rectangular SPD [|src|, |dst|]
            for cross-attention (queries x anchors). BFS still traverses
            the full graph -- pair distances reflect true graph structure.

    Cost note: scipy runs one BFS per source index, so cost scales with
    the number of SOURCE indices. The graph here is undirected, so the
    SPD is symmetric and we are free to BFS from whichever of
    {src, dst} is smaller and orient the result accordingly. On dense
    graphs (e.g. t_finance, ~21M edges) the inference call has
    |src| = #non-anchor nodes (~38k) >> |dst| = #anchors (~hundreds to
    a few thousand), so BFS-from-dst is ~10-68x cheaper for an identical
    result.
    """
    # Accept sparse adjacency directly (skips the dense N x N materialization
    # otherwise needed to thread a torch tensor / numpy array into scipy).
    if sp.issparse(adj_binary):
        adj_sparse = adj_binary if adj_binary.format == 'csr' else adj_binary.tocsr()
    else:
        if isinstance(adj_binary, torch.Tensor):
            adj_np = adj_binary.detach().cpu().numpy()
        else:
            adj_np = np.asarray(adj_binary)
        adj_np = np.asarray(adj_np).squeeze()
        adj_sparse = sp.csr_matrix((adj_np > 0).astype(np.float32))
    if indices is None:
        dist = shortest_path(adj_sparse, directed=False, unweighted=True)
    elif dst_indices is None:
        # Square subset [L, L]: BFS from the L sources, slice to the same cols.
        indices = np.asarray(indices, dtype=np.int64)
        dist = shortest_path(
            adj_sparse, directed=False, unweighted=True, indices=indices
        )
        dist = np.ascontiguousarray(dist[:, indices])
    else:
        # Rectangular [|src|, |dst|]. Undirected => symmetric SPD, so BFS
        # from the smaller index set and orient to [|src|, |dst|].
        indices = np.asarray(indices, dtype=np.int64)
        dst_indices = np.asarray(dst_indices, dtype=np.int64)
        if len(dst_indices) <= len(indices):
            # BFS from dst (anchors): [|dst|, N] -> take src cols -> transpose
            d = shortest_path(
                adj_sparse, directed=False, unweighted=True, indices=dst_indices
            )
            dist = np.ascontiguousarray(d[:, indices].T)  # [|src|, |dst|]
        else:
            d = shortest_path(
                adj_sparse, directed=False, unweighted=True, indices=indices
            )
            dist = np.ascontiguousarray(d[:, dst_indices])  # [|src|, |dst|]
    dist_f = torch.from_numpy(dist)
    unreach = torch.isinf(dist_f) | (dist_f < 0)
    dist_f[unreach] = -1.0
    return dist_f.long()


# ---------------------------------------------------------------------------
# Config + model
# ---------------------------------------------------------------------------
@dataclass
class GRPEConfig:
    d_model: int = 80
    num_layer: int = 4
    nhead: int = 8
    ffn_dim: Optional[int] = None  # defaults to d_model
    max_hop: int = 5
    dropout: float = 0.1
    attention_dropout: float = 0.1
    perturb_noise: float = 0.0
    num_epoch: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-3
    # Encoder loss = GGAD three terms (BCE + affinity-margin + ego-centric
    # reconstruction); see ggad_subset_losses.
    # Ego-centric reconstruction noise on the abnormal target (mirrors GGAD's
    # per-dataset mean/var; 0/0 = the GGAD default, i.e. fc4 transform only).
    noise_mean: float = 0.0
    noise_var: float = 0.0
    # Cross-attention inference batch size (queries per chunk, attending to
    # all L anchors). Tune for GPU memory: scores tensor is
    # [B=1, H, chunk_size, L] fp32, ~ 8 * chunk_size * L * 4 B per layer.
    chunk_size: int = 8000


class GRPEEncoder(nn.Module):
    """Full-graph GRPE encoder without TASK token.

    Forward returns (node_emb, logits):
        node_emb: [B, N, d_model]  per-node output of the final LayerNorm
        logits:   [B, N]           anomaly score from BCE head (Linear)
    """

    def __init__(
        self,
        in_dim: int,
        d_model: int = 80,
        num_layer: int = 4,
        nhead: int = 8,
        ffn_dim: Optional[int] = None,
        max_hop: int = 5,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        perturb_noise: float = 0.0,
    ):
        super().__init__()
        ffn_dim = ffn_dim if ffn_dim is not None else d_model
        self.d_model = d_model
        self.max_hop = max_hop
        self.perturb_noise = perturb_noise

        # Distance vocab: 0..max_hop real hops + TASK_DISTANCE + UNREACHABLE.
        # Edge vocab:     0..U-1 user types + TASK_EDGE + SELF_EDGE + NO_EDGE.
        # (TASK_* slots reserved for checkpoint compatibility; never indexed
        #  because we dropped the TASK token.)
        # NUM_USER_EDGE_TYPES is the single source of truth, shared with
        # build_edge_type_matrix — do not override per-instance.
        self.TASK_DISTANCE = max_hop + 1
        self.UNREACHABLE_DISTANCE = max_hop + 2
        self.num_dist_slots = max_hop + 3

        self.TASK_EDGE = NUM_USER_EDGE_TYPES + 1
        self.SELF_EDGE = NUM_USER_EDGE_TYPES + 2
        self.NO_EDGE = NUM_USER_EDGE_TYPES + 3
        self.num_edge_slots = NUM_USER_EDGE_TYPES + 4

        self.node_emb = nn.Linear(in_dim, d_model)

        self.query_hop_emb = nn.Embedding(self.num_dist_slots, d_model)
        self.key_hop_emb = nn.Embedding(self.num_dist_slots, d_model)
        self.value_hop_emb = nn.Embedding(self.num_dist_slots, d_model)
        self.query_edge_emb = nn.Embedding(self.num_edge_slots, d_model)
        self.key_edge_emb = nn.Embedding(self.num_edge_slots, d_model)
        self.value_edge_emb = nn.Embedding(self.num_edge_slots, d_model)

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, ffn_dim, dropout, attention_dropout, nhead)
            for _ in range(num_layer)
        ])
        self.final_ln = nn.LayerNorm(d_model)
        self.cls_head = nn.Linear(d_model, 1)
        # Ego-centric reconstruction transform (GGAD fc4): consumed by the
        # three-term subset loss in train_grpe_encoder, not by forward().
        self.fc4 = nn.Linear(d_model, d_model, bias=False)

    def _prepare_indices(
        self, distance: torch.Tensor, edge_attr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Remap raw distance / edge_attr into embedding-table indices,
        mirroring GRPE's construction exactly (including the hop=1 gate that
        overwrites the diagonal's SELF_EDGE assignment to NO_EDGE)."""
        B, N, _ = distance.shape

        # Distance: clamp at max_hop, -1 -> UNREACHABLE.
        dist = distance.clamp(max=self.max_hop).clone()
        dist[distance < 0] = self.UNREACHABLE_DISTANCE

        # Edge-attr: diag -> SELF_EDGE, -1 -> NO_EDGE, hop!=1 -> NO_EDGE.
        ea = edge_attr.clone()
        diag = torch.arange(N, device=ea.device)
        ea[:, diag, diag] = self.SELF_EDGE
        ea[ea == -1] = self.NO_EDGE
        ea[distance != 1] = self.NO_EDGE
        return dist, ea

    def forward(
        self,
        x: torch.Tensor,
        distance: torch.Tensor,
        edge_attr: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x:         [B, N, in_dim]
        distance:  [B, N, N] long, raw SPD (-1 = unreachable)
        edge_attr: [B, N, N] long, user edge-type ids (-1 = no edge)
        mask:      [B, N] bool (True = padding), optional
        """
        h = self.node_emb(x)
        if self.training and self.perturb_noise > 0:
            h = h + torch.empty_like(h).uniform_(
                -self.perturb_noise, self.perturb_noise
            )

        dist_idx, edge_idx = self._prepare_indices(distance, edge_attr)

        for layer in self.layers:
            h = layer(
                h,
                self.query_hop_emb.weight, self.query_edge_emb.weight,
                self.key_hop_emb.weight, self.key_edge_emb.weight,
                self.value_hop_emb.weight, self.value_edge_emb.weight,
                dist_idx, edge_idx, mask=mask,
            )
        h = self.final_ln(h)
        logits = self.cls_head(h).squeeze(-1)
        return h, logits

    def forward_anchors_capture(
        self,
        x: torch.Tensor,
        distance: torch.Tensor,
        edge_attr: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[list, torch.Tensor]:
        """Self-attention forward over anchors that records per-layer activations.

        Returns:
            anchor_states: list of len num_layer; anchor_states[l] is the
                hidden state going INTO layer l (i.e., the output of layer
                l-1's residual / FFN block, or node_emb(x) for l=0). These
                are exactly the K/V sources cross-attention should use at
                each matched layer to mirror the training compute pattern.
            final_emb: [B, L, d_model] post-final-LN output (= what
                `forward` would return as `node_emb`).
        """
        h = self.node_emb(x)
        # No perturb_noise: this path is only called in eval/no_grad after train.
        dist_idx, edge_idx = self._prepare_indices(distance, edge_attr)

        anchor_states = []
        for layer in self.layers:
            anchor_states.append(h)
            h = layer(
                h,
                self.query_hop_emb.weight, self.query_edge_emb.weight,
                self.key_hop_emb.weight, self.key_edge_emb.weight,
                self.value_hop_emb.weight, self.value_edge_emb.weight,
                dist_idx, edge_idx, mask=mask,
            )
        final_emb = self.final_ln(h)
        return anchor_states, final_emb

    def forward_cross_chunk(
        self,
        x_q: torch.Tensor,
        distance_q_kv: torch.Tensor,
        anchor_states: list,
    ) -> torch.Tensor:
        """Cross-attention forward for a chunk of query nodes against anchors.

        Edge bias is silenced ("option (i)"): edge_attr at every (q, k) pair
        is set to NO_EDGE, since the bin scheme requires both endpoints'
        labels and queries are unlabeled. The hop pathway is fully alive --
        adjacency / multi-hop topology still informs attention.

        Args:
            x_q: [B, N_q, in_dim] raw query features (e.g., features at
                idx_val / idx_test).
            distance_q_kv: [B, N_q, L] long, raw SPD between queries and
                anchors (-1 for unreachable). Anchor order must match the
                anchor_states tensors.
            anchor_states: list returned by `forward_anchors_capture`.
                Length must equal num_layer.

        Returns:
            [B, N_q, d_model] post-final-LN embeddings for the query chunk.
        """
        assert len(anchor_states) == len(self.layers), (
            f"anchor_states len {len(anchor_states)} != num_layer {len(self.layers)}"
        )
        h_q = self.node_emb(x_q)

        # Distance index: clamp at max_hop, -1 -> UNREACHABLE.
        dist = distance_q_kv.clamp(max=self.max_hop).clone()
        dist[distance_q_kv < 0] = self.UNREACHABLE_DISTANCE

        # Edge index under option (i): all NO_EDGE. Bypasses _prepare_indices
        # (which assumes a square N x N with self-loop diagonal -- not
        # meaningful in cross-attn).
        B, N_q, N_k = distance_q_kv.shape
        edge = torch.full(
            (B, N_q, N_k),
            self.NO_EDGE,
            dtype=torch.long,
            device=h_q.device,
        )

        for layer, anchor_h in zip(self.layers, anchor_states):
            h_q = layer(
                h_q,
                self.query_hop_emb.weight, self.query_edge_emb.weight,
                self.key_hop_emb.weight, self.key_edge_emb.weight,
                self.value_hop_emb.weight, self.value_edge_emb.weight,
                dist, edge, mask=None, x_kv=anchor_h,
            )
        return self.final_ln(h_q)


# ---------------------------------------------------------------------------
# Phase 0 training wrapper
# ---------------------------------------------------------------------------
def _strip_self_loops(adj_dense: np.ndarray) -> np.ndarray:
    adj_bin = (adj_dense > 0).astype(np.float32)
    np.fill_diagonal(adj_bin, 0.0)
    return adj_bin


def _strip_self_loops_sparse(adj_sparse: sp.spmatrix) -> sp.csr_matrix:
    """Sparse counterpart of `_strip_self_loops` -- never materializes [N, N]."""
    a = adj_sparse.tocsr().copy()
    a.setdiag(0)
    a.eliminate_zeros()
    # Binarize while staying sparse. Use astype(float32) on the boolean result
    # so downstream `(adj > 0).astype(np.float32)` paths also work.
    return (a != 0).astype(np.float32)


# ---------------------------------------------------------------------------
# SPD cache (per dataset/seed/max_hop) -- BFS from many sources is the dominant
# cost on large graphs (e.g. ~1-3h on t_finance). Caching turns re-runs into
# tens of seconds. Cache verifies the source/dest indices and an adj-shape/nnz
# fingerprint to catch stale entries when splits or graphs change.
# ---------------------------------------------------------------------------
def _spd_cache_path(cache_dir: str, cache_key: str, kind: str, max_hop: int) -> str:
    return os.path.join(cache_dir, f"{cache_key}_{kind}_h{max_hop}.pt")


def _adj_meta(adj) -> dict:
    if sp.issparse(adj):
        return {"shape": tuple(adj.shape), "nnz": int(adj.nnz)}
    arr = np.asarray(adj)
    return {"shape": tuple(arr.shape), "nnz": int((arr > 0).sum())}


def _try_load_spd_cache(
    path: str,
    expected_indices: np.ndarray,
    expected_dst_indices: Optional[np.ndarray],
    expected_adj_meta: dict,
) -> Optional[torch.Tensor]:
    if not os.path.exists(path):
        return None
    try:
        blob = torch.load(path, map_location="cpu")
    except Exception:
        return None
    if blob.get("adj_meta") != expected_adj_meta:
        return None
    if not np.array_equal(blob.get("indices"), expected_indices):
        return None
    cached_dst = blob.get("dst_indices")
    if expected_dst_indices is None:
        if cached_dst is not None:
            return None
    else:
        if cached_dst is None or not np.array_equal(cached_dst, expected_dst_indices):
            return None
    return blob.get("distance")


def _save_spd_cache(
    path: str,
    distance: torch.Tensor,
    indices: np.ndarray,
    dst_indices: Optional[np.ndarray],
    adj_meta: dict,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "distance": distance,
        "indices": np.asarray(indices),
        "dst_indices": None if dst_indices is None else np.asarray(dst_indices),
        "adj_meta": adj_meta,
    }, path)


def train_grpe_encoder(
    features: torch.Tensor,
    raw_adj: torch.Tensor,
    normal_label_idx: Iterable[int],
    abnormal_label_idx: Iterable[int],
    *,
    cfg: Optional[GRPEConfig] = None,
    device: Optional[torch.device] = None,
    verbose: bool = True,
    inference_idx: Optional[Iterable[int]] = None,
    raw_adj_sparse: Optional[sp.spmatrix] = None,
    cache_key: Optional[str] = None,
    cache_dir: str = "./cache/grpe_spd",
) -> Tuple[GRPEEncoder, torch.Tensor]:
    """Pretrain a GRPE encoder with BCE on labeled nodes, return embeddings.

    TRAINING. Attention runs on the LABELED SUBSET only -- the [L, L] block
    of features / SPD / edge_attr where
    L = |normal_label_idx| + |abnormal_label_idx|. Hop distances between
    labeled pairs still reflect the full graph (BFS traverses unlabeled
    nodes), preserving structural context.

    INFERENCE. The labeled (anchor) rows of the returned embedding come
    from the trained self-attention forward. If `inference_idx` is given,
    those nodes' embeddings come from cross-attention against the anchors
    (option (i): edge_attr forced to NO_EDGE; hop pathway fully alive).
    All other nodes (training-only-unlabeled, when present) receive only
    the bare-linear projection `model.node_emb(raw)`. This matches the
    semi-supervised contract: nodes that the downstream phase 2 actually
    evaluates (idx_val / idx_test) get the cross-attention treatment;
    nodes that only serve as graph-structure context for phase-2 GCN
    message passing fall through to the cheaper linear projection.

    Args:
        features: [1, N, ft_size] float (as prepared by GGAD's run.py).
        raw_adj:  [1, N, N] float, adjacency with self-loops added
            (self-loops are stripped internally before SPD so hop=1 only
            marks true graph edges).
        normal_label_idx, abnormal_label_idx: iterables of node indices
            that participate in BCE supervision. Their union defines the
            subset of nodes that MHA attends over during training.
        inference_idx: optional iterable of node indices to embed via
            cross-attention at inference time (typically idx_val ∪
            idx_test). If None, no cross-attention is run; non-anchor
            rows are filled via bare-linear node_emb.
        raw_adj_sparse: optional scipy.sparse adjacency (with self-loops,
            same convention as `raw_adj`). If provided, the dense [N, N]
            materialization is skipped end-to-end -- BFS runs on the
            sparse CSR directly, saving ~3*N*N*4 bytes of allocations
            and the O(N^2) `csr_matrix(dense)` constructor cost.
        cache_key: identifier used to scope the on-disk SPD cache
            (e.g. f"{dataset}_seed{seed}"). When set, anchor SPD ([L, L])
            and inference SPD ([|inference|, L]) are loaded from / saved
            to `cache_dir`. Re-runs with the same key + max_hop + index
            sets + adjacency fingerprint skip the BFS (the dominant cost
            on dense graphs like t_finance). When None, caching is off.
        cache_dir: directory for cached SPD tensors. Created on save.

    Returns (model, embeddings [1, N, d_model]).
    """
    cfg = cfg or GRPEConfig()
    device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    N = features.shape[1]
    ft_size = features.shape[-1]
    # Prefer sparse adjacency: avoids three [N, N] dense allocations
    # (raw_np, adj_bin, the (adj > 0).astype(float32) intermediate inside
    # build_distance_matrix). Falls back to the dense path if no sparse
    # matrix is supplied -- preserves the prior calling contract.
    if raw_adj_sparse is not None:
        adj_bin = _strip_self_loops_sparse(raw_adj_sparse)
    else:
        raw_np = raw_adj.squeeze(0).detach().cpu().numpy()
        adj_bin = _strip_self_loops(raw_np)

    normal_list = list(normal_label_idx)
    abnormal_list = list(abnormal_label_idx)
    labeled_list = normal_list + abnormal_list  # order: all N then all A
    L = len(labeled_list)
    labeled_np = np.asarray(labeled_list, dtype=np.int64)

    # Attention subset == the labeled set.
    subset_np = labeled_np
    S = len(subset_np)

    # SPD cache fingerprint: shape + nnz of the binarized adjacency. Cheap
    # to compute, catches dataset / preprocessing changes (any change in
    # nnz invalidates the cache). cache_key adds dataset/seed scope.
    adj_fp = _adj_meta(adj_bin)

    if verbose:
        print(f"[GRPE] building subset SPD / edge-type for L={L} labeled (N={N})...")
    # Subset-restricted SPD: BFS from subset sources only. Cache to disk
    # because BFS from K sources is the dominant cost on large graphs.
    distance = None
    anchor_cache_path = None
    if cache_key:
        anchor_cache_path = _spd_cache_path(
            cache_dir, cache_key, "anchor", cfg.max_hop
        )
        distance = _try_load_spd_cache(
            anchor_cache_path, subset_np, None, adj_fp
        )
        if distance is not None and verbose:
            print(f"[GRPE] anchor SPD loaded from cache: {anchor_cache_path}")
    if distance is None:
        distance = build_distance_matrix(
            adj_bin, max_hop=cfg.max_hop, indices=subset_np
        )  # [S, S]
        if anchor_cache_path is not None:
            _save_spd_cache(
                anchor_cache_path, distance, subset_np, None, adj_fp
            )
            if verbose:
                print(f"[GRPE] anchor SPD cached at: {anchor_cache_path}")
    # Edge-type on the [S, S] sub-adjacency; label sets re-expressed as
    # positions within the subset (normal = 0..Nn-1, abnormal = Nn..L-1).
    if sp.issparse(adj_bin):
        # Two-step CSR/CSC slice; densify the [S, S] block (small, ~MBs).
        adj_sub = adj_bin[subset_np, :][:, subset_np].toarray()
    else:
        adj_sub = adj_bin[np.ix_(subset_np, subset_np)]
    normal_in_sub = range(len(normal_list))
    abnormal_in_sub = range(len(normal_list), L)
    edge_attr = build_edge_type_matrix(
        adj_sub, normal_in_sub, abnormal_in_sub
    )  # [L, L]; labeled<->labeled -> NN/AA/NA

    # Subset adjacency + inverse degree for the affinity-margin / ego-centric
    # reconstruction terms (the "known partial topo"). adj_sub is already
    # binary with self-loops stripped (it comes from adj_bin); deg_inv turns
    # the neighbor sum into a mean. Fixed across epochs, so computed once.
    adj_sub_t = torch.as_tensor(
        np.asarray(adj_sub), dtype=torch.float32, device=device
    )
    deg_inv_sub = torch.pow(adj_sub_t.sum(1), -1)
    deg_inv_sub[torch.isinf(deg_inv_sub)] = 0.0

    distance = distance.unsqueeze(0).to(device)
    edge_attr = edge_attr.unsqueeze(0).to(device)

    # subset == labeled set; rows 0..Nn-1 normal, Nn..L-1 abnormal.
    x_full = features.to(device).float()                      # [1, N, ft]
    subset_t = torch.as_tensor(subset_np, device=device)
    x_sub = x_full.index_select(1, subset_t).contiguous()     # [1, S, ft]

    model = GRPEEncoder(
        in_dim=ft_size,
        d_model=cfg.d_model,
        num_layer=cfg.num_layer,
        nhead=cfg.nhead,
        ffn_dim=cfg.ffn_dim,
        max_hop=cfg.max_hop,
        dropout=cfg.dropout,
        attention_dropout=cfg.attention_dropout,
        perturb_noise=cfg.perturb_noise,
    ).to(device)

    labels = torch.cat([
        torch.zeros(len(normal_list), device=device),
        torch.ones(len(abnormal_list), device=device),
    ])

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    bce = nn.BCEWithLogitsLoss()

    if verbose:
        print(
            f"[GRPE] pretrain subset-MHA: L={L} "
            f"({len(normal_list)}N+{len(abnormal_list)}A), "
            f"d_model={cfg.d_model}, layers={cfg.num_layer}, "
            f"heads={cfg.nhead}, max_hop={cfg.max_hop}, "
            f"epochs={cfg.num_epoch}"
        )

    model.train()
    for ep in range(cfg.num_epoch):
        opt.zero_grad()
        emb_l, logits = model(x_sub, distance, edge_attr)  # emb_l:[1,L,d] logits:[1,L]
        logits_l = logits[0][:L]
        # GGAD three-term supervised loss (BCE + affinity-margin + ego-centric
        # reconstruction) on the labeled subset.
        loss, loss_bce, loss_margin, loss_rec = ggad_subset_losses(
            emb_l[0], logits_l, labels, adj_sub_t, deg_inv_sub,
            len(normal_list), model.fc4, bce,
            noise_mean=cfg.noise_mean, noise_var=cfg.noise_var,
        )
        loss.backward()
        opt.step()
        if verbose and (ep % 10 == 0 or ep == cfg.num_epoch - 1):
            with torch.no_grad():
                acc = ((logits_l.sigmoid() > 0.5).float() == labels).float().mean().item()
            print(
                f"[GRPE] ep {ep:03d}/{cfg.num_epoch}  loss={loss.item():.4f}  "
                f"bce={loss_bce.item():.4f}  margin={loss_margin.item():.4f}  "
                f"rec={loss_rec.item():.4f}  lab_acc={acc:.3f}"
            )

    model.eval()
    with torch.no_grad():
        # 1. Anchor self-attention forward -- captures per-layer hidden states
        #    so cross-attention at each layer can use the layer-matched
        #    anchor activation as K/V (mirrors training compute pattern).
        anchor_states, emb_sub = model.forward_anchors_capture(
            x_sub, distance, edge_attr
        )

        # 2. Allocate emb_full as zeros. Only anchors and inference_idx
        #    rows are filled by the encoder. Training-only-unlabeled rows
        #    (idx_train minus anchors, when inference_idx omits them) stay
        #    zero -- they aren't supervised, indexed for eval, or used by
        #    the FC-only detector. With GCN they contribute zero to
        #    neighbor aggregation, which is the desired "no bare-linear
        #    nonsense" semantics.
        emb_full = torch.zeros(
            1, N, cfg.d_model, device=device, dtype=emb_sub.dtype
        )

        # 3. Fill labeled rows with the trained self-attn embeddings.
        emb_full[:, subset_t, :] = emb_sub

        # 4. Cross-attention over `inference_idx` (typically idx_val ∪
        #    idx_test), chunked to fit GPU memory.
        if inference_idx is not None:
            inf_np = np.asarray(list(inference_idx), dtype=np.int64)
            # Drop any inference indices that overlap the subset (cross-attn
            # is for non-subset queries; subset rows already have emb_sub).
            anchor_set = set(subset_np.tolist())
            inf_np = np.asarray(
                [i for i in inf_np.tolist() if i not in anchor_set],
                dtype=np.int64,
            )
            n_inf = len(inf_np)

            if n_inf > 0:
                if verbose:
                    print(
                        f"[GRPE] cross-attn inference: {n_inf} nodes vs "
                        f"S={S} anchors, chunk_size={cfg.chunk_size}"
                    )
                # Precompute SPD between inference queries and anchors once
                # ([n_inf, S]); kept on CPU and sliced per chunk to GPU.
                # Cache: BFS from |inf_np| sources (~ tens of thousands)
                # is the single biggest cost on t_finance.
                dist_inf = None
                inf_cache_path = None
                if cache_key:
                    inf_cache_path = _spd_cache_path(
                        cache_dir, cache_key, "inference", cfg.max_hop
                    )
                    dist_inf = _try_load_spd_cache(
                        inf_cache_path, inf_np, subset_np, adj_fp
                    )
                    if dist_inf is not None and verbose:
                        print(
                            f"[GRPE] inference SPD loaded from cache: "
                            f"{inf_cache_path}"
                        )
                if dist_inf is None:
                    dist_inf = build_distance_matrix(
                        adj_bin,
                        max_hop=cfg.max_hop,
                        indices=inf_np,
                        dst_indices=subset_np,
                    )
                    if inf_cache_path is not None:
                        _save_spd_cache(
                            inf_cache_path, dist_inf, inf_np, subset_np, adj_fp
                        )
                        if verbose:
                            print(
                                f"[GRPE] inference SPD cached at: "
                                f"{inf_cache_path}"
                            )

                for start in range(0, n_inf, cfg.chunk_size):
                    end = min(start + cfg.chunk_size, n_inf)
                    chunk_ids = inf_np[start:end]
                    chunk_t = torch.as_tensor(
                        chunk_ids, device=device, dtype=torch.long
                    )
                    x_q = x_full.index_select(1, chunk_t).contiguous()
                    dist_chunk = (
                        dist_inf[start:end].unsqueeze(0).to(device)
                    )
                    emb_chunk = model.forward_cross_chunk(
                        x_q, dist_chunk, anchor_states
                    )
                    emb_full[:, chunk_t, :] = emb_chunk
                    if verbose:
                        print(
                            f"[GRPE] cross-attn chunk {start}:{end} done"
                        )

    return model, emb_full.detach()
