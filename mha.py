"""
GRPE-style attention primitives used by grpe_encoder.py.

Structurally matches GRPE's FeedForwardNetwork / MultiHeadAttention /
EncoderLayer. Split out of grpe_encoder.py so the attention stack is easy
to reuse or swap (e.g. for a labeled-subset variant at t-finance scale).

Interface supports both self-attention (default) and cross-attention via the
`x_kv` kwarg on `MultiHeadAttention.forward` / `EncoderLayer.forward`. In
self-attention mode shapes are [B, N, D] for x and [B, N, N] for distance /
edge_attr; in cross-attention mode queries are [B, N_q, D], keys/values are
[B, N_k, D], and distance / edge_attr are [B, N_q, N_k].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FFN(nn.Module):
    def __init__(self, hidden_size, ffn_size):
        super().__init__()
        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        return self.layer2(self.gelu(self.layer1(x)))


class MultiHeadAttention(nn.Module):
    """GRPE attention: q*k^T + query_hop + key_hop + query_edge + key_edge
    on the score, plus value-side scatter_add into hop/edge bins post-softmax.
    """

    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.att_size = hidden_size // num_heads
        self.scale = self.att_size ** -0.5
        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.linear_k = nn.Linear(hidden_size, hidden_size)
        self.linear_v = nn.Linear(hidden_size, hidden_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)
        self.output_layer = nn.Linear(hidden_size, hidden_size)

    def _split(self, t, num_slots):
        # [K, D] -> [1, H, K, d_k]
        return t.view(1, num_slots, self.num_heads, self.att_size).transpose(1, 2)

    def forward(
        self,
        x,
        query_hop_emb, query_edge_emb,
        key_hop_emb, key_edge_emb,
        value_hop_emb, value_edge_emb,
        distance, edge_attr,
        mask=None,
        x_kv=None,
    ):
        # x_kv=None  -> self-attention (training path; preserves prior behavior).
        # x_kv given -> cross-attention: queries from x [B,N_q,D], keys/values
        # from x_kv [B,N_k,D]; distance / edge_attr must be [B, N_q, N_k].
        self_attn = x_kv is None
        if self_attn:
            x_kv = x
        B, N_q, _ = x.shape
        _, N_k, _ = x_kv.shape
        H, d = self.num_heads, self.att_size

        q = self.linear_q(x).view(B, N_q, H, d).transpose(1, 2)        # [B,H,N_q,d]
        k = self.linear_k(x_kv).view(B, N_k, H, d).transpose(1, 2)     # [B,H,N_k,d]
        v = self.linear_v(x_kv).view(B, N_k, H, d).transpose(1, 2)
        # Key-side hop/edge bias here uses k indexed by the QUERY position
        # (k_i at score[i,j], not k_j) -- matches the existing self-attn logic.
        # In self-attn, k itself is N_q-aligned (N_q = N_k); reuse it.
        # In cross-attn, recompute linear_k(x) so the bias gather has the right
        # N_q-aligned shape; preserves training-time self-attn semantics exactly.
        k_q_bias = k if self_attn else (
            self.linear_k(x).view(B, N_q, H, d).transpose(1, 2)
        )

        n_hop = query_hop_emb.shape[0]
        n_edge = query_edge_emb.shape[0]

        qh = self._split(query_hop_emb, n_hop)
        qe = self._split(query_edge_emb, n_edge)
        kh = self._split(key_hop_emb, n_hop)
        ke = self._split(key_edge_emb, n_edge)
        vh = self._split(value_hop_emb, n_hop)
        ve = self._split(value_edge_emb, n_edge)

        dist_idx = distance.unsqueeze(1).expand(B, H, N_q, N_k)
        edge_idx = edge_attr.unsqueeze(1).expand(B, H, N_q, N_k)

        query_hop = torch.gather(
            torch.matmul(q, qh.transpose(2, 3)), 3, dist_idx
        )
        query_edge = torch.gather(
            torch.matmul(q, qe.transpose(2, 3)), 3, edge_idx
        )
        key_hop = torch.gather(
            torch.matmul(k_q_bias, kh.transpose(2, 3)), 3, dist_idx
        )
        key_edge = torch.gather(
            torch.matmul(k_q_bias, ke.transpose(2, 3)), 3, edge_idx
        )

        scores = torch.matmul(q, k.transpose(2, 3))
        scores = scores + query_hop + key_hop + query_edge + key_edge
        scores = scores * self.scale

        if mask is not None:
            scores = scores.masked_fill(
                mask.view(B, 1, 1, N_k), float("-inf")
            )

        attn = F.softmax(scores, dim=3)
        attn = self.att_dropout(attn)

        v_hop_bin = torch.zeros(
            B, H, N_q, n_hop, device=x.device, dtype=attn.dtype
        )
        v_hop_bin = v_hop_bin.scatter_add(3, dist_idx, attn)
        v_edge_bin = torch.zeros(
            B, H, N_q, n_edge, device=x.device, dtype=attn.dtype
        )
        v_edge_bin = v_edge_bin.scatter_add(3, edge_idx, attn)

        out = (
            torch.matmul(attn, v)
            + torch.matmul(v_hop_bin, vh)
            + torch.matmul(v_edge_bin, ve)
        )
        out = out.transpose(1, 2).contiguous().view(B, N_q, H * d)
        out = self.output_layer(out)
        if mask is not None and self_attn:
            # Self-attn only: zero padding-query rows. Cross-attn never pads
            # queries (chunk sizes are explicit), so this branch is skipped.
            out = out.masked_fill(mask.view(B, N_q, 1), 0.0)
        return out


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout, attention_dropout, num_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = MultiHeadAttention(hidden_size, attention_dropout, num_heads)
        self.drop1 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.ffn = FFN(hidden_size, ffn_size)
        self.drop2 = nn.Dropout(dropout)

    def forward(
        self, x, qh, qe, kh, ke, vh, ve, distance, edge_attr,
        mask=None, x_kv=None,
    ):
        # Self-attn: x_kv=None. Cross-attn: x_kv = anchor activation [B, N_k, D].
        # ln1 is applied to BOTH sides (queries and keys/values) so projections
        # see normalized inputs, matching the training pattern where k/v come
        # from ln1(x) too.
        y = self.ln1(x)
        y_kv = None if x_kv is None else self.ln1(x_kv)
        y = self.attn(
            y, qh, qe, kh, ke, vh, ve, distance, edge_attr,
            mask=mask, x_kv=y_kv,
        )
        x = x + self.drop1(y)
        y = self.ln2(x)
        y = self.ffn(y)
        x = x + self.drop2(y)
        return x
