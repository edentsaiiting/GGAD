import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias

        return self.act(out)


class Model(nn.Module):
    def __init__(self, n_in, n_h, activation, use_gcn=True,
                 highpass_ref='none'):
        super(Model, self).__init__()
        self.use_gcn = use_gcn
        # High-pass detector layer (after GCN, before FC head): subtract a
        # reference from each node rep to surface the deviation (anomaly)
        # signal. Reference:
        #   'neighbor' -> mean over the node's neighbors (local graph high-pass)
        #   'labeled'  -> mean over all labeled training nodes (global prototype)
        #   'none'     -> disabled. When on, the FC head consumes [h, h-ref].
        # Applied uniformly to the GRPE train path and the inference path;
        # vanilla's fc4 ego-centric path is left untouched.
        self.highpass_ref = highpass_ref
        if use_gcn:
            # Vanilla GGAD: two GCN message-passing layers → embedding dim = n_h
            self.gcn1 = GCN(n_in, n_h, activation)
            self.gcn2 = GCN(n_h, n_h, activation)
            feat_dim = n_h
        else:
            # fc_only detector: features consumed directly (raw or GRPE
            # embeddings) → embedding dim = n_in
            feat_dim = n_in
        # Downstream classifier — input dim depends on encoder + high-pass concat.
        head_in = feat_dim * 2 if highpass_ref != 'none' else feat_dim
        self.fc1 = nn.Linear(head_in, int(n_h / 2), bias=False)
        self.fc2 = nn.Linear(int(n_h / 2), int(n_h / 4), bias=False)
        self.fc3 = nn.Linear(int(n_h / 4), 1, bias=False)
        # Ego-centric neighbor aggregation transform — same dim as encoder output
        self.fc4 = nn.Linear(feat_dim, feat_dim, bias=False)
        self.act = nn.ReLU()

    def _highpass(self, emb, adj, labeled_idx, sparse=False):
        """High-pass aggregator layer: inserted AFTER the GCN blocks and
        BEFORE fc1. Concatenates [h, h-ref] so the FC head sees both the
        representation and its deviation from a reference. emb: [1, N, h]
        -> [1, N, h] (ref='none') or [1, N, 2h]."""
        if self.highpass_ref == 'neighbor':
            # Local graph high-pass: ref = (normalized) neighbor mean = adj @ emb.
            if sparse:
                ref = torch.unsqueeze(torch.spmm(adj, torch.squeeze(emb, 0)), 0)
            else:
                ref = torch.bmm(adj, emb)
            return torch.cat([emb, emb - ref], dim=-1)
        elif self.highpass_ref == 'labeled':
            # Global high-pass: ref = mean over labeled training nodes (fixed
            # reference applied to both train and eval nodes).
            lab = labeled_idx if len(labeled_idx) > 0 else [0]
            ref = emb[:, lab, :].mean(dim=1, keepdim=True)   # [1, 1, h]
            return torch.cat([emb, emb - ref], dim=-1)
        return emb

    def forward(self, seq1, adj, sample_abnormal_idx, normal_idx, train_flag, args, sparse=False):
        # seq1: [1, n_nodes, n_in],  adj: [1, n_nodes, n_nodes]
        n_nodes = seq1.shape[1]
        valid_abnormal_idx = [i for i in sample_abnormal_idx if 0 <= i < n_nodes]
        valid_normal_idx   = [i for i in normal_idx           if 0 <= i < n_nodes]

        if self.use_gcn:
            # Graph-aware encoder: two GCN layers → [1, n_nodes, n_h]
            h_1 = self.gcn1(seq1, adj, sparse)
            emb = self.gcn2(h_1,  adj, sparse)
        else:
            # fc_only: pass features through unchanged → [1, n_nodes, n_in]
            emb = seq1

        if len(valid_abnormal_idx) == 0:
            valid_abnormal_idx = [0]

        if train_flag:
            if len(valid_normal_idx) == 0:
                valid_normal_idx = [0]

            if getattr(args, 'use_grpe_encoder', False):
                # GRPE encoder path: discriminative signal is pre-baked into
                # the features by the upstream encoder/generation stage.
                # Detector is a pure node-wise classifier over the full
                # graph embedding — no noise, no ego-centric reconstruction,
                # no emb_combine. run.py gathers labeled rows for BCE.
                emb_abnormal = emb[:, valid_abnormal_idx, :]
                emb_con = None
                emb_combine = None
                # High-pass layer after GCN, before fc1.
                h_in = self._highpass(emb, adj, valid_normal_idx + valid_abnormal_idx, sparse)
                logits = self.fc3(self.act(self.fc2(self.act(self.fc1(h_in)))))
            else:
                # Noise on the abnormal target for ego-centric reconstruction
                emb_abnormal = emb[:, valid_abnormal_idx, :]
                noise = torch.randn(emb_abnormal.size(), device=emb_abnormal.device) * args.var + args.mean
                emb_abnormal = emb_abnormal + noise

                # Ego-centric reconstruction: aggregate neighbors of each abnormal node, then transform
                neigh_adj = adj[0, valid_abnormal_idx, :]        # [n_abn, n_nodes]
                emb_con   = torch.mm(neigh_adj, emb[0, :, :])   # [n_abn, n_h]
                emb_con   = self.act(self.fc4(emb_con))          # [n_abn, n_h]
                emb_combine = torch.cat((emb[:, valid_normal_idx, :], emb_con.unsqueeze(0)), dim=1)

                # Replace abnormal slots in emb AFTER computing logits (following GGAD-git)
                idx0 = torch.zeros(len(valid_abnormal_idx), dtype=torch.long, device=emb.device)
                idx1 = torch.tensor(valid_abnormal_idx, dtype=torch.long, device=emb.device)
                emb = emb.index_put((idx0, idx1), emb_con)

                # Logits from emb_combine only: [1, n_normal+n_abn, 1]
                logits = self.fc3(self.act(self.fc2(self.act(self.fc1(emb_combine)))))
        else:
            emb_abnormal = emb[:, valid_abnormal_idx, :]
            emb_con     = None
            emb_combine = None
            # Inference: classify all N nodes (same high-pass layer as training).
            h_in = self._highpass(emb, adj, valid_normal_idx + valid_abnormal_idx, sparse)
            logits = self.fc3(self.act(self.fc2(self.act(self.fc1(h_in)))))

        return emb, emb_combine, logits, emb_con, emb_abnormal
