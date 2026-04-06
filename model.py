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
    def __init__(self, n_in, n_h, activation, negsamp_round, readout, use_gcn=True):
        super(Model, self).__init__()
        self.use_gcn = use_gcn
        if use_gcn:
            # Vanilla GGAD: two GCN message-passing layers → embedding dim = n_h
            self.gcn1 = GCN(n_in, n_h, activation)
            self.gcn2 = GCN(n_h, n_h, activation)
            feat_dim = n_h
        else:
            # Diffusion detector: raw features fed directly → embedding dim = n_in
            feat_dim = n_in
        # Downstream classifier — input dim depends on encoder
        self.fc1 = nn.Linear(feat_dim, int(n_h / 2), bias=False)
        self.fc2 = nn.Linear(int(n_h / 2), int(n_h / 4), bias=False)
        self.fc3 = nn.Linear(int(n_h / 4), 1, bias=False)
        # Ego-centric neighbor aggregation transform — same dim as encoder output
        self.fc4 = nn.Linear(feat_dim, feat_dim, bias=False)
        self.act = nn.ReLU()

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
            # Diffusion detector: pass raw features through unchanged → [1, n_nodes, n_in]
            emb = seq1

        if len(valid_abnormal_idx) == 0:
            valid_abnormal_idx = [0]
        emb_abnormal = emb[:, valid_abnormal_idx, :]   # [1, n_abn, n_h]
        noise = torch.randn(emb_abnormal.size(), device=emb_abnormal.device) * args.var + args.mean
        emb_abnormal = emb_abnormal + noise

        if train_flag:
            # Ego-centric reconstruction: aggregate neighbors of each abnormal node, then transform
            neigh_adj = adj[0, valid_abnormal_idx, :]        # [n_abn, n_nodes]
            emb_con   = torch.mm(neigh_adj, emb[0, :, :])   # [n_abn, n_h]
            emb_con   = self.act(self.fc4(emb_con))          # [n_abn, n_h]

            if len(valid_normal_idx) == 0:
                valid_normal_idx = [0]
            # emb_combine: [1, n_normal+n_abn, n_h] — labeled nodes only (following GGAD-git)
            emb_combine = torch.cat((emb[:, valid_normal_idx, :], emb_con.unsqueeze(0)), dim=1)

            # Logits from emb_combine only: [1, n_normal+n_abn, 1] (following GGAD-git)
            logits = self.fc3(self.act(self.fc2(self.act(self.fc1(emb_combine)))))

            # Replace abnormal slots in emb AFTER computing logits (following GGAD-git)
            idx0 = torch.zeros(len(valid_abnormal_idx), dtype=torch.long, device=emb.device)
            idx1 = torch.tensor(valid_abnormal_idx, dtype=torch.long, device=emb.device)
            emb = emb.index_put((idx0, idx1), emb_con)
        else:
            emb_con     = None
            emb_combine = None
            # Inference: classify all N nodes
            logits = self.fc3(self.act(self.fc2(self.act(self.fc1(emb)))))

        return emb, emb_combine, logits, emb_con, emb_abnormal
