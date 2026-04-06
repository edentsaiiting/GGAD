import torch
from loss_guided_diffusion import AnomalyGenerator
from utils import pretrain_gcn_embeddings


class DiffusionGenerator:
    """Phases 0-1c: GCN pre-training + diffusion-based synthetic abnormal node generation."""

    def __init__(self, features, adj, raw_adj,
                 abnormal_label_idx, normal_label_idx,
                 all_abnormal_label_idx, args):
        self.features = features
        self.adj = adj
        self.raw_adj = raw_adj
        self.abnormal_label_idx = abnormal_label_idx
        self.normal_label_idx = normal_label_idx
        self.all_abnormal_label_idx = all_abnormal_label_idx
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.gcn_emb = None          # set in pretrain_gcn()
        self.gcn_emb_device = None   # set in pretrain_gcn()
        self.generator = None        # set in train_denoiser()

    # -----------------------------------------------------------------
    # PHASE 0
    # -----------------------------------------------------------------
    def pretrain_gcn(self):
        """GCN pre-training → gcn_emb [1, N, n_h] on CPU."""
        print("\n" + "=" * 80)
        print("PHASE 0: GCN Pre-training (loss_margin + loss_rec)")
        print("=" * 80)
        self.gcn_emb = pretrain_gcn_embeddings(
            self.features, self.adj, self.raw_adj,
            self.abnormal_label_idx, self.normal_label_idx, self.args,
            all_abnormal_label_idx=self.all_abnormal_label_idx,
        )
        self.gcn_emb_device = self.gcn_emb[0].to(self.device)

    # -----------------------------------------------------------------
    # PHASE 1a
    # -----------------------------------------------------------------
    def train_denoiser(self):
        """Train diffusion denoiser on GCN seed embeddings (EDMLoss)."""
        print("\n" + "=" * 80)
        print("PHASE 1a: Diffusion denoiser training on GCN seed embeddings (EDMLoss)")
        print("=" * 80)

        self.abnormal_seed_embs = self.gcn_emb_device[
            torch.tensor(self.abnormal_label_idx, device=self.device)]

        self.generator = AnomalyGenerator(
            embedding_dim=self.args.embedding_dim,
            hidden_dim=self.args.diff_hidden_dim,
            lr=self.args.lr,
            num_gen_epochs=self.args.gen_epochs,
            proto_alpha=0.5,
            device=self.device,
        )
        self.generator.initialize()

        self.args.num_gen_samples = int(len(self.abnormal_label_idx))
        print(f"Training on {len(self.abnormal_label_idx)} seed embeddings "
              f"(dim={self.args.embedding_dim})...")
        self.generator.train_unconditional(self.abnormal_seed_embs)

    # -----------------------------------------------------------------
    # PHASE 1b
    # -----------------------------------------------------------------
    def guided_sampling(self):
        """Guided reverse diffusion → synthetic_nodes [n_seeds, n_h]."""
        print("\n" + "=" * 80)
        print("PHASE 1b: Guided reverse diffusion from GCN seed embeddings")
        print("=" * 80)

        normal_embs = self.gcn_emb_device[self.normal_label_idx].detach()

        synthetic_nodes = self.generator.generate(
            num_samples=self.args.num_gen_samples,
            num_steps=self.args.num_steps,
            guidance_fn=self._ggad_guidance_loss,
            guidance_scale=self.args.loss_guidance_weight,
            normal_embs=normal_embs,
            seed_embeddings=self.abnormal_seed_embs,
        )
        print(f"Guided outliers shape: {synthetic_nodes.shape}")
        return synthetic_nodes

    # -----------------------------------------------------------------
    # PHASE 1c
    # -----------------------------------------------------------------
    def prepare_features(self, synthetic_nodes):
        """Build train / eval feature tensors [1, N, n_h] on CPU."""
        print("\n" + "=" * 80)
        print("PHASE 1c: Prepare training features (diffusion-augmented)")
        print("=" * 80)

        features_train = self.gcn_emb.clone()
        features_train[0, self.abnormal_label_idx] = synthetic_nodes.cpu()
        print(f"  train features: {features_train.shape} "
              f"(GCN embeddings, abnormals overwritten with guided synthetics)")

        features_eval = self.gcn_emb.clone()
        print(f"  eval features:  {features_eval.shape} "
              f"(raw GCN embeddings, no denoiser projection)")

        print("\n" + "=" * 80)
        print("PHASE 2: FC-only Detector Training on GCN+Diffusion Enhanced Features")
        print("=" * 80)

        return features_train.cpu(), features_eval.cpu()

    # -----------------------------------------------------------------
    # Convenience: run all phases
    # -----------------------------------------------------------------
    def run(self):
        """Execute Phases 0 → 1c. Returns (features_train, features_eval)."""
        self.pretrain_gcn()
        self.train_denoiser()
        synthetic_nodes = self.guided_sampling()
        features_train, features_eval = self.prepare_features(synthetic_nodes)
        self._cleanup()
        return features_train, features_eval

    # -----------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------
    def _cleanup(self):
        del self.generator, self.gcn_emb_device, self.abnormal_seed_embs
        self.generator = None
        self.gcn_emb_device = None
        self.abnormal_seed_embs = None

    def _ggad_guidance_loss(self, noisy_embeddings, noisy_normals, sigma):
        idx_abn = (torch.tensor(self.abnormal_label_idx, device=self.gcn_emb_device.device),)
        feat = self.gcn_emb_device.detach().clone()
        if noisy_normals is not None:
            idx_norm = (torch.tensor(self.normal_label_idx, device=self.gcn_emb_device.device),)
            feat = feat.index_put(idx_norm, noisy_normals)
        feat = feat.index_put(idx_abn, noisy_embeddings)

        feat_inf = torch.pow(torch.norm(feat, dim=-1, keepdim=True), -1)
        feat_inf[torch.isinf(feat_inf)] = 0.
        feat_norm = feat * feat_inf

        raw_adj_0 = self.raw_adj[0]
        abn_set = set(self.all_abnormal_label_idx) if len(self.all_abnormal_label_idx) > 0 else set()
        pure_normal_guide  = [i for i in self.normal_label_idx if i not in abn_set]
        real_abn_in_normal = [i for i in self.normal_label_idx if i in abn_set]
        normal_used   = pure_normal_guide                                    if len(pure_normal_guide)  > 0 else [0]
        abnormal_used = self.abnormal_label_idx + real_abn_in_normal         if len(self.abnormal_label_idx) > 0 else [0]

        def guidance_affinity(idx):
            rows     = feat_norm[idx]
            sim_rows = torch.mm(rows, feat_norm.T)
            adj_rows = raw_adj_0[idx].to(feat_norm.device)
            deg_inv  = torch.pow(adj_rows.sum(1), -1)
            deg_inv[torch.isinf(deg_inv)] = 0.
            return (sim_rows * adj_rows).sum(1) * deg_inv

        loss_margin = 0.7 - (guidance_affinity(normal_used).mean()
                             - guidance_affinity(abnormal_used).mean())

        if self.args.no_guide_rec_term:
            return loss_margin

        neigh_adj    = self.adj[0, self.abnormal_label_idx, :].to(feat.device)
        emb_con      = torch.mm(neigh_adj, feat)
        emb_abnormal = feat[self.abnormal_label_idx]
        loss_rec     = torch.mean(torch.sqrt(torch.sum(
            torch.pow(emb_con - emb_abnormal, 2), 1)))

        return loss_margin + loss_rec
