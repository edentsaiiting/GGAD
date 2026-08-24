"""Iterative GRPE encoder <-> diffusion augmentor co-training.

Replaces the single-shot Phase 0 + Phase 1 sequence with an iterative
schedule that interleaves encoder pretraining and contrastive diffusion
oversampling. Activated via --use_iter_cotrain in run.py.

Per iteration k (every iter runs BOTH steps, including the last):
  1. ENCODER STEP. Train GRPE on (N_k, A_pool_k) anchors with BCE on
     real labels (N_k -> 0, A_pool_k -> 1). Output: GRPE embeddings
     for every node in the graph (anchors via self-attn, others via
     cross-attn -- mirrors run.py's full-coverage Fix 2 for GCN/aug
     downstream).
  2. DIFFUSION STEP. Pick |A_pool_k| previously-unused real-normal
     positions; replace their feature rows with diffusion-generated
     vectors that are simultaneously far from N_k and near to real-A,
     plus heterophilic to graph neighbors. Flip their labels 0 -> 1
     ("in-place oversampling"). These positions extend A_pool for
     the next iter.

Loop stop rule: at the START of iter k+1, if |N_{k+1}| / |A_pool_{k+1}|
falls below TARGET_N_TO_A_RATIO (default 5.5), we stop without running
iter k+1; the last completed iter k supplies the final state.

Notes
-----
* In-place oversampling at real-normal positions keeps the graph
  topology intact -- GRPE's hop / edge biases still see those rows as
  real graph nodes. Their feature vector and class label flip.
* Real-positive pool (real_abnormal_pool, the leaked anchors) stays fixed
  across iters; the contrastive guidance always pulls toward these anchors.
* Negatives for guidance = N_k (current iter), not the full normal pool.
  The negative geometry tightens as iters progress.
* EDM denoiser is re-initialized + retrained each iter on the current
  encoder embeddings.
* SPD caching: each iter uses a distinct cache_key suffixed with
  iter index, so on-disk SPD files for different iter anchor sets
  do not clobber each other across runs.

Imports only stable surfaces from grpe_encoder.py and
loss_guided_diffusion.py; does not modify the GRPE encoder.
"""

import json
from dataclasses import fields as _dc_fields

import numpy as np
import torch

from grpe_encoder import GRPEConfig, train_grpe_encoder
from loss_guided_diffusion import AnomalyGenerator


GRPE_CONFIG_PATH = './grpe_config.json'

# Stop rule: terminate when |N_k| / |A_pool_k| <= TARGET_N_TO_A_RATIO
# at the START of an iter, i.e., before running that iter.
TARGET_N_TO_A_RATIO = 5.5

# Hard cap on iters; with the geometric A schedule the ratio reaches
# 5.5:1 well before 10, so this is a safety bound.
MAX_ITERS = 10

# Local-affinity guidance hyperparams.
GUIDANCE_MARGIN = 0.7   # margin threshold (matches GGAD's confidence margin)
LAMBDA_AFF = 1.0        # weight on local-affinity term

DIM_SAMPLE_RATE = 0.3   # rho: per-(node,dim) prob a dim carries the abnormal signal


class IterCotrainer:
    """Iterative encoder <-> diffusion co-trainer; see module docstring."""

    def __init__(self, features, raw_adj, raw_adj_sparse,
                 real_normal_pool, real_abnormal_pool, args):
        # CPU canonical features. Mutated in place at pseudo-abn positions
        # after each diffusion step. Initially the raw input features
        # (pre-GRPE); after iter 0's encoder step we overwrite with the
        # GRPE [1, N, d_model] embeddings -- those are what the diffusion
        # then refines and what the final detector consumes.
        self.features_cpu = features.clone()                # [1, N, ft]
        self.raw_adj = raw_adj                              # [1, N, N] dense
        self.raw_adj_sparse = raw_adj_sparse                # scipy sparse (csr)
        self.real_normal_pool = list(real_normal_pool)
        self.real_abnormal_pool = list(real_abnormal_pool)
        self.args = args
        # Raw-rolling ablation: skip the GRPE encoder entirely; rolling
        # generation + detector run on raw features (see run() loop).
        self.skip_encoder = getattr(args, 'iter_skip_encoder', False)
        # rho = fraction of dims replaced with abnormal-pool values per new
        # pseudo seed; the rest keep a shuffled-normal base. See _oversample_step.
        self.dim_sample_rate = DIM_SAMPLE_RATE
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Rolling-schedule knobs (see run.py --gen_target_ratio / --gen_frac_schedule).
        #   doubling mode  : grow A_pool geometrically, stop at |N|/|A| <= target_ratio.
        #   schedule mode  : explicit cumulative abnormal-fraction targets, one
        #                    "one-shot" generation step per entry (overrides doubling).
        self.target_ratio = getattr(args, 'gen_target_ratio', TARGET_N_TO_A_RATIO)
        _sched = (getattr(args, 'gen_frac_schedule', '') or '').strip()
        self.frac_schedule = ([float(x) for x in _sched.split(',') if x.strip()]
                              if _sched else None)
        # Slice-size knob: N_k = first min(1, (k+1)*slice_frac) of the real-normal
        # pool. Default 0.1 == the historical (k+1)/MAX_ITERS rule. 1.0 = the
        # full-pool ablation: the fill target is the WHOLE normal set, and the
        # pseudo positions come out of its tail (N shrinks by num_new).
        self.slice_frac = getattr(args, 'gen_slice_frac', 0.1)
        self.slice_full = self.slice_frac >= 1.0
        # Tag the SPD cache + logs so variants of the same (dataset, seed)
        # don't share a cache file; 'F' marks the full-pool (--gen_slice_frac
        # 1.0) ablation. Do not change the tag string: it keys on-disk caches.
        self.cache_tag = ((_sched.replace(',', '-').replace('.', 'p') + 'S'
                           + ('F' if self.slice_full else ''))
                          if self.frac_schedule is not None
                          else f"gr{self.target_ratio}")

        # Cumulative state across iters.
        self.pseudo_pool = []   # real-normal positions whose features were overridden + label flipped to 1

        # Cached sparse adj on device for affinity-margin guidance (built lazily).
        self._adj_sparse_device = None
        self._adj_deg_inv = None

        # GRPE hyperparams from JSON; tolerate keys that aren't in GRPEConfig.
        with open(GRPE_CONFIG_PATH) as f:
            _raw = json.load(f)
        _allowed = {fd.name for fd in _dc_fields(GRPEConfig)}
        self.grpe_cfg = GRPEConfig(**{k: v for k, v in _raw.items() if k in _allowed})
        self.grpe_cfg.noise_mean = getattr(args, 'mean', 0.0)   # ego-centric recon noise
        self.grpe_cfg.noise_var = getattr(args, 'var', 0.0)

        # Per-iter guidance state -- set by _setup_guidance, read by _iter_guidance_loss.
        self._g_feat = None
        self._g_N_tensor = None
        self._g_A_real_tensor = None
        self._g_A_pool_tensor = None
        self._g_new_pos_tensor = None

    # ------------------------------------------------------------------
    # N_k / A_pool bookkeeping
    # ------------------------------------------------------------------
    def _normal_subset(self, k):
        """N_k = first min(1, (k+1)*slice_frac) of real_normal_pool, excluding
        pseudo_pool. slice_frac defaults to 0.1 == the historical (k+1)/MAX_ITERS."""
        frac = min(1.0, (k + 1) * self.slice_frac)
        n_target = int(frac * len(self.real_normal_pool))
        pseudo_set = set(self.pseudo_pool)
        return [i for i in self.real_normal_pool if i not in pseudo_set][:n_target]

    def _A_pool_current(self):
        return list(self.real_abnormal_pool) + list(self.pseudo_pool)

    def _pick_new_pseudo(self, num_new, N_k):
        """Pick num_new real-normal positions not in N_k and not yet pseudo."""
        used = set(N_k) | set(self.pseudo_pool)
        candidates = [i for i in self.real_normal_pool if i not in used]
        if len(candidates) < num_new:
            return None
        # Take from the tail so the N_k prefix (head of real_normal_pool) stays stable.
        return candidates[-num_new:]

    # ------------------------------------------------------------------
    # Encoder step
    # ------------------------------------------------------------------
    def _encoder_step(self, k, N_k, A_pool, inference_idx):
        cache_key = f"{self.args.dataset}_seed{self.args.seed}_{self.cache_tag}_iter{k}"
        print(f"[iter{k}/enc] GRPE on {len(N_k)}N + {len(A_pool)}A  "
              f"(d_model={self.grpe_cfg.d_model}, epochs={self.grpe_cfg.num_epoch}, "
              f"cache={cache_key})")
        _, grpe_emb = train_grpe_encoder(
            self.features_cpu, self.raw_adj,
            N_k, A_pool,
            cfg=self.grpe_cfg,
            inference_idx=inference_idx,
            raw_adj_sparse=self.raw_adj_sparse,
            cache_key=cache_key,
        )
        return grpe_emb.detach()  # [1, N, d_model] on training device

    # ------------------------------------------------------------------
    # Oversampling step (shared shuffle, optional diffusion refinement)
    # ------------------------------------------------------------------
    def _oversample_step(self, k, grpe_emb_device, N_k, A_pool, num_new):
        """Produce `num_new` new pseudo-abnormal feature rows.

        Pick num_new unused real-normal positions, build dimsample seeds
        (shuffled-normal base, rho dims hard-replaced from the A_pool bank),
        then train an EDM denoiser and run guided reverse diffusion from the
        seeds. Returns (generated [B, ft], positions).
        """
        feat_device = grpe_emb_device[0].clone()
        ft = feat_device.shape[1]

        new_positions = self._pick_new_pseudo(num_new, N_k)
        if new_positions is None:
            return None, None

        head = new_positions[:3]
        tail = "..." if num_new > 3 else ""
        abn_frac = self.dim_sample_rate
        print(f"[iter{k}/diff] sampling {num_new} pseudo-abns at "
              f"positions {head}{tail}  "
              f"(rho={abn_frac:.2f}, |A_pool|={len(A_pool)})")

        # ---- Pseudo-abnormal seed construction (dimsample) ----
        # Applied to each new batch when it is introduced (prior pseudos keep
        # their accumulated, encoder-re-embedded representation). A per-
        # (node,dim) Bernoulli(rho) mask picks abnormal dims filled from an
        # A_pool column-sampled bank; base = row+col-shuffled clean-normal
        # values (off-manifold hard-negative floor).
        new_pos_tensor = torch.tensor(new_positions, device=self.device, dtype=torch.long)
        B = len(new_positions)

        # Abnormal bank: column-sample each dim from the current A_pool.
        abn_pool_t = torch.tensor(A_pool, device=self.device, dtype=torch.long)
        abn_bank = feat_device[abn_pool_t]                            # [|A_pool|, ft]
        col_a = torch.randint(0, abn_bank.shape[0], (B, ft), device=self.device)
        abn_vals = torch.gather(abn_bank, 0, col_a)                   # [B, ft]
        dim_mask = torch.rand(B, ft, device=self.device) < abn_frac  # [B, ft] per (node,dim)

        # Row+col-shuffled normal base, hard abnormal replace. Exclude
        # positions already flipped into pseudo_pool -- their feature rows
        # were overwritten with abnormal seeds in earlier iters, so sampling
        # them would contaminate the "normal" base.
        pseudo_set = set(self.pseudo_pool)
        clean_normal_pool = [i for i in self.real_normal_pool if i not in pseudo_set]
        normal_bank_t = torch.tensor(clean_normal_pool, device=self.device, dtype=torch.long)
        normal_bank = feat_device[normal_bank_t]                  # [|Npool|, ft]
        col_n = torch.randint(0, normal_bank.shape[0], (B, ft), device=self.device)
        base = torch.gather(normal_bank, 0, col_n)                # col-shuffle
        row_perm = torch.argsort(torch.rand(B, ft, device=self.device), dim=1)
        base = torch.gather(base, 1, row_perm)                    # row-permute
        seed = torch.where(dim_mask, abn_vals, base)
        feat_device[new_pos_tensor] = seed

        # EDM denoiser + guided sampling.
        labeled_idx = N_k + A_pool + new_positions
        labeled_t = torch.tensor(labeled_idx, device=self.device, dtype=torch.long)
        seed_feats = feat_device[labeled_t]

        gen = AnomalyGenerator(
            embedding_dim=ft,
            hidden_dim=self.args.diff_hidden_dim,
            lr=self.args.lr,
            num_gen_epochs=self.args.gen_epochs,
            device=self.device,
        )
        gen.initialize()
        gen.train_unconditional(seed_feats)

        self._setup_guidance(feat_device, N_k, A_pool, new_positions)
        N_t = torch.tensor(N_k, device=self.device, dtype=torch.long)
        generated = gen.generate(
            num_samples=len(new_positions),
            num_steps=self.args.num_steps,
            guidance_fn=self._iter_guidance_loss,
            guidance_scale=self.args.loss_guidance_weight,
            normal_embs=feat_device[N_t],
            seed_embeddings=feat_device[new_pos_tensor],
        )

        del gen
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        return generated, new_positions

    # ------------------------------------------------------------------
    # Guidance: margin-style contrastive (far from N_k, near to real A)
    # + local-affinity margin term.
    # ------------------------------------------------------------------
    def _ensure_adj_sparse(self):
        if (self._adj_sparse_device is not None
                and self._adj_sparse_device.device == self.device):
            return
        raw_cpu = self.raw_adj[0]
        if raw_cpu.is_sparse:
            sp_cpu = raw_cpu.coalesce()
            deg_cpu = torch.sparse.sum(sp_cpu, dim=1).to_dense()
        else:
            sp_cpu = raw_cpu.to_sparse().coalesce()
            deg_cpu = raw_cpu.sum(1)
        self._adj_sparse_device = sp_cpu.to(self.device)
        deg_inv = torch.pow(deg_cpu, -1)
        deg_inv[torch.isinf(deg_inv)] = 0.
        self._adj_deg_inv = deg_inv.to(self.device)

    def _setup_guidance(self, feat_device, N_k, A_pool, new_positions):
        self._ensure_adj_sparse()
        self._g_feat = feat_device.detach()
        self._g_N_tensor = torch.tensor(N_k, device=self.device, dtype=torch.long)
        self._g_A_real_tensor = torch.tensor(
            self.real_abnormal_pool, device=self.device, dtype=torch.long)
        self._g_A_pool_tensor = torch.tensor(
            A_pool, device=self.device, dtype=torch.long)
        self._g_new_pos_tensor = torch.tensor(
            new_positions, device=self.device, dtype=torch.long)

    @staticmethod
    def _normalize_rows(feat):
        inv = torch.pow(torch.norm(feat, dim=-1, keepdim=True), -1)
        inv[torch.isinf(inv)] = 0.
        return feat * inv

    # ------------------------------------------------------------------
    # Per-iter sanity signal: mean-cos separation of synthetics from real-A
    # vs N_k in the current feature space; positive sep = guidance bit.
    # ------------------------------------------------------------------
    @staticmethod
    def _cos_mean(x, y):
        """Mean of cos(x_i, y_j) over all pairs i, j."""
        if x.numel() == 0 or y.numel() == 0:
            return float("nan")
        xn = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        yn = y / y.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return (xn @ yn.t()).mean().item()

    def _log_synth_diagnostics(self, k, generated, new_positions, N_k):
        device = generated.device
        real_A_emb = self.features_cpu[0, self.real_abnormal_pool, :].to(device)
        N_k_emb    = self.features_cpu[0, N_k, :].to(device)

        # New batch: the just-generated diffusion output for this iter.
        cos_A_new = self._cos_mean(generated, real_A_emb)
        cos_N_new = self._cos_mean(generated, N_k_emb)
        sep_new   = cos_A_new - cos_N_new

        # Cumulative: union of (a) previous-iter pseudos, which now carry
        # this-iter's GRPE re-embedding from self.features_cpu, and (b)
        # this iter's newly generated batch (still in `generated`, not
        # yet written to features_cpu).
        prior_emb = None
        if len(self.pseudo_pool) > 0:
            prior_emb = self.features_cpu[0, self.pseudo_pool, :].to(device)
        if prior_emb is None:
            cum_emb = generated
        else:
            cum_emb = torch.cat([prior_emb, generated], dim=0)
        cos_A_cum = self._cos_mean(cum_emb, real_A_emb)
        cos_N_cum = self._cos_mean(cum_emb, N_k_emb)
        sep_cum   = cos_A_cum - cos_N_cum

        print(f"[iter{k}/diag] sep(new n={generated.shape[0]})={sep_new:+.4f}  "
              f"sep(cum n={cum_emb.shape[0]})={sep_cum:+.4f}")

    def _iter_guidance_loss(self, noisy_feats, noisy_normals, sigma):
        """Margin contrastive (far from N_k, near to real-A) + local-affinity margin.

        noisy_feats:   [B, D] current noisy new-pseudo-abn batch (grad-carrying).
        noisy_normals: sampler-supplied noise-matched ref for normal_embs
                       (ignored here -- our negatives are the current iter's
                       N_k cached encoder embeddings, not noise-matched).
        sigma:         current noise level (unused).
        """
        # Patch grad-carrying noisy_feats into the cached encoder embeddings at
        # the new pseudo-abn positions; everything else stays detached.
        feat = self._g_feat.clone()
        feat = feat.index_put((self._g_new_pos_tensor,), noisy_feats)
        feat_norm = self._normalize_rows(feat)

        # ---- Margin contrastive: cos(x, N_k) - cos(x, real_A) ----
        x = feat_norm[self._g_new_pos_tensor]                # [B, D]
        N_emb = feat_norm[self._g_N_tensor]                  # [|N_k|, D]
        A_emb = feat_norm[self._g_A_real_tensor]             # [|real_A|, D]
        # Mean cosine sim to each pool, per batch row.
        sim_x_N = (x @ N_emb.t()).mean(dim=1)                # [B]
        sim_x_A = (x @ A_emb.t()).mean(dim=1)                # [B]
        loss_contrastive = (sim_x_N - sim_x_A).mean()

        # ---- Local-affinity margin (graph-aware) ----
        full_agg = torch.sparse.mm(self._adj_sparse_device, feat_norm)  # [N, D]

        def aff_mean(idx_t):
            rows = feat_norm[idx_t]
            agg = full_agg[idx_t]
            per_row = (rows * agg).sum(dim=-1) * self._adj_deg_inv[idx_t]
            return per_row.mean()

        aff_normal = aff_mean(self._g_N_tensor)
        # Abnormal class for the margin = real_A union existing pseudo_pool union the new batch.
        abn_idx = torch.cat([
            self._g_A_real_tensor,
            self._g_A_pool_tensor,
            self._g_new_pos_tensor,
        ]).unique()
        aff_abnormal = aff_mean(abn_idx)
        loss_aff = (GUIDANCE_MARGIN - (aff_normal - aff_abnormal)).clamp_min(0)

        return loss_contrastive + LAMBDA_AFF * loss_aff

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self):
        """Iterate encoder + diffusion until N_k/A_pool falls below target ratio.

        Returns
        -------
        features_cpu : torch.Tensor [1, N, d_model]
            GRPE embeddings (from the last encoder step) with rows at every
            pseudo_pool position overwritten by diffusion-refined vectors.
        final_N : list[int]
            Normal-class anchor list from the last completed iter (N_k_last).
        final_A : list[int]
            Abnormal-class anchor list = real_abnormal_pool + pseudo_pool.
        """
        N = self.features_cpu.shape[1]
        # Full-graph cross-attn coverage so the downstream detector / any
        # consumer sees a non-zero embedding for every node.
        inference_idx = np.arange(N)

        sched = self.frac_schedule          # None => geometric doubling mode
        n_iters = MAX_ITERS if sched is None else len(sched)
        # --gen_max_iters: hard cap on generation iterations (doubling mode).
        # E.g. 2 => A_pool doubles twice: anchors a -> 2a -> 4a, i.e. the
        # final abnormal set = real anchors + 3x-anchors pseudos, regardless
        # of the fraction/ratio target.
        _cap = getattr(self.args, 'gen_max_iters', 0)
        if _cap and _cap > 0:
            n_iters = min(n_iters, _cap)
        last_completed = -1
        for k in range(n_iters):
            A_pool = self._A_pool_current()
            if sched is None:
                # Doubling mode: N_k grows as (k+1)/MAX_ITERS, A_pool ~doubles
                # each iter, stop when |N|/|A| falls to the target ratio.
                N_k = self._normal_subset(k)
                ratio = len(N_k) / max(1, len(A_pool))
                print(f"\n=== Iter {k}: |N_k|={len(N_k)} |A_pool|={len(A_pool)} "
                      f"ratio={ratio:.2f} (target stop <= {self.target_ratio}) ===")
                if ratio <= self.target_ratio:
                    print(f"  ratio {ratio:.2f} <= {self.target_ratio}; stop before iter {k}.")
                    break
                num_new = len(A_pool)
            else:
                # Schedule mode: SAME termination rule as doubling, one exact-fill
                # step per entry — frac f_k <=> target ratio r_k = (1-f_k)/f_k,
                # checked against the same growing slice N_k. The only difference
                # from doubling is num_new: exact fill to the ratio, not |A_pool|.
                frac = sched[k]
                r_k = (1.0 - frac) / frac
                if self.slice_full:
                    # Full-pool ablation: target pool = the WHOLE remaining normal
                    # set, but pseudo positions must come out of that same set, so
                    # solve jointly: A+num = frac * (A + |avail|)  (N = avail - num).
                    pseudo_set = set(self.pseudo_pool)
                    avail = [i for i in self.real_normal_pool if i not in pseudo_set]
                    a_target = int(np.ceil(frac * (len(avail) + len(A_pool))))
                    num_new = a_target - len(A_pool)
                    N_k = avail[: len(avail) - max(num_new, 0)]
                else:
                    N_k = self._normal_subset(k)
                    a_target = int(np.ceil(len(N_k) / r_k))
                    num_new = a_target - len(A_pool)
                ratio = len(N_k) / max(1, len(A_pool))
                print(f"\n=== Step {k}/{n_iters - 1}: target abn frac={frac:.2f} "
                      f"(rule |N_k|/|A|<={r_k:.2f}, now {ratio:.2f}; "
                      f"|A| {len(A_pool)}->{a_target}, num_new={num_new}); |N_k|={len(N_k)} ===")
                if num_new <= 0:
                    print(f"  ratio {ratio:.2f} <= {r_k:.2f} already; skip step.")
                    continue

            # Position-availability check BEFORE the (expensive) encoder step:
            # a complete step needs num_new free real-normal positions, else the
            # returned features (this iter) would be inconsistent with the
            # partition. _pick_new_pseudo is pure w.r.t. the unchanged state, so
            # _oversample_step re-picks exactly this set below.
            if self._pick_new_pseudo(num_new, N_k) is None:
                print(f"  not enough free positions for {num_new} pseudo-abns; "
                      f"stop before iter {k}.")
                break

            # Encoder step (dim may change raw-ft_size -> d_model on the first).
            # --iter_skip_encoder ("raw-rolling"): identity step — the working
            # representation stays the RAW features (with accumulated pseudo
            # rows); the rolling/diffusion machinery operates in raw space and
            # the detector consumes raw-dim features. Isolates the rolling
            # schedule's value from the (lossy) encoder space.
            if self.skip_encoder:
                grpe_emb = self.features_cpu.clone()
                print(f"[iter{k}/enc] SKIPPED (raw-rolling): features stay "
                      f"raw {tuple(grpe_emb.shape)}")
            else:
                grpe_emb = self._encoder_step(k, N_k, A_pool, inference_idx)
            self.features_cpu = grpe_emb.detach().cpu()

            # Generation step (shuffle, then optional diffusion refinement).
            grpe_emb_dev = grpe_emb if grpe_emb.device == self.device else grpe_emb.to(self.device)
            generated, new_positions = self._oversample_step(
                k, grpe_emb_dev, N_k, A_pool, num_new
            )
            if generated is None:
                print(f"  not enough free positions for {num_new} pseudo-abns; stopping.")
                break

            # Diagnostic BEFORE the in-place override (prior pseudos still distinct).
            self._log_synth_diagnostics(k, generated, new_positions, N_k)

            # In-place feature override + label flip (appending to pseudo_pool
            # makes these part of the next iter's A_pool via _A_pool_current).
            self.features_cpu[0, new_positions, :] = generated.detach().cpu()
            self.pseudo_pool.extend(new_positions)
            last_completed = k

        # Final partition (both modes now share it): last completed step's
        # slice N_k; if nothing ran (anchors already satisfy the rule — the
        # "gate"), slice-0 and the input features are returned unchanged.
        if last_completed < 0:
            final_N = self._normal_subset(0)
            final_A = self._A_pool_current()
            print("[iter] no iter completed; returning input features unchanged.")
        else:
            final_N = self._normal_subset(last_completed)
            final_A = self._A_pool_current()
        print(f"\n[iter] done. last completed={last_completed}, "
              f"|N|={len(final_N)} |A|={len(final_A)} "
              f"abn_frac={len(final_A) / max(1, len(final_A) + len(final_N)):.2f}")
        return self.features_cpu, final_N, final_A
