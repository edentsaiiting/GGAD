import json
from dataclasses import fields as _dc_fields

import torch.nn as nn

from model import Model
from utils import *
from grpe_encoder import GRPEConfig, train_grpe_encoder

from sklearn.metrics import roc_auc_score
import random
import dgl
from sklearn.metrics import average_precision_score
import argparse
from tqdm import tqdm
import time

# Set argument
parser = argparse.ArgumentParser(description='')

parser.add_argument('--dataset', type=str,
                    default='reddit')
parser.add_argument('--lr', type=float)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--embedding_dim', type=int, default=300)
parser.add_argument('--num_epoch', type=int)
parser.add_argument('--negsamp_ratio', type=int, default=1)
parser.add_argument('--diff_hidden_dim', type=int, default=512,
                    help='Hidden dimension for diffusion model')
parser.add_argument('--gen_epochs', type=int, default=50,
                    help='Epochs for synthetic abnormal node generation')
parser.add_argument('--loss_guidance_weight', type=float, default=1.0,
                    help='Weight for guidance loss in diffusion model')
parser.add_argument('--num_steps', type=int, default=50,
                    help='Number of diffusion sampling steps for synthetic generation')
parser.add_argument('--test_sample_rounds', type=int, default=10,
                    help='Number of random test subsamples to average for final test metrics')
parser.add_argument('--test_sample_ratio', type=float, default=0.8,
                    help='Fraction of idx_test to sample per round')
parser.add_argument('--fc_only', action='store_true', default=False,
                    help='FC-only detector; skip the GCN encoder in the detector. '
                         'Compatible with --use_grpe_encoder or vanilla raw features. '
                         'Default: GCN+FC.')
parser.add_argument('--use_grpe_encoder', action='store_true', default=False,
                    help='Phase 0: pretrain a GRPE-style encoder on labeled nodes, then feed '
                         'its [N, d_model] embeddings into the downstream pipeline in place '
                         'of raw features. Hyperparams live in GRPE_CONFIG_PATH (JSON).')
parser.add_argument('--use_iter_cotrain', action='store_true', default=False,
                    help='Iterative GRPE encoder <-> diffusion augmentor co-training. '
                         'Subsumes Phase 0 + Phase 1; implies --use_grpe_encoder for the '
                         'detector branch. See iter_cotrain.py.')
parser.add_argument('--gen_max_iters', type=int, default=0,
                    help='(--use_iter_cotrain, doubling mode) Hard cap on generation '
                         'iterations. 2 = double twice: anchors a -> 2a -> 4a, i.e. final '
                         'abnormal set = real anchors + 3x-anchors pseudos. 0 = no cap.')
parser.add_argument('--iter_skip_encoder', action='store_true', default=False,
                    help='Raw-rolling ablation (--use_iter_cotrain): skip the GRPE encoder '
                         'entirely — rolling diffusion generation AND the detector operate on '
                         'RAW features. Isolates the rolling schedule from the encoder space.')
parser.add_argument('--highpass_ref', type=str, default='none',
                    choices=['none', 'neighbor', 'labeled'],
                    help='Detector high-pass aggregator (after GCN, before FC): subtract a '
                         'reference and concat [h, h-ref]. neighbor=local neighbor mean, '
                         'labeled=global labeled-node mean, none=off.')
parser.add_argument('--ano_known_rate', type=float, default=0.0,
                    help='Size of the known/leaked real-abnormal pool (clip_abnormal_idx / '
                         'all_abnormal_label_idx) as a FRACTION of the training-abnormal set. '
                         '0 (default) keeps the legacy fixed pool of 3. This pool is the '
                         'real-abnormal set that seeds configs B/C pseudo-abnormal generation, '
                         'so sweeping it varies how much real-abnormal supervision they get.')
# --- iter_cotrain rolling-schedule knobs (config B / one-shot variants) ---
# The pseudo-abnormal pool is grown until it reaches a target abnormal
# fraction of the labeled pool. Two ways to drive it:
#   * doubling (default): geometric growth, stop when |N|/|A| <= gen_target_ratio.
#     ratio 5.5 ~= 15% abnormal (the current B/C stopping point); 1.0 = 50% (1:1).
#   * explicit schedule: --gen_frac_schedule lists cumulative abnormal-fraction
#     targets, one big "one-shot" generation step per entry (overrides doubling).
#     e.g. "0.15,0.30,0.50" = 3-step; "0.5" = single one-shot to 50%.
parser.add_argument('--gen_target_ratio', type=float, default=5.5,
                    help='(--use_iter_cotrain, doubling mode) stop when |N|/|A_pool| <= this. '
                         '5.5 ~15%% abnormal (default); 1.0 = 50%% (1:1).')
parser.add_argument('--gen_frac_schedule', type=str, default='',
                    help='(--use_iter_cotrain) comma-separated cumulative abnormal-fraction '
                         'targets; one one-shot generation step each (overrides doubling). '
                         'e.g. "0.15,0.30,0.50" or "0.5". Empty = doubling mode.')
parser.add_argument('--gen_slice_frac', type=float, default=0.1,
                    help='(--use_iter_cotrain) per-iter normal-slice fraction: N_k = first '
                         'min(1,(k+1)*this) of the labeled-normal pool. 0.1 (default) = the '
                         'historical (k+1)/10 slice rule; 1.0 = full-pool ablation (the fill '
                         'target is the whole normal set, pseudos carved from its tail).')
# gen_frac_schedule steps use the same termination rule as doubling:
# step frac f <=> target ratio (1-f)/f, checked against the rolling slice N_k.

# GRPE hyperparameter file. Hardcoded because it is not a frequently tuned knob.
GRPE_CONFIG_PATH = './grpe_config.json'


args = parser.parse_args()

# Iter co-training drives the encoder/augmentor loop itself; flip --use_grpe_encoder
# so the detector branch (model.py forward bypass + run.py BCE gather) routes
# through the encoder-aware path. The Phase 0 single-shot block below is then
# gated off explicitly to avoid running it twice.
if args.use_iter_cotrain:
    args.use_grpe_encoder = True

if args.lr is None:
    args.lr = 1e-3  # all shipped datasets

if args.num_epoch is None:
    if args.dataset in ['photo']:
        args.num_epoch = 100
    if args.dataset in ['elliptic']:
        args.num_epoch = 150
    if args.dataset in ['reddit']:
        args.num_epoch = 300
    elif args.dataset in ['t_finance']:
        args.num_epoch = 350
    elif args.dataset in ['Amazon']:
        args.num_epoch = 800
# Ego-centric recon noise, fixed per dataset (GGAD paper values).
if args.dataset in ['reddit', 'photo']:
    args.mean = 0.02
    args.var = 0.01
else:
    args.mean = 0.0
    args.var = 0.0


print('Dataset: ', args.dataset)

# Set random seed
dgl.random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Load and preprocess data. load_mat returns already-repartitioned label sets:
# normal_label_idx = pure real normals; abnormal_label_idx = sampled pseudos +
# leaked reals (the leaked-abnormal label fix lives inside load_mat).
adj, features, ano_label, idx_train, idx_val, idx_test, \
normal_label_idx, abnormal_label_idx, all_abnormal_label_idx = load_mat(args.dataset, ano_known_rate=args.ano_known_rate)

# NB: 'tf_finace' typo is inherited from GGAD-git — t_finance intentionally
# falls through to todense() (no row-normalize); all campaign numbers were
# produced this way. Do not 'fix'.
if args.dataset in ['Amazon', 'tf_finace', 'reddit', 'elliptic']:
    features, _ = preprocess_features(features)
else:
    features = features.todense()

nb_nodes = features.shape[0]
ft_size = features.shape[1]
raw_adj = adj
adj = normalize_adj(adj)

# Keep a sparse copy of raw_adj (with self-loops) for phase 0 GRPE so it
# never needs the dense [N, N] form. Phase 1/2 still consume the dense
# torch tensor below.
raw_adj_sparse = (raw_adj + sp.eye(raw_adj.shape[0])).tocsr()
raw_adj = raw_adj_sparse.todense()
adj = (adj + sp.eye(adj.shape[0])).todense()

features = torch.FloatTensor(features[np.newaxis])
adj = torch.FloatTensor(adj[np.newaxis])
raw_adj = torch.FloatTensor(raw_adj[np.newaxis])

# ============================================================================
# Phase 0: GRPE encoder pretraining (optional, runs before diffusion / detector)
# ============================================================================
# Data-split rule per phase: the encoder is supervised on the ground-truth
# labels (real normals + real leaked abnormals); the diffusion augmentor
# (when present) operates on the pseudo-abn split sampled from real
# normals. The detector inherits the split of the immediately upstream
# phase. So with an augmentor in the pipeline, its pseudo-abn split wins
# for the detector; with GRPE-only, the encoder's real-real split is used
# end-to-end.
#
# Skipped entirely when --use_iter_cotrain owns both the encoder and the
# augmentor; that path runs its own GRPE + diffusion schedule (see below).
if args.use_grpe_encoder and not args.use_iter_cotrain:
    with open(GRPE_CONFIG_PATH) as _f:
        _grpe_cfg_raw = json.load(_f)
    _allowed = {f.name for f in _dc_fields(GRPEConfig)}
    grpe_cfg = GRPEConfig(**{k: v for k, v in _grpe_cfg_raw.items() if k in _allowed})
    grpe_cfg.noise_mean = args.mean                     # GGAD ego-centric recon noise
    grpe_cfg.noise_var = args.var

    # load_mat returns already-cleaned splits: normal_label_idx = real normals
    # only; all_abnormal_label_idx = leaked reals. The pseudo-abn set is
    # intentionally not consulted for the encoder.
    real_normal_idx = list(normal_label_idx)
    real_abnormal_idx = list(all_abnormal_label_idx)

    print("\n" + "=" * 80)
    print(f"PHASE 0: GRPE encoder pretraining  ({GRPE_CONFIG_PATH})")
    print("=" * 80)
    print(f"  d_model={grpe_cfg.d_model}  layers={grpe_cfg.num_layer}  "
          f"heads={grpe_cfg.nhead}  max_hop={grpe_cfg.max_hop}  "
          f"epochs={grpe_cfg.num_epoch}  lr={grpe_cfg.lr}")
    print(f"  encoder split: {len(real_normal_idx)} real normals + "
          f"{len(real_abnormal_idx)} real abnormals")

    if len(real_abnormal_idx) == 0:
        # No real leaked abnormals -> single-class BCE. Skip Phase 0 and
        # leave the load_mat (pseudo-abn) split in place for downstream.
        print("[GRPE] WARNING: clip_abnormal_idx is empty (no real leaked "
              "abnormals in train); Phase 0 BCE would be single-class. "
              "Skipping encoder; using raw features.\n")
    else:
        # Cover every non-anchor node so the downstream detector / augmentor
        # never sees zero rows. With a GCN detector this is required: zero
        # features at training-only-unlabeled nodes silently corrupt
        # neighbor aggregation for adjacent labeled / val / test nodes. With
        # a raw-space augmentor the diffusion seeds train-unlabeled rows
        # too. Anchors are filtered out internally by train_grpe_encoder.
        _inference_idx = np.arange(nb_nodes)
        _, grpe_emb = train_grpe_encoder(
            features, raw_adj,
            real_normal_idx, real_abnormal_idx,
            cfg=grpe_cfg,
            inference_idx=_inference_idx,
            raw_adj_sparse=raw_adj_sparse,
            cache_key=f"{args.dataset}_seed{args.seed}",
        )
        # grpe_emb: [1, N, d_model] on training device. Downstream stages expect
        # CPU tensors (Phase 2 runs on CPU; Phase 1 moves to CUDA internally).
        features = grpe_emb.detach().cpu().float()
        ft_size = features.shape[-1]
        print(f"[GRPE] replaced raw features with encoder embeddings: "
              f"shape={tuple(features.shape)}, new ft_size={ft_size}")

        # Detector inherits the encoder's real-real split.
        normal_label_idx = real_normal_idx
        abnormal_label_idx = real_abnormal_idx
        print(f"[GRPE-only] detector inherits encoder split: "
              f"{len(normal_label_idx)}N + {len(abnormal_label_idx)}A\n")

if args.use_iter_cotrain:
    # Iterative GRPE <-> diffusion co-training (see iter_cotrain.py).
    # Inputs: real_normal_pool = normal_label_idx after the convention
    # fix-up (real normals, leaked subtracted); real_abnormal_pool =
    # all_abnormal_label_idx (the 3 leaked, fixed positives for guidance).
    from iter_cotrain import IterCotrainer
    cotrainer = IterCotrainer(
        features=features, raw_adj=raw_adj, raw_adj_sparse=raw_adj_sparse,
        real_normal_pool=normal_label_idx,
        real_abnormal_pool=all_abnormal_label_idx,
        args=args,
    )
    features, normal_label_idx, abnormal_label_idx = cotrainer.run()
    ft_size = features.shape[-1]
    # Detector consumes the iter outputs directly. features_eval is set
    # below by the vanilla branch (features_eval = features) since the
    # iter loop produces a single tensor used for both train and eval.
    print(f"[iter] feeding detector: features={tuple(features.shape)}, "
          f"|N|={len(normal_label_idx)}, |A|={len(abnormal_label_idx)}\n")

# ============================================================================
# Standard GGAD Detector Training (PHASE 2) — always runs on CPU
# ============================================================================

# Phase 2 always runs on CPU: adj/raw_adj dense [N,N] matrices exhaust GPU
# memory (mirrors GGAD-git). Release cached encoder blocks so the hours-long
# CPU detector holds no idle GPU memory on a shared box.
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ---- Analysis dump (env-gated, DUMP_FEATS=<path.npz>): save the detector's
# exact inputs then exit. Implementation lives in analysis.py.
_dump_path = os.environ.get('DUMP_FEATS')
if _dump_path:
    from analysis import dump_detector_inputs
    try:
        _feval = features_eval
    except NameError:
        _feval = features
    dump_detector_inputs(
        _dump_path, features=features, features_eval=_feval,
        normal_label_idx=normal_label_idx, abnormal_label_idx=abnormal_label_idx,
        real_anchor_idx=all_abnormal_label_idx, ano_label=ano_label,
        idx_train=idx_train, idx_test=idx_test,
    )
    import sys as _dsys
    _dsys.exit(0)

device = torch.device('cpu')

# Detector input: GCN+FC over ft_size (default) or FC-only over ft_size
# (--fc_only; features may be raw, feat-aug-refined, or GRPE embeddings).
if args.fc_only:
    n_in_detect = ft_size
    use_gcn = False
else:
    n_in_detect = ft_size
    use_gcn = True
# High-pass is only wired into the GRPE detector branch and the shared eval
# branch (model.forward). The vanilla ego-centric train branch does NOT apply
# it, yet fc1 would be sized for the [h, h-ref] concat — a guaranteed
# shape-mismatch crash (and train/eval inconsistency).
# Gate it off for that path, matching model.forward's branch condition.
_uses_highpass_path = args.use_grpe_encoder
effective_highpass = args.highpass_ref if _uses_highpass_path else 'none'
if args.highpass_ref != 'none' and not _uses_highpass_path:
    print(f"[highpass] --highpass_ref={args.highpass_ref} not supported on the "
          f"ego-centric detector path; using 'none' for this config.")
model = Model(n_in_detect, args.embedding_dim, 'prelu',
              use_gcn=use_gcn, highpass_ref=effective_highpass)
optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

model = model.to(device)

b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio], device=device))

# Set log file name
if args.use_iter_cotrain:
    log_suffix = "iter_cotrain_"
else:
    log_suffix = "vanilla_"
# GRPE encoder runs replace raw features with cross-attention embeddings
# upstream and use a BCE-only detector — distinct from true vanilla GGAD,
# so prefix the log/ckpt name to avoid collisions. (Skip the prefix for
# iter_cotrain since it already names the upstream stage.)
if args.use_grpe_encoder and not args.use_iter_cotrain:
    log_suffix = "enc_" + log_suffix

print("\n" + "=" * 80)
print("TRAINING CONFIGURATION")
print("=" * 80)
if args.use_iter_cotrain:
    method_name = 'Iterative GRPE <-> Diffusion Co-train'
else:
    method_name = 'GRPE -> Original GCN-based' if args.use_grpe_encoder else 'Original GCN-based'
print(f"Method: {method_name}")
print(f"Dataset: {args.dataset}")
print(f"Embedding Dimension: {args.embedding_dim}")
print(f"Number of Epochs: {args.num_epoch}")
print(f"Learning Rate: {args.lr}")
print(f"Number of Abnormal Nodes: {len(abnormal_label_idx)}")
print("=" * 80 + "\n")


features_eval = features   # train and eval share one tensor on every path

val_auc_history   = []
val_ap_history    = []
val_epoch_history = []
# Detector-loss curves (logged every 2 epochs alongside the prints below).
loss_epoch_history  = []
loss_total_history  = []
loss_bce_history    = []
loss_margin_history = []
loss_rec_history    = []
best_val_auc      = 0.0
# Per-run ckpt path: concurrent runs sharing (log_suffix, dataset) would
# clobber one file and load each other's weights at final test (silent wrong
# result, or fc1 shape crash when highpass differs). Tag = seed + knobs.
_ckpt_dir = './log/ckpt'
os.makedirs(_ckpt_dir, exist_ok=True)
_gs_tag = args.gen_frac_schedule.replace(',', '-').replace('.', 'p') if args.gen_frac_schedule else 'none'
if args.gen_frac_schedule and args.gen_slice_frac >= 1.0:
    _gs_tag += 'F'   # full-pool (--gen_slice_frac 1.0): separate ckpt namespace
_run_tag = (f"s{args.seed}_hp{args.highpass_ref}"
            f"_akr{args.ano_known_rate}_gr{args.gen_target_ratio}_gs{_gs_tag}")
best_ckpt_path    = f'{_ckpt_dir}/best_{log_suffix}{args.dataset}_{_run_tag}.pt'

# Train model
with tqdm(total=args.num_epoch) as pbar:
    pbar.set_description('Training')
    total_time = 0
    with open("./log/" + log_suffix + args.dataset + str(args.num_epoch) + ".txt", "a") as f:
        for epoch in range(args.num_epoch):
            start_time = time.time()
            model.train()
            optimiser.zero_grad()

            # Train model
            train_flag = True
            n_nodes_train = features.shape[1]
            if len(abnormal_label_idx) > 0 and max(abnormal_label_idx) >= n_nodes_train:
                raise ValueError(f"abnormal_label_idx out of range: max={max(abnormal_label_idx)} n_nodes={n_nodes_train}")
            if len(normal_label_idx) > 0 and max(normal_label_idx) >= n_nodes_train:
                raise ValueError(f"normal_label_idx out of range: max={max(normal_label_idx)} n_nodes={n_nodes_train}")

            emb, emb_combine, logits, emb_con, emb_abnormal = model(features, adj,
                                                                    abnormal_label_idx, normal_label_idx,
                                                                    train_flag, args)

            # BCE loss — labeled nodes only.
            if args.use_grpe_encoder:
                # abnormal_label_idx ⊂ normal_label_idx (pseudos are drawn from
                # labeled normals). With the node-wise FC head, gathering both
                # lists would fetch emb[i] twice with conflicting targets and
                # cancel gradients — dedup: pure normals → 0, abnormals → 1.
                abn_set = set(abnormal_label_idx)
                pure_normal_idx = [i for i in normal_label_idx if i not in abn_set]
                labeled_idx = pure_normal_idx + list(abnormal_label_idx)
                lbl = torch.unsqueeze(
                    torch.cat((torch.zeros(len(pure_normal_idx)), torch.ones(len(abnormal_label_idx)))), 1
                ).unsqueeze(0).to(device)
                logits = logits[:, labeled_idx, :]
            else:
                # Vanilla: logits come from emb_combine with
                # len(normal_label_idx)+len(abnormal_label_idx) rows; the
                # duplication is intentional (fc4 gives distinct targets).
                lbl = torch.unsqueeze(
                    torch.cat((torch.zeros(len(normal_label_idx)),
                               torch.ones(len(abnormal_label_idx)))), 1
                ).unsqueeze(0).to(device)

            loss_bce = b_xent(logits, lbl)
            loss_bce = torch.mean(loss_bce)

            if args.use_grpe_encoder:
                # Detector head trains on BCE only. The margin/reconstruction
                # terms live upstream (encoder / generation stages) on this path.
                loss = loss_bce
                loss_margin = torch.tensor(0.0)
                loss_rec = torch.tensor(0.0)
            else:
                # Local affinity margin loss
                # Avoid full [N, N] similarity matrix — compute only the labeled rows needed.
                emb = torch.squeeze(emb)           # [N, n_h]
                raw_adj = torch.squeeze(raw_adj)   # [N, N]

                emb_inf = torch.norm(emb, dim=-1, keepdim=True)
                emb_inf = torch.pow(emb_inf, -1)
                emb_inf[torch.isinf(emb_inf)] = 0.
                emb_norm = emb * emb_inf           # [N, n_h]

                # normal_label_idx = pure normals; abnormal_label_idx = pseudos +
                # leaked reals (repartitioned in load_mat) — use directly.
                def labeled_affinity(idx):
                    """Affinity for a subset of nodes: [n_idx, N] instead of [N, N]."""
                    rows     = emb_norm[idx]                        # [n_idx, n_h]
                    sim_rows = torch.mm(rows, emb_norm.T)          # [n_idx, N]
                    adj_rows = raw_adj[idx]                        # [n_idx, N]
                    deg_inv  = torch.pow(adj_rows.sum(1), -1)
                    deg_inv[torch.isinf(deg_inv)] = 0.
                    return (sim_rows * adj_rows).sum(1) * deg_inv  # [n_idx]

                affinity_normal_mean = (labeled_affinity(normal_label_idx).mean()
                                        if len(normal_label_idx) > 0
                                        else torch.tensor(0.0, device=device))
                affinity_abnormal_mean = (labeled_affinity(abnormal_label_idx).mean()
                                          if len(abnormal_label_idx) > 0
                                          else torch.tensor(0.0, device=device))

                # Ego Centric Loss
                confidence_margin = 0.7
                loss_margin = (confidence_margin - (affinity_normal_mean - affinity_abnormal_mean)).clamp_min(min=0)

                # emb_con is [n_abn, n_h]; emb_abnormal is [1, n_abn, n_h].
                # Squeeze the batch dim so the subtraction is element-wise per
                # node and sum(dim=1) reduces over FEATURES (per-node L2 norm),
                # not over nodes. (+1e-12 keeps sqrt's gradient finite at 0.)
                diff_attribute = torch.pow(emb_con - emb_abnormal.squeeze(0), 2)
                loss_rec = torch.mean(torch.sqrt(torch.sum(diff_attribute, 1) + 1e-12))

                loss = 1 * loss_margin + 1 * loss_bce + 1 * loss_rec

            loss.backward()
            optimiser.step()
            end_time = time.time()
            total_time += end_time - start_time
            if epoch % 2 == 0:
                logits_np = np.squeeze(logits.cpu().detach().numpy())
                lbl_np    = np.squeeze(lbl.cpu().detach().numpy())
                train_auc = roc_auc_score(lbl_np, logits_np)
                print(f"Epoch {epoch:04d}  margin={loss_margin.item():.5f} "
                      f"bce={loss_bce.item():.5f} rec={loss_rec.item():.5f} "
                      f"total={loss.item():.5f} train_auc={train_auc:.4f}")
                # Track detector-loss components for the loss-vs-epoch curves.
                loss_epoch_history.append(epoch)
                loss_total_history.append(loss.item())
                loss_bce_history.append(loss_bce.item())
                loss_margin_history.append(loss_margin.item())
                loss_rec_history.append(loss_rec.item())

                model.eval()
                train_flag = False
                _, _, logits_eval, _, _ = model(features_eval, adj,
                                                    abnormal_label_idx, normal_label_idx,
                                                    train_flag, args)
                # Validation eval for model selection
                safe_idx_val = [i for i in idx_val if i < logits_eval.shape[1]]
                val_auc = 0.0
                val_ap  = 0.0
                if len(safe_idx_val) > 0:
                    logits_val = np.squeeze(logits_eval[:, safe_idx_val, :].cpu().detach().numpy())
                    val_auc = roc_auc_score(ano_label[safe_idx_val], logits_val)
                    val_ap  = average_precision_score(ano_label[safe_idx_val], logits_val, average='macro', pos_label=1, sample_weight=None)
                    print(f'Val {args.dataset}  AUC:{val_auc:.4f}  AP:{val_ap:.4f}')
                    val_auc_history.append(val_auc)
                    val_ap_history.append(val_ap)
                    val_epoch_history.append(epoch)
                    if val_auc > best_val_auc:
                        best_val_auc = val_auc
                        torch.save(model.state_dict(), best_ckpt_path)
                        print(f'  [ckpt] best val AUC={best_val_auc:.4f} → saved to {best_ckpt_path}')
                else:
                    print('No valid idx_val in logits range; skipping val AUC/AP')

            f.write(f'{loss_margin.item():.4f} {loss_bce.item():.4f} {loss_rec.item():.4f} {loss.item():.4f} {val_auc:.4f} {val_ap:.4f}\n')

print(f'Total training time: {total_time:.1f}s')
if val_epoch_history:
    method = 'Vanilla'
    test_plotting(val_epoch_history, val_auc_history, val_ap_history,
                  args.dataset, method=method, num_epochs=args.num_epoch, log_suffix=log_suffix,
                  best_ckpt_path=best_ckpt_path, best_val_auc=best_val_auc,
                  features_eval=features_eval, adj=adj,
                  abnormal_label_idx=abnormal_label_idx, normal_label_idx=normal_label_idx,
                  idx_test=idx_test, ano_label=ano_label,
                  model=model, device=device, args=args,
                  loss_epochs=loss_epoch_history, loss_total=loss_total_history,
                  loss_bce=loss_bce_history, loss_margin=loss_margin_history,
                  loss_rec=loss_rec_history)