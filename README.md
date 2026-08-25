# Running `run.py` — GGAD *Diffusion co-training* branch

Branch: [`Diffusion-co-training`](https://github.com/edentsaiiting/GGAD/tree/Diffusion-co-training).
The original GGAD paper README (NeurIPS 2024) is on the `main` branch / [mala-lab/GGAD](https://github.com/mala-lab/GGAD).
`run.py` is the single entry point. It drives **three arms** on top of the stock GGAD detector,
selected by flags; everything else (encoder hyperparameters, diffusion knobs, the
supervision axis) is a switch on the same command line.

---

## 1. Setup

**Environment** — tested with Python 3.10, `torch 1.12.1+cu113`, `dgl 1.0.1`, `scikit-learn 1.0.2`,
`scipy 1.7.3`, `matplotlib`, `tqdm` (conda env `Graph` on the lab box). `requirements.txt` lists the
original pins; `requirements2.txt` is an older py3.7/torch-1.6 spec kept for reference.

**Datasets** — `.mat` files named `Amazon.mat`, `elliptic.mat`, `photo.mat`, `reddit.mat`,
`t_finance.mat`. They are **not in the repo**. `utils.load_mat` looks in
`../../Dataset/T/` relative to the working directory; override with an env var:

```bash
export GGAD_DATA_ROOT=/path/to/Dataset/T
```

**Where to run from** — always the repo root: `run.py` uses relative paths for
`./grpe_config.json`, `./log/`, `./plt/`, `./cache/`.

**Devices** — the encoder and diffusion phases use CUDA when available; the Phase-2 detector
**always runs on CPU** (dense N×N adjacency, mirrors GGAD-git). To force CPU everywhere:
`CUDA_VISIBLE_DEVICES=""`. On a shared box where someone else runs an MPS server, CUDA init can
hang — set `CUDA_MPS_PIPE_DIRECTORY=/tmp/no-mps-$USER` to bypass it.

**Long runs** — t_finance / Amazon take hours; detach them (`setsid nohup … &`) or they die with the
SSH session.

---

## 2. The three arms

| arm | flags | what happens |
|---|---|---|
| **D — vanilla GGAD** | *(none)* | Stock detector: 2-layer GCN + FC head, ego-centric pseudo-abnormals, three losses (BCE + affinity-margin + reconstruction). Baseline. |
| **C — GRPE encoder** | `--use_grpe_encoder` | Phase 0 pretrains a graph-Transformer encoder (hop + labeled-pair edge biases) on the labeled nodes with the same three-term loss, then the detector consumes its `[N, d_model]` embeddings instead of raw features. |
| **B — diffusion co-training** | `--use_iter_cotrain` | Iterates *encoder step → guided-diffusion step*: each iteration trains the encoder on the current labeled pool, then synthesizes new pseudo-abnormals with an EDM diffusion model guided away from normals and toward the known anchors, flips their labels, and grows the abnormal pool until a target normal:abnormal ratio is reached ("rolling"). Implies `--use_grpe_encoder` for the detector branch. |
| **B-raw — raw-rolling** | `--use_iter_cotrain --iter_skip_encoder` | Same rolling generation schedule but the encoder step is skipped: generation **and** the detector operate on raw features. Isolates the schedule from the encoder space. |

Data flow: `load_mat` → (Phase 0 encoder / iterative co-training) → Phase 2 detector on CPU →
best-val checkpoint → test evaluation → plots.

---

## 3. All arguments

### Base / training

| flag | default | meaning |
|---|---|---|
| `--dataset` | `reddit` | One of `Amazon`, `elliptic`, `photo`, `reddit`, `t_finance`. |
| `--seed` | `0` | Seeds `random`, `numpy`, `torch`, `dgl`; cuDNN set deterministic. |
| `--lr` | `1e-3` | Detector learning rate (all shipped datasets use 1e-3). |
| `--weight_decay` | `0.0` | Detector Adam weight decay. |
| `--embedding_dim` | `300` | Detector hidden width (`n_h`); the FC head is `n_h → n_h/2 → n_h/4 → 1`. |
| `--num_epoch` | per dataset | photo 100 · elliptic 150 · reddit 300 · t_finance 350 · Amazon 800. |
| `--negsamp_ratio` | `1` | `pos_weight` of the detector's BCE. |
| `--test_sample_rounds` | `10` | Final test metrics are also averaged over this many random test subsamples. |
| `--test_sample_ratio` | `0.8` | Fraction of the test set drawn per round. |

### Supervision axis

| flag | default | meaning |
|---|---|---|
| `--ano_known_rate` | `0.0` | Size of the **known / "leaked" real-abnormal pool** as a fraction of the training abnormals (rounded, ≥1). `0` keeps the legacy fixed pool of 3 anchors. This pool is what supervises the encoder and seeds generation, so sweeping it (e.g. `0.03 0.05 0.10 0.20 0.30`) varies how much real supervision the method gets — the main experimental axis of the project. |

### Detector options

| flag | default | meaning |
|---|---|---|
| `--fc_only` | off | Drop the detector's GCN layers: FC head directly over the input features (raw or encoder embeddings). |
| `--highpass_ref` | `none` | High-pass layer between GCN and FC head: subtract a reference and feed `[h, h − ref]`. `neighbor` = local neighbor mean (the `_hp` in `B_hp`/`C_hp`), `labeled` = global labeled-node mean. **Only takes effect on the encoder / co-training path**; on vanilla D it is ignored with a warning (the ego-centric branch would shape-mismatch). |

### Encoder (arm C, and the encoder step of arm B)

| flag | default | meaning |
|---|---|---|
| `--use_grpe_encoder` | off | Enable the Phase-0 GRPE encoder. |

Encoder hyperparameters are **not** CLI flags; edit `grpe_config.json`:

| key | value | note |
|---|---|---|
| `d_model` | 80 | must be divisible by `nhead` |
| `num_layer` | 4 | |
| `nhead` | 8 | |
| `ffn_dim` | null → `d_model` | |
| `max_hop` | 5 | SPD clamp for the hop bias |
| `dropout` / `attention_dropout` | 0.1 / 0.1 | |
| `perturb_noise` | 0.0 | input-embedding jitter during training |
| `num_epoch` | 50 | encoder epochs (per encoder step in arm B) |
| `lr` / `weight_decay` | 1e-3 / 1e-3 | AdamW |
| `chunk_size` | 8000 | cross-attention inference batch; lower it if the GPU OOMs |

The encoder caches shortest-path-distance matrices under `./cache/grpe_spd/` (keyed by dataset,
seed, hop, and adjacency fingerprint). First run on t_finance builds them (hours); re-runs load in
seconds. Safe to delete — they rebuild on demand.

### Co-training / generation schedule (arm B)

| flag | default | meaning |
|---|---|---|
| `--use_iter_cotrain` | off | Enable the rolling co-training loop (`iter_cotrain.py`). |
| `--iter_skip_encoder` | off | Raw-rolling: skip the encoder step; generation and detector run on raw features. |
| `--gen_target_ratio` | `5.5` | **Doubling mode** (default). Each iteration adds \|A_pool\| new pseudo-abnormals (pool doubles); stop when \|N_k\| / \|A_pool\| ≤ this. `5.5` ≈ 15 % abnormal in the labeled pool; `2.333` ≈ 30 %; `1.0` = 50 % (1:1). |
| `--gen_frac_schedule` | `""` | **Schedule mode** (overrides doubling). Comma-separated cumulative abnormal-fraction targets, one exact-fill generation step each: `"0.5"` = single one-shot to 50 %, `"0.15,0.30,0.50"` = 3 steps. Each step uses the same termination rule as doubling (frac *f* ⇔ ratio (1−*f*)/*f*), checked against the growing normal slice. |
| `--gen_slice_frac` | `0.1` | Per-iteration normal slice: N_k = first min(1, (k+1)·frac) of the labeled normals. `0.1` = the (k+1)/10 rule used in all campaigns; `1.0` = full-pool variant (pseudos carved from the tail of the whole normal set). |
| `--gen_max_iters` | `0` | Hard cap on generation iterations (doubling mode). `2` = anchors a → 2a → 4a, i.e. final abnormal set = anchors + 3× pseudos. `0` = no cap. |

**The gate.** If the known anchors already satisfy the target ratio at iteration 0 (large
`--ano_known_rate`), the loop performs **no generation** and the run reduces to
"encoder (or raw features) + true labels". This is by design — see §6.

### Diffusion model knobs (arm B)

| flag | default | meaning |
|---|---|---|
| `--diff_hidden_dim` | `512` | Width of the EDM denoiser MLP. |
| `--gen_epochs` | `50` | Denoiser training epochs per generation step. |
| `--num_steps` | `50` | Reverse-diffusion sampling steps. |
| `--loss_guidance_weight` | `1.0` | Scale of the classifier-gradient guidance (far-from-N_k, near-to-anchors, affinity margin). `0` = unguided sampling. |

### Environment hooks

| variable | effect |
|---|---|
| `GGAD_DATA_ROOT` | Dataset directory (default `../../Dataset/T`). |
| `DUMP_FEATS=<path.npz>` | After the encoder/generation phases, dump the detector's exact inputs to `<path.npz>` and exit (no detector training). Analyze offline with `python analysis.py 'dumps/*.npz'` (feature-space probes: oracle LR, anchor-centroid cosine AUC, camouflage). |
| `CUDA_VISIBLE_DEVICES` | Standard; empty string forces CPU for all phases. |
| `CUDA_MPS_PIPE_DIRECTORY` | Set to a private dir to dodge a foreign MPS server (see §1). |

---

## 4. Named configurations from the campaigns → commands

These are the exact flag sets the reported numbers were produced with (`--dataset`, `--seed`,
`--ano_known_rate` vary per run).

```bash
# D   — vanilla GGAD baseline
python run.py --dataset photo --seed 100 --ano_known_rate 0.05

# C_hp — GRPE encoder + high-pass detector
python run.py --dataset photo --seed 100 --ano_known_rate 0.05 \
    --use_grpe_encoder --highpass_ref neighbor

# B_hp — rolling co-training to ~15 % abnormal (doubling, default ratio 5.5)
python run.py --dataset photo --seed 100 --ano_known_rate 0.05 \
    --use_iter_cotrain --highpass_ref neighbor

# Broll30 / Broll50 — rolling to 30 % / 50 %
python run.py … --use_iter_cotrain --highpass_ref neighbor --gen_target_ratio 2.3333
python run.py … --use_iter_cotrain --highpass_ref neighbor --gen_target_ratio 1.0

# Bshot15s / Bshot30s / Bshot50s — single one-shot fill to 15 / 30 / 50 %
python run.py … --use_iter_cotrain --highpass_ref neighbor --gen_frac_schedule 0.15
python run.py … --use_iter_cotrain --highpass_ref neighbor --gen_frac_schedule 0.30
python run.py … --use_iter_cotrain --highpass_ref neighbor --gen_frac_schedule 0.5

# Bstep15s / Bstep30s / Bstep50s — 3-step schedules
python run.py … --use_iter_cotrain --highpass_ref neighbor --gen_frac_schedule 0.05,0.10,0.15
python run.py … --use_iter_cotrain --highpass_ref neighbor --gen_frac_schedule 0.10,0.20,0.30
python run.py … --use_iter_cotrain --highpass_ref neighbor --gen_frac_schedule 0.15,0.30,0.50

# BshotF* — full-pool one-shot variants
python run.py … --use_iter_cotrain --highpass_ref neighbor --gen_frac_schedule 0.5 --gen_slice_frac 1.0

# Bgate — forced gate: no generation at all → raw features + true labels
python run.py … --use_iter_cotrain --highpass_ref neighbor --gen_frac_schedule 0.001

# Raw-rolling (the best-performing generation arm): 2× doubling cap ("Brawd2") / uncapped
python run.py … --use_iter_cotrain --iter_skip_encoder --highpass_ref neighbor --gen_max_iters 2
python run.py … --use_iter_cotrain --iter_skip_encoder --highpass_ref neighbor
```

A supervision sweep is just a loop over `--ano_known_rate`:

```bash
for akr in 0.0 0.05 0.10 0.20 0.30; do
  for s in 100 101 102; do
    python run.py --dataset t_finance --seed $s --ano_known_rate $akr \
        --use_iter_cotrain --highpass_ref neighbor \
        > log/tfin_Bhp_s${s}_akr${akr}.log 2>&1
  done
done
```

A quick smoke test (CPU, minutes):

```bash
CUDA_VISIBLE_DEVICES="" python run.py --dataset photo --num_epoch 4 \
    --use_iter_cotrain --iter_skip_encoder --gen_max_iters 1 --gen_epochs 5 --num_steps 8
```

---

## 5. Outputs

- **stdout** ends with a `FINAL TEST RESULT` block (best-val checkpoint): full-set and
  sampled-mean±std **ROC-AUC**, **AUC-PRC**, AP. Report **AUC-PRC** (area under the PR curve),
  not AP — they differ.
- `log/<suffix><dataset><epochs>.txt` — per-epoch `margin bce rec total val_auc val_ap`, appended
  across runs. `<suffix>` is `vanilla_`, `enc_vanilla_` (arm C) or `iter_cotrain_` (arm B).
- `log/ckpt/best_<suffix><dataset>_s<seed>_hp<ref>_akr<rate>_gr<ratio>_gs<schedule>.pt` — best-val
  detector weights; the tag makes concurrent runs safe.
- `plt/<suffix><dataset>_test_metrics.png` and `plt/loss_*.png` — validation curves with the
  final test line, and the detector-loss curves.
- `cache/grpe_spd/*.pt` — SPD caches (regenerable).

---

## 6. Things to know before changing the code

- **Label convention.** `utils.load_mat` returns `normal_label_idx` = real normals only and
  `abnormal_label_idx` = sampled pseudo-abnormals **plus** the leaked real anchors
  (`all_abnormal_label_idx`). Upstream GGAD mislabeled the leaked anchors as normal in the BCE;
  this branch fixes it at the source, for every arm.
- **`'tf_finace'` in `run.py` is not a typo to fix.** t_finance intentionally falls through to
  `todense()` without row-normalization; all reported t_finance numbers were produced that way.
- **Reproducibility.** Seeds are fixed everywhere; `NUM_USER_EDGE_TYPES` in `grpe_encoder.py`
  is kept at 6 so the encoder's weight-init RNG matches the campaign runs. GPU `atomicAdd` in the
  encoder still gives small run-to-run variance — compare seeds, not single runs.
- **Where generation helps.** Across the campaigns, pseudo-abnormal generation helped only at very
  low anchor rates; once real anchors are plentiful, raw features + true labels (the gated regime)
  dominate, and the encoder space is a handicap outside t_finance at low rates. See `HANDOVER.md`
  §8 and the result dashboards.
- **Docs.** `GGAD_ARCHITECTURE.md` (stock detector), `QUICK_REFERENCE.md` and
  `DIFFUSION_IMPLEMENTATION_GUIDE.md` (EDM mechanics — note they still mention a since-removed
  `--use_loss_guided_emb_gen` arm), `HANDOVER.md` (repo history and verdicts).
