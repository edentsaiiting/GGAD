# GGAD Project Handover — File Inventory

*Written 2026-08-24 for handover to the next authors. Repo: fork of [mala-lab/GGAD](https://github.com/mala-lab/GGAD) (NeurIPS 2024) at `edentsaiiting/GGAD`, remote `upstream` = original.*

Every file in this directory falls into one of four buckets:

1. **Upstream GGAD** — untouched files from the original repo (baselines, docs, the `src/` DGraph pipeline).
2. **Upstream, modified by us** — 9 files, +1232/−260 lines total; the substance is in `run.py`, `model.py`, `utils.py`.
3. **Created by us — method code & docs** — the encoder, diffusion, co-training modules and 3 markdown docs.
4. **Created by us — experiment scripts** — ~80 shell scripts (2 reusable dispatchers, ~40 orchestrators, the rest one-off runners/mop-ups).

> **⚠ Git state (fix before handover):** the current branch `GCN-FC-diff_emb_aug` only tracks ~20 of our files. **The method modules — `iter_cotrain.py`, `grpe_encoder.py`, `mha.py`, `diff_feat_aug.py`, `analysis.py` — plus all four docs and the unstaged edits to `run.py`/`model.py`/`utils.py` are untracked or uncommitted.** They exist only on this machine. Commit them first.
>
> **Cleanup done 2026-08-24:** regenerable data deleted (SPD cache 379 G, detector checkpoints 6.3 G, DGraph-Fin pkls 1.4 G), and **all 82 campaign shell scripts were moved out of the repo to `../GGAD_scripts_archive/`** (kept there for provenance; §5–6 below describe them in their archived location). `eval/visual.py` (empty placeholder) removed. **Final pass, same day:** result/artifact dirs deleted outright per the owner's call — `log/` (342 M campaign logs + results CSVs), `dumps/` (107 M probe dumps), `plt/`, both `__pycache__/`, `src/log/`, `src/pytorch_models/` — the authors receive conclusions through the docs and dashboards, not raw run outputs. `CORRECTIONS_SUMMARY.md` dropped from the doc set. Repo now ~320 MB, mostly `.git` history.
>
> **Code deep-clean, same day:** all experimental/trial args and their code paths removed from the Python modules (encoder search & early-stop, `--diag_only`, 3-term loss-placement variants, k-hop context machinery incl. `mha.py` edge-weight plumbing, D+anchor arms in `model.py`, pseudo-init modes, InfoNCE); device/scheduler machinery (`GPU_RESERVE_MB`/exit-97) deleted; the leaked-label Convention fix-up moved INTO `utils.load_mat` (which now returns 9 values, already-repartitioned); dataset path overridable via **`GGAD_DATA_ROOT`** env. **Both single-shot generation arms deleted**: `diff_gen.py` (embedding-space) and, in a final pass, `diff_feat_aug.py` (raw-space Braw arm) — the rolling scheme (`iter_cotrain.py` on the `loss_guided_diffusion.py` engine) is the one generation path kept; full experimental versions preserved in `../GGAD_scripts_archive/` (`run.py.experimental`, `diff_gen.py.experimental`, `diff_feat_aug.py.experimental`). `NUM_USER_EDGE_TYPES` stays 6 so encoder weight-init RNG (and thus campaign numbers at a given seed) remains reproducible. All surviving arms smoke-tested end-to-end after each trim.

---

## 0. Branch map

| branch | contents |
|---|---|
| `main`, `Dataset_exp` | pristine upstream mirror — zero files beyond `mala-lab/GGAD` |
| `Normal` | small side branch (3 own commits): dgraph/normal-eval work in `src/` |
| `Abnormal&Normal` | early experiment branch, ancestor of current |
| **`GCN-FC-diff_emb_aug`** ← current | tip of tracked work (adds `diff_gen.py`, `loss_guided_diffusion.py`, `eval/`, early logs) + all the untracked work above |

---

## 1. Upstream GGAD files (unmodified)

```
README.md  framework.png  poster-NeurIPS2024.pdf
model_*.py + <name>.py pairs — competing-method baselines, each runnable standalone:
  aegis.py/model_AEGIS.py · anomalyDAE.py/model_AnomalyDAE.py · dominant.py/model_domaint.py
  gaan.py/model_gaan.py · ocgnn.py/model_ocgnn.py · tam.py/model_tam.py/utils_tam.py
src/ — upstream's separate DGraph-Fin pipeline (GraphSAGE/DOMINANT/AnomalyDAE handlers):
  main.py  model.py  model_handler*.py  graphsage*.py  layers.py  utils.py  dgraph.yml
```

## 2. Upstream files we modified

| file | what changed |
|---|---|
| `run.py` | The experiment front-end, **trimmed 2026-08-24 for handover** to the three surviving arms: vanilla D (default), GRPE encoder (`--use_grpe_encoder`), rolling co-training (`--use_iter_cotrain`, `--iter_skip_encoder` raw-rolling, `gen_max_iters`/`gen_target_ratio`/`gen_frac_schedule`/`gen_slice_frac`, diffusion knobs `diff_hidden_dim`/`gen_epochs`/`num_steps`/`loss_guidance_weight`); plus the **`--ano_known_rate` (akr) anchor-leak axis**, `--highpass_ref`, `--fc_only`, `DUMP_FEATS` hook, best-val checkpointing, memory-efficient row-wise margin, and the **leaked-abnormal label fix** (now inside `utils.load_mat`). ~30 experimental/trial flags were deleted across the cleanup passes (encoder search & early-stop, `--diag_only`, loss-placement variants, k-hop context, D+anchor arms, pseudo-init modes, both single-shot generation arms `--use_loss_guided_emb_gen`/`--use_feat_augmentor`, device/scheduler machinery); the full experimental version is preserved at `../GGAD_scripts_archive/run.py.experimental`. |
| `model.py` | +125/−142. Detector rework: `use_gcn=False` raw-feature/FC-only path, high-pass `[h, h−ref]` head, GRPE/augmentor branch, `det_real_anchor_pos`/`det_rap_bce_only` (real anchors as BCE positives ± margin sees real geometry). |
| `utils.py` | `load_mat(ano_known_rate=…)` sizes the leaked-anchor pool and **returns already-repartitioned label sets** (9 values; the leaked-label fix lives here now); dataset path `../../Dataset/T/` overridable via `GGAD_DATA_ROOT`; fast `adj_to_dgl_graph`; `test_plotting()` (multi-round test eval, ROC-AUC / **AUC-PRC** / AP + PNGs under `plt/`). Dead plotting/pretrain helpers deleted in the deep-clean. |
| `requirements.txt` / `requirements2.txt` | comment only / new alternate pinned env (py3.7, torch 1.6 + torch-geometric stack). |
| `src/graphsage.py`, `src/utils.py`, `src/dgraph.yml` | 1-line env/path/epoch tweaks for the DGraph side-experiment. |
| `.gitignore` | new — excludes `log/ plt/ __pycache__/ pytorch_models/` + data caches. |

**Vanilla-D caveat:** our "vanilla" differs deliberately from upstream — the label fix, per-node rec-loss (+1e-12 guard), t_finance epochs 500→350, and a changed pseudo-pool slice in `load_mat` — so D numbers are not bit-identical to the upstream repo even at akr=0.

## 3. Method code we created (Python)

```
iter_cotrain.py         IterCotrainer: GRPE-encoder ⇄ EDM-diffusion co-training loop (--use_iter_cotrain).
                        Per iter: train GRPE on (N_k, A_pool_k) → guided-diffuse new pseudo-abnormals in place →
                        flip labels, grow pool; stops at |N|/|A_pool| ≤ 5.5. Key knob: iter_skip_encoder
                        (RAW-ROLLING — the winning arm). Pseudo-init is dimsample (rho 0.3), the campaign default.
grpe_encoder.py         Phase-0 GRPE graph-Transformer (SPD-hop + labeled-pair edge-type biases, cached;
                        chunked cross-attention inference). Holds the 3-term encoder loss
                        (BCE + affinity-margin + ego-recon). Config from grpe_config.json (nhead=8 ⇒ d_model % 8 == 0).
mha.py                  GRPE attention primitives (hop/edge-bias MHA, self + cross attention) used by grpe_encoder.
loss_guided_diffusion.py EDM backbone (MLPDiffusion, Precond, EDMLoss, guided sampler;
                        AnomalyGenerator facade) — the generation engine iter_cotrain runs on.  [tracked in git]
analysis.py             Offline probe battery over DUMP_FEATS .npz dumps: LR_full oracle, cos_kshot,
                        camo (camouflage cosine), detector-view AUCs — source of the featAUC/camo regime metrics.
grpe_config.json        Encoder hyperparameters (d_model 80, 4 layers, nhead 8, max_hop 5, 50 ep, chunk 8000).
```

(Both single-shot generation arms were deleted: `diff_gen.py` (embedding-space) in the deep-clean and `diff_feat_aug.py` (raw-space, `--use_feat_augmentor`/Braw) in the final pass — each archived as `../GGAD_scripts_archive/*.experimental`. `loss_guided_diffusion.py` is NOT an arm and stays: it is the EDM engine `iter_cotrain.py` generates with.)

Dependency shape: `run.py` imports all of the above. Three arms remain: vanilla D (no flags), GRPE encoder (`--use_grpe_encoder`, C_hp), and rolling co-training (`--use_iter_cotrain`, B_hp; `--iter_skip_encoder` = the winning raw-rolling).

## 4. Docs we wrote — read in this order

1. `GGAD_ARCHITECTURE.md` — ground-truth walkthrough of the stock detector (`model.py`): losses, shapes, forward paths. **Read first.**
2. `QUICK_REFERENCE.md` — one-page cheat sheet for the diffusion rewrite + run commands.
3. `DIFFUSION_IMPLEMENTATION_GUIDE.md` — full spec of the EDM pipeline (for whoever modifies `loss_guided_diffusion.py`).

(`CORRECTIONS_SUMMARY.md`, a historical changelog of the old→new diffusion API, was dropped from the handover set.)

**Caveat:** all three predate the later campaigns, and QUICK_REFERENCE / DIFFUSION_IMPLEMENTATION_GUIDE still document the deleted `--use_loss_guided_emb_gen` embedding-space arm — read them for the EDM mechanics (still live in `loss_guided_diffusion.py`), not the exact run commands. The experimental verdict (see §8) is that pseudo-abnormal *generation helps only at very low anchor rates* — the diffusion arm these docs describe is not the recommended configuration.

## 5. The two dispatchers everything else wraps — *archived in `../GGAD_scripts_archive/`*

**Note:** none of the shell scripts in §5–6 live in the repo anymore. `run_numabn.sh` in the archive remains the authoritative record of the config-name → `run.py`-flag mapping until it is extracted into a doc.

```
run_numabn.sh   CORE. (gpu, dataset, config, seed, rates…) → run.py per akr point.
                Owns the authoritative config-name → flag mapping for the whole zoo:
                B_hp/C_hp, B_es/C_es, D, Dplus, DplusB, Broll*/Bshot*/Bstep* (15/30/50 endpoints,
                *s = slice rule v2, BshotF* = full-pool), Braw, Bgate. Legacy (non-s) schedule names
                carry a DO-NOT-RE-RUN warning. Resumable (skip-if-done via "Full-set  ROC-AUC" grep),
                EXTRA/TAG env hooks, MPS-bypass baked in. Logs → log/numabn.
run_3term.sh    CORE. Same shape for the 3-term-encoder-loss era (configs B_hp/C_hp + variants b/c, D).
                Logs → log/3term with done-markers the queue_* scripts poll.
```

Plus reusable ablation runners: `run_B_ablation.sh`, `run_C_ablation.sh` (InfoNCE × high-pass 4-cell), `run_D_seed.sh`, `run_amazon3.sh`, `run_reddit.sh`.

## 6. Campaign orchestrators (~40) — grouped by standing — *archived in `../GGAD_scripts_archive/`*

**Core campaigns (produced the reported results):**

```
orchestrate_bgate.sh / orchestrate_bgate_tfin.sh   Bgate forced-gate arm (raw + true labels, no gen)
orchestrate_braw.sh                                Braw raw-space gen — completes the 2×2 factorial
orchestrate_dplus.sh / orchestrate_dplusb.sh       D+anchors routing/attribution arms
orchestrate_elliptic.sh                            elliptic 5-rate campaign (6 configs, n=3)
orchestrate_photo30.sh / orchestrate_photo_highanchor.sh    photo →30% family / high-akr n=10
orchestrate_reddit30.sh / orchestrate_reddit_hi50.sh        reddit →30% overlay / high-akr →50% family
orchestrate_shotfull.sh                            full-pool vs slice ablation (BshotF*, n=3)
orchestrate_slice.sh                               slice-rule (v2) relaunch after the termination-rule fix
orchestrate_tfin5.sh                               ★ canonical t_finance scheduler — 5-rate axis
                                                   {0.0,.05,.10,.20,.30}, always safe to relaunch
orchestrate_tfin50.sh                              t_finance →50% heavy arm (Broll50+Bstep3)
```

**One-off (complete; kept for provenance):** `orchestrate_3term/B/D/reddit/reddit10/reddit10a/numabn*` (first sweep + seed extensions), `orchestrate_gen30fill/photo10_reddit5/pr10/prf10` (grid fills & mop-ups), `orchestrate_photo_encdiag/reddit_encsearch*/reddit_lrdiag/sepdiag` (diagnostics; OAT search was null, SEP ruled out, LR-diag established reddit's 0.65 ceiling).

**Superseded / retired:** `orchestrate_gen15.sh` (pre-audit slice rule), `orchestrate_tfin30.sh` (old 11-rate axis), `orchestrate_tfin30_n3.sh.retired`, `orchestrate_tfin6.sh.retired` (→ became tfin5), `watchdog_chain.sh.retired`, `watchdog_tfin30_n3.sh.retired`.

**One-off helpers/mop-ups:** `mopup_heavy.sh`, `mopup_tfin_light.sh`, `rerun_lrdiag_oom.sh`, `rerun_tfin_sep_subset.sh`, `fill_aC.sh`, `helper_tfin50_lane.sh`, `queue_reddit_3term.sh`, `queue_tfin_B.sh`, `resume_3term.sh`, `launch_tfin5_when_free.sh` (**hard-coded stale PIDs — not reusable**), `amazon10_3term.sh`, `reddit10_3term.sh`, plus early-era studies (`run_sweep_v2/v3.sh`, `run_tfinance_abcd.sh`, `run_rho_*.sh`, `run_init_photo.sh`, `run_photoB_grid.sh`, `run_tfinB_*.sh`, `run_c_variance.sh`, `run_photo_rho0_tfinA.sh`, `run_labeledHP.sh`, `run_numabn.sh` wrappers `run_C_10seed.sh`, `run_tfinD10.sh`, `run_dmodel8/20.sh`).

**Naming trap:** `run_dmodel20.sh` actually runs `--grpe_d_model 24` (logs to `log/amazon_dm24`).

## 7. Data directories & ship list

| path | size | verdict |
|---|---|---|
| `../../Dataset/T/` (i.e. `/home/edentsai/Dataset/T/`) | 5.8 G | **MUST SHIP — the only external dependency.** `utils.load_mat` hardcodes `../../Dataset/T/{dataset}.mat` — place it two levels above the repo cwd or edit that line. Not in the repo. |
| ~~`log/`~~, ~~`dumps/`~~, ~~`plt/`~~, ~~`__pycache__/`~~ | ~460 M | **deleted 2026-08-24** — raw campaign logs, results CSVs, probe dumps, and curves are not part of the handover; conclusions live in the docs and dashboards. |
| ~~`log/ckpt/`~~, ~~`cache/grpe_spd/`~~, ~~`pytorch_models/`~~ | 6.3 G + 379 G + 1.4 G | **deleted 2026-08-24** (all regenerable: checkpoints by rerunning, SPD caches auto-rebuilt on demand). |

## 8. Operational gotchas & where the science landed

- **MPS hijack:** a foreign CUDA MPS server freezes our jobs at `cuInit`. Every mature script sets `CUDA_MPS_PIPE_DIRECTORY=/tmp/no-mps-edentsai`. Keep doing this.
- **Long jobs must be `setsid`-detached** or they die at SSH session teardown.
- **Killing an orchestrator:** kill the parent first or the round loop respawns workers; beware `pgrep` flag-substring collisions (bracket-trick the pattern).
- Mature orchestrators are **idempotent**: flock'd text queue in `log/numabn`, freest-GPU dispatch, done-cell grep, ≤3 catch-up rounds — safe to relaunch anytime.
- **Metric convention:** report **AUC-PRC** (log field `AUC-PRC`, not `AP`); never compare an n=1 config against a cherry-picked baseline seed.
- **Final verdict (2×2 factorial complete):** *raw features + true labels* (gated) dominates photo/reddit at all akr; pseudo-abnormal **generation is harmful in both feature spaces** once real anchors are abundant; the encoder's only niche is t_finance at low akr; reddit is supervision-saturated at a ~0.67 ceiling (real ~0.65 feature ceiling). Generation helps only at few anchors; `B_hp` saturates ~0.98 at akr 15–30 %.
- Result dashboards (interactive): *Known-abnormal supervision sweep* and *Gated vs GGAD architectures* — Claude artifacts owned by edentsai101@gmail.com; ask for share links.
