import os
import numpy as np
import scipy.sparse as sp
import torch
import scipy.io as sio
import random
import dgl
from collections import Counter


def sparse_to_tuple(sparse_mx, insert_batch=False):
    """Convert sparse matrix to tuple representation."""
    """Set insert_batch=True if you want to insert a batch dimension."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if insert_batch:
            coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            values = mx.data
            shape = (1,) + mx.shape
        else:
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


# Dataset directory: .mat files live OUTSIDE the repo. Default matches this
# lab's layout (two levels above the repo cwd); override via GGAD_DATA_ROOT.
DATA_ROOT = os.environ.get('GGAD_DATA_ROOT', '../../Dataset/T')


def load_mat(dataset, train_rate=0.3, val_rate=0.1, ano_known_rate=0.0):

    """Load .mat dataset."""
    data = sio.loadmat("{}/{}.mat".format(DATA_ROOT, dataset))
    label = data['Label'] if ('Label' in data) else data['gnd']
    attr = data['Attributes'] if ('Attributes' in data) else data['X']
    network = data['Network'] if ('Network' in data) else data['A']

    adj = sp.csr_matrix(network)
    feat = sp.lil_matrix(attr)

    ano_labels = np.squeeze(np.array(label))

    num_node = adj.shape[0]
    num_train = int(num_node * train_rate)
    num_val = int(num_node * val_rate)
    all_idx = list(range(num_node))
    random.shuffle(all_idx)
    idx_train = all_idx[: num_train]
    idx_val = all_idx[num_train: num_train + num_val]
    idx_test = all_idx[num_train + num_val:]
    print('Training', Counter(np.squeeze(ano_labels[idx_train])))
    print('Test', Counter(np.squeeze(ano_labels[idx_test])))
    # Sample some labeled normal nodes
    all_normal_label_idx = [i for i in idx_train if ano_labels[i] == 0]
    all_abnormal_label_idx = [i for i in idx_train if ano_labels[i] == 1]
    if len(all_abnormal_label_idx) <= 1:
        clip_abnormal_idx = []
    else:
        # ano_known_rate>0 sizes the known/leaked real-abnormal pool as a fraction
        # of training abnormals (rounded, >=1, capped); <=0 keeps the legacy 3.
        if ano_known_rate and ano_known_rate > 0:
            num_abnormal = int(round(ano_known_rate * len(all_abnormal_label_idx)))
            num_abnormal = max(1, min(num_abnormal, len(all_abnormal_label_idx)))
        else:
            num_abnormal = 3  # legacy default
        print(f'[load_mat] known/leaked abnormals = {num_abnormal} '
              f'(ano_known_rate={ano_known_rate}, of {len(all_abnormal_label_idx)} '
              f'training abnormals)')
        clip_abnormal_idx = random.sample(all_abnormal_label_idx, num_abnormal)
    all_idx = all_normal_label_idx + clip_abnormal_idx
    random.shuffle(all_idx)

    rate = 0.5  # fraction of training normals used as labeled normals
    normal_label_idx = all_normal_label_idx[: int(len(all_idx) * rate)] + clip_abnormal_idx
    print('Training rate', rate)

    random.shuffle(normal_label_idx)
    # 0.05 for Amazon and 0.15 for other datasets
    if dataset in ['Amazon']:
        abnormal_label_idx = normal_label_idx[: int(len(normal_label_idx) * 0.05)]
    else:
        abnormal_label_idx = all_normal_label_idx[: int(len(all_idx) * rate * 0.15)]

    # Re-partition (moved from run.py): normal set = real normals only;
    # abnormal set = sampled pseudo-abn + leaked reals (dedup needed: Amazon's
    # pseudo slice is drawn from normal_label_idx, which still contains the
    # leaked reals at this point).
    _abn_set = set(abnormal_label_idx)
    abnormal_label_idx = list(abnormal_label_idx) + \
        [i for i in clip_abnormal_idx if i not in _abn_set]
    _leaked = set(clip_abnormal_idx)
    normal_label_idx = [i for i in normal_label_idx if i not in _leaked]
    print(f"[load_mat] abnormal class = {len(abnormal_label_idx)} "
          f"({len(_abn_set)} sampled + {len(clip_abnormal_idx)} leaked); "
          f"normal class = {len(normal_label_idx)}")

    return (adj, feat, ano_labels, idx_train, idx_val, idx_test,
            normal_label_idx, abnormal_label_idx, clip_abnormal_idx)


def adj_to_dgl_graph(adj):
    """Convert adjacency matrix to dgl format."""
    # Direct conversion from sparse adjacency to DGL to avoid networkx deep copy overhead.
    if sp.issparse(adj):
        src, dst = adj.nonzero()
    else:
        src, dst = np.nonzero(adj)
    dgl_graph = dgl.graph((src.tolist(), dst.tolist()))
    return dgl_graph


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


def test_plotting(epochs, auc_list, ap_list, dataset, method='Vanilla', num_epochs=0, log_suffix='',
                  best_ckpt_path=None, best_val_auc=0.0,
                  features_eval=None, adj=None,
                  abnormal_label_idx=None, normal_label_idx=None,
                  idx_test=None, ano_label=None,
                  model=None, device=None, args=None,
                  loss_epochs=None, loss_total=None, loss_bce=None,
                  loss_margin=None, loss_rec=None):
    """Evaluate on test set with best val checkpoint, then plot val curves with final test lines."""
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                 precision_recall_curve, auc)

    final_test_auc = 0.0
    final_test_ap  = 0.0

    # ── Load best checkpoint and run final test eval ─────────────────────────
    if model is not None and best_ckpt_path is not None and os.path.exists(best_ckpt_path):
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
        print(f'\nLoaded best checkpoint (val AUC={best_val_auc:.4f}) from {best_ckpt_path}')

    if (model is not None and features_eval is not None and adj is not None
            and idx_test is not None and ano_label is not None):
        model.eval()
        with torch.no_grad():
            _, _, logits_final, _, _ = model(
                features_eval, adj, abnormal_label_idx, normal_label_idx, False, args)
        safe_idx_test = [i for i in idx_test if i < logits_final.shape[1]]
        if len(safe_idx_test) > 0:
            logits_test = np.squeeze(logits_final[:, safe_idx_test, :].cpu().numpy())

            # Multi-round subsampled test evaluation for more robust metrics
            n_rounds    = getattr(args, 'test_sample_rounds', 10)
            sample_ratio = getattr(args, 'test_sample_ratio', 0.8)
            sample_size  = max(1, int(len(safe_idx_test) * sample_ratio))
            # AUC-PRC = area under the precision-recall curve (trapezoidal), the
            # PR analogue of ROC-AUC; reported alongside it as the headline pair.
            # Distinct from AP, the step-wise PR estimator that avoids the curve's
            # optimistic linear interpolation.
            auc_rounds, ap_rounds, prc_rounds = [], [], []

            for r in range(n_rounds):
                sub_idx = random.sample(safe_idx_test, sample_size)
                # Map sub_idx back to positions within safe_idx_test for logits indexing
                idx_map = {v: i for i, v in enumerate(safe_idx_test)}
                pos     = [idx_map[s] for s in sub_idx]
                sub_logits = logits_test[pos]
                sub_labels = ano_label[sub_idx]
                # Need both classes present for AUC
                if len(set(sub_labels)) < 2:
                    continue
                auc_rounds.append(roc_auc_score(sub_labels, sub_logits))
                ap_rounds.append(average_precision_score(sub_labels, sub_logits,
                                                          average='macro', pos_label=1, sample_weight=None))
                prec, rec, _ = precision_recall_curve(sub_labels, sub_logits, pos_label=1)
                prc_rounds.append(auc(rec, prec))

            # Also compute full-set metrics
            full_test_auc = roc_auc_score(ano_label[safe_idx_test], logits_test)
            full_test_ap  = average_precision_score(ano_label[safe_idx_test], logits_test,
                                                     average='macro', pos_label=1, sample_weight=None)
            prec, rec, _  = precision_recall_curve(ano_label[safe_idx_test], logits_test, pos_label=1)
            full_test_prc = auc(rec, prec)

            if len(auc_rounds) > 0:
                mean_auc = np.mean(auc_rounds)
                std_auc  = np.std(auc_rounds)
                mean_ap  = np.mean(ap_rounds)
                std_ap   = np.std(ap_rounds)
                mean_prc = np.mean(prc_rounds)
                std_prc  = np.std(prc_rounds)
                final_test_auc = mean_auc
                final_test_ap  = mean_ap
                print(f'\n{"=" * 60}')
                print(f'FINAL TEST RESULT  (best-val checkpoint)')
                print(f'  Dataset : {dataset}')
                print(f'  Full-set  ROC-AUC : {full_test_auc:.4f}   AUC-PRC : {full_test_prc:.4f}   AP : {full_test_ap:.4f}')
                print(f'  Sampled ({n_rounds} rounds, {sample_ratio:.0%} each):')
                print(f'    ROC-AUC : {mean_auc:.4f} ± {std_auc:.4f}')
                print(f'    AUC-PRC : {mean_prc:.4f} ± {std_prc:.4f}')
                print(f'    AP      : {mean_ap:.4f} ± {std_ap:.4f}')
                print(f'{"=" * 60}\n')
            else:
                final_test_auc = full_test_auc
                final_test_ap  = full_test_ap
                print(f'\n{"=" * 60}')
                print(f'FINAL TEST RESULT  (best-val checkpoint)')
                print(f'  Dataset : {dataset}')
                print(f'  ROC-AUC : {full_test_auc:.4f}')
                print(f'  AUC-PRC : {full_test_prc:.4f}')
                print(f'  AP      : {full_test_ap:.4f}')
                print(f'  (subsampled rounds skipped — not enough class diversity)')
                print(f'{"=" * 60}\n')
        else:
            print('No valid idx_test in logits range; skipping final test evaluation.')

    # ── Plot detector-loss curves vs epoch ────────────────────────────────────
    if loss_epochs is not None and len(loss_epochs) > 0:
        figL, axL = plt.subplots(figsize=(8, 5))
        axL.plot(loss_epochs, loss_total, color='black', lw=2.0, label='total')
        if loss_bce is not None:
            axL.plot(loss_epochs, loss_bce, color='steelblue', lw=1.5, label='bce')
        if loss_margin is not None and any(v != 0 for v in loss_margin):
            axL.plot(loss_epochs, loss_margin, color='crimson', lw=1.5, label='margin')
        if loss_rec is not None and any(v != 0 for v in loss_rec):
            axL.plot(loss_epochs, loss_rec, color='seagreen', lw=1.5, label='rec')
        axL.set_xlabel('Epoch', fontsize=12)
        axL.set_ylabel('Detector loss (every 2 ep)', fontsize=12)
        axL.set_title(f'Detector loss | {method} | {num_epochs} ep | {dataset}', fontsize=11)
        axL.legend(fontsize=10)
        axL.grid(True, alpha=0.3)
        # Unique filename per run so parallel seeds/variants don't clobber.
        _tag = f"s{getattr(args, 'seed', '')}_hp{getattr(args, 'highpass_ref', 'none')}"
        os.makedirs('./plt', exist_ok=True)
        figL.tight_layout()
        figL.savefig(f'./plt/loss_{log_suffix}{dataset}_{num_epochs}ep_{_tag}.png',
                     dpi=120, bbox_inches='tight')
        plt.close(figL)

    # ── Plot val curves with final test dashed lines ──────────────────────────
    _, axes = plt.subplots(1, 2, figsize=(14, 5))

    test_values = [final_test_auc, final_test_ap]
    for ax, values, label, color, test_val in [
        (axes[0], auc_list, 'Val ROC-AUC', 'steelblue',   test_values[0]),
        (axes[1], ap_list,  'Val AP',      'darkorange',   test_values[1]),
    ]:
        ax.plot(epochs, values, color=color, linewidth=2, marker='o', markersize=3, label='Val')

        max_val = max(values)
        max_ep  = epochs[values.index(max_val)]
        ax.scatter([max_ep], [max_val], color='red', s=60, zorder=5)
        ax.annotate(
            f'Val Max: {max_val:.4f}',
            xy=(max_ep, max_val),
            xytext=(max_ep, max_val + (max(values) - min(values)) * 0.05 + 1e-6),
            fontsize=10, color=color,
            arrowprops=dict(arrowstyle='->', color=color),
        )

        # Final test dashed line
        ax.axhline(y=test_val, color='crimson', linestyle='--', linewidth=1.5,
                   label=f'Test={test_val:.4f}')
        ax.legend(fontsize=10)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(label, fontsize=12)
        ax.set_title(
            f'{label} | {method} | {num_epochs} ep | {dataset}\n'
            f'Test AUC={final_test_auc:.4f}  AP={final_test_ap:.4f}',
            fontsize=11,
        )
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs('./plt', exist_ok=True)
    save_path = f'./plt/{log_suffix}{dataset}_test_metrics.png'
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'Test metrics plot saved to {save_path}')
