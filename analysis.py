"""Analysis utilities for the GGAD pipeline (kept OUT of run.py).

1. dump_detector_inputs(...) — env-gated hook called by run.py right before
   Phase 2 when DUMP_FEATS=<path.npz> is set: saves the detector's exact
   inputs (final features, pseudo/normal label sets, real anchors, splits).
2. Probe battery (python analysis.py [dumps_glob]) — space-quality and
   detector-view metrics over saved dumps:
     LR_full   oracle-label linear probe (space ceiling, eval view)
     cos_kshot anchor-centroid cosine AUC
     camo      cos(centroid_A, centroid_N) of normalized rows (1 = camouflaged)
     both_AUC  LR on the detector's actual label set -> real-anomaly test AUC
     pseudo_AUC / anchor_AUC  same probe with pseudos-only / anchors-only
     align     cos(mean(pseudo)-c_N, mean(real test abn)-c_N), within-space
"""
import glob
import os
import re
import sys
import warnings

import numpy as np


def dump_detector_inputs(path, *, features, features_eval, normal_label_idx,
                         abnormal_label_idx, real_anchor_idx, ano_label,
                         idx_train, idx_test):
    import torch
    def _sqz(t):
        return (t.squeeze(0).detach().cpu().numpy()
                if torch.is_tensor(t) else np.asarray(t))
    np.savez_compressed(
        path,
        features=_sqz(features),
        features_eval=_sqz(features_eval),
        normal_label_idx=np.asarray(list(normal_label_idx)),
        abnormal_label_idx=np.asarray(list(abnormal_label_idx)),
        real_anchor_idx=np.asarray(list(real_anchor_idx)),
        ano_label=np.asarray(ano_label).reshape(-1),
        idx_train=np.asarray(idx_train).reshape(-1),
        idx_test=np.asarray(idx_test).reshape(-1),
    )
    print(f"[dump] wrote {path}")


# ---------------------------------------------------------------------------
# Probe battery
# ---------------------------------------------------------------------------
def _aucsafe(y, s):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    return roc_auc_score(y, s) if len(set(y.tolist())) == 2 else float('nan')


def _lr_auc(Xtr, ytr, Xte, yte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    if len(set(ytr.tolist())) < 2:
        return float('nan')
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(class_weight='balanced', max_iter=3000).fit(
        sc.transform(Xtr), ytr)
    return _aucsafe(yte, m.predict_proba(sc.transform(Xte))[:, 1])


def _norm_rows(X):
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)


def probe_metrics(Xtrain, Xeval, z):
    """All probe metrics for one dump (see module docstring)."""
    N_lab = z['normal_label_idx']; A_lab = z['abnormal_label_idx']
    R = z['real_anchor_idx']; ano = z['ano_label']
    itr = z['idx_train']; ite = z['idx_test']
    P = np.setdiff1d(A_lab, R)
    yte = ano[ite]
    out = {}
    out['LR_full'] = _lr_auc(Xeval[itr], ano[itr], Xeval[ite], yte)
    En = _norm_rows(Xeval)
    cN = En[N_lab].mean(0)
    if len(R):
        cA = En[R].mean(0)
        out['cos_kshot'] = _aucsafe(yte, En[ite] @ cA)
        out['camo'] = float(cA @ cN / (np.linalg.norm(cA) * np.linalg.norm(cN)))
    pureN = np.setdiff1d(N_lab, A_lab)
    def probe(pos):
        idx = np.r_[pureN, pos]
        y = np.r_[np.zeros(len(pureN)), np.ones(len(pos))]
        return _lr_auc(Xtrain[idx], y, Xeval[ite], yte)
    out['both_AUC'] = probe(A_lab)
    out['pseudo_AUC'] = probe(P) if len(P) else float('nan')
    out['anchor_AUC'] = probe(R) if len(R) else float('nan')
    cN_t = Xtrain[pureN].mean(0)
    realA_te = ite[ano[ite] == 1]
    vA = Xeval[realA_te].mean(0) - cN_t
    if len(P):
        vP = Xtrain[P].mean(0) - cN_t
        out['align'] = float(vP @ vA /
                             (np.linalg.norm(vP) * np.linalg.norm(vA) + 1e-12))
    out['n_pseudo'] = len(P); out['n_anchor'] = len(R)
    return out


def main(pattern='dumps/*.npz', raw_mat=None):
    """Aggregate probe metrics over dumps; optional raw baseline via raw_mat."""
    warnings.filterwarnings('ignore')
    RAW = None
    if raw_mat:
        import scipy.io as sio, scipy.sparse as sp
        d = sio.loadmat(raw_mat)
        RAW = np.asarray(sp.lil_matrix(
            d['Attributes'] if 'Attributes' in d else d['X']).todense())
    rows = {}
    for f in sorted(glob.glob(pattern)):
        m = re.search(r'([A-Za-z]+)_([a-z0-9]+)_akr(\d+)(?:_[a-z0-9]*)?_s(\d+)\.npz',
                      os.path.basename(f))
        key = (m.group(2), m.group(3)) if m else (os.path.basename(f), '')
        z = np.load(f)
        rows.setdefault(key, []).append(probe_metrics(z['features'],
                                                      z['features_eval'], z))
        if RAW is not None:
            rows.setdefault(('raw', key[1]), []).append(
                probe_metrics(RAW, RAW, z))
    KEYS = ['LR_full', 'cos_kshot', 'camo', 'both_AUC', 'pseudo_AUC',
            'anchor_AUC', 'align']
    print(f"{'space':>8} {'akr':>4} " + " ".join(f"{k:>10}" for k in KEYS) + "   n")
    for key in sorted(rows):
        v = rows[key]
        line = f"{key[0]:>8} {key[1]:>4} "
        for k in KEYS:
            vals = [r[k] for r in v if not np.isnan(r.get(k, np.nan))]
            line += f"{np.mean(vals):>10.4f} " if vals else f"{'—':>10} "
        print(line + f"  {len(v)}")


if __name__ == '__main__':
    main(*(sys.argv[1:] or ()))
