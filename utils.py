import numpy as np
import networkx as nx
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


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def load_mat(dataset, train_rate=0.3, val_rate=0.1, ano_known_rate=0.023):

    """Load .mat dataset."""
    # data = sio.loadmat("./dataset/{}.mat".format(dataset))
    data = sio.loadmat("../../Dataset/T/{}.mat".format(dataset))
    label = data['Label'] if ('Label' in data) else data['gnd']
    attr = data['Attributes'] if ('Attributes' in data) else data['X']
    network = data['Network'] if ('Network' in data) else data['A']

    adj = sp.csr_matrix(network)
    feat = sp.lil_matrix(attr)

    # labels = np.squeeze(np.array(data['Class'], dtype=np.int64) - 1)
    # num_classes = np.max(labels) + 1
    # labels = dense_to_one_hot(labels, num_classes)

    ano_labels = np.squeeze(np.array(label))
    if 'str_anomaly_label' in data:
        str_ano_labels = np.squeeze(np.array(data['str_anomaly_label']))
        attr_ano_labels = np.squeeze(np.array(data['attr_anomaly_label']))
    else:
        str_ano_labels = None
        attr_ano_labels = None

    num_node = adj.shape[0]
    num_train = int(num_node * train_rate)
    num_val = int(num_node * val_rate)
    all_idx = list(range(num_node))
    random.shuffle(all_idx)
    idx_train = all_idx[: num_train]
    idx_val = all_idx[num_train: num_train + num_val]
    idx_test = all_idx[num_train + num_val:]
    # idx_test = all_idx[num_train:]
    print('Training', Counter(np.squeeze(ano_labels[idx_train])))
    print('Test', Counter(np.squeeze(ano_labels[idx_test])))
    # Sample some labeled normal nodes
    all_normal_label_idx = [i for i in idx_train if ano_labels[i] == 0]
    all_abnormal_label_idx = [i for i in idx_train if ano_labels[i] == 1]
    if int(len(all_abnormal_label_idx)) <= 1:
        clip_abnormal_idx = random.sample(all_abnormal_label_idx, 0) #Clip limited abnormal laebls
    else:
        num_abnormal =3#random.choice([0,1,2,3])
        clip_abnormal_idx = random.sample(all_abnormal_label_idx, num_abnormal) #Clip limited abnormal laebls
    # all_abnormal_label_idx = []
    all_idx = all_normal_label_idx + clip_abnormal_idx
    random.shuffle(all_idx)

    rate = 0.5  #  change train_rate to 0.3 0.5 0.6  0.8
    normal_label_idx = all_normal_label_idx[: int(len(all_idx) * rate)] + clip_abnormal_idx
    print('Training rate', rate)

    # normal_label_idx = all_normal_label_idx[: int(len(all_normal_label_idx) * 0.2)]
    # normal_label_idx = all_normal_label_idx[: int(len(all_normal_label_idx) * 0.25)]
    # normal_label_idx = all_normal_label_idx[: int(len(all_normal_label_idx) * 0.15)]
    # normal_label_idx = all_normal_label_idx[: int(len(all_normal_label_idx) * 0.10)]

    # contamination
    # real_abnormal_id = np.array(all_idx)[np.argwhere(ano_labels == 1).squeeze()].tolist()
    # add_rate = 0.1 * len(real_abnormal_id)
    # random.shuffle(real_abnormal_id)
    # add_abnormal_id = real_abnormal_id[:int(add_rate)]
    # normal_label_idx = normal_label_idx + add_abnormal_id
    # idx_test = np.setdiff1d(idx_test, add_abnormal_id, False)

    # contamination 
    # real_abnormal_id = np.array(all_idx)[np.argwhere(ano_labels == 1).squeeze()].tolist()
    # add_rate = 0.05 * len(real_abnormal_id)  #0.05 0.1  0.15
    # remove_rate = 0.15 * len(real_abnormal_id)
    # random.shuffle(real_abnormal_id)
    # add_abnormal_id = real_abnormal_id[:int(add_rate)]
    # remove_abnormal_id = real_abnormal_id[:int(remove_rate)]
    # normal_label_idx = normal_label_idx + add_abnormal_id
    # idx_test = np.setdiff1d(idx_test, remove_abnormal_id, False)

    # camouflage
    # real_abnormal_id = np.array(all_idx)[np.argwhere(ano_labels == 1).squeeze()].tolist()
    # normal_feat = np.mean(feat[normal_label_idx], 0)
    # replace_rate = 0.05 * normal_feat.shape[1]
    # feat[real_abnormal_id, :int(replace_rate)] = normal_feat[:, :int(replace_rate)]

    random.shuffle(normal_label_idx)
    # 0.05 for Amazon and 0.15 for other datasets
    if dataset in ['Amazon']:
        abnormal_label_idx = normal_label_idx[: int(len(normal_label_idx) * 0.05)]
    else:
        abnormal_label_idx = all_normal_label_idx[: int(len(all_idx) * rate * 0.15)]
    return adj, feat, ano_labels, all_idx, idx_train, idx_val, idx_test, ano_labels, str_ano_labels, attr_ano_labels, normal_label_idx, abnormal_label_idx, clip_abnormal_idx, len(clip_abnormal_idx)


def adj_to_dgl_graph(adj):
    """Convert adjacency matrix to dgl format."""
    # Direct conversion from sparse adjacency to DGL to avoid networkx deep copy overhead.
    if sp.issparse(adj):
        src, dst = adj.nonzero()
    else:
        src, dst = np.nonzero(adj)
    dgl_graph = dgl.graph((src.tolist(), dst.tolist()))
    return dgl_graph


def generate_rwr_subgraph(dgl_graph, subgraph_size):
    """Generate subgraph with RWR algorithm."""
    all_idx = list(range(dgl_graph.number_of_nodes()))
    reduced_size = subgraph_size - 1
    traces = dgl.contrib.sampling.random_walk_with_restart(dgl_graph, all_idx, restart_prob=1,
                                                           max_nodes_per_seed=subgraph_size * 3)
    subv = []

    for i, trace in enumerate(traces):
        subv.append(torch.unique(torch.cat(trace), sorted=False).tolist())
        retry_time = 0
        while len(subv[i]) < reduced_size:
            cur_trace = dgl.contrib.sampling.random_walk_with_restart(dgl_graph, [i], restart_prob=0.9,
                                                                      max_nodes_per_seed=subgraph_size * 5)
            subv[i] = torch.unique(torch.cat(cur_trace[0]), sorted=False).tolist()
            retry_time += 1
            if (len(subv[i]) <= 2) and (retry_time > 10):
                subv[i] = (subv[i] * reduced_size)
        subv[i] = subv[i][:reduced_size * 3]
        subv[i].append(i)

    return subv


def pretrain_gcn_embeddings(features, adj, raw_adj, abnormal_label_idx, normal_label_idx, args,
                            all_abnormal_label_idx=None):
    """Phase 0: GCN pre-training following GGAD-git convention.

    Full graph → GCN → all three GGAD losses on labeled nodes only.
    Returns GCN embeddings for all N nodes (needed by Phases 1a/1b/2).

    Args:
        features           : [1, N, ft_size]  raw node features
        adj                : [1, N, N]        normalised adjacency
        raw_adj            : [1, N, N]        raw adjacency (self-loop added)
        abnormal_label_idx : list of global abnormal node indices
        normal_label_idx   : list of global normal node indices
        args               : namespace (embedding_dim, negsamp_ratio, readout, lr, gen_epochs, mean, var)

    Returns:
        gcn_emb : [1, N, embedding_dim]  GCN embeddings for all N nodes (CPU)
    """
    from model import Model
    ft_size = features.shape[2]
    gcn_pre = Model(ft_size, args.embedding_dim, 'prelu', args.negsamp_ratio, args.readout, use_gcn=True)
    # Phase 0 only needs GCN + fc4; freeze fc1-fc3 classifier to avoid wasted gradients
    for name, param in gcn_pre.named_parameters():
        if name.startswith(('fc1.', 'fc2.', 'fc3.')):
            param.requires_grad = False
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, gcn_pre.parameters()), lr=args.lr)
    raw_adj_sq = torch.squeeze(raw_adj)   # [N, N] — local copy, caller's raw_adj unchanged

    print("Phase 0: GCN pre-training (loss_margin + loss_rec) ...")
    for epoch in range(args.gen_epochs):
        gcn_pre.train()
        opt.zero_grad()

        emb, _, _, emb_con, emb_abnormal = gcn_pre(
            features, adj, abnormal_label_idx, normal_label_idx, True, args)

        # Local affinity margin loss — following GGAD-git
        emb_sq = torch.squeeze(emb)
        emb_inf = torch.pow(torch.norm(emb_sq, dim=-1, keepdim=True), -1)
        emb_inf[torch.isinf(emb_inf)] = 0.
        emb_norm = emb_sq * emb_inf

        # Split normal_label_idx: exclude real GT abnormals (all_abnormal_label_idx)
        # from the "normal" affinity pool, matching Phase 2 logic.
        if all_abnormal_label_idx is not None and len(all_abnormal_label_idx) > 0:
            abn_set = set(all_abnormal_label_idx)
            pure_normal_idx   = [i for i in normal_label_idx if i not in abn_set]
            real_abnormal_idx = [i for i in normal_label_idx if i in abn_set]
            all_abn_idx       = abnormal_label_idx + real_abnormal_idx
        else:
            pure_normal_idx = normal_label_idx
            all_abn_idx     = abnormal_label_idx

        def labeled_affinity_p0(idx):
            rows     = emb_norm[idx]
            sim_rows = torch.mm(rows, emb_norm.T)
            adj_rows = raw_adj_sq[idx]
            deg_inv  = torch.pow(adj_rows.sum(1), -1)
            deg_inv[torch.isinf(deg_inv)] = 0.
            return (sim_rows * adj_rows).sum(1) * deg_inv

        aff_normal  = labeled_affinity_p0(pure_normal_idx).mean() if len(pure_normal_idx) > 0 else torch.tensor(0.0)
        aff_abnormal = labeled_affinity_p0(all_abn_idx).mean()    if len(all_abn_idx)     > 0 else torch.tensor(0.0)

        loss_margin = (0.7 - (aff_normal - aff_abnormal)).clamp_min(0)

        # Reconstruction loss
        loss_rec = torch.mean(torch.sqrt(
            torch.sum(torch.pow(emb_con - emb_abnormal, 2), 1)))

        (loss_margin + loss_rec).backward()
        opt.step()

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:03d}: margin={loss_margin.item():.4f}  "
                  f"rec={loss_rec.item():.4f}")

    # Extract GCN embeddings for all N nodes — needed because Phase 0 GCN is
    # separate from the Phase 2 FC-only detector (unlike GGAD-git where GCN IS
    # the detector). Phases 1a/1b/2/inference all consume these embeddings.
    gcn_pre.eval()
    with torch.no_grad():
        gcn_emb = gcn_pre.gcn2(gcn_pre.gcn1(features, adj), adj)   # [1, N, n_h]
    print(f"Phase 0 complete. GCN embedding shape: {gcn_emb.shape}")
    return gcn_emb


import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib
import os

matplotlib.use('Agg')


def test_plotting(epochs, auc_list, ap_list, dataset, method='Vanilla', num_epochs=0, log_suffix='',
                  best_ckpt_path=None, best_val_auc=0.0,
                  features_eval=None, adj=None,
                  abnormal_label_idx=None, normal_label_idx=None,
                  idx_test=None, ano_label=None,
                  model=None, device=None, args=None):
    """Evaluate on test set with best val checkpoint, then plot val curves with final test lines."""
    import torch
    from sklearn.metrics import roc_auc_score, average_precision_score
    import numpy as np

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
            auc_rounds, ap_rounds = [], []

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

            # Also compute full-set metrics
            full_test_auc = roc_auc_score(ano_label[safe_idx_test], logits_test)
            full_test_ap  = average_precision_score(ano_label[safe_idx_test], logits_test,
                                                     average='macro', pos_label=1, sample_weight=None)

            if len(auc_rounds) > 0:
                mean_auc = np.mean(auc_rounds)
                std_auc  = np.std(auc_rounds)
                mean_ap  = np.mean(ap_rounds)
                std_ap   = np.std(ap_rounds)
                final_test_auc = mean_auc
                final_test_ap  = mean_ap
                print(f'\n{"=" * 60}')
                print(f'FINAL TEST RESULT  (best-val checkpoint)')
                print(f'  Dataset : {dataset}')
                print(f'  Full-set  ROC-AUC : {full_test_auc:.4f}   AP : {full_test_ap:.4f}')
                print(f'  Sampled ({n_rounds} rounds, {sample_ratio:.0%} each):')
                print(f'    ROC-AUC : {mean_auc:.4f} ± {std_auc:.4f}')
                print(f'    AP      : {mean_ap:.4f} ± {std_ap:.4f}')
                print(f'{"=" * 60}\n')
            else:
                final_test_auc = full_test_auc
                final_test_ap  = full_test_ap
                print(f'\n{"=" * 60}')
                print(f'FINAL TEST RESULT  (best-val checkpoint)')
                print(f'  Dataset : {dataset}')
                print(f'  ROC-AUC : {full_test_auc:.4f}')
                print(f'  AP      : {full_test_ap:.4f}')
                print(f'  (subsampled rounds skipped — not enough class diversity)')
                print(f'{"=" * 60}\n')
        else:
            print('No valid idx_test in logits range; skipping final test evaluation.')

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
plt.rcParams['figure.dpi'] = 300  # 图片像素
plt.rcParams['figure.figsize'] = (8.5, 7.5)
# plt.rcParams['figure.figsize'] = (10.5, 9.5)
from matplotlib.backends.backend_pdf import PdfPages


def draw_pdf(message_normal, message_abnormal, message_real_abnormal, dataset, epoch):
    message_all = [np.squeeze(message_normal), np.squeeze(message_abnormal), np.squeeze(message_real_abnormal)]
    mu_0 = np.mean(message_all[0])  # 计算均值
    sigma_0 = np.std(message_all[0])
    # print('The mean of normal {}'.format(mu_0))
    # print('The std of normal {}'.format(sigma_0))
    mu_1 = np.mean(message_all[1])  # 计算均值
    sigma_1 = np.std(message_all[1])
    # print('The mean of abnormal {}'.format(mu_1))
    # print('The std of abnormal {}'.format(sigma_1))
    mu_2 = np.mean(message_all[2])  # 计算均值
    sigma_2 = np.std(message_all[2])
    # print('The mean of abnormal {}'.format(mu_2))
    # print('The std of abnormal {}'.format(sigma_2))
    n, bins, patches = plt.hist(message_all, bins=30, normed=1, label=['Normal', 'Outlier', 'Abnormal'])
    y_0 = mlab.normpdf(bins, mu_0, sigma_0)  # 拟合一条最佳正态分布曲线y
    y_1 = mlab.normpdf(bins, mu_1, sigma_1)  # 拟合一条最佳正态分布曲线y
    y_2 = mlab.normpdf(bins, mu_2, sigma_2)  # 拟合一条最佳正态分布曲线y
    # plt.plot(bins, y_0, 'g--', linewidth=3.5)  # 绘制y的曲线
    # plt.plot(bins, y_1, 'r--', linewidth=3.5)  # 绘制y的曲线
    plt.plot(bins, y_0, color='steelblue', linestyle='--', linewidth=7.5)  # 绘制y的曲线
    plt.plot(bins, y_1, color='darkorange', linestyle='--', linewidth=7.5)  # 绘制y的曲线
    plt.plot(bins, y_2, color='green', linestyle='--', linewidth=7.5)  # 绘制y的曲线
    plt.ylim(0, 20)

    # plt.xlabel('RAW-based Affinity', fontsize=25)
    # plt.xlabel('TAM-based Affinity', fontsize=25)
    # plt.ylabel('Number of Samples', size=25)
    plt.yticks(fontsize=30)
    plt.xticks(fontsize=30)
    # from matplotlib.pyplot import MultipleLocator
    # x_major_locator = MultipleLocator(0.02)
    # ax = plt.gca()
    # ax.xaxis.set_major_locator(x_major_locator)
    # plt.legend(loc='upper left', fontsize=30)
    # plt.title('Amazon'.format(dataset), fontsize=25)
    # plt.title('BlogCatalog', fontsize=50)
    plt.savefig('fig/{}/{}_{}.pdf'.format(dataset, dataset, epoch))
    plt.close()


def draw_pdf_methods(method, message_normal, message_abnormal, message_real_abnormal, dataset, epoch):
    message_all = [np.squeeze(message_normal), np.squeeze(message_abnormal), np.squeeze(message_real_abnormal)]
    mu_0 = np.mean(message_all[0])  # 计算均值
    sigma_0 = np.std(message_all[0])
    # print('The mean of normal {}'.format(mu_0))
    # print('The std of normal {}'.format(sigma_0))
    mu_1 = np.mean(message_all[1])  # 计算均值
    sigma_1 = np.std(message_all[1])
    # print('The mean of abnormal {}'.format(mu_1))
    # print('The std of abnormal {}'.format(sigma_1))
    mu_2 = np.mean(message_all[2])  # 计算均值
    sigma_2 = np.std(message_all[2])
    # print('The mean of abnormal {}'.format(mu_2))
    # print('The std of abnormal {}'.format(sigma_2))

    n, bins, patches = plt.hist(message_all, bins=30, normed=1, label=['Normal', 'Outlier', 'Abnormal'])
    y_0 = mlab.normpdf(bins, mu_0, sigma_0)  # 拟合一条最佳正态分布曲线y
    y_1 = mlab.normpdf(bins, mu_1, sigma_1)  # 拟合一条最佳正态分布曲线y
    y_2 = mlab.normpdf(bins, mu_2, sigma_2)  # 拟合一条最佳正态分布曲线y
    # plt.plot(bins, y_0, 'g--', linewidth=3.5)  # 绘制y的曲线
    # plt.plot(bins, y_1, 'r--', linewidth=3.5)  # 绘制y的曲线
    plt.plot(bins, y_0, color='steelblue', linestyle='--', linewidth=7.5)  # 绘制y的曲线
    plt.plot(bins, y_1, color='darkorange', linestyle='--', linewidth=7.5)  # 绘制y的曲线
    plt.plot(bins, y_2, color='green', linestyle='--', linewidth=7.5)  # 绘制y的曲线
    plt.ylim(0, 8)

    # plt.xlabel('RAW-based Affinity', fontsize=25)
    # plt.xlabel('TAM-based Affinity', fontsize=25)
    # plt.ylabel('Number of Samples', size=25)

    plt.yticks(fontsize=30)
    plt.xticks(fontsize=30)
    # plt.legend(loc='upper left', fontsize=30)
    # plt.title('Amazon'.format(dataset), fontsize=25)
    # plt.title('BlogCatalog', fontsize=50)
    plt.savefig('fig/{}/{}2/{}_{}.svg'.format(method, dataset, dataset, epoch))
    plt.close()
