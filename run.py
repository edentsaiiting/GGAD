import torch.nn as nn

from model import Model, GCN
from utils import *
from diff_gen import DiffusionGenerator

from sklearn.metrics import roc_auc_score
import random
import dgl
from sklearn.metrics import average_precision_score
import argparse
from tqdm import tqdm
import time

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [3]))
# os.environ["KMP_DUPLICATE_LnIB_OK"] = "TRUE"
# Set argument
parser = argparse.ArgumentParser(description='')

parser.add_argument('--dataset', type=str,
                    default='reddit')
parser.add_argument('--lr', type=float)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--embedding_dim', type=int, default=300)
parser.add_argument('--num_epoch', type=int)
parser.add_argument('--drop_prob', type=float, default=0.0)
parser.add_argument('--readout', type=str, default='avg')  # max min avg  weighted_sum
parser.add_argument('--auc_test_rounds', type=int, default=256)
parser.add_argument('--negsamp_ratio', type=int, default=1)
parser.add_argument('--mean', type=float, default=0.0)
parser.add_argument('--var', type=float, default=0.0)
parser.add_argument('--use_loss_guided_diffusion', action='store_true', default=False,
                    help='Use loss-guided diffusion to generate synthetic abnormal nodes')
parser.add_argument('--diff_hidden_dim', type=int, default=512,
                    help='Hidden dimension for diffusion model')
parser.add_argument('--gen_epochs', type=int, default=50,
                    help='Epochs for synthetic abnormal node generation')
parser.add_argument('--loss_guidance_weight', type=float, default=1.0,
                    help='Weight for guidance loss in diffusion model')
parser.add_argument('--num_steps', type=int, default=50,
                    help='Number of diffusion sampling steps for synthetic generation')
parser.add_argument('--no_guide_rec_term', action='store_true', default=False,
                    help='Exclude loss_rec from guidance loss; use only loss_margin for discriminative guidance')
parser.add_argument('--test_sample_rounds', type=int, default=10,
                    help='Number of random test subsamples to average for final test metrics')
parser.add_argument('--test_sample_ratio', type=float, default=0.8,
                    help='Fraction of idx_test to sample per round')



args = parser.parse_args()

if args.lr is None:
    if args.dataset in ['Amazon']:
        args.lr = 1e-3
    elif args.dataset in ['t_finance']:
        args.lr = 1e-3
    elif args.dataset in ['reddit']:
        args.lr = 1e-3
    elif args.dataset in ['photo']:
        args.lr = 1e-3
    elif args.dataset in ['elliptic']:
        args.lr = 1e-3

if args.num_epoch is None:
    if args.dataset in ['photo']:
        args.num_epoch = 100
    if args.dataset in ['elliptic']:
        args.num_epoch = 150
    if args.dataset in ['reddit']:
        args.num_epoch = 300
    elif args.dataset in ['t_finance']:
        args.num_epoch = 500
    elif args.dataset in ['Amazon']:
        args.num_epoch = 800
if args.dataset in ['reddit', 'photo']:
    args.mean = 0.02
    args.var = 0.01
else:
    args.mean = 0.0
    args.var = 0.0


print('Dataset: ', args.dataset)
print('Use loss-guided diffusion:', args.use_loss_guided_diffusion)

# Set random seed
dgl.random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
# os.environ['PYTHONHASHSEED'] = str(args.seed)
# os.environ['OMP_NUM_THREADS'] = '1'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Load and preprocess data
adj, features, labels, all_idx, idx_train, idx_val, \
idx_test, ano_label, str_ano_label, attr_ano_label, normal_label_idx, abnormal_label_idx, all_abnormal_label_idx, num_abnormal = load_mat(args.dataset)

if args.dataset in ['Amazon', 'tf_finace', 'reddit', 'elliptic']:
    features, _ = preprocess_features(features)
else:
    features = features.todense()

dgl_graph = adj_to_dgl_graph(adj)

nb_nodes = features.shape[0]
ft_size = features.shape[1]
raw_adj = adj
print(adj.sum())
adj = normalize_adj(adj)

raw_adj = (raw_adj + sp.eye(raw_adj.shape[0])).todense()
adj = (adj + sp.eye(adj.shape[0])).todense()

features = torch.FloatTensor(features[np.newaxis])
adj = torch.FloatTensor(adj[np.newaxis])
raw_adj = torch.FloatTensor(raw_adj[np.newaxis])
labels = torch.FloatTensor(labels[np.newaxis])

# idx_train = torch.LongTensor(idx_train)
# idx_val = torch.LongTensor(idx_val)
# idx_test = torch.LongTensor(idx_test)

if args.use_loss_guided_diffusion:
    diff_gen = DiffusionGenerator(
        features, adj, raw_adj,
        abnormal_label_idx, normal_label_idx,
        all_abnormal_label_idx, args,
    )
    features, features_eval = diff_gen.run()

# ============================================================================
# Standard GGAD Detector Training (PHASE 2) — always runs on CPU
# ============================================================================

# Phase 2 always runs on CPU: adj/raw_adj dense [N,N] matrices exhaust GPU memory
# for both vanilla and diffusion options (mirrors GGAD-git behavior).
device = torch.device('cpu')

# Diffusion ON:  FC-only detector, n_in = n_h (GCN+diffusion pre-processed space).
#                GCN was used in Phase 0 pre-training; removing it here prevents
#                neighbour smoothing from diluting the diffusion-enhanced embeddings.
# Diffusion OFF: GCN+FC, n_in = ft_size (raw features → GCN → FC).
n_in_detect = args.embedding_dim if args.use_loss_guided_diffusion else ft_size
model = Model(n_in_detect, args.embedding_dim, 'prelu', args.negsamp_ratio, args.readout,
              use_gcn=not args.use_loss_guided_diffusion)
optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

model = model.to(device)
#
# if torch.cuda.is_available():
#     print('Using CUDA')
#     model.cuda()
#     features = features.cuda()
#     adj = adj.cuda()
#     labels = labels.cuda()
#     raw_adj = raw_adj.cuda()

# idx_train = idx_train.cuda()
# idx_val = idx_val.cuda()
# idx_test = idx_test.cuda()
#
# if torch.cuda.is_available():
#     b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]).cuda())
# else:
#     b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]))

b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio], device=device))

# Set log file name
if args.use_loss_guided_diffusion:
    log_suffix = "diffusion_"
else:
    log_suffix = "vanilla_"

print("\n" + "=" * 80)
print("TRAINING CONFIGURATION")
print("=" * 80)
print(f"Method: {'Loss-Guided Diffusion' if args.use_loss_guided_diffusion else 'Original GCN-based'}")
print(f"Dataset: {args.dataset}")
print(f"Embedding Dimension: {args.embedding_dim}")
print(f"Number of Epochs: {args.num_epoch}")
print(f"Learning Rate: {args.lr}")
print(f"Number of Abnormal Nodes: {len(abnormal_label_idx)}")
print("=" * 80 + "\n")


if not args.use_loss_guided_diffusion:
    features_eval = features   # vanilla: train and eval use same raw features
# For diffusion, features_eval was set in Phase 1c (test nodes through frozen denoiser)

val_auc_history   = []
val_ap_history    = []
val_epoch_history = []
best_val_auc      = 0.0
best_ckpt_path    = f'./log/best_{log_suffix}{args.dataset}.pt'
auc      = 0.0
AP       = 0.0

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
            if epoch % 10 == 0:
                # save data for tsne
                pass

                # tsne_data_path = 'draw/tfinance/tsne_data_{}.mat'.format(str(epoch))
                # io.savemat(tsne_data_path, {'emb': np.array(emb.cpu().detach()), 'ano_label': ano_label,
                #                             'abnormal_label_idx': np.array(abnormal_label_idx),
                #                             'normal_label_idx': np.array(normal_label_idx)})

            # BCE loss — labeled nodes only: [zeros(n_normal), ones(n_abn)] (following GGAD-git)
            # logits: [1, n_normal+n_abn, 1] from emb_combine; lbl matches exactly
            lbl = torch.unsqueeze(
                torch.cat((torch.zeros(len(normal_label_idx)), torch.ones(len(abnormal_label_idx)))), 1
            ).unsqueeze(0).to(device)

            loss_bce = b_xent(logits, lbl)
            loss_bce = torch.mean(loss_bce)

            # Local affinity margin loss
            # Avoid full [N, N] similarity matrix — compute only the labeled rows needed.
            emb = torch.squeeze(emb)           # [N, n_h]
            raw_adj = torch.squeeze(raw_adj)   # [N, N]

            emb_inf = torch.norm(emb, dim=-1, keepdim=True)
            emb_inf = torch.pow(emb_inf, -1)
            emb_inf[torch.isinf(emb_inf)] = 0.
            emb_norm = emb * emb_inf           # [N, n_h]

            # Split normal_label_idx: pure normals vs camouflage abnormals (in all_abnormal_label_idx)
            normal_idx        = [i for i in normal_label_idx if i not in all_abnormal_label_idx]
            real_abnormal_idx = [i for i in normal_label_idx if i in all_abnormal_label_idx]
            all_abn_idx       = abnormal_label_idx + real_abnormal_idx

            def labeled_affinity(idx):
                """Affinity for a subset of nodes: [n_idx, N] instead of [N, N]."""
                rows     = emb_norm[idx]                        # [n_idx, n_h]
                sim_rows = torch.mm(rows, emb_norm.T)          # [n_idx, N]
                adj_rows = raw_adj[idx]                        # [n_idx, N]
                deg_inv  = torch.pow(adj_rows.sum(1), -1)
                deg_inv[torch.isinf(deg_inv)] = 0.
                return (sim_rows * adj_rows).sum(1) * deg_inv  # [n_idx]

            affinity_normal_mean = (labeled_affinity(normal_idx).mean()
                                    if len(normal_idx) > 0
                                    else torch.tensor(0.0, device=device))
            affinity_abnormal_mean = (labeled_affinity(all_abn_idx).mean()
                                      if len(all_abn_idx) > 0
                                      else torch.tensor(0.0, device=device))

            # if epoch % 10 == 0:
            #     real_abnormal_label_idx = np.array(all_idx)[np.argwhere(ano_label == 1).squeeze()].tolist()
            #     real_normal_label_idx = np.array(all_idx)[np.argwhere(ano_label == 0).squeeze()].tolist()
            #     overlap = list(set(real_abnormal_label_idx) & set(real_normal_label_idx))
            #
            #     real_affinity, index = torch.sort(affinity[real_abnormal_label_idx])
            #     real_affinity = real_affinity[:300]
            #     draw_pdf(np.array(affinity[real_normal_label_idx].detach().cpu()),
            #              np.array(affinity[abnormal_label_idx].detach().cpu()),
            #              np.array(real_affinity.detach().cpu()), args.dataset, epoch)

            # Ego Centric Loss
            confidence_margin = 0.7
            loss_margin = (confidence_margin - (affinity_normal_mean - affinity_abnormal_mean)).clamp_min(min=0)

            diff_attribute = torch.pow(emb_con - emb_abnormal, 2)
            loss_rec = torch.mean(torch.sqrt(torch.sum(diff_attribute, 1)))
            
            loss = 1 * loss_margin + 1 * loss_bce + 1 * loss_rec

            loss.backward()
            optimiser.step()
            end_time = time.time()
            total_time += end_time - start_time
            print('Total time is', total_time)
            if epoch % 2 == 0:
                logits = np.squeeze(logits.cpu().detach().numpy())
                lbl = np.squeeze(lbl.cpu().detach().numpy())
                auc = roc_auc_score(lbl, logits)
                # print('Traininig {} AUC:{:.4f}'.format(args.dataset, auc))
                # AP = average_precision_score(lbl, logits, average='macro', pos_label=1, sample_weight=None)
                # print('Traininig AP:', AP)

                print("Epoch:", '%04d' % (epoch), "train_loss_margin=", "{:.5f}".format(loss_margin.item()))
                print("Epoch:", '%04d' % (epoch), "train_loss_bce=", "{:.5f}".format(loss_bce.item()))
                print("Epoch:", '%04d' % (epoch), "rec_loss=", "{:.5f}".format(loss_rec.item()))
                print("Epoch:", '%04d' % (epoch), "train_loss=", "{:.5f}".format(loss.item()))
                print("=====================================================================")
            if epoch % 2 == 0:
                model.eval()
                train_flag = False
                emb, emb_combine, logits, emb_con, emb_abnormal = model(features_eval, adj, abnormal_label_idx, normal_label_idx,
                                                                        train_flag, args)
                # Validation eval for model selection
                safe_idx_val = [i for i in idx_val if i < logits.shape[1]]
                val_auc = 0.0
                val_ap  = 0.0
                if len(safe_idx_val) > 0:
                    logits_val = np.squeeze(logits[:, safe_idx_val, :].cpu().detach().numpy())
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

if val_epoch_history:
    method = 'Diffusion' if args.use_loss_guided_diffusion else 'Vanilla'
    test_plotting(val_epoch_history, val_auc_history, val_ap_history,
                  args.dataset, method=method, num_epochs=args.num_epoch, log_suffix=log_suffix,
                  best_ckpt_path=best_ckpt_path, best_val_auc=best_val_auc,
                  features_eval=features_eval, adj=adj,
                  abnormal_label_idx=abnormal_label_idx, normal_label_idx=normal_label_idx,
                  idx_test=idx_test, ano_label=ano_label,
                  model=model, device=device, args=args)