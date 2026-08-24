# GGAD Model Architecture Validation

## Overview

GGAD is a **Graph Anomaly Detection** model that uses:
1. **GCN Encoder**: Convert node features to embeddings based on graph structure
2. **FC Layers**: Binary classifier to distinguish normal from abnormal nodes
3. **Ego-Centric Training**: Train specifically on normal nodes + context-based embeddings

---

## Model Architecture

### Class: Model (from model.py)

```python
class Model(nn.Module):
    def __init__(self, n_in, n_h, activation, negsamp_round, readout):
        # GCN layers
        self.gcn1 = GCN(n_in, n_h, activation)      # Input features → hidden
        self.gcn2 = GCN(n_h, n_h, activation)       # Hidden → hidden
        self.gcn3 = GCN(n_h, n_h, activation)       # Unused in forward
        
        # FC classifier layers
        self.fc1 = nn.Linear(n_h, n_h/2, bias=False)           # Combine
        self.fc2 = nn.Linear(n_h/2, n_h/4, bias=False)         # Hidden
        self.fc3 = nn.Linear(n_h/4, 1, bias=False)             # Binary output
        
        # Context generation
        self.fc4 = nn.Linear(n_h, n_h, bias=False)  # Generate context embeddings
```

### Forward Pass (model.forward)

#### TRAINING MODE (train_flag=True)

**Input**:
- `seq1`: Node features (batch=1, num_nodes, feature_dim)
- `adj`: Adjacency matrix (batch=1, num_nodes, num_nodes)
- `sample_abnormal_idx`: Indices of abnormal nodes to sample during training
- `normal_idx`: Indices of normal/labeled nodes
- `args.var`, `args.mean`: Noise parameters

**Process**:

1. **GCN Encoding**:
   ```
   h_1 = gcn1(seq1, adj)           # (1, num_nodes, n_h)
   emb = gcn2(h_1, adj)            # (1, num_nodes, n_h) - Final embeddings
   ```

2. **Extract Abnormal Embeddings**:
   ```
   emb_abnormal = emb[:, sample_abnormal_idx, :]  # (1, num_abnormal, n_h)
   noise = randn(...) * args.var + args.mean     # Add noise
   emb_abnormal = emb_abnormal + noise            # Noisy embeddings
   ```

3. **Generate Context Embeddings** (Ego-Centric):
   ```
   neigh_adj = adj[0, sample_abnormal_idx, :]    # (num_abnormal, num_nodes)
   emb_con = neigh_adj @ emb[0, :, :]            # (num_abnormal, n_h)
                                                  # Weighted sum of neighbors
   emb_con = relu(fc4(emb_con))                   # Context embeddings
   ```

4. **Combine for Training**:
   ```
   emb_combine = [emb[:, normal_idx, :], emb_con.unsqueeze(0)]
                                                  # (1, num_normal + num_abnormal, n_h)
   ```

5. **Binary Classification**:
   ```
   f_1 = fc1(emb_combine)          # (1, num_nodes, n_h/2)
   f_1 = relu(f_1)
   f_2 = fc2(f_1)                  # (1, num_nodes, n_h/4)
   f_2 = relu(f_2)
   f_3 = fc3(f_2)                  # (1, num_nodes, 1) - Logits
   ```

**Output**:
- `emb`: All node embeddings (1, num_nodes, n_h)
- `emb_combine`: Combined normal + context embeddings (1, num_normal + num_abnormal, n_h)
- `f_3`: Classification logits (1, num_nodes, 1)
- `emb_con`: Context embeddings (num_abnormal, n_h)
- `emb_abnormal`: Noisy abnormal embeddings (1, num_abnormal, n_h)

#### TESTING MODE (train_flag=False)

**Input**: Same as training

**Process**:

1. **GCN Encoding** (same):
   ```
   h_1 = gcn1(seq1, adj)
   emb = gcn2(h_1, adj)            # (1, num_nodes, n_h)
   ```

2. **Direct Classification** (NO context generation):
   ```
   f_1 = fc1(emb)                  # Apply FC on ALL embeddings
   f_1 = relu(f_1)
   f_2 = fc2(f_1)
   f_2 = relu(f_2)
   f_3 = fc3(f_2)                  # (1, num_nodes, 1)
   ```

**Output**:
- `emb`: All node embeddings (1, num_nodes, n_h)
- `emb_combine`: None
- `f_3`: Anomaly scores for ALL nodes (1, num_nodes, 1)
- `emb_con`: None
- `emb_abnormal`: Noisy abnormal embeddings

---

## Key Differences: Training vs Testing

| Aspect | Training | Testing |
|--------|----------|---------|
| **Input nodes to classify** | normal_idx + sample_abnormal_idx | All nodes |
| **Embeddings used** | emb[:, normal_idx, :] + context embeddings (emb_con) | All embeddings emb |
| **FC layer input** | emb_combine (hybrid) | emb (pure GCN) |
| **Processing** | 2 different paths (normal vs abnormal) | Single path (all nodes) |

---

## Loss Functions (from run.py)

### 1. BCE Loss
```python
# Binary classification loss
labels = [0 if normal, 1 if abnormal]
loss_bce = BCEWithLogitsLoss(logits, labels)
```

### 2. Margin Loss (Ego-Centric)
```python
# Compute affinity from graph structure
affinity = sum(sim_matrix * adj) / degree

affinity_normal_mean = mean(affinity[normal_idx])
affinity_abnormal_mean = mean(affinity[abnormal_idx])

loss_margin = max(0, confidence_margin - (affinity_normal - affinity_abnormal))
# confidence_margin = 0.7
```

### 3. Reconstruction Loss (Ego-Centric)
```python
# Measure gap between context and sample embeddings
diff_attribute = (emb_con - emb_abnormal) ^ 2
loss_rec = mean(sqrt(sum(diff_attribute, dim=1)))
```

### Total Loss
```python
loss = loss_margin + loss_bce + loss_rec
```

---

## Data Flow Diagram

```
TRAINING:
Input: features (1, N, d), adj (1, N, N)
   ↓
GCN Layer 1: features → hidden (1, N, h)
   ↓
GCN Layer 2: hidden → embeddings (1, N, h)
   ↓
┌─ Normal path: emb[:, normal_idx, :] (num_normal, h)
│
├─ Abnormal path:
│  ├─ Extract: emb[:, abnormal_idx, :] → Add noise
│  ├─ Context: neigh_adj @ emb → relu(fc4) → emb_con (num_abnormal, h)
│  └─ Result: emb_con (num_abnormal, h)
│
└─ Combine: [normal_path, abnormal_path] (1, N_labeled, h)
   ↓
FC Layer 1: (N_labeled, h) → relu → (N_labeled, h/2)
   ↓
FC Layer 2: (N_labeled, h/2) → relu → (N_labeled, h/4)
   ↓
FC Layer 3: (N_labeled, h/4) → logits (N_labeled, 1)
   ↓
Compute losses (BCE, Margin, Reconstruction)

TESTING:
Input: features (1, N, d), adj (1, N, N)
   ↓
GCN Layer 1→2: → embeddings (1, N, h)
   ↓
FC Layer 1: (1, N, h) → relu → (1, N, h/2)
   ↓
FC Layer 2: (1, N, h/2) → relu → (1, N, h/4)
   ↓
FC Layer 3: (1, N, h/4) → logits (1, N, 1)
   ↓
Output: Anomaly scores for all nodes
```

---

## Legitimate Inputs/Outputs

### Training Forward Call
```python
emb, emb_combine, logits, emb_con, emb_abnormal = model(
    features,              # (1, num_nodes, feature_dim)
    adj,                   # (1, num_nodes, num_nodes)
    abnormal_label_idx,    # List of abnormal node indices
    normal_label_idx,      # List of normal node indices
    train_flag=True,       # Enable training mode
    args                   # Config (var, mean, ...)
)

Returns:
- emb: (1, num_nodes, embedding_dim)
- emb_combine: (1, num_normal+num_abnormal_sampled, embedding_dim)
- logits: (1, num_normal+num_abnormal_sampled, 1)
- emb_con: (num_abnormal_sampled, embedding_dim)
- emb_abnormal: (1, num_abnormal_sampled, embedding_dim)
```

### Testing Forward Call
```python
emb, emb_combine, logits, emb_con, emb_abnormal = model(
    features,              # (1, num_nodes, feature_dim)
    adj,                   # (1, num_nodes, num_nodes)
    abnormal_label_idx,    # Still passed but not used in test mode
    normal_label_idx,      # Still passed but not used in test mode
    train_flag=False,      # Disable ego-centric processing
    args
)

Returns:
- emb: (1, num_nodes, embedding_dim)
- emb_combine: None
- logits: (1, num_nodes, 1)  # Scores for ALL nodes
- emb_con: None
- emb_abnormal: (1, num_abnormal_sampled, embedding_dim) - stale, not updated
```

---

## Key Properties

1. **Ego-Centric Training**: Only trains on normal nodes + filtered abnormal nodes with their context representations
2. **Graph-Based**: Uses both GCN (structural encoding) + ego-centric context (neighborhood influence)
3. **Hybrid Embeddings during Training**: Combines pure GCN embeddings with context-generated embeddings
4. **Full-Graph Testing**: During evaluation, uses pure GCN embeddings for all nodes uniformly

---

## Integration with Loss-Guided Diffusion

When using `--use_loss_guided_emb_gen`:

1. **Phase 1**: Generate synthetic abnormal nodes as embeddings (no GCN)
2. **Phase 2**: Expand dataset and train standard GGAD model:
   - Synthetic nodes added as regular nodes in the dataset
   - Features padded with zeros
   - Adjacency expanded with sparse connections
   - Standard GGAD training/testing proceeds as normal

The synthetic nodes generated by diffusion are treated as regular nodes in the expanded graph during Phase 2.

---

## Hyperparameters

- `embedding_dim` (n_h): Output dimension of GCN layers (default 512 or 300)
- `confidence_margin`: Threshold for margin loss (0.7 in code)
- `var, mean`: Noise added to abnormal embeddings during training
- `negsamp_ratio`: Positive weight in BCE loss (class weight)
