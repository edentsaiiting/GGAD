# GGAD Diffusion Model - Implementation Guide

## Overview

This document explains the corrected implementation of loss-guided diffusion for synthetic abnormal node generation in GGAD. The key fix is **separating training objectives**: diffusion training uses EDMLoss (like DiffGAD), while GGAD losses (margin, BCE, reconstruction) are applied only to the detector.

---

## Architecture: Two-Phase Pipeline

### Phase 1: Diffusion-based Node Generation (loss_guided_diffusion.py)

**Training Objective**: EDMLoss (weighted MSE reconstruction)
- **NOT** GGAD losses
- Follows DiffGAD's approach exactly

**Models**:
1. **Unconditional Diffusion Model** (`dm_unconditional`)
   - Trained on embeddings without conditioning
   - Learns prototype as weighted average of reconstructions
   - Formula: `proto = softmax(cos_sim(proto, reconstructed), t=5) @ reconstructed`

2. **Conditional Diffusion Model** (`dm_conditional`)  
   - Trained with prototype guidance (classifier-free guidance)
   - Takes same proto from unconditional as input
   - Learns alternative denoising path conditioned on proto

**Sampling** (Generation):
- Uses both models with classifier-free guidance
- Formula: `d_curr = (1 + weight) * d_curr_free - weight * d_curr_proto`
- Generates synthetic abnormal embeddings without graph structure

**Loss Function**: 
```python
EDMLoss (Efficient Diffusion Model Loss):
- Weighted MSE: weight(sigma) * (D_yn - target)^2
- weight(sigma) = (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2
- Ensures balanced loss across different noise levels
```

### Phase 2: Standard GGAD Detector on Expanded Graph (model.py + run.py)

**Input**:
- Combined features: original + synthetic (zeros)
- Expanded adjacency: original + synthetic node connections
- Combined labels: normal (0) + synthetic abnormal (1) + real abnormal (variable)

**Architecture**:
- Two-layer GCN for structural encoding
- Ego-centric FC classifier (during training)
- Uniform FC application (during testing)

**Training Loss** (three GGAD components):
1. **Margin Loss** (Isolation):
   ```
   affinity_normal = mean(affinity[normal_nodes])
   affinity_abnormal = mean(affinity[abnormal_nodes])
   loss_margin = max(0, 0.7 - (affinity_normal - affinity_abnormal))
   ```
   Makes abnormal nodes isolated (low affinity)

2. **BCE Loss** (Classification):
   ```
   loss_bce = BCEWithLogits(logits, labels)
   ```
   Binary classification: normal (0) vs abnormal (1)

3. **Reconstruction Loss** (Deviation):
   ```
   loss_rec = mean(sqrt(sum((emb_con - emb_abnormal)^2)))
   ```
   Measures divergence between context-based and actual embeddings

**Combined Loss**:
```
total_loss = 1.0 * loss_margin + 1.0 * loss_bce + 1.0 * loss_rec
```

---

## Key Changes from Previous Version

### What Changed

| Aspect | Previous | Corrected |
|--------|----------|-----------|
| **Diffusion Loss** | GGAD losses (margin+BCE+recon) | EDMLoss (weighted MSE) |
| **Prototype** | Not learned | Weighted average during training |
| **Guidance in Training** | During diffusion training | Only during sampling (classifier-free) |
| **GGAD Losses** | Used in diffusion training | ONLY in detector training |
| **Models** | Single loss-guided model | Two models (unconditional + conditional) |
| **Sampling** | Not implemented | Classifier-free guidance blending |

### Why This is Correct

1. **DiffGAD Consistency**: Uses exact same training approach as DiffGAD
2. **Proper Separation**: Losses used where they're designed to be used
3. **Loss Independence**: Diffusion doesn't depend on downstream detector objectives
4. **Classifier-Free Guidance**: Only applied during inference, not training
5. **Prototype Learning**: Natural outcome of training, not engineered constraint

---

## Implementation Details

### AnomalyGenerator Class

Three main training phases:

```python
# Phase 1: Train unconditional model
proto = generator.train_unconditional(embeddings)

# Phase 2: Train conditional model (uses proto from phase 1)
generator.train_conditional(embeddings)

# Phase 3: Generate synthetic nodes (uses both models)
synthetic_nodes = generator.generate(num_samples, num_steps=50)
```

### Loss Function Components

**EDMLoss Forward**:
```python
def forward(self, denoise_fn, data, proto, proto_alpha):
    # Sample random noise levels from log-normal distribution
    rnd_normal = torch.randn(data.shape[0])
    sigma = exp(rnd_normal * P_std + P_mean)
    
    # Compute loss weight based on noise level
    weight = (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2
    
    # Add noise and denoise
    noisy_data = data + sigma * noise
    denoised = denoise_fn(noisy_data, sigma, proto, proto_alpha)
    
    # Weighted MSE
    loss = weight * (denoised - data)^2
    
    return loss.mean()
```

**Prototype Update**:
```python
if epoch == 0:
    proto = mean(reconstructed)
else:
    # Similarity-weighted average
    sim_scores = cos_sim(proto, reconstructed)
    weights = softmax(sim_scores / t, where t=5)
    proto = weights @ reconstructed
```

### Classifier-Free Guidance Sampling

```python
def sample_dm_free(proto_net, free_net, noise, num_steps, proto, weight):
    for t_curr, t_next in reverse_time_steps:
        # Get predictions from both models
        denoised_proto = proto_net(x, t, proto, proto_alpha)
        denoised_free = free_net(x, t)
        
        # Compute derivatives
        d_proto = (x - denoised_proto) / t
        d_free = (x - denoised_free) / t
        
        # Blend with guidance weight
        d = (1 + weight) * d_free - weight * d_proto
        
        # Update
        x = x + (t_next - t) * d
    
    return x
```

---

## Training in run.py

### Phase 1 Execution

```python
if args.use_loss_guided_emb_gen:
    # Initialize
    generator = AnomalyGenerator(
        embedding_dim=ft_size,
        hidden_dim=args.diff_hidden_dim,
        lr=args.lr,
        num_gen_epochs=args.gen_epochs
    )
    generator.initialize()
    
    # Train (no GGAD losses used here!)
    proto = generator.train_unconditional(embeddings)
    generator.train_conditional(embeddings)
    
    # Generate
    synthetic_nodes = generator.generate(num_samples=num_abnormal)
    
    # Expand graph with synthetic nodes
    features_expanded = cat([features, zeros(num_synthetic)])
    adj_expanded = expand_adjacency(adj, synthetic_nodes_connections)
```

### Phase 2 Execution

Standard GGAD detector training on expanded graph:

```python
# Model forward pass
emb, emb_combine, logits, emb_con, emb_abnormal = model(
    features_expanded, adj_expanded,
    abnormal_label_idx, normal_label_idx,
    train_flag=True
)

# Compute GGAD losses (margin + BCE + reconstruction)
loss_margin = compute_margin_loss(affinity_normal, affinity_abnormal)
loss_bce = BCELoss(logits, labels)
loss_rec = compute_reconstruction_loss(emb_con, emb_abnormal)

# Total loss
total_loss = loss_margin + loss_bce + loss_rec
total_loss.backward()
optimizer.step()
```

---

## Hyperparameters

### Diffusion Model
- `embedding_dim`: 300 (matches feature dimension)
- `hidden_dim`: 512 (denoising network dimension)
- `lr`: 1e-3 (learning rate)
- `num_gen_epochs`: 50 (training epochs)
- `proto_alpha`: 0.5 (prototype conditioning weight)
- `P_mean`: -1.2 (log-normal noise distribution mean)
- `P_std`: 1.2 (log-normal noise distribution std)
- `sigma_data`: 0.5 (data scale preconditioning)

### Detector (GGAD)
- `embedding_dim`: 300
- `num_epoch`: 300 (for reddit,  dataset-specific)
- `lr`: 1e-3
- `weight_decay`: 0.0
- `negsamp_ratio`: 1
- `confidence_margin`: 0.7 (used in margin loss)

---

## Comparison: DiffGAD vs GGAD+Diffusion

| Component | DiffGAD | GGAD+Diffusion |
|-----------|---------|----------------|
| **Purpose** | Anomaly detection via diffusion | Data augmentation via diffusion |
| **Diffusion Loss** | EDMLoss ✓ | EDMLoss ✓ |
| **Detector** | Autoencoder → anomaly scores | GCN → binary classification |
| **Input to Diffusion** | Autoencoder embeddings | Raw features |
| **Output** | Anomaly scores | Synthetic nodes → expanded graph |
| **Downstream** | Direct evaluation | GCN detector training |
| **GGAD Losses** | Not applicable | Phase 2 detector training |

---

## File Structure

```
GGAD/
├── loss_guided_diffusion.py    # Phase 1: Diffusion training (EDMLoss)
│   ├── MLPDiffusion            # Denoising network
│   ├── Precond                 # Preconditioning wrapper
│   ├── EDMLoss                 # Training loss (weighted MSE)
│   ├── DiffusionGenerator      # Model wrapper
│   ├── AnomalyGenerator        # High-level interface
│   └── Sampling functions      # Inference with classifier-free guidance
│
├── model.py                    # Phase 2: GCN + detector
│   ├── GCN                     # Graph convolutional layer
│   └── Model                   # Full detector architecture
│
└── run.py                      # Main orchestration
    ├── Phase 1: Diffusion generation
    └── Phase 2: Detector training (uses GGAD losses)
```

---

## Execution

### Basic Usage
```bash
# Without diffusion augmentation (original GGAD)
python run.py --dataset reddit --num_epoch 300

# With diffusion augmentation (GGAD + synthetic nodes)
python run.py --dataset reddit --use_loss_guided_emb_gen --num_epoch 300
```

### Expected Output

```
Phase 1: Training unconditional diffusion model...
  Unconditional Epoch 0: Loss=0.123456
  ...
  Unconditional Epoch 40: Loss=0.045678
Early stopping (unconditional)

Phase 1: Training conditional diffusion model...
  Conditional Epoch 0: Loss=0.098765
  ...
  Conditional Epoch 40: Loss=0.038901
Early stopping (conditional)

Phase 1: Generating synthetic nodes with classifier-free guidance...
Generated synthetic abnormal nodes shape: torch.Size([50, 300])

Combined dataset: 3956 nodes (original: 3906 + synthetic: 50)
Expanded adjacency: torch.Size([1, 3956, 3956])

Phase 2: Standard GGAD Detector Training on Combined Dataset
  ...
  Epoch: 0000 train_loss_margin= 0.23456
  Epoch: 0000 train_loss_bce= 0.34567
  Epoch: 0000 rec_loss= 0.12345
  Epoch: 0000 train_loss= 0.70368
  Testing reddit AUC:0.7234
```

---

## Key Insights

### Why Separate Training Objectives?

1. **Diffusion Goal**: Learn distribution of embeddings to generate realistic new samples
   - Training signal: reconstruction quality (EDMLoss)
   - Optimizer: MSE between denoised and original
   
2. **Detector Goal**: Classify nodes as normal or abnormal
   - Training signal: margin (isolation) + BCE (classification) + reconstruction (deviation)
   - Optimizer: Three complementary objectives

3. **No Interference**: Diffusion doesn't care about downstream classification
   - Diffusion only needs to generate valid embeddings
   - Detector responsibility: distinguish normal from generated abnormal

### Classifier-Free Guidance

- **Training Time**: Models learn unconditional and conditional paths independently
- **Inference Time**: Combine both for better control and quality
- **Formula**: Blend gradient from conditional (proto-guided) and unconditional paths
- **Tuning**: `weight` parameter controls how much the prototype influences generation

---

## Validation Checklist

- [x] Diffusion uses EDMLoss (not GGAD losses)
- [x] Prototype learned during training (weighted average)
- [x] Two independent diffusion models (unconditional + conditional)
- [x] Classifier-free guidance implemented correctly
- [x] GGAD losses applied only to detector training
- [x] Expanded graph properly constructed
- [x] Model forward pass handles ego-centric training correctly
- [x] Syntax validation passed

---

## References

- **DiffGAD**: [DiffGAD Paper](https://github.com/GRAND-Lab/DiffGAD)
  - Loss: EDMLoss with Precond wrapper
  - Prototype: Weighted average of reconstructions
  - Sampling: Classifier-free guidance
  
- **GGAD**: Original detector architecture used on expanded dataset
  - Three-component loss (margin + BCE + reconstruction)
  - Ego-centric training on combined nodes
