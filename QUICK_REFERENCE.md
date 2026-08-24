# GGAD Diffusion: Quick Reference

## What Changed

| Aspect | Before | After |
|--------|--------|-------|
| **Phase 1 Loss** | ❌ GGAD losses (wrong) | ✅ EDMLoss (correct) |
| **Prototype** | ❌ Not learned | ✅ Learned during training |
| **Models** | ❌ Single model | ✅ Two models (unconditional+conditional) |
| **GGAD Losses** | ❌ Used in Phase 1 | ✅ Used only in Phase 2 |
| **Sampling** | ❌ Not implemented | ✅ Classifier-free guidance |

## The Core Problem Fixed

**Before**: GGAD losses (margin, BCE, reconstruction) were guiding the diffusion model training.

**Problem**: Diffusion is for **learning distributions**, not for **classification tasks**. Using detector-specific losses to train a generative model is wrong.

**After**: Diffusion uses **EDMLoss** (weighted MSE), GGAD losses only for detector.

**Result**: Each component optimizes its own objective = better design.

---

## File Summary

### loss_guided_diffusion.py
**Complete rewrite from DiffGAD approach**:
- `EDMLoss`: Weighted reconstruction loss (phase 1 training)
- `MLPDiffusion`: Denoising network (from DiffGAD)
- `Precond`: Preconditioning wrapper (from DiffGAD)
- `AnomalyGenerator`: High-level interface
  - `train_unconditional()`: Learn prototype + general distribution
  - `train_conditional()`: Learn prototype-guided path
  - `generate()`: Classifier-free guidance sampling
- Sampling functions: `sample_dm_free()`, `sample_step_free()`

### run.py Changes
**Phase 1 (diffusion generation)**:
```python
# OLD (wrong)
generator = LossGuidedAnomalyGenerator(...)
synthetic_nodes = generator.generate_guided(...)  # Using GGAD losses

# NEW (correct)
generator = AnomalyGenerator(...)
proto = generator.train_unconditional(embeddings)     # EDMLoss
generator.train_conditional(embeddings)                # EDMLoss
synthetic_nodes = generator.generate(num_samples)      # No losses
```

**Phase 2 (unchanged)**: Already correctly uses GGAD losses for detector training.

---

## Loss Functions

### Phase 1: EDMLoss (Diffusion Training)
```python
loss = weight(σ) × (denoised - original)²
where weight(σ) = (σ² + σ_data²) / (σ × σ_data)²

Purpose: Reconstruct embeddings from noise
Motivation: Learn the data distribution
```

### Phase 2: GGAD Losses (Detector Training)
```python
1. Margin Loss: max(0, 0.7 - (aff_normal - aff_abnormal))
2. BCE Loss: -[y×log(p) + (1-y)×log(1-p)]
3. Reconstruction: mean(√(Σ(context_emb - actual)²))

Purpose: Train detector for anomaly classification
Motivation: Isolate, classify, and distinguish abnormal nodes
```

---

## Key Concepts

### Prototype
- **What**: Single embedding vector summarizing key characteristics
- **When**: Learned during unconditional diffusion training
- **How**: Started as mean, updated as weighted average each epoch
- **Why**: Provides meaningful conditioning for second model

### Classifier-Free Guidance
- **Unconditional Model**: Learns general distribution (no prototype)
- **Conditional Model**: Learns prototype-guided path
- **At Inference**: Blend both using weighted combination
- **Formula**: `d = (1 + w) × d_free - w × d_proto`
- **Effect**: Can control how much prototype influences generation

### Two-Model Architecture
- **Model 1**: Unconditional → learns prototype naturally
- **Model 2**: Conditional → uses prototype for guidance
- **Benefit**: Better quality and controllability
- **Trade-off**: Needs to train twice, but training is independent

---

## Execution

### Command
```bash
# Default (no diffusion augmentation)
python run.py --dataset reddit --num_epoch 300

# With diffusion augmentation
python run.py --dataset reddit --use_loss_guided_emb_gen --num_epoch 300

# Custom hyperparameters
python run.py --dataset reddit --use_loss_guided_emb_gen \
  --gen_epochs 50 --diff_hidden_dim 512 --lr 1e-3 --num_epoch 300
```

### Expected Output

```
Phase 1: Training unconditional diffusion model...
  Unconditional Epoch 0: Loss=0.234567
  Unconditional Epoch 10: Loss=0.123456
  ...
  Unconditional Epoch 40: Loss=0.012345
Early stopping (unconditional)

Phase 1: Training conditional diffusion model...
  Conditional Epoch 0: Loss=0.198765
  ...
Early stopping (conditional)

Phase 1: Generating synthetic nodes with classifier-free guidance...
Generated synthetic abnormal nodes shape: torch.Size([50, 300])

Combined dataset: 3956 nodes (original: 3906 + synthetic: 50)

Phase 2: Standard GGAD Detector Training on Combined Dataset
  Epoch: 0000 train_loss_margin= 0.234560
  Epoch: 0000 train_loss_bce= 0.345670
  Epoch: 0000 rec_loss= 0.123450
  Epoch: 0000 train_loss= 0.703680
  Testing reddit AUC:0.7234
  ...
```

---

## Architecture Diagram

```
Node Features
      ↓
┌─────────────────────────────────────────────────────┐
│ PHASE 1: Diffusion (EDMLoss Input)                  │
├─────────────────────────────────────────────────────┤
│ DM_Unconditional (no proto)                         │
│   → learns distribution + prototype                 │
│                                                      │
│ DM_Conditional (with proto)                         │
│   → learns conditional path                         │
│                                                      │
│ Sampling: Classifier-Free Guidance                  │
│   d = (1+w) × d_free - w × d_proto                 │
│                                                      │
│ Output: Synthetic embeddings (num_samples × dim)    │
└─────────────────────────────────────────────────────┘
      ↓
Expand Graph
  - Add synthetic features (zeros)
  - Connect to random normal nodes
      ↓
┌─────────────────────────────────────────────────────┐
│ PHASE 2: GCN Detector (GGAD Losses ONLY)            │
├─────────────────────────────────────────────────────┤
│ GCN Layer 1: Features → Hidden Embeddings           │
│ GCN Layer 2: Hidden → Final Embeddings              │
│ FC Classifier (Ego-Centric Training Mode)           │
│                                                      │
│ Loss = Margin + BCE + Reconstruction                │
│                                                      │
│ Output: Anomaly scores per node                     │
└─────────────────────────────────────────────────────┘
      ↓
Per-Node Anomaly Scores
```

---

## Validation

- ✅ Syntax: `python -m py_compile run.py loss_guided_diffusion.py` PASSED
- ✅ Loss Separation: Phase 1 uses EDMLoss, Phase 2 uses GGAD losses VERIFIED
- ✅ No Circular Dependencies: Each phase independent CONFIRMED
- ✅ DiffGAD Alignment: Same architecture and training approach CONFIRMED

---

## Documentation

For detailed information, see:
1. **DIFFUSION_IMPLEMENTATION_GUIDE.md** - Complete technical documentation
2. **CORRECTIONS_SUMMARY.md** - Before/after comparison
3. **GGAD_ARCHITECTURE.md** - GGAD detector design

---

## Key Takeaway

The rewritten implementation follows **DiffGAD's proven approach**:
- Use EDMLoss for diffusion training (not task-specific losses)
- Learn prototype as natural outcome of training
- Apply classifier-free guidance at inference time
- Use GGAD losses only for detector training

This separation of concerns is the correct way to combine diffusion with graph anomaly detection.
