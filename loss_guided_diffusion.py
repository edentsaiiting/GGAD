"""
Diffusion Model for Synthetic Abnormal Node Generation (Following DiffGAD)

This module implements a diffusion model trained with EDMLoss (like DiffGAD).
Key differences from previous version:
1. Uses EDMLoss for training (weighted MSE reconstruction), not GGAD losses
2. GGAD losses (margin, BCE, reconstruction) are for detector training ONLY
3. Learns a prototype during training as weighted average
        # Classifier-guided gradient guidance used in sampling, not training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple


class SiLU(nn.Module):
    """Swish activation function"""
    def forward(self, x):
        return x * torch.sigmoid(x)


class PositionalEmbedding(nn.Module):
    """Positional embedding for noise level conditioning"""
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels // 2,
                             dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class MLPDiffusion(nn.Module):
    """Denoising network - identical to DiffGAD"""
    def __init__(self, d_in, dim_t=512):
        super().__init__()
        self.dim_t = dim_t
        self.proj = nn.Linear(d_in, dim_t)
        self.mlp = nn.Sequential(
            nn.Linear(dim_t, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, d_in),
        )
        self.map_noise = PositionalEmbedding(num_channels=dim_t)
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t)
        )

    def forward(self, x, noise_labels, proto=None, proto_alpha=None):
        """Forward pass for unconditional diffusion denoiser."""
        emb = self.map_noise(noise_labels)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)
        emb = self.time_embed(emb)
        x = self.proj(x) + emb
        return self.mlp(x)


class Precond(nn.Module):
    """Preconditioning for diffusion training - identical to DiffGAD"""
    def __init__(self, denoise_fn, hid_dim, sigma_min=0, sigma_max=float('inf'), sigma_data=0.5):
        super().__init__()
        self.hid_dim = hid_dim
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.denoise_fn = denoise_fn

    def forward(self, x, sigma, proto=None, proto_alpha=None):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1)

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        x_in = c_in * x
        F_x = self.denoise_fn(x_in.to(torch.float32), c_noise.flatten())
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


class EDMLoss(nn.Module):
    """
    Diffusion loss - identical to DiffGAD
    Uses weighted MSE reconstruction loss with preconditioning
    """
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        super().__init__()
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, denoise_fn, data, proto=None, proto_alpha=None):
        """
        Compute diffusion loss during training
        
        Args:
            denoise_fn: Preconditioned denoising function
            data: Input data (batch of embeddings)
            proto: Prototype vector (for conditional diffusion)
            proto_alpha: Weight for prototype conditioning
        
        Returns:
            loss: Scalar loss value
            score: Per-sample reconstruction error
            reconstructed: Denoised output
        """
        # Sample random noise levels
        rnd_normal = torch.randn(data.shape[0], device=data.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        
        # Compute loss weight based on noise level
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        
        # Add noise to data
        y = data
        n = torch.randn_like(y) * sigma.unsqueeze(1)
        D_yn = denoise_fn(y + n, sigma)
        
        # Reconstruction loss: MSE between denoised and original
        target = y
        loss = weight.unsqueeze(1) * ((D_yn - target) ** 2)
        
        # Compute per-sample reconstruction error
        reconstruction_errors = (D_yn - target) ** 2
        score = torch.sqrt(torch.sum(reconstruction_errors, 1))
        
        return loss, score, D_yn


class DiffusionGenerator(nn.Module):
    """
    Diffusion model wrapper combining Precond and EDMLoss
    """
    def __init__(self, denoise_fn, hid_dim, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        super().__init__()
        self.denoise_fn_D = Precond(denoise_fn, hid_dim, sigma_data=sigma_data)
        self.loss_fn = EDMLoss(P_mean, P_std, sigma_data=sigma_data)

    def forward(self, x, proto=None, proto_alpha=None):
        """
        Compute loss and denoising
        
        Returns:
            loss: Scalar loss
            score: Per-sample error
            reconstructed: Denoised embedding
        """
        loss, score, reconstructed = self.loss_fn(self.denoise_fn_D, x)
        return loss.mean(-1).mean(), score, reconstructed


# ============================================================================
# Sampling utilities (for inference/generation)
# ============================================================================

SIGMA_MIN = 0.002
SIGMA_MAX = 80
rho = 7
S_churn = 1
S_min = 0
S_max = float('inf')
S_noise = 1


def sample_step(net, num_steps, i, t_cur, t_next, x_next, proto=None, proto_alpha=None):
    """Single sampling step (unconditional)"""
    x_cur = x_next
    gamma = min(S_churn / num_steps, math.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
    t_hat = net.round_sigma(t_cur + gamma * t_cur)
    x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)

    denoised = net(x_hat, t_hat, proto, proto_alpha).to(torch.float32)
    d_cur = (x_hat - denoised) / t_hat
    x_next = x_hat + (t_next - t_hat) * d_cur

    if i < num_steps - 1:
        denoised = net(x_next, t_next, proto, proto_alpha).to(torch.float32)
        d_prime = (x_next - denoised) / t_next
        x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next


def sample_dm(net, noise, num_steps, proto=None, proto_alpha=None, z_init=None):
    """Reverse diffusion sampling (unconditional or conditional).

    z_init: if provided, use as the starting latent (data + sigma_max noise) instead of
            scaling pure noise by t_steps[0]. Allows seeded generation from existing features.
    """
    device = z_init.device if z_init is not None else noise.device
    step_indices = torch.arange(num_steps, dtype=torch.float32, device=device)

    sigma_min = max(SIGMA_MIN, net.sigma_min)
    sigma_max = min(SIGMA_MAX, net.sigma_max)

    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
        sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    z = z_init.to(torch.float32) if z_init is not None else noise.to(torch.float32) * t_steps[0]
    with torch.no_grad():
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            z = sample_step(net, num_steps, i, t_cur, t_next, z, proto, proto_alpha)

    return z


def sample_step_guided(net, num_steps, i, t_cur, t_next, x_next, proto=None, proto_alpha=None,
                       guidance_fn=None, guidance_scale=1.0, normal_embs=None):
    """Single sampling step with gradient guidance.

    normal_embs: [n_normal, feat_dim] clean normal embeddings. When provided, they are
    noised at the same sigma t_hat so that the guidance function receives both abnormal
    and normal representations at an identical noise level — making the affinity margin
    a fair comparison between equally-corrupted normal and abnormal features.
    """
    x_cur = x_next
    gamma = min(S_churn / num_steps, math.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
    t_hat = net.round_sigma(t_cur + gamma * t_cur)
    x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)

    denoised = net(x_hat, t_hat, proto, proto_alpha).to(torch.float32)
    d_cur = (x_hat - denoised) / t_hat

    if guidance_fn is not None and guidance_scale != 0:
        x_hat_req = x_hat.detach().requires_grad_(True)
        # Noise normal embeddings at the same sigma so affinity comparison is at equal corruption
        noisy_normals = None
        if normal_embs is not None:
            noisy_normals = normal_embs + torch.randn_like(normal_embs) * t_hat
        guidance_loss = guidance_fn(x_hat_req, noisy_normals, t_hat)
        if guidance_loss is not None:
            guidance_grad = torch.autograd.grad(guidance_loss.sum(), x_hat_req)[0]
            d_cur = d_cur + guidance_scale * guidance_grad

    x_next = x_hat + (t_next - t_hat) * d_cur

    if i < num_steps - 1:
        denoised = net(x_next, t_next, proto, proto_alpha).to(torch.float32)
        d_prime = (x_next - denoised) / t_next

        # Re-evaluate guidance at the predicted x_next so the Heun correction
        # carries the full guidance signal (not just the unguided score).
        if guidance_fn is not None and guidance_scale != 0:
            x_next_req = x_next.detach().requires_grad_(True)
            noisy_normals_prime = None
            if normal_embs is not None:
                noisy_normals_prime = normal_embs + torch.randn_like(normal_embs) * t_next
            guidance_loss_prime = guidance_fn(x_next_req, noisy_normals_prime, t_next)
            if guidance_loss_prime is not None:
                guidance_grad_prime = torch.autograd.grad(guidance_loss_prime.sum(), x_next_req)[0]
                d_prime = d_prime + guidance_scale * guidance_grad_prime

        x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next


def sample_dm_guided(net, noise, num_steps, proto=None, proto_alpha=None, guidance_fn=None,
                     guidance_scale=1.0, normal_embs=None, z_init=None):
    """Reverse diffusion sampling with gradient guidance.

    z_init: if provided, use as the starting latent (seed features + sigma_max noise)
            instead of scaling pure noise. Enables seeded guided generation (Phase 1b).
    """
    device = z_init.device if z_init is not None else noise.device
    step_indices = torch.arange(num_steps, dtype=torch.float32, device=device)

    sigma_min = max(SIGMA_MIN, net.sigma_min)
    sigma_max = min(SIGMA_MAX, net.sigma_max)

    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
        sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    z = z_init.to(torch.float32) if z_init is not None else noise.to(torch.float32) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        z = sample_step_guided(net, num_steps, i, t_cur, t_next, z, proto, proto_alpha,
                               guidance_fn, guidance_scale, normal_embs)

    return z


def sample_step_free(proto_net, free_net, num_steps, i, t_cur, t_next, x_next, 
                     proto=None, proto_alpha=None, weight=None):
    """Single sampling step with classifier-free guidance"""
    x_cur = x_next
    gamma = min(S_churn / num_steps, math.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
    t_hat = proto_net.round_sigma(t_cur + gamma * t_cur)
    x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)

    # Conditional and unconditional denoising
    denoised_proto = proto_net(x_hat, t_hat, proto=proto, proto_alpha=proto_alpha).to(torch.float32)
    denoised_free = free_net(x_hat, t_hat).to(torch.float32)

    # Classifier-free guidance blend
    d_cur_proto = (x_hat - denoised_proto) / t_hat
    d_cur_free = (x_hat - denoised_free) / t_hat
    d_cur = (1 + weight) * d_cur_free - weight * d_cur_proto
    x_next = x_hat + (t_next - t_hat) * d_cur

    if i < num_steps - 1:
        denoised_proto = proto_net(x_next, t_next, proto=proto, proto_alpha=proto_alpha).to(torch.float32)
        denoised_free = free_net(x_next, t_next).to(torch.float32)
        d_prime_proto = (x_next - denoised_proto) / t_next
        d_prime_free = (x_next - denoised_free) / t_next
        d_prime = (1.0 + weight) * d_prime_free - weight * d_prime_proto
        x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next


def sample_dm_free(proto_net, free_net, noise, num_steps, proto=None, proto_alpha=None, weight=None):
    """Reverse diffusion sampling with classifier-free guidance"""
    step_indices = torch.arange(num_steps, dtype=torch.float32, device=noise.device)

    sigma_min = max(SIGMA_MIN, free_net.sigma_min)
    sigma_max = min(SIGMA_MAX, free_net.sigma_max)

    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
        sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([free_net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    z = noise.to(torch.float32) * t_steps[0]
    with torch.no_grad():
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            z = sample_step_free(proto_net, free_net, num_steps, i, t_cur, t_next, z, 
                                proto, proto_alpha=proto_alpha, weight=weight)

    return z


# ============================================================================
# High-level interface for anomaly generation
# ============================================================================

def softmax_with_temperature(x, t=1.0):
    """Softmax with temperature"""
    x = x / t
    return F.softmax(x, dim=0)


class AnomalyGenerator:
    """
    High-level interface for generating synthetic abnormal nodes using diffusion.
    
    Training follows diffusion reconstruction training (EDMLoss) and uses external
    classifier guidance during sampling to minimize GGAD loss.
    1. Uses EDMLoss (weighted MSE) for training
    2. Uses classifier gradient guidance during sampling (no prototype-conditioned guidance)
    3. GGAD losses used only for classifier guidance
    """
    
    def __init__(self, embedding_dim=300, hidden_dim=512, lr=0.001, 
                 num_gen_epochs=50, proto_alpha=0.5, device='cpu'):
        """
        Args:
            embedding_dim: Dimension of node embeddings
            hidden_dim: Hidden dimension for denoising network
            lr: Learning rate
            num_gen_epochs: Number of training epochs
            proto_alpha: Weight for prototype conditioning (in classifier-free guidance)
            device: 'cpu' or 'cuda'
        """
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.num_gen_epochs = num_gen_epochs
        self.proto_alpha = proto_alpha
        self.device = device
        
        # Models
        self.denoise_fn_unconditional = None
        self.denoise_fn_conditional = None
        self.dm_unconditional = None
        self.dm_conditional = None
        
        # Optimizers
        self.optimizer_unconditional = None
        self.optimizer_conditional = None
        self.scheduler_unconditional = None
        self.scheduler_conditional = None
        
        # Prototype (learned during training)
        self.proto = None
        
        # Generated results
        self.generated_nodes = None
        
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def initialize(self):
        """Initialize both unconditional and conditional diffusion models"""
        # Unconditional model
        self.denoise_fn_unconditional = MLPDiffusion(self.embedding_dim, dim_t=self.hidden_dim)
        self.dm_unconditional = DiffusionGenerator(
            self.denoise_fn_unconditional, 
            self.embedding_dim
        )
        self.denoise_fn_unconditional = self.denoise_fn_unconditional.to(self.device)
        self.dm_unconditional = self.dm_unconditional.to(self.device)
        
        self.optimizer_unconditional = torch.optim.Adam(
            self.dm_unconditional.parameters(),
            lr=self.lr,
            weight_decay=0.0
        )
        self.scheduler_unconditional = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_unconditional, T_max=self.num_gen_epochs, eta_min=1e-6
        )
        
        # Conditional model (prototype-guided)
        # In classifier-guided setting, we only need unconditional diffusion training.
        self.denoise_fn_conditional = None
        self.dm_conditional = None
        self.optimizer_conditional = None
        self.scheduler_conditional = None

    def train_unconditional(self, node_embeddings):
        """
        Train unconditional diffusion model on node embeddings
        
        Args:
            node_embeddings: Input embeddings (num_nodes, embedding_dim)
        
        Returns:
            prototype: Learned prototype vector
        """
        self.dm_unconditional.train()

        n_samples = node_embeddings.shape[0]
        batch_size = min(256, n_samples)
        print(f"Training unconditional diffusion model ({n_samples} samples, batch={batch_size})...")
        for epoch in range(self.num_gen_epochs):
            # Shuffle and iterate mini-batches so each epoch sees multiple σ draws
            perm = torch.randperm(n_samples, device=node_embeddings.device)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n_samples, batch_size):
                batch = node_embeddings[perm[start:start + batch_size]]
                self.optimizer_unconditional.zero_grad()
                loss, score_train, reconstructed = self.dm_unconditional(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.dm_unconditional.parameters(), 1.0)
                self.optimizer_unconditional.step()
                epoch_loss += loss.item()
                n_batches += 1
            self.scheduler_unconditional.step()

            if epoch % 10 == 0:
                print(f"  Unconditional Epoch {epoch}: Loss={epoch_loss / n_batches:.6f}")

        self.proto = None
        return None

    def train_conditional(self, node_embeddings):
        """No-op placeholder. We do not use conditional prototype training in classifier-guided setup."""
        return

    def generate(self, num_samples, num_steps=50, guidance_fn=None, guidance_scale=0.0,
                 normal_embs=None, seed_embeddings=None):
        """
        Generate synthetic abnormal nodes using gradient guidance (Phase 1b).

        Args:
            num_samples: Number of synthetic nodes (used only when seed_embeddings is None).
            num_steps: Number of diffusion steps for sampling.
            guidance_fn: Callback (noisy_abnormals, noisy_normals, sigma) -> scalar loss.
            guidance_scale: Scale for the gradient guidance term.
            normal_embs: [n_normal, feat_dim] clean normal embeddings for fair guidance comparison.
            seed_embeddings: [n_seeds, feat_dim] seed features (real_abn + 15% normals).
                When provided, the reverse ODE starts from seed_embeddings + sigma_max noise
                instead of pure noise, grounding generation in the seed distribution (Phase 1b).

        Returns:
            generated_nodes: Synthetic embeddings [n_seeds or num_samples, embedding_dim]
        """
        self.dm_unconditional.eval()
        guided_net = self.dm_unconditional.denoise_fn_D

        if seed_embeddings is not None:
            # Phase 1b: start from seed features + sigma_max noise
            seed_embeddings = seed_embeddings.to(self.device).to(torch.float32)
            step_indices = torch.arange(num_steps, dtype=torch.float32, device=self.device)
            sigma_min_eff = max(SIGMA_MIN, guided_net.sigma_min)
            sigma_max_eff = min(SIGMA_MAX, guided_net.sigma_max)
            t_steps_init = (sigma_max_eff ** (1 / rho) + step_indices / (num_steps - 1) * (
                sigma_min_eff ** (1 / rho) - sigma_max_eff ** (1 / rho))) ** rho
            t_steps_init = torch.cat([guided_net.round_sigma(t_steps_init),
                                      torch.zeros_like(t_steps_init[:1])])
            sigma_max_val = t_steps_init[0]
            z_init = seed_embeddings + torch.randn_like(seed_embeddings) * sigma_max_val
            noise = None
        else:
            z_init = None
            noise = torch.randn(num_samples, self.embedding_dim, device=self.device)

        if guidance_fn is not None and guidance_scale != 0.0:
            generated = sample_dm_guided(
                guided_net, noise, num_steps,
                proto=None, proto_alpha=None,
                guidance_fn=guidance_fn,
                guidance_scale=guidance_scale,
                normal_embs=normal_embs,
                z_init=z_init,
            )
        else:
            generated = sample_dm(
                guided_net, noise, num_steps,
                proto=None, proto_alpha=None,
                z_init=z_init,
            )

        self.generated_nodes = generated.detach()
        return self.generated_nodes

    def transform_all(self, node_embeddings, num_steps=50):
        """
        Project all node embeddings through the Phase 1a trained denoiser (unguided).
        Used at inference to create a consistent feature space with Phase 2 training.

        Args:
            node_embeddings: [N, feat_dim] raw node features (all N nodes).
            num_steps: Number of reverse ODE steps.

        Returns:
            transformed: [N, feat_dim] projected embeddings.
        """
        self.dm_unconditional.eval()
        net = self.dm_unconditional.denoise_fn_D

        node_embeddings = node_embeddings.to(self.device).to(torch.float32)
        step_indices = torch.arange(num_steps, dtype=torch.float32, device=self.device)
        sigma_min_eff = max(SIGMA_MIN, net.sigma_min)
        sigma_max_eff = min(SIGMA_MAX, net.sigma_max)
        t_steps = (sigma_max_eff ** (1 / rho) + step_indices / (num_steps - 1) * (
            sigma_min_eff ** (1 / rho) - sigma_max_eff ** (1 / rho))) ** rho
        t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])
        sigma_max_val = t_steps[0]

        z_init = node_embeddings + torch.randn_like(node_embeddings) * sigma_max_val
        transformed = sample_dm(net, None, num_steps, z_init=z_init)
        return transformed.detach()

    def get_generated_nodes(self):
        """Return generated nodes"""
        return self.generated_nodes
