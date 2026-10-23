# -*- coding: utf-8 -*-
"""
Neural SDE — Conditional Stochastic Differential Equation Generator.

Improvement #8: replaces the hand-crafted impact-driven generator
with a learned SDE whose drift and diffusion functions are neural
networks conditioned on causal factor activations.

Model
-----
The latent state ``X_t ∈ ℝ^P`` (P = 4 for OHLC) evolves as::

    dX_t = μ_θ(X_t, F_t) dt + σ_φ(X_t, F_t) dW_t + J_t dN_t

where
    - ``μ_θ``  — drift network  (MLP conditioned on factors)
    - ``σ_φ``  — diffusion network (diagonal, positive-definite)
    - ``F_t``  — factor activations at time *t*
    - ``J_t``  — Merton-style Poisson jumps (analytical, not learned)
    - ``dW_t`` — standard Wiener process
    - ``dN_t`` — Poisson process (jump indicator)

Training
--------
The model is trained by minimising the **negative log-likelihood**
of observed daily log-returns under the Euler-Maruyama discretisation::

    X_{t+1} = X_t + μ_θ(X_t, F_t)·Δt + σ_φ(X_t, F_t)·√Δt · ε_t

    ℓ = -Σ_t log 𝒩(ΔX_t | μ_θ Δt, σ_φ² Δt)

Sampling
--------
After training, new trajectories are sampled via Euler-Maruyama
with the learned drift/diffusion, optionally injecting cross-
asset correlations (#9) and jumps (#7).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ══════════════════════════════════════════════════════════════════════
# Neural SDE Networks
# ══════════════════════════════════════════════════════════════════════


class DriftNet(nn.Module):
    """Drift function μ_θ(X_t, F_t) → ℝ^P.

    Inputs:  (batch, P + K) — concatenation of state and factors.
    Outputs: (batch, P)     — predicted drift per feature.
    """

    def __init__(self, state_dim: int, factor_dim: int, hidden: int = 64):
        super().__init__()
        inp = state_dim + factor_dim
        self.net = nn.Sequential(
            nn.Linear(inp, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, state_dim),
        )
        # Initialise near zero so initial drift ≈ 0
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DiffusionNet(nn.Module):
    """Diffusion function σ_φ(X_t, F_t) → ℝ^P  (diagonal, > 0).

    Uses softplus to guarantee positivity.
    """

    def __init__(self, state_dim: int, factor_dim: int, hidden: int = 64):
        super().__init__()
        inp = state_dim + factor_dim
        self.net = nn.Sequential(
            nn.Linear(inp, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, state_dim),
        )
        # Initialise so initial σ ≈ 0.02 (typical daily vol)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, math.log(math.expm1(0.02)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.net(x))


# ══════════════════════════════════════════════════════════════════════
# Neural SDE Model
# ══════════════════════════════════════════════════════════════════════


class NeuralSDEModel(nn.Module):
    """Wrapper that holds drift + diffusion networks.

    Parameters
    ----------
    state_dim : int
        Dimension of the price state (P, typically 4 for OHLC).
    factor_dim : int
        Number of causal factors (K).
    hidden : int
        Hidden layer width.
    """

    def __init__(self, state_dim: int, factor_dim: int, hidden: int = 64):
        super().__init__()
        self.state_dim = state_dim
        self.factor_dim = factor_dim
        self.drift = DriftNet(state_dim, factor_dim, hidden)
        self.diffusion = DiffusionNet(state_dim, factor_dim, hidden)

    def forward(
        self,
        x_t: torch.Tensor,
        f_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (drift, diffusion) at state x_t, factors f_t.

        Parameters
        ----------
        x_t : (B, P) — current log-price state
        f_t : (B, K) — current factor activations

        Returns
        -------
        mu : (B, P)    — drift
        sigma : (B, P) — diffusion (positive)
        """
        inp = torch.cat([x_t, f_t], dim=-1)
        mu = self.drift(inp)
        sigma = self.diffusion(inp)
        return mu, sigma


# ══════════════════════════════════════════════════════════════════════
# Transfer Learning  (F5)
# ══════════════════════════════════════════════════════════════════════


@torch.no_grad()
def _transfer_fin_to_sde(
    fin_model: nn.Module,
    sde_model: NeuralSDEModel,
    device: str | torch.device,
    scale: float = 0.5,
) -> None:
    """Transfer learned factor representations from FIN/TFT → SDE.

    Strategy
    --------
    1. Extract the factor embedding matrix  E ∈ ℝ^{K × D}  from
       the pre-trained FIN or TFT.
    2. Compute a random projection  P ∈ ℝ^{D × H}  from embedding
       space to hidden space and construct  W_factor = (E @ P)^T.
    3. If the source is a TFT, scale columns by VSN importance so
       that unimportant factors start near zero.
    4. Copy  W_factor  into the factor-related columns (cols P: )
       of `DriftNet.net[0].weight` and
       `DiffusionNet.net[0].weight`.
    """
    K = sde_model.factor_dim
    P = sde_model.state_dim
    H = sde_model.drift.net[0].out_features          # hidden width

    # 1. Extract factor embeddings
    if hasattr(fin_model, "factor_embedding"):
        embeddings = fin_model.factor_embedding.weight.detach().to(device)
    elif hasattr(fin_model, "factor_embeddings"):
        embeddings = fin_model.factor_embeddings.weight.detach().to(device)
    else:
        return  # no embeddings to transfer

    D = embeddings.shape[1]

    # 2. Random projection  D → H
    proj = torch.randn(D, H, device=device) * (scale / (D ** 0.5))
    factor_init = embeddings @ proj                               # (K, H)

    # 3. Scale by importance
    if hasattr(fin_model, "_last_vsn_weights") and fin_model._last_vsn_weights is not None:
        importance = fin_model._last_vsn_weights.mean(dim=(0, 1)).to(device)
    elif hasattr(fin_model, "factor_impact_weight"):
        importance = fin_model.factor_impact_weight.detach().to(device).abs().mean(dim=1)
        importance = importance / (importance.max() + 1e-8)
    else:
        importance = torch.ones(K, device=device)

    factor_init = factor_init * importance.unsqueeze(-1)

    # 4. Write into SDE networks  (factor columns = indices P:P+K)
    W_drift = sde_model.drift.net[0].weight.data                  # (H, P+K)
    W_drift[:, P : P + K] = factor_init.T * scale
    sde_model.drift.net[0].weight.data = W_drift

    W_diff = sde_model.diffusion.net[0].weight.data               # (H, P+K)
    W_diff[:, P : P + K] = factor_init.T * scale * 0.5
    sde_model.diffusion.net[0].weight.data = W_diff


# ══════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════


def train_neural_sde(
    log_returns: np.ndarray,
    factor_matrix: np.ndarray,
    n_price_features: int = 4,
    hidden: int = 64,
    n_epochs: int = 200,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    val_fraction: float = 0.15,
    patience: int = 30,
    device: str = "auto",
    pretrained_fin: Optional[nn.Module] = None,
    transfer_scale: float = 0.5,
) -> Tuple[NeuralSDEModel, Dict]:
    """Train the Neural SDE on real log-returns conditioned on factors.

    Parameters
    ----------
    log_returns : (T, M) — daily log-returns (OHLCV)
    factor_matrix : (T+1, K) — daily factor presence
    n_price_features : int — use first P columns (OHLC only)
    hidden : int
    n_epochs, batch_size, learning_rate : training hyperparams
    val_fraction : float
    patience : int — early stopping
    device : str — "auto" | "cpu" | "cuda"
    pretrained_fin : nn.Module or None — trained FIN/TFT for transfer
        learning (F5).  When provided, factor embeddings are used to
        warm-start the drift and diffusion networks.
    transfer_scale : float — scale for transferred weights (0–1).

    Returns
    -------
    model : NeuralSDEModel
    history : dict with 'train_loss', 'val_loss', 'best_epoch'
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    P = min(n_price_features, log_returns.shape[1])
    T = log_returns.shape[0]
    K = factor_matrix.shape[1]

    # Align factor matrix (T+1 dates → T returns)
    F_aligned = factor_matrix[:T].astype(np.float32)

    # Build log-price state: cumulative sum of log-returns
    X = np.cumsum(log_returns[:, :P], axis=0).astype(np.float32)
    # State at time t (for conditioning): X[t-1] for t≥1, zeros for t=0
    X_prev = np.vstack([np.zeros((1, P), dtype=np.float32), X[:-1]])

    # Target: log-returns (Δt = 1 day)
    Y = log_returns[:, :P].astype(np.float32)

    # Train / val split
    n_val = max(1, int(T * val_fraction))
    n_train = T - n_val

    Xt = torch.from_numpy(X_prev[:n_train]).to(device)
    Ft = torch.from_numpy(F_aligned[:n_train]).to(device)
    Yt = torch.from_numpy(Y[:n_train]).to(device)

    Xv = torch.from_numpy(X_prev[n_train:]).to(device)
    Fv = torch.from_numpy(F_aligned[n_train:]).to(device)
    Yv = torch.from_numpy(Y[n_train:]).to(device)

    # Model
    model = NeuralSDEModel(P, K, hidden).to(device)

    # ── F5: Transfer learning from pre-trained FIN/TFT ──────────────
    if pretrained_fin is not None:
        _transfer_fin_to_sde(pretrained_fin, model, device, transfer_scale)
        print(f"    ✓ Transfer learning applied (scale={transfer_scale})")

    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, patience=patience // 3, factor=0.5
    )

    dataset = TensorDataset(Xt, Ft, Yt)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        drop_last=False)

    history = {"train_loss": [], "val_loss": [], "best_epoch": 0}
    best_val = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(n_epochs):
        # ── Training ────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for xb, fb, yb in loader:
            mu, sigma = model(xb, fb)
            # Gaussian NLL: -log N(y | mu, sigma²)
            # = 0.5 * log(2π) + log(σ) + 0.5 * ((y-μ)/σ)²
            nll = (
                torch.log(sigma + 1e-8)
                + 0.5 * ((yb - mu) / (sigma + 1e-8)) ** 2
            ).mean()
            optimiser.zero_grad()
            nll.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            epoch_loss += nll.item()
            n_batches += 1

        train_loss = epoch_loss / max(n_batches, 1)
        history["train_loss"].append(train_loss)

        # ── Validation ──────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            mu_v, sigma_v = model(Xv, Fv)
            val_nll = (
                torch.log(sigma_v + 1e-8)
                + 0.5 * ((Yv - mu_v) / (sigma_v + 1e-8)) ** 2
            ).mean().item()

        history["val_loss"].append(val_nll)
        scheduler.step(val_nll)

        if val_nll < best_val:
            best_val = val_nll
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            history["best_epoch"] = epoch
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:4d}/{n_epochs}  "
                  f"train={train_loss:.4f}  val={val_nll:.4f}")

        if no_improve >= patience:
            print(f"    Early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    return model, history


# ══════════════════════════════════════════════════════════════════════
# Sampling / Generation
# ══════════════════════════════════════════════════════════════════════


@torch.no_grad()
def sample_neural_sde(
    model: NeuralSDEModel,
    n_steps: int,
    n_samples: int,
    initial_prices: np.ndarray,
    factor_schedule: np.ndarray,
    jump_lam: float = 0.0,
    jump_mu: float = 0.0,
    jump_sig: float = 0.0,
    corr_cholesky: Optional[np.ndarray] = None,
    device: str = "cpu",
) -> np.ndarray:
    """Sample trajectories from a trained Neural SDE.

    Parameters
    ----------
    model : NeuralSDEModel
    n_steps : int — trading days
    n_samples : int — parallel trajectories
    initial_prices : (M,) — OHLCV starting values
    factor_schedule : (n_samples, n_steps, K) — factor activations
    jump_lam, jump_mu, jump_sig : Merton jump params
    corr_cholesky : (P, P) or None — Cholesky factor for cross-feature
        correlation (#9). When provided, noise ε is replaced by L @ ε
        so that Cov(ε) = L L^T.
    device : str

    Returns
    -------
    prices : (n_samples, n_steps, M) — synthetic prices
    """
    model = model.to(device)
    P = model.state_dim
    M = len(initial_prices)
    K = model.factor_dim

    # Log-price state
    log_p = np.log(np.clip(initial_prices[:P], 1e-8, None))
    X = np.tile(log_p, (n_samples, 1))  # (S, P)

    prices = np.zeros((n_samples, n_steps, M))

    for t in range(n_steps):
        x_t = torch.from_numpy(X.astype(np.float32)).to(device)
        f_t = torch.from_numpy(
            factor_schedule[:, t, :].astype(np.float32)
        ).to(device)

        mu, sigma = model(x_t, f_t)
        mu_np = mu.cpu().numpy()       # (S, P)
        sig_np = sigma.cpu().numpy()   # (S, P)

        # Noise: optionally correlated across features
        eps = np.random.normal(0, 1, (n_samples, P))
        if corr_cholesky is not None:
            eps = eps @ corr_cholesky.T  # (S, P) @ (P, P)^T

        # Euler-Maruyama: dX = μ dt + σ √dt ε
        dX = mu_np + sig_np * eps  # dt = 1

        # Jump component
        if jump_lam > 0 and jump_sig > 0:
            jump_mask = np.random.binomial(1, jump_lam, n_samples)
            jump_size = np.random.normal(jump_mu, jump_sig, n_samples)
            for p in range(P):
                dX[:, p] += jump_mask * jump_size

        # Clamp
        dX = np.clip(dX, -0.20, 0.20)

        X = X + dX
        prices[:, t, :P] = np.exp(X)

    return prices


# ══════════════════════════════════════════════════════════════════════
# Integrated Generator (wraps ImpactDrivenGenerator + NeuralSDE)
# ══════════════════════════════════════════════════════════════════════


class NeuralSDEGenerator:
    """High-level generator that uses a trained Neural SDE.

    Combines:
        - Factor schedule from ImpactDrivenGenerator._build_schedule
        - Learned drift/diffusion from NeuralSDEModel
        - Merton jumps from generate.py helpers
        - Separate volume model
        - Cross-asset correlation injection
    """

    def __init__(
        self,
        model: NeuralSDEModel,
        impact_generator,                            # ImpactDrivenGenerator
        device: str = "cpu",
    ):
        self.model = model
        self.ig = impact_generator                   # for schedule + volume
        self.device = device

    def generate(
        self,
        n_steps: int,
        n_samples: int,
        initial_prices: np.ndarray,
        factor_schedule: np.ndarray,
        real_log_returns: Optional[np.ndarray] = None,
        real_volume: Optional[np.ndarray] = None,
        corr_cholesky: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Generate synthetic OHLCV prices.

        Parameters
        ----------
        n_steps, n_samples : int
        initial_prices : (M,) — OHLCV initial values
        factor_schedule : (S, T, K)
        real_log_returns : (T, M) or None — for jump estimation
        real_volume : (T,) or None — for volume model
        corr_cholesky : (P, P) or None — cross-asset cholesky

        Returns
        -------
        prices : (S, T, M)
        """
        from .generate import _estimate_jump_params

        P = self.model.state_dim
        M = len(initial_prices)

        # Jump params
        jump_lam = jump_mu = jump_sig = 0.0
        if real_log_returns is not None and len(real_log_returns) > 30:
            jump_lam, jump_mu, jump_sig = _estimate_jump_params(
                real_log_returns[:, :P]
            )

        # Sample trajectories
        prices = sample_neural_sde(
            model=self.model,
            n_steps=n_steps,
            n_samples=n_samples,
            initial_prices=initial_prices,
            factor_schedule=factor_schedule,
            jump_lam=jump_lam,
            jump_mu=jump_mu,
            jump_sig=jump_sig,
            corr_cholesky=corr_cholesky,
            device=self.device,
        )

        # Volume (separate log-normal model from ImpactDrivenGenerator)
        if M >= 5:
            log_ret = np.diff(
                np.log(np.clip(prices[:, :, :P], 1e-8, None)), axis=1
            )
            # Pad first step with zeros
            log_ret = np.concatenate(
                [np.zeros((n_samples, 1, P)), log_ret], axis=1
            )
            prices[:, :, 4] = self.ig._generate_volume(
                log_returns=log_ret,
                n_samples=n_samples,
                n_steps=n_steps,
                real_volume=real_volume,
                initial_volume=initial_prices[4] if len(initial_prices) > 4 else 1e8,
            )

        # Enforce OHLCV
        prices = self.ig._enforce_ohlcv(prices)
        return prices
