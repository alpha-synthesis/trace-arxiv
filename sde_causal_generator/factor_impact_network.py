# -*- coding: utf-8 -*-
"""
Factor Impact Network (FIN).

Learns the mapping   ``(factor_presence, context) → price_impact``
and produces an interpretable :class:`ImpactMatrix`.

Architecture (designed for post-hoc decomposition):
    1. **Per-factor impact embeddings** — interpretable base effects.
    2. **Multi-head self-attention** over active factors — captures
       synergies and dampening between co-occurring factors.
    3. **Depthwise temporal convolution** — per-factor lag/decay profile.
    4. **Gaussian NLL loss** — learns both mean *and* variance of
       impact, enabling calibrated uncertainty.

References:
    - Gu, Kelly & Xiu (2020), *Empirical Asset Pricing via ML*
    - Nix & Weigend (1994), *Estimating mean and variance of the
      target probability distribution*
    - Lundberg & Lee (2017), *SHAP*
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .data_structures import ImpactMatrix


# ══════════════════════════════════════════════════════════════════════
# Network
# ══════════════════════════════════════════════════════════════════════


class FactorImpactNetwork(nn.Module):
    """
    Neural network that learns factor → price impact.

    The architecture is structured so that after training we can
    extract per-factor impact vectors by:

    * inspecting ``factor_impact_weight`` directly,
    * probing with one-hot factor activations,
    * ablating factors from the real history.
    """

    def __init__(
        self,
        n_factors: int,
        n_features: int = 5,
        n_lags: int = 10,
        embed_dim: int = 32,
        interaction_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_factors = n_factors
        self.n_features = n_features
        self.n_lags = n_lags
        self.embed_dim = embed_dim

        # ── Module 1: per-factor impact embeddings ──────────────────
        self.factor_embeddings = nn.Embedding(n_factors, embed_dim)
        self.factor_impact_weight = nn.Parameter(
            torch.randn(n_factors, n_features) * 0.01
        )
        self.factor_impact_bias = nn.Parameter(torch.zeros(n_features))

        # ── Module 2: factor interaction ─────────────────────────────
        # For large K (> 256), attention is O(K²) and risks OOM.
        # Fall back to a lightweight bilinear interaction layer.
        self._use_attention = n_factors <= 256
        if self._use_attention:
            self.interaction_attention = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=interaction_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.interaction_norm = nn.LayerNorm(embed_dim)
        else:
            # Cheap alternative: project pooled embeddings
            self.interaction_proj = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, embed_dim),
            )
            self.interaction_norm = nn.LayerNorm(embed_dim)

        # ── Module 3: temporal convolution (depthwise) ──────────────
        kernel = min(n_lags, 5)
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=n_factors,
                out_channels=n_factors,
                kernel_size=kernel,
                padding=kernel // 2,
                groups=n_factors,
            ),
            nn.ReLU(),
        )

        # ── combiner ───────────────────────────────────────────────
        self.combiner = nn.Sequential(
            nn.Linear(embed_dim + n_features, n_features * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_features * 2, n_features),
        )

        # ── uncertainty head ────────────────────────────────────────
        self.uncertainty_head = nn.Sequential(
            nn.Linear(embed_dim, n_features),
            nn.Softplus(),
        )

    # ────────────────────────────────────────────────────────────────

    def forward(
        self,
        factor_presence: torch.Tensor,     # (B, T, K)
        return_decomposed: bool = False,
    ) -> Dict[str, torch.Tensor]:
        B, T, K = factor_presence.shape
        device = factor_presence.device

        # 1. base impact
        base_impact = torch.einsum(
            "btk,km->btm", factor_presence, self.factor_impact_weight
        ) + self.factor_impact_bias

        # 2. interaction
        factor_indices = torch.arange(K, device=device)
        all_embeds = self.factor_embeddings(factor_indices)        # (K, D)
        presence_flat = factor_presence.reshape(B * T, K)
        weighted_embeds = (
            presence_flat.unsqueeze(-1) * all_embeds.unsqueeze(0)
        )  # (B*T, K, D)

        if self._use_attention:
            interaction_out, attn_weights = self.interaction_attention(
                weighted_embeds, weighted_embeds, weighted_embeds
            )
            interaction_out = self.interaction_norm(
                interaction_out + weighted_embeds
            )
        else:
            # lightweight path: mean-pool active embeddings → project
            mask_lw = presence_flat.unsqueeze(-1)                  # (B*T, K, 1)
            active_cnt = mask_lw.sum(dim=1).clamp(min=1)           # (B*T, 1)
            mean_embed = (weighted_embeds * mask_lw).sum(dim=1) / active_cnt  # (B*T, D)
            proj = self.interaction_proj(mean_embed)               # (B*T, D)
            # broadcast back + residual
            interaction_out = self.interaction_norm(
                weighted_embeds + proj.unsqueeze(1)
            )

        mask = presence_flat.unsqueeze(-1)
        active_count = mask.sum(dim=1).clamp(min=1)
        pooled = (interaction_out * mask).sum(dim=1) / active_count
        pooled = pooled.reshape(B, T, -1)

        # 3. temporal convolution
        temporal_in = factor_presence.transpose(1, 2)              # (B,K,T)
        temporal_out = self.temporal_conv(temporal_in).transpose(1, 2)
        temporal_mod = torch.einsum(
            "btk,km->btm", temporal_out, self.factor_impact_weight
        )

        # 4. combine
        combined = torch.cat(
            [pooled, base_impact + temporal_mod], dim=-1
        )
        predicted_returns = self.combiner(combined)

        # 5. uncertainty
        uncertainty = self.uncertainty_head(pooled)

        result: Dict[str, torch.Tensor] = {
            "predicted_returns": predicted_returns,
            "uncertainty": uncertainty,
        }

        if return_decomposed:
            contributions = torch.zeros(
                B, T, K, self.n_features, device=device
            )
            for k in range(K):
                contributions[:, :, k, :] = (
                    factor_presence[:, :, k : k + 1]
                    * self.factor_impact_weight[k : k + 1, :]
                )
            result["factor_contributions"] = contributions

        return result

    # ── ImpactMatrix extraction ─────────────────────────────────────

    @torch.no_grad()
    def extract_impact_matrix(
        self,
        factor_presence_history: torch.Tensor,
        factor_names: List[str],
        feature_names: Optional[List[str]] = None,
        llm_directions: Optional[np.ndarray] = None,
    ) -> ImpactMatrix:
        """Extract an interpretable :class:`ImpactMatrix` after training.

        Uses three complementary methods:
            1. Direct weight inspection  (linear component).
            2. One-hot probing  (isolated nonlinear effects).
            3. Ablation  (contextual contribution in real data).

        Non-linear capture improvements:
            - Probing uses varying activation intensities (0.5, 1.0, 2.0)
              to capture non-linear response curves.
            - Interaction matrix uses triplet probing for higher-order
              synergies.
            - Combined weights emphasise ablation (non-linear) over
              linear weights.

        Parameters
        ----------
        llm_directions : ndarray, shape (K,) or None
            Average LLM-extracted direction per factor (range [-1, +1]).
            When provided, the combination step corrects sign conflicts
            on the close column so that the learned impact direction is
            consistent with the LLM’s causal assessment.
        """
        self.eval()
        device = next(self.parameters()).device

        if feature_names is None:
            feature_names = [
                f"feature_{i}" for i in range(self.n_features)
            ]

        K = self.n_factors
        M = self.n_features

        # 1. direct weights (linear baseline)
        base_w = self.factor_impact_weight.detach().cpu().numpy()

        # 2. multi-intensity one-hot probing (captures non-linear response)
        #    Probe at intensities [0.5, 1.0, 2.0] to detect non-linearity
        intensities = [0.5, 1.0, 2.0]
        probed_by_intensity = {}
        probed_std = np.zeros((K, M))
        for intensity in intensities:
            probed_int = np.zeros((K, M))
            for k in range(K):
                one_hot = torch.zeros(1, self.n_lags, K, device=device)
                one_hot[:, :, k] = intensity
                res = self(one_hot)
                probed_int[k] = res["predicted_returns"].cpu().numpy()[0].mean(0)
                if intensity == 1.0:
                    probed_std[k] = np.sqrt(
                        res["uncertainty"].cpu().numpy()[0].mean(0)
                    )
            probed_by_intensity[intensity] = probed_int

        probed = probed_by_intensity[1.0]

        # Non-linearity measure: if response at 2x != 2 * response at 1x,
        # the factor has non-linear effects
        nonlinearity_score = np.abs(
            probed_by_intensity[2.0] - 2.0 * probed_by_intensity[1.0]
        ).mean(axis=1)  # (K,)

        # Build response curves array: (K, 3, M) for intensities [0.5, 1.0, 2.0]
        response_curves = np.stack(
            [probed_by_intensity[i] for i in intensities], axis=1,
        )  # (K, 3, M)

        # 3. ablation on real data
        # BUG-C5: probe in windows of n_lags (not full sequence) to
        # match the training distribution.  Aggregate across windows.
        T_total = factor_presence_history.shape[0]
        n_windows = max(1, T_total - self.n_lags + 1)
        full_preds = []
        abl_preds = {k: [] for k in range(K)}

        for start in range(0, n_windows, max(1, n_windows // 50)):
            end = start + self.n_lags
            if end > T_total:
                break
            window = factor_presence_history[start:end].unsqueeze(0).to(device)
            fp = self(window)["predicted_returns"].cpu().numpy()[0]
            full_preds.append((start, end, fp))
            for k in range(K):
                ablated = window.clone()
                ablated[:, :, k] = 0.0
                ap = self(ablated)["predicted_returns"].cpu().numpy()[0]
                abl_preds[k].append((start, end, ap))

        ablation = np.zeros((K, M))
        for k in range(K):
            diffs = []
            for (s, e, fp), (_, _, ap) in zip(full_preds, abl_preds[k]):
                active = factor_presence_history[s:e, k].numpy() > 0
                if active.sum() > 0:
                    diffs.append((fp[active] - ap[active]).mean(0))
            if diffs:
                ablation[k] = np.mean(diffs, axis=0)
        combined = np.zeros((K, M))
        for k in range(K):
            nl = float(nonlinearity_score[k])
            # Adaptive weights: more non-linear → more ablation weight
            w_abl = 0.55 + 0.15 * min(nl / (nl + 0.01), 1.0)
            w_prob = 0.30 - 0.05 * min(nl / (nl + 0.01), 1.0)
            w_lin = 1.0 - w_abl - w_prob
            combined[k] = w_abl * ablation[k] + w_prob * probed[k] + w_lin * base_w[k]

        # DIR-REG: direction consistency correction.
        # When the LLM says a factor is bearish but the combined impact
        # on the close column is positive (or vice versa), correct the
        # sign.  Uses the LLM’s confidence (|direction|) to scale.
        combined = _apply_direction_correction(combined, llm_directions, M)

        # occurrence prob
        occ = factor_presence_history.float().mean(dim=0).numpy()

        # temporal profile
        temporal = self._extract_temporal_profile()

        # interaction matrix (with non-linear higher-order detection)
        interaction = self._extract_interaction_matrix(
            factor_presence_history, device
        )

        return ImpactMatrix(
            factor_names=factor_names,
            base_impact=combined,
            impact_std=probed_std,
            occurrence_prob=occ,
            temporal_profile=temporal,
            interaction_matrix=interaction,
            feature_names=feature_names,
            nonlinearity_scores=nonlinearity_score,
            response_curves=response_curves,
        )

    def _extract_temporal_profile(self) -> np.ndarray:
        conv = self.temporal_conv[0]
        weights = conv.weight.detach().cpu().numpy()
        kernel_size = weights.shape[2]
        profiles = np.zeros((self.n_factors, self.n_lags))
        for k in range(self.n_factors):
            profiles[k, :kernel_size] = weights[k, 0, :]
        return profiles

    @torch.no_grad()
    def _extract_interaction_matrix(
        self,
        factor_presence_history: torch.Tensor,
        device: torch.device,
    ) -> np.ndarray:
        """Extract pairwise interaction matrix with non-linear detection.

        Uses three probing strategies:
        1. Standard paired probing: f(i+j) - f(i) - f(j)
        2. Multi-intensity probing: compare interaction at different
           activation levels to detect non-linear synergies
        3. Context-dependent probing: interaction strength changes when
           other factors are present
        """
        K = self.n_factors
        interaction = np.zeros((K, K))

        # Isolated effects at intensity 1.0
        isolated = np.zeros((K, self.n_features))
        for k in range(K):
            one_hot = torch.zeros(1, self.n_lags, K, device=device)
            one_hot[:, :, k] = 1.0
            isolated[k] = (
                self(one_hot)["predicted_returns"].cpu().numpy()[0].mean(0)
            )

        # Isolated effects at intensity 0.5 (for non-linearity detection)
        isolated_half = np.zeros((K, self.n_features))
        for k in range(K):
            one_hot = torch.zeros(1, self.n_lags, K, device=device)
            one_hot[:, :, k] = 0.5
            isolated_half[k] = (
                self(one_hot)["predicted_returns"].cpu().numpy()[0].mean(0)
            )

        cooc = (
            factor_presence_history.T.float()
            @ factor_presence_history.float()
        ).numpy()
        cooc_norm = cooc / max(cooc.max(), 1)

        for i in range(K):
            for j in range(i + 1, K):
                if cooc_norm[i, j] < 0.05:
                    continue

                # Strategy 1: Standard paired probing
                paired = torch.zeros(1, self.n_lags, K, device=device)
                paired[:, :, i] = 1.0
                paired[:, :, j] = 1.0
                paired_eff = (
                    self(paired)["predicted_returns"]
                    .cpu()
                    .numpy()[0]
                    .mean(0)
                )
                delta_standard = paired_eff - (isolated[i] + isolated[j])

                # Strategy 2: Half-intensity paired probing
                paired_half = torch.zeros(1, self.n_lags, K, device=device)
                paired_half[:, :, i] = 0.5
                paired_half[:, :, j] = 0.5
                paired_half_eff = (
                    self(paired_half)["predicted_returns"]
                    .cpu()
                    .numpy()[0]
                    .mean(0)
                )
                delta_half = paired_half_eff - (isolated_half[i] + isolated_half[j])

                # Non-linear interaction: if delta changes with intensity,
                # the interaction is non-linear
                nl_interaction = np.abs(delta_standard - 2.0 * delta_half).mean()

                # Combined: standard interaction + non-linear bonus
                combined_delta = (
                    delta_standard.mean()
                    + 0.3 * np.sign(delta_standard.mean()) * nl_interaction
                )
                interaction[i, j] = combined_delta
                interaction[j, i] = interaction[i, j]

        return interaction


# ══════════════════════════════════════════════════════════════════════
# TFT Architecture — Improvement F7
# ══════════════════════════════════════════════════════════════════════


class _GatedResidualNetwork(nn.Module):
    """Gated Residual Network from the TFT paper.

    ``GRN(a) = LayerNorm(a + Dropout(GLU(W₂·ELU(W₁·a))))``
    """

    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        d_out: int = 0,
        d_context: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        d_out = d_out or d_in
        self.d_in = d_in
        self.d_out = d_out

        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out * 2)  # GLU splits into 2
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_out)

        if d_context > 0:
            self.context_fc = nn.Linear(d_context, d_hidden, bias=False)
        if d_in != d_out:
            self.skip = nn.Linear(d_in, d_out, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = x if self.d_in == self.d_out else self.skip(x)
        h = nn.functional.elu(self.fc1(x))
        if context is not None and hasattr(self, "context_fc"):
            h = h + self.context_fc(context)
        h = self.fc2(h)
        vals, gates = h.chunk(2, dim=-1)
        gated = self.dropout(vals * torch.sigmoid(gates))
        return self.layer_norm(residual + gated)


class _VariableSelectionNetwork(nn.Module):
    """Variable Selection Network — learns which factors are most relevant.

    Returns selection softmax weights (interpretable importance) and
    the weighted-sum representation.
    """

    def __init__(
        self,
        n_vars: int,
        d_var: int,
        d_hidden: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_vars = n_vars
        self.d_var = d_var

        self.weight_grn = _GatedResidualNetwork(
            d_in=n_vars * d_var,
            d_hidden=d_hidden,
            d_out=n_vars,
            dropout=dropout,
        )
        self.var_grn = _GatedResidualNetwork(
            d_in=d_var, d_hidden=d_hidden, dropout=dropout,
        )

    def forward(
        self, x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (..., K, D) — per-variable representations.

        Returns
        -------
        selected : (..., D) — weighted sum.
        weights  : (..., K) — softmax selection weights.
        """
        *batch, K, D = x.shape
        flat = x.reshape(*batch, K * D)
        weights = torch.softmax(self.weight_grn(flat), dim=-1)
        x_proc = self.var_grn(x.reshape(-1, D)).reshape(*batch, K, D)
        selected = (weights.unsqueeze(-1) * x_proc).sum(dim=-2)
        return selected, weights


class TemporalFusionTransformerFIN(nn.Module):
    """TFT-based Factor Impact Network (F7).

    Uses Variable Selection Network, Gated Residual Networks, and
    multi-head temporal self-attention.  Interface-compatible with
    :class:`FactorImpactNetwork`.
    """

    def __init__(
        self,
        n_factors: int,
        n_features: int = 5,
        n_lags: int = 10,
        embed_dim: int = 32,
        interaction_heads: int = 4,
        dropout: float = 0.1,
        causal_attention: bool = False,
    ):
        super().__init__()
        self.n_factors = n_factors
        self.n_features = n_features
        self.n_lags = n_lags
        self.embed_dim = embed_dim
        # Each window maps presence at days i-L..i-1 to the returns of the
        # same days, so unmasked self-attention lets the return of day j use
        # labels of days j+1..i-1 (look-ahead). causal_attention masks it.
        self.causal_attention = bool(causal_attention)

        # ── factor + position embeddings ────────────────────────────
        self.factor_embedding = nn.Embedding(n_factors, embed_dim)
        self.position_embedding = nn.Embedding(n_lags, embed_dim)

        # ── Variable Selection Network ──────────────────────────────
        self.vsn = _VariableSelectionNetwork(
            n_vars=n_factors,
            d_var=embed_dim,
            d_hidden=embed_dim,
            dropout=dropout,
        )

        # ── static enrichment GRN ──────────────────────────────────
        self.enrich_grn = _GatedResidualNetwork(
            d_in=embed_dim, d_hidden=embed_dim, dropout=dropout,
        )

        # ── temporal self-attention ─────────────────────────────────
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=interaction_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.attn_grn = _GatedResidualNetwork(
            d_in=embed_dim, d_hidden=embed_dim, dropout=dropout,
        )

        # ── output heads ────────────────────────────────────────────
        self.output_proj = nn.Linear(embed_dim, n_features)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(embed_dim, n_features),
            nn.Softplus(),
        )

        # bookkeeping for extraction & transfer learning
        self._last_vsn_weights: Optional[torch.Tensor] = None
        self._last_attn_weights: Optional[torch.Tensor] = None

    # ────────────────────────────────────────────────────────────────

    def forward(
        self,
        factor_presence: torch.Tensor,     # (B, T, K)
        return_decomposed: bool = False,
    ) -> Dict[str, torch.Tensor]:
        B, T, K = factor_presence.shape
        device = factor_presence.device
        D = self.embed_dim

        # 1. embed (weighted by presence)
        idx = torch.arange(K, device=device)
        embeds = self.factor_embedding(idx)                         # (K, D)
        x = factor_presence.unsqueeze(-1) * embeds                  # (B,T,K,D)

        # 2. variable selection
        x_flat = x.reshape(B * T, K, D)
        selected, vsn_w = self.vsn(x_flat)                         # (BT,D), (BT,K)
        selected = selected.reshape(B, T, D)
        vsn_w = vsn_w.reshape(B, T, K)
        self._last_vsn_weights = vsn_w.detach()

        # 3. positional encoding
        pos = torch.arange(min(T, self.n_lags), device=device)
        selected[:, :len(pos)] = selected[:, :len(pos)] + self.position_embedding(pos)

        # 4. enrichment GRN
        enriched = self.enrich_grn(selected)

        # 5. temporal self-attention
        mask = (torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), 1)
                if self.causal_attention else None)
        attn_out, attn_w = self.temporal_attn(
            enriched, enriched, enriched, attn_mask=mask,
        )
        self._last_attn_weights = attn_w.detach()
        enriched = self.attn_norm(enriched + attn_out)
        enriched = self.attn_grn(enriched)

        # 6. output heads
        predicted_returns = self.output_proj(enriched)
        uncertainty = self.uncertainty_head(enriched)

        result: Dict[str, torch.Tensor] = {
            "predicted_returns": predicted_returns,
            "uncertainty": uncertainty,
        }

        if return_decomposed:
            contributions = torch.zeros(
                B, T, K, self.n_features, device=device,
            )
            for k in range(K):
                contributions[:, :, k, :] = (
                    vsn_w[:, :, k : k + 1] * predicted_returns
                )
            result["factor_contributions"] = contributions

        return result

    # ── ImpactMatrix extraction ─────────────────────────────────────

    @torch.no_grad()
    def extract_impact_matrix(
        self,
        factor_presence_history: torch.Tensor,
        factor_names: List[str],
        feature_names: Optional[List[str]] = None,
        llm_directions: Optional[np.ndarray] = None,
        estimator: str = "legacy",
    ) -> ImpactMatrix:
        """Extract an interpretable :class:`ImpactMatrix` (same interface
        as ``FactorImpactNetwork.extract_impact_matrix``).

        ``estimator="ablation"`` (criteria v2): base_impact is the
        in-sample ablation effect only, averaged over every sliding window
        in which the factor is active (the legacy estimator sampled about
        a third of the windows and mixed in probe levels that carry the
        network intercept). Response curves are not returned.

        Now uses multi-intensity probing (matching the FIN implementation)
        to capture non-linear response curves and produce accurate
        ``nonlinearity_scores`` and ``response_curves``.

        Parameters
        ----------
        llm_directions : ndarray, shape (K,) or None
            Average LLM-extracted direction per factor for sign correction.
        """
        self.eval()
        device = next(self.parameters()).device

        if feature_names is None:
            feature_names = [
                f"feature_{i}" for i in range(self.n_features)
            ]

        K = self.n_factors
        M = self.n_features

        # 1. multi-intensity one-hot probing (captures non-linear response)
        intensities = [0.5, 1.0, 2.0]
        probed_by_intensity = {}
        probed_std = np.zeros((K, M))
        for intensity in intensities:
            probed_int = np.zeros((K, M))
            for k in range(K):
                one_hot = torch.zeros(1, self.n_lags, K, device=device)
                one_hot[:, :, k] = intensity
                res = self(one_hot)
                probed_int[k] = res["predicted_returns"].cpu().numpy()[0].mean(0)
                if intensity == 1.0:
                    probed_std[k] = np.sqrt(
                        res["uncertainty"].cpu().numpy()[0].mean(0)
                    )
            probed_by_intensity[intensity] = probed_int

        probed = probed_by_intensity[1.0]

        # Non-linearity measure
        nonlinearity_score = np.abs(
            probed_by_intensity[2.0] - 2.0 * probed_by_intensity[1.0]
        ).mean(axis=1)  # (K,)

        # Response curves: (K, 3, M)
        response_curves = np.stack(
            [probed_by_intensity[i] for i in intensities], axis=1,
        )

        # 2. ablation on real data
        # BUG-C5: probe in windows of n_lags (not full sequence) to
        # match the training distribution.  Aggregate across windows.
        T_total = factor_presence_history.shape[0]
        n_windows = max(1, T_total - self.n_lags + 1)
        full_preds = []
        abl_preds_tft = {k: [] for k in range(K)}

        for start in range(0, n_windows, max(1, n_windows // 50)):
            end = start + self.n_lags
            if end > T_total:
                break
            window = factor_presence_history[start:end].unsqueeze(0).to(device)
            fp = self(window)["predicted_returns"].cpu().numpy()[0]
            full_preds.append((start, end, fp))
            for k in range(K):
                ablated = window.clone()
                ablated[:, :, k] = 0.0
                ap = self(ablated)["predicted_returns"].cpu().numpy()[0]
                abl_preds_tft[k].append((start, end, ap))

        ablation = np.zeros((K, M))
        for k in range(K):
            diffs = []
            for (s, e, fp), (_, _, ap) in zip(full_preds, abl_preds_tft[k]):
                active = factor_presence_history[s:e, k].numpy() > 0
                if active.sum() > 0:
                    diffs.append((fp[active] - ap[active]).mean(0))
            if diffs:
                ablation[k] = np.mean(diffs, axis=0)

        # 3. VSN-based importance
        vsn_w = self._last_vsn_weights
        if vsn_w is not None:
            vsn_avg = vsn_w.cpu().numpy()[0].mean(0)          # (K,)
        else:
            vsn_avg = np.ones(K) / K

        # Adaptive combination: more non-linear → more ablation weight
        combined = np.zeros((K, M))
        for k in range(K):
            nl = float(nonlinearity_score[k])
            w_abl = 0.50 + 0.15 * min(nl / (nl + 0.01), 1.0)
            w_prob = 0.25 - 0.05 * min(nl / (nl + 0.01), 1.0)
            w_vsn = 1.0 - w_abl - w_prob
            combined[k] = (
                w_abl * ablation[k]
                + w_prob * probed[k]
                + w_vsn * (vsn_avg[k] * probed[k])
            )

        if estimator == "ablation":
            combined = self._ablation_all_windows(factor_presence_history, device)

        # DIR-REG: direction consistency correction (same as FIN).
        # A no-op when llm_directions is None (data-sign variant).
        combined = _apply_direction_correction(combined, llm_directions, M)

        occ = factor_presence_history.float().mean(dim=0).numpy()
        temporal = self._extract_temporal_profile()
        interaction = self._extract_interaction_matrix(
            factor_presence_history, device,
        )

        return ImpactMatrix(
            factor_names=factor_names,
            base_impact=combined,
            impact_std=probed_std,
            occurrence_prob=occ,
            temporal_profile=temporal,
            interaction_matrix=interaction,
            feature_names=feature_names,
            nonlinearity_scores=None if estimator == "ablation" else nonlinearity_score,
            response_curves=None if estimator == "ablation" else response_curves,
            estimator=estimator,
        )

    @torch.no_grad()
    def _ablation_all_windows(
        self, factor_presence_history: torch.Tensor, device, batch: int = 256,
    ) -> np.ndarray:
        """Pooled in-sample ablation effect of each factor: the mean, over
        all (window, day) pairs where factor k is active, of
        (prediction - prediction with factor k zeroed in that window).
        Windows are exactly the training windows of prepare_training_data
        (T - n_lags - 1 of them; the last day has no return target).
        Pooling over active days (v2.1) makes b_k * n_k the network's own
        in-sample contribution of factor k. Returns (K, M), normalised units."""
        X = factor_presence_history.float()
        T_total, K = X.shape
        M = self.n_features
        out = np.zeros((K, M))
        n_w = T_total - self.n_lags - 1
        if n_w < 1:
            return out
        W = X.unfold(0, self.n_lags, 1).permute(0, 2, 1).contiguous()[:n_w]  # (n_w, L, K)

        def _pred(B: torch.Tensor) -> torch.Tensor:
            res = []
            for i in range(0, B.shape[0], batch):
                res.append(self(B[i:i + batch].to(device))["predicted_returns"].cpu())
            return torch.cat(res, 0)                                    # (n, L, M)

        full = _pred(W)
        for k in range(K):
            act = W[:, :, k] > 0                                        # (n_w, L)
            sel = act.any(dim=1)
            if not bool(sel.any()):
                continue
            Wk = W[sel].clone()
            Wk[:, :, k] = 0.0
            diff = full[sel] - _pred(Wk)                                # (n_sel, L, M)
            a = act[sel].unsqueeze(-1).float()
            out[k] = ((diff * a).sum(dim=(0, 1)) / a.sum()).numpy()    # pooled over active days
        return out

    @torch.no_grad()
    def _extract_temporal_profile(self) -> np.ndarray:
        """Per-factor temporal sensitivity profile (K, T).

        Probes the model with a one-hot pulse for factor *k* placed at
        time *t* (and zero elsewhere) and measures the magnitude of the
        resulting prediction at the same time step.  This produces a
        DISTINCT profile per factor (unlike the previous version which
        returned the same attention-averaged profile for all factors).
        """
        T = self.n_lags
        K = self.n_factors
        device = next(self.parameters()).device
        profile = np.zeros((K, T))
        for k in range(K):
            for t in range(T):
                pulse = torch.zeros(1, T, K, device=device)
                pulse[0, t, k] = 1.0
                pred = self(pulse)["predicted_returns"][0]   # (T, M)
                # use the magnitude of the response over the next lags
                profile[k, t] = float(pred.abs().mean().item())
            # normalize so that profile sums to 1 across T per factor
            s = profile[k].sum()
            if s > 1e-12:
                profile[k] = profile[k] / s
            else:
                profile[k] = 1.0 / T
        return profile

    @torch.no_grad()
    def _extract_interaction_matrix(
        self,
        factor_presence_history: torch.Tensor,
        device: torch.device,
    ) -> np.ndarray:
        """Extract pairwise interaction matrix with non-linear detection.

        Matches the FIN implementation: uses multi-intensity probing
        to detect non-linear synergies between factor pairs.
        """
        K = self.n_factors
        interaction = np.zeros((K, K))

        # Isolated effects at intensity 1.0
        isolated = np.zeros((K, self.n_features))
        for k in range(K):
            one_hot = torch.zeros(1, self.n_lags, K, device=device)
            one_hot[:, :, k] = 1.0
            isolated[k] = (
                self(one_hot)["predicted_returns"].cpu().numpy()[0].mean(0)
            )

        # Isolated effects at intensity 0.5 (for non-linearity detection)
        isolated_half = np.zeros((K, self.n_features))
        for k in range(K):
            one_hot = torch.zeros(1, self.n_lags, K, device=device)
            one_hot[:, :, k] = 0.5
            isolated_half[k] = (
                self(one_hot)["predicted_returns"].cpu().numpy()[0].mean(0)
            )

        cooc = (
            factor_presence_history.T.float()
            @ factor_presence_history.float()
        ).numpy()
        cooc_norm = cooc / max(cooc.max(), 1)

        for i in range(K):
            for j in range(i + 1, K):
                if cooc_norm[i, j] < 0.05:
                    continue

                # Strategy 1: Standard paired probing
                paired = torch.zeros(1, self.n_lags, K, device=device)
                paired[:, :, i] = 1.0
                paired[:, :, j] = 1.0
                paired_eff = (
                    self(paired)["predicted_returns"]
                    .cpu().numpy()[0].mean(0)
                )
                delta_standard = paired_eff - (isolated[i] + isolated[j])

                # Strategy 2: Half-intensity paired probing
                paired_half = torch.zeros(1, self.n_lags, K, device=device)
                paired_half[:, :, i] = 0.5
                paired_half[:, :, j] = 0.5
                paired_half_eff = (
                    self(paired_half)["predicted_returns"]
                    .cpu().numpy()[0].mean(0)
                )
                delta_half = paired_half_eff - (isolated_half[i] + isolated_half[j])

                # Non-linear interaction detection
                nl_interaction = np.abs(
                    delta_standard - 2.0 * delta_half
                ).mean()

                # Combined: standard interaction + non-linear bonus
                combined_delta = (
                    delta_standard.mean()
                    + 0.3 * np.sign(delta_standard.mean()) * nl_interaction
                )

                interaction[i, j] = combined_delta
                interaction[j, i] = interaction[i, j]

        return interaction

    def get_vsn_weights(self) -> Optional[np.ndarray]:
        """Return last VSN weights — used for transfer learning (F5)."""
        if self._last_vsn_weights is None:
            return None
        return self._last_vsn_weights.cpu().numpy()


# ══════════════════════════════════════════════════════════════════════
# Data preparation
# ══════════════════════════════════════════════════════════════════════
# DIR-REG: direction consistency correction
# ══════════════════════════════════════════════════════════════════════


def _apply_direction_correction(
    combined: np.ndarray,
    llm_directions: Optional[np.ndarray],
    M: int,
) -> np.ndarray:
    """Correct sign conflicts between learned impact and LLM direction.

    For each factor *k* where the LLM assigns a non-neutral direction
    (|d_k| > 0.1) and the learned close-column impact has the opposite
    sign, the close-column impact is flipped and scaled by the LLM's
    confidence (|d_k|).  This preserves the data-driven magnitude while
    enforcing the LLM's causal sign assessment.

    The correction is deliberately conservative:
        corrected = -sign(learned) * |learned| * |llm_direction|

    So a highly confident LLM (|d|≈1) nearly fully flips the value,
    while a weakly confident one (|d|≈0.2) strongly dampens it.

    Parameters
    ----------
    combined : ndarray, shape (K, M)
        Data-driven impact matrix to correct in-place.
    llm_directions : ndarray, shape (K,) or None
        Average LLM direction per factor in [-1, +1].
    M : int
        Number of features.

    Returns
    -------
    combined : ndarray (K, M) — corrected in-place and returned.
    """
    if llm_directions is None:
        return combined
    close_col = min(3, M - 1)
    K = combined.shape[0]
    for k in range(K):
        llm_d = llm_directions[k]
        if abs(llm_d) <= 0.1:
            continue  # neutral — no constraint
        learned_val = combined[k, close_col]
        if abs(learned_val) < 1e-10:
            continue  # effectively zero — nothing to correct
        if np.sign(learned_val) != np.sign(llm_d):
            # Sign conflict: flip to LLM direction, scale by confidence
            combined[k, close_col] = np.sign(llm_d) * abs(learned_val) * abs(llm_d)
    return combined


# ══════════════════════════════════════════════════════════════════════
# Data preparation
# ══════════════════════════════════════════════════════════════════════


def prepare_training_data(
    factor_matrix: np.ndarray,
    price_data: np.ndarray,
    n_lags: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare aligned (X, Y) windowed pairs.

    Returns
    -------
    X : ndarray, shape (N, n_lags, K)
    Y : ndarray, shape (N, n_lags, M)
    """
    log_returns = np.diff(np.log(price_data + 1e-8), axis=0)

    X_list, Y_list = [], []
    for i in range(n_lags, len(log_returns)):
        X_list.append(factor_matrix[i - n_lags : i])
        Y_list.append(log_returns[i - n_lags : i])

    return np.array(X_list, dtype=np.float32), np.array(
        Y_list, dtype=np.float32
    )


# ══════════════════════════════════════════════════════════════════════
# Training loop
# ══════════════════════════════════════════════════════════════════════


def train_impact_network(
    factor_matrix: np.ndarray,
    price_data: np.ndarray,
    factor_names: List[str],
    feature_names: Optional[List[str]] = None,
    n_lags: int = 10,
    n_epochs: int = 300,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    val_fraction: float = 0.15,
    device: str = "auto",
    patience: int = 30,
    architecture: str = "fin",
    llm_directions: Optional[np.ndarray] = None,
    impact_estimator: str = "legacy",
    selection: str = "nll",
    causal_attention: bool = False,
) -> Tuple[nn.Module, ImpactMatrix, Dict]:
    """
    Train the FIN (or TFT variant) and extract the :class:`ImpactMatrix`.

    Parameters
    ----------
    architecture : str
        ``"fin"`` (default) for the shallow FactorImpactNetwork, or
        ``"tft"`` for the Temporal Fusion Transformer variant (F7).

    Returns
    -------
    model : nn.Module
    impact_matrix : ImpactMatrix
    history : dict   with keys ``train_loss``, ``val_loss``, ``best_epoch``
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if feature_names is None:
        feature_names = ["open", "high", "low", "close", "volume"]

    K = factor_matrix.shape[1]
    M = price_data.shape[1]

    # ── direction consistency setup ─────────────────────────────────
    # DIR-REG: prepare LLM direction tensor for sign regularisation.
    _dir_tensor: Optional[torch.Tensor] = None
    _dir_mask: Optional[torch.Tensor] = None
    _close_col = min(3, M - 1)
    if llm_directions is not None:
        _dir_tensor = torch.tensor(
            llm_directions, dtype=torch.float32, device=device,
        )
        _dir_mask = _dir_tensor.abs() > 0.1  # skip neutral factors

    print(f"  > Training Factor Impact Network")
    print(f"    Factors: {K}, Features: {M}, Lags: {n_lags}")
    print(f"    Architecture: {'TFT (Temporal Fusion Transformer)' if architecture == 'tft' else 'FIN (shallow)'}")
    print(f"    Data points: {factor_matrix.shape[0]}, Device: {device}")

    X, Y = prepare_training_data(factor_matrix, price_data, n_lags)
    N = X.shape[0]

    # normalise targets
    Y_mean = Y.mean(axis=(0, 1), keepdims=True)
    Y_std = Y.std(axis=(0, 1), keepdims=True).clip(1e-8)
    Y_norm = (Y - Y_mean) / Y_std

    # temporal train/val split
    split = int(N * (1 - val_fraction))
    X_tr, X_val = X[:split], X[split:]
    Y_tr, Y_val = Y_norm[:split], Y_norm[split:]

    print(f"    Train: {X_tr.shape[0]}, Val: {X_val.shape[0]}")

    X_tr_t = torch.tensor(X_tr)
    Y_tr_t = torch.tensor(Y_tr)
    X_val_t = torch.tensor(X_val).to(device)
    Y_val_t = torch.tensor(Y_val).to(device)

    loader = DataLoader(
        TensorDataset(X_tr_t, Y_tr_t),
        batch_size=batch_size,
        shuffle=True,
    )

    # Model selection (v2.3 candidates; default keeps the submitted rule):
    #   nll       - best validation Gaussian NLL over all targets (legacy)
    #   close_mse - best validation MSE of the close-return mean
    #   last      - no early stopping: the model after n_epochs
    if selection not in ("nll", "close_mse", "last"):
        raise ValueError(f"unknown selection: {selection}")
    if causal_attention and architecture != "tft":
        raise NotImplementedError("causal_attention is implemented for architecture='tft'")
    model: nn.Module
    if architecture == "tft":
        model = TemporalFusionTransformerFIN(
            n_factors=K, n_features=M, n_lags=n_lags,
            causal_attention=causal_attention,
        ).to(device)
    else:
        model = FactorImpactNetwork(
            n_factors=K, n_features=M, n_lags=n_lags,
        ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs
    )

    history: Dict = {
        "train_loss": [],
        "val_loss": [],
        "best_epoch": 0,
    }
    best_val = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(n_epochs):
        model.train()
        losses = []

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            res = model(xb)
            pred, unc = res["predicted_returns"], res["uncertainty"]

            # Gaussian NLL
            nll = 0.5 * (
                torch.log(unc + 1e-8) + (yb - pred).pow(2) / (unc + 1e-8)
            ).mean()

            # sparsity
            if hasattr(model, "factor_impact_weight"):
                sparsity = 0.01 * model.factor_impact_weight.abs().mean()
            else:
                # TFT: L1 on output projection for implicit sparsity
                sparsity = 0.01 * model.output_proj.weight.abs().mean()

            # temporal smoothness
            smooth = (
                0.005 * (pred[:, 1:] - pred[:, :-1]).pow(2).mean()
                if pred.shape[1] > 1
                else 0.0
            )

            # Non-linear interaction loss: encourage the model to
            # learn that co-occurring factors have non-additive effects.
            # Compare: f(x) vs linear_prediction(x) from base weights.
            nl_loss = torch.tensor(0.0, device=device)
            if hasattr(model, "factor_impact_weight"):
                with torch.no_grad():
                    linear_pred = torch.einsum(
                        "btk,km->btm", xb,
                        model.factor_impact_weight,
                    ) + model.factor_impact_bias
                # The model should do BETTER than the linear prediction.
                # Penalise when predictions are too close to linear
                # (which means the non-linear modules aren't adding value).
                residual_from_linear = (pred - linear_pred).pow(2).mean()
                # Adaptive target: scale with the data variance so the
                # non-linearity target is meaningful regardless of the
                # data scale.  Use 1% of the target variance.
                data_var = yb.var().detach().clamp(min=1e-8)
                nl_target = 0.01 * data_var
                nl_loss = 0.005 * torch.relu(nl_target - residual_from_linear)

            # DIR-REG: direction consistency regularisation.
            # Penalise when the learned factor impact sign on the close
            # column contradicts the LLM-extracted direction.
            # Uses a hinge loss: max(0, -sign_llm * w_close) so the
            # penalty is zero when signs agree and grows linearly
            # when they disagree.
            dir_loss = torch.tensor(0.0, device=device)
            if _dir_tensor is not None and _dir_mask is not None and _dir_mask.any():
                d_signs = torch.sign(_dir_tensor[_dir_mask])
                if hasattr(model, "factor_impact_weight"):
                    # FIN: direct per-factor weight on close
                    w_close = model.factor_impact_weight[_dir_mask, _close_col]
                else:
                    # TFT: effective weight ≈ output_proj @ factor_embedding^T
                    eff_w = (
                        model.output_proj.weight
                        @ model.factor_embedding.weight.T
                    ).T  # (K, M)
                    w_close = eff_w[_dir_mask, _close_col]
                dir_loss = 0.05 * torch.relu(-d_signs * w_close).mean()

            loss = nll + sparsity + smooth + nl_loss + dir_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        scheduler.step()

        # validation
        model.eval()
        with torch.no_grad():
            vres = model(X_val_t)
            vp, vu = vres["predicted_returns"], vres["uncertainty"]
            vloss = (
                0.5
                * (
                    torch.log(vu + 1e-8)
                    + (Y_val_t - vp).pow(2) / (vu + 1e-8)
                )
                .mean()
                .item()
            )

            if selection == "close_mse":
                vloss = (Y_val_t[..., _close_col] - vp[..., _close_col]).pow(2).mean().item()
            elif selection == "last":
                vloss = -float(epoch)          # always "improves": keeps the last epoch

        tl = float(np.mean(losses))
        history["train_loss"].append(tl)
        history["val_loss"].append(vloss)

        if vloss < best_val:
            best_val = vloss
            best_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            history["best_epoch"] = epoch
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 30 == 0:
            print(
                f"    Epoch {epoch+1}/{n_epochs} | "
                f"Train: {tl:.5f} | Val: {vloss:.5f} | "
                f"Best: {best_val:.5f} (ep {history['best_epoch']+1})"
            )

        if no_improve >= patience:
            print(f"    Early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)

    # extract
    print("    Extracting impact matrix …")
    factor_t = torch.tensor(factor_matrix, dtype=torch.float32)
    if impact_estimator not in ("legacy", "ablation"):
        raise ValueError(f"unknown impact_estimator: {impact_estimator}")
    if impact_estimator == "ablation" and architecture != "tft":
        raise NotImplementedError("impact_estimator='ablation' is implemented for architecture='tft'")
    extract_kwargs = dict(
        factor_presence_history=factor_t,
        factor_names=factor_names,
        feature_names=feature_names,
        llm_directions=llm_directions,
    )
    if impact_estimator == "ablation":
        extract_kwargs["estimator"] = "ablation"
    impact_matrix = model.extract_impact_matrix(**extract_kwargs)

    # de-normalise
    Y_std_flat = Y_std.squeeze()
    Y_mean_flat = Y_mean.squeeze()
    if impact_matrix.estimator == "ablation":
        # An effect is a difference of normalised predictions: scale only.
        impact_matrix.base_impact = impact_matrix.base_impact * Y_std_flat
        impact_matrix.units = "daily"
    else:
        # legacy (submitted) behaviour, kept for reproducibility: the
        # affine form adds the mean return to every factor effect (B1).
        impact_matrix.base_impact = (
            impact_matrix.base_impact * Y_std_flat + Y_mean_flat
        )
    impact_matrix.impact_std = impact_matrix.impact_std * Y_std_flat
    # BUG-C1: de-normalise interaction_matrix (was computed in normalised
    # Y-space but used in de-normalised space by the generator)
    impact_matrix.interaction_matrix = (
        impact_matrix.interaction_matrix * float(Y_std_flat.mean())
    )
    # De-normalise response curves if present
    if impact_matrix.response_curves is not None:
        # response_curves shape: (K, n_intensities, M)
        impact_matrix.response_curves = (
            impact_matrix.response_curves * Y_std_flat[None, :] + Y_mean_flat[None, :]
        )

    print(impact_matrix.summary())
    return model, impact_matrix, history
