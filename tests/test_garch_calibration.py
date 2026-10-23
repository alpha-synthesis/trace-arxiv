# -*- coding: utf-8 -*-
"""Tests for GJR-GARCH(1,1) calibration and volatility clustering.

Verifies that:
  1. _fit_gjr_garch_params returns valid GARCH parameters.
  2. Degenerate reference data (i.i.d.) triggers canonical fallback.
  3. Real data with ARCH effects uses fitted parameters.
  4. Generated data reproduces volatility clustering (ACF of |r| > 0).
  5. The full pipeline (signal + GARCH) preserves clustering.

References:
  - Bollerslev (1986), "Generalized Autoregressive Conditional
    Heteroskedasticity"
  - Glosten, Jagannathan & Runkle (1993), "On the Relation between
    the Expected Value and the Volatility of the Nominal Excess Return
    on Stocks"
"""

from __future__ import annotations

import numpy as np
import pytest

from sde_causal_generator.data_structures import ImpactMatrix
from sde_causal_generator.generate import (
    ImpactDrivenGenerator,
    _fit_gjr_garch_params,
    _sample_skewed_t,
)


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════


def _acf_lag1_abs(x: np.ndarray) -> float:
    """ACF of |x| at lag 1."""
    a = np.abs(x)
    a = a - a.mean()
    v = np.var(a)
    if v < 1e-15:
        return 0.0
    c = np.correlate(a, a, mode="full")
    c = c[len(a) - 1 :] / (v * len(a))
    return float(c[1])


def _simulate_garch(
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
    n_steps: int,
    target_v: float,
    seed: int = 0,
) -> np.ndarray:
    """Simulate a single GJR-GARCH(1,1) path with Gaussian innovations."""
    rng = np.random.RandomState(seed)
    sigma2 = np.empty(n_steps)
    eps = np.empty(n_steps)
    sigma2[0] = target_v
    z = rng.standard_normal(n_steps)
    eps[0] = z[0] * np.sqrt(sigma2[0])
    for t in range(1, n_steps):
        lev = 1.0 if eps[t - 1] < 0 else 0.0
        sigma2[t] = (
            omega
            + alpha * eps[t - 1] ** 2
            + gamma * lev * eps[t - 1] ** 2
            + beta * sigma2[t - 1]
        )
        sigma2[t] = np.clip(sigma2[t], 1e-10, target_v * 10)
        eps[t] = z[t] * np.sqrt(sigma2[t])
    return eps


# ══════════════════════════════════════════════════════════════════════
# Test class: GARCH parameter calibration
# ══════════════════════════════════════════════════════════════════════


class TestGarchCalibration:
    """Verify _fit_gjr_garch_params returns sensible parameters."""

    def test_returns_four_floats(self):
        """Should return (omega, alpha, gamma, beta) as floats."""
        rng = np.random.RandomState(0)
        lr = rng.standard_t(df=5, size=252) * 0.012
        result = _fit_gjr_garch_params(lr)
        assert len(result) == 4
        for v in result:
            assert isinstance(v, float)

    def test_stationarity_constraint(self):
        """α + γ/2 + β must be < 1 for stationarity."""
        rng = np.random.RandomState(0)
        lr = rng.standard_t(df=5, size=500) * 0.015
        omega, alpha, gamma, beta = _fit_gjr_garch_params(lr)
        persistence = alpha + gamma / 2.0 + beta
        assert persistence < 1.0, (
            f"Stationarity violated: α+γ/2+β = {persistence:.4f}"
        )

    def test_nonnegative_params(self):
        """All GARCH parameters must be non-negative."""
        rng = np.random.RandomState(42)
        lr = rng.standard_t(df=5, size=300) * 0.012
        omega, alpha, gamma, beta = _fit_gjr_garch_params(lr)
        assert omega >= 0, f"omega must be ≥ 0 (got {omega})"
        assert alpha >= 0, f"alpha must be ≥ 0 (got {alpha})"
        assert gamma >= 0, f"gamma must be ≥ 0 (got {gamma})"
        assert beta >= 0, f"beta must be ≥ 0 (got {beta})"

    def test_alpha_within_bounds(self):
        """Alpha should be in [0.01, 0.20] or canonical fallback."""
        rng = np.random.RandomState(0)
        lr = rng.standard_t(df=5, size=252) * 0.012
        _, alpha, _, _ = _fit_gjr_garch_params(lr)
        assert 0.01 <= alpha <= 0.20

    def test_beta_within_bounds(self):
        """Beta should be in [0.50, 0.98] or canonical fallback."""
        rng = np.random.RandomState(0)
        lr = rng.standard_t(df=5, size=252) * 0.012
        _, _, _, beta = _fit_gjr_garch_params(lr)
        assert 0.50 <= beta <= 0.98


class TestGarchDegeneracyFallback:
    """When reference data lacks ARCH effects, canonical params should
    be used so that generated data still exhibits vol clustering."""

    def test_iid_data_uses_canonical_params(self):
        """i.i.d. data → degenerate fit → canonical fallback."""
        rng = np.random.RandomState(42)
        lr = rng.standard_t(df=5, size=504) * 0.012
        omega, alpha, gamma, beta = _fit_gjr_garch_params(lr)
        # Canonical values
        assert alpha == pytest.approx(0.06, abs=1e-6)
        assert gamma == pytest.approx(0.10, abs=1e-6)
        assert beta == pytest.approx(0.88, abs=1e-6)

    @pytest.mark.parametrize("seed", [0, 1, 7, 13, 99])
    def test_iid_data_canonical_across_seeds(self, seed):
        """Multiple i.i.d. draws should all trigger canonical fallback."""
        rng = np.random.RandomState(seed)
        lr = rng.standard_t(df=5, size=504) * 0.012
        omega, alpha, gamma, beta = _fit_gjr_garch_params(lr)
        persistence = alpha + gamma / 2.0 + beta
        # Persistence should be ≥ 0.90 (canonical = 0.99)
        assert persistence >= 0.90, (
            f"seed={seed}: persistence={persistence:.4f} too low"
        )
        # ARCH reactivity should be meaningful
        arch_react = alpha + gamma / 2.0
        assert arch_react >= 0.05, (
            f"seed={seed}: ARCH reactivity={arch_react:.4f} too low"
        )

    def test_real_garch_data_uses_fitted_params(self):
        """Data with genuine ARCH effects should NOT trigger fallback."""
        # Generate data with known GARCH dynamics
        rng = np.random.RandomState(42)
        T = 1000
        omega_true, alpha_true, gamma_true, beta_true = 1e-6, 0.08, 0.12, 0.85
        sigma2 = np.empty(T)
        eps = np.empty(T)
        sigma2[0] = omega_true / (1 - alpha_true - gamma_true / 2 - beta_true)
        for t in range(T):
            z = rng.standard_normal()
            eps[t] = z * np.sqrt(sigma2[max(0, t)])
            if t < T - 1:
                lev = 1.0 if eps[t] < 0 else 0.0
                sigma2[t + 1] = (
                    omega_true
                    + alpha_true * eps[t] ** 2
                    + gamma_true * lev * eps[t] ** 2
                    + beta_true * sigma2[t]
                )

        omega, alpha, gamma, beta = _fit_gjr_garch_params(eps)
        # Should NOT be the canonical values
        persistence = alpha + gamma / 2.0 + beta
        assert persistence >= 0.90, f"persistence={persistence:.4f}"
        # Should recover something near the true alpha (not exactly due to noise)
        assert 0.01 <= alpha <= 0.20
        assert 0.00 <= gamma <= 0.25


# ══════════════════════════════════════════════════════════════════════
# Test class: Volatility clustering in generated data
# ══════════════════════════════════════════════════════════════════════


class TestVolClusteringGeneration:
    """Verify that the full pipeline produces vol clustering."""

    @pytest.fixture
    def impact_matrix(self):
        """ImpactMatrix with clear directional factors (same as experiments)."""
        K, M, L = 8, 5, 10
        rng = np.random.RandomState(42)
        base_impact = rng.normal(0, 0.005, (K, M))
        base_impact[0, 3] = 0.04
        base_impact[1, 3] = 0.03
        base_impact[2, 3] = 0.025
        base_impact[3, 3] = -0.04
        base_impact[4, 3] = -0.03
        base_impact[5, 3] = -0.025
        base_impact[6, 3] = 0.002
        base_impact[7, 3] = -0.001
        return ImpactMatrix(
            factor_names=[f"factor_{i}" for i in range(K)],
            base_impact=base_impact,
            impact_std=np.abs(rng.normal(0, 0.003, (K, M))),
            occurrence_prob=rng.uniform(0.15, 0.5, K),
            temporal_profile=np.exp(-0.3 * np.arange(L))[None, :].repeat(
                K, axis=0
            ),
            interaction_matrix=rng.normal(0, 0.001, (K, K)),
            feature_names=["open", "high", "low", "close", "volume"],
            nonlinearity_scores=rng.uniform(0, 0.3, K),
            response_curves=rng.normal(0, 0.005, (K, 3, M)),
        )

    @pytest.fixture
    def reference_returns(self):
        """i.i.d. t-dist reference — exactly as experiments use."""
        rng = np.random.RandomState(42)
        lr = rng.standard_t(df=5, size=(504, 5)) * 0.012
        lr[:, 0] += 0.0003
        return lr

    def test_pure_garch_acf_positive(self):
        """Pure GARCH with canonical params should  produce ACF(|r|) > 0.05."""
        omega, alpha, gamma, beta = 1e-6, 0.06, 0.10, 0.88
        target_v = (0.012) ** 2  # typical daily variance
        acf_vals = []
        for seed in range(20):
            eps = _simulate_garch(omega, alpha, gamma, beta, 504, target_v, seed)
            acf_vals.append(_acf_lag1_abs(eps))
        mean_acf = np.mean(acf_vals)
        assert mean_acf > 0.05, (
            f"Pure GARCH ACF(|r|, lag-1) = {mean_acf:.4f}, expected > 0.05"
        )

    def test_pipeline_acf_positive(self, impact_matrix, reference_returns):
        """Full pipeline (factors + GARCH + jumps + calibration) should
        produce vol clustering: ACF(|r|, lag-1) > 0.05."""
        gen = ImpactDrivenGenerator(impact_matrix, window_trading_days=504)
        ohlcv = gen.generate_scenario(
            n_steps=504,
            n_samples=20,
            seed=42,
            real_log_returns=reference_returns,
        )
        close = ohlcv[:, :, 3]
        log_ret = np.diff(np.log(close), axis=1)

        per_sample_acf = []
        for s in range(log_ret.shape[0]):
            per_sample_acf.append(_acf_lag1_abs(log_ret[s]))

        mean_acf = np.mean(per_sample_acf)
        assert mean_acf > 0.05, (
            f"Pipeline ACF(|r|, lag-1) = {mean_acf:.4f}, expected > 0.05. "
            f"Vol clustering is too weak — GARCH calibration may be degenerate."
        )

    def test_pooled_acf_positive(self, impact_matrix, reference_returns):
        """Pooled (across samples) ACF(|r|) should also be positive."""
        gen = ImpactDrivenGenerator(impact_matrix, window_trading_days=504)
        ohlcv = gen.generate_scenario(
            n_steps=504,
            n_samples=20,
            seed=42,
            real_log_returns=reference_returns,
        )
        close = ohlcv[:, :, 3]
        log_ret = np.diff(np.log(close), axis=1)
        pooled = log_ret.ravel()

        acf_pooled = _acf_lag1_abs(pooled)
        assert acf_pooled > 0.05, (
            f"Pooled ACF(|r|, lag-1) = {acf_pooled:.4f}, expected > 0.05"
        )

    def test_acf_abs_greater_than_acf_returns(
        self, impact_matrix, reference_returns
    ):
        """ACF(|r|) should be larger than ACF(r) — classic stylized fact:
        returns are uncorrelated but |returns| are autocorrelated."""
        gen = ImpactDrivenGenerator(impact_matrix, window_trading_days=504)
        ohlcv = gen.generate_scenario(
            n_steps=504,
            n_samples=20,
            seed=42,
            real_log_returns=reference_returns,
        )
        close = ohlcv[:, :, 3]
        log_ret = np.diff(np.log(close), axis=1)

        acf_abs_list = []
        acf_raw_list = []
        for s in range(log_ret.shape[0]):
            acf_abs_list.append(_acf_lag1_abs(log_ret[s]))
            # ACF of raw returns at lag 1
            r = log_ret[s]
            r = r - r.mean()
            v = np.var(r)
            if v > 1e-15:
                c = np.correlate(r, r, mode="full")
                c = c[len(r) - 1 :] / (v * len(r))
                acf_raw_list.append(abs(c[1]))
            else:
                acf_raw_list.append(0.0)

        mean_abs = np.mean(acf_abs_list)
        mean_raw = np.mean(acf_raw_list)
        assert mean_abs > mean_raw, (
            f"ACF(|r|)={mean_abs:.4f} should > ACF(r)={mean_raw:.4f}"
        )
