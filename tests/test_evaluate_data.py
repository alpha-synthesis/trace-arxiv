# -*- coding: utf-8 -*-
"""Tests for evaluate_data.py — metrics, bootstrap CIs, predictive score.

Tests verify:
- BUG-C3: predictive_score uses linear output (no sigmoid) + MSE
- STAT-4: kl_divergence uses density=False and manual normalisation
- STAT-5: bootstrap_ci returns point, ci_lo, ci_hi, se
- Metric functions return expected keys
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from sde_causal_generator.evaluate_data import (
    _acf,
    _log_returns,
    acf_distance,
    bootstrap_ci,
    financial_metrics,
    kl_divergence,
    ks_test,
    moments_comparison,
    tail_comparison,
    volatility_comparison,
    wasserstein_distance,
)


@pytest.fixture
def real_returns(rng):
    return rng.normal(0, 0.015, 500)


@pytest.fixture
def synth_returns(rng):
    return rng.normal(0.0001, 0.016, 500)


# ══════════════════════════════════════════════════════════════════════
# Low-level helpers
# ══════════════════════════════════════════════════════════════════════

class TestLogReturns:

    def test_length(self):
        prices = np.array([100, 101, 99, 102], dtype=float)
        lr = _log_returns(prices)
        assert len(lr) == 3

    def test_values(self):
        prices = np.array([100.0, 110.0])
        lr = _log_returns(prices)
        assert lr[0] == pytest.approx(np.log(1.1))


class TestACF:

    def test_lag_zero_is_one(self, real_returns):
        acf = _acf(real_returns, 10)
        assert acf[0] == pytest.approx(1.0, abs=1e-6)

    def test_length(self, real_returns):
        acf = _acf(real_returns, 15)
        assert len(acf) == 16  # 0..15

    def test_constant_series(self):
        acf = _acf(np.ones(100), 5)
        np.testing.assert_allclose(acf, 0.0, atol=1e-10)


# ══════════════════════════════════════════════════════════════════════
# Statistical metrics
# ══════════════════════════════════════════════════════════════════════

class TestKSTest:

    def test_keys(self, real_returns, synth_returns):
        result = ks_test(real_returns, synth_returns)
        assert "ks_stat" in result
        assert "ks_pval" in result

    def test_identical_data(self, real_returns):
        result = ks_test(real_returns, real_returns)
        assert result["ks_stat"] == 0.0
        assert result["ks_pval"] == 1.0


class TestWasserstein:

    def test_zero_for_identical(self, real_returns):
        result = wasserstein_distance(real_returns, real_returns)
        assert result["wasserstein"] == pytest.approx(0.0)

    def test_positive_for_different(self, real_returns, synth_returns):
        result = wasserstein_distance(real_returns, synth_returns)
        assert result["wasserstein"] > 0


class TestKLDivergence:

    def test_sym_kl_positive(self, real_returns, synth_returns):
        """STAT-4: KL should be non-negative."""
        result = kl_divergence(real_returns, synth_returns)
        assert result["kl_sym"] >= 0
        assert result["kl_pq"] >= 0
        assert result["kl_qp"] >= 0

    def test_near_zero_for_identical(self, real_returns):
        result = kl_divergence(real_returns, real_returns)
        # With identical data, KL should be very close to 0
        assert result["kl_sym"] < 0.01

    def test_no_density_true_in_source(self):
        """STAT-4: must NOT use density=True."""
        import inspect
        src = inspect.getsource(kl_divergence)
        assert "density=True" not in src


class TestMoments:

    def test_keys(self, real_returns, synth_returns):
        result = moments_comparison(real_returns, synth_returns)
        assert "real_mean" in result
        assert "rel_diff_std" in result


class TestACFDistance:

    def test_zero_for_identical(self, real_returns):
        result = acf_distance(real_returns, real_returns)
        assert result["acf_dist_returns"] == pytest.approx(0.0)


class TestVolatilityComparison:

    def test_keys(self, real_returns, synth_returns):
        result = volatility_comparison(real_returns, synth_returns)
        assert "vol_rmse" in result
        assert "vol_corr" in result


class TestTailComparison:

    def test_keys(self, real_returns, synth_returns):
        result = tail_comparison(real_returns, synth_returns)
        assert "tail_2sigma_diff" in result
        assert "tail_3sigma_diff" in result


class TestFinancialMetrics:

    def test_keys(self):
        prices_r = 100 * np.exp(np.cumsum(np.random.normal(0, 0.01, 252)))
        prices_s = 100 * np.exp(np.cumsum(np.random.normal(0, 0.01, 252)))
        result = financial_metrics(prices_r, prices_s)
        assert "sharpe_diff" in result
        assert "max_dd_diff" in result
        assert "var95_diff" in result


# ══════════════════════════════════════════════════════════════════════
# STAT-5: bootstrap_ci
# ══════════════════════════════════════════════════════════════════════

class TestBootstrapCI:

    def test_returns_expected_keys(self, real_returns, synth_returns):
        fn = lambda r, s: float(stats.ks_2samp(r, s)[0])
        result = bootstrap_ci(fn, real_returns, synth_returns, n_boot=100)
        assert "point" in result
        assert "ci_lo" in result
        assert "ci_hi" in result
        assert "se" in result

    def test_ci_contains_point(self, real_returns, synth_returns):
        fn = lambda r, s: float(stats.wasserstein_distance(r, s))
        result = bootstrap_ci(fn, real_returns, synth_returns, n_boot=200)
        assert result["ci_lo"] <= result["point"] <= result["ci_hi"]

    def test_se_positive(self, real_returns, synth_returns):
        fn = lambda r, s: float(stats.wasserstein_distance(r, s))
        result = bootstrap_ci(fn, real_returns, synth_returns, n_boot=200)
        assert result["se"] > 0

    def test_narrow_ci_for_identical(self, real_returns):
        """CI should be very narrow when comparing same data."""
        fn = lambda r, s: float(stats.wasserstein_distance(r, s))
        result = bootstrap_ci(fn, real_returns, real_returns, n_boot=200)
        assert result["ci_hi"] - result["ci_lo"] < 0.01


# ══════════════════════════════════════════════════════════════════════
# BUG-C3: predictive_score — no sigmoid
# ══════════════════════════════════════════════════════════════════════

class TestPredictiveScore:

    def test_no_sigmoid_in_source(self):
        """BUG-C3: predictive_score must use linear output, not sigmoid."""
        from sde_causal_generator.evaluate_data import predictive_score
        import inspect
        src = inspect.getsource(predictive_score)
        # Check that sigmoid is NOT called on predictions.
        # Comments mentioning "sigmoid" are OK — only actual calls matter.
        assert "torch.sigmoid(" not in src and ".sigmoid()" not in src, (
            "BUG-C3: predictive_score should NOT apply sigmoid to output"
        )

    def test_uses_mse(self):
        """BUG-C3: should use MSELoss for regression."""
        from sde_causal_generator.evaluate_data import predictive_score
        import inspect
        src = inspect.getsource(predictive_score)
        assert "MSELoss" in src
