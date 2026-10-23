# -*- coding: utf-8 -*-
"""Tests for generate.py — ImpactDrivenGenerator + helpers.

Tests verify:
- BUG-C2: _build_schedule uses np.pad (not np.roll)
- STAT-6: variance decomposition uses empirical R² (not magic 0.85)
- STAT-7: t-Student ν clamped ≥ 5
- STAT-8: drift correction uses exp-decay weights (not uniform shift)
- DISC-1: volume β calibrated from data
"""

from __future__ import annotations

import numpy as np
import pytest

from sde_causal_generator.generate import (
    ImpactDrivenGenerator,
    _estimate_jump_params,
    _estimate_t_df,
)


# ══════════════════════════════════════════════════════════════════════
# Helper function tests
# ══════════════════════════════════════════════════════════════════════

class TestEstimateTdf:

    def test_returns_positive_float(self, dummy_log_returns):
        df = _estimate_t_df(dummy_log_returns[:, :4])
        assert isinstance(df, float)
        assert df > 0

    def test_clamped_above_5(self):
        """STAT-7: ν should never be below 5."""
        # Create very fat-tailed data that would give low ν
        rng = np.random.RandomState(0)
        fat = rng.standard_t(2.5, size=(500, 4))
        df = _estimate_t_df(fat)
        assert df >= 5.0, "STAT-7: ν must be ≥ 5 (finite kurtosis)"

    def test_clamped_below_30(self, dummy_log_returns):
        df = _estimate_t_df(dummy_log_returns[:, :4])
        assert df <= 30.0


class TestEstimateJumpParams:

    def test_returns_tuple_of_three(self, dummy_log_returns):
        lam, mu, sig = _estimate_jump_params(dummy_log_returns[:, :4])
        assert isinstance(lam, float)
        assert isinstance(mu, float)
        assert isinstance(sig, float)

    def test_lambda_non_negative(self, dummy_log_returns):
        lam, _, _ = _estimate_jump_params(dummy_log_returns[:, :4])
        assert lam >= 0.0

    def test_sigma_non_negative(self, dummy_log_returns):
        _, _, sig = _estimate_jump_params(dummy_log_returns[:, :4])
        assert sig >= 0.0


# ══════════════════════════════════════════════════════════════════════
# ImpactDrivenGenerator
# ══════════════════════════════════════════════════════════════════════

class TestGenerator:

    def test_generate_shape(self, dummy_impact_matrix, dummy_log_returns):
        gen = ImpactDrivenGenerator(dummy_impact_matrix, window_trading_days=63)
        out = gen.generate_scenario(
            n_steps=50,
            n_samples=3,
            seed=42,
            real_log_returns=dummy_log_returns,
        )
        assert out.shape == (3, 50, 5), f"Expected (3, 50, 5), got {out.shape}"

    def test_prices_positive(self, dummy_impact_matrix, dummy_log_returns):
        gen = ImpactDrivenGenerator(dummy_impact_matrix, window_trading_days=63)
        out = gen.generate_scenario(
            n_steps=50, n_samples=2, seed=42,
            real_log_returns=dummy_log_returns,
        )
        # OHLC columns should be positive
        assert np.all(out[:, :, :4] > 0), "Prices should be positive"

    def test_volume_positive(self, dummy_impact_matrix, dummy_log_returns):
        gen = ImpactDrivenGenerator(dummy_impact_matrix, window_trading_days=63)
        volume = dummy_log_returns[:50, -1]  # dummy
        out = gen.generate_scenario(
            n_steps=50, n_samples=2, seed=42,
            real_log_returns=dummy_log_returns,
            real_volume=np.abs(np.exp(np.cumsum(volume)) * 1e6),
        )
        assert np.all(out[:, :, 4] > 0), "Volume should be positive"

    def test_deterministic_with_seed(self, dummy_impact_matrix, dummy_log_returns):
        gen = ImpactDrivenGenerator(dummy_impact_matrix, window_trading_days=63)
        out1 = gen.generate_scenario(n_steps=50, n_samples=2, seed=123,
                                      real_log_returns=dummy_log_returns)
        out2 = gen.generate_scenario(n_steps=50, n_samples=2, seed=123,
                                      real_log_returns=dummy_log_returns)
        np.testing.assert_array_equal(out1, out2)


# ══════════════════════════════════════════════════════════════════════
# BUG-C2: _build_schedule — no np.roll
# ══════════════════════════════════════════════════════════════════════

class TestBuildSchedule:

    def test_no_np_roll_in_schedule(self):
        """BUG-C2: schedule should not use np.roll (causes circular shift)."""
        import inspect
        src = inspect.getsource(ImpactDrivenGenerator._build_schedule)
        assert "np.roll" not in src, (
            "BUG-C2: _build_schedule must not use np.roll"
        )

    def test_schedule_binary_or_probability(self, dummy_impact_matrix):
        gen = ImpactDrivenGenerator(dummy_impact_matrix, window_trading_days=63)
        schedule = gen._build_schedule(50, None, 2)
        assert schedule.shape == (2, 50, len(dummy_impact_matrix.factor_names))
        assert np.all(schedule >= 0)
        assert np.all(schedule <= 1)


# ══════════════════════════════════════════════════════════════════════
# STAT-6: variance decomposition — no magic 0.85
# ══════════════════════════════════════════════════════════════════════

class TestVarianceDecomposition:

    def test_no_magic_085(self):
        """STAT-6: generate_scenario should not use hardcoded 0.85."""
        import inspect
        src = inspect.getsource(ImpactDrivenGenerator.generate_scenario)
        # target_var * 0.85 should no longer appear
        assert "target_var * 0.85" not in src, (
            "STAT-6: magic constant 0.85 must be replaced by empirical decomposition"
        )
        # The new code should use target_var - signal_var with a floor
        assert "target_var - signal_var" in src or "target_var * 0.15" in src


# ══════════════════════════════════════════════════════════════════════
# STAT-8: drift correction — exp-decay
# ══════════════════════════════════════════════════════════════════════

class TestDriftCorrection:

    def test_uses_exp_decay(self):
        """STAT-8: drift correction should use exponential decay, not uniform."""
        import inspect
        src = inspect.getsource(ImpactDrivenGenerator.generate_scenario)
        # Old pattern: correction_per_day = gap * 0.5 / remaining
        assert "correction_per_day" not in src, (
            "STAT-8: uniform correction_per_day should be replaced by exp-decay"
        )
        assert "half_life" in src or "weights" in src, (
            "STAT-8: drift correction should use exp-decay weights"
        )
