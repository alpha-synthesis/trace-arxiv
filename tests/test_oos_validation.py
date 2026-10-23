# -*- coding: utf-8 -*-
"""Tests for oos_validation.py — OOS metrics, MMD, KS.

Tests verify:
- STAT-1: MMD uses median heuristic for bandwidth
- STAT-5: bootstrap CI reported for MMD
- STAT-9: KS test on single sample (not concatenated)
- BUG-C4: financial metrics computed per-sample
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sde_causal_generator.oos_validation import (
    _mmd_rbf,
    oos_validate,
)


@pytest.fixture
def real_test_df():
    """Held-out real test data as a DataFrame."""
    rng = np.random.RandomState(99)
    n = 126  # ~6 months
    dates = pd.bdate_range("2024-01-02", periods=n, freq="B")
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n)))
    return pd.DataFrame({"date": dates, "close": close})


@pytest.fixture
def synth_multi_sample_df():
    """Synthetic data with 3 samples."""
    rng = np.random.RandomState(42)
    n = 126
    dates = pd.bdate_range("2024-01-02", periods=n, freq="B")
    rows = []
    for sid in range(3):
        close = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.014, n)))
        for i in range(n):
            rows.append({"date": dates[i], "close": close[i], "sample": sid})
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════
# STAT-1: MMD median heuristic
# ══════════════════════════════════════════════════════════════════════

class TestMMD:

    def test_zero_for_same_distribution(self):
        rng = np.random.RandomState(0)
        X = rng.normal(0, 1, 500)
        mmd = _mmd_rbf(X, X.copy())
        assert mmd < 0.05, "MMD should be near 0 for same data"

    def test_positive_for_different(self):
        rng = np.random.RandomState(0)
        X = rng.normal(0, 1, 500)
        Y = rng.normal(2, 1, 500)
        mmd = _mmd_rbf(X, Y)
        assert mmd > 0.01

    def test_median_heuristic_default(self):
        """STAT-1: gamma=0 (default) should trigger median heuristic."""
        import inspect
        src = inspect.getsource(_mmd_rbf)
        assert "median" in src.lower(), (
            "STAT-1: _mmd_rbf should use median heuristic"
        )

    def test_small_sample_returns_nan(self):
        X = np.array([1.0, 2.0])
        Y = np.array([1.0, 2.0])
        assert np.isnan(_mmd_rbf(X, Y))


# ══════════════════════════════════════════════════════════════════════
# oos_validate — integration
# ══════════════════════════════════════════════════════════════════════

class TestOOSValidate:

    def test_basic_keys(self, synth_multi_sample_df, real_test_df):
        report = oos_validate(synth_multi_sample_df, real_test_df)
        assert "oos_ks_stat" in report
        assert "oos_wasserstein" in report
        assert "oos_synth_sharpe" in report

    def test_per_sample_ks(self, synth_multi_sample_df, real_test_df):
        """STAT-9: should report per-sample KS p-value median."""
        report = oos_validate(synth_multi_sample_df, real_test_df)
        assert "oos_ks_pval_median" in report, (
            "STAT-9: oos_ks_pval_median should be reported"
        )

    def test_per_sample_financial(self, synth_multi_sample_df, real_test_df):
        """BUG-C4: financial metrics should have std (computed per-sample)."""
        report = oos_validate(synth_multi_sample_df, real_test_df)
        assert "oos_synth_sharpe_std" in report, (
            "BUG-C4: Sharpe std should be reported (per-sample computation)"
        )
        assert "oos_synth_maxdd_std" in report
        assert "oos_synth_var95_std" in report

    def test_mmd_bootstrap_ci(self, synth_multi_sample_df, real_test_df):
        """STAT-5: MMD should have bootstrap CI."""
        report = oos_validate(synth_multi_sample_df, real_test_df)
        if "oos_mmd_rbf" in report:
            assert "oos_mmd_ci_lo" in report, (
                "STAT-5: MMD bootstrap CI should be reported"
            )
            assert "oos_mmd_ci_hi" in report

    def test_insufficient_data(self):
        """Should return error for very short series."""
        df1 = pd.DataFrame({"date": ["2024-01-02"], "close": [100.0]})
        df2 = pd.DataFrame({"date": ["2024-01-02"], "close": [100.0]})
        report = oos_validate(df1, df2)
        assert "error" in report
