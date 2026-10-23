# -*- coding: utf-8 -*-
"""Tests for cross_asset.py — correlation estimation, Cholesky, DCC-GARCH.

Covers:
- Pearson/Spearman estimation
- Cholesky decomposition preserves correlation
- PSD projection
- Re-correlation of returns and prices
- DCC-GARCH estimation smoke test
"""

from __future__ import annotations

import numpy as np
import pytest

from sde_causal_generator.cross_asset import (
    _nearest_psd,
    cholesky_factor,
    estimate_cross_asset_correlation,
    extract_returns_from_dataframes,
    recorrelate_prices,
    recorrelate_returns,
)


# ══════════════════════════════════════════════════════════════════════
# Correlation estimation
# ══════════════════════════════════════════════════════════════════════


class TestEstimateCrossAssetCorrelation:

    def test_identity_for_single_ticker(self):
        """Single ticker returns identity correlation."""
        ret = {"AAPL": np.random.normal(0, 0.01, 252)}
        corr, tickers = estimate_cross_asset_correlation(ret)
        assert corr.shape == (1, 1)
        assert corr[0, 0] == pytest.approx(1.0)
        assert tickers == ["AAPL"]

    def test_shape_for_two_tickers(self):
        rng = np.random.RandomState(42)
        ret = {
            "AAPL": rng.normal(0, 0.01, 252),
            "GOOG": rng.normal(0, 0.01, 252),
        }
        corr, tickers = estimate_cross_asset_correlation(ret)
        assert corr.shape == (2, 2)
        assert len(tickers) == 2

    def test_diagonal_is_one(self):
        rng = np.random.RandomState(42)
        ret = {
            "AAPL": rng.normal(0, 0.01, 252),
            "GOOG": rng.normal(0, 0.01, 252),
            "MSFT": rng.normal(0, 0.01, 252),
        }
        corr, _ = estimate_cross_asset_correlation(ret)
        np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-6)

    def test_symmetric(self):
        rng = np.random.RandomState(42)
        ret = {
            "AAPL": rng.normal(0, 0.01, 252),
            "GOOG": rng.normal(0, 0.01, 252),
        }
        corr, _ = estimate_cross_asset_correlation(ret)
        np.testing.assert_allclose(corr, corr.T, atol=1e-10)

    def test_correlated_inputs_detected(self):
        """Highly correlated inputs should show high correlation."""
        rng = np.random.RandomState(42)
        base = rng.normal(0, 0.01, 252)
        ret = {
            "A": base,
            "B": base + rng.normal(0, 0.001, 252),
        }
        corr, _ = estimate_cross_asset_correlation(ret)
        assert corr[0, 1] > 0.8, "Highly correlated data should show corr > 0.8"

    def test_spearman_method(self):
        rng = np.random.RandomState(42)
        ret = {
            "AAPL": rng.normal(0, 0.01, 252),
            "GOOG": rng.normal(0, 0.01, 252),
        }
        corr, _ = estimate_cross_asset_correlation(ret, method="spearman")
        assert corr.shape == (2, 2)
        np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-6)


# ══════════════════════════════════════════════════════════════════════
# Cholesky and PSD
# ══════════════════════════════════════════════════════════════════════


class TestCholeskyFactor:

    def test_roundtrip(self):
        """L @ L.T should reproduce the original correlation matrix."""
        rng = np.random.RandomState(42)
        # Build a valid correlation matrix
        A = rng.normal(0, 1, (3, 10))
        C = np.corrcoef(A)
        L = cholesky_factor(C)
        reconstructed = L @ L.T
        np.testing.assert_allclose(reconstructed, C, atol=1e-6)

    def test_lower_triangular(self):
        C = np.eye(3)
        L = cholesky_factor(C)
        assert np.allclose(L, np.tril(L))


class TestNearestPSD:

    def test_makes_non_psd_matrix_psd(self):
        """A matrix with negative eigenvalues should become PSD."""
        bad = np.array([[1.0, 1.5], [1.5, 1.0]])  # not PSD
        psd = _nearest_psd(bad)
        eigvals = np.linalg.eigvalsh(psd)
        assert np.all(eigvals >= 0), "All eigenvalues should be non-negative"

    def test_diagonal_stays_one(self):
        bad = np.array([[1.0, 1.5], [1.5, 1.0]])
        psd = _nearest_psd(bad)
        np.testing.assert_allclose(np.diag(psd), 1.0, atol=1e-6)

    def test_identity_unchanged(self):
        I = np.eye(3)
        psd = _nearest_psd(I)
        np.testing.assert_allclose(psd, I, atol=1e-6)


# ══════════════════════════════════════════════════════════════════════
# Re-correlation
# ══════════════════════════════════════════════════════════════════════


class TestRecorrelateReturns:

    def test_preserves_individual_stats(self):
        """Re-correlation should roughly preserve mean and std of
        individual ticker returns."""
        rng = np.random.RandomState(42)
        tickers = ["A", "B"]
        indep = {
            "A": rng.normal(0.001, 0.01, 252),
            "B": rng.normal(-0.001, 0.02, 252),
        }
        target_corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        L = cholesky_factor(target_corr)

        corr_ret = recorrelate_returns(indep, L, tickers)
        for t in tickers:
            assert len(corr_ret[t]) == 252
            # Std should be in same ballpark
            orig_std = indep[t].std()
            new_std = corr_ret[t].std()
            ratio = new_std / orig_std
            assert 0.3 < ratio < 3.0, (
                f"Std ratio for {t} is {ratio:.2f} (too far from 1)"
            )

    def test_single_ticker_passthrough(self):
        """Single ticker should be returned unchanged."""
        ret = {"A": np.array([0.01, -0.02, 0.03])}
        L = np.eye(1)
        result = recorrelate_returns(ret, L, ["A"])
        np.testing.assert_array_equal(result["A"], ret["A"])


class TestRecorrelatePrices:

    def test_preserves_initial_price(self):
        """Initial prices should be preserved after re-correlation."""
        rng = np.random.RandomState(42)
        tickers = ["A", "B"]
        prices = {
            "A": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 100))),
            "B": 50 * np.exp(np.cumsum(rng.normal(0, 0.01, 100))),
        }
        target_corr = np.array([[1.0, 0.3], [0.3, 1.0]])
        L = cholesky_factor(target_corr)

        corr_prices = recorrelate_prices(prices, L, tickers)
        for t in tickers:
            assert corr_prices[t][0] == pytest.approx(prices[t][0], rel=1e-6)

    def test_positive_prices(self):
        """Prices should remain positive after re-correlation."""
        rng = np.random.RandomState(42)
        tickers = ["A", "B"]
        prices = {
            "A": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 100))),
            "B": 50 * np.exp(np.cumsum(rng.normal(0, 0.01, 100))),
        }
        target_corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        L = cholesky_factor(target_corr)

        corr_prices = recorrelate_prices(prices, L, tickers)
        for t in tickers:
            assert np.all(corr_prices[t] > 0), f"Prices for {t} should be positive"


# ══════════════════════════════════════════════════════════════════════
# extract_returns_from_dataframes
# ══════════════════════════════════════════════════════════════════════


class TestExtractReturns:

    def test_basic_extraction(self):
        import pandas as pd
        dates = pd.bdate_range("2023-01-02", periods=100)
        df_a = pd.DataFrame({"date": dates, "close": 100 * np.exp(np.cumsum(np.random.normal(0, 0.01, 100)))})
        df_b = pd.DataFrame({"date": dates, "close": 50 * np.exp(np.cumsum(np.random.normal(0, 0.01, 100)))})

        result = extract_returns_from_dataframes({"A": df_a, "B": df_b})
        assert "A" in result
        assert "B" in result
        assert len(result["A"]) == 99  # T-1 log-returns

    def test_empty_input(self):
        result = extract_returns_from_dataframes({})
        assert result == {}
