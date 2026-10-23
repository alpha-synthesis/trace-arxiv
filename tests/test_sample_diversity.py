# -*- coding: utf-8 -*-
"""Tests for sample diversity and overfitting detection.

B2/B3 from the gap analysis:
- B2: Verify generator doesn't memorize training data (privacy/overfitting)
- B3: Multiple samples should be diverse (not identical copies)
- B4: Metrics should be stable across seeds
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as sp_stats

from sde_causal_generator.data_structures import ImpactMatrix
from sde_causal_generator.generate import ImpactDrivenGenerator


# ══════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════


@pytest.fixture
def sample_impact_matrix():
    """Standard ImpactMatrix for diversity tests."""
    K, M, L = 4, 5, 10
    rng = np.random.RandomState(42)
    return ImpactMatrix(
        factor_names=[f"factor_{i}" for i in range(K)],
        base_impact=rng.normal(0, 0.005, (K, M)),
        impact_std=np.abs(rng.normal(0, 0.003, (K, M))),
        occurrence_prob=rng.uniform(0.2, 0.6, K),
        temporal_profile=np.exp(-0.3 * np.arange(L))[None, :].repeat(K, axis=0),
        interaction_matrix=rng.normal(0, 0.001, (K, K)),
        feature_names=["open", "high", "low", "close", "volume"],
        nonlinearity_scores=rng.uniform(0, 0.3, K),
        response_curves=rng.normal(0, 0.005, (K, 3, M)),
    )


@pytest.fixture
def sample_log_returns():
    rng = np.random.RandomState(42)
    return rng.normal(0.0003, 0.015, (252, 5))


# ══════════════════════════════════════════════════════════════════════
# B3: Inter-sample diversity
# ══════════════════════════════════════════════════════════════════════


class TestSampleDiversity:

    def test_samples_are_not_identical(
        self, sample_impact_matrix, sample_log_returns
    ):
        """Multiple samples from same seed batch should differ."""
        gen = ImpactDrivenGenerator(
            sample_impact_matrix, window_trading_days=63
        )
        ohlcv = gen.generate_scenario(
            n_steps=63, n_samples=5, seed=42,
            real_log_returns=sample_log_returns,
        )
        # Compare each pair
        for i in range(ohlcv.shape[0]):
            for j in range(i + 1, ohlcv.shape[0]):
                assert not np.allclose(ohlcv[i], ohlcv[j], atol=1e-6), (
                    f"Samples {i} and {j} should not be identical"
                )

    def test_inter_sample_variance_positive(
        self, sample_impact_matrix, sample_log_returns
    ):
        """Variance across samples should be positive (real diversity)."""
        gen = ImpactDrivenGenerator(
            sample_impact_matrix, window_trading_days=63
        )
        ohlcv = gen.generate_scenario(
            n_steps=63, n_samples=10, seed=42,
            real_log_returns=sample_log_returns,
        )
        close = ohlcv[:, :, 3]  # (10, 63)
        log_ret = np.diff(np.log(close), axis=1)

        # Variance across samples at each time step
        inter_sample_var = np.var(log_ret, axis=0)
        mean_var = inter_sample_var.mean()
        assert mean_var > 1e-8, (
            f"Inter-sample variance should be positive (got {mean_var:.2e})"
        )

    def test_coverage_of_return_space(
        self, sample_impact_matrix, sample_log_returns
    ):
        """Samples should cover a reasonable range of cumulative returns."""
        gen = ImpactDrivenGenerator(
            sample_impact_matrix, window_trading_days=126
        )
        ohlcv = gen.generate_scenario(
            n_steps=126, n_samples=20, seed=42,
            real_log_returns=sample_log_returns,
        )
        close = ohlcv[:, :, 3]

        # Cumulative log-returns for each sample
        cum_returns = np.log(close[:, -1] / close[:, 0])

        # Range should be meaningful
        ret_range = cum_returns.max() - cum_returns.min()
        assert ret_range > 0.01, (
            f"Range of cumulative returns across samples should be > 1% "
            f"(got {ret_range:.4f})"
        )


# ══════════════════════════════════════════════════════════════════════
# B2: Non-memorization / privacy
# ══════════════════════════════════════════════════════════════════════


class TestNonMemorization:

    def test_nearest_neighbor_distance_ratio(
        self, sample_impact_matrix, sample_log_returns
    ):
        """Synthetic samples should not be exact copies of reference data.
        Nearest-neighbor distance ratio checks for memorization."""
        gen = ImpactDrivenGenerator(
            sample_impact_matrix, window_trading_days=63
        )
        ohlcv = gen.generate_scenario(
            n_steps=63, n_samples=10, seed=42,
            real_log_returns=sample_log_returns,
        )
        synth_close = ohlcv[:, :, 3]
        synth_lr = np.diff(np.log(synth_close), axis=1)

        # Reference data: first 63 days of log returns
        real_lr = sample_log_returns[:62, 3]  # close column

        # Compute L2 distance from each synthetic sample to reference
        distances = []
        for i in range(synth_lr.shape[0]):
            dist = np.linalg.norm(synth_lr[i] - real_lr)
            distances.append(dist)

        min_dist = min(distances)
        # If min distance is 0, the generator memorized the training data
        assert min_dist > 1e-6, (
            f"Minimum distance to reference data should be > 0 "
            f"(got {min_dist:.2e}) — possible memorization"
        )


# ══════════════════════════════════════════════════════════════════════
# B4: Seed stability
# ══════════════════════════════════════════════════════════════════════


class TestSeedStability:

    def test_metrics_stable_across_seeds(
        self, sample_impact_matrix, sample_log_returns
    ):
        """Key metrics should have bounded CV across different seeds."""
        gen = ImpactDrivenGenerator(
            sample_impact_matrix, window_trading_days=63
        )

        sharpes = []
        mean_returns = []
        volatilities = []

        for seed in range(20):
            ohlcv = gen.generate_scenario(
                n_steps=63, n_samples=1, seed=seed,
                real_log_returns=sample_log_returns,
            )
            close = ohlcv[0, :, 3]
            lr = np.diff(np.log(close))
            mean_ret = lr.mean()
            vol = lr.std()
            sharpe = mean_ret / vol if vol > 1e-10 else 0.0

            mean_returns.append(mean_ret)
            volatilities.append(vol)
            sharpes.append(sharpe)

        # Volatility should not vary wildly across seeds
        vol_array = np.array(volatilities)
        vol_cv = vol_array.std() / vol_array.mean() if vol_array.mean() > 0 else 0
        assert vol_cv < 2.0, (
            f"Volatility CV across seeds should be < 2.0 (got {vol_cv:.2f})"
        )

    def test_reproducibility_same_seed(
        self, sample_impact_matrix, sample_log_returns
    ):
        """Same seed should always produce identical results."""
        gen = ImpactDrivenGenerator(
            sample_impact_matrix, window_trading_days=63
        )
        out1 = gen.generate_scenario(
            n_steps=63, n_samples=3, seed=42,
            real_log_returns=sample_log_returns,
        )
        out2 = gen.generate_scenario(
            n_steps=63, n_samples=3, seed=42,
            real_log_returns=sample_log_returns,
        )
        np.testing.assert_array_equal(out1, out2)
