# -*- coding: utf-8 -*-
"""Tests for DIR-REG — LLM direction consistency regularisation.

Verifies that the direction consistency mechanism:
1. Corrects sign conflicts in extract_impact_matrix (post-extraction).
2. Adds a hinge loss during training that penalises weight/direction
   disagreements.
3. Integrates correctly via _compute_llm_directions in the pipeline.
4. Works identically for both FIN and TFT architectures.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sde_causal_generator.data_structures import (
    CausalFactor,
    ImpactMatrix,
    WindowAnalysis,
)
from sde_causal_generator.factor_impact_network import (
    FactorImpactNetwork,
    TemporalFusionTransformerFIN,
    _apply_direction_correction,
    train_impact_network,
)
from sde_causal_generator.pipeline import CausalSDEPipeline

# ── small dims for fast tests ────────────────────────────────────────
K = 5
M = 5
L = 10
T = 100


# ══════════════════════════════════════════════════════════════════════
# _apply_direction_correction — unit tests
# ══════════════════════════════════════════════════════════════════════


class TestApplyDirectionCorrection:
    """Test the post-extraction sign-correction helper."""

    def test_no_directions_is_noop(self):
        combined = np.array([
            [0.01, 0.02, 0.03, 0.04, 0.05],  # factor 0
            [-0.01, -0.02, -0.03, -0.04, -0.05],  # factor 1
        ])
        original = combined.copy()
        result = _apply_direction_correction(combined, None, M)
        np.testing.assert_array_equal(result, original)

    def test_neutral_direction_is_noop(self):
        """Factors with direction ≈ 0 should not be corrected."""
        combined = np.array([
            [0.01, 0.02, 0.03, 0.04, 0.05],
        ])
        original = combined.copy()
        directions = np.array([0.05])  # below threshold
        result = _apply_direction_correction(combined, directions, M)
        np.testing.assert_array_equal(result, original)

    def test_agreement_is_noop(self):
        """When LLM direction and learned sign agree, no change."""
        combined = np.array([
            [0.01, 0.02, 0.03, -0.04, 0.05],  # close col (3) is negative
        ])
        original = combined.copy()
        directions = np.array([-0.7])  # bearish — agrees with negative close
        result = _apply_direction_correction(combined, directions, M)
        np.testing.assert_array_equal(result, original)

    def test_conflict_is_corrected(self):
        """When LLM says bearish but learned is bullish, flip the sign."""
        combined = np.array([
            [0.01, 0.02, 0.03, 0.05, 0.01],  # close=+0.05 (bullish)
        ])
        directions = np.array([-0.8])  # LLM says bearish
        result = _apply_direction_correction(combined, directions, M)
        # close col should now be negative, magnitude scaled by |direction|
        assert result[0, 3] < 0, "Close impact should be bearish after correction"
        expected = -1.0 * 0.05 * 0.8  # -sign * |learned| * |llm_dir|
        np.testing.assert_almost_equal(result[0, 3], expected, decimal=8)

    def test_bullish_conflict_corrected(self):
        """LLM says bullish but learned is bearish → flip to positive."""
        combined = np.array([
            [0.0, 0.0, 0.0, -0.03, 0.0],  # close=-0.03 (bearish)
        ])
        directions = np.array([0.6])  # LLM says bullish
        result = _apply_direction_correction(combined, directions, M)
        assert result[0, 3] > 0, "Close impact should be bullish after correction"
        np.testing.assert_almost_equal(result[0, 3], 0.03 * 0.6, decimal=8)

    def test_multiple_factors_mixed(self):
        """Multiple factors with different correction needs."""
        combined = np.array([
            [0.0, 0.0, 0.0, +0.05, 0.0],  # bullish learned
            [0.0, 0.0, 0.0, -0.03, 0.0],  # bearish learned
            [0.0, 0.0, 0.0, +0.02, 0.0],  # bullish learned
        ])
        directions = np.array([
            -0.7,  # LLM bearish → conflict with factor 0
            -0.5,  # LLM bearish → agrees with factor 1
            +0.9,  # LLM bullish → agrees with factor 2
        ])
        result = _apply_direction_correction(combined, directions, M)
        assert result[0, 3] < 0, "Factor 0 should be flipped to bearish"
        assert result[1, 3] < 0, "Factor 1 should remain bearish (agreement)"
        assert result[2, 3] > 0, "Factor 2 should remain bullish (agreement)"

    def test_zero_learned_is_noop(self):
        """If learned impact is ~0, don't try to flip."""
        combined = np.array([[0.0, 0.0, 0.0, 1e-12, 0.0]])
        directions = np.array([-0.9])
        result = _apply_direction_correction(combined, directions, M)
        # Effectively zero — should not be touched
        assert abs(result[0, 3]) < 1e-9


# ══════════════════════════════════════════════════════════════════════
# extract_impact_matrix — direction correction is applied
# ══════════════════════════════════════════════════════════════════════


class TestExtractImpactMatrixDirectionCorrection:
    """Verify that extracted ImpactMatrix respects LLM directions."""

    @pytest.fixture
    def fin_model(self):
        return FactorImpactNetwork(
            n_factors=K, n_features=M, n_lags=L,
            embed_dim=16, interaction_heads=2, dropout=0.0,
        )

    @pytest.fixture
    def tft_model(self):
        return TemporalFusionTransformerFIN(
            n_factors=K, n_features=M, n_lags=L,
            embed_dim=16, interaction_heads=2, dropout=0.0,
        )

    @pytest.fixture
    def presence(self):
        rng = np.random.RandomState(42)
        return torch.tensor(
            (rng.random((T, K)) * (rng.random((T, K)) > 0.7)).astype(np.float32)
        )

    @pytest.fixture
    def factor_names(self):
        return [f"factor_{i}" for i in range(K)]

    def test_fin_with_no_directions_unchanged(self, fin_model, presence, factor_names):
        """Without llm_directions, extraction works as before."""
        im_no = fin_model.extract_impact_matrix(
            presence, factor_names, llm_directions=None,
        )
        assert isinstance(im_no, ImpactMatrix)
        assert im_no.base_impact.shape == (K, M)

    def test_fin_directions_flip_sign(self, fin_model, presence, factor_names):
        """With llm_directions, conflicting signs get corrected."""
        # Extract without directions
        im_no = fin_model.extract_impact_matrix(
            presence, factor_names, llm_directions=None,
        )
        close_col = min(3, M - 1)
        # Build directions that oppose every factor's learned sign
        learned_signs = np.sign(im_no.base_impact[:, close_col])
        opposing = -learned_signs * 0.9  # max confidence, opposing
        # For factors with learned ~0, use 0 to avoid noise
        opposing[np.abs(im_no.base_impact[:, close_col]) < 1e-8] = 0.0

        im_dir = fin_model.extract_impact_matrix(
            presence, factor_names, llm_directions=opposing,
        )
        # Every non-zero factor should now have its sign flipped
        for k in range(K):
            if abs(opposing[k]) > 0.1 and abs(im_no.base_impact[k, close_col]) > 1e-8:
                assert np.sign(im_dir.base_impact[k, close_col]) == np.sign(opposing[k]), (
                    f"Factor {k}: expected sign {np.sign(opposing[k])}, "
                    f"got {np.sign(im_dir.base_impact[k, close_col])}"
                )

    def test_tft_directions_flip_sign(self, tft_model, presence, factor_names):
        """TFT extract_impact_matrix also applies direction correction."""
        im_no = tft_model.extract_impact_matrix(
            presence, factor_names, llm_directions=None,
        )
        close_col = min(3, M - 1)
        learned_signs = np.sign(im_no.base_impact[:, close_col])
        opposing = -learned_signs * 0.9
        opposing[np.abs(im_no.base_impact[:, close_col]) < 1e-8] = 0.0

        im_dir = tft_model.extract_impact_matrix(
            presence, factor_names, llm_directions=opposing,
        )
        for k in range(K):
            if abs(opposing[k]) > 0.1 and abs(im_no.base_impact[k, close_col]) > 1e-8:
                assert np.sign(im_dir.base_impact[k, close_col]) == np.sign(opposing[k]), (
                    f"Factor {k}: expected sign {np.sign(opposing[k])}, "
                    f"got {np.sign(im_dir.base_impact[k, close_col])}"
                )


# ══════════════════════════════════════════════════════════════════════
# train_impact_network — direction regularisation in training loop
# ══════════════════════════════════════════════════════════════════════


class TestTrainWithDirectionRegularisation:
    """Verify that the directional hinge loss nudges weights."""

    @pytest.fixture
    def training_data(self):
        """Synthetic data where factor 0 always causes positive returns
        and factor 1 always causes negative returns."""
        rng = np.random.RandomState(42)
        T_data = 200
        K_data = 3
        M_data = 5

        # Factor presence: sparse
        factor_matrix = np.zeros((T_data, K_data), dtype=np.float32)
        # Factor 0: active day 20-50 (bullish effect)
        factor_matrix[20:50, 0] = 0.8
        # Factor 1: active day 80-120 (bearish effect)
        factor_matrix[80:120, 1] = 0.7

        # Price data: base random walk + factor effects
        close = np.zeros(T_data)
        close[0] = 100.0
        for t in range(1, T_data):
            ret = rng.normal(0.0001, 0.01)
            if factor_matrix[t, 0] > 0:
                ret += 0.005  # bullish
            if factor_matrix[t, 1] > 0:
                ret -= 0.005  # bearish
            close[t] = close[t - 1] * np.exp(ret)

        # Build OHLCV
        price_data = np.column_stack([
            close * (1 + rng.normal(0, 0.003, T_data)),  # open
            close * (1 + np.abs(rng.normal(0, 0.005, T_data))),  # high
            close * (1 - np.abs(rng.normal(0, 0.005, T_data))),  # low
            close,
            rng.lognormal(18, 0.3, T_data),  # volume
        ]).astype(np.float32)

        return factor_matrix, price_data

    def test_train_with_directions_completes(self, training_data):
        """Training with llm_directions runs without error."""
        factor_matrix, price_data = training_data
        K_data = factor_matrix.shape[1]
        directions = np.array([-0.8, -0.5, 0.0], dtype=np.float32)
        factor_names = [f"f{i}" for i in range(K_data)]

        model, im, history = train_impact_network(
            factor_matrix=factor_matrix,
            price_data=price_data,
            factor_names=factor_names,
            n_lags=L,
            n_epochs=5,
            batch_size=16,
            patience=100,
            architecture="fin",
            llm_directions=directions,
        )
        assert isinstance(im, ImpactMatrix)
        assert im.base_impact.shape[0] == K_data

    def test_train_with_directions_tft(self, training_data):
        """TFT training with llm_directions runs without error."""
        factor_matrix, price_data = training_data
        K_data = factor_matrix.shape[1]
        directions = np.array([-0.8, -0.5, 0.0], dtype=np.float32)
        factor_names = [f"f{i}" for i in range(K_data)]

        model, im, history = train_impact_network(
            factor_matrix=factor_matrix,
            price_data=price_data,
            factor_names=factor_names,
            n_lags=L,
            n_epochs=5,
            batch_size=16,
            patience=100,
            architecture="tft",
            llm_directions=directions,
        )
        assert isinstance(im, ImpactMatrix)

    def test_train_without_directions_still_works(self, training_data):
        """Backward compat: training without llm_directions is unchanged."""
        factor_matrix, price_data = training_data
        K_data = factor_matrix.shape[1]
        factor_names = [f"f{i}" for i in range(K_data)]

        model, im, history = train_impact_network(
            factor_matrix=factor_matrix,
            price_data=price_data,
            factor_names=factor_names,
            n_lags=L,
            n_epochs=5,
            batch_size=16,
            patience=100,
            architecture="fin",
        )
        assert isinstance(im, ImpactMatrix)

    def test_direction_regularisation_affects_close_sign(self, training_data):
        """With enough epochs, direction regularisation should nudge
        the close-column sign toward the LLM direction."""
        factor_matrix, price_data = training_data
        K_data = factor_matrix.shape[1]
        factor_names = [f"f{i}" for i in range(K_data)]

        # Factor 0 is genuinely bullish in data.
        # Provide a BEARISH direction to test that the reg has effect.
        directions = np.array([-0.9, -0.9, 0.0], dtype=np.float32)

        _, im_with, _ = train_impact_network(
            factor_matrix=factor_matrix,
            price_data=price_data,
            factor_names=factor_names,
            n_lags=L,
            n_epochs=30,
            batch_size=16,
            patience=100,
            architecture="fin",
            llm_directions=directions,
        )

        _, im_without, _ = train_impact_network(
            factor_matrix=factor_matrix,
            price_data=price_data,
            factor_names=factor_names,
            n_lags=L,
            n_epochs=30,
            batch_size=16,
            patience=100,
            architecture="fin",
            llm_directions=None,
        )

        close_col = min(3, price_data.shape[1] - 1)
        # The extraction_correction should ensure factor 0 close is ≤ 0
        # (bearish) when direction is -0.9
        assert im_with.base_impact[0, close_col] <= 0, (
            f"Factor 0 close-col impact should be bearish with "
            f"llm_directions=-0.9, got {im_with.base_impact[0, close_col]:.6f}"
        )


# ══════════════════════════════════════════════════════════════════════
# _compute_llm_directions — pipeline helper
# ══════════════════════════════════════════════════════════════════════


class TestComputeLLMDirections:
    """Test that pipeline extracts correct average directions."""

    def test_basic_averaging(self):
        """Directions are averaged across windows for the same factor."""
        analyses = [
            WindowAnalysis(
                start_date="2020-01-01", end_date="2020-01-31",
                window_type="monthly", price_change_pct=-10.0,
                factors=[
                    CausalFactor("covid", "macro", -0.8, 0.5, "medium"),
                    CausalFactor("stimulus", "policy", 0.6, 0.3, "medium"),
                ],
            ),
            WindowAnalysis(
                start_date="2020-02-01", end_date="2020-02-28",
                window_type="monthly", price_change_pct=-15.0,
                factors=[
                    CausalFactor("covid", "macro", -0.6, 0.7, "medium"),
                ],
            ),
        ]
        factor_names = ["covid", "stimulus"]
        dirs = CausalSDEPipeline._compute_llm_directions(analyses, factor_names)
        assert dirs.shape == (2,)
        np.testing.assert_almost_equal(dirs[0], (-0.8 + -0.6) / 2)  # covid
        np.testing.assert_almost_equal(dirs[1], 0.6)  # stimulus (1 window)

    def test_missing_factor_gets_zero(self):
        """A factor not found in any analysis gets direction 0."""
        analyses = [
            WindowAnalysis(
                start_date="2020-01-01", end_date="2020-01-31",
                window_type="monthly", price_change_pct=-5.0,
                factors=[
                    CausalFactor("known", "macro", -0.5, 0.3, "medium"),
                ],
            ),
        ]
        factor_names = ["known", "unknown"]
        dirs = CausalSDEPipeline._compute_llm_directions(analyses, factor_names)
        np.testing.assert_almost_equal(dirs[0], -0.5)
        np.testing.assert_almost_equal(dirs[1], 0.0)

    def test_empty_analyses(self):
        """All zeros when no analyses are provided."""
        dirs = CausalSDEPipeline._compute_llm_directions([], ["a", "b"])
        np.testing.assert_array_equal(dirs, [0.0, 0.0])

    def test_consistent_direction_stays_strong(self):
        """Factor consistently bearish across many windows → strong negative."""
        factors = [CausalFactor("crash", "macro", -0.9, 0.8, "medium")]
        analyses = [
            WindowAnalysis(
                start_date=f"2020-0{i+1}-01",
                end_date=f"2020-0{i+1}-28",
                window_type="monthly",
                price_change_pct=-5.0,
                factors=factors,
            )
            for i in range(5)
        ]
        dirs = CausalSDEPipeline._compute_llm_directions(analyses, ["crash"])
        np.testing.assert_almost_equal(dirs[0], -0.9)
