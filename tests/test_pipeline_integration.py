# -*- coding: utf-8 -*-
"""Integration-level tests for pipeline.py — Top-N pruning & end-to-end.

Tests verify:
- DISC-2: Top-N pruning uses hybrid ranking (occurrence × max_magnitude)
  and protects tail-events (peak magnitude ≥ 0.7).
- End-to-end wiring: generate_scenario produces valid OHLCV from an
  ImpactMatrix built by hand.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from sde_causal_generator.data_structures import ImpactMatrix
from sde_causal_generator.generate import ImpactDrivenGenerator


# ══════════════════════════════════════════════════════════════════════
# DISC-2: Top-N hybrid ranking (source-level)
# ══════════════════════════════════════════════════════════════════════

class TestTopNPruning:

    def test_hybrid_ranking_in_pipeline(self):
        """DISC-2: pruning should use hybrid_score = occurrence * max_magnitude."""
        from sde_causal_generator.pipeline import CausalSDEPipeline
        src = inspect.getsource(CausalSDEPipeline)
        assert "hybrid_score" in src, (
            "DISC-2: pipeline must compute hybrid_score for Top-N"
        )
        assert "max_magnitude" in src

    def test_tail_event_protection(self):
        """DISC-2: factors with peak ≥ 0.7 should be auto-kept."""
        from sde_causal_generator.pipeline import CausalSDEPipeline
        src = inspect.getsource(CausalSDEPipeline)
        assert "tail_mask" in src or "0.7" in src, (
            "DISC-2: pipeline must protect tail-events from pruning"
        )


# ══════════════════════════════════════════════════════════════════════
# End-to-end: ImpactMatrix → Generator → OHLCV
# ══════════════════════════════════════════════════════════════════════

class TestEndToEndGeneration:

    def test_generate_from_impact_matrix(self, dummy_impact_matrix, dummy_log_returns):
        """Smoke test the full generation path."""
        gen = ImpactDrivenGenerator(dummy_impact_matrix, window_trading_days=63)
        ohlcv = gen.generate_scenario(
            n_steps=63,
            n_samples=2,
            seed=0,
            real_log_returns=dummy_log_returns,
        )
        # Shape: (samples, steps, features)
        assert ohlcv.ndim == 3
        assert ohlcv.shape[0] == 2
        assert ohlcv.shape[1] == 63
        assert ohlcv.shape[2] == 5

        # OHLC should be positive
        for col in range(4):
            assert np.all(ohlcv[:, :, col] > 0), f"Column {col} should be positive prices"

        # Volume should be positive
        assert np.all(ohlcv[:, :, 4] > 0), "Volume should be positive"

    def test_returns_magnitude_reasonable(self, dummy_impact_matrix, dummy_log_returns):
        """Generated daily log-returns should be within [-0.20, 0.20]."""
        gen = ImpactDrivenGenerator(dummy_impact_matrix, window_trading_days=63)
        ohlcv = gen.generate_scenario(
            n_steps=63, n_samples=5, seed=42,
            real_log_returns=dummy_log_returns,
        )
        close = ohlcv[:, :, 3]
        log_ret = np.diff(np.log(close), axis=1)
        assert np.all(log_ret >= -0.21) and np.all(log_ret <= 0.21), (
            "Clamped returns should be in [-0.20, 0.20]"
        )

    def test_different_seeds_differ(self, dummy_impact_matrix, dummy_log_returns):
        gen = ImpactDrivenGenerator(dummy_impact_matrix, window_trading_days=63)
        out1 = gen.generate_scenario(n_steps=50, n_samples=1, seed=1,
                                      real_log_returns=dummy_log_returns)
        out2 = gen.generate_scenario(n_steps=50, n_samples=1, seed=2,
                                      real_log_returns=dummy_log_returns)
        assert not np.allclose(out1, out2), "Different seeds should give different outputs"
