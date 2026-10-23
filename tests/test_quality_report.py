# -*- coding: utf-8 -*-
"""Tests for quality_report.py — ACF fix, volume log-transform.

Tests verify:
- BUG-C8: _proper_acf uses full-series variance (np.correlate, not np.corrcoef)
- DISC-1: volume log-transformed (np.log1p) before z-scoring in discriminator
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest


# ══════════════════════════════════════════════════════════════════════
# BUG-C8: proper ACF
# ══════════════════════════════════════════════════════════════════════

class TestProperACF:

    def test_no_corrcoef_for_acf(self):
        """BUG-C8: _plot_stylized_facts should not use np.corrcoef for ACF."""
        from sde_causal_generator.quality_report import GenerationQualityReport
        src = inspect.getsource(GenerationQualityReport)
        # The old broken ACF pattern:
        assert "np.corrcoef(real_lr" not in src, (
            "BUG-C8: must use _proper_acf instead of np.corrcoef sub-slicing"
        )

    def test_proper_acf_uses_np_correlate(self):
        """BUG-C8: _proper_acf should use np.correlate with full-series variance."""
        from sde_causal_generator.quality_report import GenerationQualityReport
        src = inspect.getsource(GenerationQualityReport)
        assert "np.correlate" in src, (
            "BUG-C8: _proper_acf should use np.correlate"
        )


# ══════════════════════════════════════════════════════════════════════
# DISC-1: volume log-transform in discriminator
# ══════════════════════════════════════════════════════════════════════

class TestDiscriminatorVolumeTransform:

    def test_log1p_in_make_windows(self):
        """DISC-1: _make_windows should log-transform volume."""
        from sde_causal_generator.quality_report import GenerationQualityReport
        src = inspect.getsource(GenerationQualityReport)
        assert "log1p" in src, (
            "DISC-1: _make_windows must apply np.log1p to volume"
        )

    def test_log1p_before_z_scoring(self):
        """DISC-1: log-transform should come before z-scoring."""
        from sde_causal_generator.quality_report import GenerationQualityReport
        src = inspect.getsource(GenerationQualityReport)
        # log1p should appear BEFORE the z-scoring mu/sigma lines
        log1p_pos = src.find("log1p")
        zscore_pos = src.find("mu = vals.mean")
        if log1p_pos >= 0 and zscore_pos >= 0:
            assert log1p_pos < zscore_pos, (
                "DISC-1: log1p must appear before z-scoring"
            )
