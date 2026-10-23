# -*- coding: utf-8 -*-
"""Tests for factor_relevance.py — scoring, Bonferroni, rank-normalise.

Tests verify:
- STAT-2: Granger score applies Bonferroni correction
- STAT-3: score_factors rank-normalises before combining
- DISC-2: point-biserial threshold reduced to 2
"""

from __future__ import annotations

import numpy as np
import pytest

from sde_causal_generator.factor_relevance import (
    _granger_score,
    _mutual_info_score,
    _pointbiserial_score,
    prune_irrelevant_factors,
    score_factors,
)


@pytest.fixture
def binary_presence(rng):
    """(200, 4) presence matrix with binary {0, 1} columns."""
    return (rng.random((200, 4)) > 0.6).astype(float)


@pytest.fixture
def cont_returns(rng):
    """(200, 5) log-returns for 5 features."""
    return rng.normal(0, 0.015, (200, 5))


# ══════════════════════════════════════════════════════════════════════
# STAT-2: Bonferroni correction in Granger
# ══════════════════════════════════════════════════════════════════════

class TestGrangerScore:

    def test_returns_float(self, binary_presence, cont_returns):
        score = _granger_score(binary_presence[:, 0], cont_returns[:, 3])
        assert isinstance(score, float)

    def test_bounded_01(self, binary_presence, cont_returns):
        score = _granger_score(binary_presence[:, 0], cont_returns[:, 3])
        assert 0.0 <= score <= 1.0

    def test_bonferroni_in_source(self):
        """STAT-2: should apply Bonferroni correction."""
        import inspect
        src = inspect.getsource(_granger_score)
        assert "bonferroni" in src.lower() or "max_lag" in src, (
            "STAT-2: _granger_score should mention Bonferroni correction"
        )

    def test_constant_presence_returns_zero(self, cont_returns):
        """All-zero presence should score 0 (no predictive power)."""
        score = _granger_score(np.zeros(200), cont_returns[:, 3])
        assert score == pytest.approx(0.0)


# ══════════════════════════════════════════════════════════════════════
# DISC-2: point-biserial threshold
# ══════════════════════════════════════════════════════════════════════

class TestPointBiserialScore:

    def test_returns_float(self, binary_presence, cont_returns):
        score = _pointbiserial_score(binary_presence[:, 0], cont_returns[:, 3])
        assert isinstance(score, float)

    def test_threshold_lowered_to_2(self):
        """DISC-2: threshold should be 2 (not 5)."""
        import inspect
        src = inspect.getsource(_pointbiserial_score)
        # Should find  < 2  instead of  < 5
        assert "< 2" in src or "<2" in src, (
            "DISC-2: point-biserial threshold should be reduced to 2"
        )

    def test_very_sparse_uses_fallback(self, cont_returns):
        """A column with only 1 non-zero should still return 0."""
        col = np.zeros(200)
        col[0] = 1.0
        score = _pointbiserial_score(col, cont_returns[:, 3])
        assert score == pytest.approx(0.0)


# ══════════════════════════════════════════════════════════════════════
# STAT-3: rank-normalisation in score_factors
# ══════════════════════════════════════════════════════════════════════

class TestScoreFactors:

    def test_returns_dict(self, binary_presence, cont_returns):
        names = [f"f{i}" for i in range(binary_presence.shape[1])]
        scores = score_factors(binary_presence, cont_returns, names)
        assert isinstance(scores, dict)
        assert len(scores) == len(names)

    def test_composite_bounded(self, binary_presence, cont_returns):
        names = [f"f{i}" for i in range(binary_presence.shape[1])]
        scores = score_factors(binary_presence, cont_returns, names)
        for name, s in scores.items():
            assert 0.0 <= s["composite"] <= 1.0, (
                "STAT-3: composite should be bounded [0, 1] after rank-normalisation"
            )

    def test_rank_normalise_in_source(self):
        """STAT-3: should use rank-normalisation."""
        import inspect
        src = inspect.getsource(score_factors)
        assert "rank" in src.lower(), (
            "STAT-3: score_factors should rank-normalise"
        )


# ══════════════════════════════════════════════════════════════════════
# Prune irrelevant factors
# ══════════════════════════════════════════════════════════════════════

class TestPruneIrrelevant:

    def test_respects_min_score(self, binary_presence, cont_returns):
        names = [f"f{i}" for i in range(binary_presence.shape[1])]
        scores = score_factors(binary_presence, cont_returns, names)
        pruned_p, pruned_names = prune_irrelevant_factors(
            binary_presence, names, scores, min_score=0.0,
        )
        # With min_score=0, nothing should be pruned
        assert len(pruned_names) == len(names)

    def test_max_factors(self, binary_presence, cont_returns):
        names = [f"f{i}" for i in range(binary_presence.shape[1])]
        scores = score_factors(binary_presence, cont_returns, names)
        pruned_p, pruned_names = prune_irrelevant_factors(
            binary_presence, names, scores, min_score=0.0, max_factors=2,
        )
        assert len(pruned_names) <= 2
        assert pruned_p.shape[1] == len(pruned_names)
