# -*- coding: utf-8 -*-
"""Tests for data_structures.py — CausalFactor, ImpactMatrix, Scenario."""

import json
import tempfile

import numpy as np
import pytest

from sde_causal_generator.data_structures import (
    FACTOR_CATEGORIES,
    CausalFactor,
    ImpactMatrix,
    Scenario,
    ScenarioSet,
    WindowAnalysis,
)


# ══════════════════════════════════════════════════════════════════════
# CausalFactor
# ══════════════════════════════════════════════════════════════════════

class TestCausalFactor:

    def test_roundtrip_dict(self, dummy_factors):
        for f in dummy_factors:
            d = f.to_dict()
            f2 = CausalFactor.from_dict(d)
            assert f2.name == f.name
            assert f2.category == f.category
            assert f2.direction == pytest.approx(f.direction)
            assert f2.magnitude == pytest.approx(f.magnitude)

    def test_categories_valid(self, dummy_factors):
        for f in dummy_factors:
            assert f.category in FACTOR_CATEGORIES

    def test_magnitude_range(self, dummy_factors):
        for f in dummy_factors:
            assert 0.0 <= f.magnitude <= 1.0

    def test_direction_range(self, dummy_factors):
        for f in dummy_factors:
            assert -1.0 <= f.direction <= 1.0


# ══════════════════════════════════════════════════════════════════════
# WindowAnalysis
# ══════════════════════════════════════════════════════════════════════

class TestWindowAnalysis:

    def test_roundtrip_dict(self, dummy_window_analyses):
        for wa in dummy_window_analyses:
            d = wa.to_dict()
            wa2 = WindowAnalysis.from_dict(d)
            assert wa2.start_date == wa.start_date
            assert wa2.end_date == wa.end_date
            assert len(wa2.factors) == len(wa.factors)


# ══════════════════════════════════════════════════════════════════════
# ImpactMatrix
# ══════════════════════════════════════════════════════════════════════

class TestImpactMatrix:

    def test_roundtrip_dict(self, dummy_impact_matrix):
        d = dummy_impact_matrix.to_dict()
        im2 = ImpactMatrix.from_dict(d)
        assert im2.factor_names == dummy_impact_matrix.factor_names
        np.testing.assert_allclose(im2.base_impact, dummy_impact_matrix.base_impact)
        np.testing.assert_allclose(im2.interaction_matrix, dummy_impact_matrix.interaction_matrix)

    def test_save_load(self, dummy_impact_matrix, tmp_path):
        path = str(tmp_path / "im.json")
        dummy_impact_matrix.save(path)
        loaded = ImpactMatrix.load(path)
        np.testing.assert_allclose(loaded.base_impact, dummy_impact_matrix.base_impact)
        assert loaded.factor_names == dummy_impact_matrix.factor_names

    def test_shapes(self, dummy_impact_matrix):
        im = dummy_impact_matrix
        K = len(im.factor_names)
        M = len(im.feature_names)
        assert im.base_impact.shape == (K, M)
        assert im.impact_std.shape == (K, M)
        assert im.occurrence_prob.shape == (K,)
        assert im.interaction_matrix.shape == (K, K)

    def test_summary(self, dummy_impact_matrix):
        s = dummy_impact_matrix.summary()
        assert "Factor Impact Matrix" in s
        for name in dummy_impact_matrix.factor_names:
            assert name in s


# ══════════════════════════════════════════════════════════════════════
# ScenarioSet
# ══════════════════════════════════════════════════════════════════════

class TestScenarioSet:

    def test_weighted_probs(self, dummy_scenario_set):
        wp = dummy_scenario_set.weighted_probs()
        # All probs should be between 0 and 1
        for v in wp.values():
            assert 0.0 <= v <= 1.0
        # bull(0.6)*0.5 + bear(0.4)*0.3 = 0.42
        assert wp["fed_rate_hike"] == pytest.approx(0.42)

    def test_roundtrip_dict(self, dummy_scenario_set):
        d = dummy_scenario_set.to_dict()
        ss2 = ScenarioSet.from_dict(d)
        assert len(ss2.scenarios) == 2
        assert ss2.factor_names == dummy_scenario_set.factor_names

    def test_save_load(self, dummy_scenario_set, tmp_path):
        path = str(tmp_path / "scenarios.json")
        dummy_scenario_set.save(path)
        loaded = ScenarioSet.load(path)
        assert len(loaded.scenarios) == len(dummy_scenario_set.scenarios)
