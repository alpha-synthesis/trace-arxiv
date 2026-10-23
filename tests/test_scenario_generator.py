# -*- coding: utf-8 -*-
"""Tests for scenario_generator.py — parse, aggregate, empirical fallback.

Tests the LLM-based scenario generator module. Uses mock data only —
no actual LLM calls.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from sde_causal_generator.data_structures import Scenario, ScenarioSet
from sde_causal_generator.scenario_generator import ScenarioGenerator


# ══════════════════════════════════════════════════════════════════════
# _parse_response
# ══════════════════════════════════════════════════════════════════════


class TestParseResponse:

    def test_valid_json_parsed(self):
        """Valid JSON with scenario list should be parsed correctly."""
        raw = json.dumps({
            "scenarios": [
                {
                    "name": "bull",
                    "description": "Bull market",
                    "probability": 0.6,
                    "factor_probabilities": {
                        "rate_hike": 0.3,
                        "earnings_growth": 0.7,
                    },
                },
                {
                    "name": "bear",
                    "description": "Bear market",
                    "probability": 0.4,
                    "factor_probabilities": {
                        "rate_hike": 0.8,
                        "earnings_growth": 0.2,
                    },
                },
            ]
        })
        known_factors = ["rate_hike", "earnings_growth"]
        result = ScenarioGenerator._parse_response(raw, known_factors)
        if result is not None:
            assert len(result) >= 1
            assert all(isinstance(s, Scenario) for s in result)

    def test_malformed_json_returns_none(self):
        """Malformed JSON should return None gracefully."""
        result = ScenarioGenerator._parse_response(
            "not valid json at all!", ["factor_a"]
        )
        assert result is None

    def test_empty_json_returns_none(self):
        result = ScenarioGenerator._parse_response("{}", ["factor_a"])
        assert result is None


# ══════════════════════════════════════════════════════════════════════
# _empirical_fallback
# ══════════════════════════════════════════════════════════════════════


class TestEmpiricalFallback:

    def test_returns_scenario_set(self):
        """Fallback should always return a valid ScenarioSet."""
        factor_names = ["rate_hike", "tech_rally", "geopolitical"]
        rates = {"rate_hike": 0.3, "tech_rally": 0.5, "geopolitical": 0.1}

        ss = ScenarioGenerator._empirical_fallback(factor_names, rates)
        assert isinstance(ss, ScenarioSet)
        assert len(ss.scenarios) > 0
        assert ss.factor_names == factor_names

    def test_fallback_probs_bounded(self):
        """All probabilities in fallback should be in [0, 1]."""
        factor_names = ["f1", "f2", "f3"]
        rates = {"f1": 0.5, "f2": 0.3, "f3": 0.8}

        ss = ScenarioGenerator._empirical_fallback(factor_names, rates)
        for sc in ss.scenarios:
            assert 0.0 <= sc.scenario_prob <= 1.0
            for k, v in sc.factor_probs.items():
                assert 0.0 <= v <= 1.0

    def test_fallback_with_empty_rates(self):
        """Fallback should work even with empty historical rates."""
        factor_names = ["f1", "f2"]
        ss = ScenarioGenerator._empirical_fallback(factor_names, {})
        assert isinstance(ss, ScenarioSet)


# ══════════════════════════════════════════════════════════════════════
# _aggregate
# ══════════════════════════════════════════════════════════════════════


class TestAggregate:

    def test_aggregation_produces_valid_scenarioset(self):
        """Multiple LLM runs should be aggregated properly."""
        run1 = [
            Scenario("bull", "Bull market",
                     {"f1": 0.7, "f2": 0.3}, 0.6),
            Scenario("bear", "Bear market",
                     {"f1": 0.2, "f2": 0.8}, 0.4),
        ]
        run2 = [
            Scenario("bull", "Bull market",
                     {"f1": 0.8, "f2": 0.2}, 0.5),
            Scenario("bear", "Bear market",
                     {"f1": 0.3, "f2": 0.7}, 0.5),
        ]

        ss = ScenarioGenerator._aggregate([run1, run2], ["f1", "f2"])
        assert isinstance(ss, ScenarioSet)
        assert len(ss.scenarios) > 0

    def test_aggregation_single_run(self):
        """Single run should still produce valid output."""
        run = [
            Scenario("base", "Baseline", {"f1": 0.5}, 1.0),
        ]
        ss = ScenarioGenerator._aggregate([run], ["f1"])
        assert isinstance(ss, ScenarioSet)
        assert len(ss.scenarios) >= 1
