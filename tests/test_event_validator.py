# -*- coding: utf-8 -*-
"""Tests for event_validator.py — UserEvent, presence matrix, merge, dedup.

B6 (partial) from the gap analysis: Tests the event validation module
that handles user-defined events, merging with LLM-extracted factors,
and deduplication.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sde_causal_generator.event_validator import (
    UserEvent,
    deduplicate_events,
    events_to_presence_matrix,
    events_to_scenario_set,
    merge_llm_and_user_events,
)


# ══════════════════════════════════════════════════════════════════════
# UserEvent
# ══════════════════════════════════════════════════════════════════════


class TestUserEvent:

    def test_roundtrip_dict(self):
        """UserEvent should survive to_dict / from_dict roundtrip."""
        ev = UserEvent(
            name="covid_crash",
            start_date="2020-03-01",
            end_date="2020-04-15",
            direction="bearish",
            probability=0.95,
            category="geopolitical",
            description="COVID-19 pandemic crash",
            source="user",
        )
        d = ev.to_dict()
        ev2 = UserEvent.from_dict(d)
        assert ev2.name == ev.name
        assert ev2.start_date == ev.start_date
        assert ev2.end_date == ev.end_date
        assert ev2.direction == ev.direction
        assert ev2.probability == pytest.approx(ev.probability)

    def test_default_values(self):
        ev = UserEvent(
            name="test",
            start_date="2023-01-01",
            end_date="2023-01-31",
        )
        assert ev.direction == "bearish"
        assert ev.probability == 1.0
        assert ev.source == "user"


# ══════════════════════════════════════════════════════════════════════
# events_to_presence_matrix
# ══════════════════════════════════════════════════════════════════════


class TestEventsToPresenceMatrix:

    def test_basic_shape(self):
        dates = pd.bdate_range("2023-01-02", periods=20)
        events = [
            UserEvent("event_a", "2023-01-02", "2023-01-10"),
            UserEvent("event_b", "2023-01-15", "2023-01-20"),
        ]
        presence, names = events_to_presence_matrix(events, dates)
        assert presence.shape == (20, 2)
        assert len(names) == 2

    def test_correct_activation_dates(self):
        dates = pd.bdate_range("2023-01-02", periods=10)
        events = [
            UserEvent("event_a", "2023-01-02", "2023-01-04"),
        ]
        presence, names = events_to_presence_matrix(events, dates)
        # First 3 business days should be active
        assert presence[0, 0] == 1.0
        assert presence[1, 0] == 1.0
        assert presence[2, 0] == 1.0
        # Later days should be inactive
        assert presence[5, 0] == 0.0

    def test_empty_events(self):
        dates = pd.bdate_range("2023-01-02", periods=10)
        presence, names = events_to_presence_matrix([], dates)
        assert presence.shape == (10, 0)
        assert names == []

    def test_values_binary(self):
        dates = pd.bdate_range("2023-01-02", periods=20)
        events = [
            UserEvent("ev", "2023-01-05", "2023-01-15"),
        ]
        presence, _ = events_to_presence_matrix(events, dates)
        unique_vals = set(np.unique(presence))
        assert unique_vals <= {0.0, 1.0}


# ══════════════════════════════════════════════════════════════════════
# events_to_scenario_set
# ══════════════════════════════════════════════════════════════════════


class TestEventsToScenarioSet:

    def test_basic_scenario_set(self):
        events = [
            UserEvent("ev_a", "2023-01-01", "2023-01-31", probability=0.8),
            UserEvent("ev_b", "2023-02-01", "2023-02-28", probability=0.5),
        ]
        ss = events_to_scenario_set(events)
        assert len(ss.scenarios) == 1
        assert ss.scenarios[0].name == "user_defined"
        assert "ev_a" in ss.scenarios[0].factor_probs
        assert ss.scenarios[0].factor_probs["ev_a"] == pytest.approx(0.8)

    def test_factor_names_sorted(self):
        events = [
            UserEvent("zebra", "2023-01-01", "2023-01-31"),
            UserEvent("alpha", "2023-01-01", "2023-01-31"),
        ]
        ss = events_to_scenario_set(events)
        assert ss.factor_names == ["alpha", "zebra"]


# ══════════════════════════════════════════════════════════════════════
# merge_llm_and_user_events
# ══════════════════════════════════════════════════════════════════════


class TestMergeLLMAndUserEvents:

    def test_user_overrides_llm(self):
        """User events should override LLM factors with the same name."""
        dates = pd.bdate_range("2023-01-02", periods=10)
        llm_presence = np.ones((10, 2), dtype=np.float32)
        llm_names = ["factor_a", "factor_b"]

        user_events = [
            UserEvent("factor_a", "2023-01-02", "2023-01-04"),
        ]

        merged, names = merge_llm_and_user_events(
            llm_presence, llm_names, user_events, dates
        )
        assert "factor_a" in names
        idx = names.index("factor_a")
        # User event only covers first 3 days
        assert merged[0, idx] == 1.0
        # Should be 0 after the event ends
        assert merged[5, idx] == 0.0

    def test_new_user_event_appended(self):
        """New user events should be appended to merged matrix."""
        dates = pd.bdate_range("2023-01-02", periods=10)
        llm_presence = np.ones((10, 1), dtype=np.float32)
        llm_names = ["factor_a"]

        user_events = [
            UserEvent("new_event", "2023-01-02", "2023-01-06"),
        ]

        merged, names = merge_llm_and_user_events(
            llm_presence, llm_names, user_events, dates
        )
        assert "new_event" in names
        assert merged.shape[1] == 2  # original + new

    def test_llm_preserved_when_no_conflict(self):
        """LLM factors not in user events should be preserved."""
        dates = pd.bdate_range("2023-01-02", periods=10)
        llm_presence = np.ones((10, 2), dtype=np.float32) * 0.5
        llm_names = ["factor_a", "factor_b"]

        user_events = [
            UserEvent("new_event", "2023-01-02", "2023-01-06"),
        ]

        merged, names = merge_llm_and_user_events(
            llm_presence, llm_names, user_events, dates
        )
        idx_b = names.index("factor_b")
        # factor_b should be unchanged
        np.testing.assert_allclose(merged[:, idx_b], 0.5)


# ══════════════════════════════════════════════════════════════════════
# deduplicate_events
# ══════════════════════════════════════════════════════════════════════


class TestDeduplicateEvents:

    def test_removes_noise_factors(self):
        """Factors with noise-related names should be filtered."""
        events = [
            UserEvent("market_noise", "2023-01-01", "2023-01-31"),
            UserEvent("real_event", "2023-01-01", "2023-01-31"),
            UserEvent("random_residual", "2023-01-01", "2023-01-31"),
        ]
        deduped = deduplicate_events(events)
        names = [e.name for e in deduped]
        assert "real_event" in names
        assert "market_noise" not in names

    def test_merges_same_name_events(self):
        """Events with the same name should be merged."""
        events = [
            UserEvent("rate_hike", "2023-01-01", "2023-01-31", probability=0.8),
            UserEvent("rate_hike", "2023-03-01", "2023-03-31", probability=0.6),
        ]
        deduped = deduplicate_events(events)
        rate_hikes = [e for e in deduped if e.name == "rate_hike"]
        assert len(rate_hikes) == 1
        # Merged should have widest span
        assert rate_hikes[0].start_date <= "2023-01-01"
        assert rate_hikes[0].end_date >= "2023-03-31"

    def test_filters_low_probability(self):
        """Events below min_probability should be removed."""
        events = [
            UserEvent("high_prob", "2023-01-01", "2023-01-31", probability=0.5),
            UserEvent("low_prob", "2023-01-01", "2023-01-31", probability=0.01),
        ]
        deduped = deduplicate_events(events, min_probability=0.08)
        names = [e.name for e in deduped]
        assert "high_prob" in names
        assert "low_prob" not in names

    def test_max_events_limit(self):
        """Should respect max_events limit."""
        events = [
            UserEvent(f"event_{i}", "2023-01-01", "2023-12-31", probability=0.5)
            for i in range(20)
        ]
        deduped = deduplicate_events(events, max_events=5)
        assert len(deduped) <= 5
