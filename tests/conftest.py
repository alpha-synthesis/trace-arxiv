# -*- coding: utf-8 -*-
"""
Shared fixtures for Causal SDE test suite.

Provides deterministic dummy data for all test modules so that
tests are self-contained and never require API keys, GPUs, or
network access.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sde_causal_generator.data_structures import (
    FACTOR_CATEGORIES,
    CausalFactor,
    ImpactMatrix,
    Scenario,
    ScenarioSet,
    WindowAnalysis,
)


# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

N_DAYS = 252          # 1 year of trading data
N_FACTORS = 5         # small number for fast tests
N_FEATURES = 5        # OHLCV
N_LAGS = 10
RNG_SEED = 42


# ──────────────────────────────────────────────────────────────────────
# Fixtures: raw data
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture
def rng():
    """Seeded numpy random state."""
    return np.random.RandomState(RNG_SEED)


@pytest.fixture
def dummy_ohlcv_df(rng) -> pd.DataFrame:
    """FinRL-format OHLCV DataFrame with 252 days of synthetic prices.

    Columns: date, tic, open, high, low, close, volume
    """
    dates = pd.bdate_range("2023-01-02", periods=N_DAYS, freq="B")
    log_returns = rng.normal(0.0003, 0.015, N_DAYS)
    close = 100.0 * np.exp(np.cumsum(log_returns))
    high = close * (1.0 + np.abs(rng.normal(0, 0.008, N_DAYS)))
    low = close * (1.0 - np.abs(rng.normal(0, 0.008, N_DAYS)))
    open_ = close * (1.0 + rng.normal(0, 0.005, N_DAYS))
    volume = rng.lognormal(18, 0.4, N_DAYS).astype(int)

    df = pd.DataFrame({
        "date": dates,
        "tic": "TEST",
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })
    return df


@pytest.fixture
def dummy_ohlcv_array(dummy_ohlcv_df) -> np.ndarray:
    """(T, 5) array of [open, high, low, close, volume]."""
    return dummy_ohlcv_df[["open", "high", "low", "close", "volume"]].values


@pytest.fixture
def dummy_log_returns(dummy_ohlcv_df) -> np.ndarray:
    """(T-1, 5) array of log-returns per feature."""
    prices = dummy_ohlcv_df[["open", "high", "low", "close", "volume"]].values
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.log(prices[1:] / prices[:-1])
    return np.nan_to_num(lr, nan=0.0, posinf=0.0, neginf=0.0)


# ──────────────────────────────────────────────────────────────────────
# Fixtures: factors & presence
# ──────────────────────────────────────────────────────────────────────

DUMMY_FACTOR_NAMES = [
    "fed_rate_hike",
    "earnings_surprise",
    "geopolitical_tension",
    "tech_momentum",
    "market_noise",
]


@pytest.fixture
def dummy_factors() -> list[CausalFactor]:
    """Five CausalFactor instances spanning several categories."""
    return [
        CausalFactor("fed_rate_hike", "macroeconomic", -0.6, 0.35, "medium",
                      "Federal Reserve raises rates"),
        CausalFactor("earnings_surprise", "company_specific", 0.8, 0.50, "short",
                      "Better-than-expected quarterly earnings"),
        CausalFactor("geopolitical_tension", "geopolitical", -0.4, 0.25, "long",
                      "Escalating geopolitical tensions"),
        CausalFactor("tech_momentum", "market_sentiment", 0.5, 0.30, "medium",
                      "Tech sector momentum"),
        CausalFactor("market_noise", "market_sentiment", 0.0, 0.10, "transient",
                      "Market noise / random variance"),
    ]


@pytest.fixture
def dummy_window_analyses(dummy_factors) -> list[WindowAnalysis]:
    """List of WindowAnalysis covering 12 monthly windows."""
    analyses = []
    base_date = pd.Timestamp("2023-01-02")
    for i in range(12):
        start = base_date + pd.DateOffset(months=i)
        end = start + pd.DateOffset(months=1) - pd.DateOffset(days=1)
        # Vary which factors appear in each window
        n = max(1, (i % 4) + 1)
        factors = dummy_factors[:n]
        analyses.append(WindowAnalysis(
            start_date=start.strftime("%Y-%m-%d"),
            end_date=end.strftime("%Y-%m-%d"),
            window_type="monthly",
            price_change_pct=float(np.random.RandomState(i).normal(0, 5)),
            factors=factors,
        ))
    return analyses


@pytest.fixture
def dummy_presence_matrix(rng) -> np.ndarray:
    """(T, K) impact-weighted presence matrix with values in [0, 1]."""
    # Sparse — about 30 % of entries are non-zero
    raw = rng.random((N_DAYS, N_FACTORS))
    mask = rng.random((N_DAYS, N_FACTORS)) > 0.7
    return raw * mask


# ──────────────────────────────────────────────────────────────────────
# Fixtures: impact matrix
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture
def dummy_impact_matrix(rng) -> ImpactMatrix:
    """Minimal ImpactMatrix for generation tests."""
    K, M, L = N_FACTORS, N_FEATURES, N_LAGS
    return ImpactMatrix(
        factor_names=list(DUMMY_FACTOR_NAMES),
        base_impact=rng.normal(0, 0.01, (K, M)),
        impact_std=np.abs(rng.normal(0, 0.005, (K, M))),
        occurrence_prob=rng.uniform(0.1, 0.8, K),
        temporal_profile=np.exp(-0.3 * np.arange(L))[None, :].repeat(K, axis=0),
        interaction_matrix=rng.normal(0, 0.002, (K, K)),
        feature_names=["open", "high", "low", "close", "volume"],
        nonlinearity_scores=rng.uniform(0, 0.5, K),
        response_curves=rng.normal(0, 0.01, (K, 3, M)),
    )


# ──────────────────────────────────────────────────────────────────────
# Fixtures: scenario
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture
def dummy_scenario_set() -> ScenarioSet:
    """Simple ScenarioSet with 2 scenarios."""
    return ScenarioSet(
        scenarios=[
            Scenario(
                name="bull",
                description="Bull market",
                factor_probs={n: 0.5 for n in DUMMY_FACTOR_NAMES},
                scenario_prob=0.6,
            ),
            Scenario(
                name="bear",
                description="Bear market",
                factor_probs={n: 0.3 for n in DUMMY_FACTOR_NAMES},
                scenario_prob=0.4,
            ),
        ],
        factor_names=list(DUMMY_FACTOR_NAMES),
    )


# ──────────────────────────────────────────────────────────────────────
# Fixtures: mock LLM response
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_llm_response_json() -> str:
    """Valid JSON string as an LLM would produce.

    Note: _parse_factors expects a top-level JSON array, not {"factors": [...]}.
    """
    return '''
    [
      {
        "name": "fed_rate_hike",
        "category": "macroeconomic",
        "direction": -0.6,
        "magnitude": 0.35,
        "confidence": 0.85,
        "persistence": "medium",
        "description": "Rate increase announced"
      },
      {
        "name": "tech_earnings",
        "category": "company_specific",
        "direction": 0.8,
        "magnitude": 0.50,
        "confidence": 0.90,
        "persistence": "short",
        "description": "Strong tech earnings"
      },
      {
        "name": "noise_residual",
        "category": "market_sentiment",
        "direction": 0.0,
        "magnitude": 0.15,
        "confidence": 0.50,
        "persistence": "transient",
        "description": "Random market noise"
      }
    ]
    '''
