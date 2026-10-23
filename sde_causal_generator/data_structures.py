# -*- coding: utf-8 -*-
"""
Core data structures for the Causal SDE Generator.

Defines the shared data classes used across all three phases
(extraction, impact learning, generation).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

FACTOR_CATEGORIES = [
    "macroeconomic",        # interest rates, inflation, GDP
    "geopolitical",         # wars, sanctions, trade agreements
    "sector_specific",      # industry trends, regulation
    "company_specific",     # earnings, products, management
    "market_sentiment",     # fear/greed, momentum, herding
    "technical",            # support/resistance, volume patterns
    "liquidity",            # market microstructure, flow
]


# ──────────────────────────────────────────────────────────────────────
# Phase 1: Factor Extraction
# ──────────────────────────────────────────────────────────────────────

@dataclass
class CausalFactor:
    """A single causal factor identified by the LLM."""

    name: str
    category: str       # one of FACTOR_CATEGORIES
    direction: float    # -1.0 (bearish) to +1.0 (bullish)
    magnitude: float    # 0.0 to 1.0 (normalized impact strength)
    persistence: str    # "transient", "short", "medium", "long"
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "category": self.category,
            "direction": self.direction,
            "magnitude": self.magnitude,
            "persistence": self.persistence,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CausalFactor":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


@dataclass
class WindowAnalysis:
    """Analysis result for a single time window."""

    start_date: str
    end_date: str
    window_type: str          # "daily", "monthly", "quarterly"
    price_change_pct: float
    factors: List[CausalFactor] = field(default_factory=list)
    raw_llm_response: str = ""

    def to_dict(self) -> dict:
        return {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "window_type": self.window_type,
            "price_change_pct": self.price_change_pct,
            "factors": [f.to_dict() for f in self.factors],
            "raw_llm_response": self.raw_llm_response,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WindowAnalysis":
        factors = [CausalFactor.from_dict(f) for f in d.get("factors", [])]
        return cls(
            start_date=d["start_date"],
            end_date=d["end_date"],
            window_type=d["window_type"],
            price_change_pct=d["price_change_pct"],
            factors=factors,
            raw_llm_response=d.get("raw_llm_response", ""),
        )


def normalize_article_text(text: str) -> str:
    """Normalise article text for identity and de-duplication.

    Lowercase, collapse every run of whitespace (including newlines, tabs
    and non-breaking spaces) to one space, and strip.
    """
    return " ".join(str(text).lower().split())


def compute_article_id(
    source: str, published_date: str, title: str, content: str
) -> str:
    """Content-deterministic article id (corpus v2).

    ``md5(source | date | title | normalized_content)``.  There is no
    process counter, so the same article gets the same id in every
    instance and every process, and re-ingesting it is an idempotent
    upsert instead of a new document.
    """
    raw = (
        f"{str(source).strip()}|{str(published_date).strip()}|"
        f"{str(title).strip()}|{normalize_article_text(content)}"
    )
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


@dataclass
class NewsArticle:
    """A single news article with temporal metadata.

    ``article_id`` is content-deterministic (:func:`compute_article_id`).
    An id passed explicitly to the constructor is kept, but
    :meth:`from_dict` always recomputes it: ids stored in cache files were
    produced by an older, counter-based scheme and are never trusted.
    """

    title: str
    content: str
    source: str
    published_date: str  # ISO format YYYY-MM-DD
    tickers_mentioned: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    url: str = ""
    article_id: str = ""

    def __post_init__(self):
        if not self.article_id:
            self.article_id = self.content_id()

    def content_id(self) -> str:
        """The content-deterministic id of this article."""
        return compute_article_id(
            self.source, self.published_date, self.title, self.content
        )

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "content": self.content,
            "source": self.source,
            "published_date": self.published_date,
            "tickers_mentioned": self.tickers_mentioned,
            "categories": self.categories,
            "url": self.url,
            "article_id": self.article_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NewsArticle":
        """Reconstruct a NewsArticle from a dict (e.g. JSON cache).

        The cached ``article_id`` is ignored and recomputed from content.
        """
        return cls(
            title=d["title"],
            content=d["content"],
            source=d["source"],
            published_date=d["published_date"],
            tickers_mentioned=list(d.get("tickers_mentioned", []) or []),
            categories=list(d.get("categories", []) or []),
            url=d.get("url", ""),
        )


# ──────────────────────────────────────────────────────────────────────
# Phase 2: Impact Learning
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ImpactMatrix:
    """
    Per-factor impact characterisation produced by the FIN.

    Attributes:
        factor_names: List of K factor names.
        base_impact: (K, M) — average directional impact of each factor
                     on each feature (signed magnitude).
        impact_std: (K, M) — uncertainty / variance of estimated impact.
        occurrence_prob: (K,) — empirical probability of each factor
                        being active in a random window.
        temporal_profile: (K, L) — how the impact of each factor decays
                         over L lag steps.
        interaction_matrix: (K, K) — pairwise factor interaction strengths.
                           Positive = amplifying, negative = dampening.
        feature_names: List of M feature names.
        nonlinearity_scores: (K,) — per-factor non-linearity score.
                            Higher values indicate the factor has
                            strongly non-linear effects (response at
                            2× intensity ≠ 2× response at 1×).
        response_curves: (K, n_intensities, M) — factor response at
                        multiple activation intensities [0.5, 1.0, 2.0].
                        Enables non-linear interpolation during generation
                        instead of purely linear factor application.
    """

    factor_names: List[str]
    base_impact: np.ndarray         # (K, M)
    impact_std: np.ndarray          # (K, M)
    occurrence_prob: np.ndarray     # (K,)
    temporal_profile: np.ndarray    # (K, L)
    interaction_matrix: np.ndarray  # (K, K)
    feature_names: List[str]
    nonlinearity_scores: Optional[np.ndarray] = None  # (K,)
    response_curves: Optional[np.ndarray] = None      # (K, n_intensities, M)
    # Provenance of base_impact (criteria v2):
    #   estimator "legacy"   = R0 mix of ablation + probe levels, +mu
    #             "ablation" = in-sample ablation over all windows, x Y_std
    #   units     "legacy"   = generator divides by window_days
    #             "daily"    = per-day log-return effect of an active factor
    estimator: str = "legacy"
    units: str = "legacy"

    def summary(self) -> str:
        """Human-readable summary of top factors and their impacts."""
        lines = ["═══ Factor Impact Matrix ═══", ""]
        K = len(self.factor_names)

        avg_impact = np.abs(self.base_impact).mean(axis=1)
        order = np.argsort(avg_impact)[::-1]

        header = f"{'Factor':<30} {'P(occur)':<10} {'Avg |Impact|':<14} {'Direction':<10}"
        lines.append(header)
        lines.append("─" * len(header))

        for i in order:
            name = self.factor_names[i]
            prob = self.occurrence_prob[i]
            avg_abs = avg_impact[i]
            avg_dir = self.base_impact[i].mean()
            if avg_dir > 0.01:
                direction = "▲ bullish"
            elif avg_dir < -0.01:
                direction = "▼ bearish"
            else:
                direction = "─ neutral"
            lines.append(
                f"{name:<30} {prob:<10.3f} {avg_abs:<14.5f} {direction}"
            )

        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = {
            "factor_names": self.factor_names,
            "base_impact": self.base_impact.tolist(),
            "impact_std": self.impact_std.tolist(),
            "occurrence_prob": self.occurrence_prob.tolist(),
            "temporal_profile": self.temporal_profile.tolist(),
            "interaction_matrix": self.interaction_matrix.tolist(),
            "feature_names": self.feature_names,
        }
        if self.nonlinearity_scores is not None:
            d["nonlinearity_scores"] = self.nonlinearity_scores.tolist()
        if self.response_curves is not None:
            d["response_curves"] = self.response_curves.tolist()
        d["estimator"] = self.estimator
        d["units"] = self.units
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ImpactMatrix":
        kwargs = dict(
            factor_names=d["factor_names"],
            base_impact=np.array(d["base_impact"]),
            impact_std=np.array(d["impact_std"]),
            occurrence_prob=np.array(d["occurrence_prob"]),
            temporal_profile=np.array(d["temporal_profile"]),
            interaction_matrix=np.array(d["interaction_matrix"]),
            feature_names=d["feature_names"],
        )
        if "nonlinearity_scores" in d:
            kwargs["nonlinearity_scores"] = np.array(d["nonlinearity_scores"])
        if "response_curves" in d:
            kwargs["response_curves"] = np.array(d["response_curves"])
        kwargs["estimator"] = d.get("estimator", "legacy")
        kwargs["units"] = d.get("units", "legacy")
        return cls(**kwargs)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ImpactMatrix":
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))


# ──────────────────────────────────────────────────────────────────────
# Phase 3: Scenario Generation
# ──────────────────────────────────────────────────────────────────────

@dataclass
class Scenario:
    """A named scenario with factor activation probabilities."""

    name: str
    description: str
    factor_probs: Dict[str, float]          # factor_name → P(active)
    scenario_prob: float                    # P(this scenario occurring)
    correlations: Dict[str, float] = field(default_factory=dict)
    # "factor1__factor2" → correlation coefficient

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "factor_probs": self.factor_probs,
            "scenario_prob": self.scenario_prob,
            "correlations": self.correlations,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Scenario":
        return cls(
            name=d["name"],
            description=d.get("description", ""),
            factor_probs=d.get("factor_probs", {}),
            scenario_prob=d.get("scenario_prob", 0.2),
            correlations=d.get("correlations", {}),
        )


@dataclass
class ScenarioSet:
    """A weighted set of scenarios that spans plausible futures."""

    scenarios: List[Scenario]
    factor_names: List[str]

    def weighted_probs(self) -> Dict[str, float]:
        """Scenario-weighted factor probabilities.

        ``P(factor) = Σ_s  P(scenario_s) × P(factor | scenario_s)``
        """
        result = {f: 0.0 for f in self.factor_names}
        total_weight = sum(s.scenario_prob for s in self.scenarios)
        if total_weight == 0:
            return result

        for scenario in self.scenarios:
            weight = scenario.scenario_prob / total_weight
            for factor, prob in scenario.factor_probs.items():
                if factor in result:
                    result[factor] += weight * prob
        return result

    def summary(self) -> str:
        lines = ["═══ Scenario Set ═══", ""]

        for s in sorted(self.scenarios, key=lambda x: -x.scenario_prob):
            lines.append(f"┌─ {s.name} (P={s.scenario_prob:.0%})")
            lines.append(f"│  {s.description}")
            for factor, prob in sorted(
                s.factor_probs.items(), key=lambda x: -x[1]
            ):
                bar = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
                lines.append(f"│    {factor:<25} {bar} {prob:.0%}")
            lines.append("└───")
            lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "scenarios": [s.to_dict() for s in self.scenarios],
            "factor_names": self.factor_names,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ScenarioSet":
        return cls(
            scenarios=[Scenario.from_dict(s) for s in d.get("scenarios", [])],
            factor_names=d.get("factor_names", []),
        )

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ScenarioSet":
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))
