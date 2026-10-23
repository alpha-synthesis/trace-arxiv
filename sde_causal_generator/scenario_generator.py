# -*- coding: utf-8 -*-
"""
LLM-Based Scenario Generator.

Uses an LLM to generate plausible future scenarios — combinations of
causal factors with estimated probabilities — closing the loop:

    LLM extracts factors from the past  →  FIN learns their impacts  →
    LLM generates plausible future factor combinations  →  Generator
    produces synthetic data.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import numpy as np

from .data_structures import Scenario, ScenarioSet


# ══════════════════════════════════════════════════════════════════════
# Prompt
# ══════════════════════════════════════════════════════════════════════

_SCENARIO_PROMPT = """\
You are a financial scenario analyst.

CONTEXT:
You have analyzed {ticker} during {historical_period} and identified
these causal factors that affected its price:

{factor_list}

Historical occurrence rates:
{historical_rates}

Learned impact directions:
{impact_directions}

TASK:
Generate {n_scenarios} plausible scenarios for {ticker} over the
NEXT {forward_period}.  Each scenario should represent a distinct
"world state" that could realistically occur.

REQUIREMENTS:
1. Include a "base case" (most likely continuation of current trends).
2. Include at least one "stress scenario" (adverse conditions).
3. Include at least one "opportunity scenario" (favourable conditions).
4. If appropriate, include a "tail risk scenario" (low probability,
   high impact — events NOT seen in the historical data).
5. Scenario probabilities must sum to 1.0.
6. For each scenario, specify the probability that each factor is
   active (0.0 to 1.0).
7. Note any strong correlations between factors within each scenario.

CRITICAL:
- You MAY introduce NEW factors not seen in the historical data for
  tail-risk scenarios (e.g., pandemic, regulatory change, new
  competitor).
- For new factors, estimate their likely impact direction.
- Be specific about WHY each scenario is plausible.

Respond with valid JSON:
{{
  "scenarios": [
    {{
      "name": "short_identifier",
      "description": "One paragraph explaining this scenario",
      "probability": 0.XX,
      "factor_activations": {{
        "factor_name": probability_float
      }},
      "new_factors": [
        {{
          "name": "new_factor_name",
          "activation_probability": 0.XX,
          "estimated_direction": "bullish or bearish",
          "estimated_magnitude": "low or medium or high",
          "reasoning": "why this could happen"
        }}
      ],
      "factor_correlations": {{
        "factor1__factor2": correlation_float
      }}
    }}
  ]
}}

Respond ONLY with valid JSON.
"""


# ══════════════════════════════════════════════════════════════════════
# Generator
# ══════════════════════════════════════════════════════════════════════


class ScenarioGenerator:
    """
    Generates plausible future scenarios via LLM.

    Combines:
        * Historical factor occurrence rates (from data).
        * Learned impact directions (from FIN).
        * LLM world knowledge (novel factors, plausibility).

    Parameters
    ----------
    llm_model : str
        LiteLLM-compatible model identifier.
    temperature : float
        Higher than extraction (0.7) — we *want* creative diversity.
    n_samples : int
        Multiple LLM calls; results are aggregated.
    """

    def __init__(
        self,
        llm_model: str = "openrouter/anthropic/claude-sonnet-4",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        cache_dir: str = "cache/scenarios",
    ):
        self.llm_model = llm_model
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.temperature = temperature
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    # ── public ──────────────────────────────────────────────────────

    def generate(
        self,
        ticker: str,
        factor_names: List[str],
        historical_rates: Dict[str, float],
        impact_directions: Dict[str, str],
        historical_period: str = "2005-2006",
        forward_period: str = "1 year",
        n_scenarios: int = 5,
        n_samples: int = 3,
    ) -> ScenarioSet:
        """Generate a weighted set of plausible scenarios."""
        factor_list = "\n".join(f"  - {n}" for n in factor_names)
        hist_str = "\n".join(
            f"  - {n}: {r:.1%} of days active"
            for n, r in historical_rates.items()
        )
        impact_str = "\n".join(
            f"  - {n}: {d}" for n, d in impact_directions.items()
        )

        prompt = _SCENARIO_PROMPT.format(
            ticker=ticker,
            historical_period=historical_period,
            factor_list=factor_list,
            historical_rates=hist_str,
            impact_directions=impact_str,
            n_scenarios=n_scenarios,
            forward_period=forward_period,
        )

        all_runs: List[List[Scenario]] = []
        try:
            from tqdm import tqdm  # type: ignore[import-untyped]
            pbar = tqdm(
                total=n_samples,
                desc="    Generating scenarios",
                unit="sample",
                ncols=90,
            )
        except ImportError:
            pbar = None

        # Parallel LLM calls for scenario samples
        with ThreadPoolExecutor(max_workers=n_samples) as pool:
            futures = [
                pool.submit(self._call_llm, prompt)
                for _ in range(n_samples)
            ]
            for fut in futures:
                raw = fut.result()
                if pbar is not None:
                    pbar.update(1)
                parsed = self._parse_response(raw, factor_names)
                if parsed:
                    all_runs.append(parsed)

        if pbar is not None:
            pbar.close()

        if not all_runs:
            print(
                "    WARNING: LLM scenario generation failed. "
                "Using empirical fallback."
            )
            return self._empirical_fallback(factor_names, historical_rates)

        return self._aggregate(all_runs, factor_names)

    # ── LLM ─────────────────────────────────────────────────────────

    def _call_llm(self, prompt: str) -> str:
        try:
            import litellm  # type: ignore[import-untyped]

            litellm.suppress_debug_info = True
            resp = litellm.completion(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                api_key=self.api_key,
                max_tokens=4096,
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"    LLM call failed: {e}")
            return ""

    # ── parse ───────────────────────────────────────────────────────

    @staticmethod
    def _parse_response(
        raw: str, known_factors: List[str]
    ) -> Optional[List[Scenario]]:
        try:
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]

            data = json.loads(raw)
            scenarios: List[Scenario] = []

            for s in data.get("scenarios", []):
                factor_probs: Dict[str, float] = {}
                activations = s.get("factor_activations", {})
                for name in known_factors:
                    factor_probs[name] = float(
                        activations.get(name, 0.0)
                    )

                for nf in s.get("new_factors", []):
                    factor_probs[nf["name"]] = float(
                        nf.get("activation_probability", 0.0)
                    )

                correlations = {
                    k: float(v)
                    for k, v in s.get("factor_correlations", {}).items()
                }

                scenarios.append(
                    Scenario(
                        name=s["name"],
                        description=s.get("description", ""),
                        factor_probs=factor_probs,
                        scenario_prob=float(s.get("probability", 0.2)),
                        correlations=correlations,
                    )
                )

            total = sum(sc.scenario_prob for sc in scenarios)
            if total > 0:
                for sc in scenarios:
                    sc.scenario_prob /= total

            return scenarios or None
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"    Parse failed: {e}")
            return None

    # ── aggregate ───────────────────────────────────────────────────

    @staticmethod
    def _aggregate(
        all_runs: List[List[Scenario]], factor_names: List[str]
    ) -> ScenarioSet:
        name_groups: Dict[str, List[Scenario]] = {}
        all_factor_names = set(factor_names)

        for run in all_runs:
            for sc in run:
                key = sc.name.lower().replace(" ", "_")
                name_groups.setdefault(key, []).append(sc)
                all_factor_names.update(sc.factor_probs.keys())

        averaged: List[Scenario] = []
        for name, group in name_groups.items():
            avg_prob = float(np.mean([s.scenario_prob for s in group]))

            all_keys_in_group = set()
            for s in group:
                all_keys_in_group.update(s.factor_probs.keys())

            avg_factors: Dict[str, float] = {}
            for f in all_keys_in_group:
                vals = [s.factor_probs.get(f, 0.0) for s in group]
                avg_factors[f] = float(np.mean(vals))

            best_desc = max((s.description for s in group), key=len)
            averaged.append(
                Scenario(
                    name=name,
                    description=best_desc,
                    factor_probs=avg_factors,
                    scenario_prob=avg_prob,
                )
            )

        total = sum(s.scenario_prob for s in averaged)
        if total > 0:
            for s in averaged:
                s.scenario_prob /= total

        return ScenarioSet(
            scenarios=averaged,
            factor_names=sorted(all_factor_names),
        )

    # ── fallback ────────────────────────────────────────────────────

    @staticmethod
    def _empirical_fallback(
        factor_names: List[str],
        historical_rates: Dict[str, float],
    ) -> ScenarioSet:
        scenarios = [
            Scenario(
                name="historical_base",
                description="Continuation of historical patterns.",
                factor_probs=dict(historical_rates),
                scenario_prob=0.5,
            ),
            Scenario(
                name="stress",
                description="All adverse factors maximised.",
                factor_probs={
                    f: min(1.0, r * 1.5)
                    for f, r in historical_rates.items()
                },
                scenario_prob=0.2,
            ),
            Scenario(
                name="benign",
                description="All adverse factors minimised.",
                factor_probs={
                    f: max(0.0, r * 0.3)
                    for f, r in historical_rates.items()
                },
                scenario_prob=0.3,
            ),
        ]
        return ScenarioSet(
            scenarios=scenarios, factor_names=factor_names
        )
