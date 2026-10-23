# -*- coding: utf-8 -*-
"""
Causal SDE Pipeline Orchestrator.

End-to-end pipeline that chains all three phases:

    Phase 1 — Factor Extraction (LLM + RAG)
    Phase 2 — Impact Learning (FIN training)
    Phase 3 — Generation, Validation & Evaluation

All configuration is driven by a YAML file (see ``configs/causal_sde_config.yaml``).
No CLI entry point — use the generation script::

    python scripts/generation/generate_causal_sde_data.py \\
        --config configs/causal_sde_config.yaml

Or from Python::

    from sde_causal_generator.pipeline import CausalSDEPipeline, PipelineConfig

    cfg = PipelineConfig.from_yaml("configs/causal_sde_config.yaml")
    pipe = CausalSDEPipeline(config=cfg)
    report = pipe.run(df, ticker="AAPL")
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml


# ══════════════════════════════════════════════════════════════════════
# Pipeline configuration
# ══════════════════════════════════════════════════════════════════════


@dataclass
class PipelineConfig:
    """
    All tuneable knobs for the Causal SDE pipeline.

    Override any field via ``PipelineConfig(field=value)`` or pass a
    dict to ``CausalSDEPipeline.run(config={...})``.
    """

    # ── General ─────────────────────────────────────────────────────
    output_dir: str = "results/causal_sde"
    cache_dir: str = "cache/causal_sde"
    seed: int = 42

    # ── LLM ─────────────────────────────────────────────────────────
    llm_model: str = "openrouter/anthropic/claude-sonnet-4"
    llm_api_key: str = ""    # falls back to OPENROUTER_API_KEY env
    llm_temperature: float = 0.1
    llm_n_samples: int = 3   # LLM calls per window (averaged)
    llm_max_workers: int = 4   # concurrent window extractions
    llm_provider: str = "openrouter"   # "openrouter" (litellm) | "ollama"
    ollama_base_url: str = ""          # e.g. "http://host:11434"
    llm_prompt_mode: str = "with_prices"  # "with_prices" | "price_blind" (no OHLC in prompt)
    rag_redact_market_moves: Union[bool, str] = False  # price_blind: False/"off" | True/"legacy" | "moves" | "strict"

    # ── RAG ─────────────────────────────────────────────────────────
    use_rag: bool = True
    rag_persist_dir: str = "cache/rag_store"
    rag_sources: List[str] = field(
        default_factory=lambda: ["wikipedia", "fred"]
    )
    rag_cache_mode: str = "reuse"  # "reuse" | "refresh"
    rag_max_articles: int = 50       # max articles per window for LLM context
    rag_content_limit: int = 500     # max chars per article in LLM prompt
    # Corpus v2 (R2): precomputed, redacted contexts from
    # scripts/precompute_contexts.py. Replaces the vector store (use_rag must
    # be false); the corpus tag enters the LLM cache key.
    rag_context_dir: str = ""
    rag_corpus_tag: str = ""         # must match the contexts' corpus_tag
    llm_call_log: str = ""           # JSONL log of every LLM call attempt
    fred_api_key: str = ""   # falls back to FRED_API_KEY env
    sec_email: str = ""      # falls back to SEC_EMAIL env

    # ── Factor extraction ───────────────────────────────────────────
    window_sizes: List[str] = field(
        default_factory=lambda: ["1ME", "1QE"]
    )
    factor_threshold: float = 0.0  # min magnitude to keep factor
    max_factors: int = 64          # keep top-N factors by occurrence (0 = no limit)
    cross_window_validation: bool = True  # penalise coarse-only factors

    # ── FIN training ────────────────────────────────────────────────
    fin_n_lags: int = 10
    fin_epochs: int = 300
    fin_batch_size: int = 64
    fin_lr: float = 1e-3
    fin_patience: int = 30
    fin_device: str = "auto"
    # Corrections (criteria v2). Defaults keep R0's behaviour.
    fin_impact_estimator: str = "legacy"   # "legacy" | "ablation"
    fin_direction_prior: str = "llm"       # "llm" (DIR-REG on) | "none" (sign learned from data)
    gen_mode: str = ""                     # "" = follow impact-matrix units; "legacy" | "corrected"
    # Reuse saved labels: <dir>/<ticker>/analyses.json. Skips RAG and LLM
    # entirely (no store opened, no network); presence building and pruning
    # are unchanged.
    precomputed_analyses_dir: str = ""

    # ── Scenario generation ─────────────────────────────────────────
    n_scenarios: int = 5
    generate_scenarios: bool = True  # False: skip the scenario LLM call (not used by the factual data)
    scenario_llm_samples: int = 3

    # ── Data generation ─────────────────────────────────────────────
    gen_n_steps: int = 0           # 0 = auto-match real data length
    gen_n_samples: int = 10        # 10 synthetic samples for envelope
    gen_noise_scale: float = 1.0
    gen_start_date: str = ""       # empty = use training start date

    # ── Evaluation ──────────────────────────────────────────────────
    eval_include_deep: bool = True

    # ── Neural SDE (improvement #8) ─────────────────────────────────
    use_neural_sde: bool = False        # enable Neural SDE generator
    nsde_hidden: int = 64               # hidden layer width
    nsde_epochs: int = 200              # training epochs
    nsde_batch_size: int = 256
    nsde_lr: float = 1e-3
    nsde_patience: int = 30            # early stopping patience

    # ── Cross-asset correlations (improvement #9) ───────────────────
    use_cross_asset: bool = False       # imposed automatically when >1 ticker
    cross_asset_method: str = "pearson" # "pearson" or "spearman"

    # ── Factor relevance scoring (F4) ───────────────────────────────
    use_factor_scoring: bool = True         # score & prune factors pre-training
    factor_min_score: float = 0.10          # min composite score to keep

    # ── FIN architecture (F7) ───────────────────────────────────────
    fin_architecture: str = "tft"           # "fin" (shallow) or "tft" (Temporal Fusion Transformer)

    # ── Transfer learning (F5) ──────────────────────────────────────
    use_transfer_learning: bool = True      # warm-start Neural SDE from FIN/TFT
    transfer_scale: float = 0.5             # scale for transferred weights

    # ── DCC-GARCH (F6) ──────────────────────────────────────────────
    use_dcc_garch: bool = True              # DCC-GARCH dynamic correlations
    dcc_garch_p: int = 1
    dcc_garch_q: int = 1

    # ── Publication quality report ──────────────────────────────────
    publish_quality_report: bool = False       # generate llm_quality/ & generation_quality/ folders
    extra_llm_caches: Dict[str, str] = field(default_factory=dict)  # {label: cache_dir}
    disc_epochs: int = 100                     # LSTM discriminator training epochs
    disc_hidden: int = 64                      # LSTM hidden dimension
    disc_window_size: int = 20                 # sliding window for discriminator
    disc_n_folds: int = 5                      # k-fold cross-validation

    # ── Interactive editors ─────────────────────────────────────────
    interactive_post_extraction: bool = False  # Point 1: edit LLM factors before FIN
    interactive_pre_generation: bool = False   # Point 2: edit factors before generation
    interactive_port: int = 8050               # Dash server port

    # ── User-provided factors (YAML) ────────────────────────────────
    user_factors_file: str = ""                 # path to YAML with user-defined factors
    user_factors_mode: str = "merge"            # "merge" | "only" | "disabled"

    _SECRET_FIELDS = ("llm_api_key", "fred_api_key")

    def to_dict(self) -> dict:
        """Serializable view of the config; secrets are masked."""
        d = {}
        for k in self.__dataclass_fields__:
            v = getattr(self, k)
            if k in self._SECRET_FIELDS and v:
                v = "***redacted***"
            d[k] = v
        return d

    def save(self, path: str) -> None:
        """Save config as YAML (or JSON if path ends with .json)."""
        with open(path, "w") as f:
            if path.endswith(".json"):
                json.dump(self.to_dict(), f, indent=2)
            else:
                yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load(cls, path: str) -> "PipelineConfig":
        """Load from YAML or JSON (auto-detected by extension)."""
        with open(path) as f:
            if path.endswith(".json"):
                raw = json.load(f)
            else:
                raw = yaml.safe_load(f) or {}
        return cls._from_flat_or_nested(raw)

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        """Load configuration from a YAML file.

        Handles both flat keys and nested sections (``fin:``,
        ``generation:``, ``evaluation:``, ``interactive:``).
        """
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls._from_flat_or_nested(raw)

    @classmethod
    def _from_flat_or_nested(cls, raw: dict) -> "PipelineConfig":
        """Map a YAML dict (with nested sections) to flat dataclass fields."""
        flat: Dict[str, Any] = {}

        # Nested section mappings -> flat PipelineConfig field names
        _SECTION_MAP = {
            "fin": {
                "n_lags": "fin_n_lags",
                "epochs": "fin_epochs",
                "batch_size": "fin_batch_size",
                "learning_rate": "fin_lr",
                "patience": "fin_patience",
                "device": "fin_device",
                "architecture": "fin_architecture",
                "impact_estimator": "fin_impact_estimator",
                "direction_prior": "fin_direction_prior",
            },
            "generation": {
                "n_steps": "gen_n_steps",
                "n_samples": "gen_n_samples",
                "mode": "gen_mode",
                "noise_scale": "gen_noise_scale",
                "start_date": "gen_start_date",
            },
            "evaluation": {
                "include_deep_metrics": "eval_include_deep",
            },
            "neural_sde": {
                "enabled": "use_neural_sde",
                "hidden": "nsde_hidden",
                "epochs": "nsde_epochs",
                "batch_size": "nsde_batch_size",
                "learning_rate": "nsde_lr",
                "patience": "nsde_patience",
            },
            "cross_asset": {
                "enabled": "use_cross_asset",
                "method": "cross_asset_method",
            },
            "factor_scoring": {
                "enabled": "use_factor_scoring",
                "min_score": "factor_min_score",
            },
            "transfer_learning": {
                "enabled": "use_transfer_learning",
                "scale": "transfer_scale",
            },
            "dcc_garch": {
                "enabled": "use_dcc_garch",
                "garch_p": "dcc_garch_p",
                "garch_q": "dcc_garch_q",
            },
            "interactive": {
                "post_extraction": "interactive_post_extraction",
                "pre_generation": "interactive_pre_generation",
                "port": "interactive_port",
            },
            "quality_report": {
                "enabled": "publish_quality_report",
                "extra_llm_caches": "extra_llm_caches",
                "disc_epochs": "disc_epochs",
                "disc_hidden": "disc_hidden",
                "disc_window_size": "disc_window_size",
                "disc_n_folds": "disc_n_folds",
            },
            "user_factors": {
                "file": "user_factors_file",
                "mode": "user_factors_mode",
            },
        }

        for key, value in raw.items():
            if key in _SECTION_MAP and isinstance(value, dict):
                for sub_key, sub_val in value.items():
                    flat_key = _SECTION_MAP[key].get(sub_key)
                    if flat_key:
                        flat[flat_key] = sub_val
            elif key in cls.__dataclass_fields__:
                flat[key] = value
            # Silently ignore unknown keys (tickers, train_start, etc.
            # are handled by the generation script, not pipeline config).

        return cls(**flat)


# ══════════════════════════════════════════════════════════════════════
# Pipeline
# ══════════════════════════════════════════════════════════════════════


class CausalSDEPipeline:
    """
    Orchestrates the complete Causal SDE data generation pipeline.

    Parameters
    ----------
    config : PipelineConfig | dict | None
        Pipeline configuration.  Accepts a ``PipelineConfig`` instance,
        a kwargs dict, or ``None`` (defaults).
    """

    def __init__(
        self,
        config: Optional[PipelineConfig | dict] = None,
    ):
        if config is None:
            self.cfg = PipelineConfig()
        elif isinstance(config, dict):
            self.cfg = PipelineConfig._from_flat_or_nested(config)
        else:
            self.cfg = config

        self._setup_dirs()

    # ── directory setup ─────────────────────────────────────────────

    def _setup_dirs(self):
        for d in [
            self.cfg.output_dir,
            self.cfg.cache_dir,
            os.path.join(self.cfg.output_dir, "plots"),
        ]:
            os.makedirs(d, exist_ok=True)

    # ── FULL RUN ────────────────────────────────────────────────────

    def run(
        self,
        df: pd.DataFrame,
        ticker: str,
        interactive: bool = False,
        test_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Execute the complete pipeline.

        Parameters
        ----------
        df : pd.DataFrame
            Historical price data with columns:
            ``date, open, high, low, close, volume``.
        ticker : str
            Stock ticker symbol.
        interactive : bool
            Legacy flag. When True, activates post_extraction editor.
            Prefer setting ``interactive_post_extraction`` /
            ``interactive_pre_generation`` in the YAML config.

        Returns
        -------
        report : dict
            Full pipeline report with timing, metrics, and paths.
        """
        # Legacy compat: if interactive=True passed directly, honour it
        if interactive and not self.cfg.interactive_post_extraction:
            self.cfg.interactive_post_extraction = True
        np.random.seed(self.cfg.seed)
        # v2.1: seed torch as well (the TFT was previously unseeded, so every
        # run was a different draw) and request deterministic cuDNN kernels.
        import random as _random
        import torch as _torch
        _random.seed(self.cfg.seed)
        _torch.manual_seed(self.cfg.seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(self.cfg.seed)
        _torch.backends.cudnn.deterministic = True
        _torch.backends.cudnn.benchmark = False
        if getattr(self.cfg, "fin_direction_prior", "llm") not in ("llm", "none"):
            raise ValueError(f"fin.direction_prior must be 'llm' or 'none', got "
                             f"{self.cfg.fin_direction_prior!r}")
        if getattr(self.cfg, "fin_impact_estimator", "legacy") not in ("legacy", "ablation"):
            raise ValueError(f"fin.impact_estimator must be 'legacy' or 'ablation', got "
                             f"{self.cfg.fin_impact_estimator!r}")
        report: Dict[str, Any] = {
            "ticker": ticker,
            "config": self.cfg.to_dict(),
            "phases": {},
        }

        t0 = time.time()

        # ── Load user-provided factors (if any) ────────────────────
        user_factors_events = self._load_user_factors()
        skip_llm = (
            self.cfg.user_factors_mode == "only"
            and len(user_factors_events) > 0
        )

        # ── Phase 1: Factor Extraction ──────────────────────────────
        print("\n" + "═" * 60)
        if skip_llm:
            print("  PHASE 1 — User-Provided Factors (LLM skipped)")
        else:
            print("  PHASE 1 — Factor Extraction (LLM + RAG)")
        print("═" * 60)

        t1 = time.time()
        if skip_llm:
            # Build presence matrix from user factors only
            from .event_validator import (
                UserEvent, events_to_presence_matrix,
            )
            dates = pd.to_datetime(sorted(df["date"].unique()))
            presence, factor_names = events_to_presence_matrix(
                user_factors_events, dates,
            )
            analyses = []  # no LLM analyses
            self._llm_consistency_stats = []
            print(f"    Using {len(user_factors_events)} user-provided factors "
                  f"(LLM extraction skipped)")
        else:
            presence, factor_names, dates, analyses = self._phase1_extract(
                df, ticker
            )
            # Merge user factors into LLM results
            if (self.cfg.user_factors_mode == "merge"
                    and len(user_factors_events) > 0):
                from .event_validator import merge_llm_and_user_events
                n_before = len(factor_names)
                presence, factor_names = merge_llm_and_user_events(
                    presence, factor_names, user_factors_events, dates,
                )
                n_added = len(factor_names) - n_before
                print(f"    Merged {len(user_factors_events)} user factors "
                      f"({n_added} new, "
                      f"{len(user_factors_events) - n_added} overridden)")

        dt1 = time.time() - t1

        report["phases"]["extraction"] = {
            "n_factors": len(factor_names),
            "factor_names": factor_names,
            "presence_shape": list(presence.shape),
            "n_windows_analysed": len(analyses),
            "time_seconds": round(dt1, 1),
            "llm_consistency": self._get_consistency_summary(
                getattr(self, '_llm_consistency_stats', [])
            ),
        }
        print(f"\n  ✓ Phase 1 complete: {len(factor_names)} factors, "
              f"{presence.shape[0]} days, {dt1:.1f}s")

        # ── Interactive Validation (Point 1) ────────────────────────
        validated_events_dicts = None
        if self.cfg.interactive_post_extraction:
            presence, factor_names, validated_events_dicts = (
                self._interactive_validation(
                    df, ticker, presence, factor_names, dates, analyses,
                    user_factors_events=user_factors_events,
                )
            )

        # ── F4: Factor Relevance Scoring ────────────────────────────
        if self.cfg.use_factor_scoring:
            print("\n  Scoring factor relevance (F4)...")
            from .factor_relevance import (
                score_factors,
                prune_irrelevant_factors,
                print_relevance_report,
            )

            # Compute log-returns for scoring
            feature_cols = [c for c in ["open", "high", "low", "close", "volume"]
                           if c in df.columns]
            df_sorted = df.sort_values("date")
            _prices = df_sorted[feature_cols].values.astype(np.float64)
            _prices = np.clip(_prices, 1e-8, None)
            _log_ret = np.diff(np.log(_prices), axis=0)

            n_pre = len(factor_names)
            factor_scores = score_factors(
                presence, _log_ret, factor_names,
            )
            print_relevance_report(factor_scores)

            presence, factor_names = prune_irrelevant_factors(
                presence, factor_names, factor_scores,
                min_score=self.cfg.factor_min_score,
                max_factors=self.cfg.max_factors,
            )
            print(f"    Factors after scoring: {n_pre} → {len(factor_names)}")
            report["phases"]["factor_scoring"] = {
                "scores": {k: v["composite"] for k, v in factor_scores.items()},
                "n_before": n_pre,
                "n_after": len(factor_names),
            }

        # ── Phase 2: Impact Learning ────────────────────────────────
        print("\n" + "═" * 60)
        print("  PHASE 2 — Factor Impact Network Training")
        print("═" * 60)

        # DIR-REG: compute average LLM direction per factor from the
        # extraction analyses.  Used to regularise FIN training and
        # correct sign conflicts in the extracted impact matrix.
        llm_directions = self._compute_llm_directions(
            analyses, factor_names,
        )
        n_directional = int((np.abs(llm_directions) > 0.1).sum())
        print(f"  Direction prior: {n_directional}/{len(factor_names)} "
              f"factors have LLM directional signal")
        if getattr(self.cfg, "fin_direction_prior", "llm") == "none":
            # data-sign variant: LLM directions never enter training or extraction
            print("  Direction prior DISABLED (fin.direction_prior: none)")
            llm_directions = None

        t2 = time.time()
        impact_matrix, fin_model, train_history = self._phase2_train(
            df, presence, factor_names, llm_directions=llm_directions,
        )
        dt2 = time.time() - t2

        report["phases"]["impact_learning"] = {
            "train_loss_final": train_history["train_loss"][-1] if train_history["train_loss"] else None,
            "val_loss_final": train_history["val_loss"][-1] if train_history["val_loss"] else None,
            "best_epoch": train_history.get("best_epoch", -1),
            "time_seconds": round(dt2, 1),
        }
        print(f"\n  ✓ Phase 2 complete: {dt2:.1f}s")

        # Impact matrix summary
        print(f"\n{impact_matrix.summary()}")

        # ── Interactive Generation Editor (Point 2) ─────────────────
        generation_factor_configs = None
        if self.cfg.interactive_pre_generation:
            presence, factor_names, generation_factor_configs = (
                self._interactive_generation_edit(
                    df, ticker, impact_matrix, presence, factor_names, dates,
                    validated_events=validated_events_dicts,
                )
            )

        # Build generation factor configs from presence matrix if not
        # already set by the interactive editor (so the box PNG is
        # always produced with the actual factors used for generation).
        if generation_factor_configs is None and len(factor_names) > 0:
            # Build direction map from user-validated events
            _val_dirs: Dict[str, str] = {}
            if validated_events_dicts:
                _val_dirs = {
                    e["name"]: e.get("direction", "neutral")
                    for e in validated_events_dicts
                }

            generation_factor_configs = []
            for k, fname in enumerate(factor_names):
                active_rows = np.where(presence[:, k] > 0)[0]
                if len(active_rows) == 0:
                    continue
                start_d = str(dates[active_rows[0]])[:10]
                end_d = str(dates[active_rows[-1]])[:10]

                # Priority: user-validated direction > FIN close impact
                if fname in _val_dirs:
                    direction = _val_dirs[fname]
                else:
                    n_features = impact_matrix.base_impact.shape[1]
                    close_idx = min(3, n_features - 1)
                    close_impact = float(
                        impact_matrix.base_impact[k, close_idx]
                    )
                    direction = "bearish" if close_impact < 0 else "bullish"

                prob = float(
                    impact_matrix.occurrence_prob[k]
                    if impact_matrix.occurrence_prob is not None
                    else 1.0
                )
                generation_factor_configs.append({
                    "name": fname,
                    "start_date": start_d,
                    "end_date": end_d,
                    "direction": direction,
                    "probability": prob,
                    "active": True,
                })

        # ── Phase 3: Generation + Scenarios ─────────────────────────
        print("\n" + "═" * 60)
        print("  PHASE 3 — Scenario Generation & Synthetic Data")
        print("═" * 60)

        t3 = time.time()
        scenario_set, synth_df = self._phase3_generate(
            df, ticker, impact_matrix, factor_names, presence,
            fin_model=fin_model,
        )
        dt3 = time.time() - t3

        report["phases"]["generation"] = {
            "n_scenarios": len(scenario_set.scenarios),
            "n_samples": self.cfg.gen_n_samples,
            "n_steps": self.cfg.gen_n_steps,
            "synth_rows": len(synth_df),
            "time_seconds": round(dt3, 1),
        }
        print(f"\n  ✓ Phase 3 complete: {len(synth_df)} rows, {dt3:.1f}s")

        # ── Evaluation ──────────────────────────────────────────────
        print("\n" + "═" * 60)
        print("  EVALUATION — Quality Assessment")
        print("═" * 60)

        t4 = time.time()
        eval_report = self._evaluate(df, synth_df)
        dt4 = time.time() - t4

        report["phases"]["evaluation"] = {
            "metrics": eval_report.get("_aggregate", {}),
            "time_seconds": round(dt4, 1),
        }

        # ── OOS Validation (#10) ───────────────────────────────────
        if test_df is not None and len(test_df) > 10:
            print("\n" + "═" * 60)
            print("  OUT-OF-SAMPLE VALIDATION (#10)")
            print("═" * 60)

            from .oos_validation import oos_validate, plot_oos_comparison, print_oos_report

            t5 = time.time()
            oos_report = oos_validate(
                synth_df=synth_df,
                test_df=test_df,
                train_df=df,
            )
            dt5 = time.time() - t5
            print_oos_report(oos_report, ticker=ticker)

            plot_dir = os.path.join(self.cfg.output_dir, "plots")
            plot_oos_comparison(
                synth_df=synth_df,
                test_df=test_df,
                train_df=df,
                ticker=ticker,
                output_dir=plot_dir,
            )

            report["phases"]["oos_validation"] = {
                "metrics": oos_report,
                "time_seconds": round(dt5, 1),
            }

        # ── Save artefacts ──────────────────────────────────────────
        total = time.time() - t0
        report["total_time_seconds"] = round(total, 1)

        self._save_artefacts(
            report, impact_matrix, scenario_set, synth_df, df, ticker
        )

        # ── Publication quality report ──────────────────────────
        if self.cfg.publish_quality_report:
            print("\n" + "═" * 60)
            print("  PUBLICATION QUALITY REPORT")
            print("═" * 60)

            from .quality_report import QualityReportGenerator

            eval_output = os.path.join(self.cfg.output_dir, "evaluation")
            qr = QualityReportGenerator(output_dir=eval_output)

            # Build factor_cache_dirs
            factor_cache_dirs = {}
            current_cache = os.path.join(self.cfg.cache_dir, ticker, "llm_factors")
            if os.path.isdir(current_cache):
                model_label = self.cfg.llm_model.split("/")[-1]
                factor_cache_dirs[model_label] = current_cache
            # Add all extra caches from config
            for ref_label, ref_path in self.cfg.extra_llm_caches.items():
                if os.path.isdir(ref_path):
                    factor_cache_dirs[ref_label] = ref_path
                else:
                    print(f"    ⚠ Extra cache not found: {ref_label} -> {ref_path}")

            try:
                qr.run(
                    real_df=df,
                    synth_df=synth_df,
                    ticker=ticker,
                    factor_cache_dirs=factor_cache_dirs or None,
                    disc_epochs=self.cfg.disc_epochs,
                    disc_hidden=self.cfg.disc_hidden,
                    disc_window_size=self.cfg.disc_window_size,
                    disc_n_folds=self.cfg.disc_n_folds,
                )
            except Exception as e:
                print(f"    ⚠ Quality report failed: {e}")

        # ── Comparison plots with causal annotations ────────────────
        print("\n  Generating comparison plots...")
        from .evaluate_data import plot_causal_comparison

        plot_dir = os.path.join(self.cfg.output_dir, "plots")
        analyses_dicts = [a.to_dict() for a in analyses]
        try:
            plot_causal_comparison(
                real_df=df,
                synth_df=synth_df,
                analyses=analyses_dicts,
                ticker=ticker,
                output_dir=plot_dir,
                validated_events=validated_events_dicts,
                generation_factors=generation_factor_configs,
            )
        except Exception as e:
            print(f"    ⚠ Plot generation failed: {e}")

        print(f"\n{'═' * 60}")
        print(f"  PIPELINE COMPLETE — {total:.1f}s total")
        print(f"  Output: {self.cfg.output_dir}/")
        print(f"{'═' * 60}\n")

        return report

    # ══════════════════════════════════════════════════════════════════
    # Phase implementations
    # ══════════════════════════════════════════════════════════════════

    def _phase1_extract(
        self,
        df: pd.DataFrame,
        ticker: str,
    ) -> Tuple[np.ndarray, List[str], pd.DatetimeIndex, list]:
        """
        Phase 1: LLM-based factor extraction with optional RAG.

        Returns (presence_matrix, factor_names, dates, analyses).
        """
        from .news_rag import NewsRAGManager
        from .factor_extractor import LLMFactorExtractor

        pre_dir = getattr(self.cfg, "precomputed_analyses_dir", "")
        ctx_dir = getattr(self.cfg, "rag_context_dir", "")
        if ctx_dir and self.cfg.use_rag:
            raise ValueError("rag_context_dir replaces the RAG store: set use_rag: false")
        # ── optional RAG ────────────────────────────────────────────
        rag = None
        if self.cfg.use_rag and not pre_dir:
            print("\n  Building news corpus (RAG)...")
            rag = NewsRAGManager(persist_dir=self.cfg.rag_persist_dir)
            dates_list = pd.to_datetime(df["date"])
            start = dates_list.min().strftime("%Y-%m-%d")
            end = dates_list.max().strftime("%Y-%m-%d")
            try:
                rag.build_corpus(
                    ticker=ticker,
                    start_date=start,
                    end_date=end,
                    sources=self.cfg.rag_sources,
                    fred_api_key=self.cfg.fred_api_key or None,
                    cache_mode=self.cfg.rag_cache_mode,
                )
                print(f"    RAG stats: {rag.store.get_stats()}")
            except Exception as e:
                print(f"    ⚠ RAG build failed: {e}. Continuing without RAG.")
                rag = None

        # ── factor extraction ───────────────────────────────────────
        context_provider = None
        if ctx_dir and not pre_dir:
            from .factor_extractor import PrecomputedContexts
            context_provider = PrecomputedContexts(
                ctx_dir, expected_corpus_tag=getattr(self.cfg, "rag_corpus_tag", "") or None)
            print(f"\n  Precomputed contexts: {ctx_dir} (corpus {context_provider.corpus_tag})")
        extractor = LLMFactorExtractor(
            llm_model=self.cfg.llm_model,
            api_key=self.cfg.llm_api_key or None,
            cache_dir=os.path.join(self.cfg.cache_dir, "llm_factors"),
            temperature=self.cfg.llm_temperature,
            n_samples=self.cfg.llm_n_samples,
            news_rag=rag,
            max_articles=self.cfg.rag_max_articles,
            llm_provider=getattr(self.cfg, "llm_provider", "openrouter"),
            ollama_base_url=getattr(self.cfg, "ollama_base_url", ""),
            prompt_mode=getattr(self.cfg, "llm_prompt_mode", "with_prices"),
            redact_market_moves=getattr(self.cfg, "rag_redact_market_moves", False),
            context_provider=context_provider,
            call_log_path=getattr(self.cfg, "llm_call_log", "") or None,
        )

        if pre_dir:
            import json as _json
            from .data_structures import WindowAnalysis
            src = os.path.join(pre_dir, ticker, "analyses.json")
            with open(src) as fh:
                analyses = [WindowAnalysis.from_dict(d) for d in _json.load(fh)]
            print(f"\n  Loaded {len(analyses)} saved window analyses from {src} "
                  f"(no RAG, no LLM)")
        else:
            print(f"\n  Extracting causal factors (windows: {self.cfg.window_sizes})...")
            try:
                analyses = extractor.extract_all_windows(
                    df, ticker,
                    window_sizes=self.cfg.window_sizes,
                    max_workers=self.cfg.llm_max_workers,
                )
            finally:
                extractor.close()   # long-lived HTTP client (litellm fd-reuse fix)
            print(f"    Analysed {len(analyses)} windows "
                  f"({extractor.n_llm_calls} LLM calls sent)")

        # ── build presence matrix ───────────────────────────────────
        presence, factor_names, dates = extractor.build_presence_matrix(
            analyses, df, threshold=self.cfg.factor_threshold
        )
        print(f"    Presence matrix: {presence.shape}")
        print(f"    Factors found: {len(factor_names)}")

        # ── prune to top-N factors by hybrid ranking ──────────────────
        # DISC-2: use freq × max_magnitude instead of pure occurrence
        # count, so rare-but-intense events are protected from pruning.
        max_k = self.cfg.max_factors
        if max_k > 0 and len(factor_names) > max_k:
            occurrence = presence.sum(axis=0)            # (K,)
            max_magnitude = presence.max(axis=0)         # (K,)
            # Hybrid score: combines frequency and peak intensity
            hybrid_score = occurrence * max_magnitude    # (K,)

            # DISC-2: tail-event protection — any factor with peak
            # magnitude above 0.7 is kept regardless of frequency.
            tail_mask = max_magnitude >= 0.7
            n_tail = int(tail_mask.sum())

            # Reserve slots for tail-events, fill rest by hybrid score
            non_tail_budget = max(max_k - n_tail, 1)
            non_tail_idx = np.where(~tail_mask)[0]
            non_tail_scores = hybrid_score[non_tail_idx]
            top_non_tail = non_tail_idx[
                np.argsort(non_tail_scores)[::-1][:non_tail_budget]
            ]

            top_idx = np.sort(np.concatenate([
                np.where(tail_mask)[0], top_non_tail
            ]))[:max_k]

            presence = presence[:, top_idx]
            factor_names = [factor_names[i] for i in top_idx]
            print(f"    Pruned to top {len(top_idx)} factors "
                  f"(hybrid ranking, {n_tail} tail-events protected)")
            print(f"    Presence matrix after pruning: {presence.shape}")

        # ── cross-window validation ─────────────────────────────────
        # If multiple window sizes are used (e.g. 2W, 1ME, 1QE), check
        # consistency: a factor in a quarterly window should also appear
        # in at least one of its constituent monthly/biweekly windows.
        if self.cfg.cross_window_validation and len(self.cfg.window_sizes) > 1:
            presence, factor_names = self._cross_window_validate(
                analyses, presence, factor_names, dates,
            )

        # Save analyses cache
        analyses_path = os.path.join(self.cfg.output_dir, "analyses.json")
        with open(analyses_path, "w") as f:
            json.dump([a.to_dict() for a in analyses], f, indent=2)

        # Store LLM consistency stats for the report
        self._llm_consistency_stats = extractor._consistency_stats

        return presence, factor_names, dates, analyses

    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _get_consistency_summary(stats: List[Dict]) -> Dict:
        """Aggregate LLM consistency stats into a report-friendly dict."""
        if not stats:
            return {}
        parse_rates = [s["parse_rate"] for s in stats]
        jaccards = [s["jaccard_similarity"] for s in stats]
        dir_agrees = [s["direction_agreement"] for s in stats]
        mag_cvs = [s["magnitude_cv"] for s in stats]
        return {
            "n_windows": len(stats),
            "parse_success_rate": round(float(np.mean(parse_rates)), 4),
            "avg_jaccard_similarity": round(float(np.mean(jaccards)), 4),
            "avg_direction_agreement": round(float(np.mean(dir_agrees)), 4),
            "avg_magnitude_cv": round(float(np.mean(mag_cvs)), 4),
        }

    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_llm_directions(
        analyses: List,
        factor_names: List[str],
    ) -> np.ndarray:
        """Compute average LLM direction per factor from window analyses.

        For each factor in *factor_names*, averages the ``direction``
        field across all :class:`WindowAnalysis` windows where that
        factor was extracted.  Factors not found in any window get 0.

        Returns
        -------
        directions : ndarray, shape (K,)
            Values in [-1, +1].  Positive = bullish, negative = bearish.
        """
        K = len(factor_names)
        name_to_idx = {n: i for i, n in enumerate(factor_names)}
        sums = np.zeros(K, dtype=np.float64)
        counts = np.zeros(K, dtype=np.float64)
        for analysis in analyses:
            for f in analysis.factors:
                if f.name in name_to_idx:
                    idx = name_to_idx[f.name]
                    sums[idx] += f.direction
                    counts[idx] += 1
        with np.errstate(divide="ignore", invalid="ignore"):
            directions = np.where(counts > 0, sums / counts, 0.0)
        return directions.astype(np.float32)

    # ─────────────────────────────────────────────────────────────────

    def _load_user_factors(self) -> list:
        """Load user-provided factors from a YAML or JSON file.

        **YAML** format::

            factors:
              - name: "federal_reserve_rate_hike"
                start_date: "2022-03-15"
                end_date: "2023-07-26"
                direction: "bearish"
                probability: 0.9
                category: "macroeconomic"
                description: "Fed aggressive rate hiking cycle"

        **JSON** format (same structure — also accepts a bare list)::

            {"factors": [{"name": "...", ...}]}
            # or simply:
            [{"name": "...", ...}]

        The format is auto-detected by file extension (``.yaml`` /
        ``.yml`` → YAML, ``.json`` → JSON).

        Returns a list of UserEvent objects (possibly empty).
        """
        if self.cfg.user_factors_mode == "disabled":
            return []

        path = self.cfg.user_factors_file
        if not path or not os.path.exists(path):
            if path and self.cfg.user_factors_mode == "only":
                print(f"  ⚠ user_factors_file '{path}' not found — "
                      f"falling back to LLM extraction.")
            return []

        from .event_validator import UserEvent

        with open(path) as f:
            if path.endswith(".json"):
                raw = json.load(f)
            else:
                raw = yaml.safe_load(f) or {}

        # Accept both {"factors": [...]} and a bare list [...]
        if isinstance(raw, list):
            factors_list = raw
        else:
            factors_list = raw.get("factors", [])

        if not factors_list:
            print(f"  ⚠ No factors found in '{path}'.")
            return []

        events = []
        for fc in factors_list:
            try:
                events.append(UserEvent(
                    name=fc["name"],
                    start_date=str(fc["start_date"]),
                    end_date=str(fc["end_date"]),
                    direction=fc.get("direction", "neutral"),
                    probability=float(fc.get("probability", 1.0)),
                    category=fc.get("category", "macroeconomic"),
                    description=fc.get("description", ""),
                    source="user_yaml",
                ))
            except (KeyError, ValueError) as e:
                print(f"  ⚠ Skipping malformed factor: {e}")

        print(f"  ✓ Loaded {len(events)} user-provided factors from '{path}'")
        return events

    def _cross_window_validate(
        self,
        analyses: list,
        presence: np.ndarray,
        factor_names: List[str],
        dates: pd.DatetimeIndex,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Cross-validate factors across overlapping window sizes.

        For each factor that only appears in large windows (quarterly)
        but never in any of its constituent smaller windows (monthly /
        bi-weekly), reduce its presence weight or drop it entirely.

        This catches LLM "hallucinations" that name a broad macro factor
        for a quarter but none of the shorter windows within that
        quarter mention it — a strong signal of confabulation.

        Returns
        -------
        presence : ndarray (T, K')
            Possibly pruned presence matrix.
        factor_names : list[str]
            Possibly pruned factor name list.
        """
        from collections import defaultdict

        # Group analyses by window type
        by_type: Dict[str, list] = defaultdict(list)
        for a in analyses:
            wtype = a.window_type if hasattr(a, "window_type") else "unknown"
            by_type[wtype].append(a)

        # Only validate when we have both fine and coarse windows
        fine_types = {"daily", "monthly"}
        coarse_types = {"quarterly"}
        has_fine = bool(fine_types & set(by_type.keys()))
        has_coarse = bool(coarse_types & set(by_type.keys()))

        if not (has_fine and has_coarse):
            return presence, factor_names

        # Build factor sets per type
        fine_factors: set = set()
        for wtype in fine_types:
            for a in by_type.get(wtype, []):
                for f in a.factors:
                    fine_factors.add(f.name)

        coarse_only: set = set()
        for wtype in coarse_types:
            for a in by_type.get(wtype, []):
                for f in a.factors:
                    if f.name not in fine_factors:
                        coarse_only.add(f.name)

        if not coarse_only:
            print("    Cross-window validation: all factors consistent ✓")
            return presence, factor_names

        # Penalise coarse-only factors: halve their presence
        # (rather than hard-drop, to be conservative)
        penalised = 0
        for fname in coarse_only:
            if fname in factor_names:
                idx = factor_names.index(fname)
                presence[:, idx] *= 0.5
                penalised += 1

        print(
            f"    Cross-window validation: "
            f"{penalised} factors penalised (coarse-only), "
            f"{len(fine_factors)} confirmed by fine windows ✓"
        )

        # Drop factors whose total presence fell below 1 day
        keep_mask = presence.sum(axis=0) >= 1.0
        if not keep_mask.all():
            n_drop = int((~keep_mask).sum())
            presence = presence[:, keep_mask]
            factor_names = [
                n for n, k in zip(factor_names, keep_mask) if k
            ]
            print(f"    Dropped {n_drop} factors with near-zero presence")

        return presence, factor_names

    # ─────────────────────────────────────────────────────────────────

    def _interactive_validation(
        self,
        df: pd.DataFrame,
        ticker: str,
        presence: np.ndarray,
        factor_names: List[str],
        dates: pd.DatetimeIndex,
        analyses: list,
        user_factors_events: Optional[list] = None,
    ) -> Tuple[np.ndarray, List[str], Optional[List[dict]]]:
        """
        Launch the Dash UI for interactive event validation.

        Returns the (possibly modified) presence matrix, factor names,
        and the list of validated event dicts (or None if not exported).
        """
        from .event_validator import (
            UserEvent,
            create_dash_app,
            merge_llm_and_user_events,
            deduplicate_events,
        )
        from .data_structures import CausalFactor

        # Convert LLM analyses to initial UI events (tagged as LLM source)
        initial_events = []
        for a in analyses:
            for f in a.factors:
                initial_events.append(
                    UserEvent(
                        name=f.name,
                        start_date=a.start_date,
                        end_date=a.end_date,
                        direction={
                            True: "bullish",
                            False: "bearish",
                        }.get(f.direction > 0, "neutral"),
                        probability=f.magnitude,
                        category=f.category,
                        description=f.description,
                        source="llm",
                    )
                )

        # Deduplicate: merge same-name factors across windows, filter noise
        n_raw = len(initial_events)
        initial_events = deduplicate_events(
            initial_events,
            min_probability=0.08,
            max_events=0,
        )
        print(f"  Events: {n_raw} raw -> {len(initial_events)} after dedup")

        # Append user-provided YAML factors (tagged as "user_yaml")
        if user_factors_events:
            for uf in user_factors_events:
                # Skip if already present (by name) from LLM
                existing_names = {e.name for e in initial_events}
                if uf.name not in existing_names:
                    initial_events.append(uf)
                else:
                    # User YAML overrides LLM event with same name
                    initial_events = [
                        e for e in initial_events if e.name != uf.name
                    ]
                    initial_events.append(uf)
            print(f"  Events after user YAML merge: {len(initial_events)}")

        port = self.cfg.interactive_port
        export_dir = os.path.join(self.cfg.output_dir, "validated_events")

        print("\n  ┌─────────────────────────────────────────────────┐")
        print(f"  │  INTERACTIVE MODE — Point 1 (Post-Extraction)   │")
        print(f"  │  Opening Event Editor at http://localhost:{port}  │")
        print("  │  Add / edit / delete events on the chart.        │")
        print("  │  Click 'Export & Continue' when done, then       │")
        print("  │  close the browser or press Ctrl+C to continue.  │")
        print("  └─────────────────────────────────────────────────┘\n")

        app = create_dash_app(
            df=df,
            ticker=ticker,
            initial_events=initial_events,
            output_dir=export_dir,
            port=port,
        )

        try:
            app.run(debug=False, port=port)
        except KeyboardInterrupt:
            print("\n  Event editor closed.")

        # Check if user exported events
        events_file = os.path.join(export_dir, f"{ticker}_events.json")
        validated_events_dicts = None
        if os.path.exists(events_file):
            with open(events_file) as f:
                validated_events_dicts = json.load(f)
            user_events = [UserEvent.from_dict(d) for d in validated_events_dicts]
            print(f"  ✓ Loaded {len(user_events)} validated events")
            presence, factor_names = merge_llm_and_user_events(
                presence, factor_names, user_events, dates
            )
        else:
            print("  ℹ No exported events found — using LLM-only factors.")

        return presence, factor_names, validated_events_dicts

    # ─────────────────────────────────────────────────────────────────

    def _interactive_generation_edit(
        self,
        df: pd.DataFrame,
        ticker: str,
        impact_matrix,
        presence: np.ndarray,
        factor_names: List[str],
        dates: pd.DatetimeIndex,
        validated_events: Optional[List[dict]] = None,
    ) -> Tuple[np.ndarray, List[str], Optional[List[dict]]]:
        """
        Launch the Dash UI for pre-generation factor editing (Point 2).

        The user can toggle factors on/off, adjust probabilities,
        and move temporal windows before generation starts.

        Returns the (possibly modified) presence matrix, factor names,
        and generation factor configs (or None if not exported).
        """
        from .generation_editor import (
            create_generation_editor_app,
            _build_edited_presence,
        )

        port = self.cfg.interactive_port + 1  # Point 2 uses port+1
        export_dir = os.path.join(self.cfg.output_dir, "generation_config")

        print("\n  ┌─────────────────────────────────────────────────┐")
        print(f"  │  INTERACTIVE MODE — Point 2 (Pre-Generation)    │")
        print(f"  │  Opening Factor Editor at http://localhost:{port}  │")
        print("  │  Toggle / adjust / reposition trained factors.   │")
        print("  │  Click 'Export & Continue' when done, then       │")
        print("  │  close the browser or press Ctrl+C to continue.  │")
        print("  └─────────────────────────────────────────────────┘\n")

        # Build direction map from user-validated events (Point 1)
        validated_directions: Optional[Dict[str, str]] = None
        if validated_events:
            validated_directions = {
                e["name"]: e.get("direction", "neutral")
                for e in validated_events
            }

        app = create_generation_editor_app(
            df=df,
            ticker=ticker,
            impact_matrix=impact_matrix,
            historical_presence=presence,
            output_dir=export_dir,
            port=port,
            validated_directions=validated_directions,
        )

        try:
            app.run(debug=False, port=port)
        except KeyboardInterrupt:
            print("\n  Generation editor closed.")

        # Check if user exported factor config
        config_file = os.path.join(
            export_dir, f"{ticker}_generation_config.json",
        )
        generation_factor_configs = None
        if os.path.exists(config_file):
            import json as _json
            with open(config_file) as f:
                factor_configs = _json.load(f)
            active = [fc for fc in factor_configs if fc.get("active", True)]
            generation_factor_configs = active
            print(f"  ✓ Loaded {len(active)}/{len(factor_configs)} "
                  f"active factors from generation editor")
            presence, factor_names = _build_edited_presence(
                active, dates,
            )
        else:
            print("  ℹ No generation config exported — using "
                  "original trained factors.")

        return presence, factor_names, generation_factor_configs

    # ─────────────────────────────────────────────────────────────────

    def _phase2_train(
        self,
        df: pd.DataFrame,
        presence: np.ndarray,
        factor_names: List[str],
        llm_directions: Optional[np.ndarray] = None,
    ) -> Tuple[Any, Any, Dict]:
        """Phase 2: Train the Factor Impact Network."""
        from .factor_impact_network import train_impact_network

        # Build price matrix (T × M OHLCV)
        feature_cols = ["open", "high", "low", "close", "volume"]
        available = [c for c in feature_cols if c in df.columns]
        if not available:
            raise ValueError(
                f"DataFrame must have at least one of {feature_cols}."
            )

        dates_sorted = pd.to_datetime(df["date"]).sort_values()
        df_sorted = df.set_index(pd.to_datetime(df["date"])).sort_index()
        price_data = df_sorted[available].values.astype(np.float32)

        # Align sizes (presence and price must have same T)
        min_T = min(presence.shape[0], price_data.shape[0])
        presence = presence[:min_T]
        price_data = price_data[:min_T]

        model, impact_matrix, history = train_impact_network(
            factor_matrix=presence,
            price_data=price_data,
            factor_names=factor_names,
            feature_names=available,
            n_lags=self.cfg.fin_n_lags,
            n_epochs=self.cfg.fin_epochs,
            batch_size=self.cfg.fin_batch_size,
            learning_rate=self.cfg.fin_lr,
            val_fraction=0.15,
            device=self.cfg.fin_device,
            patience=self.cfg.fin_patience,
            architecture=self.cfg.fin_architecture,
            llm_directions=llm_directions,
            impact_estimator=getattr(self.cfg, "fin_impact_estimator", "legacy"),
        )

        # Save
        impact_matrix.save(
            os.path.join(self.cfg.output_dir, "impact_matrix.json")
        )

        return impact_matrix, model, history

    # ─────────────────────────────────────────────────────────────────

    def _phase3_generate(
        self,
        df: pd.DataFrame,
        ticker: str,
        impact_matrix: Any,
        factor_names: List[str],
        historical_presence: Any = None,
        fin_model: Any = None,
    ) -> Tuple[Any, pd.DataFrame]:
        """Phase 3: Scenario generation + synthetic data."""
        from .scenario_generator import ScenarioGenerator
        from .generate import ImpactDrivenGenerator

        # ── scenarios ───────────────────────────────────────────────
        if (getattr(self.cfg, "precomputed_analyses_dir", "")
                or not getattr(self.cfg, "generate_scenarios", True)):
            # v2.1: runs on saved labels make no LLM call at all; scenarios
            # do not feed the factual synthetic data.
            from .data_structures import ScenarioSet as _ScenarioSet
            scenario_set = _ScenarioSet(scenarios=[], factor_names=list(factor_names))
            print("\n  Scenario generation skipped (no scenario LLM call)")
        else:
            print("\n  Generating scenarios via LLM...")
            sg = ScenarioGenerator(
                llm_model=self.cfg.llm_model,
                api_key=self.cfg.llm_api_key or None,
                temperature=0.7,
                cache_dir=os.path.join(self.cfg.cache_dir, "scenarios"),
            )

            # Compute historical rates from presence matrix
            historical_rates = {
                fn: float(impact_matrix.occurrence_prob[i])
                for i, fn in enumerate(factor_names)
            }

            # Impact directions
            impact_directions = {}
            for i, fn in enumerate(factor_names):
                avg = impact_matrix.base_impact[i].mean()
                if avg > 0.01:
                    impact_directions[fn] = "bullish"
                elif avg < -0.01:
                    impact_directions[fn] = "bearish"
                else:
                    impact_directions[fn] = "neutral"

            dates = pd.to_datetime(df["date"])
            hist_period = (
                f"{dates.min().strftime('%Y-%m-%d')} to "
                f"{dates.max().strftime('%Y-%m-%d')}"
            )

            scenario_set = sg.generate(
                ticker=ticker,
                factor_names=factor_names,
                historical_rates=historical_rates,
                impact_directions=impact_directions,
                historical_period=hist_period,
                n_scenarios=self.cfg.n_scenarios,
                n_samples=self.cfg.scenario_llm_samples,
            )

        print(f"\n{scenario_set.summary()}")
        scenario_set.save(
            os.path.join(self.cfg.output_dir, "scenarios.json")
        )

        # ── generate ────────────────────────────────────────────────
        print("  Generating synthetic data...")

        # Estimate window trading days from largest window_size for scaling
        _window_days = 63  # default: ~3 months
        for ws in self.cfg.window_sizes:
            if "Q" in ws.upper():
                _window_days = max(_window_days, 63)
            elif "M" in ws.upper():
                _window_days = max(_window_days, 21)
        generator = ImpactDrivenGenerator(
            impact_matrix,
            window_trading_days=_window_days,
            mode=(getattr(self.cfg, "gen_mode", "") or None),
        )

        # Use real data to determine generation period and initial prices
        feature_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        df_sorted = df.sort_values("date")
        first_row = df_sorted.iloc[0]
        initial_prices = np.array(
            [first_row.get(c, 100.0) for c in feature_cols], dtype=np.float32
        )

        # Compute real log-returns for drift / vol calibration
        real_prices = df_sorted[feature_cols].values.astype(np.float64)
        real_prices = np.clip(real_prices, 1e-8, None)  # avoid log(0)
        real_log_returns = np.diff(np.log(real_prices), axis=0)  # (T-1, M)

        # Use training period dates so synthetic covers the same range
        real_dates = pd.to_datetime(df_sorted["date"])
        gen_start = self.cfg.gen_start_date or real_dates.min().strftime("%Y-%m-%d")
        n_steps = self.cfg.gen_n_steps
        if n_steps <= 0:
            # auto: match real trading days count
            n_steps = len(real_dates)

        # Extract real volume and dates for the new generator
        real_volume = df_sorted["volume"].values.astype(np.float64) if "volume" in df_sorted.columns else None
        real_dates_arr = pd.to_datetime(df_sorted["date"]).values

        # ── Neural SDE (#8) — optionally train & use ────────────────
        if self.cfg.use_neural_sde:
            print("  Training Neural SDE model...")
            from .neural_sde import train_neural_sde, NeuralSDEGenerator

            # Build factor matrix aligned to returns
            factor_matrix = historical_presence if historical_presence is not None else np.zeros((n_steps, len(factor_names)))

            # F5: Transfer learning from pre-trained FIN/TFT
            _pretrained = fin_model if self.cfg.use_transfer_learning else None

            nsde_model, nsde_history = train_neural_sde(
                log_returns=real_log_returns,
                factor_matrix=factor_matrix,
                n_price_features=min(4, len(feature_cols)),
                hidden=self.cfg.nsde_hidden,
                n_epochs=self.cfg.nsde_epochs,
                batch_size=self.cfg.nsde_batch_size,
                learning_rate=self.cfg.nsde_lr,
                patience=self.cfg.nsde_patience,
                device=self.cfg.fin_device,
                pretrained_fin=_pretrained,
                transfer_scale=self.cfg.transfer_scale,
            )
            print(f"    ✓ Neural SDE trained (best epoch {nsde_history['best_epoch']})")

            # Save model checkpoint
            import torch
            ckpt_path = os.path.join(self.cfg.output_dir, "neural_sde.pt")
            torch.save(nsde_model.state_dict(), ckpt_path)
            print(f"    ✓ Checkpoint: {ckpt_path}")

            # Build factor schedule for generation
            schedule = generator._build_schedule(
                n_steps, None, self.cfg.gen_n_samples,
                historical_presence=historical_presence,
            )

            nsde_gen = NeuralSDEGenerator(
                model=nsde_model,
                impact_generator=generator,
                device=self.cfg.fin_device if self.cfg.fin_device != "auto"
                    else ("cuda" if __import__("torch").cuda.is_available() else "cpu"),
            )
            prices = nsde_gen.generate(
                n_steps=n_steps,
                n_samples=self.cfg.gen_n_samples,
                initial_prices=initial_prices,
                factor_schedule=schedule,
                real_log_returns=real_log_returns,
                real_volume=real_volume,
            )

            # Convert to FinRL DataFrame
            if real_dates_arr is not None:
                gen_dates = pd.to_datetime(real_dates_arr[:n_steps])
            else:
                gen_dates = pd.bdate_range(start=gen_start, periods=n_steps)

            dfs: List[pd.DataFrame] = []
            for s in range(prices.shape[0]):
                sdf = pd.DataFrame(
                    prices[s],
                    columns=feature_cols[:prices.shape[2]],
                )
                sdf["date"] = gen_dates.strftime("%Y-%m-%d")
                sdf["tic"] = ticker
                sdf["sample"] = s
                dfs.append(sdf)

            synth_df = pd.concat(dfs, ignore_index=True)
        else:
            synth_df = generator.generate_finrl_df(
                ticker=ticker,
                n_steps=n_steps,
                initial_prices=initial_prices,
                n_samples=self.cfg.gen_n_samples,
                real_dates=real_dates_arr,
                start_date=gen_start,
                real_log_returns=real_log_returns,
                real_volume=real_volume,
                historical_presence=historical_presence,
                real_prices=real_prices,
            )

        # Save
        synth_path = os.path.join(
            self.cfg.output_dir, "synthetic_data_causal_sde.csv"
        )
        synth_df.to_csv(synth_path, index=False)
        print(f"  ✓ Synthetic data: {synth_path} ({len(synth_df)} rows)")

        return scenario_set, synth_df

    # ─────────────────────────────────────────────────────────────────

    def _evaluate(
        self,
        real_df: pd.DataFrame,
        synth_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Phase 4: Quality evaluation."""
        from .evaluate_data import CausalSDEEvaluator

        eval_dir = os.path.join(self.cfg.output_dir, "evaluation")
        evaluator = CausalSDEEvaluator(
            output_dir=eval_dir,
            include_deep_metrics=self.cfg.eval_include_deep,
        )

        # Use only sample 0 for evaluation
        if "sample" in synth_df.columns:
            synth_eval = synth_df[synth_df["sample"] == 0].copy()
        else:
            synth_eval = synth_df.copy()

        report = evaluator.evaluate(real_df, synth_eval)
        evaluator.print_report(report)
        return report

    # ── artefact saving ─────────────────────────────────────────────

    def _save_artefacts(
        self,
        report: dict,
        impact_matrix: Any,
        scenario_set: Any,
        synth_df: pd.DataFrame,
        real_df: pd.DataFrame,
        ticker: str,
    ) -> None:
        """Save all pipeline outputs."""
        out = self.cfg.output_dir

        # Full report
        report_path = os.path.join(out, "pipeline_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        # Config
        self.cfg.save(os.path.join(out, "config.yaml"))

        # Real data reference
        real_path = os.path.join(out, "real_data_reference.csv")
        real_df.to_csv(real_path, index=False)

        print(f"\n  Artefacts saved to {out}/")

    # ── convenience class methods ───────────────────────────────────

    @classmethod
    def from_csv(
        cls,
        csv_path: str,
        ticker: str,
        config: Optional[PipelineConfig | dict] = None,
        interactive: bool = False,
    ) -> Dict[str, Any]:
        """
        Run the pipeline from a CSV file.

        Parameters
        ----------
        csv_path : str
            Path to CSV with columns: date, open, high, low, close, volume.
        ticker : str
        config : PipelineConfig | dict | None
        interactive : bool
            Open the event-validation UI.

        Returns
        -------
        report : dict
        """
        df = pd.read_csv(csv_path)
        df["date"] = pd.to_datetime(df["date"])
        pipe = cls(config=config)
        return pipe.run(df, ticker=ticker, interactive=interactive)

    @classmethod
    def from_finrl(
        cls,
        df: pd.DataFrame,
        ticker: str,
        config: Optional[PipelineConfig | dict] = None,
    ) -> Dict[str, Any]:
        """
        Run on a FinRL-format DataFrame (multi-ticker supported).

        Filters to the specified ticker and runs the pipeline.
        """
        if "tic" in df.columns:
            df = df[df["tic"] == ticker].copy()

        if "timestamp" in df.columns and "date" not in df.columns:
            df = df.rename(columns={"timestamp": "date"})

        pipe = cls(config=config)
        return pipe.run(df, ticker=ticker)

