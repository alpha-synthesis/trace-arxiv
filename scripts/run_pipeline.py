#!/usr/bin/env python3
"""
Causal SDE Synthetic Data Generator — Script

Standalone script that:
1. Downloads real market data (via Yahoo Finance)
2. For each ticker:
   a. Extracts causal factors via LLM + RAG (Phase 1)
   b. Trains a Factor Impact Network (Phase 2)
   c. Generates N synthetic trajectories (Phase 3)
3. Evaluates quality (KS, Wasserstein, discriminative/predictive scores)
4. Saves everything into ``results/<output-dir>/``:
       results/<output-dir>/
           <TICKER>/
               synthetic_data_0.csv
               synthetic_data_1.csv
               ...
               pipeline_report.json
               impact_matrix.json
               scenarios.json
               config.yaml
               evaluation/
                   quality_report.json
                   ...

The generated CSVs are compatible with ``load_synthetic_csvs()``
and the existing evaluation pipeline.

All configuration is driven by the YAML file — no CLI flags needed.

Usage:
    # Default config
    python scripts/run_pipeline.py

    # Custom config path
    python scripts/run_pipeline.py configs/my_config.yaml
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR.parent          # scripts/ -> repo root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load .env from project root (OPENROUTER_API_KEY, OPENROUTER_MODEL_HEAVY, etc.)
load_dotenv(PROJECT_ROOT / ".env")

# Default config path (relative to project root)
DEFAULT_CONFIG = str(PROJECT_ROOT / "configs" / "causal_sde_config.yaml")


# ══════════════════════════════════════════════════════════════════════
# Config loading
# ══════════════════════════════════════════════════════════════════════


def _load_config_file(path: str) -> dict:
    """Load a YAML config file and return it as a dict."""
    if not os.path.isfile(path):
        path = os.path.join(str(PROJECT_ROOT), "configs", path)
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    return cfg if cfg else {}


def load_config(config_path: Optional[str] = None) -> dict:
    """Load config from YAML file. All settings come from YAML — no CLI overrides."""
    path = config_path or DEFAULT_CONFIG
    cfg = _load_config_file(path)
    cfg["_config_path"] = path
    return cfg


# ══════════════════════════════════════════════════════════════════════
# Checkpoint / resume helpers
# ══════════════════════════════════════════════════════════════════════


def _is_ticker_complete(ticker_output: str, cfg: dict) -> bool:
    """Check whether a ticker's generation has already completed.

    A ticker is considered complete when:
      1. ``synthetic_data_causal_sde.csv`` exists, AND
      2. ``pipeline_report.json`` exists, AND
      3. The expected number of per-sample CSVs are present.
    """
    synth_csv = os.path.join(ticker_output, "synthetic_data_causal_sde.csv")
    report_json = os.path.join(ticker_output, "pipeline_report.json")
    if not os.path.exists(synth_csv) or not os.path.exists(report_json):
        return False

    # Check that per-sample CSVs exist
    n_samples = cfg.get("generation", {}).get("n_samples", 10)
    for i in range(n_samples):
        sample_csv = os.path.join(ticker_output, f"synthetic_data_{i}.csv")
        if not os.path.exists(sample_csv):
            return False
    return True


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════


def main(config_path: Optional[str] = None):
    cfg = load_config(config_path)

    t_start = time.time()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # ── Extract top-level settings ──────────────────────────────────
    tickers = cfg.get("tickers", ["AAPL"])
    train_start = cfg.get("train_start", "2004-01-01")
    train_end = cfg.get("train_end", "2020-06-30")
    test_start = cfg.get("test_start")
    test_end = cfg.get("test_end")
    output_name = cfg.get("output_dir", "causal_sde_data")
    seed = cfg.get("seed", 42)
    validate = cfg.get("validate", True)
    interactive = cfg.get("interactive", {}).get("post_extraction", False)
    interactive_pre = cfg.get("interactive", {}).get("pre_generation", False)
    interactive_port = cfg.get("interactive", {}).get("port", 8050)
    resume = cfg.get("resume", False)

    # ── Resolve LLM model & API key from env vars ──────────────────
    # If llm_model is empty in YAML, use OPENROUTER_MODEL_HEAVY env var
    llm_model = cfg.get("llm_model", "") or ""
    if not llm_model:
        env_model = os.environ.get("OPENROUTER_MODEL_HEAVY", "")
        if env_model:
            llm_model = f"openrouter/{env_model}" if not env_model.startswith("openrouter/") else env_model
        else:
            llm_model = "openrouter/anthropic/claude-sonnet-4"
    cfg["llm_model"] = llm_model

    # API key: YAML → OPENROUTER_API_KEY env var
    llm_api_key = cfg.get("llm_api_key", "") or ""
    if not llm_api_key:
        llm_api_key = os.environ.get("OPENROUTER_API_KEY", "")
    cfg["llm_api_key"] = llm_api_key

    output_dir = os.path.join(str(PROJECT_ROOT), "results", output_name)
    cache_dir = cfg.get("cache_dir", "cache/causal_sde")

    np.random.seed(seed)

    print("═" * 64)
    print("  CAUSAL SDE SYNTHETIC DATA GENERATOR")
    print("═" * 64)
    print(f"  Config:      {cfg.get('_config_path', '?')}")
    print(f"  Tickers:     {tickers}")
    print(f"  Train:       {train_start} → {train_end}")
    if test_start and test_end:
        print(f"  Test:        {test_start} → {test_end}")
    print(f"  LLM model:   {llm_model}")
    print(f"  Output:      {output_dir}")
    print(f"  Interactive: post={interactive}, pre={interactive_pre}")
    print(f"  Resume:      {resume}")
    print(f"  Seed:        {seed}")
    print("═" * 64)

    # ── 1. Download real data ───────────────────────────────────────
    print("\n" + "─" * 64)
    print("  STEP 1 — Downloading real market data")
    print("─" * 64)

    from sde_causal_generator.download_data import (
        download_training_data,
        download_test_data,
        split_by_ticker,
        save_training_data,
        validate_data,
    )

    train_df = download_training_data(
        tickers=tickers,
        start_date=train_start,
        end_date=train_end,
        cache_dir=cache_dir,
    )
    validate_data(train_df)
    save_training_data(train_df, output_dir, "training_data.csv")

    test_df = None
    if test_start and test_end:
        test_df = download_test_data(
            tickers=tickers,
            start_date=test_start,
            end_date=test_end,
            cache_dir=cache_dir,
        )
        save_training_data(test_df, output_dir, "test_data.csv")

    ticker_dfs = split_by_ticker(train_df)

    # ── 2. Build PipelineConfig from YAML ───────────────────────────
    from sde_causal_generator.pipeline import PipelineConfig, CausalSDEPipeline

    all_reports: Dict[str, Any] = {}

    for ticker in tickers:
        if ticker not in ticker_dfs:
            print(f"\n  ⚠ Ticker {ticker} not found in downloaded data, skipping.")
            continue

        df_ticker = ticker_dfs[ticker]
        ticker_output = os.path.join(output_dir, ticker)

        # ── Checkpoint/resume: skip completed tickers ───────────────
        if resume and _is_ticker_complete(ticker_output, cfg):
            print(f"\n  ✓ [{ticker}] already complete — skipping (resume: true)")
            # Load existing report for the summary
            report_path = os.path.join(ticker_output, "pipeline_report.json")
            if os.path.exists(report_path):
                with open(report_path) as f:
                    all_reports[ticker] = json.load(f)
            continue
        os.makedirs(ticker_output, exist_ok=True)

        print(f"\n{'═' * 64}")
        print(f"  TICKER: {ticker}  ({len(df_ticker)} rows)")
        print(f"{'═' * 64}")

        # Build per-ticker config
        ticker_cfg = dict(cfg)
        ticker_cfg["output_dir"] = ticker_output
        ticker_cfg["cache_dir"] = os.path.join(cache_dir, ticker)

        # Load PipelineConfig (handles nested YAML sections)
        pipe_cfg = PipelineConfig._from_flat_or_nested(ticker_cfg)

        # ── Run pipeline ────────────────────────────────────────────
        pipe = CausalSDEPipeline(config=pipe_cfg)

        # Pass test data for OOS validation (#10)
        ticker_test_df = None
        if test_df is not None and "tic" in test_df.columns:
            td = test_df[test_df["tic"] == ticker]
            if len(td) > 10:
                ticker_test_df = td

        report = pipe.run(
            df=df_ticker,
            ticker=ticker,
            test_df=ticker_test_df,
        )

        # ── Save individual synthetic CSVs ──────────────────────────
        synth_path = os.path.join(ticker_output, "synthetic_data_causal_sde.csv")
        if os.path.exists(synth_path):
            synth_df = pd.read_csv(synth_path)
            n_samples = pipe_cfg.gen_n_samples
            if "sample" in synth_df.columns:
                for i in range(n_samples):
                    sample_df = synth_df[synth_df["sample"] == i].copy()
                    sample_df = sample_df.drop(columns=["sample"], errors="ignore")
                    sample_path = os.path.join(
                        ticker_output, f"synthetic_data_{i}.csv"
                    )
                    sample_df.to_csv(sample_path, index=False)
                print(f"  ✓ Saved {n_samples} individual CSVs for {ticker}")

        all_reports[ticker] = report

    # ── 3. Cross-asset re-correlation (#9) ────────────────────────────
    if len(tickers) > 1 and cfg.get("cross_asset", {}).get("enabled", True):
        print(f"\n{'─' * 64}")
        print("  STEP 3 — Cross-asset correlation (#9)")
        print("─" * 64)

        from sde_causal_generator.cross_asset import (
            extract_returns_from_dataframes,
            estimate_cross_asset_correlation,
            cholesky_factor,
            recorrelate_prices,
        )

        # Estimate correlation from real training data
        real_ticker_dfs = {t: ticker_dfs[t] for t in tickers if t in ticker_dfs}
        ca_method = cfg.get("cross_asset", {}).get("method", "pearson")
        ticker_returns = extract_returns_from_dataframes(real_ticker_dfs)

        if len(ticker_returns) >= 2:
            corr_matrix, ordered_tickers = estimate_cross_asset_correlation(
                ticker_returns, method=ca_method,
            )
            L = cholesky_factor(corr_matrix)

            print(f"    Correlation matrix ({ca_method}):")
            for i, t in enumerate(ordered_tickers):
                vals = " ".join(f"{corr_matrix[i,j]:+.3f}" for j in range(len(ordered_tickers)))
                print(f"      {t:>6s}: {vals}")

            # Re-correlate each sample's close prices
            for sample_i in range(cfg.get("generation", {}).get("n_samples", 10)):
                sample_prices = {}
                for t in ordered_tickers:
                    ticker_out = os.path.join(output_dir, t)
                    csv_path = os.path.join(ticker_out, f"synthetic_data_{sample_i}.csv")
                    if os.path.exists(csv_path):
                        sdf = pd.read_csv(csv_path)
                        if "close" in sdf.columns:
                            sample_prices[t] = sdf["close"].values

                if len(sample_prices) == len(ordered_tickers):
                    corr_prices = recorrelate_prices(sample_prices, L, ordered_tickers)
                    for t in ordered_tickers:
                        ticker_out = os.path.join(output_dir, t)
                        csv_path = os.path.join(ticker_out, f"synthetic_data_{sample_i}.csv")
                        sdf = pd.read_csv(csv_path)
                        sdf["close"] = corr_prices[t][:len(sdf)]
                        sdf.to_csv(csv_path, index=False)

            print(f"    ✓ Cross-asset correlation applied to {len(ordered_tickers)} tickers")

            # Save correlation matrix
            import json as _json
            corr_path = os.path.join(output_dir, "cross_asset_correlation.json")
            with open(corr_path, "w") as f:
                _json.dump({
                    "tickers": ordered_tickers,
                    "method": ca_method,
                    "correlation_matrix": corr_matrix.tolist(),
                }, f, indent=2)
            print(f"    ✓ Correlation matrix saved: {corr_path}")
        else:
            print("    ⚠ Not enough tickers with overlapping data for cross-asset correlation")

    # ── 4. Save global summary ──────────────────────────────────────
    total_time = time.time() - t_start

    summary = {
        "timestamp": timestamp,
        "config_file": cfg.get("_config_path", "?"),
        "tickers": tickers,
        "train_period": f"{train_start} → {train_end}",
        "test_period": f"{test_start} → {test_end}" if test_start else None,
        "total_time_seconds": round(total_time, 1),
        "per_ticker": {
            t: {
                "total_time": r.get("total_time_seconds"),
                "n_factors": r.get("phases", {}).get("extraction", {}).get("n_factors"),
                "n_scenarios": r.get("phases", {}).get("generation", {}).get("n_scenarios"),
            }
            for t, r in all_reports.items()
        },
    }

    summary_path = os.path.join(output_dir, "generation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Save a copy of the config used
    config_path = os.path.join(output_dir, "config_used.yaml")
    cfg_to_save = {k: ("***redacted***" if k in ("llm_api_key", "fred_api_key") and v else v)
                   for k, v in cfg.items()}
    with open(config_path, "w") as f:
        yaml.dump(cfg_to_save, f, default_flow_style=False, sort_keys=False)

    print(f"\n{'═' * 64}")
    print(f"  ALL DONE — {len(all_reports)} tickers, {total_time:.1f}s total")
    print(f"  Results: {output_dir}/")
    print(f"  Summary: {summary_path}")
    print(f"{'═' * 64}\n")


if __name__ == "__main__":
    import argparse
    _parser = argparse.ArgumentParser(description="Run the full ECSDE pipeline")
    _parser.add_argument("--config", default=None, help="Path to YAML config")
    # Legacy: also accept positional arg for backward compat
    _parser.add_argument("config_positional", nargs="?", default=None)
    _args = _parser.parse_args()
    _path = _args.config or _args.config_positional
    main(config_path=_path)
