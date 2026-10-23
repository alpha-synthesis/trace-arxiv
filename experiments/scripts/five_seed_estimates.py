#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase-2-only retraining with 5 seeds (criteria v2.1).

For every ticker of a run, retrain the corrected TFT impact network
(impact_estimator="ablation") on the run's labels and prices with seeds 0-4,
and save one impact matrix per seed. Hyperparameters match the pipeline config
(n_lags 10, 300 epochs, batch 64, lr 1e-3, patience 30, val 0.15). No LLM, no
RAG, no generation.

Usage:
  TRACE_RUN=r1_datasign python five_seed_estimates.py --variant data
  TRACE_RUN=r1_llmsign  python five_seed_estimates.py --variant llm
Output: results/five_seed__<RUN>__<variant>/<TICKER>/impact_matrix_seed<s>.json
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, RUN_TAG, build_presence, load_ticker_artifacts, save_json, ticker_df, tickers_28  # noqa: E402
from shared import FEATS, _llm_directions  # noqa: E402

SEEDS = (0, 1, 2, 3, 4)


def seed_all(s: int):
    import torch
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["data", "llm"], required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tickers", default="")
    args = ap.parse_args()
    from sde_causal_generator.factor_impact_network import train_impact_network

    out_root = OUT / f"five_seed__{RUN_TAG}__{args.variant}"
    tickers = args.tickers.split(",") if args.tickers else tickers_28()
    log = []
    for t in tickers:
        im0, _, analyses = load_ticker_artifacts(t)
        assert im0.units == "daily", f"{t}: run is not a corrected (daily-unit) run"
        names = im0.factor_names
        df = ticker_df(t)
        prices = np.clip(df[FEATS].values.astype(np.float64), 1e-8, None)
        P = build_presence(analyses, df, names)
        dirs = _llm_directions(analyses, names) if args.variant == "llm" else None
        for s in SEEDS:
            f = out_root / t / f"impact_matrix_seed{s}.json"
            if f.exists():
                continue
            t0 = time.time()
            seed_all(s)
            _, im, hist = train_impact_network(
                factor_matrix=P, price_data=prices, factor_names=names,
                feature_names=FEATS, n_lags=10, n_epochs=300, batch_size=64,
                learning_rate=1e-3, val_fraction=0.15, device=args.device,
                patience=30, architecture="tft", llm_directions=dirs,
                impact_estimator="ablation",
            )
            f.parent.mkdir(parents=True, exist_ok=True)
            im.save(str(f))
            log.append({"ticker": t, "seed": s, "best_epoch": hist.get("best_epoch"),
                        "elapsed_s": round(time.time() - t0, 1)})
            print(f"{t} seed {s}: best_epoch={hist.get('best_epoch')} ({log[-1]['elapsed_s']}s)", flush=True)
    save_json({"run": RUN_TAG, "variant": args.variant, "seeds": list(SEEDS), "log": log},
              out_root / f"multiseed_log_{int(time.time())}.json")


if __name__ == "__main__":
    main()
