#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Estimator comparison on the power-check synthetic panels (synthetic only).

Context: the sign-recovery check (sign_recovery.py) found
that the corrected TFT recovers planted signs at 0.56-0.72 for beta 0.05-0.3
sigma, never >= 0.80, while OLS and the univariate in-window mean reach
0.85-0.95 at 0.2-0.3 sigma on the same data. Diagnostics: best epochs of 0-7
on many fits; the model is selected on the validation Gaussian NLL over all
five targets (volume included), where the factor-mean signal is negligible;
the temporal self-attention has no causal mask (look-ahead within a window).

This script compares, on exactly the same synthetic panels as sign_recovery.py
(same presence, same GJR-GARCH-t draws, same planted factors and signs), the
TFT under alternative training rules. No real return is used.

  nll_plain      : the estimator fixed in advance (results reused from sign_recovery)
  mse_plain      : model selected on validation close-return MSE
  mse_causal     : close-return MSE selection + causal attention mask
  last60_causal  : causal mask, no early stopping, 60 epochs (cosine schedule
                   over 60 epochs), the model after the last epoch

Seed 0, same hyperparameters otherwise. Usage:
  SDECG_ROOT=<generator worktree> TRACE_RUN=r1_datasign \
      python sign_recovery_estimators.py --tickers AAPL,MSFT [--device cuda]
  ... --aggregate
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np

if os.environ.get("SDECG_ROOT"):
    sys.path.insert(0, os.environ["SDECG_ROOT"])
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, RUN_TAG, build_presence, load_ticker_artifacts, save_json, ticker_df, tickers_28  # noqa: E402
from shared import FEATS  # noqa: E402
from sign_recovery import BASELINE_NAMES, BETAS, MIN_ACTIVE, N_PLANT, gjr_t_returns, seed_all  # noqa: E402

if os.environ.get("SDECG_ROOT"):            # common.py puts the main checkout first
    sys.path.insert(0, os.environ["SDECG_ROOT"])
    for _m in [m for m in sys.modules if m.startswith("sde_causal_generator")]:
        del sys.modules[_m]

VARIANTS = {
    "mse_plain": dict(selection="close_mse", causal_attention=False, n_epochs=300),
    "mse_causal": dict(selection="close_mse", causal_attention=True, n_epochs=300),
    "last60_causal": dict(selection="last", causal_attention=True, n_epochs=60),
}
OUT_DIR = OUT / f"sign_recovery_estimators__{RUN_TAG}"
PC_DIR = OUT / f"sign_recovery__{RUN_TAG}"


def panel(ticker, beta):
    im0, _, analyses = load_ticker_artifacts(ticker)
    names = im0.factor_names
    df = ticker_df(ticker)
    real = np.clip(df[FEATS].values.astype(np.float64), 1e-8, None)
    P = build_presence(analyses, df, names)
    T = len(real)
    r_real = np.diff(np.log(real[:, 3]))
    sigma = float(r_real.std())
    occ = (P > 0).mean(axis=0)
    cand = [k for k, n in enumerate(names)
            if n not in BASELINE_NAMES and occ[k] <= 0.8 and (P[:, k] > 0).sum() >= MIN_ACTIVE]
    rng = np.random.RandomState(zlib.crc32(ticker.encode()))
    plant = rng.choice(cand, size=min(N_PLANT, len(cand)), replace=False)
    signs = rng.choice([-1.0, 1.0], size=len(plant))
    eps, _ = gjr_t_returns(r_real, T - 1, rng)
    r = eps + ((P[:-1, plant] > 0).astype(float) * (beta * sigma * signs)).sum(axis=1)
    close = real[0, 3] * np.exp(np.concatenate([[0.0], np.cumsum(r)]))
    prices = np.column_stack([real[:, :3] / real[:, [3]] * close[:, None], close, real[:, 4]])
    return P, prices, names, plant, signs, sigma


def run_ticker(ticker, device):
    from sde_causal_generator.factor_impact_network import train_impact_network
    pc = json.loads((PC_DIR / f"ticker_{ticker}.json").read_text())
    out = {"ticker": ticker, "per_beta": {}}
    for beta in BETAS:
        P, prices, names, plant, signs, sigma = panel(ticker, beta)
        assert [names[k] for k in plant] == pc["planted"] and signs.tolist() == pc["signs"]
        cell = {"nll_plain": {"sign_recovery": pc["per_beta"][str(beta)]["sign_recovery"]}}
        for v, kw in VARIANTS.items():
            t0 = time.time()
            seed_all(0)
            _, im, hist = train_impact_network(
                factor_matrix=P, price_data=prices, factor_names=names, feature_names=FEATS,
                n_lags=10, n_epochs=kw["n_epochs"], batch_size=64, learning_rate=1e-3,
                val_fraction=0.15, device=device, patience=30, architecture="tft",
                llm_directions=None, impact_estimator="ablation",
                selection=kw["selection"], causal_attention=kw["causal_attention"])
            rec = im.base_impact[:, 3][plant] / sigma
            cell[v] = {"sign_recovery": float((np.sign(rec) == signs).mean()),
                       "corr": float(np.corrcoef(rec, signs)[0, 1]),
                       "recovered_sigma": rec.tolist(), "best_epoch": hist.get("best_epoch"),
                       "elapsed_s": round(time.time() - t0, 1)}
        out["per_beta"][str(beta)] = cell
        print(f"  {ticker} beta={beta}: " + " ".join(f"{v}={c['sign_recovery']:.2f}" for v, c in cell.items()),
              flush=True)
    return out


def aggregate():
    rows = [json.loads(f.read_text()) for f in sorted(OUT_DIR.glob("ticker_*.json"))]
    pcs = {r["ticker"]: json.loads((PC_DIR / f"ticker_{r['ticker']}.json").read_text()) for r in rows}
    rep = {"run": RUN_TAG, "n_tickers": len(rows), "per_beta": {}}
    for beta in BETAS:
        s = str(beta)
        cell = {}
        for v in ["nll_plain", *VARIANTS]:
            hits = []
            for r in rows:
                if v == "nll_plain":
                    hits += list(np.sign(pcs[r["ticker"]]["per_beta"][s]["recovered_sigma"]) == np.array(pcs[r["ticker"]]["signs"]))
                else:
                    hits += list(np.sign(r["per_beta"][s][v]["recovered_sigma"]) == np.array(pcs[r["ticker"]]["signs"]))
            cell[v] = float(np.mean(hits))
        cell["ols"] = float(np.mean(np.concatenate([
            np.repeat(pcs[r["ticker"]]["per_beta"][s]["ols_sign_recovery"], len(pcs[r["ticker"]]["signs"])) for r in rows])))
        cell["univariate"] = float(np.mean(np.concatenate([
            np.repeat(pcs[r["ticker"]]["per_beta"][s]["univariate_sign_recovery"], len(pcs[r["ticker"]]["signs"])) for r in rows])))
        rep["per_beta"][s] = cell
    save_json(rep, OUT_DIR / "summary.json")
    print(json.dumps(rep, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()
    if a.aggregate:
        aggregate(); return
    import sde_causal_generator
    print("generator code:", sde_causal_generator.__file__, flush=True)
    if os.environ.get("SDECG_ROOT"):
        assert sde_causal_generator.__file__.startswith(os.environ["SDECG_ROOT"])
    for t in (a.tickers.split(",") if a.tickers else tickers_28()):
        f = OUT_DIR / f"ticker_{t}.json"
        if not f.exists():
            save_json(run_ticker(t, a.device), f)


if __name__ == "__main__":
    main()
