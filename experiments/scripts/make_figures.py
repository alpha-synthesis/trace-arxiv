#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figures for the TRACE paper, from saved results only.

Writes vector PDFs to experiments/figures/ and every plotted number to
results/figure_data/figure_data.json, so each figure can be checked against
its source files.

  fig_b1_leak.pdf          B1: label-outcome agreement vs independence, per arm,
                           with calendar-window cluster-bootstrap 95% CIs on the
                           excess and on the next-window placebo excess
  fig_power.pdf            power gate: planted-sign recovery vs beta (TFT 5-seed
                           majority on R1 and R2 presence, OLS pooled over planted
                           factors, ideal in-window-mean bound for a 10-day factor),
                           80% line and the realistic effect band
  fig_recurrence.pdf       factor-name recurrence: share of ticker-factor names by
                           number of windows in which they occur, R1 vs R2
  fig_tracking.pdf         B2: |terminal tracking error| per ticker, R1 vs R2
  fig_counterfactual.pdf   B2 example: history vs factual paths (R2), and the
                           cumulative effect of removing the eligible bull group
                           (R1, R2) against path noise
  fig_estimators.pdf       appendix: estimator comparison on the power panels

Bootstrap draws use fixed seeds; the CIs reported in the paper's tables are
the ones written here.

Usage: python make_figures.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ORIG, OUT, build_presence  # noqa: E402
from common import ROOT as RUNS_ROOT  # noqa: E402  (TRACE_ROOT; default ORIG/results)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

IMG = Path(__file__).resolve().parents[1] / "figures"
DATA = OUT / "figure_data"
IMG.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

# palette: reference categorical slots 1-3 (validated all-pairs, light), text tokens
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#a3a29c", "#e6e5e1"
COL_W = 3.33          # ACM single column, inches

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 7.5,
    "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.8, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.6,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
FIG_DATA: dict = {}
NOISE = {"market_noise", "noise_residual"}


def _grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


# ── B1 ───────────────────────────────────────────────────────────────────
ARMS = [("A0", "r0", "A0: prices in prompt"),
        ("A1", "a1", "A1: price-blind, uncurated"),
        ("R2", "r2_datasign", "R2: price-blind, curated")]


def _arm_windows(run):
    rows = []
    for tdir in sorted((RUNS_ROOT / run).iterdir()):
        f = tdir / "analyses.json"
        if not f.exists():
            continue
        ws = sorted(json.load(open(f)), key=lambda x: x["start_date"])
        for i, w in enumerate(ws):
            fs = [x for x in w["factors"] if x["name"] not in NOISE and x["magnitude"] > 0]
            s = sum(x["direction"] * x["magnitude"] for x in fs)
            rows.append({"ticker": tdir.name, "win": i, "s": s, "r": float(w["price_change_pct"]),
                         "r_next": float(ws[i + 1]["price_change_pct"]) if i + 1 < len(ws) else np.nan})
    return pd.DataFrame(rows)


def _excess(df, rcol):
    d = df[(df.s != 0) & (df[rcol] != 0) & df[rcol].notna()]
    s, r = np.sign(d.s.values), np.sign(d[rcol].values)
    agree = float((s == r).mean())
    up, bull = (r > 0).mean(), (s > 0).mean()
    ind = float(up * bull + (1 - up) * (1 - bull))
    return agree, ind, agree - ind, len(d)


def _cluster_ci(df, rcol, n_boot=4000, seed=0):
    """Calendar-window cluster bootstrap of agreement minus independence."""
    d = df[(df.s != 0) & (df[rcol] != 0) & df[rcol].notna()]
    s, r = np.sign(d.s.values), np.sign(d[rcol].values)
    wins = np.array(sorted(d.win.unique()))
    idx = np.searchsorted(wins, d.win.values)
    W = len(wins)
    n_w = np.bincount(idx, minlength=W).astype(float)
    a_w = np.bincount(idx, weights=(s == r).astype(float), minlength=W)
    u_w = np.bincount(idx, weights=(r > 0).astype(float), minlength=W)
    b_w = np.bincount(idx, weights=(s > 0).astype(float), minlength=W)
    rng = np.random.RandomState(seed)
    ex = np.empty(n_boot)
    for i in range(n_boot):
        c = np.bincount(rng.randint(0, W, W), minlength=W)
        n = c @ n_w
        up, bull = (c @ u_w) / n, (c @ b_w) / n
        ex[i] = (c @ a_w) / n - (up * bull + (1 - up) * (1 - bull))
    return [float(np.percentile(ex, 2.5)), float(np.percentile(ex, 97.5))]


def fig_b1():
    out = {}
    for key, run, label in ARMS:
        df = _arm_windows(run)
        a, i, e, n = _excess(df, "r")
        _, _, pe, pn = _excess(df, "r_next")
        out[key] = {"label": label, "agreement": a, "independence": i, "excess": e, "n_scored": n,
                    "excess_ci95": _cluster_ci(df, "r"), "placebo_excess": pe,
                    "placebo_ci95": _cluster_ci(df, "r_next"), "n_placebo": pn}
    FIG_DATA["b1"] = out
    fig, ax = plt.subplots(figsize=(COL_W, 1.75))
    keys = [k for k, _, _ in ARMS]
    y = np.arange(len(keys))[::-1]
    for yi, k in zip(y, keys):
        o = out[k]
        lo, hi = o["excess_ci95"]
        ax.plot([lo * 100, hi * 100], [yi + 0.12] * 2, color=BLUE, lw=2, solid_capstyle="round")
        ax.plot(o["excess"] * 100, yi + 0.12, "o", color=BLUE, ms=5, mec="white", mew=0.8)
        plo, phi = o["placebo_ci95"]
        ax.plot([plo * 100, phi * 100], [yi - 0.14] * 2, color=ORANGE, lw=2, solid_capstyle="round")
        ax.plot(o["placebo_excess"] * 100, yi - 0.14, "s", color=ORANGE, ms=4.5, mec="white", mew=0.8)
        ax.text(hi * 100 + 1.2, yi + 0.12, f"{o['excess']*100:+.1f}", va="center", fontsize=6.5, color=INK2)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([out[k]["label"] for k in keys])
    ax.set_xlabel("Excess over independence baseline (pp)")
    ax.set_xlim(-6, 45)
    _grid(ax, "x")
    ax.plot([], [], "o-", color=BLUE, ms=4, label="same window")
    ax.plot([], [], "s-", color=ORANGE, ms=4, label="next window (placebo)")
    ax.legend(loc="lower right", frameon=False, bbox_to_anchor=(1.0, 0.18))
    fig.savefig(IMG / "fig_b1_leak.pdf")
    plt.close(fig)


# ── power ────────────────────────────────────────────────────────────────
BETAS = ["0.05", "0.1", "0.2", "0.3"]


def _pooled_rate(run, field, n_boot=4000, seed=0):
    rows = [json.load(open(f)) for f in sorted((OUT / f"sign_recovery__{run}").glob("ticker_*.json"))]
    res = {}
    rng = np.random.RandomState(seed)
    for b in BETAS:
        n = np.array([len(r["signs"]) for r in rows], dtype=float)
        hits = np.array([round(r["per_beta"][b][field] * len(r["signs"])) for r in rows], dtype=float)
        boots = []
        for _ in range(n_boot):
            k = rng.randint(0, len(rows), len(rows))
            boots.append(hits[k].sum() / n[k].sum())
        res[b] = {"rate": float(hits.sum() / n.sum()), "ci95": [float(np.percentile(boots, 2.5)),
                                                              float(np.percentile(boots, 97.5))],
                  "n_planted": int(n.sum())}
    return res


def fig_power():
    tft = {}
    for arm, run in (("R1", "r1_datasign"), ("R2", "r2_datasign")):
        s = json.load(open(OUT / f"sign_recovery__{run}" / "summary_majority5.json"))
        tft[arm] = {b: {"rate": s["per_beta"][b]["majority_sign_recovery"],
                        "ci95": s["per_beta"][b]["ci95_ticker_bootstrap"]} for b in BETAS}
    ols = {arm: _pooled_rate(run, "ols_sign_recovery") for arm, run in
           (("R1", "r1_datasign"), ("R2", "r2_datasign"))}
    grid = np.linspace(0.0, 0.32, 65)
    ideal10 = [0.5 * (1 + math.erf(b * math.sqrt(10) / math.sqrt(2))) for b in grid]
    FIG_DATA["power"] = {"tft_majority5": tft, "ols_pooled_over_planted": ols,
                         "ideal_bound_n10": {f"{b:.2f}": v for b, v in zip(grid[::8], ideal10[::8])},
                         "realistic_band_sd": [0.06, 0.12]}
    x = [float(b) for b in BETAS]
    fig, ax = plt.subplots(figsize=(COL_W, 2.25))
    # realistic event sizes: a 1% one-day move spread over ten days for the median
    # stock (0.056 s.d.; 0.033-0.078 across the panel) up to the largest effect
    # implied by the labelled factors' in-sample excess returns (about 0.12 s.d.)
    ax.axvspan(0.06, 0.12, color="#f0efec", zorder=0)
    ax.text(0.09, 0.462, "realistic\nevent size", ha="center", va="bottom", fontsize=6.3, color=INK2)
    ax.axhline(0.8, color=INK2, lw=0.8, ls=(0, (4, 3)))
    ax.text(0.003, 0.807, "80% threshold", fontsize=6.3, color=INK2, va="bottom")
    ax.axhline(0.5, color=MUTED, lw=0.6)
    ax.plot(grid, ideal10, color=MUTED, lw=1.2, ls=(0, (2, 2)))
    ax.text(0.215, 0.655, "ideal (Gaussian),\n10-day factor", fontsize=6.3, color=INK2, ha="left")
    for arm, col, mk in (("R1", BLUE, "o"), ("R2", ORANGE, "s")):
        r = [tft[arm][b]["rate"] for b in BETAS]
        lo = [tft[arm][b]["rate"] - tft[arm][b]["ci95"][0] for b in BETAS]
        hi = [tft[arm][b]["ci95"][1] - tft[arm][b]["rate"] for b in BETAS]
        ax.errorbar(x, r, yerr=[lo, hi], color=col, marker=mk, ms=4.5, mec="white", mew=0.8,
                    capsize=0, elinewidth=1.0, label=f"TFT, 5-seed majority ({arm})")
    ax.plot(x, [ols["R1"][b]["rate"] for b in BETAS], color=AQUA, marker="^", ms=4.5, mec="white",
            mew=0.8, label="OLS on presence (R1)")
    y1, y2, yo = tft["R1"]["0.3"]["rate"], tft["R2"]["0.3"]["rate"], ols["R1"]["0.3"]["rate"]
    if abs(y1 - y2) < 0.035:
        mid = (y1 + y2) / 2
        y1, y2 = (mid + 0.024, mid - 0.024) if y1 >= y2 else (mid - 0.024, mid + 0.024)
    for lab, yy in (("TFT R1", y1), ("TFT R2", y2), ("OLS R1", yo)):
        ax.text(0.307, yy, lab, fontsize=6.3, color=INK2, va="center")
    ax.set_xlim(0, 0.36)
    ax.set_ylim(0.45, 1.0)
    ax.set_xticks([0, 0.05, 0.1, 0.2, 0.3])
    ax.set_xlabel(r"Planted effect $\beta$ (s.d. per active day)")
    ax.set_ylabel("Planted signs recovered")
    _grid(ax)
    ax.legend(loc="upper left", frameon=False)
    fig.savefig(IMG / "fig_power.pdf")
    plt.close(fig)


# ── recurrence ───────────────────────────────────────────────────────────
def _recurrence(run):
    counts = []
    for tdir in sorted((ORIG / "results" / run).iterdir()):
        f = tdir / "analyses.json"
        if not f.exists():
            continue
        c: dict = {}
        for w in json.load(open(f)):
            for x in {x["name"] for x in w["factors"] if x["magnitude"] > 0}:
                c[x] = c.get(x, 0) + 1
        counts += list(c.values())
    return np.array(counts)


def fig_recurrence():
    bins = [("1", 1, 1), ("2", 2, 2), ("3", 3, 3), ("4-9", 4, 9), (">=10", 10, 10**9)]
    out = {}
    for arm, run in (("R1", "r1_datasign"), ("R2", "r2_datasign")):
        c = _recurrence(run)
        out[arm] = {"n_ticker_factor_names": int(len(c)),
                    "share": {lab: float(((c >= lo) & (c <= hi)).mean()) for lab, lo, hi in bins}}
    FIG_DATA["recurrence"] = out
    fig, ax = plt.subplots(figsize=(COL_W, 1.8))
    xs = np.arange(len(bins))
    w = 0.36
    for j, (arm, col) in enumerate((("R1", BLUE), ("R2", ORANGE))):
        vals = [out[arm]["share"][lab] * 100 for lab, _, _ in bins]
        ax.bar(xs + (j - 0.5) * (w + 0.03), vals, width=w, color=col,
               label=f"{arm} ({out[arm]['n_ticker_factor_names']:,} stock-factor names)")
        ax.text(xs[0] + (j - 0.5) * (w + 0.03), vals[0] + 2, f"{vals[0]:.0f}%", ha="center",
                fontsize=6.5, color=INK2)
    ax.set_xticks(xs)
    ax.set_xticklabels([lab.replace(">=", "≥") for lab, _, _ in bins])
    ax.set_xlabel("Two-week windows in which a factor name occurs")
    ax.set_ylabel("Share of names (%)")
    ax.set_ylim(0, 105)
    _grid(ax)
    ax.legend(loc="upper right", frameon=False)
    fig.savefig(IMG / "fig_recurrence.pdf")
    plt.close(fig)


# ── B2 tracking ──────────────────────────────────────────────────────────
def fig_tracking():
    r1 = pd.read_csv(OUT / "b2_counterfactual__r1_datasign" / "per_ticker.csv").set_index("ticker")
    r2 = pd.read_csv(OUT / "b2_counterfactual__r2_datasign" / "per_ticker.csv").set_index("ticker")
    d = pd.DataFrame({"r1": r1.track_err_factual.abs(), "r2": r2.track_err_factual.abs()})
    d["fail"] = (d.r2 - d.r1) > 0.05
    FIG_DATA["tracking"] = {t: {"abs_track_r1": float(v.r1), "abs_track_r2": float(v.r2),
                                "fails": bool(v.fail)} for t, v in d.iterrows()}
    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    m = max(d.r1.max(), d.r2.max()) * 1.08
    xx = np.linspace(0, m, 10)
    ax.plot(xx, xx, color=MUTED, lw=0.8)
    ax.plot(xx, xx + 0.05, color=INK2, lw=0.8, ls=(0, (4, 3)), label="tolerance: R2 worse by 0.05")
    ok = d[~d.fail]
    bad = d[d.fail]
    ax.plot(ok.r1, ok.r2, "o", color=BLUE, ms=4.5, mec="white", mew=0.8, label=f"within tolerance ({len(ok)})")
    ax.plot(bad.r1, bad.r2, "s", color=ORANGE, ms=5, mec="white", mew=0.8, label=f"fails ({len(bad)})")
    for t, v in bad.iterrows():
        ax.annotate(t, (v.r1, v.r2), xytext=(4, -1), textcoords="offset points", fontsize=6.3, color=INK2)
    ax.set_xlim(0, m)
    ax.set_ylim(0, m)
    ax.set_xlabel("|Terminal tracking error|, R1 (log units)")
    ax.set_ylabel("|Terminal tracking error|, R2")
    _grid(ax, "both")
    ax.legend(loc="lower right", frameon=False)
    fig.savefig(IMG / "fig_tracking.pdf")
    plt.close(fig)


# ── B2 counterfactual example ────────────────────────────────────────────
def _paths_for(run, ticker, S=10, seed=1):
    import os
    os.environ["TRACE_RUN"] = run
    from sde_causal_generator.data_structures import ImpactMatrix, WindowAnalysis
    from sde_causal_generator.generate import ImpactDrivenGenerator
    rdir = ORIG / "results" / run / ticker
    im = ImpactMatrix.load(str(rdir / "impact_matrix.json"))
    an = [WindowAnalysis.from_dict(x) for x in json.load(open(rdir / "analyses.json"))]
    panel = pd.read_csv(ORIG / "results" / run / "training_data.csv")
    df = panel[panel.tic == ticker].sort_values("date").reset_index(drop=True)
    rp = np.clip(df[["open", "high", "low", "close", "volume"]].values.astype(float), 1e-8, None)
    rlr = np.diff(np.log(rp), axis=0)
    names = list(im.factor_names)
    P = build_presence(an, df, names)
    b = im.base_impact[:, 3]
    occ = (P > 0).mean(axis=0)
    elig = [k for k, n in enumerate(names) if n not in NOISE and occ[k] <= 0.8
            and (P[:, k] > 0).sum() >= 20 and abs(b[k]) >= 1e-8]
    bulls = [names[k] for k in elig if b[k] > 0]
    gen = ImpactDrivenGenerator(im)

    def run_paths(**kw):
        np.random.seed(seed)
        o = gen.generate_scenario(n_steps=len(rp), initial_prices=rp[0], n_samples=S, real_log_returns=rlr,
                                  real_volume=rp[:, 4], historical_presence=P, real_prices=rp, **kw)
        return o[:, :, 3]
    fac = run_paths(record_ref=True)
    ref = gen._last_ref
    cf = run_paths(matched_ref=ref, remove_factors=bulls) if bulls else fac
    return df, rp[:, 3], fac, cf, len(bulls)


def fig_counterfactual():
    rep = json.load(open(OUT / "b2_counterfactual__r2_datasign" / "report.json"))["rows"]
    eff = {r["ticker"]: abs(r["groups"]["bulls"]["d_path"]) for r in rep if r["groups"]["bulls"]}
    tick = sorted(eff, key=eff.get)[len(eff) // 2]        # median R2 bull-group effect
    df, hist, fac2, cf2, nb2 = _paths_for("r2_datasign", tick)
    _, _, fac1, cf1, nb1 = _paths_for("r1_datasign", tick)
    dates = pd.to_datetime(df.date).values
    lh = np.log(hist / hist[0])
    lf = np.log(fac2 / fac2[:, :1])
    eff2 = (np.log(cf2) - np.log(fac2)).mean(axis=0)
    eff1 = (np.log(cf1) - np.log(fac1)).mean(axis=0)
    noise = lf.std(axis=0)
    FIG_DATA["counterfactual"] = {
        "ticker": tick, "rule": "median |R2 bull-group effect| among tickers with a bull group",
        "n_bull_factors": {"R1": nb1, "R2": nb2}, "paths": 10, "seed": 1,
        "terminal_cum_effect": {"R1": float(eff1[-1]), "R2": float(eff2[-1])},
        "terminal_path_sd_R2": float(noise[-1]),
        "terminal_history_log_return": float(lh[-1]),
        "terminal_factual_median_log_return_R2": float(np.median(lf[:, -1]))}
    fig, axes = plt.subplots(2, 1, figsize=(COL_W, 3.5), sharex=True,
                             gridspec_kw={"height_ratios": [1.15, 1], "hspace": 0.32})
    ax = axes[0]
    ax.fill_between(dates, np.percentile(lf, 10, axis=0), np.percentile(lf, 90, axis=0),
                    color=BLUE, alpha=0.18, lw=0)
    ax.plot(dates, np.median(lf, axis=0), color=BLUE, lw=1.4)
    ax.plot(dates, lh, color=INK, lw=1.0)
    ax.text(dates[-1], lh[-1], "  history", fontsize=6.3, color=INK, va="center")
    ax.text(0.02, 0.93, "factual paths (R2): median, 10-90%", fontsize=6.3, color=BLUE, va="top",
            transform=ax.transAxes)
    ax.set_ylabel("Cumulative log return")
    ax.set_title(f"(a) {tick}: factual paths track history", loc="left", color=INK)
    _grid(ax)
    ax = axes[1]
    ax.fill_between(dates, -noise, noise, color="#f0efec", lw=0)
    ax.plot(dates, eff1, color=BLUE, lw=1.4)
    ax.plot(dates, eff2, color=ORANGE, lw=1.4)
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.text(dates[-1], eff1[-1], f"  R1 ({nb1} factors)", fontsize=6.3, color=INK2, va="center")
    ax.text(dates[-1], eff2[-1], f"  R2 ({nb2} factors)", fontsize=6.3, color=INK2, va="center")
    ax.text(dates[int(len(dates) * 0.03)], noise[int(len(dates) * 0.5)] * 0.75,
            "±1 s.d. of factual paths", fontsize=6.3, color=INK2)
    ax.set_ylabel("Change from removal (log)")
    ax.set_title("(b) removing the eligible positive-effect group", loc="left", color=INK)
    _grid(ax)
    fig.savefig(IMG / "fig_counterfactual.pdf")
    plt.close(fig)


# ── appendix: estimator comparison ───────────────────────────────────────
def fig_estimators():
    s = json.load(open(OUT / "sign_recovery_estimators__r1_datasign" / "summary.json"))["per_beta"]
    series = [("nll_plain", "TFT, primary", BLUE, "o", 1.6),
              ("mse_plain", "TFT, three other training rules", MUTED, None, 1.0),
              ("mse_causal", "_nolegend_", MUTED, None, 1.0),
              ("last60_causal", "_nolegend_", MUTED, None, 1.0),
              ("ols", "OLS on presence", AQUA, "^", 1.6),
              ("univariate", "in-window mean", ORANGE, "s", 1.6)]
    FIG_DATA["estimators"] = {k: {b: s[b][k] for b in BETAS} for k, *_ in series}
    x = [float(b) for b in BETAS]
    fig, ax = plt.subplots(figsize=(COL_W, 2.1))
    ax.axhline(0.8, color=INK2, lw=0.8, ls=(0, (4, 3)))
    for k, lab, col, mk, lw in series:
        ax.plot(x, [s[b][k] for b in BETAS], color=col, lw=lw, marker=mk, ms=4 if mk else 0,
                mec="white", mew=0.7, label=lab, zorder=3 if mk else 2)
    ax.text(0.302, s["0.3"]["univariate"], "in-window mean", fontsize=6.3, color=INK2, va="center")
    ax.text(0.302, s["0.3"]["ols"] - 0.012, "OLS", fontsize=6.3, color=INK2, va="center")
    ax.text(0.302, s["0.3"]["nll_plain"], "TFT (primary)", fontsize=6.3, color=INK2, va="center")
    ax.text(0.302, (s["0.3"]["mse_causal"] + s["0.3"]["last60_causal"]) / 2 + 0.01, "other TFT rules",
            fontsize=6.3, color=INK2, va="center")
    ax.set_xlim(0.03, 0.38)
    ax.set_ylim(0.5, 1.0)
    ax.set_xticks(x)
    ax.set_xlabel(r"Planted effect $\beta$ (s.d. per active day)")
    ax.set_ylabel("Planted signs recovered")
    _grid(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.27), frameon=False, ncol=2)
    fig.savefig(IMG / "fig_estimators.pdf")
    plt.close(fig)


# ── positive control ─────────────────────────────────────────────────────
def fig_pc():
    """B3 pass rate on the synthetic positive-control worlds (results/positive_control)."""
    j = json.load(open(OUT / "positive_control" / "pc_summary.json"))
    pts = sorted(j["curve"].values(), key=lambda v: v["beta"])
    nul = j["null_cell"]["B3"]
    xs = [0.0] + [v["beta"] for v in pts]
    ys = [nul["rate"]] + [v["rate"] for v in pts]
    lo = [nul["rate"] - nul["cp95"][0]] + [v["rate"] - v["cp95"][0] for v in pts]
    hi = [nul["cp95"][1] - nul["rate"]] + [v["cp95"][1] - v["rate"] for v in pts]
    adj = [nul["rate"]] + [v["size_adjusted"] / v["n"] for v in pts]
    ev = {b: j["cells"][k]["B3"]["rate"] for b, k in ((0.0, "EV0@0"), (0.12, "EV@0.12"), (0.25, "EV@0.25"))}
    tft = {0.0: [j["T2"]["world"]["B3"] == "PASS"], 0.25: [j["T1"]["world"]["B3"] == "PASS"]}
    for tag, w in j["TFT_other_worlds"].items():
        b = float(tag.rsplit("_b", 1)[1])
        if b in tft:
            tft[b].append(w["B3"] == "PASS")
    FIG_DATA["positive_control"] = {"recurring_ols": dict(zip(map(str, xs), ys)),
                                    "size_adjusted": dict(zip(map(str, xs), adj)),
                                    "window_specific_ols": {str(k): v for k, v in ev.items()},
                                    "tft_pass_share": {str(k): float(np.mean(v)) for k, v in tft.items()},
                                    "tft_worlds": {str(k): len(v) for k, v in tft.items()}}
    fig, ax = plt.subplots(figsize=(COL_W, 2.45))
    ax.axvspan(0.06, 0.12, color="#f0efec", zorder=0)
    ax.text(0.09, 0.07, "realistic\nevent size", ha="center", va="bottom", fontsize=6.3, color=INK2)
    ax.axhline(0.8, color=INK2, lw=0.8, ls=(0, (4, 3)))
    ax.errorbar(xs, ys, yerr=[lo, hi], color=BLUE, marker="o", ms=4.5, mec="white", mew=0.8, capsize=0,
                elinewidth=1.0, label="OLS, recurring names")
    ax.plot(xs, adj, color=BLUE, lw=1.0, ls=(0, (2, 2)), label="OLS, size-adjusted")
    ax.plot(list(ev), list(ev.values()), color=ORANGE, marker="s", ms=4.5, mec="white", mew=0.8,
            label="OLS, window-specific names")
    tx = sorted(tft)
    ax.plot(tx, [float(np.mean(tft[b])) for b in tx], ls="none", marker="D", ms=5, mfc="white", mec=INK, mew=1.0,
            label="TFT, recurring names (3 worlds)")
    ax.set_xlim(-0.01, 0.27)
    ax.set_ylim(-0.04, 1.04)
    ax.set_xticks([0, 0.06, 0.09, 0.12, 0.18, 0.25])
    ax.set_xlabel(r"Planted effect $\beta$ (s.d. per active day)")
    ax.set_ylabel("Worlds where B3 passes")
    _grid(ax)
    h, l = ax.get_legend_handles_labels()
    order = [l.index(x) for x in ("OLS, recurring names", "OLS, size-adjusted", "OLS, window-specific names",
                                  "TFT, recurring names (3 worlds)")]
    ax.legend([h[i] for i in order], [l[i] for i in order], loc="lower center", frameon=False,
              bbox_to_anchor=(0.45, 1.0), ncol=2, columnspacing=1.0, handlelength=1.8)
    fig.savefig(IMG / "fig_pc.pdf")
    plt.close(fig)


def main():
    fig_b1()
    fig_power()
    fig_recurrence()
    fig_tracking()
    fig_counterfactual()
    fig_estimators()
    fig_pc()
    (DATA / "figure_data.json").write_text(json.dumps(FIG_DATA, indent=1))
    print(json.dumps({k: (v if k in ("b1", "counterfactual", "recurrence") else "...") for k, v in FIG_DATA.items()},
                     indent=1))


if __name__ == "__main__":
    main()
