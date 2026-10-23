#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Positive control: aggregation (design 2026-10-08, step 6).

Reads the OLS JSONL (positive_control_ols.py) and the TFT queue's work directory
(positive_control_tft.py) and computes, with Clopper-Pearson 95% intervals:

  G-leak  mean B1 excess and mean placebo excess over the M0 worlds, each with a
          99.5% t-interval that must cover 0 (read first: --gleak-only reads B1
          fields of M0 rows only and nothing else)
  S       composite B3 false passes in the null worlds (M0 + M3): <= 2.5%
  N       RO and DE false passes, each <= 2.5% of the null worlds
  P       B3 passes in >= 160/200 M1 worlds (beta 0.25)
  curve   B3 passes at beta 0.06, 0.09, 0.12, 0.18, 0.25
  size-adjusted power (only with all 1,000 null worlds): RO uses c_RO on its
          pooled p (with >= 2 positive origins), DE uses c_DE on the pair
          (p_cluster_boot, -pooled agreement) ordered lexicographically; the
          thresholds are the largest values such that <= 1% of the null worlds
          pass. Operationally (the set of such c is open on the right): a world
          passes iff its statistic is strictly below the 11th smallest null
          value (k = floor(0.01 N) + 1) among null worlds meeting the test's
          other conditions; if no more than floor(0.01 N) null worlds meet
          them, every world meeting them passes. A test that was not run
          (p = 1) never passes.
  T1/T2   TFT B3 at beta 0.25 (PASS expected) and at beta 0 (FAIL expected), r = 0;
          RO and DE separately; beta 0.12; stretch worlds if present
  reported, not criteria: IN, HET, RR, EV, EV0; B1 readings; B2 (b_hat/b*, sign
          agreement with the truth, tracking against the b = 0 engine); not-run
          counts; DE cluster counts; the label-only predictor; sign_centered
and fills the design's fixed sentences.

Rows finished after --finished-before, when given, are dropped (reported as not run).

Usage:
  python positive_control_aggregate.py --ols results.jsonl --gleak-only
  python positive_control_aggregate.py --ols results.jsonl --tft-work runs/tft --out DIR
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

NULL_ARMS = ("M0", "M3")
CURVE = [(0.25, "c"), (0.18, "d"), (0.12, "e"), (0.09, "g"), (0.06, "f")]


def cp(k: int, n: int, alpha: float = 0.05):
    from scipy.stats import beta
    if n == 0:
        return [None, None]
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return [lo, hi]


def rate(k, n):
    return {"k": int(k), "n": int(n), "rate": (k / n) if n else None, "cp95": cp(int(k), int(n))}


def load_rows(path: Path, finished_before):
    rows, late, errors, seen = [], [], [], set()
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r.get("status") != "ok":
            errors.append(r.get("tag"))
            continue
        if r["tag"] in seen:
            continue
        if finished_before is not None and datetime.fromisoformat(r["finished_at"]) > finished_before:
            late.append(r["tag"])
            continue
        seen.add(r["tag"])
        rows.append(r)
    errors = [t for t in errors if t not in seen]
    return rows, late, errors


def by_arm_beta(rows):
    g = {}
    for r in rows:
        g.setdefault((r["arm"], float(r["beta"])), []).append(r)
    return g


def gleak(rows, conf=0.995, arm="M0", r_lt=None):
    from scipy.stats import t as student_t
    m0 = [r for r in rows if r["arm"] == arm and r["beta"] == 0.0 and (r_lt is None or r["r"] < r_lt)]
    out = {"n_worlds": len(m0), "confidence": conf}
    for key, f in (("b1_excess", lambda r: r["b1"]["excess"]), ("placebo_excess", lambda r: r["b1"]["placebo_excess"])):
        x = np.array([f(r) for r in m0], dtype=float)
        n = len(x)
        if n < 2:
            out[key] = None
            continue
        m, s = float(x.mean()), float(x.std(ddof=1))
        h = float(student_t.ppf(0.5 + conf / 2, n - 1) * s / math.sqrt(n))
        out[key] = {"mean": m, "sd": s, "interval": [m - h, m + h], "covers_0": bool(m - h <= 0 <= m + h)}
    out["gate_pass"] = bool(out.get("b1_excess") and out.get("placebo_excess")
                            and out["b1_excess"]["covers_0"] and out["placebo_excess"]["covers_0"])
    out["complete"] = len(m0) == 200
    return out


# ── size adjustment ──────────────────────────────────────────────────────

def ro_stat(r):
    ro = r["ro"]
    run = ro["status"] in ("PASS", "FAIL")
    p = ro["p"] if (run and ro["p"] is not None) else 1.0
    return run, p, (run and ro["pos_origins"] >= 2)


def de_stat(r):
    de = r["de"]
    run = de["status"] in ("PASS", "FAIL")
    pooled = de["pooled"] or {}
    p = pooled.get("p_cluster_boot", 1.0) if run else 1.0
    agree = pooled.get("agree", 0.0) if run else 0.0
    cond = run and agree >= 0.60 and not de["bad_classes"]
    return run, (p, -agree), cond


def thresholds(null_rows):
    N = len(null_rows)
    k = int(math.floor(0.01 * N))
    ro_vals = sorted(ro_stat(r)[1] for r in null_rows if ro_stat(r)[2])
    de_vals = sorted(de_stat(r)[1] for r in null_rows if de_stat(r)[2])
    c_ro = ro_vals[k] if len(ro_vals) > k else None              # pass iff p < c_ro (None: all pass)
    c_de = de_vals[k] if len(de_vals) > k else None
    return {"N_null": N, "k_max_null_passes": k, "c_RO_exclusive": c_ro, "c_DE_exclusive": c_de,
            "n_null_meeting_RO_conditions": len(ro_vals), "n_null_meeting_DE_conditions": len(de_vals)}


def adj_pass(r, th):
    _, p, ok_ro = ro_stat(r)
    _, key, ok_de = de_stat(r)
    ro = ok_ro and (th["c_RO_exclusive"] is None or p < th["c_RO_exclusive"])
    de = ok_de and (th["c_DE_exclusive"] is None or tuple(key) < tuple(th["c_DE_exclusive"]))
    return ro, de, ro and de


# ── per-cell summaries ───────────────────────────────────────────────────

def cell_summary(rs, th=None):
    n = len(rs)
    b3 = sum(r["b3"] == "PASS" for r in rs)
    ro_pass = sum(r["ro"]["status"] == "PASS" for r in rs)
    de_pass = sum(r["de"]["status"] == "PASS" for r in rs)
    ro_nr = sum(r["ro"]["status"].startswith("NOT_RUN") for r in rs)
    ro_nr0 = sum(r["ro"]["status"] == "NOT_RUN_0" for r in rs)
    de_nr = sum(r["de"]["status"] == "NOT_RUN" for r in rs)
    reruns = sum(bool(r.get("rerun")) for r in rs)
    b1 = [r["b1"] for r in rs]

    def b1_reading(x):
        lo, hi = x["excess_ci95"]
        plo, phi = x["placebo_ci95"]
        if lo > 0 or hi < 0:
            return "undetermined (excess interval excludes 0)"
        return "pass" if plo <= 0 <= phi else "placebo off baseline"
    readings = {}
    for x in b1:
        readings[b1_reading(x)] = readings.get(b1_reading(x), 0) + 1
    agree_read = {}
    for x in b1:
        agree_read[x["reading"]] = agree_read.get(x["reading"], 0) + 1
    clusters = [r["de"]["pooled"]["n_clusters"] for r in rs if r["de"]["pooled"]]
    fed_cl = {c: [r["de"]["per_class"][c]["n_clusters"] for r in rs if r["de"]["per_class"].get(c)]
              for c in ("E1_tightening", "E2_easing")}
    lo_mean = [np.mean([v["mean_label_only_agreement"] for v in r["ro"]["per_origin"].values()
                        if v.get("mean_label_only_agreement") is not None]) for r in rs
               if any(v.get("mean_label_only_agreement") is not None for v in r["ro"]["per_origin"].values())]
    out = {"n": n, "B3": rate(b3, n), "RO_pass": rate(ro_pass, n), "DE_pass": rate(de_pass, n),
           "DE_alone_pass": rate(de_pass, n), "RO_not_run": ro_nr, "RO_not_run_no_stock": ro_nr0,
           "DE_not_run": de_nr, "exp_n_incomplete_reruns": reruns,
           "RO_entering_median": float(np.median([r["ro"]["n_entering"] for r in rs])) if n else None,
           "DE_tested_clusters_median": float(np.median(clusters)) if clusters else None,
           "fed_class_clusters": {c: sorted(set(v)) for c, v in fed_cl.items()},
           "DE_sign_centered_pass": rate(sum(r["de"]["sign_centered"]["status"] == "PASS" for r in rs), n),
           "B1": {"readings": readings, "agreement_readings": agree_read,
                  "p_label_shuffle_lt_0.05": rate(sum(x["p_label_shuffle"] < 0.05 for x in b1), n),
                  "mean_excess": float(np.mean([x["excess"] for x in b1])) if n else None,
                  "mean_placebo_excess": float(np.mean([x["placebo_excess"] for x in b1])) if n else None},
           "label_only_mean_agreement": float(np.mean(lo_mean)) if lo_mean else None,
           "label_only_pass": rate(sum(bool((r["ro"]["label_only"] or {}).get("pass")) for r in rs), n)}
    if th is not None:
        a = [adj_pass(r, th) for r in rs]
        out["size_adjusted"] = {"RO": rate(sum(x[0] for x in a), n), "DE": rate(sum(x[1] for x in a), n),
                                "B3": rate(sum(x[2] for x in a), n)}
    return out


# ── TFT worlds ───────────────────────────────────────────────────────────

def tft_world(work: Path, tag: str, ols_rows_by_tag: dict):
    from positive_control_ols import parse_outputs
    out = work / "out" / tag
    res = {"tag": tag, "present": out.exists()}
    if not out.exists():
        return res
    for est in ("tft", "ols"):
        try:
            res[est] = parse_outputs(out, tag, est)
        except FileNotFoundError as e:
            res[est] = {"missing": str(e)}
    man_f = out / f"manifest_{tag}.json"
    man = json.loads(man_f.read_text()) if man_f.exists() else None
    if man:
        # identical data to the M-arm world of the same r and beta
        arm = "M1" if man["beta"] == 0.25 else "M0" if man["beta"] == 0 else "M2"
        from positive_control_worlds import tag_of
        mrow = ols_rows_by_tag.get(tag_of(arm, man["r"], man["beta"]))
        res["same_data_as"] = None if mrow is None else {
            "tag": mrow["tag"], "prices_hash_equal": mrow["prices_hash"] == man["prices_hash"],
            "label_hash_equal": mrow["label_hash"] == man["label_hash"],
            "ols_ro_p_equal": (res.get("ols", {}).get("ro", {}).get("p") == mrow["ro"]["p"]),
            "ols_de_pooled_equal": (res.get("ols", {}).get("de", {}).get("pooled") == mrow["de"]["pooled"])}
        # B2: b_hat / b* and sign agreement (P2 matrices vs the truth)
        ratios, agree = [], []
        root = work / "root" / tag
        for t, tr in man["truth"].items():
            f = root / t / "impact_matrix.json"
            if not f.exists():
                continue
            im = json.loads(f.read_text())
            if im.get("estimator") != "ablation":
                continue
            bh = dict(zip(im["factor_names"], [row[3] for row in im["base_impact"]]))
            for n, b in tr.items():
                if n == "market_noise" or b == 0:
                    continue
                ratios.append(bh[n] / b)
                agree.append(np.sign(bh[n]) == np.sign(b))
        res["b2_truth"] = {"n_factors": len(ratios),
                           "median_bhat_over_bstar": float(np.median(ratios)) if ratios else None,
                           "sign_agreement": float(np.mean(agree)) if agree else None}
        # tracking against the b = 0 engine (per_ticker.csv)
        import pandas as pd
        f_t, f_0 = out / f"b2_counterfactual__{tag}" / "per_ticker.csv", out / f"b2_counterfactual__{tag}_b0eng" / "per_ticker.csv"
        if f_t.exists() and f_0.exists():
            a, b = pd.read_csv(f_t).set_index("ticker"), pd.read_csv(f_0).set_index("ticker")
            d = (a.track_err_factual.abs() - b.track_err_factual.abs())
            res["b2_tracking_vs_b0"] = {"median_abs_track_tft": float(a.track_err_factual.abs().median()),
                                        "median_abs_track_b0": float(b.track_err_factual.abs().median()),
                                        "median_diff": float(d.median()),
                                        "stocks_tft_closer": int((d < 0).sum()), "n": int(len(d))}
        for suffix in ("", "_oracle", "_b0eng"):
            f = out / f"b2_counterfactual__{tag}{suffix}" / "report.json"
            if f.exists():
                res[f"exp_o{suffix or '_tft'}"] = json.loads(f.read_text())["summary"]
    pw = out / f"sign_recovery__{tag}" / "summary_majority5.json"
    if pw.exists():
        res["sign_recovery_majority5"] = json.loads(pw.read_text())["per_beta"]
    return res


def fmt(x, nd=3):
    return "n/a" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def worlds_csv(rows, th, path: Path):
    """One compact line per scored world (the per-world record kept in the repo)."""
    import pandas as pd
    out = []
    for r in rows:
        ro, de, b1 = r["ro"], r["de"], r["b1"]
        a = adj_pass(r, th) if th else (None, None, None)
        out.append({"tag": r["tag"], "arm": r["arm"], "r": r["r"], "beta": r["beta"], "noise": r["noise"],
                    "names": r["names"], "label_sub": r["label_sub"], "label_hash": r["label_hash"][:16],
                    "noise_hash": r["noise_hash"][:16], "prices_hash": r["prices_hash"][:16],
                    "B3": r["b3"], "RO": ro["status"], "RO_p": ro["p"], "RO_pos_origins": ro["pos_origins"],
                    "RO_entering": ro["n_entering"], "RO_mean_diff": ro["mean_diff"],
                    "DE": de["status"], "DE_agree": (de["pooled"] or {}).get("agree"),
                    "DE_p_cluster_boot": (de["pooled"] or {}).get("p_cluster_boot"),
                    "DE_n": (de["pooled"] or {}).get("n"), "DE_n_clusters": (de["pooled"] or {}).get("n_clusters"),
                    "DE_bad_classes": ";".join(de["bad_classes"]), "DE_centered": de["sign_centered"]["status"],
                    "label_only_pass": bool((ro["label_only"] or {}).get("pass")),
                    "B1_excess": b1["excess"], "B1_excess_lo": b1["excess_ci95"][0], "B1_excess_hi": b1["excess_ci95"][1],
                    "B1_placebo_excess": b1["placebo_excess"], "B1_placebo_lo": b1["placebo_ci95"][0],
                    "B1_placebo_hi": b1["placebo_ci95"][1], "B1_p_label_shuffle": b1["p_label_shuffle"],
                    "adj_RO": a[0], "adj_DE": a[1], "adj_B3": a[2], "rerun": r.get("rerun"),
                    "finished_at": r["finished_at"]})
    pd.DataFrame(out).sort_values(["arm", "beta", "r"]).to_csv(path, index=False)


def figure(rep, path_base: Path):
    """B3 pass rate against beta (least squares, noise C), nominal and size-adjusted."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, ORANGE, INK2, MUTED, GRID = "#2a78d6", "#eb6834", "#52514e", "#a3a29c", "#e6e5e1"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
                         "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
    nul = rep["S"]
    pts = [(0.0, nul["k"], nul["n"], None)] + [(v["beta"], v["k"], v["n"], v["size_adjusted"])
                                                for v in sorted(rep["curve"].values(), key=lambda x: x["beta"])]
    pts = [p for p in pts if p[2]]
    fig, ax = plt.subplots(figsize=(3.33, 2.2))
    ax.axvspan(0.06, 0.12, color="#f0efec", zorder=0)
    ax.axhline(0.80, color=INK2, lw=0.8, ls=(0, (4, 3)))
    x = [p[0] for p in pts]
    y = [p[1] / p[2] for p in pts]
    ci = [cp(p[1], p[2]) for p in pts]
    ax.errorbar(x, y, yerr=[[yy - c[0] for yy, c in zip(y, ci)], [c[1] - yy for yy, c in zip(y, ci)]],
                color=BLUE, marker="o", ms=4.5, mec="white", mew=0.8, capsize=0, elinewidth=1.0,
                label="B3 pass rate (least squares)")
    adj = [(p[0], p[3] / p[2]) for p in pts if p[3] is not None]
    if adj:
        ax.plot([a[0] for a in adj], [a[1] for a in adj], color=ORANGE, marker="s", ms=4, mec="white", mew=0.8,
                label="size-adjusted")
    ax.set_xlabel(r"Planted effect $\beta$ (s.d. per active day)")
    ax.set_ylabel("Share of worlds where B3 passed")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks([0, 0.06, 0.09, 0.12, 0.18, 0.25])
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, fontsize=6.8)
    for ext in ("pdf", "png"):
        fig.savefig(f"{path_base}.{ext}", dpi=200)
    plt.close(fig)


# ── real-calendar setting: arm CAL (calendar-only taxonomy) ────────────────

CAL_CURVE = [(0.25, "c"), (0.18, "d"), (0.12, "e"), (0.09, "g"), (0.06, "f")]


def m_tag_for(r, beta):
    from positive_control_worlds import tag_of
    if beta == 0:
        return tag_of("M0" if r < 200 else "M3", r, 0.0)
    return tag_of("M1" if beta == 0.25 else "M2", r, beta)


def cal_main(a, finished_before):
    import hashlib
    rows, late, errors = load_rows(Path(a.ols), finished_before)
    arm_name = a.cal_arm
    rows = [r for r in rows if r["arm"] == arm_name]
    for r in rows:                                                    # CALF is reported under the CAL keys
        r["arm"] = "CAL"
    mrows = {r["tag"]: r for r in load_rows(Path(a.pair), finished_before)[0]} if a.pair else {}
    if a.gleak_only:
        print(json.dumps({f"gleak_{arm_name}": gleak(rows, arm="CAL", r_lt=200), "late": len(late)}, indent=1))
        return
    g = by_arm_beta(rows)
    null = g.get(("CAL", 0.0), [])
    N = len(null)
    rep = {"generated": datetime.now().astimezone().isoformat(timespec="seconds"), "finished_before": a.finished_before or None,
           "aggregator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "arm": ("CAL (seven real-calendar types + market_noise)" if arm_name == "CAL" else
                   f"{arm_name} (CAL with a per-world microstructure offset)"),
           "rows_ok": len(rows), "rows_late_dropped": late, "tags_with_errors_only": errors,
           "gleak": gleak(rows, arm="CAL", r_lt=200),
           "gleak_all_null_worlds": gleak(rows, arm="CAL")}
    s = sum(r["b3"] == "PASS" for r in null)
    a_ = sum(r["ro"]["status"] == "PASS" for r in null)
    b_ = sum(r["de"]["status"] == "PASS" for r in null)
    lim = 0.025 * N
    rep["S"] = {**rate(s, N), "limit": lim, "holds": bool(N and s <= lim)}
    rep["N"] = {"RO": {**rate(a_, N), "holds": bool(N and a_ <= lim)},
                "DE": {**rate(b_, N), "holds": bool(N and b_ <= lim)}}
    rep["N"]["holds"] = rep["N"]["RO"]["holds"] and rep["N"]["DE"]["holds"]
    top = g.get(("CAL", 0.25), [])
    c = sum(r["b3"] == "PASS" for r in top)
    rep["P"] = {**rate(c, len(top)), "holds": bool(top and c / len(top) >= 0.80)}
    th = thresholds(null) if N == 1000 else None
    rep["size_adjusted_thresholds"] = th if th else f"not run ({N} of 1,000 null worlds)"
    if th:
        an = [adj_pass(r, th) for r in null]
        rep["size_adjusted_null_rates"] = {"RO": rate(sum(x[0] for x in an), N), "DE": rate(sum(x[1] for x in an), N),
                                           "B3": rate(sum(x[2] for x in an), N)}
    rep["cells"] = {f"CAL@{b:g}": cell_summary(rs, th) for (arm, b), rs in sorted(g.items())}
    curve = {}
    for b, letter in CAL_CURVE:
        rs = g.get(("CAL", b), [])
        k = sum(r["b3"] == "PASS" for r in rs)
        ka = sum(adj_pass(r, th)[2] for r in rs) if th else None
        curve[letter] = {"beta": b, **rate(k, len(rs)), "size_adjusted": ka,
                         "size_adjusted_cp95": cp(ka, len(rs)) if th else None,
                         "RO_not_run": sum(r["ro"]["status"].startswith("NOT_RUN") for r in rs),
                         "DE_alone_pass": sum(r["de"]["status"] == "PASS" for r in rs)}
    rep["curve"] = curve
    # rolling origin's eligible factors per stock-origin (b3_rolling_origin's n_eligible_factors) and entering stocks
    elig = {}
    for r in rows:
        for o, v in r["ro"]["per_origin"].items():
            for n_, cnt in (v.get("n_eligible_hist") or {}).items():
                elig.setdefault(o, {}).setdefault(int(n_), 0)
                elig[o][int(n_)] += cnt

    def med(h):
        xs = sorted(k for k, cnt in h.items() for _ in range(cnt))
        return xs[len(xs) // 2] if xs else None
    allh = {}
    for h in elig.values():
        for k, cnt in h.items():
            allh[k] = allh.get(k, 0) + cnt
    rep["eligible_factors_per_stock_origin"] = {
        "by_origin": {o: {"hist": dict(sorted(h.items())), "median": med(h),
                          "share_ge5": sum(cnt for k, cnt in h.items() if k >= 5) / max(sum(h.values()), 1)}
                      for o, h in sorted(elig.items())},
        "all": {"hist": dict(sorted(allh.items())), "median": med(allh),
                "share_ge5": sum(cnt for k, cnt in allh.items() if k >= 5) / max(sum(allh.values()), 1)}}
    ent = [r["ro"]["n_entering"] for r in rows]
    rep["RO_entering"] = {"median": float(np.median(ent)) if ent else None, "min": min(ent) if ent else None,
                          "max": max(ent) if ent else None, "worlds_lt6": sum(e < 6 for e in ent), "n": len(ent)}
    u = sum(r["ro"]["status"].startswith("NOT_RUN") for r in rows)
    w = sum(r["de"]["status"] == "NOT_RUN" for r in rows)
    rep["not_run"] = {"RO": u, "DE": w, "N": len(rows),
                      "RO_by_beta": {f"{b:g}": sum(r["ro"]["status"].startswith("NOT_RUN") for r in rs)
                                     for (_, b), rs in sorted(g.items())}}
    # paired comparison with the M arms (same r and beta)
    pairs = {}
    for (arm, b), rs in sorted(g.items()):
        both = m_only = c_only = neither = missing = 0
        hash_ok = True
        for r in rs:
            m = mrows.get(m_tag_for(r["r"], b))
            if m is None:
                missing += 1
                continue
            hash_ok &= (m["label_hash_recurring"] == r["label_hash_recurring"] and m["noise_hash"] == r["noise_hash"]
                        and (b != 0 or m["prices_hash"] == r["prices_hash"]))
            pm, pc_ = m["b3"] == "PASS", r["b3"] == "PASS"
            both += pm and pc_
            m_only += pm and not pc_
            c_only += pc_ and not pm
            neither += (not pm) and (not pc_)
        pairs[f"{b:g}"] = {"both": both, "M_only": m_only, "CAL_only": c_only, "neither": neither,
                           "missing_M_rows": missing, "hashes_match_M": bool(hash_ok)}
    rep["paired_with_M"] = pairs
    Mc = {"f": 89, "g": 155, "e": 177}                                   # main-setting counts (M arms)
    if mrows:
        for b, letter in CAL_CURVE:
            Mc[letter] = sum(mrows[m_tag_for(r_, b)]["b3"] == "PASS" for r_ in range(200) if m_tag_for(r_, b) in mrows)
    rep["M_curve_same_seeds"] = Mc
    # readings fixed for the real-calendar setting
    C = {k: v["k"] for k, v in curve.items()}
    Cn = {k: v["n"] for k, v in curve.items()}
    Ca = {k: v["size_adjusted"] for k, v in curve.items()}
    adj = (lambda L: ", ".join(str(Ca[x]) for x in L)) if th else (lambda L: "not run")
    adj_and = (lambda L: ", ".join(str(Ca[x]) for x in L[:-1]) + f" and {Ca[L[-1]]}") if th else (lambda L: "not run")
    ns = sorted({Cn[x] for x in "cdegf"})
    of_n = str(ns[0]) if len(ns) == 1 else ", ".join(str(Cn[x]) for x in "cdegf")
    j_ = {k: v["DE_alone_pass"] for k, v in curve.items()}
    med_all = rep["eligible_factors_per_stock_origin"]["all"]["median"]
    sent = {
        "S_N_P_hold": (f"With only real-calendar event types (seven per stock), B3's tests rejected in {a_}/{N:,} (RO) and "
                       f"{b_}/{N:,} (DE) worlds without effects, and B3 in {s}/{N:,}. B3 passed in {C['c']}, {C['d']}, "
                       f"{C['e']}, {C['g']} and {C['f']} of {of_n} worlds at β = 0.25, 0.18, 0.12, 0.09 and 0.06 "
                       f"(least squares; size-adjusted {adj('cdegf')})."),
        "N_fails": (f"With only real-calendar event types, RO rejected in {a_}/{N:,} worlds without effects, above its "
                    f"nominal 1%, and DE in {b_}/{N:,}; B3 stayed at {s}/{N:,}. Size-adjusted, B3 passed in "
                    f"{adj_and('cdegf')} of {of_n} worlds at β = 0.25, 0.18, 0.12, 0.09 and 0.06 (least squares)."),
        "S_fails": f"With only real-calendar event types, B3 passed in {s}/{N:,} worlds without effects (limit 25/1,000).",
        "P_fails_RO_not_run": (f"With seven real-calendar types, the rolling-origin test could not run in {u}/{len(rows):,} "
                               f"worlds, so B3 passed in only {C['c']}/{Cn['c']} at β = 0.25; DE alone passed in "
                               f"{j_['c']}/{Cn['c']}."),
        "P_fails_otherwise": (f"With only real-calendar event types, B3 passed in only {C['c']}/{Cn['c']} worlds at β = 0.25, "
                              f"where every test is expected to pass. The defect found was …"),
        "curve": (f"With only real-calendar event types, B3 passed in {C['f']}/{Cn['f']}, {C['g']}/{Cn['g']} and "
                  f"{C['e']}/{Cn['e']} worlds at β = 0.06, 0.09 and 0.12 s.d. per active day (size-adjusted {adj('fge')}), "
                  f"against {Mc['f']}/200, {Mc['g']}/200 and {Mc['e']}/200 with the five synthetic types (same seeds). "
                  f"At 0.18 and 0.25 it passed in {C['d']}/{Cn['d']} and {C['c']}/{Cn['c']}."),
        "RO_mostly_not_run": (f"With seven real-calendar types, the rolling-origin test could not run in {u}/{len(rows):,} "
                              f"worlds, so B3 cannot pass on such panels whatever the effects; DE alone passed in "
                              f"{j_['c']}/{Cn['c']}, {j_['e']}/{Cn['e']} and {j_['f']}/{Cn['f']} at β = 0.25, 0.12 and 0.06."),
        "not_run": (f"RO was not run in {u}, and DE in {w}, of {len(rows):,} worlds with only real-calendar types; "
                    f"stock-origins had a median of {med_all} eligible factors (10–11 with the synthetic types)."),
    }
    applicable = []
    S_, N_, P_ = rep["S"]["holds"], rep["N"]["holds"], rep["P"]["holds"]
    if S_ and N_ and P_:
        applicable.append("S_N_P_hold")
    if (not N_) and S_ and P_:
        applicable.append("N_fails")
    if N and not S_:
        applicable.append("S_fails")
    top_nr = sum(r["ro"]["status"].startswith("NOT_RUN") for r in top)
    if top and not P_:
        applicable.append("P_fails_RO_not_run" if top_nr > len(top) / 2 else "P_fails_otherwise")
    applicable.append("curve")
    if rows and u > len(rows) / 2:
        applicable.append("RO_mostly_not_run")
    applicable.append("not_run")
    rep["sentences"] = sent
    rep["sentences_applicable"] = applicable
    txt = json.dumps(rep, indent=1, default=str)
    if a.out:
        od = Path(a.out)
        od.mkdir(parents=True, exist_ok=True)
        pre = "pc_cal" if arm_name == "CAL" else f"pc_{arm_name.lower()}"
        (od / f"{pre}_summary.json").write_text(txt)
        lines = ["# Positive control, arm CAL (real-calendar setting): filled sentences", "",
                 f"Generated {rep['generated']}; rows finished before {a.finished_before or 'any time'}.", ""]
        for k in applicable:
            lines += [f"**{k}**", "", sent[k], ""]
        lines += ["## Not applicable (filled for reference)", ""]
        for k, v in sent.items():
            if k not in applicable:
                lines += [f"**{k}**", "", v, ""]
        lines[0] = f"# Positive control, arm {arm_name}: filled sentences (real-calendar readings)"
        (od / f"{pre}_sentences.md").write_text("\n".join(lines))
        worlds_csv(rows, th, od / f"{pre}_worlds.csv")
        figure(rep, od / f"fig_{pre}_b3_curve")
    print(txt)


B3_FIELDS = (("ro", "status"), ("ro", "p"), ("ro", "pos_origins"), ("ro", "n_entering"), ("ro", "mean_diff"),
             ("ro", "verdict"), ("ro", "label_only"), ("de", "status"), ("de", "pooled"), ("de", "per_class"),
             ("de", "bad_classes"), ("de", "sign_centered"), ("b3",))


def compare_b3(a, finished_before):
    """Number of worlds where any least-squares B3 field differs between the
    fixed (CALF) and the unfixed (CAL) world with the same r and beta (expected 0)."""
    fx = load_rows(Path(a.compare[0]), finished_before)[0]
    un = {(r["r"], r["beta"]): r for r in load_rows(Path(a.compare[1]), finished_before)[0] if r["arm"] == "CAL"}
    per_field, differ, missing, n = {}, [], 0, 0
    for r in fx:
        if r["arm"] != "CALF":
            continue
        u = un.get((r["r"], r["beta"]))
        if u is None:
            missing += 1
            continue
        n += 1
        bad = False
        for path in B3_FIELDS:
            x, y = r, u
            for k in path:
                x, y = x[k], y[k]
            if x != y:
                per_field[".".join(path)] = per_field.get(".".join(path), 0) + 1
                bad = True
        same_inputs = (r["noise_hash"] == u["noise_hash"] and r["label_hash"] == u["label_hash"])
        if not same_inputs:
            per_field["inputs(noise/label hash)"] = per_field.get("inputs(noise/label hash)", 0) + 1
            bad = True
        if bad:
            differ.append(r["tag"])
    th_f = thresholds([r for r in fx if r["arm"] == "CALF" and r["beta"] == 0.0])
    th_u = thresholds([r for r in un.values() if r["beta"] == 0.0])
    out = {"worlds_compared": n, "missing_unfixed": missing, "worlds_with_any_B3_field_different": len(differ),
           "per_field": per_field, "examples": differ[:10],
           "size_adjusted_thresholds_equal": th_f == th_u, "thresholds_fixed": th_f, "thresholds_unfixed": th_u}
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "calf_vs_cal_b3_comparison.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ols", required=True)
    ap.add_argument("--tft-work", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--finished-before", default="", help="optional ISO time; later rows are dropped")
    ap.add_argument("--gleak-only", action="store_true")
    ap.add_argument("--cal", action="store_true", help="report arm CAL (real-calendar setting)")
    ap.add_argument("--pair", default="", help="--cal: the M arms' JSONL for the paired comparison")
    ap.add_argument("--cal-arm", default="CAL", help="--cal: CAL or CALF (regenerated)")
    ap.add_argument("--compare", nargs=2, default=None, metavar=("FIXED_JSONL", "UNFIXED_JSONL"),
                    help="world-by-world comparison of the B3 fields of CALF and CAL")
    a = ap.parse_args()
    finished_before = datetime.fromisoformat(a.finished_before) if a.finished_before else None
    if a.compare:
        compare_b3(a, finished_before)
        return
    if a.cal:
        cal_main(a, finished_before)
        return
    rows, late, errors = load_rows(Path(a.ols), finished_before)
    if a.gleak_only:
        print(json.dumps({"gleak": gleak(rows), "late": len(late)}, indent=1))
        return
    g = by_arm_beta(rows)
    import hashlib
    rep = {"generated": datetime.now().astimezone().isoformat(timespec="seconds"), "finished_before": a.finished_before or None,
           "aggregator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "rows_ok": len(rows), "rows_late_dropped": late, "tags_with_errors_only": errors,
           "gleak": gleak(rows)}
    null = [r for r in rows if r["arm"] in NULL_ARMS and r["beta"] == 0.0]
    N = len(null)
    s = sum(r["b3"] == "PASS" for r in null)
    a_ = sum(r["ro"]["status"] == "PASS" for r in null)
    b_ = sum(r["de"]["status"] == "PASS" for r in null)
    lim = 0.025 * N
    rep["S"] = {**rate(s, N), "limit": lim, "holds": bool(N and s <= lim)}
    rep["N"] = {"RO": {**rate(a_, N), "holds": bool(N and a_ <= lim)},
                "DE": {**rate(b_, N), "holds": bool(N and b_ <= lim)}}
    rep["N"]["holds"] = rep["N"]["RO"]["holds"] and rep["N"]["DE"]["holds"]
    m1 = g.get(("M1", 0.25), [])
    c = sum(r["b3"] == "PASS" for r in m1)
    rep["P"] = {**rate(c, len(m1)), "holds": bool(m1 and c / len(m1) >= 0.80)}
    th = thresholds(null) if N == 1000 else None
    rep["size_adjusted_thresholds"] = th if th else f"not run ({N} of 1,000 null worlds)"
    if th:
        an = [adj_pass(r, th) for r in null]
        rep["size_adjusted_null_rates"] = {"RO": rate(sum(x[0] for x in an), N), "DE": rate(sum(x[1] for x in an), N),
                                           "B3": rate(sum(x[2] for x in an), N)}
    rep["cells"] = {f"{arm}@{b:g}": cell_summary(rs, th) for (arm, b), rs in sorted(g.items())}
    rep["null_cell"] = cell_summary(null, th)
    curve = {}
    for b, letter in CURVE:
        rs = g.get(("M1" if b == 0.25 else "M2", b), [])
        k = sum(r["b3"] == "PASS" for r in rs)
        curve[letter] = {"beta": b, **rate(k, len(rs)),
                         "size_adjusted": (sum(adj_pass(r, th)[2] for r in rs) if th else None),
                         "size_adjusted_cp95": (cp(sum(adj_pass(r, th)[2] for r in rs), len(rs)) if th else None)}
    rep["curve"] = curve

    # TFT
    tft = {}
    if a.tft_work:
        by_tag = {r["tag"]: r for r in rows}
        for tag in sorted(p.name for p in (Path(a.tft_work) / "out").iterdir() if p.name.startswith("pc_TFT")):
            tft[tag] = tft_world(Path(a.tft_work), tag, by_tag)
    rep["tft"] = tft

    def tstat(tag):
        w = tft.get(tag) or {}
        x = w.get("tft") or {}
        if "ro" not in x:
            return None
        return {"B3": x["b3"], "RO": x["ro"]["status"], "RO_p": x["ro"]["p"], "RO_entering": x["ro"]["n_entering"],
                "DE": x["de"]["status"], "DE_pooled": x["de"]["pooled"]}
    t1, t2, t12 = tstat("pc_TFT_r000_b0.25"), tstat("pc_TFT_r000_b0"), tstat("pc_TFT_r000_b0.12")
    rep["T1"] = {"world": t1, "holds": (t1 or {}).get("B3") == "PASS" if t1 else None}
    rep["T2"] = {"world": t2, "holds": (t2 or {}).get("B3") != "PASS" if t2 else None}
    rep["TFT_beta_0.12"] = t12
    # TFT verdicts other than T1/T2 are reported with their powered status (design: "read at a powered beta")
    rep["TFT_other_worlds"] = {tag: tstat(tag) for tag in sorted(tft) if tag not in
                               ("pc_TFT_r000_b0.25", "pc_TFT_r000_b0")}
    w0 = tft.get("pc_TFT_r000_b0") or {}
    sr = w0.get("sign_recovery_majority5")
    rep["TFT_powered"] = None if not sr else {
        b: {"grid": gb, "majority_sign_recovery": sr[gb]["majority_sign_recovery"],
            "powered": sr[gb]["majority_sign_recovery"] >= 0.80}
        for b, gb in (("0.25", "0.2"), ("0.18", "0.1"), ("0.12", "0.1"), ("0.09", "0.05"), ("0.06", "0.05"))}
    for tag, w in rep["TFT_other_worlds"].items():
        if w is not None:
            b = tag.rsplit("_b", 1)[1]
            w["powered_beta"] = (None if b == "0" or not rep["TFT_powered"]
                                 else rep["TFT_powered"].get(b, {}).get("powered"))
    if t1 is not None and rep["TFT_powered"]:
        rep["T1"]["read_at_powered_beta"] = rep["TFT_powered"]["0.25"]["powered"]

    # fixed sentences
    C = {k: v["k"] for k, v in curve.items()}
    Cn = {k: v["n"] for k, v in curve.items()}
    Ca = {k: v["size_adjusted"] for k, v in curve.items()}
    sent = {}
    ns = sorted({Cn[x] for x in "cdegf"})
    of_n = str(ns[0]) if len(ns) == 1 else ", ".join(str(Cn[x]) for x in "cdegf")
    adj = (lambda L: ", ".join(str(Ca[x]) for x in L)) if th else (lambda L: "not run")
    sent["S_N_P_hold"] = (f"On synthetic panels with the real calendar, recurring names and constant planted effects, "
                          f"B3's tests rejected in {a_}/{N:,} (RO) and {b_}/{N:,} (DE) worlds without effects, and B3 in "
                          f"{s}/{N:,}. B3 passed in {C['c']}, {C['d']}, {C['e']}, {C['g']} and {C['f']} of "
                          f"{of_n} worlds at β = 0.25, 0.18, 0.12, 0.09 "
                          f"and 0.06 (least squares; size-adjusted {adj('cdegf')}).")
    sent["curve"] = (f"In the realistic band, B3 passed in {C['f']}/{Cn['f']}, {C['g']}/{Cn['g']} and {C['e']}/{Cn['e']} "
                     f"worlds at β = 0.06, 0.09 and 0.12 s.d. per active day (size-adjusted {adj('fge')}). "
                     f"At 0.18 and 0.25 it passed in {C['d']}/{Cn['d']} and {C['c']}/{Cn['c']}.")
    sent["N_fails"] = (f"… RO rejected in {a_}/{N:,} worlds without effects, above its nominal 1%: stocks share event "
                       f"dates and correlated returns, so its Wilcoxon test over stocks is anti-conservative. B3 stayed at "
                       f"{s}/{N:,} only because DE rarely passed without effects ({b_}/{N:,}). Size-adjusted, B3 passed in "
                       f"{adj('cdegf')} of 200 worlds …")
    sent["N_fails_assembled"] = (
        f"On synthetic panels with the real calendar, recurring names and constant planted effects, RO rejected in "
        f"{a_}/{N:,} worlds without effects, above its nominal 1%: stocks share event dates and correlated returns, so "
        f"its Wilcoxon test over stocks is anti-conservative. B3 stayed at {s}/{N:,} only because DE rarely passed "
        f"without effects ({b_}/{N:,}). Size-adjusted, B3 passed in {adj('cdegf')} of {of_n} worlds at β = 0.25, 0.18, "
        f"0.12, 0.09 and 0.06 (least squares; unadjusted {C['c']}, {C['d']}, {C['e']}, {C['g']} and {C['f']}).")
    sent["S_fails"] = (f"B3 passed in {s}/{N:,} worlds without effects (limit 25/1,000). At this panel size, a B3 pass "
                       f"is not evidence of an effect.")
    sent["P_fails"] = (f"B3 passed in only {c}/{len(m1)} worlds at β = 0.25, where every test is expected to pass. "
                       f"The defect found was …")
    x02 = (rep["TFT_powered"] or {}).get("0.25", {}).get("majority_sign_recovery")
    if t1:
        sent["T1_fails"] = (f"Least squares passed B3 in {C['c']}/{Cn['c']} worlds at β = 0.25, but the five-seed TFT "
                            f"failed on the same world (RO {t1['RO']}, p = {fmt(t1['RO_p'])}; DE {t1['DE']}). Its sign "
                            f"recovery was {fmt(x02, 2)} at 0.2, so the estimator, not the tests, limits B3 at this effect size.")
    if t2:
        sent["T2_fails"] = (f"The TFT passed B3 on the one world without effects (RO {t2['RO']}, DE {t2['DE']}). One world "
                            f"cannot estimate its size. Least squares had a false-pass rate of {s}/{N:,} on such worlds.")
    sent["T1_T2_hold"] = "The five-seed TFT passed B3 at β = 0.25 and failed it at β = 0, one world each."
    allm = [r for r in rows if r["arm"] in ("M0", "M1", "M2", "M3")]
    u = sum(r["ro"]["status"].startswith("NOT_RUN") for r in allm)
    w = sum(r["de"]["status"] == "NOT_RUN" for r in allm)
    kcl = rep["null_cell"]["DE_tested_clusters_median"]
    sent["not_run"] = (f"RO was not run in {u}, and DE in {w}, of {len(allm):,} worlds. Tested DE rows spanned a median "
                       f"of {fmt(kcl, 0)} clusters, and each Fed class was one cluster.")
    rep["not_run_scope"] = {"worlds": "M0, M1, M2, M3 (least squares, noise C, recurring names)", "N": len(allm),
                            "fed_class_clusters_observed": rep["null_cell"]["fed_class_clusters"]}
    ev25, ev12 = g.get(("EV", 0.25), []), g.get(("EV", 0.12), [])
    if ev25 or ev12:
        h = sum(r["b3"] == "PASS" for r in ev25)
        i = sum(r["b3"] == "PASS" for r in ev12)
        u25 = sum(r["ro"]["status"] == "NOT_RUN_0" for r in ev25)
        u12 = sum(r["ro"]["status"] == "NOT_RUN_0" for r in ev12)
        j = sum(r["de"]["status"] == "PASS" for r in ev25)
        k2 = sum(r["de"]["status"] == "PASS" for r in ev12)
        sent["EV"] = (f"With the same effects and window-specific names, B3 passed in {h}/{len(ev25)} worlds at β = 0.25 "
                      f"and {i}/{len(ev12)} at 0.12. RO had no eligible stock in {u25}/{len(ev25)} and {u12}/{len(ev12)}, and DE alone "
                      f"passed in {j}/{len(ev25)} and {k2}/{len(ev12)}.")
    applicable = []
    if rep["S"]["holds"] and rep["N"]["holds"] and rep["P"]["holds"]:
        applicable.append("S_N_P_hold")
    if (not rep["N"]["holds"]) and rep["S"]["holds"] and rep["P"]["holds"]:
        applicable.append("N_fails")
    if N and not rep["S"]["holds"]:
        applicable.append("S_fails")
    if m1 and not rep["P"]["holds"]:
        applicable.append("P_fails")
    if rep["T1"]["holds"] is False:
        applicable.append("T1_fails")
    if rep["T2"]["holds"] is False:
        applicable.append("T2_fails")
    if rep["T1"]["holds"] and rep["T2"]["holds"]:
        applicable.append("T1_T2_hold")
    if "N_fails" in applicable:
        applicable.append("N_fails_assembled")
    applicable += ["curve", "not_run"] + (["EV"] if "EV" in sent else [])
    rep["sentences"] = sent
    rep["sentences_applicable"] = applicable
    txt = json.dumps(rep, indent=1, default=str)
    if a.out:
        od = Path(a.out)
        od.mkdir(parents=True, exist_ok=True)
        (od / "pc_summary.json").write_text(txt)
        lines = ["# Positive control: filled sentences", "", f"Generated {rep['generated']}; rows finished before {a.finished_before or 'any time'}.", ""]
        for k in applicable:
            lines += [f"**{k}**", "", sent[k], ""]
        lines += ["## Not applicable (filled for reference)", ""]
        for k, v in sent.items():
            if k not in applicable:
                lines += [f"**{k}**", "", v, ""]
        (od / "pc_sentences.md").write_text("\n".join(lines))
        worlds_csv(rows, th, od / "pc_worlds.csv")
        figure(rep, od / "fig_pc_b3_curve")
    print(txt)


if __name__ == "__main__":
    main()
