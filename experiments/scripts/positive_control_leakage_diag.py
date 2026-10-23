#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Investigation: why the real-calendar arm (CAL) failed its leakage gate (B1 inputs only).

Rebuilds the beta-0 labels and prices of the CAL and M0 worlds r = 0..199 in
memory (positive_control_worlds's own functions; no test script, no B3 field) and recomputes
B1's same-window excess and next-window placebo excess (make_figures._excess)
with three window returns:
  first_open   (last close - first open) / first open   (price_change_pct, as B1 reads it)
  close2close  last close of the window / last close before the window - 1
  no_ratio     first_open with the first day's open/close ratio replaced by 1
               (i.e. last close / first close - 1)
and with each label group's contribution to the window direction removed in turn.
Prints the mean, s.d. and 99.5% t-interval over the 200 worlds of each variant.

Usage: TRACE_ORIG=... python positive_control_leakage_diag.py [--workers 16] [--out FILE]
"""
from __future__ import annotations

import os

os.environ.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
REF = None
GROUPS = {"fomc_fixed_sign": ("hike", "cut"), "hold": ("hold",), "cpi": ("hot", "cool"),
          "earnings": ("beat", "miss"), "synthetic": ("a", "b", "c", "d", "e")}


def _init():
    global REF
    import positive_control_worlds
    REF = positive_control_worlds.Ref()
    REF.garch()


def _excess(s, r, rn):
    from make_figures import _excess as ex
    rows = []
    for i in range(s.shape[0]):
        for w in range(s.shape[1]):
            rows.append({"ticker": i, "win": w, "s": s[i, w], "r": r[i, w], "r_next": rn[i, w]})
    df = pd.DataFrame(rows)
    return ex(df, "r")[2], ex(df, "r_next")[2]


def one(r):
    import positive_control_worlds as pw
    ref = REF
    lab_full = pw.draw_labels(r, ref)
    eps, meta = pw.draw_noise(r, "C", ref)
    px = pw.prices_from(ref, eps, meta["micro_shift"])                  # beta 0: no effect
    ds = np.array(ref.date_str)
    first = np.array([np.where((ds >= s) & (ds <= e))[0][0] for s, e in ref.windows])
    last = np.array([np.where((ds >= s) & (ds <= e))[0][-1] for s, e in ref.windows])
    O, C = px[:, :, 0], px[:, :, 3]
    ret = {"first_open": ((C[last] - O[first]) / O[first] * 100).T,
           "close2close": ((C[last] - C[first - 1]) / C[first - 1] * 100).T,
           "no_ratio": ((C[last] - C[first]) / C[first] * 100).T}
    out = {"r": r}
    for arm, lab in (("CAL", pw.calendar_only(lab_full)), ("M0", lab_full)):
        contrib = lab["direction"] * lab["mag"][:, :, :pw.NT]              # (N, W, 12)
        variants = {"all": contrib.sum(axis=2)}
        for gname, keys in GROUPS.items():
            idx = [pw.TKEY.index(k) for k in keys]
            keep = [k for k in range(pw.NT) if k not in idx]
            variants[f"without_{gname}"] = contrib[:, :, keep].sum(axis=2)
        for vname, s in variants.items():
            for rname, rr in ret.items():
                rn = np.concatenate([rr[:, 1:], np.full((rr.shape[0], 1), np.nan)], axis=1)
                e, p = _excess(s, rr, rn)
                out[f"{arm}|{vname}|{rname}"] = (e, p)
    return out


def summarize(vals, conf=0.995):
    from scipy.stats import t as student_t
    x = np.array(vals, dtype=float)
    n = len(x)
    m, sd = float(x.mean()), float(x.std(ddof=1))
    h = float(student_t.ppf(0.5 + conf / 2, n - 1) * sd / math.sqrt(n))
    return {"mean": m, "sd": sd, "t": m / (sd / math.sqrt(n)), "interval": [m - h, m + h], "covers_0": m - h <= 0 <= m + h}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--r0", type=int, default=0, help="first replicate (1000+: fresh worlds outside every arm)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    with ProcessPoolExecutor(a.workers, initializer=_init) as ex:
        res = list(ex.map(one, range(a.r0, a.r0 + a.n)))
    keys = [k for k in res[0] if k != "r"]
    rep = {}
    for k in keys:
        rep[k] = {"excess": summarize([x[k][0] for x in res]), "placebo": summarize([x[k][1] for x in res])}
    txt = json.dumps({"n_worlds": a.n, "r0": a.r0, "results": rep}, indent=1)
    if a.out:
        Path(a.out).write_text(txt)
    for k in keys:
        e, p = rep[k]["excess"], rep[k]["placebo"]
        print(f"{k:45s} excess {e['mean']:+.5f} (t {e['t']:+.2f}) {'ok ' if e['covers_0'] else 'OUT'} | "
              f"placebo {p['mean']:+.5f} (t {p['t']:+.2f}) {'ok' if p['covers_0'] else 'OUT'}")


if __name__ == "__main__":
    main()
