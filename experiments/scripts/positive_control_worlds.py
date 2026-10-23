#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Positive control: synthetic worlds with planted, recurring event effects.

Design: the paper's appendix on the positive control, with positive_control_calendars.csv.
Section references below are to the design document fixed before any code. Nothing here changes P2/P3 or the B1-B3 scripts; the worlds are
written in the run layout those scripts read (TRACE_ROOT/<tag>/...).

Subcommands
  build           one world (labels -> presence -> effects -> noise -> prices ->
                  analyses.json, training_data.csv, stub impact matrices; with
                  --b2 also the oracle and b = 0 directories) and its manifest
  p2              the pipeline's own Phase 1 (saved labels) + Phase 2 with
                  configs/r2_datasign.yaml and run()'s seed-42 block,
                  through CausalSDEPipeline; the trained matrix replaces the stub
                  atomically (P3 is not run: B2 regenerates its paths itself)
  b1ci            the B1 calendar-window CI and placebo of one world
                  (make_figures._excess / _cluster_ci); run with TRACE_ROOT and
                  TRACE_OUT set to the world's
  classify-freeze copy b3_documented_effects's classifier output verbatim to classified_FROZEN.csv
                  and assert the design's class mapping
  selftest        the seven checks of step 1, plus the calendar counts of the
                  design's label table

World construction (section "World", "Labels", "Leakage control")
  * labels: numpy default_rng([r, 1]) (identifiability redraws: [r, 1, sub]),
    drawn by make_labels() with no price or return argument, before any return;
  * noise: default_rng([r, 2, variant]) with variant C = 0, I = 1, RR = 2, one
    draw per world; HET's eta: default_rng([r, 3]);
  * effects r(t -> t+1) += sum_k 1[F_tk > 0] beta sigma_i s_ik, added last;
  * O/H/L = real (O, H, L)/C ratios x synthetic close and real volume, both
    circularly shifted by 843 trading days (np.roll: real day i lands on day
    i + 843); arm RR uses its own offset for them;
  * the label hash (SHA-256 of every window's start, end and sorted factor
    records, price_change_pct excluded) depends on r only and is asserted.

Usage (TRACE_ORIG points to the generator checkout; set TRACE_GEN_COMMIT to pin its commit):
  python positive_control_worlds.py build --arm M1 --r 0 --beta 0.25 --root R --out O [--b2] [--tickers AAPL,JPM]
  TRACE_ROOT=R TRACE_OUT=O python positive_control_worlds.py p2 --tag TAG [--tickers ...]
  TRACE_ROOT=R TRACE_OUT=O python positive_control_worlds.py b1ci --tag TAG
  TRACE_ROOT=R TRACE_OUT=O python positive_control_worlds.py classify-freeze --tag TAG
  python positive_control_worlds.py selftest [--quick]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import common  # noqa: E402  (puts TRACE_ORIG on sys.path)
from common import ORIG, build_presence  # noqa: E402

GEN_COMMIT = os.environ.get("TRACE_GEN_COMMIT", "")   # optional pin of the generator commit
ARCH_VERSION = "8.0.0"
R2_TAG = "r2_datasign"
CAL_CSV = SCRIPTS / "positive_control_calendars.csv"
FEATS = ["open", "high", "low", "close", "volume"]
SHIFT = 843                              # trading days, section "Why 843 trading days"
N_EARN = 24
ORIGINS = ("2021-01-01", "2022-01-01", "2023-01-01")

# ── taxonomy (section "Labels") ──────────────────────────────────────────
# key, recurring name ({co} = company stem), planted sign (0 = random per stock),
# calendar source, category, persistence, eta scope (HET)
TYPES = [
    ("hike", "fed_rate_hike", -1, "fomc_hike", "macroeconomic", "medium", "common"),
    ("cut", "fed_rate_cut", +1, "fomc_cut", "macroeconomic", "medium", "common"),
    ("hold", "fomc_rate_hold", 0, "fomc_hold", "macroeconomic", "short", "common"),
    ("hot", "cpi_inflation_hot", -1, "cpi", "macroeconomic", "short", "common"),
    ("cool", "cpi_inflation_cool", +1, "cpi", "macroeconomic", "short", "common"),
    ("beat", "{co}_earnings_beat", +1, "earnings", "company_specific", "short", "stock"),
    ("miss", "{co}_earnings_miss", -1, "earnings", "company_specific", "short", "stock"),
    ("a", "recurring_type_a", 0, "syn_common", "sector_specific", "short", "common"),
    ("b", "recurring_type_b", 0, "syn_common", "sector_specific", "short", "common"),
    ("c", "recurring_type_c", 0, "syn_common", "sector_specific", "short", "common"),
    ("d", "recurring_type_d", 0, "syn_stock", "sector_specific", "short", "stock"),
    ("e", "recurring_type_e", 0, "syn_stock", "sector_specific", "short", "stock"),
]
NT = len(TYPES)                          # 12 non-baseline types; market_noise is the 13th
TKEY = [t[0] for t in TYPES]
NOISE_NAME, NOISE_CAT, NOISE_PERS = "market_noise", "market_sentiment", "transient"
EXPECTED_CLASS = {"hike": "E1_tightening", "cut": "E2_easing", "beat": "E4_earnings_beat",
                  "miss": "E5_earnings_miss"}
P_FLIP, P_BEAT, P_SYN, P_HOT = 0.2, 0.7, 0.25, 0.5
NOISE_VARIANT = {"C": 0, "I": 1, "RR": 2}

# ── arms (section "Arms"); task order = CPU priority, then beta as listed, then r ──
ARMS = {
    "M0": dict(noise="C", names="recurring", betas=[0.0], rs=range(0, 200)),
    "M1": dict(noise="C", names="recurring", betas=[0.25], rs=range(0, 200)),
    "M2": dict(noise="C", names="recurring", betas=[0.12, 0.06, 0.09, 0.18], rs=range(0, 200)),
    "M3": dict(noise="C", names="recurring", betas=[0.0], rs=range(200, 1000)),
    "EV": dict(noise="C", names="event", betas=[0.25, 0.12], rs=range(0, 200)),
    "IN": dict(noise="I", names="recurring", betas=[0.0, 0.25], rs=range(0, 200)),
    "HET": dict(noise="C", names="recurring", betas=[0.25, 0.12], rs=range(0, 200), het=True),
    "RR": dict(noise="RR", names="recurring", betas=[0.0, 0.12], rs=range(0, 200)),
    "EV0": dict(noise="C", names="event", betas=[0.0], rs=range(0, 200)),
    "TFT": dict(noise="C", names="recurring", betas=[0.25, 0.0, 0.12], rs=range(0, 3)),
    # real-calendar setting (CAL): calendar-only taxonomy (the five synthetic types removed); 1,000 null worlds
    "CAL": dict(noise="C", names="recurring", betas=[0.0, 0.25, 0.18, 0.12, 0.09, 0.06], rs=range(0, 200),
                rs_beta={0.0: range(0, 1000)}, calendar_only=True),
    # CALF: CAL regenerated with a per-world microstructure offset (O/H/L ratios, volume) from default_rng([r, 4])
    "CALF": dict(noise="C", names="recurring", betas=[0.0, 0.25, 0.18, 0.12, 0.09, 0.06], rs=range(0, 200),
                 rs_beta={0.0: range(0, 1000)}, calendar_only=True, micro_random=True),
}
CPU_PRIORITY = ["M0", "M1", "M2", "M3", "EV", "IN", "HET", "RR", "EV0"]


def tag_of(arm: str, r: int, beta: float, prefix: str = "pc") -> str:
    return f"{prefix}_{arm}_r{r:03d}_b{beta:g}"


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _company_stem(ticker: str) -> str:
    from b3_documented_effects import COMPANY
    return COMPANY[ticker].split("|")[0]


# ══════════════════════════════════════════════════════════════════════════
# Reference data: R2's calendar, windows and real prices (read-only)
# ══════════════════════════════════════════════════════════════════════════

class Ref:
    """R2's 28 stocks, 1,509 trading days and 155 windows; the calendars."""

    def __init__(self, tickers: list[str] | None = None, cache_dir: Path | None = None):
        rdir = ORIG / "results" / R2_TAG
        panel = pd.read_csv(rdir / "training_data.csv")
        all_tickers = sorted(panel["tic"].unique().tolist())
        assert len(all_tickers) == 28, len(all_tickers)
        self.all_tickers = all_tickers
        self.tickers = list(tickers) if tickers else all_tickers
        assert all(t in all_tickers for t in self.tickers)
        self.N = len(self.tickers)
        dates = sorted(panel["date"].unique().tolist())
        self.date_str = dates
        self.dates = pd.to_datetime(dates)
        self.T = len(dates)
        assert self.T == 1509 and dates[0] == "2018-01-02" and dates[-1] == "2023-12-29", (self.T, dates[0], dates[-1])

        # windows: identical for every stock (asserted), Sunday to Sunday
        wins = None
        for t in all_tickers:
            w = [(a["start_date"], a["end_date"]) for a in
                 sorted(json.load(open(rdir / t / "analyses.json")), key=lambda x: x["start_date"])]
            if wins is None:
                wins = w
            assert w == wins, f"R2 windows differ for {t}"
        assert len(wins) == 155
        for s, e in wins:
            assert pd.Timestamp(s).dayofweek == 6 and pd.Timestamp(e).dayofweek == 6
            assert s not in set(dates) and e not in set(dates)
        self.windows = wins
        self.W = len(wins)
        starts = pd.to_datetime([s for s, _ in wins])
        ends = pd.to_datetime([e for _, e in wins])
        widx = np.full(self.T, -1, dtype=int)
        for w in range(self.W):              # half-open [start, end) on calendar days
            m = (self.dates >= starts[w]) & (self.dates < ends[w])
            assert (widx[m] == -1).all()
            widx[m] = w
        self.widx = widx
        self.win_rows = [np.where(widx == w)[0] for w in range(self.W)]
        assert all(len(x) > 0 for x in self.win_rows)

        # real prices of the selected stocks (as sign_recovery: clipped at 1e-8)
        real = np.zeros((self.T, self.N, 5))
        for j, t in enumerate(self.tickers):
            df = panel[panel["tic"] == t].sort_values("date")
            assert df["date"].tolist() == dates
            real[:, j, :] = np.clip(df[FEATS].values.astype(np.float64), 1e-8, None)
        self.real = real
        self.r_real = np.diff(np.log(real[:, :, 3]), axis=0)          # (T-1, N)
        self.sigma = self.r_real.std(axis=0)                           # sign_recovery: r_real.std()
        self.ratios = real[:, :, :3] / real[:, :, [3]]
        self.volume = real[:, :, 4]

        # calendars
        cal = pd.read_csv(CAL_CSV)
        self.cal_sha = sha256_bytes(CAL_CSV.read_bytes())
        self.events: dict[str, list[tuple[str, int]]] = {}
        for ev in ("fomc_hike", "fomc_cut", "fomc_hold", "cpi_release"):
            self.events[ev] = [(d, self.window_of_date(d)) for d in cal[cal["event"] == ev]["date"]]
        assert len(self.events["cpi_release"]) == 72
        qends = [pd.Timestamp(y, m, 1) - pd.Timedelta(days=1) for y in range(2018, 2024) for m in (1, 4, 7, 10)]
        self.earnings: dict[str, list[tuple[str, int]]] = {}
        for t in all_tickers:
            off = 15 + zlib.crc32(t.encode()) % 31
            ed = [(q + pd.Timedelta(days=off)).strftime("%Y-%m-%d") for q in qends]
            self.earnings[t] = [(d, self.window_of_date(d)) for d in ed]
            assert len(self.earnings[t]) == N_EARN
        self.stem = {t: _company_stem(t) for t in all_tickers}
        self.cache_dir = Path(cache_dir or os.environ.get("TRACE_PC_CACHE", SCRIPTS.parent / "results" / "pc_cache"))
        self._garch = None

    def first_trading_row(self, d: str) -> int:
        i = int(self.dates.searchsorted(pd.Timestamp(d), side="left"))
        assert i < self.T, d
        return i

    def window_of_date(self, d: str) -> int:
        """Date-to-window rule: first trading day on or after d, then its window."""
        w = int(self.widx[self.first_trading_row(d)])
        assert w >= 0, f"event {d} falls outside the 155 windows"
        return w

    # ── GJR-GARCH(1,1)-t fits on real close returns (sign_recovery's spec) ──
    def garch(self) -> dict:
        if self._garch is not None:
            return self._garch
        import arch
        key = sha256_bytes(self.r_real.tobytes() + json.dumps(self.tickers).encode() + arch.__version__.encode())[:16]
        f = self.cache_dir / f"pc_garch_fit_{key}.json"
        if f.exists():
            g = json.loads(f.read_text())
        else:
            from arch import arch_model
            params, std_res = [], []
            for j in range(self.N):
                am = arch_model(self.r_real[:, j] * 100, vol="Garch", p=1, o=1, q=1, dist="t",
                                mean="Zero", rescale=False)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = am.fit(disp="off", show_warning=False)
                params.append([float(res.params[k]) for k in ("omega", "alpha[1]", "gamma[1]", "beta[1]", "nu")])
                std_res.append(np.asarray(res.std_resid, dtype=float))
            S = np.column_stack(std_res)
            corr = np.corrcoef(S, rowvar=False)
            g = {"tickers": self.tickers, "params": params, "corr": corr.tolist(),
                 "arch_version": arch.__version__, "key": key}
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            tmp = f.with_suffix(f".tmp{os.getpid()}")
            tmp.write_text(json.dumps(g))
            os.replace(tmp, f)
        g["params_arr"] = np.array(g["params"])
        g["corr_arr"] = np.array(g["corr"])
        g["chol"] = np.linalg.cholesky(g["corr_arr"])
        g["hash"] = sha256_bytes(json.dumps({"p": g["params"], "c": g["corr"]}).encode())[:16]
        self._garch = g
        return g


# ══════════════════════════════════════════════════════════════════════════
# Labels (no price or return argument: section "Leakage control", item 1)
# ══════════════════════════════════════════════════════════════════════════

def make_labels(r: int, ref: Ref, sub: int = 0) -> dict:
    """Draw one replicate's labels from default_rng([r, 1]) ([r, 1, sub] on redraw).

    Fixed draw order: CPI surprise signs (72, common), common synthetic types
    a-c (3 x W), then per stock in ticker order: hold sign, signs of a-e,
    per-stock types d-e (2 x W), beat/miss of the 24 earnings events, and for
    every (window, type) cell U(0.3, 0.8) direction size, sign flip (p 0.2),
    U(0.2, 0.8) magnitude, plus U(0.3, 0.7) for market_noise.
    """
    rng = np.random.default_rng([r, 1] if sub == 0 else [r, 1, sub])
    W, N = ref.W, ref.N
    cpi_hot = rng.random(72) < P_HOT
    common_syn = rng.random((3, W)) < P_SYN
    active = np.zeros((N, W, NT), dtype=bool)
    evdate = np.full((N, W, NT), "", dtype=object)
    signs = np.zeros((N, NT))
    direction = np.zeros((N, W, NT))
    mag = np.zeros((N, W, NT + 1))
    beat_all = np.zeros((N, N_EARN), dtype=bool)
    fomc = {"hike": ref.events["fomc_hike"], "cut": ref.events["fomc_cut"], "hold": ref.events["fomc_hold"]}
    for i, t in enumerate(ref.tickers):
        hold_sign = 1.0 if rng.random() < 0.5 else -1.0
        syn_sign = np.where(rng.random(5) < 0.5, 1.0, -1.0)
        stock_syn = rng.random((2, W)) < P_SYN
        beat = rng.random(N_EARN) < P_BEAT
        U = rng.uniform(0.3, 0.8, (W, NT))
        FL = rng.random((W, NT)) < P_FLIP
        MG = rng.uniform(0.2, 0.8, (W, NT))
        MN = rng.uniform(0.3, 0.7, W)
        beat_all[i] = beat
        for k, (key, _, s, *_r) in enumerate(TYPES):
            signs[i, k] = s if s != 0 else (hold_sign if key == "hold" else syn_sign["abcde".index(key)])
        for key in ("hike", "cut", "hold"):
            k = TKEY.index(key)
            for d, w in fomc[key]:
                assert not active[i, w, k], "two events of one type in one window"
                active[i, w, k] = True
                evdate[i, w, k] = d
        for (d, w), hot in zip(ref.events["cpi_release"], cpi_hot):
            k = TKEY.index("hot" if hot else "cool")
            assert not active[i, w, k]
            active[i, w, k] = True
            evdate[i, w, k] = d
        for (d, w), b in zip(ref.earnings[t], beat):
            k = TKEY.index("beat" if b else "miss")
            assert not active[i, w, TKEY.index("beat")] and not active[i, w, TKEY.index("miss")]
            active[i, w, k] = True
            evdate[i, w, k] = d
        for j, key in enumerate("abc"):
            k = TKEY.index(key)
            active[i, :, k] = common_syn[j]
            evdate[i, common_syn[j], k] = [ref.windows[w][0] for w in np.where(common_syn[j])[0]]
        for j, key in enumerate("de"):
            k = TKEY.index(key)
            active[i, :, k] = stock_syn[j]
            evdate[i, stock_syn[j], k] = [ref.windows[w][0] for w in np.where(stock_syn[j])[0]]
        q = np.where(FL, -1.0, 1.0)
        direction[i] = np.where(active[i], signs[i][None, :] * U * q, 0.0)
        raw = np.concatenate([np.where(active[i], MG, 0.0), MN[:, None]], axis=1)
        mag[i] = raw / raw.sum(axis=1, keepdims=True)                 # renormalized per window
    return {"r": r, "sub": sub, "cpi_hot": cpi_hot, "common_syn": common_syn, "beat": beat_all,
            "active": active, "evdate": evdate, "signs": signs, "direction": direction, "mag": mag}


def day_activity(lab: dict, ref: Ref) -> np.ndarray:
    """(N, T, 12) indicator of each type on each trading day (0 outside windows)."""
    A = np.zeros((ref.N, ref.T, NT), dtype=bool)
    inw = ref.widx >= 0
    A[:, inw, :] = lab["active"][:, ref.widx[inw], :]
    return A


def identifiable(lab: dict, ref: Ref) -> bool:
    """No stock has two identical non-baseline presence columns (recurring names)."""
    A = day_activity(lab, ref)
    for i in range(ref.N):
        cols = [A[i, :, k] for k in range(NT) if A[i, :, k].any()]
        for a in range(len(cols)):
            for b in range(a + 1, len(cols)):
                if np.array_equal(cols[a], cols[b]):
                    return False
    return True


def draw_labels(r: int, ref: Ref) -> dict:
    sub = 0
    while True:
        lab = make_labels(r, ref, sub)
        if identifiable(lab, ref):
            return lab
        sub += 1
        assert sub < 100


SYNTHETIC_KEYS = ("a", "b", "c", "d", "e")
MICRO_RANGE = (126, 1383)                                             # CALF, inclusive


def micro_offset(r: int) -> int:
    """Arm CALF (regenerated real-calendar worlds): per-world circular offset of the real O/H/L ratios and
    volume, uniform on [126, 1383], from its own stream default_rng([r, 4])."""
    return int(np.random.default_rng([r, 4]).integers(MICRO_RANGE[0], MICRO_RANGE[1] + 1))


def calendar_only(lab: dict) -> dict:
    """Real-calendar setting: the same draw with the five synthetic types switched off and the
    magnitudes of each window renormalized over the remaining factors (market_noise
    included). Activity, signs and directions of the 7 calendar types are unchanged."""
    out = dict(lab)
    syn = [TKEY.index(k) for k in SYNTHETIC_KEYS]
    out["active"] = lab["active"].copy()
    out["active"][:, :, syn] = False
    out["evdate"] = lab["evdate"].copy()
    out["evdate"][:, :, syn] = ""
    out["direction"] = lab["direction"].copy()
    out["direction"][:, :, syn] = 0.0
    mag = lab["mag"].copy()
    mag[:, :, syn] = 0.0
    out["mag"] = mag / mag.sum(axis=2, keepdims=True)
    out["calendar_only"] = True
    return out


def factor_name(key_idx: int, ticker: str, ref: Ref, regime: str, date: str = "") -> str:
    base = TYPES[key_idx][1].format(co=ref.stem[ticker])
    if regime == "event":
        return f"{base}_{date.replace('-', '_')}"
    return base


def window_records(lab: dict, ref: Ref, i: int, w: int, regime: str) -> list[dict]:
    t = ref.tickers[i]
    recs = []
    for k in range(NT):
        if lab["active"][i, w, k]:
            recs.append({"name": factor_name(k, t, ref, regime, lab["evdate"][i, w, k]),
                         "category": TYPES[k][4], "direction": float(lab["direction"][i, w, k]),
                         "magnitude": float(lab["mag"][i, w, k]), "persistence": TYPES[k][5],
                         "description": f"synthetic positive-control label ({TYPES[k][0]})"})
    recs.append({"name": NOISE_NAME, "category": NOISE_CAT, "direction": 0.0,
                 "magnitude": float(lab["mag"][i, w, NT]), "persistence": NOISE_PERS,
                 "description": "synthetic positive-control label (baseline)"})
    return recs


def label_hash(per_ticker_windows: dict) -> str:
    """Canonical label JSON: every window's start, end and sorted factor records
    (price_change_pct excluded), SHA-256 over all stocks."""
    canon = {t: [{"start": w["start_date"], "end": w["end_date"],
                  "factors": sorted(w["factors"], key=lambda f: f["name"])}
                 for w in sorted(ws, key=lambda x: x["start_date"])]
             for t, ws in sorted(per_ticker_windows.items())}
    return sha256_bytes(json.dumps(canon, sort_keys=True, separators=(",", ":")).encode())


# ══════════════════════════════════════════════════════════════════════════
# Noise (section "World")
# ══════════════════════════════════════════════════════════════════════════

def gjr_path(w: float, a: float, g: float, b: float, z: np.ndarray) -> np.ndarray:
    """sign_recovery.gjr_t_returns' recursion, verbatim, on given innovations z."""
    n = len(z)
    h = np.empty(n); e = np.empty(n)
    h[0] = w / max(1 - a - g / 2 - b, 1e-3)
    for t in range(n):
        if t:
            h[t] = w + (a + g * (e[t - 1] < 0)) * e[t - 1] ** 2 + b * h[t - 1]
        e[t] = np.sqrt(h[t]) * z[t]
    return e[500:] / 100


def copula_t_innovations(G: np.ndarray, chol: np.ndarray, nu: np.ndarray) -> np.ndarray:
    """Gaussian copula (correlation chol chol') joining unit-variance Student-t margins."""
    from scipy.stats import norm, t as student_t
    Z = G @ chol.T
    out = np.empty_like(Z)
    for j in range(Z.shape[1]):
        lo = norm.cdf(-np.abs(Z[:, j]))                      # lower-tail probability, precise
        q = -student_t.ppf(lo, nu[j])                         # >= 0
        out[:, j] = np.where(Z[:, j] < 0, -q, q) * np.sqrt((nu[j] - 2) / nu[j])
    return out


def draw_noise(r: int, variant: str, ref: Ref) -> tuple[np.ndarray, dict]:
    """(T-1, N) close log-return noise, one draw per world, and its metadata."""
    T1, N = ref.T - 1, ref.N
    rng = np.random.default_rng([r, 2, NOISE_VARIANT[variant]])
    meta = {"variant": variant, "micro_shift": SHIFT}
    if variant == "C":
        g = ref.garch()
        G = rng.standard_normal((T1 + 500, N))             # the one draw
        idx = [g["tickers"].index(t) for t in ref.tickers]
        P = g["params_arr"][idx]
        C = g["corr_arr"][np.ix_(idx, idx)]
        chol = np.linalg.cholesky(C)
        Zt = copula_t_innovations(G, chol, P[:, 4])
        eps = np.column_stack([gjr_path(*P[j, :4], Zt[:, j]) for j in range(N)])
        meta["garch_fit_hash"] = g["hash"]
    elif variant == "I":
        from sign_recovery import gjr_t_returns            # unchanged, independent across stocks
        cols = []
        for j in range(N):
            e, _ = gjr_t_returns(ref.r_real[:, j], T1, rng)
            cols.append(e)
        eps = np.column_stack(cols)
    elif variant == "RR":
        off = int(rng.integers(126, 1384))                 # uniform on [126, 1383]
        eps = np.roll(ref.r_real, off, axis=0)
        meta["micro_shift"] = off
        meta["rr_offset"] = off
    else:
        raise ValueError(variant)
    meta["noise_hash"] = sha256_bytes(np.ascontiguousarray(eps).tobytes())
    return eps, meta


def draw_eta(r: int, ref: Ref) -> np.ndarray:
    """HET: (N, W, 12) eta_e ~ N(0, 1); common across stocks for common-date types."""
    rng = np.random.default_rng([r, 3])
    common_k = [k for k in range(NT) if TYPES[k][6] == "common"]
    stock_k = [k for k in range(NT) if TYPES[k][6] == "stock"]
    ec = rng.standard_normal((ref.W, len(common_k)))
    es = rng.standard_normal((ref.N, ref.W, len(stock_k)))
    eta = np.zeros((ref.N, ref.W, NT))
    eta[:, :, common_k] = ec[None, :, :]
    eta[:, :, stock_k] = es
    return eta


def effects(lab: dict, ref: Ref, beta: float, eta: np.ndarray | None = None) -> np.ndarray:
    """(T-1, N): r(t -> t+1) += sum_k 1[F_tk > 0] beta sigma_i (s_ik [+ eta_e])."""
    A = day_activity(lab, ref)[:, :-1, :].astype(float)            # presence on day t
    eff = np.zeros((ref.T - 1, ref.N))
    if beta == 0:
        return eff
    for i in range(ref.N):
        if eta is None:
            coef = beta * ref.sigma[i] * lab["signs"][i]
            eff[:, i] = A[i] @ coef
        else:
            wi = ref.widx[:-1]
            c = np.zeros((ref.T - 1, NT))
            ok = wi >= 0
            c[ok] = beta * ref.sigma[i] * (lab["signs"][i][None, :] + eta[i, wi[ok], :])
            eff[:, i] = (A[i] * c).sum(axis=1)
    return eff


def prices_from(ref: Ref, r_tot: np.ndarray, micro_shift: int) -> np.ndarray:
    """(T, N, 5): close from returns; shifted real ratios and volume."""
    close = ref.real[0, :, 3][None, :] * np.exp(np.vstack([np.zeros((1, ref.N)), np.cumsum(r_tot, axis=0)]))
    ratios = np.roll(ref.ratios, micro_shift, axis=0)
    vol = np.roll(ref.volume, micro_shift, axis=0)
    out = np.empty((ref.T, ref.N, 5))
    out[:, :, :3] = ratios * close[:, :, None]
    out[:, :, 3] = close
    out[:, :, 4] = vol
    return out


# ══════════════════════════════════════════════════════════════════════════
# Writing a world
# ══════════════════════════════════════════════════════════════════════════

def _impact_matrix(names, P, b_close=None, estimator="pc_stub"):
    from sde_causal_generator.data_structures import ImpactMatrix
    K = len(names)
    base = np.zeros((K, 5))
    if b_close is not None:
        base[:, :4] = np.asarray(b_close, dtype=float)[:, None]    # O/H/L move with the close
    return ImpactMatrix(factor_names=list(names), base_impact=base, impact_std=np.zeros((K, 5)),
                        occurrence_prob=(P > 0).mean(axis=0).astype(float),
                        temporal_profile=np.zeros((K, 10)), interaction_matrix=np.zeros((K, K)),
                        feature_names=list(FEATS), estimator=estimator, units="daily")


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def build_world(arm: str, r: int, beta: float, root: Path, out: Path, ref: Ref,
                b2: bool = False, tag: str | None = None, keep_arrays: bool = False) -> dict:
    """Build one world in root/<tag> (+ _oracle, _b0eng with b2); manifest in out."""
    from sde_causal_generator.data_structures import WindowAnalysis, CausalFactor
    spec = ARMS[arm]
    regime = spec["names"]
    het = bool(spec.get("het"))
    tag = tag or tag_of(arm, r, beta)
    t0 = time.time()

    lab = draw_labels(r, ref)                                         # before any return exists
    recs = {t: [window_records(lab, ref, i, w, "recurring") for w in range(ref.W)]
            for i, t in enumerate(ref.tickers)}
    lhash_rec = label_hash({t: [{"start_date": s, "end_date": e, "factors": recs[t][w]}
                                for w, (s, e) in enumerate(ref.windows)] for t in ref.tickers})
    cal = bool(spec.get("calendar_only"))
    if cal:                                                           # real-calendar setting (arm CAL)
        assert regime == "recurring" and not het
        lab_full = lab
        lab = calendar_only(lab)
        recs = {t: [window_records(lab, ref, i, w, "recurring") for w in range(ref.W)]
                for i, t in enumerate(ref.tickers)}
    if regime == "event":
        recs = {t: [window_records(lab, ref, i, w, "event") for w in range(ref.W)]
                for i, t in enumerate(ref.tickers)}
    lhash = label_hash({t: [{"start_date": s, "end_date": e, "factors": recs[t][w]}
                            for w, (s, e) in enumerate(ref.windows)] for t in ref.tickers})

    eps, nmeta = draw_noise(r, spec["noise"], ref)
    if spec.get("micro_random"):                                      # CALF: separate stream
        nmeta["micro_shift"] = micro_offset(r)
    eta = draw_eta(r, ref) if het else None
    eff = effects(lab, ref, beta, eta)
    r_tot = eps + eff                                                 # effects added last
    px = prices_from(ref, r_tot, nmeta["micro_shift"])
    assert np.isfinite(px).all() and (px > 0).all()

    # training_data.csv (date, tic, OHLCV), stocks in ticker order
    world = root / tag
    if world.exists():
        shutil.rmtree(world)
    world.mkdir(parents=True)
    frames = [pd.DataFrame({"date": ref.date_str, "tic": t, "open": px[:, i, 0], "high": px[:, i, 1],
                            "low": px[:, i, 2], "close": px[:, i, 3], "volume": px[:, i, 4]})
              for i, t in enumerate(ref.tickers)]
    pd.concat(frames, ignore_index=True).to_csv(world / "training_data.csv", index=False)
    prices_hash = sha256_bytes((world / "training_data.csv").read_bytes())

    truth, occ_out, names_out, signs_out = {}, {}, {}, {}
    ds = np.array(ref.date_str)
    dfs = {}
    for i, t in enumerate(ref.tickers):
        an = []
        for w, (s, e) in enumerate(ref.windows):
            rows = np.where((ds >= s) & (ds <= e))[0]                  # the extractor's window mask
            o, c = px[rows[0], i, 0], px[rows[-1], i, 3]
            pc = float((c - o) / o * 100)                              # (last close - first open)/first open
            an.append(WindowAnalysis(start_date=s, end_date=e, window_type="monthly", price_change_pct=pc,
                                     factors=[CausalFactor.from_dict(x) for x in recs[t][w]],
                                     raw_llm_response=""))
        tdir = world / t
        _write_json(tdir / "analyses.json", [a.to_dict() for a in an])
        names = sorted({f.name for a in an for f in a.factors if f.magnitude > 0})
        df = frames[i]
        dfs[t] = df
        P = build_presence(an, df, names)
        _impact_matrix(names, P).save(str(tdir / "impact_matrix.json"))
        # truth per name: b = beta sigma_i s_ik (recurring, constant); HET per event
        tr = {}
        for name in names:
            if name == NOISE_NAME:
                tr[name] = 0.0
                continue
            k = _type_of_name(name, t, ref, regime)
            if het:
                ws = [w for w in range(ref.W) if lab["active"][i, w, k]
                      and (regime == "recurring" or factor_name(k, t, ref, regime, lab["evdate"][i, w, k]) == name)]
                tr[name] = [float(beta * ref.sigma[i] * (lab["signs"][i, k] + eta[i, w, k])) for w in ws]
            else:
                tr[name] = float(beta * ref.sigma[i] * lab["signs"][i, k])
        truth[t] = tr
        occ_out[t] = {n: float(o) for n, o in zip(names, (P > 0).mean(axis=0))}
        names_out[t] = names
        signs_out[t] = {TKEY[k]: float(lab["signs"][i, k]) for k in range(NT)}
        if regime == "event":                                          # event columns sum to the recurring column
            rec_names = sorted({x["name"] for w in range(ref.W) for x in window_records(lab, ref, i, w, "recurring")})
            an_rec = [WindowAnalysis(start_date=s, end_date=e, window_type="monthly", price_change_pct=0.0,
                                     factors=[CausalFactor.from_dict(x) for x in window_records(lab, ref, i, w, "recurring")])
                      for w, (s, e) in enumerate(ref.windows)]
            Prec = build_presence(an_rec, df, rec_names)
            for kk, rn in enumerate(rec_names):
                if rn == NOISE_NAME:
                    continue
                k = _type_of_name(rn, t, ref, "recurring")
                cols = [j for j, n in enumerate(names) if n != NOISE_NAME and _type_of_name(n, t, ref, "event") == k]
                assert np.array_equal(P[:, cols].sum(axis=1), Prec[:, kk]), (t, rn)
        if b2:
            assert not het, "oracle needs constant effects"
            b_or = np.array([truth[t][n] for n in names])
            for suffix, b, est in (("_oracle", b_or, "pc_oracle"), ("_b0eng", np.zeros(len(names)), "pc_b0")):
                d2 = root / (tag + suffix) / t
                d2.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(tdir / "analyses.json", d2 / "analyses.json")
                _impact_matrix(names, P, b, est).save(str(d2 / "impact_matrix.json"))
    if b2:
        for suffix in ("_oracle", "_b0eng"):
            shutil.copyfile(world / "training_data.csv", root / (tag + suffix) / "training_data.csv")

    # re-read the labels from disk and re-hash (excluding price_change_pct)
    disk = {t: json.load(open(world / t / "analyses.json")) for t in ref.tickers}
    assert label_hash(disk) == lhash, "label hash of the written analyses differs"

    man = {"tag": tag, "arm": arm, "r": r, "beta": beta, "noise": spec["noise"], "names": regime, "het": het,
           "tickers": ref.tickers, "label_sub": lab["sub"], "label_hash": lhash,
           "label_hash_recurring": lhash_rec, "noise_hash": nmeta["noise_hash"],
           "eta_hash": sha256_bytes(eta.tobytes()) if het else None, "prices_hash": prices_hash,
           "micro_shift": nmeta["micro_shift"], "rr_offset": nmeta.get("rr_offset"),
           "garch_fit_hash": nmeta.get("garch_fit_hash"), "calendar_sha256": ref.cal_sha,
           "generator_commit": GEN_COMMIT, "sigma": {t: float(s) for t, s in zip(ref.tickers, ref.sigma)},
           "signs": signs_out, "truth": truth, "occupancy": occ_out,
           "n_names": {t: len(v) for t, v in names_out.items()},
           "ro_eligibility": ro_eligibility(world, ref, names_out),
           "de_rows": de_rows(world, ref, names_out, dfs),
           "b2_dirs": [tag + "_oracle", tag + "_b0eng"] if b2 else [],
           "calendar_only": cal,
           "n_types_present": int(len({_type_of_name(n, t, ref, regime) for t in ref.tickers
                                       for n in names_out[t] if n != NOISE_NAME})),
           "build_s": round(time.time() - t0, 2)}
    out.mkdir(parents=True, exist_ok=True)
    (out / f"manifest_{tag}.json").write_text(json.dumps(man))
    if keep_arrays:
        man["_arrays"] = {"lab": lab, "eps": eps, "eff": eff, "px": px, "eta": eta,
                          "lab_full": lab_full if cal else lab}
    return man


def _type_of_name(name: str, ticker: str, ref: Ref, regime: str) -> int:
    for k in range(NT):
        base = TYPES[k][1].format(co=ref.stem[ticker])
        if name == base or (regime == "event" and name.startswith(base + "_")
                            and len(name) == len(base) + 11):
            return k
    raise KeyError(name)


def ro_eligibility(world: Path, ref: Ref, names: dict) -> dict:
    """The rolling-origin eligibility counted from the labels: per origin, stocks with
    >= 5 eligible factors (>= 5 training days, >= 5 test-year days, occ <= 0.8)."""
    from sde_causal_generator.data_structures import WindowAnalysis
    out = {}
    dates = ref.dates
    per = {}
    for t in ref.tickers:
        an = [WindowAnalysis.from_dict(d) for d in json.load(open(world / t / "analyses.json"))]
        df = pd.DataFrame({"date": ref.date_str})
        P_eval = build_presence(an, df, names[t])
        for o in ORIGINS:
            ot = pd.Timestamp(o)
            n_is = int((dates < ot).sum())
            n_end = int((dates < ot + pd.DateOffset(years=1)).sum())
            P_true = build_presence([a for a in an if pd.Timestamp(a.end_date) < ot], df, names[t])
            occ_is = (P_true[:n_is] > 0).mean(axis=0)
            is_days = (P_true[:n_is] > 0).sum(axis=0)
            test_days = (P_eval[n_is:n_end - 1] > 0).sum(axis=0)
            el = [(n != NOISE_NAME) and occ_is[k] <= 0.8 and is_days[k] >= 5 and test_days[k] >= 5
                  for k, n in enumerate(names[t])]
            per.setdefault(o, {})[t] = int(sum(el))
    for o in ORIGINS:
        out[o] = {"n_eligible": per[o], "stocks_with_ge5": int(sum(v >= 5 for v in per[o].values()))}
    return out


def de_rows(world: Path, ref: Ref, names: dict, dfs: dict) -> dict:
    """Expected documented-effects rows (class) with their first-activity window (cluster)."""
    from b3_documented_effects import classify
    rows = []
    for i, t in enumerate(ref.tickers):
        an = json.load(open(world / t / "analyses.json"))
        first = {}
        for w, a in enumerate(an):
            for f in a["factors"]:
                if f["magnitude"] > 0 and f["name"] not in first:
                    first[f["name"]] = w
        for n in names[t]:
            if n == NOISE_NAME:
                continue
            c = classify(n, t)
            if c is not None:
                rows.append((t, n, c, first[n]))
    by = {}
    for t, n, c, w in rows:
        by.setdefault(c, {"n": 0, "clusters": set()})
        by[c]["n"] += 1
        by[c]["clusters"].add(w)
    return {c: {"n": v["n"], "n_clusters": len(v["clusters"])} for c, v in sorted(by.items())}


# ══════════════════════════════════════════════════════════════════════════
# p2: the pipeline's Phase 2 on a built world
# ══════════════════════════════════════════════════════════════════════════

def cmd_p2(a):
    import random
    import torch
    import yaml
    from sde_causal_generator.pipeline import CausalSDEPipeline, PipelineConfig
    world = common.ROOT / a.tag
    cfg_path = ORIG / "configs" / "r2_datasign.yaml"
    cfg = yaml.safe_load(open(cfg_path))
    panel = pd.read_csv(world / "training_data.csv")
    tickers = a.tickers.split(",") if a.tickers else sorted(panel["tic"].unique())
    log = []
    for t in tickers:
        t0 = time.time()
        stage = world / t / "_p2"
        if stage.exists():
            shutil.rmtree(stage)
        ticker_cfg = dict(cfg)
        # I/O fields only (as scripts/run_pipeline.py sets output_dir / cache_dir per ticker)
        ticker_cfg.update(output_dir=str(stage), cache_dir=str(stage / "cache"),
                          precomputed_analyses_dir=str(world), llm_call_log="")
        pipe = CausalSDEPipeline(config=PipelineConfig._from_flat_or_nested(ticker_cfg))
        c = pipe.cfg
        assert (c.seed, c.fin_architecture, c.fin_impact_estimator, c.fin_direction_prior, c.fin_n_lags,
                c.fin_epochs, c.fin_batch_size, c.fin_lr, c.fin_patience) == \
            (42, "tft", "ablation", "none", 10, 300, 64, 1e-3, 30), "unexpected pipeline config"
        assert not c.use_factor_scoring and c.max_factors == 0
        df = panel[panel["tic"] == t].sort_values("date").reset_index(drop=True)    # split_by_ticker
        # CausalSDEPipeline.run(): seeding block, verbatim
        np.random.seed(c.seed)
        random.seed(c.seed)
        torch.manual_seed(c.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(c.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        assert not pipe._load_user_factors()
        presence, names, dates, analyses = pipe._phase1_extract(df, t)
        pipe._compute_llm_directions(analyses, names)                  # computed by run(), then dropped
        im, _model, hist = pipe._phase2_train(df, presence, names, llm_directions=None)
        stub = json.load(open(world / t / "impact_matrix.json"))["factor_names"]
        assert list(names) == stub, f"{t}: P2 factor set differs from the stub"
        assert (stage / "analyses.json").read_bytes() == (world / t / "analyses.json").read_bytes()
        os.replace(stage / "impact_matrix.json", world / t / "impact_matrix.json")    # atomic
        shutil.rmtree(stage)
        log.append({"ticker": t, "best_epoch": hist.get("best_epoch"), "elapsed_s": round(time.time() - t0, 1)})
        print(f"P2 {a.tag}/{t}: best_epoch={hist.get('best_epoch')} ({log[-1]['elapsed_s']}s)", flush=True)
    common.OUT.mkdir(parents=True, exist_ok=True)
    (common.OUT / f"p2_log_{a.tag}_{int(time.time() * 1000)}.json").write_text(json.dumps(log))


# ══════════════════════════════════════════════════════════════════════════
# b1ci and classify-freeze (run with the world's TRACE_ROOT / TRACE_OUT)
# ══════════════════════════════════════════════════════════════════════════

def cmd_b1ci(a):
    import make_figures as mf
    df = mf._arm_windows(a.tag)
    ag, ind, ex, n = mf._excess(df, "r")
    _, _, pex, pn = mf._excess(df, "r_next")
    res = {"tag": a.tag, "agreement": ag, "independence": ind, "excess": ex, "n_scored": n,
           "excess_ci95": mf._cluster_ci(df, "r"), "placebo_excess": pex,
           "placebo_ci95": mf._cluster_ci(df, "r_next"), "n_placebo": pn}
    common.save_json(res, common.OUT / f"b1_ci_{a.tag}.json")
    print(json.dumps(res))


def expected_class(name: str, ticker: str, ref: Ref, regime: str):
    from b3_documented_effects import FINANCIALS
    k = _type_of_name(name, ticker, ref, regime)
    key = TKEY[k]
    if key in ("hike", "cut") and ticker in FINANCIALS:
        return None
    return EXPECTED_CLASS.get(key)


def cmd_classify_freeze(a):
    od = common.OUT / f"b3_documented_effects__{a.tag}"
    src, dst = od / "classified_FOR_AUDIT.csv", od / "classified_FROZEN.csv"
    shutil.copyfile(src, dst)                                         # verbatim
    fz = pd.read_csv(dst)
    regime = "event" if "_EV" in a.tag else "recurring"
    ref = Ref()
    got = {(r.ticker, r.factor): r["class"] for _, r in fz.iterrows()}
    assert (fz["keep"] == 1).all() and "AMBIGUOUS" not in set(fz["class"])
    exp = {}
    for t in sorted({p.name for p in (common.ROOT / a.tag).iterdir() if (p / "analyses.json").exists()}):
        names = json.load(open(common.ROOT / a.tag / t / "impact_matrix.json"))["factor_names"]
        for n in names:
            if n == NOISE_NAME:
                continue
            c = expected_class(n, t, ref, regime)
            if c is not None:
                exp[(t, n)] = c
    assert got == exp, f"classifier mapping differs: {len(got)} rows vs {len(exp)} expected"
    print(f"frozen list = classifier output verbatim; mapping asserted ({len(got)} rows)")


# ══════════════════════════════════════════════════════════════════════════
# selftest (section "Implementation plan", step 1)
# ══════════════════════════════════════════════════════════════════════════

def _shift_alignment(ref: Ref, s: int) -> dict:
    """The design's alignment search statistic at shift s (real dates and volume only)."""
    T, dates = ref.T, ref.dates

    def occ(ws):
        m = np.zeros(T, bool)
        for w in set(ws):
            m |= ref.widx == w
        return m
    fomc_w = [w for ev in ("fomc_hike", "fomc_cut", "fomc_hold") for _, w in ref.events[ev]]
    cpi_w = [w for _, w in ref.events["cpi_release"]]
    fomc_all, cpi_all = occ(fomc_w), occ(cpi_w)
    fomc_days = [ref.first_trading_row(d) for ev in ("fomc_hike", "fomc_cut", "fomc_hold") for d, _ in ref.events[ev]]
    cpi_days = [ref.first_trading_row(d) for d, _ in ref.events["cpi_release"]]
    f = float(np.mean([fomc_all[(i + s) % T] for i in fomc_days]))
    c = float(np.mean([cpi_all[(i + s) % T] for i in cpi_days]))
    lv = np.log(ref.volume)
    lv = lv - pd.DataFrame(lv).rolling(63, min_periods=10, center=True).median().values
    sh = np.roll(lv, s, axis=0)                                       # sh[t] = lv[t - s]
    dv_f = float(np.nanmean(sh[fomc_all]) - np.nanmean(sh[~fomc_all]))
    dv_c = float(np.nanmean(sh[cpi_all]) - np.nanmean(sh[~cpi_all]))
    dv_e = []
    for j, t in enumerate(ref.tickers):
        m = occ([w for _, w in ref.earnings[t]])
        dv_e.append(np.nanmean(sh[m, j]) - np.nanmean(sh[~m, j]))
    return {"fomc_align": f, "cpi_align": c, "base_fomc": float(fomc_all.mean()), "base_cpi": float(cpi_all.mean()),
            "dlogvol_fomc": dv_f, "dlogvol_cpi": dv_c, "dlogvol_earnings": float(np.mean(dv_e))}


def cmd_selftest(a):
    import arch
    from b3_documented_effects import classify
    from sign_recovery import gjr_t_returns
    import sde_causal_generator
    t_start = time.time()
    rep = {}
    ref = Ref()

    # 7. generator commit, import path, arch
    gen_file = Path(sde_causal_generator.__file__).resolve()
    assert str(gen_file).startswith(str(ORIG.resolve()) + os.sep), (gen_file, ORIG)
    head = subprocess.run(["git", "-C", str(ORIG), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if GEN_COMMIT:
        assert head == GEN_COMMIT, head
    assert arch.__version__ == ARCH_VERSION, arch.__version__
    import scipy
    import torch
    rep["7_versions"] = {"generator_head": head, "generator_file": str(gen_file), "arch": arch.__version__,
                         "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__,
                         "torch": torch.__version__, "python": sys.version.split()[0]}
    print("7 ok", rep["7_versions"], flush=True)

    # design counts (label table)
    cnt = {}
    for ev, key in (("fomc_hike", "hike"), ("fomc_cut", "cut"), ("fomc_hold", "hold")):
        ws = {w for _, w in ref.events[ev]}
        days = int(np.isin(ref.widx, list(ws)).sum())
        cnt[key] = (len(ws), days, round(days / ref.T, 3))
    assert cnt == {"hike": (15, 149, 0.099), "cut": (5, 50, 0.033), "hold": (29, 286, 0.19)}, cnt
    assert len({w for _, w in ref.events["cpi_release"]}) == 72
    n_sun = sum(pd.Timestamp(d).dayofweek == 6 and d in {s for s, _ in ref.windows}
                for t in ref.tickers for d, _ in ref.earnings[t])
    assert n_sun == 51, n_sun
    assert round(float((ref.widx >= 0).mean()), 3) == 0.991
    assert ref.window_of_date("2020-03-15") == [s for s, _ in ref.windows].index("2020-03-15")
    rep["design_counts"] = {"fomc": cnt, "cpi_windows": 72, "earnings_boundary_sundays": n_sun}
    print("counts ok", cnt, flush=True)

    # 1. classifier mapping, recurring and event-dated, for all 28 stems
    for t in ref.tickers:
        for regime in ("recurring", "event"):
            for k in range(NT):
                name = factor_name(k, t, ref, regime, "2022-06-15")
                assert classify(name, t) == expected_class(name, t, ref, regime), (t, name, classify(name, t))
    rep["1_classifier"] = "ok"
    print("1 ok", flush=True)

    # 3. the recursion reproduces gjr_t_returns given the same innovations
    g = ref.garch()
    for j in (range(ref.N) if not a.quick else (0, 7, 15)):
        e1, par = gjr_t_returns(ref.r_real[:, j], ref.T - 1, np.random.default_rng(123 + j))
        nu = par["nu"]
        z = np.random.default_rng(123 + j).standard_t(nu, size=ref.T - 1 + 500) * np.sqrt((nu - 2) / nu)
        e2 = gjr_path(par["omega"], par["alpha"], par["gamma"], par["beta"], z)
        assert np.array_equal(e1, e2)
        assert np.allclose([par[k] for k in ("omega", "alpha", "gamma", "beta", "nu")], g["params_arr"][j], rtol=0, atol=0)
    rep["3_recursion"] = "ok (bit-identical; cached fit = fresh fit)"
    print("3 ok", flush=True)

    # 2. label-hash invariance across beta and noise (and EV collapses to recurring)
    tmp = Path(tempfile.mkdtemp(prefix="pc_selftest_"))
    try:
        hashes = {}
        builds = [("M0", 0, 0.0), ("M1", 0, 0.25), ("M2", 0, 0.12), ("IN", 0, 0.25), ("RR", 0, 0.12),
                  ("HET", 0, 0.25), ("EV", 0, 0.25), ("EV0", 0, 0.0)]
        if a.quick:
            builds = [("M0", 0, 0.0), ("M1", 0, 0.25), ("RR", 0, 0.12), ("EV", 0, 0.25)]
        mans = {}
        for arm, r, b in builds:
            m = build_world(arm, r, b, tmp / "root", tmp / "out", ref, b2=(arm == "M1"), keep_arrays=True)
            mans[(arm, b)] = m
            hashes[(arm, b)] = (m["label_hash"], m["label_hash_recurring"], m["noise_hash"])
            print(f"  built {m['tag']} in {m['build_s']}s", flush=True)
        rec = {v[1] for v in hashes.values()}
        assert len(rec) == 1, "recurring label hash differs across arms"
        assert len({v[0] for (arm, _), v in hashes.items() if arm not in ("EV", "EV0")}) == 1
        cnoise = {v[2] for (arm, _), v in hashes.items() if arm in ("M0", "M1", "M2", "HET", "EV", "EV0")}
        assert len(cnoise) == 1, "noise C differs across beta"
        m1 = mans[("M1", 0.25)]
        m0 = mans[("M0", 0.0)]
        # pairing: the beta 0.25 minus beta 0 close returns are the planted effects
        lr = lambda m: np.diff(np.log(m["_arrays"]["px"][:, :, 3]), axis=0)      # noqa: E731
        assert np.allclose(lr(m1) - lr(m0), m1["_arrays"]["eff"], rtol=0, atol=1e-12)
        assert np.abs(m1["_arrays"]["eff"]).max() > 0 and not m0["_arrays"]["eff"].any()
        rep["2_label_hash"] = {"recurring": next(iter(rec)), "noise_C": next(iter(cnoise)),
                               "worlds": [f"{k[0]}@{k[1]}" for k in hashes]}
        print("2 ok", flush=True)

        # 4. regressing the noise-free drift on presence returns b* exactly (t -> t+1)
        from b3_documented_effects import _ols_coefs
        from b3_rolling_origin import ols_signs_input
        from sde_causal_generator.data_structures import WindowAnalysis
        lab = m1["_arrays"]["lab"]
        eff = effects(lab, ref, 0.25)
        close = ref.real[0, :, 3][None, :] * np.exp(np.vstack([np.zeros((1, ref.N)), np.cumsum(eff, axis=0)]))
        world = tmp / "root" / m1["tag"]
        worst = 0.0
        for i, t in enumerate(ref.tickers):
            an = [WindowAnalysis.from_dict(x) for x in json.load(open(world / t / "analyses.json"))]
            names = json.load(open(world / t / "impact_matrix.json"))["factor_names"]
            P = build_presence(an, pd.DataFrame({"date": ref.date_str}), names)
            for coefs in (_ols_coefs(P, close[:, i]), ols_signs_input(P, close[:, i])):
                b_true = np.array([m1["truth"][t][n] for n in names])
                nz = [j for j, n in enumerate(names) if n != NOISE_NAME]
                worst = max(worst, float(np.abs(coefs[nz] - b_true[nz]).max() / ref.sigma[i]))
        assert worst < 1e-9, worst
        rep["4_drift_ols"] = {"max_abs_error_sd": worst}
        print("4 ok", worst, flush=True)

        # oracle / b0 directories, stub = names with magnitude > 0
        for suffix in ("_oracle", "_b0eng"):
            assert (tmp / "root" / (m1["tag"] + suffix) / "AAPL" / "impact_matrix.json").exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # 5. shift alignment and volume
    al = _shift_alignment(ref, SHIFT)
    assert abs(al["fomc_align"] - al["base_fomc"]) <= 0.10 and abs(al["cpi_align"] - al["base_cpi"]) <= 0.10, al
    assert max(abs(al[k]) for k in ("dlogvol_fomc", "dlogvol_cpi", "dlogvol_earnings")) <= 0.03, al
    rep["5_shift"] = {k: round(v, 4) for k, v in al.items()}
    rep["5_shift_unshifted"] = {k: round(v, 4) for k, v in _shift_alignment(ref, 0).items()}
    print("5 ok", rep["5_shift"], flush=True)

    # 6. build_presence bit-identical (R1, R2, synthetic with overlaps); b1_event_grounding reproduces its output
    from common import _build_presence_loop
    from sde_causal_generator.data_structures import WindowAnalysis, CausalFactor, ImpactMatrix
    n_ok = 0
    for run in ("r1_datasign", R2_TAG):
        panel = pd.read_csv(ORIG / "results" / run / "training_data.csv")
        for t in sorted(panel["tic"].unique()):
            df = panel[panel["tic"] == t].sort_values("date").reset_index(drop=True)
            im = ImpactMatrix.load(str(ORIG / "results" / run / t / "impact_matrix.json"))
            an = [WindowAnalysis.from_dict(x) for x in json.load(open(ORIG / "results" / run / t / "analyses.json"))]
            assert build_presence(an, df, im.factor_names).tobytes() == \
                _build_presence_loop(an, df, im.factor_names).tobytes(), (run, t)
            n_ok += 1
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"date": ref.date_str})
    syn = []
    for _ in range(60):                                     # overlapping windows, NaN and ties
        s = int(rng.integers(0, ref.T - 30)); e = s + int(rng.integers(0, 40))
        fs = [CausalFactor(name=f"f{int(rng.integers(0, 6))}", category="technical",
                           direction=0.1, magnitude=float(rng.choice([0.0, 0.3, 0.3, np.nan, rng.random()])),
                           persistence="short") for _ in range(4)]
        syn.append(WindowAnalysis(start_date=ref.date_str[s], end_date=ref.date_str[min(e, ref.T - 1)],
                                  window_type="monthly", price_change_pct=0.0, factors=fs))
    nm = [f"f{i}" for i in range(5)]                       # f5 not in the list on purpose
    assert build_presence(syn, df, nm).tobytes() == _build_presence_loop(syn, df, nm).tobytes()
    tmpo = Path(tempfile.mkdtemp(prefix="pc_selftest_expp_"))
    try:
        env = {k: v for k, v in os.environ.items() if k not in ("TRACE_ROOT", "TRACE_B1_ARMS", "TRACE_RUN")}
        env["TRACE_OUT"] = str(tmpo)                         # default ROOT and arms; output aside
        subprocess.run([sys.executable, str(SCRIPTS / "b1_event_grounding.py")], env=env, check=True,
                       capture_output=True, text=True)
        ref_file = SCRIPTS.parent / "results" / "b1_event_grounding" / "summary.json"
        assert (tmpo / "b1_event_grounding" / "summary.json").read_bytes() == ref_file.read_bytes(), "b1_event_grounding differs"
    finally:
        shutil.rmtree(tmpo, ignore_errors=True)
    rep["6_build_presence"] = {"stocks_bit_identical": n_ok, "synthetic_overlaps": "ok",
                               "b1_event_grounding_committed_output": "reproduced"}
    print("6 ok", flush=True)
    rep["elapsed_s"] = round(time.time() - t_start, 1)
    print(json.dumps(rep, indent=1, default=str))
    if a.report:
        Path(a.report).write_text(json.dumps(rep, indent=1, default=str))


def cmd_selftest_cal(a):
    """Real-calendar setting: the M arms are unaffected by the CAL change (rebuilt M worlds reproduce
    the hashes of the M JSONL), and CAL worlds have exactly the 7 calendar types, the M
    draw's calendar activity, signs and directions, M's noise and, at beta 0, M0's prices."""
    rows = {}
    for line in Path(a.m_jsonl).read_text().splitlines():
        r = json.loads(line)
        if r.get("status") == "ok":
            rows[r["tag"]] = r
    ref = Ref()
    rep = {"m_arms_unchanged": [], "cal": []}
    tmp = Path(tempfile.mkdtemp(prefix="pc_selftest_cal_"))
    try:
        for arm, r, b in (("M0", 0, 0.0), ("M1", 0, 0.25), ("M2", 5, 0.12), ("M3", 437, 0.0), ("EV", 3, 0.25),
                          ("HET", 2, 0.25), ("RR", 1, 0.12), ("IN", 4, 0.25), ("EV0", 7, 0.0)):
            m = build_world(arm, r, b, tmp / "root", tmp / "out", ref)
            row = rows[tag_of(arm, r, b)]
            for k in ("prices_hash", "label_hash", "label_hash_recurring", "noise_hash"):
                assert m[k] == row[k], (arm, r, b, k)
            rep["m_arms_unchanged"].append(m["tag"])
            shutil.rmtree(tmp / "root" / m["tag"])
        cal_names = {TYPES[k][1] for k in range(NT) if TKEY[k] not in SYNTHETIC_KEYS}
        for r, b in ((0, 0.0), (0, 0.25), (437, 0.0), (11, 0.06)):
            mc = build_world("CAL", r, b, tmp / "root", tmp / "out", ref, keep_arrays=True)
            mm = build_world("M0" if b == 0 else "M1", r, b, tmp / "root", tmp / "out", ref, keep_arrays=True,
                             tag=f"cmp_M_r{r}_b{b:g}")
            lc, lm = mc["_arrays"]["lab"], mm["_arrays"]["lab"]
            ci = [k for k in range(NT) if TKEY[k] not in SYNTHETIC_KEYS]
            si = [k for k in range(NT) if TKEY[k] in SYNTHETIC_KEYS]
            assert np.array_equal(lc["active"][:, :, ci], lm["active"][:, :, ci])
            assert not lc["active"][:, :, si].any()
            assert np.array_equal(lc["signs"], lm["signs"])
            assert np.array_equal(lc["direction"][:, :, ci], lm["direction"][:, :, ci])
            assert np.allclose(lc["mag"].sum(axis=2), 1.0)
            assert mc["label_hash_recurring"] == mm["label_hash_recurring"] == \
                rows[tag_of("M0" if r < 200 else "M3", r, 0.0)]["label_hash_recurring"]
            assert mc["noise_hash"] == mm["noise_hash"]
            types_present = set()
            for t in ref.tickers:
                names = json.load(open(tmp / "root" / mc["tag"] / t / "impact_matrix.json"))["factor_names"]
                nb = [n for n in names if n != NOISE_NAME]
                assert not any(n.startswith("recurring_type") for n in nb)
                bases = {TYPES[_type_of_name(n, t, ref, "recurring")][1] for n in nb}
                assert bases <= cal_names and len(nb) in (6, 7), (t, nb)
                types_present |= bases
            assert types_present == cal_names and len(cal_names) == 7 and mc["n_types_present"] == 7
            if b == 0:
                m0 = rows[tag_of("M0" if r < 200 else "M3", r, 0.0)]
                assert mc["prices_hash"] == m0["prices_hash"], "CAL beta-0 prices differ from M0/M3"
            else:
                lr = lambda m: np.diff(np.log(m["_arrays"]["px"][:, :, 3]), axis=0)     # noqa: E731
                assert np.allclose(lr(mc) - mc["_arrays"]["eps"], mc["_arrays"]["eff"], rtol=0, atol=1e-12)
                assert not np.array_equal(mc["_arrays"]["eff"], mm["_arrays"]["eff"])
            rep["cal"].append({"tag": mc["tag"], "n_types": mc["n_types_present"],
                               "label_hash_recurring_equals_M": True, "noise_equals_M": True})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(json.dumps(rep, indent=1))
    if a.report:
        Path(a.report).write_text(json.dumps(rep, indent=1))


def cmd_selftest_calf(a):
    """CALF worlds equal the unfixed CAL worlds in close noise, labels, presence
    and closes, and differ only in O/H/L/volume; offsets vary across r within [126, 1383];
    distribution-level alignment and volume checks; M and CAL arms unchanged."""
    def load(f):
        out = {}
        for line in Path(f).read_text().splitlines():
            r = json.loads(line)
            if r.get("status") == "ok":
                out[r["tag"]] = r
        return out
    mrows, crows = load(a.m_jsonl), load(a.cal_jsonl)
    ref = Ref()
    rep = {"unchanged": [], "calf_vs_cal": [], "offsets": {}}
    tmp = Path(tempfile.mkdtemp(prefix="pc_selftest_calf_"))
    try:
        for arm, r, b, rows in (("M0", 0, 0.0, mrows), ("M1", 0, 0.25, mrows), ("CAL", 0, 0.0, crows),
                                ("CAL", 5, 0.12, crows), ("CAL", 437, 0.0, crows)):
            m = build_world(arm, r, b, tmp / "root", tmp / "out", ref)
            row = rows[tag_of(arm, r, b)]
            for k in ("prices_hash", "label_hash", "label_hash_recurring", "noise_hash"):
                assert m[k] == row[k], (arm, r, b, k)
            rep["unchanged"].append(m["tag"])
            shutil.rmtree(tmp / "root" / m["tag"])
        for r, b in ((0, 0.0), (5, 0.12), (437, 0.0), (11, 0.06), (12, 0.06), (199, 0.25)):
            mf = build_world("CALF", r, b, tmp / "root", tmp / "out", ref, keep_arrays=True)
            mc = build_world("CAL", r, b, tmp / "root", tmp / "out", ref, keep_arrays=True)
            for k in ("noise_hash", "label_hash", "label_hash_recurring"):
                assert mf[k] == mc[k], (r, b, k)
            pf, pc = mf["_arrays"]["px"], mc["_arrays"]["px"]
            assert np.array_equal(pf[:, :, 3], pc[:, :, 3]), "closes differ"
            if mf["micro_shift"] != SHIFT:
                assert mf["prices_hash"] != mc["prices_hash"]
                assert not np.array_equal(pf[:, :, [0, 1, 2, 4]], pc[:, :, [0, 1, 2, 4]])
            else:                                                     # the draw can land on 843 (r = 11, 686)
                assert mf["prices_hash"] == mc["prices_hash"]
            assert mc["micro_shift"] == SHIFT and MICRO_RANGE[0] <= mf["micro_shift"] <= MICRO_RANGE[1]
            assert mf["micro_shift"] == micro_offset(r)
            cf = pd.read_csv(tmp / "root" / mf["tag"] / "training_data.csv")
            cc = pd.read_csv(tmp / "root" / mc["tag"] / "training_data.csv")
            assert cf["close"].equals(cc["close"]) and cf["date"].equals(cc["date"])
            for t in ref.tickers:
                nf = json.load(open(tmp / "root" / mf["tag"] / t / "impact_matrix.json"))["factor_names"]
                nc = json.load(open(tmp / "root" / mc["tag"] / t / "impact_matrix.json"))["factor_names"]
                assert nf == nc
                from sde_causal_generator.data_structures import WindowAnalysis
                af = [WindowAnalysis.from_dict(x) for x in json.load(open(tmp / "root" / mf["tag"] / t / "analyses.json"))]
                ac = [WindowAnalysis.from_dict(x) for x in json.load(open(tmp / "root" / mc["tag"] / t / "analyses.json"))]
                df = pd.DataFrame({"date": ref.date_str})
                assert build_presence(af, df, nf).tobytes() == build_presence(ac, df, nc).tobytes()
            rep["calf_vs_cal"].append({"r": r, "beta": b, "offset": mf["micro_shift"], "identical_close_noise_labels_presence": True})
            for tg in (mf["tag"], mc["tag"]):
                shutil.rmtree(tmp / "root" / tg)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    offs = [micro_offset(r) for r in range(1000)]
    assert len(set(offs)) > 500 and min(offs) >= MICRO_RANGE[0] and max(offs) <= MICRO_RANGE[1]
    rep["offsets"] = {"n_distinct_r0_999": len(set(offs)), "min": min(offs), "max": max(offs),
                      "first_10": offs[:10]}
    # distribution-level alignment and volume checks
    cache = {}

    def al(sft):
        if sft not in cache:
            cache[sft] = _shift_alignment(ref, sft)
        return cache[sft]
    out = {}
    for name, dist in (("uniform_126_1383", list(range(MICRO_RANGE[0], MICRO_RANGE[1] + 1))), ("realized_r0_999", offs)):
        m = {k: float(np.mean([al(x)[k] for x in dist])) for k in
             ("fomc_align", "cpi_align", "dlogvol_fomc", "dlogvol_cpi", "dlogvol_earnings")}
        base = al(SHIFT)
        m["base_fomc"], m["base_cpi"] = base["base_fomc"], base["base_cpi"]
        assert abs(m["fomc_align"] - m["base_fomc"]) <= 0.10 and abs(m["cpi_align"] - m["base_cpi"]) <= 0.10, (name, m)
        assert max(abs(m[k]) for k in ("dlogvol_fomc", "dlogvol_cpi", "dlogvol_earnings")) <= 0.03, (name, m)
        out[name] = {k: round(v, 4) for k, v in m.items()}
    rep["distribution_level_checks"] = out
    print(json.dumps(rep, indent=1))
    if a.report:
        Path(a.report).write_text(json.dumps(rep, indent=1))


def cmd_build(a):
    ref = Ref(tickers=a.tickers.split(",") if a.tickers else None)
    m = build_world(a.arm, a.r, a.beta, Path(a.root), Path(a.out), ref, b2=a.b2, tag=a.tag or None)
    print(json.dumps({k: m[k] for k in ("tag", "label_hash", "noise_hash", "prices_hash", "label_sub", "build_s")}))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build")
    b.add_argument("--arm", required=True, choices=list(ARMS))
    b.add_argument("--r", type=int, required=True)
    b.add_argument("--beta", type=float, required=True)
    b.add_argument("--root", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--b2", action="store_true")
    b.add_argument("--tickers", default="")
    b.add_argument("--tag", default="")
    p = sp.add_parser("p2")
    p.add_argument("--tag", required=True)
    p.add_argument("--tickers", default="")
    c = sp.add_parser("b1ci")
    c.add_argument("--tag", required=True)
    f = sp.add_parser("classify-freeze")
    f.add_argument("--tag", required=True)
    s = sp.add_parser("selftest")
    s.add_argument("--quick", action="store_true")
    s.add_argument("--report", default="")
    sf = sp.add_parser("selftest-calf")
    sf.add_argument("--m-jsonl", required=True)
    sf.add_argument("--cal-jsonl", required=True)
    sf.add_argument("--report", default="")
    sc = sp.add_parser("selftest-cal")
    sc.add_argument("--m-jsonl", required=True)
    sc.add_argument("--report", default="")
    a = ap.parse_args()
    {"selftest-calf": cmd_selftest_calf, "selftest-cal": cmd_selftest_cal, "build": cmd_build, "p2": cmd_p2, "b1ci": cmd_b1ci, "classify-freeze": cmd_classify_freeze,
     "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    main()
