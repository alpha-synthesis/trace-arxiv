#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Positive control, OLS arms: one world per task (design 2026-10-08, step 3).

A process pool runs one world per task. Each task builds the world in its own
TRACE_ROOT / TRACE_OUT (positive_control_worlds.build_world, in the worker), then runs the
unchanged scripts through subprocesses with OMP/MKL/OPENBLAS_NUM_THREADS=1:

  b3_rolling_origin.py --origin {2021,2022,2023}-01-01 --estimator ols, then --aggregate
  b3_documented_effects.py --classify -> positive_control_worlds.py classify-freeze (verbatim copy, mapping
              asserted) -> b3_documented_effects.py --score --variant data --estimator ols
  b1_event_grounding.py (TRACE_B1_ARMS={"PC": tag}) and positive_control_worlds.py b1ci

and maps the scripts' own outputs to the composite B3 status (section "Blocks
and the composite B3 verdict"). The parent appends one JSONL row per world,
checks that every world of a replicate r carries the same recurring label hash
(and the same noise hash per noise variant), stores the gzipped manifest and
deletes the world. The only rerun is the design's: a world whose rolling-origin test is INCOMPLETE.

Tasks run in the design's CPU priority (M0, M1, M2, M3, EV, IN, HET, RR, EV0;
within an arm, beta in the listed order, then r). No task starts after
--stop-at, when given, and rows record their finish time.

Usage:
  python positive_control_ols.py --work W --jsonl results.jsonl --manifests M \
      [--arms M0,M1,...] [--workers 32] [--stop-at ISO_TIME]
      [--rs 9000-9002 --betas 0,0.25 --prefix pcsmoke]   (smoke test)
"""
from __future__ import annotations

import os

THREAD_ENV = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
os.environ.update(THREAD_ENV)                     # before numpy is imported anywhere

import argparse  # noqa: E402
import gzip  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import shutil  # noqa: E402
import socket  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
REF = None


def _init_worker():
    global REF
    import positive_control_worlds
    REF = positive_control_worlds.Ref()
    REF.garch()


def _run(name, argv, env, logs: Path, timing: dict):
    t0 = time.time()
    with open(logs / f"{name}.log", "w") as fh:
        p = subprocess.run(argv, env=env, stdout=fh, stderr=subprocess.STDOUT, cwd=str(logs))
    timing[name] = round(time.time() - t0, 1)
    if p.returncode != 0:
        tail = (logs / f"{name}.log").read_text()[-3000:]
        raise RuntimeError(f"step {name} exited {p.returncode}:\n{tail}")


def ro_status(verdict: dict) -> str:
    """b3_rolling_origin's own outputs -> rolling-origin status (design table)."""
    if verdict["verdict"] == "INCOMPLETE":
        return "DEFECT_INCOMPLETE"
    n = verdict["pooled"]["control_minus_window_permuted"]["n_tickers"]
    if n == 0:
        return "NOT_RUN_0"
    if n < 6:
        return "NOT_RUN_1_5"
    return verdict["verdict"]                     # PASS / FAIL (p None -> FAIL)


def de_status(report: dict, col: str = "sign") -> str:
    per = report[col]["per_class"]
    if not any(s and s["n"] >= 15 for s in per.values()):
        return "NOT_RUN"
    return report[col]["verdict"]


def parse_outputs(out: Path, tag: str, estimator: str = "ols") -> dict:
    suf = "" if estimator == "tft" else f"__{estimator}"
    nd = out / f"b3_rolling_origin__{tag}__data{suf}"
    v = json.loads((nd / "verdict.json").read_text())
    cw = v["pooled"]["control_minus_window_permuted"]
    lo = v["pooled"]["control_minus_label_only"]
    per_origin = {}
    for o in ("2021", "2022", "2023"):
        f = nd / f"origin_{o}.json"
        if not f.exists():
            continue
        rows = json.loads(f.read_text())["per_ticker"]

        def ag(r, arm):
            m = (r.get(arm) or {}).get("metric") or {}
            return m.get("agreement") if m.get("n_factors", 0) >= 5 else None
        ctrl = [ag(r, "control") for r in rows if ag(r, "control") is not None]
        lab = [ag(r, "label_only") for r in rows if ag(r, "label_only") is not None]
        hist = {}
        for r in rows:
            hist[str(r["n_eligible_factors"])] = hist.get(str(r["n_eligible_factors"]), 0) + 1
        per_origin[o] = {"n_tickers_ge5_factors": len(ctrl), "n_eligible_hist": hist,
                         "median_eligible_factors": float(sorted(r["n_eligible_factors"] for r in rows)[len(rows) // 2]),
                         "mean_control_agreement": (sum(ctrl) / len(ctrl)) if ctrl else None,
                         "mean_label_only_agreement": (sum(lab) / len(lab)) if lab else None,
                         **v["per_origin"].get(f"{o}-01-01", {}).get("control_minus_window_permuted", {})}
    ro = {"status": ro_status(v), "verdict": v["verdict"], "n_entering": cw["n_tickers"],
          "p": cw["p_one_sided"], "mean_diff": cw["mean_diff"], "pos_origins": cw["origins_with_positive_median"],
          "pass": cw["pass"], "per_origin": per_origin, "label_only": lo}
    qd = out / f"b3_documented_effects__{tag}__data{suf}"
    q = json.loads((qd / "report.json").read_text())
    import pandas as pd
    rows = pd.read_csv(qd / "rows.csv") if (qd / "rows.csv").exists() else None
    de = {"status": de_status(q), "verdict": q["sign"]["verdict"],
          "pooled": q["sign"]["pooled_tested_classes"], "per_class": q["sign"]["per_class"],
          "bad_classes": q["sign"]["classes_n20_below_half"],
          "sign_centered": {"status": de_status(q, "sign_centered"), "verdict": q["sign_centered"]["verdict"],
                            "pooled": q["sign_centered"]["pooled_tested_classes"]},
          "median_seed_agreement": q.get("median_seed_agreement"),
          "n_rows": int(len(rows)) if rows is not None else None}
    b3 = "PASS" if ro["status"] == "PASS" and de["status"] == "PASS" else "NOT_PASS"
    return {"ro": ro, "de": de, "b3": b3}


def parse_b1(out: Path, tag: str) -> dict:
    s = json.loads((out / "b1_event_grounding" / "summary.json").read_text())["PC"]
    ci = json.loads((out / f"b1_ci_{tag}.json").read_text())
    return {"agreement": s["pooled"]["agreement"], "independence": s["pooled"]["independence"],
            "n_scored": s["pooled"]["n_scored"], "p_label_shuffle": s["pooled"]["p_label_shuffle"],
            "null_mean": s["pooled"]["null_mean"], "placebo": s["next_window_placebo"],
            "reading": s["reading"], "market_reading_share": s["market_reading_share"],
            "noise_only_window_share": s["noise_only_window_share"],
            "excess": ci["excess"], "excess_ci95": ci["excess_ci95"], "placebo_excess": ci["placebo_excess"],
            "placebo_ci95": ci["placebo_ci95"], "n_placebo": ci["n_placebo"]}


def run_task(arm: str, r: int, beta: float, opts: dict) -> dict:
    import positive_control_worlds
    tag = positive_control_worlds.tag_of(arm, r, beta, opts["prefix"])
    wdir = Path(opts["work"]) / tag
    py = opts["python"]
    rerun = None
    for attempt in (0, 1):
        shutil.rmtree(wdir, ignore_errors=True)
        root, out, logs = wdir / "root", wdir / "out", wdir / "logs"
        logs.mkdir(parents=True)
        t0 = time.time()
        timing = {}
        man = positive_control_worlds.build_world(arm, r, beta, root, out, REF, tag=tag)
        timing["build"] = round(time.time() - t0, 1)
        env = dict(os.environ, TRACE_ROOT=str(root), TRACE_OUT=str(out), TRACE_RUN=tag,
                   TRACE_B1_ARMS=json.dumps({"PC": tag}), **THREAD_ENV)
        frozen = out / f"b3_documented_effects__{tag}" / "classified_FROZEN.csv"
        steps = [(f"ro_{o[:4]}", [py, str(SCRIPTS / "b3_rolling_origin.py"), "--origin", o, "--estimator", "ols"])
                 for o in positive_control_worlds.ORIGINS]
        steps += [("ro_aggregate", [py, str(SCRIPTS / "b3_rolling_origin.py"), "--aggregate", "--estimator", "ols"]),
                  ("de_classify", [py, str(SCRIPTS / "b3_documented_effects.py"), "--classify"]),
                  ("de_freeze", [py, str(SCRIPTS / "positive_control_worlds.py"), "classify-freeze", "--tag", tag]),
                  ("de_score", [py, str(SCRIPTS / "b3_documented_effects.py"), "--score", "--variant", "data",
                                "--frozen", str(frozen), "--estimator", "ols"]),
                  ("b1", [py, str(SCRIPTS / "b1_event_grounding.py")]),
                  ("b1ci", [py, str(SCRIPTS / "positive_control_worlds.py"), "b1ci", "--tag", tag])]
        for name, argv in steps:
            _run(name, argv, env, logs, timing)
        res = parse_outputs(out, tag, "ols")
        if res["ro"]["status"] == "DEFECT_INCOMPLETE" and attempt == 0:
            rerun = "exp_n INCOMPLETE (the design's only allowed rerun)"
            continue
        break
    row = {"tag": tag, "arm": arm, "r": r, "beta": beta, "status": "ok", "estimator": "ols",
           "calendar_only": man.get("calendar_only", False), "n_types_present": man.get("n_types_present"),
           "noise": man["noise"], "names": man["names"], "het": man["het"],
           "label_sub": man["label_sub"], "label_hash": man["label_hash"],
           "label_hash_recurring": man["label_hash_recurring"], "noise_hash": man["noise_hash"],
           "eta_hash": man["eta_hash"], "prices_hash": man["prices_hash"], "micro_shift": man["micro_shift"],
           "rr_offset": man["rr_offset"], "garch_fit_hash": man["garch_fit_hash"],
           "ro_eligible_stocks": {o: man["ro_eligibility"][o]["stocks_with_ge5"] for o in man["ro_eligibility"]},
           "de_expected": man["de_rows"], **res, "b1": parse_b1(out, tag), "rerun": rerun,
           "timing_s": timing, "wall_s": round(sum(timing.values()), 1), "host": socket.gethostname(),
           "finished_at": datetime.now().astimezone().isoformat(timespec="seconds")}
    if opts.get("manifests"):
        md = Path(opts["manifests"])
        md.mkdir(parents=True, exist_ok=True)
        with gzip.open(md / f"{tag}.json.gz", "wt") as fh:
            json.dump({k: v for k, v in man.items() if not k.startswith("_")}, fh)
    if not opts.get("keep"):
        shutil.rmtree(wdir, ignore_errors=True)
    return row


def task_list(arms, rs_override=None, betas_override=None):
    import positive_control_worlds
    tasks = []
    for arm in arms:
        spec = positive_control_worlds.ARMS[arm]
        for b in (betas_override if betas_override is not None else spec["betas"]):
            rs_b = spec.get("rs_beta", {}).get(b, spec["rs"])
            for r in (rs_override if rs_override is not None else rs_b):
                tasks.append((arm, int(r), float(b)))
    return tasks


def _parse_range(s):
    if not s:
        return None
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main():
    import positive_control_worlds
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--manifests", default="")
    ap.add_argument("--arms", default=",".join(positive_control_worlds.CPU_PRIORITY))
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--stop-at", default="", help="optional ISO time after which no task starts")
    ap.add_argument("--prefix", default="pc")
    ap.add_argument("--rs", default="")
    ap.add_argument("--betas", default="")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--pair-jsonl", default="", help="rows whose hashes CAL worlds must match (M arms)")
    a = ap.parse_args()
    stop_at = datetime.fromisoformat(a.stop_at) if a.stop_at else None
    betas = [float(x) for x in a.betas.split(",")] if a.betas else None
    tasks = task_list(a.arms.split(","), _parse_range(a.rs), betas)
    jl = Path(a.jsonl)
    jl.parent.mkdir(parents=True, exist_ok=True)
    done, lab_by_r, noise_by = set(), {}, {}

    def register(row):
        """Pairing checks across worlds of one replicate (labels; noise per variant)."""
        h = lab_by_r.setdefault(row["r"], row["label_hash_recurring"])
        if h != row["label_hash_recurring"]:
            raise AssertionError(f"{row['tag']}: recurring label hash differs from replicate {row['r']}'s")
        key = (row["r"], row["noise"])
        n = noise_by.setdefault(key, row["noise_hash"])
        if n != row["noise_hash"]:
            raise AssertionError(f"{row['tag']}: noise hash differs within replicate {key}")

    if a.pair_jsonl:
        for line in Path(a.pair_jsonl).read_text().splitlines():
            row = json.loads(line)
            if row.get("status") == "ok" and row["noise"] == "C":
                register(row)
    if jl.exists():
        for line in jl.read_text().splitlines():
            row = json.loads(line)
            if row.get("status") == "ok":
                done.add(row["tag"])
                register(row)
    todo = [t for t in tasks if positive_control_worlds.tag_of(*t, a.prefix) not in done]
    print(f"[{datetime.now():%F %T}] {len(tasks)} tasks, {len(done)} done, {len(todo)} to run, "
          f"{a.workers} workers, stop at {stop_at.isoformat() if stop_at else 'none'}", flush=True)
    _init_worker()                                  # GARCH fit cached once, before the workers
    opts = {"work": a.work, "prefix": a.prefix, "python": a.python, "manifests": a.manifests, "keep": a.keep}
    ctx = mp.get_context("spawn")
    n_ok = n_err = 0
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=ctx, initializer=_init_worker) as ex:
        it = iter(todo)
        inflight = {}

        def submit_next():
            if stop_at is not None and datetime.now().astimezone() >= stop_at:
                return False
            t = next(it, None)
            if t is None:
                return False
            inflight[ex.submit(run_task, *t, opts)] = t
            return True
        for _ in range(a.workers):
            if not submit_next():
                break
        with open(jl, "a") as fh:
            while inflight:
                fin, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
                for f in fin:
                    arm, r, b = inflight.pop(f)
                    tag = positive_control_worlds.tag_of(arm, r, b, a.prefix)
                    try:
                        row = f.result()
                        register(row)
                        n_ok += 1
                    except Exception as e:                      # noqa: BLE001
                        row = {"tag": tag, "arm": arm, "r": r, "beta": b, "status": "error",
                               "error": repr(e)[:4000], "traceback": traceback.format_exc()[-4000:],
                               "finished_at": datetime.now().astimezone().isoformat(timespec="seconds")}
                        n_err += 1
                        print(f"ERROR {tag}: {repr(e)[:500]}", flush=True)
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())
                    el = time.time() - t_start
                    print(f"[{datetime.now():%F %T}] {tag} {row['status']} "
                          f"({row.get('wall_s', '-')}s) | ok {n_ok} err {n_err} | "
                          f"{(n_ok + n_err) / max(el, 1) * 3600:.0f} worlds/h", flush=True)
                    submit_next()
    print(f"[{datetime.now():%F %T}] finished: ok {n_ok}, errors {n_err}", flush=True)


if __name__ == "__main__":
    main()
