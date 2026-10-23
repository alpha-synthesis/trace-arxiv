#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Positive control, TFT queue on the GPU, plus B2 and sign recovery (design
2026-10-08, steps 4-5). Started by launch_positive_control.sh.

Worlds: arm TFT (noise C, recurring names). Priorities (section "GPU budget"):
  G1  r = 0 at beta 0.25 and beta 0: P2, five-seed estimates, rolling origin
  G2  sign recovery on the r = 0, beta-0 world: sign_recovery.py --seed 0..4
  G3  r = 0 at beta 0.12
  G4  (stretch) r = 1, 2 at beta 0 and 0.25
Per world, GPU jobs (each one subprocess, 2 threads, run in --streams parallel
streams): positive_control_worlds.py p2 (stock chunks), five_seed_estimates.py --variant data
(one stock per job), b3_rolling_origin.py --origin O (TFT, one job per origin). CPU jobs
(run alongside): the build (with the oracle and b = 0 directories), b3_rolling_origin
--aggregate, b3_documented_effects --classify -> classify-freeze -> --score (TFT), the same
world through the OLS path (its numbers must equal the M-arm row of the same
r and beta), b1_event_grounding + b1ci, and b2_counterfactual in the B2 order: oracle
(TRACE_B2_REF = oracle tag), then b = 0, then TFT (TRACE_B2_REF = oracle tag).

A job starts when its dependencies are done; among ready GPU jobs the lowest
priority group goes first, then the longest. Each finished job leaves a marker
(resumable: done jobs are skipped, never rerun). No job starts after
--stop-at, when given.

Usage:
  python positive_control_tft.py --work W [--streams 8] [--groups G1,G2,G3,G4]
  python positive_control_tft.py --work W --smoke   (2 stocks, the design's smoke pipeline)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
GPU_THREADS = {"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"}
CPU_THREADS = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
ORIGINS = ("2021-01-01", "2022-01-01", "2023-01-01")


class Job:
    def __init__(self, jid, kind, prio, cost, argv, env, deps=()):
        self.jid, self.kind, self.prio, self.cost = jid, kind, prio, cost
        self.argv, self.env, self.deps = argv, env, list(deps)


def world_jobs(tag, arm, r, beta, prio, root: Path, outw: Path, py, tickers, origins=ORIGINS,
               power=False, power_seeds=(0, 1, 2, 3, 4), p2_chunk=7, smoke=False):
    """All jobs of one TFT world."""
    S = lambda f: str(SCRIPTS / f)                                  # noqa: E731
    base = {"TRACE_ROOT": str(root), "TRACE_OUT": str(outw), "TRACE_RUN": tag}
    J = []
    bid = f"{tag}:build"
    bargv = [py, S("positive_control_worlds.py"), "build", "--arm", arm, "--r", str(r), "--beta", str(beta),
             "--root", str(root), "--out", str(outw), "--b2", "--tag", tag]
    if smoke:
        bargv += ["--tickers", ",".join(tickers)]
    J.append(Job(bid, "cpu", prio, 1, bargv, {}))
    # GPU: P2, five-seed estimates, rolling origin (TFT)
    p2 = []
    for i in range(0, len(tickers), p2_chunk):
        ch = tickers[i:i + p2_chunk]
        jid = f"{tag}:p2:{ch[0]}"
        J.append(Job(jid, "gpu", prio, len(ch), [py, S("positive_control_worlds.py"), "p2", "--tag", tag, "--tickers", ",".join(ch)],
                     base, [bid]))
        p2.append(jid)
    ms = []
    for t in tickers:
        jid = f"{tag}:multiseed:{t}"
        J.append(Job(jid, "gpu", prio, 5, [py, S("five_seed_estimates.py"), "--variant", "data", "--tickers", t],
                     base, [bid]))
        ms.append(jid)
    ro = []
    for o in origins:
        jid = f"{tag}:ro_tft:{o[:4]}"
        J.append(Job(jid, "gpu", prio, 10 * len(tickers), [py, S("b3_rolling_origin.py"), "--origin", o], base, [bid]))
        ro.append(jid)
    # CPU follow-ups
    J.append(Job(f"{tag}:ro_tft_agg", "cpu", prio, 1, [py, S("b3_rolling_origin.py"), "--aggregate"], base, ro))
    J.append(Job(f"{tag}:de_classify", "cpu", prio, 1, [py, S("b3_documented_effects.py"), "--classify"], base, [bid]))
    J.append(Job(f"{tag}:de_freeze", "cpu", prio, 1, [py, S("positive_control_worlds.py"), "classify-freeze", "--tag", tag],
                 base, [f"{tag}:de_classify"]))
    frozen = str(outw / f"b3_documented_effects__{tag}" / "classified_FROZEN.csv")
    J.append(Job(f"{tag}:de_score_tft", "cpu", prio, 1, [py, S("b3_documented_effects.py"), "--score", "--variant", "data",
                                                       "--frozen", frozen], base, ms + [f"{tag}:de_freeze"]))
    rols = []
    for o in origins:
        jid = f"{tag}:ro_ols:{o[:4]}"
        J.append(Job(jid, "cpu", prio, 1, [py, S("b3_rolling_origin.py"), "--origin", o, "--estimator", "ols"], base, [bid]))
        rols.append(jid)
    J.append(Job(f"{tag}:ro_ols_agg", "cpu", prio, 1, [py, S("b3_rolling_origin.py"), "--aggregate", "--estimator", "ols"],
                 base, rols))
    J.append(Job(f"{tag}:de_score_ols", "cpu", prio, 1, [py, S("b3_documented_effects.py"), "--score", "--variant", "data",
                                                       "--frozen", frozen, "--estimator", "ols"],
                 base, [f"{tag}:de_freeze"]))
    J.append(Job(f"{tag}:b1", "cpu", prio, 1, [py, S("b1_event_grounding.py")],
                 {**base, "TRACE_B1_ARMS": json.dumps({"PC": tag})}, [bid]))
    J.append(Job(f"{tag}:b1ci", "cpu", prio, 1, [py, S("positive_control_worlds.py"), "b1ci", "--tag", tag], base, [bid]))
    orc, b0 = tag + "_oracle", tag + "_b0eng"
    J.append(Job(f"{tag}:b2_oracle", "cpu", prio, 1, [py, S("b2_counterfactual.py"), "--workers", "6"],
                 {**base, "TRACE_RUN": orc, "TRACE_B2_REF": orc}, [bid]))
    J.append(Job(f"{tag}:b2_b0eng", "cpu", prio, 1, [py, S("b2_counterfactual.py"), "--workers", "6"],
                 {**base, "TRACE_RUN": b0, "TRACE_B2_REF": orc}, [f"{tag}:b2_oracle"]))
    J.append(Job(f"{tag}:b2_tft", "cpu", prio, 1, [py, S("b2_counterfactual.py"), "--workers", "6"],
                 {**base, "TRACE_B2_REF": orc}, p2 + [f"{tag}:b2_b0eng"]))
    if power:
        pw = []
        gprio = "G2" if not smoke else prio
        for s in power_seeds:
            for i in range(0, len(tickers), 4):
                ch = tickers[i:i + 4]
                jid = f"{tag}:power:s{s}:{ch[0]}"
                J.append(Job(jid, "gpu", gprio, 4 * len(ch),
                             [py, S("sign_recovery.py"), "--tickers", ",".join(ch), "--seed", str(s)], base, [bid]))
                pw.append(jid)
        if not smoke:
            J.append(Job(f"{tag}:power_agg", "cpu", gprio, 1, [py, S("sign_recovery.py"), "--aggregate-majority"],
                         base, pw))
            J.append(Job(f"{tag}:power_agg1", "cpu", gprio, 1, [py, S("sign_recovery.py"), "--aggregate"], base, pw))
    return J


def plan(work: Path, py: str, groups, smoke=False, tickers=None):
    import positive_control_worlds
    root = work / "root"
    tick = tickers or positive_control_worlds.Ref().tickers
    jobs = []
    if smoke:
        tag = "pcsmoke_TFT_r000_b0.25"
        jobs += world_jobs(tag, "TFT", 0, 0.25, "G1", root, work / "out" / tag, py, tick,
                           origins=("2021-01-01",), power=True, power_seeds=(0,), smoke=True)
        return jobs
    worlds = []
    if "G1" in groups:
        worlds += [("G1", 0, 0.25), ("G1", 0, 0.0)]
    if "G3" in groups:
        worlds += [("G3", 0, 0.12)]
    if "G4" in groups:
        worlds += [("G4", 1, 0.0), ("G4", 1, 0.25), ("G4", 2, 0.0), ("G4", 2, 0.25)]
    for g, r, b in worlds:
        tag = positive_control_worlds.tag_of("TFT", r, b)
        jobs += world_jobs(tag, "TFT", r, b, g, root, work / "out" / tag, py, tick,
                           power=("G2" in groups and r == 0 and b == 0.0))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--streams", type=int, default=8)
    ap.add_argument("--cpu-slots", type=int, default=4)
    ap.add_argument("--groups", default="G1,G2,G3,G4")
    ap.add_argument("--stop-at", default="", help="optional ISO time after which no task starts")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-tickers", default="AAPL,JPM")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    work = Path(a.work)
    (work / "markers").mkdir(parents=True, exist_ok=True)
    (work / "logs").mkdir(parents=True, exist_ok=True)
    stop_at = datetime.fromisoformat(a.stop_at) if a.stop_at else None
    tick = a.smoke_tickers.split(",") if a.smoke else None
    jobs = plan(work, a.python, a.groups.split(","), smoke=a.smoke, tickers=tick)
    byid = {j.jid: j for j in jobs}
    order = {"G1": 1, "G2": 2, "G3": 3, "G4": 4}
    marker = lambda jid: work / "markers" / (jid.replace(":", "__").replace("/", "_") + ".done")   # noqa: E731
    done = {j.jid for j in jobs if marker(j.jid).exists()}
    print(f"[{datetime.now():%F %T}] {len(jobs)} jobs ({sum(j.kind == 'gpu' for j in jobs)} GPU), "
          f"{len(done)} already done; streams {a.streams}, cpu slots {a.cpu_slots}", flush=True)
    if a.dry_run:
        for j in jobs:
            print(j.jid, j.kind, j.prio, j.cost, j.deps)
        return
    running = {}                                     # jid -> (Popen, t0, fh)
    failed = set()
    t_start = time.time()
    while True:
        # reap
        for jid, (p, t0, fh) in list(running.items()):
            rc = p.poll()
            if rc is None:
                continue
            fh.close()
            del running[jid]
            el = round(time.time() - t0, 1)
            if rc == 0:
                marker(jid).write_text(json.dumps({"elapsed_s": el, "finished_at": datetime.now().astimezone().isoformat()}))
                done.add(jid)
                print(f"[{datetime.now():%F %T}] done {jid} ({el}s)", flush=True)
            else:
                failed.add(jid)
                print(f"[{datetime.now():%F %T}] FAILED {jid} rc={rc} ({el}s) log {work / 'logs'}", flush=True)
        pending = [j for j in jobs if j.jid not in done and j.jid not in running and j.jid not in failed]
        blocked = {j.jid for j in pending if any(d in failed for d in j.deps)}
        ready = [j for j in pending if j.jid not in blocked and all(d in done for d in j.deps)]
        now_ok = stop_at is None or datetime.now().astimezone() < stop_at
        for kind, slots in (("gpu", a.streams), ("cpu", a.cpu_slots)):
            n_run = sum(byid[x].kind == kind for x in running)
            cand = sorted([j for j in ready if j.kind == kind], key=lambda j: (order.get(j.prio, 9), -j.cost, j.jid))
            for j in cand:
                if n_run >= slots or not now_ok:
                    break
                env = dict(os.environ, **(GPU_THREADS if kind == "gpu" else CPU_THREADS), **j.env)
                logf = work / "logs" / (j.jid.replace(":", "__") + ".log")
                fh = open(logf, "a")
                fh.write(f"# {datetime.now().isoformat()} {' '.join(j.argv)}\n# env {json.dumps(j.env)}\n")
                fh.flush()
                running[j.jid] = (subprocess.Popen(j.argv, env=env, stdout=fh, stderr=subprocess.STDOUT,
                                                   cwd=str(work)), time.time(), fh)
                n_run += 1
                print(f"[{datetime.now():%F %T}] start {j.jid}", flush=True)
        if not running and (not ready or not now_ok):
            break
        time.sleep(2)
    left = [j.jid for j in jobs if j.jid not in done]
    print(f"[{datetime.now():%F %T}] queue finished in {time.time() - t_start:.0f}s: done {len(done)}, "
          f"failed {len(failed)}, not run {len(left) - len(failed)}", flush=True)
    (work / f"queue_status_{int(time.time())}.json").write_text(json.dumps(
        {"done": sorted(done), "failed": sorted(failed), "not_run": sorted(set(left) - failed)}, indent=1))


if __name__ == "__main__":
    main()
