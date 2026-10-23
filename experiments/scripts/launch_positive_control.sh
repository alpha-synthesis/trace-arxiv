#!/bin/bash
# Positive control launcher (design: the paper's appendix on the positive control).
#
# Two queues: CPU (OLS arms, positive_control_ols.py, 32 workers x 1 thread) and
# GPU (TFT worlds, positive_control_tft.py, 8 streams x 2 threads). Long jobs run under
# nohup with logs in $PC/logs; every launch writes a manifest with the SHA-256
# of every script, the paper and generator commits and the package versions.
#
#   launch_positive_control.sh selftest      seven checks (step 1)
#   launch_positive_control.sh smoke         smoke test (step 2): 2-stock TFT pipeline,
#                                            OLS driver on 3 replicates x 2 beta (r 9000-9002,
#                                            outside the run's replicates), fit timing at 1/4/8 streams
#   launch_positive_control.sh cpu           OLS arms in priority order (step 3)
#   launch_positive_control.sh gpu           TFT queue (step 4, with B2 and sign recovery)
#   launch_positive_control.sh cal           arm CAL, real-calendar types only, least squares
#   launch_positive_control.sh calf          arm CALF, CAL regenerated with a per-world microstructure offset
#   launch_positive_control.sh status
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
PC=${PC_HOME:-$REPO/pc_runs}
export TRACE_ORIG=$REPO
export TRACE_PC_CACHE=$PC/cache
PY=${PYTHON:-python}
S=$REPO/experiments/scripts
LOGS=$PC/logs
STOP=${PC_STOP_AT:+--stop-at $PC_STOP_AT}
mkdir -p "$LOGS" "$PC/runs"
unset TRACE_ROOT TRACE_OUT TRACE_RUN TRACE_B1_ARMS TRACE_B2_REF

launch_manifest() {   # $1 = name
  local f=$LOGS/launch_${1}_$(date +%Y%m%dT%H%M%S).json
  {
    echo "{"
    echo "  \"what\": \"$1\", \"host\": \"$(hostname)\", \"date\": \"$(date -Iseconds)\","
    echo "  \"repository_head\": \"$(git -C $REPO rev-parse HEAD 2>/dev/null || echo unknown)\","
    echo "  \"stop_at\": \"${PC_STOP_AT:-none}\","
    echo "  \"versions\": \"$($PY -c 'import sys,torch,numpy,scipy,pandas,arch;print(sys.version.split()[0],torch.__version__,numpy.__version__,scipy.__version__,pandas.__version__,arch.__version__)')\","
    echo "  \"pip_freeze_sha256\": \"$($PY -m pip freeze 2>/dev/null | sha256sum | cut -c1-64)\","
    echo "  \"gpu\": \"$(nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ' ')\","
    echo "  \"scripts_sha256\": {"
    (cd "$S" && sha256sum *.py positive_control_calendars.csv launch_positive_control.sh | awk '{printf "    \"%s\": \"%s\",\n", $2, $1}' | sed '$ s/,$//')
    echo "  }"
    echo "}"
  } > "$f"
  echo "launch manifest: $f"
}

gpu_free() {
  local apps
  apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader)
  if [ -n "$apps" ]; then
    echo "GPU busy: $apps"
    return 1
  fi
  return 0
}

case "${1:-}" in
  selftest)
    launch_manifest selftest
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
      $PY $S/positive_control_worlds.py selftest --report $LOGS/selftest_report.json 2>&1 | tee $LOGS/selftest.log
    ;;
  smoke)
    launch_manifest smoke
    gpu_free || { echo "refusing: GPU in use"; exit 2; }
    SM=$PC/runs/smoke
    mkdir -p $SM
    # (a) the 2-stock TFT pipeline: build -> P2 -> five-seed estimates -> rolling origin 2021 (TFT, OLS) -> documented effects (TFT, OLS)
    #     -> B2 (oracle, b = 0, TFT) -> sign recovery seed 0
    $PY $S/positive_control_tft.py --work $SM/tft --smoke --streams 4 --cpu-slots 4 $STOP \
      > $LOGS/smoke_tft.log 2>&1
    # (b) fit timing at 1, 4 and 8 streams: k processes, each 5 five-seed fits on AAPL of the smoke world
    for k in 1 4 8; do
      t0=$(date +%s.%N)
      for i in $(seq 1 $k); do
        TRACE_ROOT=$SM/tft/root TRACE_OUT=$SM/streams/k${k}_$i TRACE_RUN=pcsmoke_TFT_r000_b0.25 \
          OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
          $PY $S/five_seed_estimates.py --variant data --tickers AAPL > $LOGS/smoke_streams_k${k}_$i.log 2>&1 &
      done
      wait
      t1=$(date +%s.%N)
      echo "streams=$k wall_s=$(awk "BEGIN{print $t1 - $t0}") fits=$((5 * k))" | tee -a $LOGS/smoke_streams.txt
    done
    # (c) OLS driver: 3 replicates x 2 beta, full 28-stock worlds
    $PY $S/positive_control_ols.py --work $SM/ols_work --jsonl $SM/ols_smoke.jsonl --manifests $SM/ols_manifests \
      --arms M0 --rs 9000-9002 --betas 0,0.25 --prefix pcsmoke --workers 6 $STOP \
      > $LOGS/smoke_ols.log 2>&1
    echo "smoke done"
    ;;
  cpu)
    launch_manifest cpu
    mkdir -p $PC/runs/ols
    nohup $PY $S/positive_control_ols.py --work $PC/runs/ols/work --jsonl $PC/runs/ols/results.jsonl \
      --manifests $PC/runs/ols/manifests --workers ${PC_CPU_WORKERS:-32} $STOP \
      >> $LOGS/cpu_queue.log 2>&1 &
    echo "CPU queue pid $!"
    ;;
  cal)
    # arm CAL (real-calendar types only), least squares only; paired with the M rows
    launch_manifest cal
    mkdir -p $PC/runs/cal
    nohup $PY $S/positive_control_ols.py --arms CAL --work $PC/runs/cal/work --jsonl $PC/runs/cal/results.jsonl \
      --manifests $PC/runs/cal/manifests --pair-jsonl $PC/runs/ols/results.jsonl \
      --workers ${PC_CPU_WORKERS:-32} $STOP >> $LOGS/cal_queue.log 2>&1 &
    echo "CAL queue pid $!"
    ;;
  calf)
    # arm CALF (CAL with a per-world microstructure offset), least squares; paired with the M rows
    launch_manifest calf
    mkdir -p $PC/runs/calf
    nohup $PY $S/positive_control_ols.py --arms CALF --work $PC/runs/calf/work --jsonl $PC/runs/calf/results.jsonl \
      --manifests $PC/runs/calf/manifests --pair-jsonl $PC/runs/ols/results.jsonl \
      --workers ${PC_CPU_WORKERS:-32} $STOP >> $LOGS/calf_queue.log 2>&1 &
    echo "CALF queue pid $!"
    ;;
  gpu)
    launch_manifest gpu
    gpu_free || { echo "refusing: GPU in use (report, do not kill)"; exit 2; }
    mkdir -p $PC/runs/tft
    nohup $PY $S/positive_control_tft.py --work $PC/runs/tft --streams ${PC_STREAMS:-8} --cpu-slots ${PC_CPU_SLOTS:-2} \
      --groups ${PC_GROUPS:-G1,G2,G3,G4} $STOP >> $LOGS/gpu_queue.log 2>&1 &
    echo "GPU queue pid $!"
    ;;
  status)
    echo "== CPU"; tail -n 3 $LOGS/cpu_queue.log 2>/dev/null || true
    [ -f $PC/runs/ols/results.jsonl ] && wc -l $PC/runs/ols/results.jsonl
    echo "== GPU"; tail -n 5 $LOGS/gpu_queue.log 2>/dev/null || true
    nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
    ;;
  *)
    echo "usage: $0 {selftest|smoke|cpu|cal|calf|gpu|status}"; exit 1 ;;
esac
