#!/usr/bin/env bash
# R2: price-blind labels on the precomputed news contexts (cache/contexts_v2.1)
# + corrected code. Extraction calls an LLM (OPENROUTER_API_KEY). Data-sign run (extraction + training + generation),
# then the LLM-sign variant on the same labels (no LLM). Real exit codes.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
mkdir -p logs
STORE=cache/rag_store_v2/chroma.sqlite3
echo "=== $(date -Is) R2 preflight"
$PY scripts/preflight_r2.py configs/r2_datasign.yaml configs/r2_llmsign.yaml || exit 3
sha_before=$( [ -f "$STORE" ] && sha256sum "$STORE" | cut -d' ' -f1 || echo "store not present")
echo "store sha256 before: $sha_before"
echo "generator commit: $(git rev-parse HEAD)"
for v in datasign llmsign; do
  echo "=== $(date -Is) R2 $v start"
  $PY scripts/run_pipeline.py configs/r2_$v.yaml > logs/r2_$v.log 2>&1
  rc=$?
  echo "=== $(date -Is) R2 $v exit=$rc"
  [ $rc -eq 0 ] || exit $rc
done
sha_after=$( [ -f "$STORE" ] && sha256sum "$STORE" | cut -d' ' -f1 || echo "store not present")
echo "store sha256 after:  $sha_after  unchanged=$([ "$sha_before" = "$sha_after" ] && echo yes || echo NO)"
echo "=== LLM call counts (expected ~155 x 5 = 775 sent per ticker)"
$PY - <<'PYEOF'
import json, re, collections
st = collections.Counter(json.loads(l).get("status") for l in open("logs/llm_calls_r2.jsonl"))
print("call log:", dict(st))
tick = None
for line in open("logs/r2_datasign.log"):
    m = re.search(r"TICKER: (\w+)", line)
    if m: tick = m.group(1)
    m = re.search(r"Analysed (\d+) windows \((\d+) LLM calls sent\)", line)
    if m: print(f"  {tick:5s} windows={m.group(1)} sent={m.group(2)}")
PYEOF
