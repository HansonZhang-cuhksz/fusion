#!/usr/bin/env bash
# Serialized RTX 4060 campaign. ONE GPU -- never run two concurrently.
# Resumable: a family whose result JSON already exists is skipped.
set -u
cd /home/shuhan/fusion
export GLM52_RESULTS_DIR=/home/shuhan/fusion/results/rtx4060
mkdir -p "$GLM52_RESULTS_DIR" log/run4060
run () {  # name script resultjson args...
  local name=$1 script=$2 res=$3; shift 3
  if [ -f "$GLM52_RESULTS_DIR/$res" ]; then echo "=== [$name] skip (have $res) ==="; return; fi
  echo "=== [$name] $(date +%H:%M:%S) start ==="
  timeout 21600 python3 "$script" "$@" > "log/run4060/$name.log" 2>&1
  echo "=== [$name] $(date +%H:%M:%S) exit=$? ==="
  tail -3 "log/run4060/$name.log"
}
run f03    glm52/bench/bench_f03_resadd_rmsnorm.py   f03_resadd_rmsnorm.json
run f10    glm52/bench/bench_f10_merge_resadd.py     f10_merge_resadd.json
run f01    glm52/bench/bench_f01_oproj_resadd.py     f01_oproj_resadd.json
run f04f05 glm52/bench/bench_f04f05_norm_router.py   f04f05_norm_router.json
run f11    glm52/bench/bench_f11_lazy_prenorm.py     f11_lazy_prenorm.json --router-only
echo "ALL DONE $(date +%H:%M:%S)"
