#!/usr/bin/env bash
# Parallel-safe episode runner.
# Usage: run_episode_p.sh SCRIPT TASK_SUBPATH MODEL OUT_NAME PORT [extra claude_agent args...]
set -uo pipefail
S=${WORK:?Set WORK to the scratch directory}
EXP=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
script=$1 task=$2 model=$3 name=$4 port=$5; shift 5
cd "$EXP/../../.."
export ROBOUSE_PYTHON=$S/robouse-mujoco/bin/python MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
T=$S/robouse-mujoco/lib/python3.12/site-packages/robouse/bundled_tasks
out=$S/runs/batch/$name
rm -rf "$out"; mkdir -p "$out"
setsid bash scripts/robouse/run_${script}.sh --task-dir $T/$task --output-dir $out/env --run-id $name --port $port > $out/env.log 2>&1 &
envpid=$!
cleanup() { kill -- -$envpid 2>/dev/null; sleep 2; kill -9 -- -$envpid 2>/dev/null; }
trap cleanup EXIT
for i in $(seq 150); do
  curl -s -o /dev/null http://127.0.0.1:$port/ && break
  kill -0 $envpid 2>/dev/null || { echo "env failed to start" > $out/startup_error.txt; exit 3; }
  sleep 2
done
source .venv/bin/activate
python -u $EXP/claude_agent.py --env-url http://127.0.0.1:$port --model $model --out $out/agent "$@" 2>&1 | grep --line-buffered -v " INFO" > $out/agent.log
