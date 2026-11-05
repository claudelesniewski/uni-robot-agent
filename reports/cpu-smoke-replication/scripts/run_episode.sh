#!/usr/bin/env bash
# Usage: run_episode.sh TASK_SUBPATH MODEL OUT_NAME [extra claude_agent args...]
set -uo pipefail
S=${WORK:?Set WORK to a scratch directory containing robouse-mujoco/ and runs/}
EXP=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$EXP/../../.." && pwd)
task=$1 model=$2 name=$3; shift 3
cd "$REPO"
pkill -f environment.server; sleep 2
export ROBOUSE_PYTHON=$S/robouse-mujoco/bin/python MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
T=$S/robouse-mujoco/lib/python3.12/site-packages/robouse/bundled_tasks
backend=$(basename $(dirname $task))
mkdir -p $S/runs/$name
bash scripts/robouse/run_${backend}.sh --task-dir $T/$task --output-dir $S/runs/$name/env --run-id $name --port 18780 > $S/runs/$name/env.log 2>&1 &
until curl -s -o /dev/null http://127.0.0.1:18780/; do sleep 2; done
source .venv/bin/activate
python -u $EXP/claude_agent.py --model $model --out $S/runs/$name/agent "$@" 2>&1 | grep -v " INFO" | tee $S/runs/$name/agent.log
pkill -f environment.server; sleep 3
echo "native: $(python3 -c "import json;r=json.load(open('$S/runs/$name/env/native/result.json'));print({k:r[k] for k in ('success','success_ever','steps_used')}, 'progress', r['metrics'].get('progress'))" 2>&1)"
