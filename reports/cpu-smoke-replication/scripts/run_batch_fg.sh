#!/usr/bin/env bash
# Foreground (harness-tracked) batch run: stops any previous run, then runs the resumable batch.
S=${WORK:?Set WORK to the scratch directory}
EXP=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if grep -q "^finished" $S/runs/batch_opus_latest.log 2>/dev/null && ! grep -q "HALT\|budget reached" $S/runs/batch_opus_latest.log; then echo "already finished"; exit 0; fi
pkill -f "batch.py --model opus"; sleep 1; pkill -f "run_episode_p.sh"; pkill -f "claude_agent.py"; pkill -f "environment.server"; sleep 2
log=$S/runs/batch_opus_$(date -u +%Y%m%dT%H%M%S).log
ln -sfn $log $S/runs/batch_opus_latest.log
cd $EXP && exec python -u batch.py --model opus --workers 2 --budget-usd 150 > $log 2>&1 < /dev/null
