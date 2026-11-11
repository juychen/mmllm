#!/bin/bash
# Wait for the three running region jobs, then start the last two on GPU 6/7.
cd /home/junyichen/code/mmllm || exit 1
while pgrep -f "run_region_bp0_scratch.sh" > /dev/null; do sleep 300; done
sleep 60
REGION=PFC_MW GPU=6 setsid nohup ./run_region_bp0_scratch.sh > /tmp/launch_PFC_MW.log 2>&1 < /dev/null &
disown
REGION=PFC_MC GPU=7 setsid nohup ./run_region_bp0_scratch.sh > /tmp/launch_PFC_MC.log 2>&1 < /dev/null &
disown
echo "[queue] PFC runs started at $(date)"
