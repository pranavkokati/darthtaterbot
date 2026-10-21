#!/usr/bin/env bash
# Supervisor: restarts the bot whenever it exits or the network drops (new env/proxy is re-read on each start).
# usage: LICHESS_TOKEN=lip_... scripts/run_bot.sh [lichess_bot.py args]
cd "$(dirname "$0")/.." || exit 1
while true; do
  python3 python/lichess_bot.py "$@"
  echo "$(date +%T) bot exited ($?); restarting in 5 s" >&2
  sleep 5
done
