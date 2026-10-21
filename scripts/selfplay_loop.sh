#!/bin/bash
# One iteration of the bootstrap loop (AlphaZero-style "engine generates data -> learn -> gate -> repeat").
#   usage: scripts/selfplay_loop.sh <iter> [positions] [depth]
# Every stage is also runnable by hand; see README.md. Requires: build/heretic, python3 + numpy + python-chess.
set -euo pipefail
cd "$(dirname "$0")/.."
IT=${1:?iteration number}; POS=${2:-1000000}; DEPTH=${3:-6}
BIN=./build/heretic
mkdir -p data tuned nets results

# 1. self-play data from the current engine (embedded HCE params, plus any net given via $NNUE)
OPT=""; [ -n "${NNUE:-}" ] && OPT="setoption name NNUEFile value $NNUE\n"
printf "${OPT}datagen $POS 2 $DEPTH data/it$IT.bin $((100+IT))\n" | $BIN | tail -1

# 2a. Texel-tune the HCE on all data so far (game-wise validation, L2 prior to the current parameters)
printf "tune 400000 4 2 tuned/hce_it$IT.params 1500 0.5 5e-9 $(ls data/it*.bin | tr '\n' ' ')\n" | $BIN | tail -2

# 2b. (experimental) residual NNUE on top of the HCE
printf "hceeval data/it$IT.bin data/it$IT.bin.hce\n" | $BIN | tail -1
python3 python/train_nnue.py --data data/it$IT.bin --residual --hidden 128 --epochs 30 --batch 2048 --lr 2e-3 --out nets/res_it$IT.nnue | tail -2

# 3. gate: a candidate is promoted only if it beats the incumbent in a fixed-node SPRT (0,5 Elo bounds)
python3 scripts/match.py --a $BIN --b $BIN --a-opt "EvalParams=$PWD/tuned/hce_it$IT.params" \
  --a-name candidate --b-name incumbent --games 400 --nodes 20000 --concurrency 2 --sprt 0,10 --quiet \
  --pgn results/gate_it$IT.pgn | tail -3
echo "If the gate passed: cp tuned/hce_it$IT.params.inc src/tuned_params.inc && make"
