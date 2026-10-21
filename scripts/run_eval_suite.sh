#!/bin/bash
# Reproduces the headline measurements in README.md (2 cores, 3s+0.1s). Logs in results/.
cd "$(dirname "$0")/.."
H=./build/heretic
SF=/usr/games/stockfish
python3 scripts/match.py --a $H --b $SF --b-opt "UCI_LimitStrength=true,UCI_Elo=2800" --a-name heretic --b-name sf2800 --games 100 --tc 3+0.1 --concurrency 2 --quiet --pgn results/cal_sf2800_100.pgn > results/cal_sf2800_100.txt 2>&1
python3 scripts/match.py --a $H --b $H --a-opt "QuantalSwindle=true,OpponentElo=2800" --a-name swindle --b-name plain --games 100 --tc 3+0.1 --concurrency 2 --quiet --pgn results/swindle_vs_plain.pgn > results/swindle_vs_plain.txt 2>&1
python3 scripts/match.py --a $H --b $H --a-opt "SearchMode=1" --a-name mcts --b-name alphabeta --games 60 --tc 3+0.1 --concurrency 2 --quiet --pgn results/mcts_vs_ab.pgn > results/mcts_vs_ab.txt 2>&1
python3 scripts/match.py --a $H --b $SF --a-opt "QuantalSwindle=true,OpponentElo=2200" --b-opt "UCI_LimitStrength=true,UCI_Elo=2400" --a-name swindle --b-name sf2400 --games 60 --tc 3+0.1 --concurrency 2 --quiet --pgn results/swindle_vs_sf2400.pgn > results/swindle_vs_sf2400.txt 2>&1
python3 scripts/match.py --a $H --b $SF --b-opt "UCI_LimitStrength=true,UCI_Elo=2400" --a-name plain --b-name sf2400 --games 60 --tc 3+0.1 --concurrency 2 --quiet --pgn results/plain_vs_sf2400.pgn > results/plain_vs_sf2400.txt 2>&1
echo SUITE DONE > results/suite.done
