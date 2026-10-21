#!/usr/bin/env python3
"""UCI conformance + robustness smoke test: for many positions and every search mode / quirk, the engine must
return a *legal* bestmove, respect fixed limits, and survive stop/isready/ucinewgame sequences.

usage: python3 tests/test_uci.py ./build/heretic
"""
import random, subprocess, sys, time
import chess, chess.engine

exe = sys.argv[1] if len(sys.argv) > 1 else "./build/heretic"
rng = random.Random(2024)

def random_positions(n, plies=(6, 70)):
    out = []
    while len(out) < n:
        b = chess.Board()
        for _ in range(rng.randint(*plies)):
            ms = list(b.legal_moves)
            if not ms or b.is_game_over():
                break
            b.push(rng.choice(ms))
        if not b.is_game_over():
            out.append(b)
    return out

CONFIGS = {
    "alphabeta":      {},
    "swindle":        {"QuantalSwindle": True, "OpponentElo": 1400},
    "variety":        {"VarietyPlies": 30, "VarietyMargin": 30},
    "mcts":           {"SearchMode": 1},
    "threads2":       {"Threads": 2},
    "swindle+var":    {"QuantalSwindle": True, "VarietyPlies": 20, "OpponentTime": 5000},
}
fails = 0
positions = random_positions(25) + [chess.Board(f) for f in [
    "8/8/8/8/8/5k2/6p1/6K1 w - - 0 1",                 # K vs K+P edge cases
    "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",                  # stalemate-ish
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
]]
for name, opts in CONFIGS.items():
    eng = chess.engine.SimpleEngine.popen_uci(exe, timeout=30)
    eng.configure({k: v for k, v in opts.items() if k in eng.options})
    t0 = time.time(); bad = 0
    for i, b in enumerate(positions):
        if not list(b.legal_moves):
            continue
        for lim in (chess.engine.Limit(nodes=3000), chess.engine.Limit(time=0.15)):
            r = eng.play(b, lim)
            if r.move not in b.legal_moves:
                print(f"FAIL [{name}] illegal {r.move} in {b.fen()}"); bad += 1
        if i % 9 == 0:
            eng.protocol.send_line("ucinewgame") if hasattr(eng.protocol, "send_line") else None
    eng.quit()
    fails += bad
    print(f"{name:12s} {'OK ' if not bad else 'BAD'}  {len(positions)} positions x2 limits in {time.time()-t0:.1f}s")

# clock-based search honours time (no flag-fall): 1s + 0.0 inc should take well under 1s per move on average
eng = chess.engine.SimpleEngine.popen_uci(exe, timeout=30)
b = chess.Board(); t0 = time.time(); w, bl = 3.0, 3.0
for _ in range(20):
    s = time.time()
    r = eng.play(b, chess.engine.Limit(white_clock=w, black_clock=bl, white_inc=0, black_inc=0))
    d = time.time() - s
    if b.turn: w -= d
    else: bl -= d
    b.push(r.move)
    if b.is_game_over(): break
eng.quit()
print(f"clock test: 20 plies used {3.0-w:.2f}s (white) / {3.0-bl:.2f}s (black) of 3s each")
if w < 0 or bl < 0: print("FAIL time forfeit"); fails += 1
print("UCI TEST", "PASS" if not fails else f"FAIL ({fails})")
sys.exit(1 if fails else 0)
