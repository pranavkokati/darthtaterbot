#!/usr/bin/env python3
"""Perft regression suite. Reference node counts are the standard Chess Programming Wiki values."""
import subprocess, sys, re

BIN = sys.argv[1] if len(sys.argv) > 1 else "./build/perft"
MAXD = int(sys.argv[2]) if len(sys.argv) > 2 else 5

SUITE = [
    ("startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
     [20, 400, 8902, 197281, 4865609, 119060324]),
    ("kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
     [48, 2039, 97862, 4085603, 193690690]),
    ("pos3", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
     [14, 191, 2812, 43238, 674624, 11030083]),
    ("pos4", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
     [6, 264, 9467, 422333, 15833292]),
    ("pos5", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
     [44, 1486, 62379, 2103487, 89941194]),
    ("pos6", "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
     [46, 2079, 89890, 3894594, 164075551]),
]

fails = 0
for name, fen, expected in SUITE:
    for d in range(1, min(MAXD, len(expected)) + 1):
        out = subprocess.run([BIN, "perft", str(d), fen], capture_output=True, text=True).stdout
        m = re.search(r"= (\d+)", out)
        got = int(m.group(1)) if m else None
        ok = got == expected[d - 1]
        if not ok:
            fails += 1
        print(f"{name:9s} d{d}: {got} (expected {expected[d-1]}) {'OK' if ok else 'FAIL'}")
print("ALL PERFT PASSED" if not fails else f"{fails} PERFT FAILURES")
sys.exit(1 if fails else 0)
