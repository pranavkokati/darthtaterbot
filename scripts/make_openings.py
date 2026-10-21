#!/usr/bin/env python3
"""Generate a balanced, diverse opening suite.

Random legal-move prefixes are filtered with a reference engine so that every opening
starts within +-`--window` centipawns. This keeps A/B matches informative without relying on
a hand-typed book. Output: one line per opening, "uci1 uci2 ... uciN".
"""
import argparse, random, sys
import chess, chess.engine

ap = argparse.ArgumentParser()
ap.add_argument("--engine", default="/usr/games/stockfish")
ap.add_argument("--count", type=int, default=200)
ap.add_argument("--plies", type=int, default=6)
ap.add_argument("--window", type=int, default=45)
ap.add_argument("--depth", type=int, default=10)
ap.add_argument("--seed", type=int, default=2026)
ap.add_argument("--out", default="data/openings.txt")
a = ap.parse_args()

rng = random.Random(a.seed)
eng = chess.engine.SimpleEngine.popen_uci(a.engine)
seen, lines = set(), []
tries = 0
while len(lines) < a.count and tries < a.count * 60:
    tries += 1
    b = chess.Board()
    mv = []
    ok = True
    for _ in range(a.plies):
        legal = list(b.legal_moves)
        if not legal:
            ok = False
            break
        m = rng.choice(legal)
        # avoid silly first plies: no hanging queen moves, no early king walks
        if b.piece_type_at(m.from_square) == chess.KING and not b.is_castling(m):
            ok = False
            break
        mv.append(m)
        b.push(m)
    if not ok or b.is_game_over():
        continue
    key = b.board_fen() + str(b.turn)
    if key in seen:
        continue
    info = eng.analyse(b, chess.engine.Limit(depth=a.depth))
    sc = info["score"].white().score(mate_score=10000)
    if sc is None or abs(sc) > a.window:
        continue
    seen.add(key)
    lines.append(" ".join(m.uci() for m in mv))
eng.quit()
with open(a.out, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"wrote {len(lines)} balanced openings to {a.out} ({tries} candidates tried)")
