#!/usr/bin/env python3
"""UCI engine match runner with Elo estimation, SPRT and paired openings.

Examples
  python3 scripts/match.py --a ./build/heretic --b /usr/games/stockfish \
      --b-opt UCI_LimitStrength=true,UCI_Elo=2000 --games 40 --tc 3+0.05
  python3 scripts/match.py --a ./build/heretic --b ./build_old/heretic --sprt 0,5 --games 2000 --tc 1+0.02
"""
import argparse, concurrent.futures as cf, math, os, random, sys, threading, time
import chess, chess.engine, chess.pgn

ap = argparse.ArgumentParser()
ap.add_argument("--a", required=True, help="engine A binary")
ap.add_argument("--b", required=True, help="engine B binary")
ap.add_argument("--a-opt", default="", help="k=v,k=v UCI options for A")
ap.add_argument("--b-opt", default="")
ap.add_argument("--a-name", default="A")
ap.add_argument("--b-name", default="B")
ap.add_argument("--games", type=int, default=20, help="total games (rounded to even)")
ap.add_argument("--tc", default="2+0.05", help="base+inc in seconds, e.g. 10+0.1")
ap.add_argument("--movetime", type=float, default=0, help="fixed seconds per move (overrides --tc)")
ap.add_argument("--nodes", type=int, default=0, help="fixed nodes per move for A and B")
ap.add_argument("--concurrency", type=int, default=2)
ap.add_argument("--openings", default="data/openings.txt")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--sprt", default="", help="elo0,elo1 e.g. 0,5 (alpha=beta=0.05)")
ap.add_argument("--pgn", default="")
ap.add_argument("--adjudicate", type=int, default=1)
ap.add_argument("--quiet", action="store_true")
args = ap.parse_args()


def parse_opts(s):
    d = {}
    for kv in filter(None, s.split(",")):
        k, v = kv.split("=", 1)
        v = v.strip()
        d[k.strip()] = (v.lower() == "true") if v.lower() in ("true", "false") else (int(v) if v.lstrip("-").isdigit() else v)
    return d


optsA, optsB = parse_opts(args.a_opt), parse_opts(args.b_opt)
base, inc = (0.0, 0.0)
if args.movetime == 0 and args.nodes == 0:
    b_, i_ = args.tc.split("+")
    base, inc = float(b_), float(i_)

openings = []
if os.path.exists(args.openings):
    openings = [l.split() for l in open(args.openings) if l.strip()]
if not openings:
    openings = [[]]
rng = random.Random(args.seed)
order = list(range(len(openings)))
rng.shuffle(order)

lock = threading.Lock()
results = []  # from A's perspective: 1, 0.5, 0
tally = {"W": 0, "D": 0, "L": 0, "crash": 0, "time": 0, "illegal": 0}
pgns = []
stop_flag = threading.Event()


def open_engine(path, opts):
    e = chess.engine.SimpleEngine.popen_uci(path, timeout=30)
    cfg = {k: v for k, v in opts.items() if k in e.options}
    if len(cfg) != len(opts):
        missing = set(opts) - set(cfg)
        print(f"[warn] {path}: unknown options {missing}", file=sys.stderr)
    if cfg:
        e.configure(cfg)
    return e


def play_game(idx, a_white):
    opening = openings[order[(idx // 2) % len(order)]]
    board = chess.Board()
    for u in opening:
        board.push_uci(u)
    try:
        ea = open_engine(args.a, optsA)
        eb = open_engine(args.b, optsB)
    except Exception as ex:
        print("engine start failure:", ex, file=sys.stderr)
        return None
    clocks = {chess.WHITE: base, chess.BLACK: base}
    engs = {chess.WHITE: ea if a_white else eb, chess.BLACK: eb if a_white else ea}
    result, reason = None, ""
    adj = {chess.WHITE: 0, chess.BLACK: 0}
    node = chess.pgn.Game()
    node.setup(chess.Board())
    nd = node
    for u in opening:
        nd = nd.add_variation(chess.Move.from_uci(u))
    try:
        while not board.is_game_over(claim_draw=True):
            side = board.turn
            eng = engs[side]
            if args.nodes:
                lim = chess.engine.Limit(nodes=args.nodes)
            elif args.movetime:
                lim = chess.engine.Limit(time=args.movetime)
            else:
                lim = chess.engine.Limit(white_clock=clocks[chess.WHITE], black_clock=clocks[chess.BLACK],
                                         white_inc=inc, black_inc=inc)
            t0 = time.time()
            try:
                r = eng.play(board, lim, info=chess.engine.INFO_SCORE)
            except (chess.engine.EngineTerminatedError, chess.engine.EngineError):
                with lock:
                    tally["crash"] += 1
                result = "0-1" if side == chess.WHITE else "1-0"
                reason = "engine crash"
                break
            dt = time.time() - t0
            if not args.nodes and not args.movetime:
                clocks[side] = clocks[side] - dt + inc
                if clocks[side] < 0:
                    with lock:
                        tally["time"] += 1
                    result = "0-1" if side == chess.WHITE else "1-0"
                    reason = "time forfeit"
                    break
            if r.move is None or r.move not in board.legal_moves:
                with lock:
                    tally["illegal"] += 1
                result = "0-1" if side == chess.WHITE else "1-0"
                reason = "illegal move"
                break
            sc = r.info.get("score")
            board.push(r.move)
            nd = nd.add_variation(r.move)
            if args.adjudicate and sc is not None:
                v = sc.white().score(mate_score=10000)
                if v is not None and abs(v) >= 1000 and len(board.move_stack) > 20:
                    adj[chess.WHITE if v > 0 else chess.BLACK] += 1
                    adj[chess.BLACK if v > 0 else chess.WHITE] = 0
                else:
                    adj = {chess.WHITE: 0, chess.BLACK: 0}
                if adj[chess.WHITE] >= 8:
                    result, reason = "1-0", "adjudicated"
                    break
                if adj[chess.BLACK] >= 8:
                    result, reason = "0-1", "adjudicated"
                    break
        if result is None:
            o = board.outcome(claim_draw=True)
            result = o.result() if o else "1/2-1/2"
            reason = o.termination.name if o else "unknown"
    finally:
        for e in (ea, eb):
            try:
                e.quit()
            except Exception:
                try:
                    e.close()
                except Exception:
                    pass
    node.headers["White"] = args.a_name if a_white else args.b_name
    node.headers["Black"] = args.b_name if a_white else args.a_name
    node.headers["Result"] = result
    node.headers["Termination"] = reason
    score_a = {"1-0": 1.0 if a_white else 0.0, "0-1": 0.0 if a_white else 1.0, "1/2-1/2": 0.5}[result]
    return score_a, str(node)


def elo_from_score(s):
    s = min(max(s, 1e-6), 1 - 1e-6)
    return -400 * math.log10(1 / s - 1)


def stats(res):
    n = len(res)
    if n == 0:
        return None
    w = sum(1 for r in res if r == 1)
    d = sum(1 for r in res if r == 0.5)
    l = n - w - d
    s = (w + 0.5 * d) / n
    var = (w * (1 - s) ** 2 + d * (0.5 - s) ** 2 + l * s ** 2) / n
    se = math.sqrt(var / n) if n > 1 else 1
    lo, hi = elo_from_score(s - 1.96 * se), elo_from_score(s + 1.96 * se)
    los = 0.5 * (1 + math.erf((w - l) / math.sqrt(2 * (w + l)))) if (w + l) > 0 else 0.5
    return dict(n=n, w=w, d=d, l=l, score=s, elo=elo_from_score(s), lo=lo, hi=hi, los=los, var=var)


def sprt_llr(res, elo0, elo1):
    st = stats(res)
    if not st or st["n"] < 2 or st["var"] == 0:
        return 0.0
    s0 = 1 / (1 + 10 ** (-elo0 / 400))
    s1 = 1 / (1 + 10 ** (-elo1 / 400))
    return (st["n"] * (s1 - s0) * (2 * st["score"] - s0 - s1)) / (2 * st["var"])


sprt = None
if args.sprt:
    e0, e1 = map(float, args.sprt.split(","))
    sprt = (e0, e1, math.log(0.05 / 0.95), math.log(0.95 / 0.05))

total = args.games + (args.games % 2)
t_start = time.time()


def worker(idx):
    if stop_flag.is_set():
        return
    out = play_game(idx, a_white=(idx % 2 == 0))
    if out is None:
        return
    sc, pgn = out
    with lock:
        results.append(sc)
        pgns.append(pgn)
        tally["W" if sc == 1 else "D" if sc == 0.5 else "L"] += 1
        st = stats(results)
        if not args.quiet:
            line = (f"[{len(results):4d}/{total}] +{st['w']} ={st['d']} -{st['l']}  score {st['score']*100:5.1f}%  "
                    f"Elo {st['elo']:+6.1f} [{st['lo']:+.0f},{st['hi']:+.0f}]  LOS {st['los']*100:4.1f}%")
            if sprt:
                llr = sprt_llr(results, sprt[0], sprt[1])
                line += f"  LLR {llr:+.2f} [{sprt[2]:.2f},{sprt[3]:.2f}]"
            print(line, flush=True)
        if sprt:
            llr = sprt_llr(results, sprt[0], sprt[1])
            if llr >= sprt[3] or llr <= sprt[2]:
                stop_flag.set()


with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
    list(ex.map(worker, range(total)))

st = stats(results)
print("=" * 78)
print(f"{args.a_name} vs {args.b_name}: {st['n']} games  +{st['w']} ={st['d']} -{st['l']}  "
      f"score {st['score']*100:.1f}%  Elo {st['elo']:+.1f}  95% CI [{st['lo']:+.1f}, {st['hi']:+.1f}]  LOS {st['los']*100:.1f}%")
print(f"anomalies: crashes={tally['crash']} time-forfeits={tally['time']} illegal={tally['illegal']}  "
      f"wall {time.time()-t_start:.0f}s")
if sprt:
    llr = sprt_llr(results, sprt[0], sprt[1])
    verdict = "H1 accepted (improvement)" if llr >= sprt[3] else "H0 accepted (no improvement)" if llr <= sprt[2] else "inconclusive"
    print(f"SPRT({sprt[0]},{sprt[1]}) LLR {llr:+.2f}: {verdict}")
if args.pgn:
    with open(args.pgn, "w") as f:
        f.write("\n\n".join(pgns) + "\n")
