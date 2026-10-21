#!/usr/bin/env python3
"""Heretic <-> Lichess Bot API bridge.

* Streams events, applies a challenge policy, plays each game with the UCI engine.
* Opponent-aware: feeds the opponent's rating/clock to the engine's "quirk" options.
* Persistent experience memory: remembers (position, move) pairs that lost and steers away from them.

Endpoints follow the official OpenAPI spec (lichess-org/api): /api/stream/event, /api/bot/game/stream/{id},
/api/bot/game/{id}/move/{move}, /api/bot/game/{id}/draw/{yes|no}, /api/bot/game/{id}/claim-victory,
/api/challenge/{id}/accept|decline, /api/bot/account/upgrade.

SECURITY: the OAuth token (scope `bot:play`) is read from the environment (default $LICHESS_TOKEN) and is never
written to disk or logs.   Upgrading an account to BOT is irreversible; it only happens with --upgrade.
"""
import argparse, json, logging, os, signal, sys, threading, time, tomllib
from collections import defaultdict

import chess, chess.engine, requests

log = logging.getLogger("heretic-bot")

DEFAULT_CONFIG = {
    "url": "https://lichess.org",
    "token_env": "LICHESS_TOKEN",
    "engine": {
        "path": "./build/heretic",
        "options": {"Hash": 128, "Threads": 1, "Move Overhead": 100, "Contempt": 0},
        "nnue": "",                       # optional path to a .nnue file
    },
    "games": {"max_concurrent": 1, "greeting": "Heretic here (bitboards + NNUE + a few heresies). Good luck, have fun!",
              "goodbye": "Good game!", "min_think_ms": 50, "default_correspondence_ms": 5000},
    "challenge": {
        "variants": ["standard"],
        "speeds": ["bullet", "blitz", "rapid", "classical"],
        "modes": ["casual", "rated"],
        "accept_bots": True, "accept_humans": True,
        "min_initial_s": 30, "max_initial_s": 1800, "min_increment_s": 0, "max_increment_s": 30,
        "max_games_per_opponent_per_hour": 4,
    },
    "behaviour": {
        "dynamic_contempt": True,         # contempt follows the rating gap
        "quantal_swindle": False,         # opt-in; applied only against non-BOT opponents (measured to COST Elo vs engines)
        "feed_opponent_clock": True,
        "accept_draw_below_cp": 10,       # accept draw offers when our score is <= this (and move >= 30)
        "resign_below_cp": -100000,       # disabled
        "use_experience": True, "experience_file": "data/experience.json", "experience_plies": 30,
    },
}


def merge(base, over):
    out = dict(base)
    for k, v in over.items():
        out[k] = merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


class LichessAPI:
    def __init__(self, token, base):
        self.base = base.rstrip("/")
        self.s = requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {token}", "User-Agent": "heretic-bot/1.0"})

    def _req(self, method, path, **kw):
        for attempt in range(4):
            try:
                r = self.s.request(method, self.base + path, timeout=kw.pop("timeout", 30), **kw)
            except requests.RequestException as e:
                log.warning("%s %s failed: %s", method, path, e)
                time.sleep(1 + attempt)
                continue
            if r.status_code == 429:
                log.warning("rate limited (429); sleeping 60s as required by Lichess")
                time.sleep(60)
                continue
            return r
        return None

    def account(self):
        r = self._req("GET", "/api/account")
        r.raise_for_status()
        return r.json()

    def upgrade(self):
        r = self._req("POST", "/api/bot/account/upgrade")
        return r.status_code, r.text

    def stream(self, path):
        r = self.s.get(self.base + path, stream=True, timeout=(10, 90))
        if r.status_code != 200:
            raise requests.HTTPError(f"{path}: HTTP {r.status_code} {r.text[:200]}")
        for line in r.iter_lines(chunk_size=None):
            if not line:
                continue  # keep-alive newline
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                log.warning("bad json line: %r", line[:100])

    def accept(self, cid):
        return self._req("POST", f"/api/challenge/{cid}/accept")

    def decline(self, cid, reason="generic"):
        return self._req("POST", f"/api/challenge/{cid}/decline", data={"reason": reason})

    def move(self, gid, uci, draw=False):
        return self._req("POST", f"/api/bot/game/{gid}/move/{uci}", params={"offeringDraw": "true"} if draw else None)

    def draw(self, gid, yes):
        return self._req("POST", f"/api/bot/game/{gid}/draw/{'yes' if yes else 'no'}")

    def resign(self, gid):
        return self._req("POST", f"/api/bot/game/{gid}/resign")

    def claim_victory(self, gid):
        return self._req("POST", f"/api/bot/game/{gid}/claim-victory")

    def challenge(self, username, seconds, inc, rated=False, color="random"):
        return self._req("POST", f"/api/challenge/{username}", data={
            "rated": "true" if rated else "false", "clock.limit": seconds, "clock.increment": inc, "color": color,
            "variant": "standard"})

    def chat(self, gid, text, room="player"):
        return self._req("POST", f"/api/bot/game/{gid}/chat", data={"room": room, "text": text})


class Experience:
    """(epd, move) -> [wins, draws, losses] from our point of view; JSON on disk."""

    def __init__(self, path):
        self.path, self.lock = path, threading.Lock()
        self.d = {}
        if path and os.path.exists(path):
            try:
                self.d = json.load(open(path))
            except Exception:
                log.warning("could not read experience file %s", path)

    def bad_moves(self, epd):
        with self.lock:
            return {m for m, (w, d, l) in self.d.get(epd, {}).items() if l > 0 and w == 0}

    def record(self, items, result):  # result in {1, 0.5, 0}
        idx = 0 if result == 1 else 1 if result == 0.5 else 2
        with self.lock:
            for epd, mv in items:
                e = self.d.setdefault(epd, {}).setdefault(mv, [0, 0, 0])
                e[idx] += 1
            if self.path:
                os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
                tmp = self.path + ".tmp"
                json.dump(self.d, open(tmp, "w"))
                os.replace(tmp, self.path)


class Bot:
    def __init__(self, cfg, api, me):
        self.cfg, self.api, self.me = cfg, api, me
        self.my_id = me["id"]
        self.games = {}
        self.lock = threading.Lock()
        self.recent = defaultdict(list)  # opponent id -> timestamps of accepted challenges
        self.exp = Experience(cfg["behaviour"]["experience_file"]) if cfg["behaviour"]["use_experience"] else None
        self.stop = threading.Event()
        self.my_ratings = {k: v.get("rating") for k, v in (me.get("perfs") or {}).items() if isinstance(v, dict)}

    # ------------------------------------------------------------ challenges
    def challenge_decision(self, ch):
        c = self.cfg["challenge"]
        v = ch.get("variant", {}).get("key", "standard")
        if v not in c["variants"]:
            return "variant"
        tc = ch.get("timeControl", {})
        if tc.get("type") not in ("clock",):
            return "timeControl" if tc.get("type") == "unlimited" else "later"
        init, incr = tc.get("limit", 0), tc.get("increment", 0)
        if init < c["min_initial_s"] or incr < c["min_increment_s"]:
            return "tooFast"
        if init > c["max_initial_s"] or incr > c["max_increment_s"]:
            return "tooSlow"
        if ch.get("speed") not in c["speeds"]:
            return "timeControl"
        mode = "rated" if ch.get("rated") else "casual"
        if mode not in c["modes"]:
            return mode if mode in ("rated", "casual") else "generic"
        ch_user = ch.get("challenger") or {}
        is_bot = ch_user.get("title") == "BOT"
        if is_bot and not c["accept_bots"]:
            return "noBot"
        if not is_bot and not c["accept_humans"]:
            return "onlyBot"
        with self.lock:
            if len(self.games) >= self.cfg["games"]["max_concurrent"]:
                return "later"
            now = time.time()
            hist = [t for t in self.recent[ch_user.get("id")] if now - t < 3600]
            if len(hist) >= c["max_games_per_opponent_per_hour"]:
                return "later"
        return None

    def on_challenge(self, ch):
        if (ch.get("challenger") or {}).get("id") == self.my_id:
            return  # our own outgoing challenge
        reason = self.challenge_decision(ch)
        cid = ch["id"]
        who = (ch.get("challenger") or {}).get("name", "?")
        if reason is None:
            log.info("accepting challenge %s from %s (%s %s)", cid, who, ch.get("speed"), "rated" if ch.get("rated") else "casual")
            r = self.api.accept(cid)
            if r is not None and r.status_code == 200:
                with self.lock:
                    self.recent[(ch.get("challenger") or {}).get("id")].append(time.time())
            else:
                log.warning("accept failed: %s", r.text[:200] if r is not None else "no response")
        else:
            log.info("declining challenge %s from %s: %s", cid, who, reason)
            self.api.decline(cid, reason)

    # ------------------------------------------------------------------ main
    def run(self):
        backoff = 1
        while not self.stop.is_set():
            try:
                for ev in self.api.stream("/api/stream/event"):
                    backoff = 1
                    t = ev.get("type")
                    if t == "challenge":
                        self.on_challenge(ev["challenge"])
                    elif t == "gameStart":
                        gid = ev["game"]["id"]
                        with self.lock:
                            if gid in self.games:
                                continue
                            g = Game(self, gid)
                            self.games[gid] = g
                        g.start()
                    elif t == "gameFinish":
                        log.info("game %s finished (%s)", ev["game"]["id"], ev["game"].get("status", {}))
                    if self.stop.is_set():
                        break
            except (requests.RequestException, ConnectionError) as e:
                log.warning("event stream error: %s; reconnecting in %ss", e, backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
        log.info("shutting down; waiting for %d game(s)", len(self.games))
        for g in list(self.games.values()):
            g.join(timeout=30)


class Game(threading.Thread):
    def __init__(self, bot, gid):
        super().__init__(daemon=True, name=f"game-{gid}")
        self.bot, self.api, self.gid = bot, bot.api, gid
        self.cfg = bot.cfg
        self.color = None
        self.board = None
        self.engine = None
        self.opp = {}
        self.last_len = -1
        self.lag_samples = []
        self.rtt_probe_ms = None
        self.last_score = None
        self.exp_items = []
        self.finished = False
        self.opp_gone_timer = None

    # --------------------------------------------------------------- engine
    def open_engine(self):
        e = self.cfg["engine"]
        self.engine = chess.engine.SimpleEngine.popen_uci(e["path"], timeout=60)
        try:  # one cheap round-trip probe so the very first move already uses a measured overhead
            t0 = time.time()
            self.api.account()
            self.rtt_probe_ms = (time.time() - t0) * 1000.0
        except Exception:
            pass
        opts = dict(e["options"])
        if e.get("nnue"):
            opts["NNUEFile"] = e["nnue"]
        threads = max(1, int(opts.get("Threads", 1)) // max(1, self.cfg["games"]["max_concurrent"]))
        opts["Threads"] = threads
        known = self.engine.options
        self.engine.configure({k: v for k, v in opts.items() if k in known})

    def tune_for_opponent(self, opp_clock_ms):
        """Per-move opponent modelling. Silently skips options the engine lacks."""
        b = self.cfg["behaviour"]
        known = self.engine.options
        cfg = {}
        opp_rating = self.opp.get("rating") or 0  # AI/anonymous opponents have no rating
        speed = self.speed
        my_rating = self.bot.my_ratings.get(speed) or 0
        if "OpponentElo" in known:
            cfg["OpponentElo"] = int(opp_rating)
        if "OpponentTime" in known:
            ok = b["feed_opponent_clock"] and opp_clock_ms and 0 < opp_clock_ms < 36_000_000  # unlimited games send ~2^31
            cfg["OpponentTime"] = int(opp_clock_ms) if ok else 0
        if "QuantalSwindle" in known:
            # the quantal-response model assumes a fallible opponent: never use it against other engines
            cfg["QuantalSwindle"] = bool(b["quantal_swindle"]) and self.opp.get("title") != "BOT"
        if b["dynamic_contempt"] and "Contempt" in known and opp_rating and my_rating:
            gap = my_rating - opp_rating
            cfg["Contempt"] = int(max(-20, min(60, gap / 8)))
        for k, v in cfg.items():  # never let an opponent-modelling option kill the game
            try:
                self.engine.configure({k: v})
            except Exception as e:
                log.warning("could not set %s=%s: %s", k, v, e)

    def measured_overhead_ms(self):
        """Move Overhead from *measured* round-trips (recent max blended with the mean, +30 % margin), never a constant."""
        if not self.lag_samples:
            return None
        mean = sum(self.lag_samples) / len(self.lag_samples)
        return int(max(15, min(2000, 1.3 * (0.5 * max(self.lag_samples) + 0.5 * mean))))

    def choose_move(self, wtime, btime, winc, binc, ply):
        us = self.color
        oh = self.measured_overhead_ms()
        if oh is None and self.rtt_probe_ms is not None:
            oh = int(max(15, 1.5 * self.rtt_probe_ms))
        if oh is not None and not self.cfg["games"].get("fixed_overhead"):
            try:
                self.engine.configure({"Move Overhead": oh})
            except Exception as e:
                log.warning("could not set Move Overhead: %s", e)
        opp_clock = btime if us == chess.WHITE else wtime
        self.tune_for_opponent(opp_clock)
        if wtime is None or btime is None or min(wtime, btime) > 36_000_000:  # unlimited / correspondence
            lim = chess.engine.Limit(time=self.cfg["games"]["default_correspondence_ms"] / 1000)
        else:
            lim = chess.engine.Limit(white_clock=wtime / 1000, black_clock=btime / 1000,
                                     white_inc=(winc or 0) / 1000, black_inc=(binc or 0) / 1000)
        res = self.engine.play(self.board, lim, info=chess.engine.INFO_SCORE)
        move = res.move
        sc = res.info.get("score")
        if sc is not None:
            self.last_score = sc.pov(us).score(mate_score=100000)
        # experience: avoid replaying a move that has lost from this exact position before
        exp = self.bot.exp
        if exp and move is not None and ply < self.cfg["behaviour"]["experience_plies"] * 2:
            epd = self.board.epd()
            bad = exp.bad_moves(epd)
            if move.uci() in bad:
                alts = [m for m in self.board.legal_moves if m.uci() not in bad]
                if alts:
                    remain = min((wtime if us == chess.WHITE else btime) or 60000, 600000) / 1000
                    t = max(0.05, min(remain / 30, 3.0))
                    res2 = self.engine.play(self.board, chess.engine.Limit(time=t), root_moves=alts,
                                            info=chess.engine.INFO_SCORE)
                    sc2 = res2.info.get("score")
                    s2 = sc2.pov(us).score(mate_score=100000) if sc2 else None
                    if res2.move and (s2 is None or self.last_score is None or s2 >= self.last_score - 80):
                        log.info("experience: avoiding previously-losing %s in %s -> %s", move.uci(), epd, res2.move.uci())
                        move = res2.move
                        if s2 is not None:
                            self.last_score = s2
            if self.cfg["behaviour"]["use_experience"]:
                self.exp_items.append((epd, move.uci()))
        return move

    # ------------------------------------------------------------- game loop
    def handle_state(self, st):
        status = st.get("status", "started")
        if status != "started":
            self.finish(st)
            return True
        moves = st.get("moves", "").split()
        board = chess.Board(self.initial_fen) if self.initial_fen else chess.Board()
        for m in moves:
            board.push_uci(m)
        self.board = board
        # draw offers from the opponent
        opp_offers = st.get("bdraw") if self.color == chess.WHITE else st.get("wdraw")
        if opp_offers:
            thr = self.cfg["behaviour"]["accept_draw_below_cp"]
            ok = self.last_score is not None and self.last_score <= thr and len(moves) >= 60
            log.info("opponent offers draw; our last score %s -> %s", self.last_score, "accept" if ok else "decline")
            self.api.draw(self.gid, ok)
        if board.turn == self.color and not board.is_game_over() and len(moves) != self.last_len:
            self.last_len = len(moves)
            t0 = time.time()
            try:
                move = self.choose_move(st.get("wtime"), st.get("btime"), st.get("winc", 0), st.get("binc", 0), len(moves))
            except Exception as e:
                # Never forfeit on a bridge bug: fall back to a fast engine move, then to any legal move.
                log.exception("choose_move failed in %s (%s); falling back", self.gid, e)
                move = None
                try:
                    move = self.engine.play(self.board, chess.engine.Limit(time=0.1)).move
                except Exception:
                    log.error("fallback engine search failed too; playing first legal move")
                if move is None:
                    move = next(iter(self.board.legal_moves))
            resign_thr = self.cfg["behaviour"]["resign_below_cp"]
            if self.last_score is not None and self.last_score <= resign_thr and len(moves) > 40:
                log.info("resigning (score %s)", self.last_score)
                self.api.resign(self.gid)
                return True
            t_post = time.time()
            r = self.api.move(self.gid, move.uci())
            self.lag_samples.append((time.time() - t_post) * 1000.0)
            self.lag_samples = self.lag_samples[-8:]
            if r is None or r.status_code != 200:
                log.error("move %s rejected: %s", move.uci(), r.text[:200] if r is not None else "no response")
            else:
                log.info("%s: played %s in %.2fs (score %s, clocks w=%s b=%s)", self.gid, move.uci(), time.time() - t0,
                         self.last_score, st.get("wtime"), st.get("btime"))
        return False

    def finish(self, st):
        if self.finished:
            return
        self.finished = True
        winner = st.get("winner")
        mine = "white" if self.color == chess.WHITE else "black"
        res = 0.5 if winner is None else (1 if winner == mine else 0)
        log.info("game %s over: %s, winner=%s -> we scored %s", self.gid, st.get("status"), winner, res)
        if self.bot.exp and self.exp_items:
            self.bot.exp.record(self.exp_items, res)
        try:
            g = self.cfg["games"]["goodbye"]
            if g:
                self.api.chat(self.gid, g)
        except Exception:
            pass

    def run(self):
        try:
            self.open_engine()
            done = False
            for attempt in range(6):
                try:
                    for ev in self.api.stream(f"/api/bot/game/stream/{self.gid}"):
                        t = ev.get("type")
                        if t == "gameFull":
                            self.initial_fen = None if ev.get("initialFen", "startpos") == "startpos" else ev["initialFen"]
                            white, black = ev["white"], ev["black"]
                            self.color = chess.WHITE if white.get("id") == self.bot.my_id else chess.BLACK
                            self.opp = black if self.color == chess.WHITE else white
                            self.speed = ev.get("speed", "blitz")
                            log.info("game %s: playing %s vs %s (%s)", self.gid, "white" if self.color else "black",
                                     self.opp.get("name"), self.opp.get("rating"))
                            if self.cfg["games"]["greeting"] and not ev["state"].get("moves"):
                                self.api.chat(self.gid, self.cfg["games"]["greeting"])
                            done = self.handle_state(ev["state"])
                        elif t == "gameState":
                            done = self.handle_state(ev)
                        elif t == "opponentGone" and ev.get("gone"):
                            secs = ev.get("claimWinInSeconds", 60)
                            self.opp_gone_timer = threading.Timer(secs + 1, lambda: self.api.claim_victory(self.gid))
                            self.opp_gone_timer.daemon = True
                            self.opp_gone_timer.start()
                        elif t == "opponentGone" and self.opp_gone_timer:
                            self.opp_gone_timer.cancel()
                        if done:
                            break
                    if done or self.finished:
                        break
                except (requests.RequestException, ConnectionError) as e:
                    log.warning("game stream %s dropped: %s (retry %d)", self.gid, e, attempt + 1)
                    time.sleep(1 + attempt)
        except Exception:
            log.exception("game %s crashed", self.gid)
        finally:
            if self.opp_gone_timer:
                self.opp_gone_timer.cancel()
            try:
                if self.engine:
                    self.engine.quit()
            except Exception:
                pass
            with self.bot.lock:
                self.bot.games.pop(self.gid, None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.toml")
    ap.add_argument("--upgrade", action="store_true", help="IRREVERSIBLY upgrade this account to a BOT account")
    ap.add_argument("--challenge", metavar="USER", action="append", default=[],
                    help="challenge this user on startup (repeatable; needs the challenge:write scope)")
    ap.add_argument("--time", type=int, default=180, help="initial seconds for --challenge")
    ap.add_argument("--inc", type=int, default=2, help="increment seconds for --challenge")
    ap.add_argument("--rated", action="store_true", help="make --challenge games rated")
    ap.add_argument("--check", action="store_true", help="verify token/account and exit")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    cfg = DEFAULT_CONFIG
    if os.path.exists(a.config):
        cfg = merge(DEFAULT_CONFIG, tomllib.load(open(a.config, "rb")))
    if os.environ.get("LICHESS_URL"):
        cfg["url"] = os.environ["LICHESS_URL"]
    token = os.environ.get(cfg["token_env"])
    if not token:
        sys.exit(f"Set ${cfg['token_env']} to a Lichess API token with the bot:play scope "
                 f"(https://lichess.org/account/oauth/token/create?scopes[]=bot:play).")
    api = LichessAPI(token, cfg["url"])
    me = api.account()
    is_bot = me.get("title") == "BOT"
    log.info("logged in as %s (title=%s)", me.get("username"), me.get("title"))
    if not is_bot:
        if a.upgrade:
            code, text = api.upgrade()
            log.info("upgrade -> HTTP %s %s", code, text[:200])
            if code != 200:
                sys.exit("upgrade failed (account must have played zero games)")
            me = api.account()
        else:
            sys.exit("This account is not a BOT account. Re-run with --upgrade to convert it "
                     "(irreversible, and the account must have played no games).")
    if a.check:
        return
    bot = Bot(cfg, api, me)
    for u in a.challenge:
        r = api.challenge(u, a.time, a.inc, a.rated)
        log.info("challenge %s -> HTTP %s %s", u, r.status_code if r is not None else "-", (r.text[:150] if r is not None else ""))
    signal.signal(signal.SIGINT, lambda *_: bot.stop.set())
    signal.signal(signal.SIGTERM, lambda *_: bot.stop.set())
    bot.run()


if __name__ == "__main__":
    main()
