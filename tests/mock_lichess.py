#!/usr/bin/env python3
"""Minimal in-process mock of the Lichess Bot API (the subset lichess_bot.py uses) for offline testing.

Scenario: a human-ish opponent ("randomguy") challenges the bot twice:
  1. a rated 3+0 blitz challenge  -> accepted, bot plays White vs. a random-move opponent
  2. a variant (chess960) challenge -> must be declined with reason "variant"
The game is capped at MAX_PLIES plies and then ends as a draw (status "draw") so the test is bounded.
The mock verifies that every move the bot posts is legal and that it is posted only on its turn.
"""
import json, queue, random, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import chess

MAX_PLIES = int(sys.argv[2]) if len(sys.argv) > 2 else 24
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
TOKEN = "mock-token"
RESULT = {"declines": [], "bot_moves": 0, "illegal": 0, "chat": [], "status": None, "accepted": []}
LOCK = threading.Lock()

events = queue.Queue()
games = {}


class G:
    def __init__(self, gid):
        self.id = gid
        self.board = chess.Board()
        self.moves = []
        self.wtime = self.btime = 60000
        self.inc = 0
        self.status = "started"
        self.cv = threading.Condition()
        self.last = time.time()
        self.rng = random.Random(7)

    def state(self):
        return {"type": "gameState", "moves": " ".join(self.moves), "wtime": self.wtime, "btime": self.btime,
                "winc": 0, "binc": 0, "status": self.status,
                **({"winner": None} if False else {})}

    def push(self, uci, who):
        with self.cv:
            now = time.time()
            dt = int((now - self.last) * 1000)
            self.last = now
            if self.board.turn:
                self.wtime -= dt
            else:
                self.btime -= dt
            self.board.push_uci(uci)
            self.moves.append(uci)
            if self.board.is_game_over():
                self.status = "mate" if self.board.is_checkmate() else "draw"
                RESULT["status"] = self.status
            elif len(self.moves) >= MAX_PLIES:
                self.status = "draw"
                RESULT["status"] = "draw(cap)"
            self.cv.notify_all()

    def opponent_loop(self):
        # opponent plays Black with random legal moves after a short delay
        while True:
            with self.cv:
                self.cv.wait_for(lambda: self.status != "started" or not self.board.turn, timeout=1)
                if self.status != "started":
                    return
                if self.board.turn:
                    continue
            time.sleep(0.15)
            with self.cv:
                if self.status != "started" or self.board.turn:
                    continue
                mv = self.rng.choice(list(self.board.legal_moves)).uci()
            self.push(mv, "opp")


def ndjson(h, items_iter):
    h.send_response(200)
    h.send_header("Content-Type", "application/x-ndjson")
    h.send_header("Transfer-Encoding", "chunked")  # like the real server
    h.end_headers()
    try:
        for it in items_iter:
            b = (json.dumps(it) + "\n").encode()
            h.wfile.write(b"%x\r\n" % len(b) + b + b"\r\n")
            h.wfile.flush()
    except (BrokenPipeError, ConnectionResetError):
        pass


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def auth_ok(self):
        return self.headers.get("Authorization") == f"Bearer {TOKEN}"

    def reply(self, code=200, body=None):
        data = json.dumps(body if body is not None else {"ok": True}).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if not self.auth_ok():
            return self.reply(401, {"error": "bad token"})
        p = self.path
        if p == "/api/account":
            return self.reply(200, {"id": "hereticbot", "username": "HereticBot", "title": "BOT",
                                    "perfs": {"blitz": {"rating": 2400}, "bullet": {"rating": 2400}}})
        if p == "/api/stream/event":
            def gen():
                yield from self._events()
            return ndjson(self, gen())
        if p.startswith("/api/bot/game/stream/"):
            gid = p.rsplit("/", 1)[1]
            g = games[gid]

            def gen():
                yield {"type": "gameFull", "id": gid, "variant": {"key": "standard"}, "speed": "blitz", "rated": True,
                       "initialFen": "startpos",
                       "white": {"id": "hereticbot", "name": "HereticBot", "title": "BOT", "rating": 2400},
                       "black": {"id": "randomguy", "name": "randomguy", "rating": 1500},
                       "state": g.state()}
                seen = len(g.moves)
                while True:
                    with g.cv:
                        g.cv.wait_for(lambda: len(g.moves) != seen or g.status != "started", timeout=5)
                        st = g.state()
                        seen = len(g.moves)
                    yield st
                    if st["status"] != "started":
                        return
            return ndjson(self, gen())
        self.reply(404, {"error": "not found"})

    def _events(self):
        yield {"type": "challenge", "challenge": {
            "id": "ch1", "status": "created", "challenger": {"id": "randomguy", "name": "randomguy", "rating": 1500},
            "variant": {"key": "standard"}, "rated": True, "speed": "blitz",
            "timeControl": {"type": "clock", "limit": 30, "increment": 0}}}
        yield {"type": "challenge", "challenge": {
            "id": "ch2", "status": "created", "challenger": {"id": "randomguy", "name": "randomguy", "rating": 1500},
            "variant": {"key": "chess960"}, "rated": False, "speed": "blitz",
            "timeControl": {"type": "clock", "limit": 30, "increment": 0}}}
        while True:
            try:
                ev = events.get(timeout=2)
            except queue.Empty:
                yield {"type": "challengeCanceled", "challenge": {"id": "none"}}  # harmless keep-alive-ish event
                continue
            yield ev

    def do_POST(self):
        if not self.auth_ok():
            return self.reply(401, {"error": "bad token"})
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n).decode() if n else ""
        p = self.path.split("?")[0]
        parts = p.strip("/").split("/")
        if parts[:2] == ["api", "challenge"] and parts[3] == "accept":
            RESULT["accepted"].append(parts[2])
            gid = "game0001"
            g = games[gid] = G(gid)
            threading.Thread(target=g.opponent_loop, daemon=True).start()
            events.put({"type": "gameStart", "game": {"id": gid}})
            return self.reply()
        if parts[:2] == ["api", "challenge"] and parts[3] == "decline":
            RESULT["declines"].append((parts[2], body))
            return self.reply()
        if parts[:3] == ["api", "bot", "game"] and len(parts) >= 5 and parts[4] == "move":
            g = games[parts[3]]
            uci = parts[5]
            with g.cv:
                legal = chess.Move.from_uci(uci) in g.board.legal_moves
                if not g.board.turn or not legal or g.status != "started":
                    RESULT["illegal"] += 1
                    return self.reply(400, {"error": "illegal or not your turn"})
            RESULT["bot_moves"] += 1
            g.push(uci, "bot")
            return self.reply()
        if parts[:3] == ["api", "bot", "game"] and parts[4] == "chat":
            RESULT["chat"].append(body)
            return self.reply()
        return self.reply()

    def do_PUT(self):
        self.do_POST()


def main():
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
    srv.daemon_threads = True
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    print(f"mock lichess on :{PORT}", flush=True)
    while RESULT["status"] is None:
        time.sleep(0.5)
    time.sleep(2)  # let the bot send goodbye chat
    print("RESULT", json.dumps(RESULT), flush=True)
    ok = RESULT["bot_moves"] >= MAX_PLIES // 2 - 1 and RESULT["illegal"] == 0 and \
        any(d[0] == "ch2" and "variant" in d[1] for d in RESULT["declines"]) and RESULT["accepted"] == ["ch1"]
    print("MOCK TEST", "PASS" if ok else "FAIL", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
