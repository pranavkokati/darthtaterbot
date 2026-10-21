// Heretic - UCI front-end
#include <chrono>
#include <iostream>
#include <sstream>

#include "movegen.h"
#include "nnue.h"
#include "search.h"
#include "tools.h"

using namespace heretic;

namespace {

const char* ENGINE_NAME = "Heretic 1.0";
const char* ENGINE_AUTHOR = "Pranav Kokati";

const char* BENCH_FENS[] = {
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "rnbqkb1r/pp2pppp/3p1n2/8/3NP3/8/PPP2PPP/RNBQKB1R w KQkq - 0 5",
    "r2q1rk1/ppp2ppp/2n1bn2/2bpp3/4P3/2PP1N2/PP1NBPPP/R1BQ1RK1 w - - 0 9",
    "8/8/4k3/8/2p5/8/B2P2K1/8 w - - 0 1",
    "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1",
    "2r3k1/pp3ppp/2n1p3/3p4/3P4/2P1PN2/P4PPP/2R3K1 w - - 0 1",
    "r1b2rk1/pp1nqppp/2p1pn2/3p4/2PP4/2NBPN2/PP3PPP/R2Q1RK1 w - - 0 10",
    "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
    "r5k1/5ppp/p7/1p6/8/P4P2/1P3KPP/3R4 w - - 0 1",
    "rnbqkbnr/ppp2ppp/8/3pp3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq d6 0 3",
};

struct Uci {
  Engine eng;
  Uci() {
    bitboards_init();
    Zobrist::init();
    eval_init();
    eng.set_position(Position::StartFEN, {});
  }

  void go(std::istringstream& is) {
    SearchLimits lim;
    std::string tok;
    while (is >> tok) {
      if (tok == "wtime") is >> lim.time[WHITE];
      else if (tok == "btime") is >> lim.time[BLACK];
      else if (tok == "winc") is >> lim.inc[WHITE];
      else if (tok == "binc") is >> lim.inc[BLACK];
      else if (tok == "movestogo") is >> lim.movestogo;
      else if (tok == "depth") is >> lim.depth;
      else if (tok == "nodes") is >> lim.nodes;
      else if (tok == "movetime") is >> lim.movetime;
      else if (tok == "mate") is >> lim.mate;
      else if (tok == "infinite") lim.infinite = true;
      else if (tok == "ponder") lim.infinite = true;
      else if (tok == "searchmoves") {
        std::string m;
        while (is >> m) {
          Move mv = parse_move(eng.pos, m);
          if (mv != MOVE_NONE) lim.searchmoves.push_back(mv);
        }
      }
    }
    eng.go(lim);
  }

  void position(std::istringstream& is) {
    std::string tok, fen;
    is >> tok;
    if (tok == "startpos") {
      fen = Position::StartFEN;
      is >> tok;  // "moves" or nothing
    } else if (tok == "fen") {
      while (is >> tok && tok != "moves") fen += tok + " ";
    }
    std::vector<std::string> moves;
    while (is >> tok) moves.push_back(tok);
    eng.set_position(fen, moves);
  }

  void setoption(std::istringstream& is) {
    std::string tok, name, value;
    is >> tok;  // "name"
    while (is >> tok && tok != "value") name += (name.empty() ? "" : " ") + tok;
    while (is >> tok) value += (value.empty() ? "" : " ") + tok;
    auto toInt = [&](int def) { try { return std::stoi(value); } catch (...) { return def; } };
    auto toBool = [&]() { return value == "true" || value == "1"; };
    EngineOptions& o = eng.opt;
    if (name == "Hash") eng.set_hash(std::max(1, toInt(64)));
    else if (name == "Threads") o.threads = std::max(1, std::min(64, toInt(1)));
    else if (name == "MultiPV") o.multiPV = std::max(1, std::min(64, toInt(1)));
    else if (name == "Contempt") o.contempt = toInt(0);
    else if (name == "Move Overhead") o.moveOverhead = toInt(60);
    else if (name == "TimeMtgBase") o.timeMtgBase = std::max(5, std::min(200, toInt(45)));
    else if (name == "TimeMtgDecay") o.timeMtgDecay = std::max(0, std::min(200, toInt(50)));
    else if (name == "TimeMtgFloor") o.timeMtgFloor = std::max(2, std::min(100, toInt(16)));
    else if (name == "TimeIncPct") o.timeIncPct = std::max(0, std::min(200, toInt(80)));
    else if (name == "TimeMaxPct") o.timeMaxPct = std::max(5, std::min(90, toInt(30)));
    else if (name == "QuantalSwindle") o.quantalSwindle = toBool();
    else if (name == "OpponentElo") o.opponentElo = toInt(0);
    else if (name == "SwindleMaxLoss") o.swindleMaxLoss = toInt(60);
    else if (name == "SwindleMinGain") o.swindleMinGain = toInt(60);
    else if (name == "OpponentTime") o.oppTimeMs = toInt(0);
    else if (name == "VarietyPlies") o.varietyPlies = toInt(0);
    else if (name == "VarietyMargin") o.varietyMargin = toInt(15);
    else if (name == "SearchMode") o.searchMode = toInt(0);
    else if (name == "NNUEFile") { if (!value.empty() && value != "<empty>") nnue::load(value); else nnue::unload(); }
    else if (name == "EvalParams") { if (!value.empty() && value != "<empty>") eval_load_params(value); }
  }

  void bench(int depth) {
    Engine e;
    e.silent = true;
    e.opt.hash = 64;
    uint64_t total = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (const char* fen : BENCH_FENS) {
      e.new_game();
      e.set_position(fen, {});
      SearchLimits lim;
      lim.depth = depth;
      SearchResult r = e.search_blocking(lim);
      total += r.nodes;
      printf("%-72s best %s score %d nodes %llu\n", fen, move_str(r.best).c_str(), r.score, (unsigned long long)r.nodes);
    }
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("===========================\nTotal nodes: %llu\nTime: %.2fs\nNodes/second: %.0f\n", (unsigned long long)total, s,
           total / s);
  }

  void loop() {
    std::string line, cmd;
    while (std::getline(std::cin, line)) {
      std::istringstream is(line);
      cmd.clear();
      is >> cmd;
      if (cmd == "uci") {
        printf("id name %s\nid author %s\n", ENGINE_NAME, ENGINE_AUTHOR);
        printf("option name Hash type spin default 64 min 1 max 65536\n");
        printf("option name Threads type spin default 1 min 1 max 64\n");
        printf("option name MultiPV type spin default 1 min 1 max 64\n");
        printf("option name Contempt type spin default 0 min -200 max 200\n");
        printf("option name Move Overhead type spin default 60 min 0 max 5000\n");
        printf("option name TimeMtgBase type spin default 45 min 5 max 200\n");
        printf("option name TimeMtgDecay type spin default 50 min 0 max 200\n");
        printf("option name TimeMtgFloor type spin default 16 min 2 max 100\n");
        printf("option name TimeIncPct type spin default 80 min 0 max 200\n");
        printf("option name TimeMaxPct type spin default 30 min 5 max 90\n");
        printf("option name QuantalSwindle type check default false\n");
        printf("option name OpponentElo type spin default 0 min 0 max 3500\n");
        printf("option name OpponentTime type spin default 0 min 0 max 36000000\n");
        printf("option name VarietyPlies type spin default 0 min 0 max 40\n");
        printf("option name VarietyMargin type spin default 15 min 0 max 200\n");
        printf("option name SwindleMaxLoss type spin default 30 min 0 max 300\n");
        printf("option name SwindleMinGain type spin default 30 min 0 max 1000\n");
        printf("option name SearchMode type spin default 0 min 0 max 1\n");
        printf("option name NNUEFile type string default <empty>\n");
        printf("option name EvalParams type string default <empty>\n");
        printf("uciok\n");
        fflush(stdout);
      } else if (cmd == "isready") {
        eng.wait();
        printf("readyok\n");
        fflush(stdout);
      } else if (cmd == "ucinewgame") eng.new_game();
      else if (cmd == "position") position(is);
      else if (cmd == "go") go(is);
      else if (cmd == "stop") eng.stop();
      else if (cmd == "setoption") setoption(is);
      else if (cmd == "quit") { eng.stop(); eng.wait(); break; }
      else if (cmd == "d") {
        printf("FEN: %s\nKey: %016llx\nEval (stm): %d\n", eng.pos.fen().c_str(), (unsigned long long)eng.pos.key(), evaluate(eng.pos));
        fflush(stdout);
      } else if (cmd == "eval") {
        printf("hce %d  final %d\n", evaluate_hce(eng.pos), evaluate(eng.pos));
        fflush(stdout);
      } else if (cmd == "perft") {
        int d = 1; is >> d;
        printf("%llu\n", (unsigned long long)perft(eng.pos, d));
        fflush(stdout);
      } else if (cmd == "bench") {
        int d = 10; is >> d;
        bench(d);
        fflush(stdout);
      } else if (!cmd.empty()) {
        if (!tools_command(cmd, is, eng)) { /* unknown command: ignore per UCI */ }
        fflush(stdout);
      }
    }
    eng.wait();  // EOF on stdin: let a running search finish and print its bestmove
  }
};

}  // namespace

int main(int argc, char** argv) {
  std::ios::sync_with_stdio(true);
  setvbuf(stdout, nullptr, _IOLBF, 0);
  Uci u;
  if (argc > 1 && std::string(argv[1]) == "bench") {
    u.bench(argc > 2 ? atoi(argv[2]) : 10);
    return 0;
  }
  u.loop();
  return 0;
}
