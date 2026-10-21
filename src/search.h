// Alpha-beta search engine (PVS, Lazy SMP) + engine front-end
#pragma once
#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "eval.h"
#include "movegen.h"
#include "position.h"
#include "tt.h"

namespace heretic {

typedef int64_t TimePoint;
TimePoint now_ms();

struct SearchLimits {
  int64_t time[2] = {0, 0};
  int64_t inc[2] = {0, 0};
  int movestogo = 0;
  int depth = 0;
  uint64_t nodes = 0;
  int64_t movetime = 0;
  bool infinite = false;
  int mate = 0;
  std::vector<Move> searchmoves;
};

struct EngineOptions {
  int hash = 64;
  int threads = 1;
  int multiPV = 1;
  int contempt = 0;       // centipawns; positive = avoid draws
  int moveOverhead = 60;  // ms (the Lichess bridge overwrites this every move with the *measured* round-trip lag)
  // Time-management knobs (all UCI options; defaults are calibrated by scripts/tune_time.py self-play, not guessed)
  int timeMtgBase = 45;     // expected moves to go at the start of the game
  int timeMtgDecay = 50;    // moves-to-go lost per 100 full moves played (hundredths of a move per move)
  int timeMtgFloor = 16;    // never plan for fewer moves than this
  int timeIncPct = 80;      // % of the increment spent per move
  int timeMaxPct = 30;      // hard cap on one move, % of usable clock
  // ---- Heretic quirks (see docs/ALGORITHMS.md)
  bool quantalSwindle = false;   // bounded-rational opponent model at the root
  int opponentElo = 0;           // 0 = unknown; drives swindle temperature & dynamic contempt
  int swindleMaxLoss = 30;       // cp the swindle may give up versus the best move
  int swindleMinGain = 30;       // required expected-score gain, in per-mille of win probability
  int oppTimeMs = 0;             // opponent clock (ms) at the start of the move; 0 = unknown
  int varietyPlies = 0;          // opening variety: sample near-best root moves for the first N plies
  int varietyMargin = 15;        // cp
  int searchMode = 0;            // 0 = alpha-beta, 1 = MCTS (PUCT + Gumbel root)
};

struct RootMove {
  Move move = MOVE_NONE;
  Value score = -VALUE_INFINITE;
  Value prevScore = -VALUE_INFINITE;
  int selDepth = 0;
  std::vector<Move> pv;
  RootMove() {}
  explicit RootMove(Move m) : move(m), pv(1, m) {}
};

struct SearchResult {
  Move best = MOVE_NONE;
  Move ponder = MOVE_NONE;
  Value score = 0;
  int depth = 0;
  uint64_t nodes = 0;
  std::vector<RootMove> rootMoves;  // final root moves with scores (MultiPV)
};

class Engine;

class Worker {
 public:
  Worker(Engine& e, int id);
  void iterate(const Position& root, const SearchLimits& lim);
  void mcts(const Position& root, const SearchLimits& lim);  // PUCT tree search with Gumbel root (SearchMode 1)
  bool prepare(const Position& root, const SearchLimits& lim);
  uint64_t node_count() const { return nodes.load(std::memory_order_relaxed); }

  Engine& eng;
  int id;
  Position pos;
  std::vector<RootMove> rootMoves;
  std::atomic<uint64_t> nodes;
  int selDepth;
  int rootDepth;
  int completedDepth;
  int pvIdx;
  Color rootColor;
  int rootPly;
  bool stopped;

  // ---- history tables
  struct PieceToHist { int16_t v[PIECE_NB][64]; };
  int16_t mainHist[2][4096];
  int16_t capHist[PIECE_NB][64][PT_NB];
  PieceToHist* contHist;  // [PIECE_NB*64]
  Move counterMoves[PIECE_NB][64];
  int32_t pawnCorr[2][16384];
  int32_t nonPawnCorr[2][2][16384];
  void clear_histories();

  struct Stack {
    Move pv[MAX_PLY + 2];
    int ply;
    Move currentMove;
    Move excludedMove;
    Move killers[2];
    Value staticEval;
    int statScore;
    int moveCount;
    bool inCheck;
    bool ttPv;
    PieceToHist* cont;
  };

  enum NodeType { NonPV, PV, Root };
  template <NodeType NT> Value search(Stack* ss, Value alpha, Value beta, int depth, bool cutNode);
  template <NodeType NT> Value qsearch(Stack* ss, Value alpha, Value beta);

  Value draw_value(int ply) const;
  Value corrected_eval(Value raw) const;
  void update_correction(Value bestValue, Value staticEval, int depth);
  void update_quiet_stats(Stack* ss, Move best, Move* quiets, int nq, int depth);
  void update_capture_stats(Move best, Move* caps, int nc, int depth);
  void score_moves(const Stack* ss, ExtMove* list, int n, Move ttMove, bool qs);
  int quiet_score(const Stack* ss, Move m) const;
  void check_time();
  void print_info(int depth, int pvIndex, Value score, int bound) const;

 private:
  std::unique_ptr<PieceToHist[]> contStorage;
  Stack stackStorage[MAX_PLY + 10];
  int nmpMinPly;
  SearchLimits limits;
  TimePoint startTime;
};

class Engine {
 public:
  Engine();
  ~Engine();

  void new_game();
  void set_position(const std::string& fen, const std::vector<std::string>& moves);
  void go(const SearchLimits& lim);       // async
  SearchResult search_blocking(const SearchLimits& lim);
  void stop();
  void wait();
  void set_hash(int mb);
  bool searching() const { return running.load(); }

  EngineOptions opt;
  TranspositionTable tt;
  Position pos;
  std::atomic<bool> stopFlag;
  std::function<void(const std::string&)> out;  // sink for UCI output (info/bestmove)
  bool silent = false;

  // time management (set by go)
  TimePoint startTime = 0, optimumTime = 0, maximumTime = 0;
  bool timeManaged = false;
  int effMultiPV = 1;  // internal MultiPV actually used by workers (raised when root-policy quirks are active)
  uint64_t nodeLimit = 0;
  int depthLimit = 0;
  Color rootColor = WHITE;
  int rootGamePly = 0;
  std::vector<Move> gameMoves;  // moves from the start position (for repetition / book)
  std::string startFen = Position::StartFEN;

 private:
  void run_search(SearchLimits lim);
  void apply_root_policy(SearchResult& res);
  std::unique_ptr<Engine> probe;  // reply-probe engine for the quantal-response swindle
  std::thread mainThread;
  std::atomic<bool> running;
  std::vector<std::unique_ptr<Worker>> workers;
  SearchResult lastResult;
  friend class Worker;
 public:
  const SearchResult& last_result() const { return lastResult; }
};

void search_init();
std::string pv_string(const std::vector<Move>& pv);

}  // namespace heretic
