#include "search.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <sstream>

namespace heretic {

TimePoint now_ms() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

namespace {

int LMR[64][64];

inline Value value_to_tt(Value v, int ply) {
  return v >= VALUE_MATE_IN_MAX_PLY ? v + ply : v <= -VALUE_MATE_IN_MAX_PLY ? v - ply : v;
}
inline Value value_from_tt(Value v, int ply, int r50) {
  if (v == VALUE_NONE) return VALUE_NONE;
  if (v >= VALUE_MATE_IN_MAX_PLY) return (VALUE_MATE - v > 100 - r50) ? VALUE_MATE_IN_MAX_PLY - 1 : v - ply;
  if (v <= -VALUE_MATE_IN_MAX_PLY) return (VALUE_MATE + v > 100 - r50) ? -VALUE_MATE_IN_MAX_PLY + 1 : v + ply;
  return v;
}
inline void gravity(int16_t& h, int bonus) {
  int b = std::max(-16384, std::min(16384, bonus));
  h = int16_t(h + b - h * std::abs(b) / 16384);
}
constexpr int CORR_SCALE = 64;
constexpr int CORR_LIMIT = 256 * CORR_SCALE;

inline void update_pv(Move* pv, Move m, const Move* child) {
  *pv++ = m;
  while (child && *child != MOVE_NONE) *pv++ = *child++;
  *pv = MOVE_NONE;
}

}  // namespace

void search_init() {
  for (int d = 1; d < 64; ++d)
    for (int m = 1; m < 64; ++m) LMR[d][m] = int(0.75 + std::log(double(d)) * std::log(double(m)) / 2.25);
}

std::string pv_string(const std::vector<Move>& pv) {
  std::string s;
  for (Move m : pv) {
    if (!s.empty()) s += ' ';
    s += move_str(m);
  }
  return s;
}

// ======================================================================== Worker

Worker::Worker(Engine& e, int id_) : eng(e), id(id_), nodes(0), selDepth(0), rootDepth(0), completedDepth(0), pvIdx(0),
    rootColor(WHITE), rootPly(0), stopped(false), nmpMinPly(0), startTime(0) {
  contStorage.reset(new PieceToHist[PIECE_NB * 64]);
  contHist = contStorage.get();
  clear_histories();
}

void Worker::clear_histories() {
  std::memset(mainHist, 0, sizeof(mainHist));
  std::memset(capHist, 0, sizeof(capHist));
  std::memset(contHist, 0, sizeof(PieceToHist) * PIECE_NB * 64);
  std::memset(counterMoves, 0, sizeof(counterMoves));
  std::memset(pawnCorr, 0, sizeof(pawnCorr));
  std::memset(nonPawnCorr, 0, sizeof(nonPawnCorr));
}

Value Worker::draw_value(int ply) const {
  // Contempt: from the engine's point of view a draw is worth -contempt (so it avoids draws when > 0).
  // The +-1 node-parity jitter prevents blind three-fold repetition loops (as in Stockfish).
  int c = eng.opt.contempt;
  Value v = (ply & 1) ? c : -c;
  return v + ((nodes.load(std::memory_order_relaxed) & 2) ? 1 : -1);
}

Value Worker::corrected_eval(Value raw) const {
  Color us = pos.side_to_move();
  int pc = pawnCorr[us][pos.pawn_key() & 16383];
  int nw = nonPawnCorr[us][WHITE][pos.non_pawn_key(WHITE) & 16383];
  int nb = nonPawnCorr[us][BLACK][pos.non_pawn_key(BLACK) & 16383];
  int corr = (2 * pc + nw + nb) / (4 * CORR_SCALE);
  int v = raw + corr;
  return std::max(-VALUE_MATE_IN_MAX_PLY + 1, std::min(VALUE_MATE_IN_MAX_PLY - 1, v));
}

void Worker::update_correction(Value bestValue, Value staticEval, int depth) {
  Color us = pos.side_to_move();
  int diff = std::max(-128, std::min(128, bestValue - staticEval)) * CORR_SCALE;
  int w = std::min(2 * depth, 32);
  auto upd = [&](int32_t& e) {
    e += (diff - e) * w / 256;
    e = std::max(-CORR_LIMIT, std::min(CORR_LIMIT, e));
  };
  upd(pawnCorr[us][pos.pawn_key() & 16383]);
  upd(nonPawnCorr[us][WHITE][pos.non_pawn_key(WHITE) & 16383]);
  upd(nonPawnCorr[us][BLACK][pos.non_pawn_key(BLACK) & 16383]);
}

int Worker::quiet_score(const Stack* ss, Move m) const {
  Color us = pos.side_to_move();
  Piece pc = pos.piece_on(from_sq(m));
  Square to = to_sq(m);
  int s = 2 * mainHist[us][from_sq(m) * 64 + to];
  s += 2 * (*(ss - 1)->cont).v[pc][to];
  s += (*(ss - 2)->cont).v[pc][to];
  s += (*(ss - 4)->cont).v[pc][to] / 2;
  return s;
}

void Worker::score_moves(const Stack* ss, ExtMove* list, int n, Move ttMove, bool qs) {
  Move counter = MOVE_NONE;
  Move prev = (ss - 1)->currentMove;
  if (prev != MOVE_NONE) {
    Piece pp = pos.piece_on(to_sq(prev));
    counter = counterMoves[pp][to_sq(prev)];
  }
  for (int i = 0; i < n; ++i) {
    Move m = list[i].move;
    if (m == ttMove) {
      list[i].score = 1 << 30;
      continue;
    }
    Piece pc = pos.piece_on(from_sq(m));
    if (is_noisy(m)) {
      int f = flags_of(m);
      PieceType victim = (f == MF_EP) ? PAWN : type_of(pos.piece_on(to_sq(m)));
      int mvv = SeeValue[victim] * 16 - int(type_of(pc));
      if (is_promo(m)) {
        list[i].score = (promo_type(m) == QUEEN ? (1 << 26) : -(1 << 26)) + mvv;
      } else {
        bool good = pos.see_ge(m, 0);
        list[i].score = (good ? (1 << 27) : -(1 << 27)) + mvv + capHist[pc][to_sq(m)][victim] / 16;
      }
    } else {
      int s = quiet_score(ss, m);
      if (!qs) {
        if (m == ss->killers[0]) s += 1 << 25;
        else if (m == ss->killers[1]) s += 1 << 24;
        else if (m == counter) s += 1 << 23;
      }
      list[i].score = s;
    }
  }
}

void Worker::update_quiet_stats(Stack* ss, Move best, Move* quiets, int nq, int depth) {
  int bonus = std::min(300 * depth - 250, 2400);
  if (ss->killers[0] != best) {
    ss->killers[1] = ss->killers[0];
    ss->killers[0] = best;
  }
  Color us = pos.side_to_move();
  auto upd = [&](Move m, int b) {
    Piece pc = pos.piece_on(from_sq(m));
    Square to = to_sq(m);
    gravity(mainHist[us][from_sq(m) * 64 + to], b);
    gravity((*(ss - 1)->cont).v[pc][to], b);
    gravity((*(ss - 2)->cont).v[pc][to], b / 2 * 2);
    gravity((*(ss - 4)->cont).v[pc][to], b / 2);
  };
  upd(best, bonus);
  Move prev = (ss - 1)->currentMove;
  if (prev != MOVE_NONE) counterMoves[pos.piece_on(to_sq(prev))][to_sq(prev)] = best;
  for (int i = 0; i < nq; ++i)
    if (quiets[i] != best) upd(quiets[i], -bonus);
}

void Worker::update_capture_stats(Move best, Move* caps, int nc, int depth) {
  int bonus = std::min(300 * depth - 250, 2400);
  auto upd = [&](Move m, int b) {
    Piece pc = pos.piece_on(from_sq(m));
    PieceType victim = flags_of(m) == MF_EP ? PAWN : type_of(pos.piece_on(to_sq(m)));
    gravity(capHist[pc][to_sq(m)][victim], b);
  };
  if (best != MOVE_NONE && is_capture(best)) upd(best, bonus);
  for (int i = 0; i < nc; ++i)
    if (caps[i] != best) upd(caps[i], -bonus);
}

void Worker::check_time() {
  if (id != 0) return;
  if ((nodes.load(std::memory_order_relaxed) & 1023) != 0) return;
  if (eng.nodeLimit && nodes.load(std::memory_order_relaxed) >= eng.nodeLimit) {
    eng.stopFlag = true;
    return;
  }
  if (eng.timeManaged && now_ms() - startTime >= eng.maximumTime) eng.stopFlag = true;
}

void Worker::print_info(int depth, int pvIndex, Value score, int bound) const {
  if (eng.silent || !eng.out) return;
  const RootMove& rm = rootMoves[pvIndex];
  TimePoint elapsed = std::max<TimePoint>(1, now_ms() - startTime);
  uint64_t total = 0;
  for (auto& w : eng.workers) total += w->node_count();
  std::ostringstream ss;
  ss << "info depth " << depth << " seldepth " << rm.selDepth << " multipv " << (pvIndex + 1) << " score ";
  if (std::abs(score) >= VALUE_MATE_IN_MAX_PLY) {
    int mv = (VALUE_MATE - std::abs(score) + 1) / 2;
    ss << "mate " << (score > 0 ? mv : -mv);
  } else
    ss << "cp " << score;
  if (bound == BOUND_LOWER) ss << " lowerbound";
  if (bound == BOUND_UPPER) ss << " upperbound";
  ss << " nodes " << total << " nps " << (total * 1000 / elapsed) << " hashfull " << eng.tt.hashfull() << " time "
     << elapsed << " pv " << pv_string(rm.pv);
  eng.out(ss.str());
}

// ---------------------------------------------------------------------- qsearch
template <Worker::NodeType NT>
Value Worker::qsearch(Stack* ss, Value alpha, Value beta) {
  constexpr bool PvNode = NT == PV;
  nodes.fetch_add(1, std::memory_order_relaxed);
  check_time();
  if (PvNode) {
    ss->pv[0] = MOVE_NONE;
    if (selDepth < ss->ply + 1) selDepth = ss->ply + 1;
  }
  ss->inCheck = pos.in_check();
  if (eng.stopFlag.load(std::memory_order_relaxed)) {
    stopped = true;
    return 0;
  }
  if (pos.is_draw(ss->ply)) {
    if (pos.rule50() > 99 && ss->inCheck) {
      MoveList ml(pos);
      if (ml.size() == 0) return mated_in(ss->ply);
    }
    return draw_value(ss->ply);
  }
  if (ss->ply >= MAX_PLY) return ss->inCheck ? draw_value(ss->ply) : corrected_eval(evaluate(pos));

  Key key = pos.key();
  TTData tte;
  bool ttHit = eng.tt.probe(key, tte);
  Value ttValue = ttHit ? value_from_tt(tte.value, ss->ply, pos.rule50()) : VALUE_NONE;
  Move ttMove = ttHit ? tte.move : MOVE_NONE;
  if (ttMove && !(pos.pseudo_legal(ttMove) && pos.legal(ttMove))) ttMove = MOVE_NONE;
  if (!PvNode && ttHit && tte.depth >= 0 && ttValue != VALUE_NONE &&
      (tte.bound & (ttValue >= beta ? BOUND_LOWER : BOUND_UPPER)))
    return ttValue;

  Value bestValue, futilityBase;
  Value rawEval = VALUE_NONE;
  if (ss->inCheck) {
    bestValue = -VALUE_INFINITE;
    futilityBase = -VALUE_INFINITE;
    ss->staticEval = VALUE_NONE;
  } else {
    rawEval = (ttHit && tte.eval != VALUE_NONE) ? tte.eval : evaluate(pos);
    ss->staticEval = bestValue = corrected_eval(rawEval);
    if (ttValue != VALUE_NONE && (tte.bound & (ttValue > bestValue ? BOUND_LOWER : BOUND_UPPER))) bestValue = ttValue;
    if (bestValue >= beta) {
      if (!ttHit) eng.tt.store(key, value_to_tt(bestValue, ss->ply), false, BOUND_LOWER, 0, MOVE_NONE, rawEval);
      return bestValue;
    }
    if (PvNode && bestValue > alpha) alpha = bestValue;
    futilityBase = ss->staticEval + 200;
  }

  ExtMove list[256];
  int n = generate(pos, list, GEN_NOISY);
  score_moves(ss, list, n, ttMove, true);
  Move bestMove = MOVE_NONE;
  int moveCount = 0;
  for (int i = 0; i < n; ++i) {
    ExtMove* best = &list[i];
    for (int j = i + 1; j < n; ++j)
      if (list[j].score > best->score) best = &list[j];
    std::swap(list[i], *best);
    Move move = list[i].move;
    ++moveCount;
    if (!ss->inCheck) {
      // skip clearly losing captures and hopeless ones
      if (list[i].score < 0 && !(is_promo(move))) continue;
      if (!is_promo(move) && futilityBase > -VALUE_MATE_IN_MAX_PLY) {
        PieceType victim = flags_of(move) == MF_EP ? PAWN : type_of(pos.piece_on(to_sq(move)));
        if (futilityBase + SeeValue[victim] <= alpha) {
          bestValue = std::max(bestValue, futilityBase + SeeValue[victim]);
          continue;
        }
      }
    }
    Piece pc = pos.piece_on(from_sq(move));
    ss->currentMove = move;
    ss->cont = &contHist[pc * 64 + to_sq(move)];
    eng.tt.prefetch(pos.key_after(move));
    pos.do_move(move);
    Value value = -qsearch<NT>(ss + 1, -beta, -alpha);
    pos.undo_move(move);
    if (stopped || eng.stopFlag.load(std::memory_order_relaxed)) return 0;
    if (value > bestValue) {
      bestValue = value;
      if (value > alpha) {
        bestMove = move;
        if (PvNode) update_pv(ss->pv, move, (ss + 1)->pv);
        if (value >= beta) break;
        alpha = value;
      }
    }
  }
  if (ss->inCheck && bestValue == -VALUE_INFINITE) return mated_in(ss->ply);
  int bound = bestValue >= beta ? BOUND_LOWER : (PvNode && bestMove) ? BOUND_EXACT : BOUND_UPPER;
  eng.tt.store(key, value_to_tt(bestValue, ss->ply), false, bound, 0, bestMove, rawEval);
  return bestValue;
}

// ----------------------------------------------------------------------- search
template <Worker::NodeType NT>
Value Worker::search(Stack* ss, Value alpha, Value beta, int depth, bool cutNode) {
  constexpr bool PvNode = NT != NonPV;
  constexpr bool rootNode = NT == Root;

  if (depth <= 0) return qsearch<PvNode ? PV : NonPV>(ss, alpha, beta);

  Color us = pos.side_to_move();
  nodes.fetch_add(1, std::memory_order_relaxed);
  check_time();
  ss->inCheck = pos.in_check();
  bool excluded = ss->excludedMove != MOVE_NONE;
  if (PvNode) {
    ss->pv[0] = MOVE_NONE;
    if (selDepth < ss->ply + 1) selDepth = ss->ply + 1;
  }
  if (!rootNode) {
    if (eng.stopFlag.load(std::memory_order_relaxed)) {
      stopped = true;
      return 0;
    }
    if (pos.is_draw(ss->ply)) {
      if (pos.rule50() > 99 && ss->inCheck) {
        MoveList ml(pos);
        if (ml.size() == 0) return mated_in(ss->ply);
      }
      return draw_value(ss->ply);
    }
    if (ss->ply >= MAX_PLY) return ss->inCheck ? draw_value(ss->ply) : corrected_eval(evaluate(pos));
    alpha = std::max(mated_in(ss->ply), alpha);
    beta = std::min(mate_in(ss->ply + 1), beta);
    if (alpha >= beta) return alpha;
  }
  (ss + 1)->excludedMove = MOVE_NONE;
  (ss + 2)->killers[0] = (ss + 2)->killers[1] = MOVE_NONE;

  Key key = pos.key();
  if (excluded) key ^= Key(ss->excludedMove) * 0x9E3779B97F4A7C15ULL;
  TTData tte;
  bool ttHit = eng.tt.probe(key, tte);
  Value ttValue = ttHit ? value_from_tt(tte.value, ss->ply, pos.rule50()) : VALUE_NONE;
  Move ttMove = rootNode ? rootMoves[pvIdx].move : (ttHit ? tte.move : MOVE_NONE);
  if (ttMove && !rootNode && !(pos.pseudo_legal(ttMove) && pos.legal(ttMove))) ttMove = MOVE_NONE;
  bool ttCapture = ttMove && is_capture(ttMove);
  ss->ttPv = PvNode || (ttHit && tte.pv);

  if (!PvNode && !excluded && ttHit && tte.depth >= depth && ttValue != VALUE_NONE && pos.rule50() < 90 &&
      (tte.bound & (ttValue >= beta ? BOUND_LOWER : BOUND_UPPER)))
    return ttValue;

  Value eval = VALUE_NONE, rawEval = VALUE_NONE;
  bool improving = false;
  if (ss->inCheck) {
    ss->staticEval = VALUE_NONE;
  } else {
    rawEval = (ttHit && tte.eval != VALUE_NONE) ? tte.eval : evaluate(pos);
    eval = ss->staticEval = corrected_eval(rawEval);
    if (!ttHit && !excluded) eng.tt.store(key, VALUE_NONE, ss->ttPv, BOUND_NONE, -16, MOVE_NONE, rawEval);
    if (ttValue != VALUE_NONE && (tte.bound & (ttValue > eval ? BOUND_LOWER : BOUND_UPPER))) eval = ttValue;
    improving = (ss - 2)->staticEval != VALUE_NONE && ss->staticEval > (ss - 2)->staticEval;
  }

  if (!PvNode && !ss->inCheck && !excluded) {
    // Razoring
    if (depth <= 3 && eval < alpha - 350 * depth) {
      Value v = qsearch<NonPV>(ss, alpha - 1, alpha);
      if (v < alpha || stopped) return v;
    }
    // Reverse futility pruning
    if (depth <= 8 && eval - 90 * depth + 60 * improving >= beta && beta > -VALUE_MATE_IN_MAX_PLY &&
        eval < VALUE_MATE_IN_MAX_PLY)
      return (eval + beta) / 2;
    // Null-move pruning
    if ((ss - 1)->currentMove != MOVE_NONE && eval >= beta && ss->staticEval >= beta - 20 * depth + 200 && depth >= 3 &&
        pos.has_non_pawn_material(us) && ss->ply >= nmpMinPly && beta > -VALUE_MATE_IN_MAX_PLY) {
      int R = 3 + depth / 3 + std::min((eval - beta) / 200, 3);
      ss->currentMove = MOVE_NONE;
      ss->cont = &contHist[0];
      pos.do_null();
      Value v = -search<NonPV>(ss + 1, -beta, -beta + 1, depth - R, !cutNode);
      pos.undo_null();
      if (stopped) return 0;
      if (v >= beta && v < VALUE_MATE_IN_MAX_PLY) {
        if (nmpMinPly || depth < 14) return v;
        nmpMinPly = ss->ply + 3 * (depth - R) / 4;
        Value v2 = search<NonPV>(ss, beta - 1, beta, depth - R, false);
        nmpMinPly = 0;
        if (stopped) return 0;
        if (v2 >= beta) return v;
      }
    }
    // ProbCut
    Value probCutBeta = beta + 200 - 60 * improving;
    if (depth >= 5 && std::abs(beta) < VALUE_MATE_IN_MAX_PLY &&
        !(ttHit && tte.depth >= depth - 3 && ttValue != VALUE_NONE && ttValue < probCutBeta)) {
      ExtMove pl[256];
      int pn = generate(pos, pl, GEN_NOISY);
      score_moves(ss, pl, pn, ttMove, true);
      for (int i = 0; i < pn; ++i) {
        ExtMove* best = &pl[i];
        for (int j = i + 1; j < pn; ++j)
          if (pl[j].score > best->score) best = &pl[j];
        std::swap(pl[i], *best);
        Move m = pl[i].move;
        if (!is_noisy(m) || !pos.see_ge(m, probCutBeta - ss->staticEval)) continue;
        Piece pc = pos.piece_on(from_sq(m));
        ss->currentMove = m;
        ss->cont = &contHist[pc * 64 + to_sq(m)];
        pos.do_move(m);
        Value v = -qsearch<NonPV>(ss + 1, -probCutBeta, -probCutBeta + 1);
        if (v >= probCutBeta && !stopped) v = -search<NonPV>(ss + 1, -probCutBeta, -probCutBeta + 1, depth - 4, !cutNode);
        pos.undo_move(m);
        if (stopped) return 0;
        if (v >= probCutBeta) {
          eng.tt.store(key, value_to_tt(v, ss->ply), ss->ttPv, BOUND_LOWER, depth - 3, m, rawEval);
          return v - (probCutBeta - beta);
        }
      }
    }
  }

  // Internal iterative reduction
  if (!ttMove && depth >= 6 && (PvNode || cutNode) && !rootNode) --depth;

  ExtMove list[256];
  int n;
  if (rootNode) {
    n = int(rootMoves.size()) - pvIdx;
    for (int i = 0; i < n; ++i) {
      list[i].move = rootMoves[pvIdx + i].move;
      list[i].score = 1000000 - i;
    }
  } else {
    n = generate(pos, list, GEN_ALL);
    score_moves(ss, list, n, ttMove, false);
  }

  Move bestMove = MOVE_NONE;
  Value bestValue = -VALUE_INFINITE;
  int moveCount = 0;
  Move quietsTried[32], capsTried[32];
  int nq = 0, nc = 0;
  bool skipQuiets = false;
  const Value origAlpha = alpha;

  for (int i = 0; i < n; ++i) {
    if (!rootNode) {
      ExtMove* best = &list[i];
      for (int j = i + 1; j < n; ++j)
        if (list[j].score > best->score) best = &list[j];
      std::swap(list[i], *best);
    }
    Move move = list[i].move;
    if (move == ss->excludedMove) continue;
    ++moveCount;
    ss->moveCount = moveCount;
    bool capture = is_capture(move);
    bool noisy = is_noisy(move);
    Piece pc = pos.piece_on(from_sq(move));
    Square to = to_sq(move);
    bool givesCheck = pos.gives_check(move);
    int newDepth = depth - 1;
    int histScore = noisy ? 0 : quiet_score(ss, move);

    if (!rootNode && bestValue > -VALUE_MATE_IN_MAX_PLY && pos.has_non_pawn_material(us)) {
      if (skipQuiets && !noisy) continue;
      int lmrDepth = std::max(0, newDepth - LMR[std::min(depth, 63)][std::min(moveCount, 63)]);
      if (!noisy) {
        int lmpLimit = improving ? 3 + depth * depth : (3 + depth * depth) / 2;
        if (moveCount >= lmpLimit) {
          skipQuiets = true;
          continue;
        }
        if (!ss->inCheck && !givesCheck && lmrDepth <= 8 && ss->staticEval + 90 + 90 * lmrDepth <= alpha) continue;
        if (lmrDepth <= 5 && histScore < -3500 * depth) continue;
        if (!pos.see_ge(move, -25 * lmrDepth * lmrDepth)) continue;
      } else if (!givesCheck) {
        if (!pos.see_ge(move, -90 * depth)) continue;
      }
    }

    int extension = 0;
    if (!rootNode && !excluded && move == ttMove && depth >= 8 && ttValue != VALUE_NONE &&
        std::abs(ttValue) < VALUE_MATE_IN_MAX_PLY && (tte.bound & BOUND_LOWER) && tte.depth >= depth - 3) {
      Value sBeta = ttValue - (1 + (ss->ttPv && !PvNode)) * depth;
      int sDepth = (depth - 1) / 2;
      ss->excludedMove = move;
      Value v = search<NonPV>(ss, sBeta - 1, sBeta, sDepth, cutNode);
      ss->excludedMove = MOVE_NONE;
      if (stopped) return 0;
      if (v < sBeta) {
        extension = 1;
        if (!PvNode && v < sBeta - 25) extension = 2;
      } else if (sBeta >= beta)
        return sBeta;
      else if (ttValue >= beta)
        extension = -2;
      else if (cutNode)
        extension = -1;
    } else if (givesCheck && pos.see_ge(move, 0)) {
      extension = 1;
    }
    if (ss->ply > 2 * rootDepth + 6) extension = std::min(extension, 0);
    newDepth += extension;

    ss->currentMove = move;
    ss->cont = &contHist[pc * 64 + to];
    eng.tt.prefetch(pos.key_after(move));
    pos.do_move(move);

    Value value = -VALUE_INFINITE;
    if (depth >= 2 && moveCount > 1 + rootNode) {
      int r = LMR[std::min(depth, 63)][std::min(moveCount, 63)];
      if (PvNode) r -= 1;
      if (cutNode) r += 2;
      if (givesCheck) r -= 1;
      if (!improving) r += 1;
      if (ss->ttPv) r -= 1;
      if (ttCapture && !capture) r += 1;
      if (capture)
        r -= 1 + capHist[pc][to][type_of(pos.state().captured)] / 8192;
      else
        r -= histScore / 8000;
      int d = std::max(1, std::min(newDepth - r, newDepth + 1));
      value = -search<NonPV>(ss + 1, -(alpha + 1), -alpha, d, true);
      if (value > alpha && d < newDepth) value = -search<NonPV>(ss + 1, -(alpha + 1), -alpha, newDepth, !cutNode);
    } else if (!PvNode || moveCount > 1) {
      value = -search<NonPV>(ss + 1, -(alpha + 1), -alpha, newDepth, !cutNode);
    }
    if (PvNode && (moveCount == 1 || value > alpha)) {
      (ss + 1)->pv[0] = MOVE_NONE;
      value = -search<PV>(ss + 1, -beta, -alpha, newDepth, false);
    }
    pos.undo_move(move);
    if (stopped || eng.stopFlag.load(std::memory_order_relaxed)) {
      stopped = true;
      return 0;
    }

    if (rootNode) {
      RootMove* rm = nullptr;
      for (auto& r : rootMoves)
        if (r.move == move) { rm = &r; break; }
      if (moveCount == 1 || value > alpha) {
        rm->score = value;
        rm->selDepth = selDepth;
        rm->pv.resize(1);
        rm->pv[0] = move;
        for (Move* p = (ss + 1)->pv; *p != MOVE_NONE; ++p) rm->pv.push_back(*p);
      } else
        rm->score = -VALUE_INFINITE;
    }

    if (value > bestValue) {
      bestValue = value;
      if (value > alpha) {
        bestMove = move;
        if (PvNode) update_pv(ss->pv, move, (ss + 1)->pv);
        if (value >= beta) break;
        alpha = value;
      }
    }
    if (move != bestMove) {
      if (capture && nc < 32) capsTried[nc++] = move;
      else if (!noisy && nq < 32) quietsTried[nq++] = move;
    }
  }

  if (moveCount == 0) return excluded ? alpha : ss->inCheck ? mated_in(ss->ply) : draw_value(ss->ply);

  if (bestMove != MOVE_NONE && bestValue >= beta) {
    if (!is_noisy(bestMove)) update_quiet_stats(ss, bestMove, quietsTried, nq, depth);
    update_capture_stats(bestMove, capsTried, nc, depth);
  }

  int bound = bestValue >= beta ? BOUND_LOWER : (PvNode && bestMove != MOVE_NONE && bestValue > origAlpha) ? BOUND_EXACT : BOUND_UPPER;
  if (!excluded && !(rootNode && pvIdx > 0))
    eng.tt.store(key, value_to_tt(bestValue, ss->ply), ss->ttPv, bound, depth, bestMove, rawEval);

  if (!ss->inCheck && !excluded && (bestMove == MOVE_NONE || !is_capture(bestMove)) &&
      !(bound == BOUND_LOWER && bestValue <= ss->staticEval) && !(bound == BOUND_UPPER && bestValue >= ss->staticEval))
    update_correction(bestValue, ss->staticEval, depth);
  return bestValue;
}

// ------------------------------------------------------- iterative deepening
bool Worker::prepare(const Position& root, const SearchLimits& lim) {
  pos = root;
  limits = lim;
  startTime = eng.startTime;
  nodes = 0;
  selDepth = 0;
  stopped = false;
  completedDepth = 0;
  nmpMinPly = 0;
  rootColor = pos.side_to_move();
  rootPly = 0;

  rootMoves.clear();
  {
    MoveList ml(pos);
    for (auto& em : ml) {
      if (!lim.searchmoves.empty() && std::find(lim.searchmoves.begin(), lim.searchmoves.end(), em.move) == lim.searchmoves.end())
        continue;
      rootMoves.emplace_back(em.move);
    }
  }
  if (rootMoves.empty()) return false;

  for (int i = 0; i < MAX_PLY + 10; ++i) {
    Stack& s = stackStorage[i];
    s.ply = i - 4;
    s.currentMove = MOVE_NONE;
    s.excludedMove = MOVE_NONE;
    s.killers[0] = s.killers[1] = MOVE_NONE;
    s.staticEval = VALUE_NONE;
    s.statScore = 0;
    s.moveCount = 0;
    s.inCheck = false;
    s.ttPv = false;
    s.cont = &contHist[0];
    s.pv[0] = MOVE_NONE;
  }
  return true;
}

void Worker::iterate(const Position& root, const SearchLimits& lim) {
  if (!prepare(root, lim)) return;
  Stack* ss = stackStorage + 4;

  int multiPV = std::min<int>(std::max(1, eng.effMultiPV), int(rootMoves.size()));
  int maxDepth = lim.depth ? std::min(lim.depth, MAX_PLY - 8) : MAX_PLY - 8;
  Move lastBest = MOVE_NONE;
  int stability = 0;
  Value prevIterScore = 0;

  for (rootDepth = 1; rootDepth <= maxDepth && !eng.stopFlag.load(); ++rootDepth) {
    // Lazy SMP: helper threads skip some depths to diversify the shared TT
    if (id > 0 && ((rootDepth + id) % 3 == 0)) continue;
    for (auto& rm : rootMoves) rm.prevScore = rm.score;
    for (pvIdx = 0; pvIdx < multiPV && !eng.stopFlag.load(); ++pvIdx) {
      selDepth = 0;
      Value avg = rootMoves[pvIdx].prevScore == -VALUE_INFINITE ? 0 : rootMoves[pvIdx].prevScore;
      Value delta = 10 + std::abs(avg) / 128;
      Value alpha = -VALUE_INFINITE, beta = VALUE_INFINITE;
      if (rootDepth >= 5) {
        alpha = std::max<Value>(avg - delta, -VALUE_INFINITE);
        beta = std::min<Value>(avg + delta, VALUE_INFINITE);
      }
      int failedHigh = 0;
      Value bestValue;
      while (true) {
        int adj = std::max(1, rootDepth - failedHigh);
        bestValue = search<Root>(ss, alpha, beta, adj, false);
        std::stable_sort(rootMoves.begin() + pvIdx, rootMoves.end(), [](const RootMove& a, const RootMove& b) {
          return a.score != b.score ? a.score > b.score : a.prevScore > b.prevScore;
        });
        if (eng.stopFlag.load()) break;
        if (bestValue <= alpha) {
          beta = (alpha + beta) / 2;
          alpha = std::max<Value>(bestValue - delta, -VALUE_INFINITE);
          failedHigh = 0;
          if (id == 0 && rootDepth > 8) print_info(rootDepth, pvIdx, bestValue, BOUND_UPPER);
        } else if (bestValue >= beta) {
          beta = std::min<Value>(bestValue + delta, VALUE_INFINITE);
          ++failedHigh;
          if (id == 0 && rootDepth > 8) print_info(rootDepth, pvIdx, bestValue, BOUND_LOWER);
        } else
          break;
        delta += delta / 3;
        if (alpha <= -VALUE_MATE_IN_MAX_PLY) alpha = -VALUE_INFINITE;
        if (beta >= VALUE_MATE_IN_MAX_PLY) beta = VALUE_INFINITE;
      }
      std::stable_sort(rootMoves.begin(), rootMoves.begin() + pvIdx + 1, [](const RootMove& a, const RootMove& b) {
        return a.score != b.score ? a.score > b.score : a.prevScore > b.prevScore;
      });
      if (id == 0 && (eng.stopFlag.load() || pvIdx + 1 == multiPV)) {
        for (int k = 0; k <= pvIdx; ++k)
          if (rootMoves[k].score != -VALUE_INFINITE) print_info(rootDepth, k, rootMoves[k].score, BOUND_EXACT);
      }
    }
    if (!eng.stopFlag.load()) completedDepth = rootDepth;

    if (id == 0 && !eng.stopFlag.load()) {
      Value cur = rootMoves[0].score;
      if (rootMoves[0].move == lastBest) ++stability; else stability = 0;
      lastBest = rootMoves[0].move;
      if (eng.timeManaged) {
        TimePoint elapsed = now_ms() - startTime;
        double scale = 1.0;
        scale *= stability >= 4 ? 0.7 : stability >= 2 ? 0.85 : stability == 0 && rootDepth > 6 ? 1.35 : 1.0;
        if (rootDepth > 6 && cur < prevIterScore - 25) scale *= 1.25;
        if (rootMoves.size() == 1 && rootDepth >= 6) scale = 0.0;
        if (std::abs(cur) >= VALUE_MATE_IN_MAX_PLY && rootDepth >= 2 * (VALUE_MATE - std::abs(cur)) + 4) scale = 0.0;
        scale = std::min(scale, 1.6);
        // an iteration costs ~2x the previous one: do not begin one that cannot plausibly finish inside the budget
        if (double(elapsed) >= double(eng.optimumTime) * scale || (scale <= 1.0 && double(elapsed) >= double(eng.optimumTime) * 0.55)) eng.stopFlag = true;
      }
      prevIterScore = cur;
    }
  }
}

// ======================================================================== Engine

Engine::Engine() : stopFlag(false), running(false) {
  search_init();
  tt.resize(opt.hash);
  out = [](const std::string& s) { std::cout << s << std::endl; };
}

Engine::~Engine() {
  stop();
  wait();
}

void Engine::set_hash(int mb) {
  wait();
  opt.hash = mb;
  tt.resize(mb);
}

void Engine::new_game() {
  wait();
  tt.clear();
  workers.clear();
}

void Engine::set_position(const std::string& fen, const std::vector<std::string>& moves) {
  wait();
  startFen = fen;
  pos.set_fen(fen);
  gameMoves.clear();
  for (auto& s : moves) {
    Move m = parse_move(pos, s);
    if (m == MOVE_NONE) break;
    pos.do_move(m);
    gameMoves.push_back(m);
  }
}

void Engine::stop() { stopFlag = true; }

void Engine::wait() {
  if (mainThread.joinable()) mainThread.join();
}

void Engine::go(const SearchLimits& lim) {
  wait();
  stopFlag = false;
  running = true;
  mainThread = std::thread([this, lim]() { run_search(lim); });
}

SearchResult Engine::search_blocking(const SearchLimits& lim) {
  bool oldSilent = silent;
  go(lim);
  wait();
  silent = oldSilent;
  return lastResult;
}

void Engine::run_search(SearchLimits lim) {
  startTime = now_ms();
  rootColor = pos.side_to_move();
  nodeLimit = lim.nodes;
  depthLimit = lim.depth;
  tt.new_search();

  // ---- time management
  timeManaged = false;
  if (lim.movetime > 0) {
    timeManaged = true;
    optimumTime = maximumTime = std::max<int64_t>(1, lim.movetime - opt.moveOverhead);
  } else if (!lim.infinite && lim.depth == 0 && lim.nodes == 0 && (lim.time[0] || lim.time[1])) {
    timeManaged = true;
    int us = pos.side_to_move();
    int64_t myTime = lim.time[us], myInc = lim.inc[us];
    // Bullet-aware allocation: ~1/mtg of the usable clock plus most of the increment, with hard fractions of the clock as
    // caps, and a "hurry" regime when the clock is nearly gone. Think time scales DOWN with the clock, never up.
    int mtg = lim.movestogo ? std::min(lim.movestogo, 50) : 45;
    int64_t avail = std::max<int64_t>(1, myTime - opt.moveOverhead);
    double incPart = double(myInc) * 0.8;
    double opt_ = double(avail) / mtg + incPart;
    if (avail < 10000) opt_ = double(avail) / 55.0 + incPart * 0.9;       // hurry: < 10 s on the clock
    if (avail < 3000)  opt_ = double(avail) / 70.0 + incPart * 0.8;       // panic: < 3 s
    double cap = std::min(double(avail) * 0.18 + double(myInc) * 0.5, double(avail) * 0.5);
    opt_ = std::min(opt_, std::max(cap, 1.0));
    double max_ = std::min(double(avail) * 0.30, opt_ * 2.0);
    if (lim.movestogo == 1) opt_ = max_ = avail * 0.8;
    optimumTime = std::max<int64_t>(2, int64_t(opt_));
    maximumTime = std::max<int64_t>(optimumTime, int64_t(max_));
    maximumTime = std::min<int64_t>(maximumTime, avail);
  }

  // ---- root-policy quirks need exact scores for several root moves -> internal MultiPV, and a smaller main-search budget
  const bool varietyOn = opt.varietyPlies > 0 && pos.game_ply() < opt.varietyPlies;
  const bool policyOn = (opt.quantalSwindle || varietyOn) && lim.searchmoves.empty() && opt.searchMode != 1;
  effMultiPV = std::max(1, opt.multiPV);
  if (policyOn && varietyOn) effMultiPV = std::max(effMultiPV, 3);  // opening variety needs exact scores of a few root moves
  // (the swindle does NOT touch the main search: its candidates come from a separate probe search afterwards — an early
  //  design that used MultiPV=3 and a 60 % budget for the main search cost >200 Elo, see docs/ALGORITHMS.md)

  int nthreads = std::max(1, opt.threads);
  if ((int)workers.size() != nthreads) {
    workers.clear();
    for (int i = 0; i < nthreads; ++i) workers.emplace_back(new Worker(*this, i));
  }
  std::vector<std::thread> helpers;
  for (int i = 1; i < nthreads && opt.searchMode != 1; ++i) helpers.emplace_back([this, i, &lim]() { workers[i]->iterate(pos, lim); });
  if (opt.searchMode == 1) workers[0]->mcts(pos, lim); else workers[0]->iterate(pos, lim);

  if (lim.infinite || (lim.depth == 0 && lim.nodes == 0 && !timeManaged)) {
    // "go infinite": must wait for stop before reporting
    while (!stopFlag.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  stopFlag = true;
  for (auto& t : helpers) t.join();

  SearchResult res;
  Worker& w0 = *workers[0];
  res.rootMoves = w0.rootMoves;
  if (!w0.rootMoves.empty()) {
    const RootMove& best = w0.rootMoves[0];
    res.best = best.move;
    res.score = best.score;
    res.ponder = best.pv.size() > 1 ? best.pv[1] : MOVE_NONE;
    res.depth = w0.completedDepth;
  }
  for (auto& w : workers) res.nodes += w->node_count();
  if (policyOn && !res.rootMoves.empty() && std::abs(res.score) < VALUE_MATE_IN_MAX_PLY) apply_root_policy(res);
  lastResult = res;
  if (out && !silent) {
    std::string s = "bestmove " + move_str(res.best);
    if (res.ponder != MOVE_NONE) s += " ponder " + move_str(res.ponder);
    out(s);
  }
  running = false;
}

// ============================================================ MCTS: PUCT + Gumbel root (SearchMode 1)
//
// An AlphaZero-flavoured tree search that needs no trained policy network:
//   * priors  = softmax over a cheap move heuristic (MVV, SEE sign, checks, promotions, butterfly history)
//   * values  = win probability of the quiescence-searched static evaluation (HCE or NNUE)
//   * non-root selection: PUCT with first-play-urgency reduction
//   * root selection: Gumbel-Top-k sampling + Sequential Halving ("Policy improvement by planning with Gumbel",
//     Danihelka et al., ICLR 2022), which is provably better than PUCT at small simulation budgets.
void Worker::mcts(const Position& root, const SearchLimits& lim) {
  if (!prepare(root, lim)) return;
  struct MNode {
    Move move;
    float prior;
    int visits;
    float W;       // sum of values from the viewpoint of the player who chose `move`
    int child;     // index of first child, -1 if unexpanded
    int nchild;
    int8_t term;   // 0 = not terminal, 1 = terminal with termV
    float termV;   // value for the player who chose `move`
  };
  std::vector<MNode> pool;
  pool.reserve(1 << 20);
  Stack* ss = stackStorage + 4;
  PRNG rng(now_ms() * 2654435761ULL + 99);
  auto uni = [&]() { return (double(rng.below(1u << 24)) + 0.5) / double(1u << 24); };
  auto wp = [](int cp) { return 1.0f / (1.0f + std::exp(-float(cp) * 0.0057565f)); };
  static const float VAL[8] = {0, 1, 3, 3.1f, 5, 9, 0, 0};

  // value of the current position for its side to move, in [0,1]
  auto leaf = [&]() -> float {
    ss->ply = 0;
    Value v = qsearch<PV>(ss, -VALUE_INFINITE, VALUE_INFINITE);
    if (v >= VALUE_MATE_IN_MAX_PLY) return 1.0f;
    if (v <= -VALUE_MATE_IN_MAX_PLY) return 0.0f;
    return wp(v);
  };
  // create children of node `n` (pool[n]) for the current position; returns false if the position is terminal
  auto expand = [&](int n, double* logits) -> bool {
    MoveList ml(pos);
    if (ml.size() == 0) {
      pool[n].term = 1;
      pool[n].termV = pos.in_check() ? 1.0f : 0.5f;  // the side to move is mated -> the chooser of this edge won
      return false;
    }
    int first = int(pool.size());
    double tot = 0, mx = -1e9;
    std::vector<double> sc(ml.size());
    Color us = pos.side_to_move();
    for (size_t i = 0; i < ml.size(); ++i) {
      Move m = ml.moves[i].move;
      double s = 0;
      if (is_capture(m)) {
        PieceType victim = flags_of(m) == MF_EP ? PAWN : type_of(pos.piece_on(to_sq(m)));
        s += 1.0 + 0.45 * VAL[victim] + (pos.see_ge(m, 0) ? 0.8 : -1.8);
      }
      if (is_promo(m)) s += promo_type(m) == QUEEN ? 3.0 : -1.0;
      if (pos.gives_check(m)) s += 0.9;
      if (!is_noisy(m)) s += std::clamp(double(mainHist[us][from_sq(m) * 64 + to_sq(m)]) / 6000.0, -1.5, 1.5);
      sc[i] = s;
      mx = std::max(mx, s);
    }
    for (size_t i = 0; i < ml.size(); ++i) tot += (sc[i] = std::exp(sc[i] - mx));
    for (size_t i = 0; i < ml.size(); ++i) {
      MNode c{ml.moves[i].move, float(sc[i] / tot), 0, 0.f, -1, 0, 0, 0.f};
      pool.push_back(c);
      if (logits && n == 0) logits[i] = std::log(sc[i] / tot);
    }
    pool[n].child = first;
    pool[n].nchild = int(ml.size());
    return true;
  };

  pool.push_back(MNode{MOVE_NONE, 1.f, 0, 0.f, -1, 0, 0, 0.f});
  std::vector<double> rootLogits(256);
  expand(0, rootLogits.data());
  // honour searchmoves: drop disallowed root children by zeroing their prior (handled at candidate selection)
  const int nRoot = pool[0].nchild;

  int64_t simCount = 0;
  auto simulate = [&](int rootChild) {  // one simulation that starts by playing root child `rootChild` (index into pool)
    int path[256];
    int depth = 0;
    int cur = rootChild;
    path[depth++] = cur;
    pos.do_move(pool[cur].move);
    float v;  // value for the player who chose pool[cur].move (i.e. who just moved)
    while (true) {
      MNode& nd = pool[cur];
      if (nd.term) { v = nd.termV; break; }
      if (pos.is_draw(depth)) { v = 0.5f; nd.term = 1; nd.termV = 0.5f; break; }
      if (nd.child < 0) {  // leaf: expand and evaluate
        if (!expand(cur, nullptr)) { v = pool[cur].termV; break; }
        float lv = leaf();           // value for the side to move now = the opponent of the chooser
        if (stopped) { v = 0.5f; break; }
        v = 1.0f - lv;
        break;
      }
      // PUCT selection among nd's children (stats are from the chooser's viewpoint = the mover at the child)
      int bestC = -1;
      float bestScore = -1e9f;
      float parentN = float(std::max(1, nd.visits));
      float sq = std::sqrt(parentN);
      float parentQ = nd.visits ? nd.W / nd.visits : 0.5f;  // parent's chooser-view; children are the opposite side
      float fpu = (1.0f - parentQ) - 0.2f;
      for (int k = 0; k < nd.nchild; ++k) {
        MNode& ch = pool[nd.child + k];
        float q = ch.visits ? ch.W / ch.visits : fpu;
        float u = 1.6f * ch.prior * sq / (1.0f + ch.visits);
        float s = q + u;
        if (s > bestScore) { bestScore = s; bestC = nd.child + k; }
      }
      cur = bestC;
      path[depth++] = cur;
      pos.do_move(pool[cur].move);
      if (depth >= 200) { v = 0.5f; break; }
    }
    // undo moves and back-propagate; v is from the viewpoint of the chooser at the deepest node
    for (int i = depth - 1; i >= 0; --i) {
      pool[path[i]].visits += 1;
      pool[path[i]].W += v;
      v = 1.0f - v;
      pos.undo_move(pool[path[i]].move);
    }
    pool[0].visits += 1;
    ++simCount;
  };

  // candidate set: Gumbel-Top-k over root logits
  std::vector<int> idx;
  for (int k = 0; k < nRoot; ++k) {
    Move m = pool[1 + k].move;
    if (!lim.searchmoves.empty() && std::find(lim.searchmoves.begin(), lim.searchmoves.end(), m) == lim.searchmoves.end()) continue;
    idx.push_back(1 + k);
  }
  if (idx.empty()) return;
  std::vector<double> gum(pool.size(), 0.0);
  for (int k : idx) gum[k] = -std::log(-std::log(uni()));
  const int m0 = std::min<int>(16, int(idx.size()));
  std::sort(idx.begin(), idx.end(), [&](int a, int b) { return gum[a] + rootLogits[a - 1] > gum[b] + rootLogits[b - 1]; });
  std::vector<int> cand(idx.begin(), idx.begin() + m0);

  // simulation budget: nodes limit, depth limit (as sims), or time (calibrated on the first few simulations)
  int64_t budget = lim.nodes ? int64_t(lim.nodes) : lim.depth ? int64_t(lim.depth) * 400 : 0;
  auto time_up = [&]() { return eng.stopFlag.load(std::memory_order_relaxed) || (eng.timeManaged && now_ms() - startTime >= std::min<TimePoint>(eng.optimumTime * 13 / 10, eng.maximumTime)); };
  auto qhat = [&](int c) { return pool[c].visits ? pool[c].W / pool[c].visits : 0.5f; };
  auto sigma = [&](int c, int maxN) { return (50.0 + maxN) * 0.1 * qhat(c); };

  auto report = [&](bool final_) {
    // order root moves: best candidate first by (visits, then Q)
    std::vector<int> all(idx.begin(), idx.end());
    std::sort(all.begin(), all.end(), [&](int a, int b) {
      if (pool[a].visits != pool[b].visits) return pool[a].visits > pool[b].visits;
      return qhat(a) > qhat(b);
    });
    rootMoves.clear();
    for (int c : all) {
      RootMove rm(pool[c].move);
      float q = std::clamp(qhat(c), 0.0005f, 0.9995f);
      rm.score = pool[c].visits ? Value(std::lround(std::log(q / (1.0f - q)) / 0.0057565f)) : -VALUE_INFINITE;
      // principal variation: follow most visited children
      int cur = c;
      while (pool[cur].child >= 0 && rm.pv.size() < 12) {
        int bc = -1, bv = 0;
        for (int k = 0; k < pool[cur].nchild; ++k)
          if (pool[pool[cur].child + k].visits > bv) { bv = pool[pool[cur].child + k].visits; bc = pool[cur].child + k; }
        if (bc < 0) break;
        rm.pv.push_back(pool[bc].move);
        cur = bc;
      }
      rootMoves.push_back(rm);
    }
    completedDepth = rootMoves.empty() ? 0 : int(rootMoves[0].pv.size());
    if (final_ || true) {
      if (!eng.silent && eng.out && !rootMoves.empty()) {
        TimePoint el = std::max<TimePoint>(1, now_ms() - startTime);
        std::ostringstream os;
        os << "info depth " << std::max(1, completedDepth) << " seldepth " << selDepth << " multipv 1 score cp " << rootMoves[0].score
           << " nodes " << nodes.load() << " nps " << (nodes.load() * 1000 / el) << " time " << el << " pv " << pv_string(rootMoves[0].pv);
        eng.out(os.str());
      }
    }
  };

  if (cand.size() == 1) {
    for (int i = 0; i < 4 && !time_up(); ++i) simulate(cand[0]);
    report(true);
    return;
  }

  // calibrate: one simulation per candidate, then derive the budget from the measured speed
  for (int c : cand) { if (time_up()) break; simulate(c); }
  auto per_sim_ms = [&]() { return double(std::max<TimePoint>(1, now_ms() - startTime)) / double(std::max<int64_t>(1, simCount)); };
  auto budget_for_remaining = [&](double frac) -> int64_t {
    TimePoint el = now_ms() - startTime;
    double left = double(eng.optimumTime) * frac - double(el);
    return left <= 0 ? 0 : std::max<int64_t>(8, int64_t(left / per_sim_ms()));
  };
  if (budget == 0) budget = eng.timeManaged ? budget_for_remaining(0.8) : 2000;

  // one Sequential-Halving round over `cs` with roughly `bud` simulations; returns the surviving candidate
  auto halving_round = [&](std::vector<int> cs, int64_t bud) -> int {
    int phases = 1;
    for (int s = int(cs.size()); s > 2; s = (s + 1) / 2) ++phases;
    int64_t perPhase = std::max<int64_t>(int64_t(cs.size()), bud / phases);
    while (cs.size() > 1 && !time_up()) {
      int64_t each = std::max<int64_t>(1, perPhase / int64_t(cs.size()));
      for (int64_t r = 0; r < each && !time_up(); ++r)
        for (int c : cs) { if (time_up()) break; simulate(c); }
      int maxN = 0;
      for (int c : cs) maxN = std::max(maxN, pool[c].visits);
      std::sort(cs.begin(), cs.end(), [&](int a, int b) {
        return gum[a] + rootLogits[a - 1] + sigma(a, maxN) > gum[b] + rootLogits[b - 1] + sigma(b, maxN);
      });
      cs.resize(cs.size() == 2 ? 1 : (cs.size() + 1) / 2);
    }
    return cs[0];
  };

  int winner = halving_round(cand, budget);
  report(false);
  // use the remaining clock: further rounds restricted to the most promising moves (re-seeded by visit count)
  while (eng.timeManaged && !time_up()) {
    int64_t bud = budget_for_remaining(0.85);
    if (bud < 24) break;
    std::vector<int> top(idx.begin(), idx.end());
    std::sort(top.begin(), top.end(), [&](int a, int b) { return pool[a].visits > pool[b].visits; });
    top.resize(std::min<size_t>(top.size(), 6));
    if (std::find(top.begin(), top.end(), winner) == top.end()) top.push_back(winner);
    winner = halving_round(top, bud);
    report(false);
  }
  report(true);
  if (!rootMoves.empty()) {
    Move w = pool[winner].move;
    auto it = std::find_if(rootMoves.begin(), rootMoves.end(), [&](const RootMove& r) { return r.move == w; });
    if (it != rootMoves.end()) std::rotate(rootMoves.begin(), it, it + 1);
  }
}

// ============================================================ root policy (Heretic quirks)
//
// 1. Opening variety: for the first N plies choose among root moves within `varietyMargin` cp of the best, weighted
//    by a softmax of their scores (so book-like diversity without a book).
// 2. Quantal-response swindle: model the opponent as a *bounded-rational* chooser (logit / quantal response):
//    P(reply i) ∝ exp(s_i / T), where s_i is the reply's value for the opponent and T grows as the opponent's Elo or
//    clock shrinks.  For each near-best candidate we run a full-width shallow search of the opponent's replies and
//    maximise the *expected* win probability instead of the minimax value.  With T -> 0 this reduces to ordinary
//    minimax; with T > 0 it prefers lines where the opponent has to find a narrow path ("only moves").
//    Guards: never concede more than kSwindleMaxLoss cp versus the best move, never when clearly winning.
namespace {
inline std::string pct(double v) { char b[16]; snprintf(b, sizeof b, "%.1f%%", v * 100); return b; }
inline double wp_of(double cp) { return 1.0 / (1.0 + std::exp(-cp * 0.0057565)); }  // ln(10)/400
}  // namespace

void Engine::apply_root_policy(SearchResult& res) {
  static thread_local PRNG rng(uint64_t(now_ms()) * 0x9E3779B97F4A7C15ULL + 12345);
  auto& rms = res.rootMoves;  // sorted best-first, exact scores for the first effMultiPV entries
  const int nExact = std::min<int>(int(rms.size()), effMultiPV);
  const Value best = rms[0].score;
  if (best <= -VALUE_MATE_IN_MAX_PLY) return;

  // ---- opening variety
  if (opt.varietyPlies > 0 && pos.game_ply() < opt.varietyPlies) {
    if (nExact < 2) return;
    std::vector<int> cand;
    for (int i = 0; i < nExact; ++i)
      if (rms[i].score != -VALUE_INFINITE && rms[i].score >= best - opt.varietyMargin) cand.push_back(i);
    if (cand.size() > 1) {
      double tot = 0, w[8];
      int n = std::min<int>(8, int(cand.size()));
      for (int k = 0; k < n; ++k) tot += (w[k] = std::exp(-double(best - rms[cand[k]].score) / 8.0));
      double r = (rng.below(1 << 24) / double(1 << 24)) * tot;
      int pick = 0;
      for (int k = 0; k < n; ++k) { if ((r -= w[k]) <= 0) { pick = k; break; } }
      std::swap(rms[0], rms[cand[pick]]);
      res.best = rms[0].move;
      res.ponder = rms[0].pv.size() > 1 ? rms[0].pv[1] : MOVE_NONE;
      res.score = rms[0].score;
    }
    return;
  }

  if (!opt.quantalSwindle) return;
  const int kSwindleMaxLoss = opt.swindleMaxLoss;   // cp we are willing to give up for practical chances
  constexpr int kNoSwindleAbove = 250;  // do not gamble when clearly better
  if (best > kNoSwindleAbove) return;

  // ---- temperature (cp) from opponent strength and clock
  double T = 60.0;
  if (opt.opponentElo > 0) T = std::max(25.0, 40.0 + (2400.0 - opt.opponentElo) * 0.12);
  if (opt.oppTimeMs > 0 && opt.oppTimeMs < 10000) T *= 1.0 + (10000.0 - opt.oppTimeMs) / 10000.0;

  if (timeManaged && optimumTime < 60) return;  // too little time to think twice
  if (!probe) {
    probe.reset(new Engine());
    probe->silent = true;
    probe->opt.threads = 1;
    probe->opt.moveOverhead = 0;
    probe->set_hash(16);
  }
  // Extra think time on top of the main search: ~25 % of the move's optimum for candidates, ~25 % for reply probes
  // (or fixed depths when the search is not time-managed).
  const int64_t slice = timeManaged ? std::max<int64_t>(4, optimumTime / 4) : 0;
  const int mainDepth = std::max(4, res.depth);

  // ---- phase 1: candidate scores from a separate MultiPV probe of the root (apples-to-apples across candidates)
  probe->pos = pos;
  probe->opt.multiPV = 4;
  probe->effMultiPV = 4;
  SearchLimits cl;
  if (slice > 0) cl.movetime = slice; else cl.depth = std::max(4, mainDepth - 3);
  SearchResult cr = probe->search_blocking(cl);
  struct Cand { Move m; int score; double U; };
  std::vector<Cand> cand;
  int pbest = -VALUE_INFINITE;
  for (auto& rm : cr.rootMoves) if (rm.score != -VALUE_INFINITE) pbest = std::max<int>(pbest, rm.score);
  bool haveMain = false;
  for (auto& rm : cr.rootMoves)
    if (rm.score != -VALUE_INFINITE && rm.score >= pbest - kSwindleMaxLoss && std::abs(rm.score) < VALUE_MATE_IN_MAX_PLY) {
      cand.push_back({rm.move, rm.score, 0});
      if (rm.move == res.best) haveMain = true;
    }
  if (cand.size() < 2 || !haveMain) return;  // nothing to choose between, or the probe disagrees with the main search: trust it

  // ---- phase 2: for each candidate, probe the opponent's replies (full width)
  for (auto& c : cand) {
    Position p = pos;
    p.do_move(c.m);
    MoveList replies(p);
    double bonus = 0;
    if (replies.size() > 0) {
      probe->pos = p;
      probe->opt.multiPV = int(replies.size());
      probe->effMultiPV = int(replies.size());
      SearchLimits pl;
      if (slice > 0) pl.movetime = std::max<int64_t>(3, slice / int64_t(cand.size())); else pl.depth = std::max(3, mainDepth - 4);
      SearchResult pr = probe->search_blocking(pl);
      double top = -1e9;
      for (auto& rm : pr.rootMoves) if (rm.score != -VALUE_INFINITE) top = std::max(top, double(std::clamp<int>(rm.score, -1500, 1500)));
      double Z = 0, EV = 0;
      for (auto& rm : pr.rootMoves) {
        double sc = rm.score == -VALUE_INFINITE ? top - 600.0 : double(std::clamp<int>(rm.score, -1500, 1500));
        double w = std::exp((sc - top) / T);
        Z += w;
        EV += w * wp_of(-sc);
      }
      // swindle bonus = E[wp] against the fallible opponent minus the same probe's minimax value
      bonus = Z > 0 ? std::max(0.0, EV / Z - wp_of(-top)) : 0.0;
    }
    c.U = wp_of(c.score) + bonus;
  }
  size_t bi = 0, mi = 0;
  for (size_t i = 0; i < cand.size(); ++i) {
    if (cand[i].U > cand[bi].U) bi = i;
    if (cand[i].m == res.best) mi = i;
  }
  if (out && !silent) {
    std::string d = "info string swindle probe T=" + std::to_string(int(T)) + "cp";
    for (auto& c : cand) d += " " + move_str(c.m) + "(" + std::to_string(c.score) + "cp,U=" + pct(c.U) + ")";
    out(d);
  }
  if (bi != mi && cand[bi].U - cand[mi].U >= opt.swindleMinGain / 1000.0) {
    for (auto& rm : rms)
      if (rm.move == cand[bi].m) { std::swap(rms[0], rm); break; }
    res.best = cand[bi].m;
    res.ponder = rms[0].pv.size() > 1 ? rms[0].pv[1] : MOVE_NONE;
    res.score = rms[0].score;
    if (out && !silent) out("info string swindle: chose " + move_str(res.best) + " over " + move_str(cand[mi].m));
  }
}

}  // namespace heretic
