#include "tools.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <thread>

#include "eval.h"
#include "movegen.h"

namespace heretic {

// 40-byte training record. Board is 64 nibbles (piece codes 0..14), side-to-move relative result.
struct PackedPos {
  uint8_t sq[32];
  uint8_t stm;
  uint8_t flags;   // bit0: best move is noisy (capture/promo), bit1: in check
  int16_t score;   // search score, side-to-move POV (cp)
  uint8_t result;  // 0 = stm loses, 1 = draw, 2 = stm wins
  uint8_t ply;
  uint8_t pad[2];
};
static_assert(sizeof(PackedPos) == 40, "PackedPos must be 40 bytes");

static PackedPos pack(const Position& p, int score, uint8_t flags, uint8_t ply) {
  PackedPos r;
  std::memset(&r, 0, sizeof(r));
  for (int s = 0; s < 64; s += 2) r.sq[s / 2] = uint8_t(p.piece_on(s) | (p.piece_on(s + 1) << 4));
  r.stm = uint8_t(p.side_to_move());
  r.flags = flags;
  r.score = int16_t(std::max(-32000, std::min(32000, score)));
  r.ply = ply;
  return r;
}

namespace {

std::mutex fileMutex;

struct GameRecord {
  std::vector<PackedPos> pos;
};

// Plays one self-play game; appends labelled positions to `out`. Returns false if the game was discarded.
bool play_game(Engine& eng, PRNG& rng, int depth, std::vector<PackedPos>& out) {
  Position pos;
  pos.set_fen(Position::StartFEN);
  // random opening (balanced check below)
  int randPlies = 8 + int(rng.below(5));
  for (int i = 0; i < randPlies; ++i) {
    MoveList ml(pos);
    if (ml.size() == 0) return false;
    pos.do_move(ml.moves[rng.below(ml.size())].move);
  }
  {
    MoveList ml(pos);
    if (ml.size() == 0 || pos.is_draw(0)) return false;
    eng.pos = pos;
    SearchLimits lim;
    lim.depth = 6;
    SearchResult r = eng.search_blocking(lim);
    if (std::abs(r.score) > 350) return false;  // unbalanced opening
  }

  std::vector<PackedPos> rec;
  std::vector<int> recStm;
  int winStreak = 0, drawStreak = 0;
  int result = 1;  // from White's point of view: 0 black wins, 1 draw, 2 white wins
  bool decided = false;
  for (int ply = 0; ply < 400; ++ply) {
    MoveList ml(pos);
    if (ml.size() == 0) {
      result = pos.in_check() ? (pos.side_to_move() == WHITE ? 0 : 2) : 1;
      decided = true;
      break;
    }
    if (pos.is_draw(0) || pos.rule50() >= 100) {
      result = 1;
      decided = true;
      break;
    }
    eng.pos = pos;
    SearchLimits lim;
    lim.depth = depth;
    SearchResult r = eng.search_blocking(lim);
    int score = r.score;
    Move best = r.best;
    if (best == MOVE_NONE) break;

    bool inCheck = pos.in_check();
    if (!inCheck && std::abs(score) < 3000 && ply >= 0) {
      uint8_t fl = (is_noisy(best) ? 1 : 0) | (inCheck ? 2 : 0);
      rec.push_back(pack(pos, score, fl, uint8_t(std::min(255, int(pos.game_ply())))));
      recStm.push_back(pos.side_to_move());
    }
    // adjudication
    int whiteScore = pos.side_to_move() == WHITE ? score : -score;
    if (std::abs(score) >= 1200) {
      ++winStreak;
      if (winStreak >= 5) {
        result = whiteScore > 0 ? 2 : 0;
        decided = true;
        break;
      }
    } else
      winStreak = 0;
    if (std::abs(score) <= 6 && pos.game_ply() > 100) {
      if (++drawStreak >= 30) {
        result = 1;
        decided = true;
        break;
      }
    } else
      drawStreak = 0;
    if (std::abs(score) >= VALUE_MATE_IN_MAX_PLY) {
      result = whiteScore > 0 ? 2 : 0;
      decided = true;
      break;
    }

    // a little extra randomness early on to diversify
    Move play = best;
    if (pos.game_ply() < 40 && rng.below(48) == 0) play = ml.moves[rng.below(ml.size())].move;
    pos.do_move(play);
  }
  if (!decided) return false;
  for (size_t i = 0; i < rec.size(); ++i) {
    int stmResult = recStm[i] == WHITE ? result : 2 - result;
    rec[i].result = uint8_t(stmResult);
    out.push_back(rec[i]);
  }
  return true;
}

void datagen(int64_t targetPositions, int threads, int depth, const std::string& path, uint64_t seed) {
  std::atomic<int64_t> total(0);
  std::atomic<int64_t> games(0);
  FILE* f = fopen(path.c_str(), "ab");
  if (!f) {
    printf("cannot open %s\n", path.c_str());
    return;
  }
  auto t0 = std::chrono::steady_clock::now();
  std::vector<std::thread> pool;
  for (int t = 0; t < threads; ++t) {
    pool.emplace_back([&, t]() {
      Engine eng;
      eng.silent = true;
      eng.opt.hash = 16;
      eng.opt.threads = 1;
      PRNG rng(seed * 1000003ULL + 7919ULL * (t + 1));
      std::vector<PackedPos> buf;
      while (total.load() < targetPositions) {
        buf.clear();
        eng.new_game();
        if (!play_game(eng, rng, depth, buf)) continue;
        {
          std::lock_guard<std::mutex> lk(fileMutex);
          fwrite(buf.data(), sizeof(PackedPos), buf.size(), f);
          fflush(f);
        }
        total += int64_t(buf.size());
        int64_t g = ++games;
        if (g % 25 == 0) {
          double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
          printf("datagen: %lld games, %lld positions, %.0f pos/s\n", (long long)g, (long long)total.load(),
                 total.load() / s);
          fflush(stdout);
        }
      }
    });
  }
  for (auto& th : pool) th.join();
  fclose(f);
  double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  printf("datagen done: %lld games, %lld positions in %.0fs -> %s\n", (long long)games.load(), (long long)total.load(), s,
         path.c_str());
}


// hceeval <in.bin> <out.i16> : handcrafted evaluation (side-to-move POV, cp) of every PackedPos record, for residual-NNUE training
void hceeval(const std::string& in, const std::string& out) {
  FILE* f = fopen(in.c_str(), "rb");
  FILE* g = fopen(out.c_str(), "wb");
  if (!f || !g) { printf("cannot open files\n"); return; }
  Position pos;
  PackedPos r;
  long long n = 0;
  std::vector<int16_t> buf;
  while (fread(&r, sizeof(r), 1, f) == 1) {
    Piece b[64];
    for (int s = 0; s < 64; s += 2) { b[s] = Piece(r.sq[s / 2] & 15); b[s + 1] = Piece(r.sq[s / 2] >> 4); }
    pos.set_board(b, Color(r.stm), 0, SQ_NONE);
    int v = evaluate_hce(pos);
    buf.push_back(int16_t(std::max(-32000, std::min(32000, v))));
    if (buf.size() == 65536) { fwrite(buf.data(), 2, buf.size(), g); buf.clear(); }
    ++n;
  }
  fwrite(buf.data(), 2, buf.size(), g);
  fclose(f);
  fclose(g);
  printf("hceeval: %lld records -> %s\n", n, out.c_str());
}

// ------------------------------------------------------------------------------------------------------------
// Texel-style tuning of the handcrafted evaluation: minimise  mean (R - sigmoid(K * eval))^2  over quiet self-play
// positions, where R is the game result from the side to move's point of view (0, 1/2, 1).  Coordinate descent
// over every EvalParams integer with a train/validation split; the best-validation parameter set is kept.
struct TPos {
  uint8_t b[64];
  uint8_t stm;
  float R;
};

double tune_error(const std::vector<TPos>& v, size_t lo, size_t hi, double K, int threads) {
  std::vector<double> part(threads, 0.0);
  std::vector<std::thread> th;
  size_t n = hi - lo;
  for (int t = 0; t < threads; ++t) {
    th.emplace_back([&, t]() {
      Position pos;
      size_t a = lo + n * t / threads, e = lo + n * (t + 1) / threads;
      double sum = 0;
      Piece b[64];
      for (size_t i = a; i < e; ++i) {
        for (int k = 0; k < 64; ++k) b[k] = Piece(v[i].b[k]);
        pos.set_board(b, Color(v[i].stm), 0, SQ_NONE);
        double x = evaluate_hce(pos);
        double pr = 1.0 / (1.0 + std::exp(-K * x));
        double d = v[i].R - pr;
        sum += d * d;
      }
      part[t] = sum;
    });
  }
  for (auto& t : th) t.join();
  double s = 0;
  for (double x : part) s += x;
  return s / double(n);
}

void write_inc(const std::string& path) {
  FILE* f = fopen(path.c_str(), "w");
  if (!f) return;
  int n = eval_param_count();
  fprintf(f, "// Generated by `heretic tune`.\nstatic const int TUNED_COUNT = %d;\nstatic const int TUNED[%d] = {\n", n, n);
  for (int i = 0; i < n; ++i) fprintf(f, "%d,%s", eval_param_ptr()[i], (i % 16 == 15) ? "\n" : " ");
  fprintf(f, "};\n");
  fclose(f);
}

void tune(const std::vector<std::string>& paths, long long maxPos, int sweeps, int threads, const std::string& outPath,
          int maxAbs, double beta, double mu) {
  // ---- load; game boundaries are detected by the ply counter resetting; the split is BY GAME (positions of one game
  //      share a result, so a per-position split leaks and rewards memorisation)
  std::vector<TPos> trainSet, valSet;
  std::vector<TPos> all;
  long long games = 0;
  PRNG rng(12345);
  for (auto& path : paths) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { printf("cannot open %s\n", path.c_str()); return; }
    PackedPos r;
    int lastPly = 1 << 30;
    bool gameVal = false;
    while (fread(&r, sizeof(r), 1, f) == 1) {
      if (r.ply <= lastPly) { ++games; gameVal = (rng.below(10) == 0); }
      lastPly = r.ply;
      if ((r.flags & 3) || std::abs(int(r.score)) > maxAbs) continue;
      TPos t;
      for (int s = 0; s < 64; s += 2) { t.b[s] = r.sq[s / 2] & 15; t.b[s + 1] = r.sq[s / 2] >> 4; }
      t.stm = r.stm;
      // blended target: game result and the (low-variance) search score of the labelling engine
      double ps = 1.0 / (1.0 + std::exp(-0.0057565 * r.score));
      t.R = float((1.0 - beta) * (r.result * 0.5) + beta * ps);
      (gameVal ? valSet : trainSet).push_back(t);
    }
    fclose(f);
  }
  auto shuffle = [&](std::vector<TPos>& v) { for (size_t i = v.size(); i > 1; --i) std::swap(v[i - 1], v[rng.below(i)]); };
  shuffle(trainSet);
  shuffle(valSet);
  if (maxPos > 0 && (long long)trainSet.size() > maxPos) trainSet.resize(size_t(maxPos));
  if (valSet.size() > 60000) valSet.resize(60000);
  size_t nTrain = trainSet.size();
  all = trainSet;
  all.insert(all.end(), valSet.begin(), valSet.end());
  size_t nAll = all.size();
  trainSet.clear(); trainSet.shrink_to_fit();
  printf("tune: %lld games; %zu train / %zu val positions (game-wise split), %d params, %d threads, beta=%.2f mu=%.1e\n", games,
         nTrain, nAll - nTrain, eval_param_count(), threads, beta, mu);
  fflush(stdout);

  int n = eval_param_count();
  int* P_ = eval_param_ptr();
  const std::vector<int> p0(P_, P_ + n);
  auto penalty = [&]() { double s = 0; for (int i = 0; i < n; ++i) { double d = P_[i] - p0[i]; s += d * d; } return mu * s; };

  double bestK = 0.0057, bestE = 1e9;
  for (double K = 0.002; K <= 0.012; K += 0.0004) {
    double e = tune_error(all, 0, nTrain, K, threads);
    if (e < bestE) { bestE = e; bestK = K; }
  }
  double lo = bestK - 0.0004, hi = bestK + 0.0004;
  for (int it = 0; it < 10; ++it) {
    double m1 = lo + (hi - lo) / 3, m2 = hi - (hi - lo) / 3;
    if (tune_error(all, 0, nTrain, m1, threads) < tune_error(all, 0, nTrain, m2, threads)) hi = m2; else lo = m1;
  }
  const double K = 0.5 * (lo + hi);
  double E0 = tune_error(all, 0, nTrain, K, threads), V0 = tune_error(all, nTrain, nAll, K, threads);
  printf("tune: K=%.5f  train=%.6f val=%.6f (initial)\n", K, E0, V0);
  fflush(stdout);

  std::vector<int> bestParams(P_, P_ + n);
  double bestVal = V0, cur = E0;  // `cur` = train error (without penalty); objective = cur + penalty()
  for (int sweep = 0; sweep < sweeps; ++sweep) {
    int step = sweep == 0 ? 3 : (sweep == 1 ? 2 : 1);
    int changed = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n; ++i) {
      const int old = P_[i];
      const double pen0 = penalty();
      bool improved = false;
      for (int dir : {+1, -1}) {
        P_[i] = old + dir * step;
        double e = tune_error(all, 0, nTrain, K, threads);
        if (e + penalty() < cur + pen0 - 1e-12) {
          cur = e; improved = true; ++changed;
          while (true) {
            int prev = P_[i];
            double penPrev = penalty();
            P_[i] = prev + dir * step;
            double e2 = tune_error(all, 0, nTrain, K, threads);
            if (e2 + penalty() < cur + penPrev - 1e-12) cur = e2; else { P_[i] = prev; break; }
          }
          break;
        }
        P_[i] = old;
      }
      if (!improved) P_[i] = old;
      if (i % 200 == 199) {
        double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        printf("tune: sweep %d  %d/%d params  train=%.6f  (%.0fs)\n", sweep + 1, i + 1, n, cur, el);
        fflush(stdout);
      }
    }
    double V = tune_error(all, nTrain, nAll, K, threads);
    printf("tune: === sweep %d done: step %d, %d changes, train=%.6f val=%.6f pen=%.6f (best val %.6f)\n", sweep + 1, step, changed, cur,
           V, penalty(), bestVal);
    fflush(stdout);
    if (V < bestVal - 1e-7) {
      bestVal = V;
      bestParams.assign(P_, P_ + n);
      eval_save_params(outPath);
      write_inc(outPath + ".inc");
    } else {
      printf("tune: held-out error did not improve; reverting this sweep and stopping\n");
      break;
    }
    if (changed == 0) break;
  }
  std::copy(bestParams.begin(), bestParams.end(), P_);
  eval_save_params(outPath);
  write_inc(outPath + ".inc");
  printf("tune: done. held-out error %.6f -> %.6f ; wrote %s and %s.inc\n", V0, bestVal, outPath.c_str(), outPath.c_str());
}

}  // namespace

bool tools_command(const std::string& cmd, std::istringstream& args, Engine& eng) {
  if (cmd == "datagen") {
    // datagen <positions> <threads> <depth> <outfile> [seed]
    long long n = 100000;
    int threads = 2, depth = 8;
    std::string path = "data/train.bin";
    uint64_t seed = 1;
    args >> n >> threads >> depth >> path >> seed;
    datagen(n, threads, depth, path, seed);
    return true;
  }
  if (cmd == "tune") {
    // tune <maxTrainPositions> <sweeps> <threads> <out.params> <maxAbsScore> <beta> <mu> <data1.bin> [data2.bin ...]
    std::string out, f;
    long long maxPos = 300000;
    int sweeps = 4, threads = 2, maxAbs = 1500;
    double beta = 0.5, mu = 5e-9;
    args >> maxPos >> sweeps >> threads >> out >> maxAbs >> beta >> mu;
    std::vector<std::string> files;
    while (args >> f) files.push_back(f);
    tune(files, maxPos, sweeps, threads, out, maxAbs, beta, mu);
    return true;
  }
  if (cmd == "hceeval") {
    std::string in, out;
    args >> in >> out;
    hceeval(in, out);
    return true;
  }
  return false;
}

}  // namespace heretic
