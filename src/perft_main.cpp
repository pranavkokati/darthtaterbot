// Temporary validation driver: perft + movegen fuzz
#include <chrono>
#include <iostream>

#include "movegen.h"

using namespace heretic;

int main(int argc, char** argv) {
  bitboards_init();
  Zobrist::init();
  std::string cmd = argc > 1 ? argv[1] : "perft";
  if (cmd == "perft") {
    int depth = argc > 2 ? atoi(argv[2]) : 5;
    std::string fen = argc > 3 ? argv[3] : Position::StartFEN;
    Position pos;
    pos.set_fen(fen);
    auto t0 = std::chrono::steady_clock::now();
    uint64_t n = perft(pos, depth);
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("perft(%d) = %llu  (%.2fs, %.1f Mnps)\n", depth, (unsigned long long)n, s, n / s / 1e6);
  } else if (cmd == "fuzz") {
    int games = argc > 2 ? atoi(argv[2]) : 200;
    PRNG rng(12345);
    uint64_t positions = 0, checksSeen = 0, epSeen = 0, castleSeen = 0, promoSeen = 0, errors = 0;
    for (int g = 0; g < games; ++g) {
      Position pos;
      for (int ply = 0; ply < 300; ++ply) {
        MoveList ml(pos);
        if (ml.size() == 0) break;
        ++positions;
        std::string fen0 = pos.fen();
        Key key0 = pos.key();
        // key consistency from scratch
        {
          Position p2;
          p2.set_fen(fen0);
          if (p2.key() != key0 || p2.pawn_key() != pos.pawn_key() ||
              p2.non_pawn_key(WHITE) != pos.non_pawn_key(WHITE) || p2.non_pawn_key(BLACK) != pos.non_pawn_key(BLACK)) {
            printf("KEY MISMATCH at %s\n", fen0.c_str());
            ++errors;
          }
        }
        for (auto& em : ml) {
          Move m = em.move;
          if (!pos.pseudo_legal(m) || !pos.legal(m)) {
            printf("legal move rejected: %s in %s\n", move_str(m).c_str(), fen0.c_str());
            ++errors;
          }
          bool gc = pos.gives_check(m);
          pos.do_move(m);
          if (gc != pos.in_check()) {
            printf("gives_check mismatch %s in %s (pred %d actual %d)\n", move_str(m).c_str(), fen0.c_str(), gc, pos.in_check());
            ++errors;
          }
          checksSeen += pos.in_check();
          if (flags_of(m) == MF_EP) ++epSeen;
          if (is_castle(m)) ++castleSeen;
          if (is_promo(m)) ++promoSeen;
          pos.undo_move(m);
          if (pos.key() != key0 || pos.fen() != fen0) {
            printf("UNDO MISMATCH %s in %s -> %s\n", move_str(m).c_str(), fen0.c_str(), pos.fen().c_str());
            ++errors;
          }
        }
        // random 16-bit moves: accepted by pseudo_legal && legal must be a generated legal move
        for (int i = 0; i < 3000; ++i) {
          Move m = Move(rng.rand64() & 0xFFFF);
          if (pos.pseudo_legal(m) && pos.legal(m)) {
            bool found = false;
            for (auto& em : ml)
              if (em.move == m) found = true;
            if (!found) {
              printf("PHANTOM move accepted: %s (flags %d) in %s\n", move_str(m).c_str(), flags_of(m), fen0.c_str());
              ++errors;
            }
          }
        }
        // play a random move (bias toward captures to reach endgames)
        Move pick = ml.moves[rng.below(ml.size())].move;
        for (int tries = 0; tries < 3; ++tries) {
          Move c = ml.moves[rng.below(ml.size())].move;
          if (is_capture(c)) { pick = c; break; }
        }
        pos.do_move(pick);
        if (pos.is_draw(0) && ply > 40) break;
      }
      if (errors > 20) break;
    }
    printf("fuzz: %llu positions, %llu checks, %llu ep, %llu castles, %llu promos, %llu ERRORS\n",
           (unsigned long long)positions, (unsigned long long)checksSeen, (unsigned long long)epSeen,
           (unsigned long long)castleSeen, (unsigned long long)promoSeen, (unsigned long long)errors);
    return errors ? 1 : 0;
  } else if (cmd == "divide") {
    int depth = atoi(argv[2]);
    Position pos;
    pos.set_fen(argv[3]);
    MoveList ml(pos);
    uint64_t total = 0;
    for (auto& em : ml) {
      pos.do_move(em.move);
      uint64_t n = depth > 1 ? perft(pos, depth - 1) : 1;
      pos.undo_move(em.move);
      printf("%s: %llu\n", move_str(em.move).c_str(), (unsigned long long)n);
      total += n;
    }
    printf("total %llu\n", (unsigned long long)total);
  }
  return 0;
}
