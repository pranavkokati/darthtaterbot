#include "bitboard.h"

namespace heretic {

Bitboard PawnAttacks[COLOR_NB][64];
Bitboard KnightAttacks[64];
Bitboard KingAttacks[64];
Bitboard BetweenBB[64][64];
Bitboard LineBB[64][64];
Magic RookMagics[64];
Magic BishopMagics[64];

namespace {

Bitboard RookTable[0x19000];
Bitboard BishopTable[0x1480];

Bitboard sliding_attack(PieceType pt, Square sq, Bitboard occupied) {
  static const int rd[4][2] = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
  static const int bd[4][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
  Bitboard attacks = 0;
  for (int d = 0; d < 4; ++d) {
    int df = pt == ROOK ? rd[d][0] : bd[d][0];
    int dr = pt == ROOK ? rd[d][1] : bd[d][1];
    int f = file_of(sq) + df, r = rank_of(sq) + dr;
    while (f >= 0 && f < 8 && r >= 0 && r < 8) {
      Square s = make_square(f, r);
      attacks |= sq_bb(s);
      if (occupied & sq_bb(s)) break;
      f += df;
      r += dr;
    }
  }
  return attacks;
}

void init_magics(PieceType pt, Bitboard table[], Magic magics[]) {
  // Per-rank seeds known to find magics quickly with sparse random numbers.
  static const int seeds[8] = {728, 10316, 55013, 32803, 12281, 15100, 16645, 255};
  static Bitboard occupancy[4096], reference[4096];
  int epoch[4096] = {}, cnt = 0, size = 0;
  for (Square s = 0; s < 64; ++s) {
    Bitboard edges = ((Rank1BB | Rank8BB) & ~rank_bb(rank_of(s))) | ((FileABB | FileHBB) & ~file_bb(file_of(s)));
    Magic& m = magics[s];
    m.mask = sliding_attack(pt, s, 0) & ~edges;
    m.shift = 64 - popcount(m.mask);
    m.attacks = s == 0 ? table : magics[s - 1].attacks + size;
    Bitboard b = 0;
    size = 0;
    do {
      occupancy[size] = b;
      reference[size] = sliding_attack(pt, s, b);
      ++size;
      b = (b - m.mask) & m.mask;
    } while (b);
    PRNG rng(seeds[rank_of(s)]);
    for (int i = 0; i < size;) {
      for (m.magic = 0; popcount((m.magic * m.mask) >> 56) < 6;) m.magic = rng.sparse_rand();
      for (++cnt, i = 0; i < size; ++i) {
        unsigned idx = unsigned(((occupancy[i] & m.mask) * m.magic) >> m.shift);
        if (epoch[idx] < cnt) {
          epoch[idx] = cnt;
          m.attacks[idx] = reference[i];
        } else if (m.attacks[idx] != reference[i])
          break;
      }
    }
  }
}

}  // namespace

void bitboards_init() {
  static bool done = false;
  if (done) return;
  done = true;
  init_magics(ROOK, RookTable, RookMagics);
  init_magics(BISHOP, BishopTable, BishopMagics);

  for (Square s = 0; s < 64; ++s) {
    Bitboard b = sq_bb(s);
    PawnAttacks[WHITE][s] = shift<NORTH_EAST>(b) | shift<NORTH_WEST>(b);
    PawnAttacks[BLACK][s] = shift<SOUTH_EAST>(b) | shift<SOUTH_WEST>(b);
    KnightAttacks[s] = 0;
    KingAttacks[s] = 0;
    static const int kn[8][2] = {{1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2}};
    for (auto& d : kn) {
      int f = file_of(s) + d[0], r = rank_of(s) + d[1];
      if (f >= 0 && f < 8 && r >= 0 && r < 8) KnightAttacks[s] |= sq_bb(make_square(f, r));
    }
    for (int df = -1; df <= 1; ++df)
      for (int dr = -1; dr <= 1; ++dr) {
        if (!df && !dr) continue;
        int f = file_of(s) + df, r = rank_of(s) + dr;
        if (f >= 0 && f < 8 && r >= 0 && r < 8) KingAttacks[s] |= sq_bb(make_square(f, r));
      }
  }

  for (Square a = 0; a < 64; ++a)
    for (Square b = 0; b < 64; ++b) {
      BetweenBB[a][b] = 0;
      LineBB[a][b] = 0;
      if (a == b) continue;
      if (rook_attacks(a, 0) & sq_bb(b)) {
        BetweenBB[a][b] = rook_attacks(a, sq_bb(b)) & rook_attacks(b, sq_bb(a));
        LineBB[a][b] = (rook_attacks(a, 0) & rook_attacks(b, 0)) | sq_bb(a) | sq_bb(b);
      } else if (bishop_attacks(a, 0) & sq_bb(b)) {
        BetweenBB[a][b] = bishop_attacks(a, sq_bb(b)) & bishop_attacks(b, sq_bb(a));
        LineBB[a][b] = (bishop_attacks(a, 0) & bishop_attacks(b, 0)) | sq_bb(a) | sq_bb(b);
      }
    }
}

}  // namespace heretic
