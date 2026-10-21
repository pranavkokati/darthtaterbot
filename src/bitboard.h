// Bitboards, attack tables and magic bitboards
#pragma once
#include "types.h"

namespace heretic {

constexpr Bitboard FileABB = 0x0101010101010101ULL;
constexpr Bitboard FileHBB = FileABB << 7;
constexpr Bitboard Rank1BB = 0xFFULL;
constexpr Bitboard Rank2BB = Rank1BB << 8;
constexpr Bitboard Rank3BB = Rank1BB << 16;
constexpr Bitboard Rank4BB = Rank1BB << 24;
constexpr Bitboard Rank5BB = Rank1BB << 32;
constexpr Bitboard Rank6BB = Rank1BB << 40;
constexpr Bitboard Rank7BB = Rank1BB << 48;
constexpr Bitboard Rank8BB = Rank1BB << 56;

inline Bitboard sq_bb(Square s) { return 1ULL << s; }
inline int popcount(Bitboard b) { return __builtin_popcountll(b); }
inline Square lsb(Bitboard b) { return __builtin_ctzll(b); }
inline Square msb(Bitboard b) { return 63 ^ __builtin_clzll(b); }
inline Square pop_lsb(Bitboard& b) {
  Square s = lsb(b);
  b &= b - 1;
  return s;
}
inline bool more_than_one(Bitboard b) { return b & (b - 1); }
inline Bitboard file_bb(int f) { return FileABB << f; }
inline Bitboard rank_bb(int r) { return Rank1BB << (8 * r); }

enum Direction : int {
  NORTH = 8, SOUTH = -8, EAST = 1, WEST = -1,
  NORTH_EAST = 9, NORTH_WEST = 7, SOUTH_EAST = -7, SOUTH_WEST = -9
};

template <int D> constexpr Bitboard shift(Bitboard b) {
  return D == NORTH        ? b << 8
         : D == SOUTH      ? b >> 8
         : D == NORTH_EAST ? (b & ~FileHBB) << 9
         : D == NORTH_WEST ? (b & ~FileABB) << 7
         : D == SOUTH_EAST ? (b & ~FileHBB) >> 7
         : D == SOUTH_WEST ? (b & ~FileABB) >> 9
         : D == EAST       ? (b & ~FileHBB) << 1
         : D == WEST       ? (b & ~FileABB) >> 1
                           : 0;
}

struct Magic {
  Bitboard mask;
  Bitboard magic;
  Bitboard* attacks;
  unsigned shift;
};

extern Bitboard PawnAttacks[COLOR_NB][64];
extern Bitboard KnightAttacks[64];
extern Bitboard KingAttacks[64];
extern Bitboard BetweenBB[64][64];  // squares strictly between a and b if aligned, else 0
extern Bitboard LineBB[64][64];     // full line through a and b (including both) if aligned, else 0
extern Magic RookMagics[64];
extern Magic BishopMagics[64];

void bitboards_init();

inline Bitboard rook_attacks(Square s, Bitboard occ) {
  const Magic& m = RookMagics[s];
  return m.attacks[((occ & m.mask) * m.magic) >> m.shift];
}
inline Bitboard bishop_attacks(Square s, Bitboard occ) {
  const Magic& m = BishopMagics[s];
  return m.attacks[((occ & m.mask) * m.magic) >> m.shift];
}
inline Bitboard queen_attacks(Square s, Bitboard occ) { return rook_attacks(s, occ) | bishop_attacks(s, occ); }

inline Bitboard attacks_bb(PieceType pt, Square s, Bitboard occ) {
  switch (pt) {
    case KNIGHT: return KnightAttacks[s];
    case BISHOP: return bishop_attacks(s, occ);
    case ROOK: return rook_attacks(s, occ);
    case QUEEN: return queen_attacks(s, occ);
    case KING: return KingAttacks[s];
    default: return 0;
  }
}

inline bool aligned(Square a, Square b, Square c) { return LineBB[a][b] & sq_bb(c); }

}  // namespace heretic
