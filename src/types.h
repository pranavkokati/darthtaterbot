// Heretic chess engine - core types
#pragma once
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace heretic {

typedef uint64_t Bitboard;
typedef uint64_t Key;
typedef int Value;
typedef int Square;
typedef uint16_t Move;

enum Color : int { WHITE = 0, BLACK = 1, COLOR_NB = 2 };
inline Color operator~(Color c) { return Color(c ^ 1); }

enum PieceType : int { NO_PT = 0, PAWN = 1, KNIGHT = 2, BISHOP = 3, ROOK = 4, QUEEN = 5, KING = 6, PT_NB = 7 };

// Piece = (color << 3) | type.  White: 1..6, Black: 9..14.
typedef int Piece;
constexpr Piece NO_PIECE = 0;
constexpr int PIECE_NB = 16;
constexpr Piece make_piece(Color c, PieceType pt) { return (c << 3) | pt; }
constexpr PieceType type_of(Piece p) { return PieceType(p & 7); }
constexpr Color color_of(Piece p) { return Color(p >> 3); }

constexpr Square SQ_A1 = 0, SQ_C1 = 2, SQ_D1 = 3, SQ_E1 = 4, SQ_F1 = 5, SQ_G1 = 6, SQ_H1 = 7;
constexpr Square SQ_A8 = 56, SQ_C8 = 58, SQ_D8 = 59, SQ_E8 = 60, SQ_F8 = 61, SQ_G8 = 62, SQ_H8 = 63;
constexpr Square SQ_NONE = 64;
constexpr int file_of(Square s) { return s & 7; }
constexpr int rank_of(Square s) { return s >> 3; }
constexpr Square make_square(int f, int r) { return r * 8 + f; }
constexpr int relative_rank(Color c, Square s) { return c == WHITE ? rank_of(s) : 7 - rank_of(s); }

// Castling rights bits
constexpr int WHITE_OO = 1, WHITE_OOO = 2, BLACK_OO = 4, BLACK_OOO = 8;

// Move encoding: from(6) | to(6) << 6 | flags(4) << 12
enum MoveFlag : int {
  MF_QUIET = 0, MF_DOUBLE = 1, MF_KCASTLE = 2, MF_QCASTLE = 3,
  MF_CAPTURE = 4, MF_EP = 5,
  MF_PROMO = 8,       // 8..11 = N,B,R,Q
  MF_PROMO_CAP = 12   // 12..15 = N,B,R,Q with capture
};
constexpr Move MOVE_NONE = 0;
constexpr Move make_move(Square f, Square t, int flags = 0) { return Move(f | (t << 6) | (flags << 12)); }
constexpr Square from_sq(Move m) { return m & 63; }
constexpr Square to_sq(Move m) { return (m >> 6) & 63; }
constexpr int flags_of(Move m) { return m >> 12; }
constexpr bool is_capture_flag(int f) { return f & 4; }
constexpr bool is_promo_flag(int f) { return f & 8; }
constexpr PieceType promo_type(Move m) { return PieceType((flags_of(m) & 3) + KNIGHT); }
inline bool is_capture(Move m) { return flags_of(m) & 4; }
inline bool is_promo(Move m) { return flags_of(m) & 8; }
inline bool is_castle(Move m) { int f = flags_of(m); return f == MF_KCASTLE || f == MF_QCASTLE; }
// "noisy" = changes material (capture or promotion)
inline bool is_noisy(Move m) { return flags_of(m) & 12; }

constexpr Value VALUE_ZERO = 0;
constexpr Value VALUE_DRAW = 0;
constexpr Value VALUE_MATE = 32000;
constexpr int MAX_PLY = 128;
constexpr Value VALUE_MATE_IN_MAX_PLY = VALUE_MATE - MAX_PLY;
constexpr Value VALUE_INFINITE = 32001;
constexpr Value VALUE_NONE = 32002;
constexpr Value VALUE_TB_WIN = VALUE_MATE_IN_MAX_PLY - 1;  // unused; reserved

constexpr Value mate_in(int ply) { return VALUE_MATE - ply; }
constexpr Value mated_in(int ply) { return -VALUE_MATE + ply; }

struct ExtMove {
  Move move;
  int score;
};

inline std::string square_str(Square s) {
  std::string r;
  r += char('a' + file_of(s));
  r += char('1' + rank_of(s));
  return r;
}

inline std::string move_str(Move m) {
  if (m == MOVE_NONE) return "0000";
  std::string r = square_str(from_sq(m)) + square_str(to_sq(m));
  if (is_promo(m)) r += "nbrq"[promo_type(m) - KNIGHT];
  return r;
}

// xorshift64* PRNG (deterministic)
struct PRNG {
  uint64_t s;
  explicit PRNG(uint64_t seed) : s(seed) { assert(seed); }
  uint64_t rand64() {
    s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
    return s * 2685821657736338717ULL;
  }
  template <typename T> T rand() { return T(rand64()); }
  uint64_t sparse_rand() { return rand64() & rand64() & rand64(); }
  // uniform in [0, n)
  uint32_t below(uint32_t n) { return uint32_t((rand64() >> 32) * uint64_t(n) >> 32); }
  double uniform() { return (rand64() >> 11) * (1.0 / 9007199254740992.0); }
};

}  // namespace heretic
