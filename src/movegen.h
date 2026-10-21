// Legal move generation
#pragma once
#include "position.h"

namespace heretic {

enum GenType { GEN_ALL, GEN_NOISY };

// Fills `list` with fully legal moves; returns the count.
//  GEN_ALL   : every legal move
//  GEN_NOISY : captures, en passant, queen promotions (all legal evasions when in check)
int generate(const Position& pos, ExtMove* list, GenType type);

// perft utility
uint64_t perft(Position& pos, int depth);

// Parse a UCI move string against the legal moves of a position; MOVE_NONE if illegal.
Move parse_move(const Position& pos, const std::string& s);

struct MoveList {
  ExtMove moves[256];
  int count;
  MoveList(const Position& pos, GenType t = GEN_ALL) { count = generate(pos, moves, t); }
  ExtMove* begin() { return moves; }
  ExtMove* end() { return moves + count; }
  int size() const { return count; }
};

}  // namespace heretic
