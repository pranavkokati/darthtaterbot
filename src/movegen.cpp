#include "movegen.h"

namespace heretic {

namespace {

inline void add(ExtMove*& out, Move m) {
  out->move = m;
  out->score = 0;
  ++out;
}

template <Color Us>
void gen_pawns(const Position& pos, ExtMove*& out, GenType type, Bitboard targetMask) {
  constexpr Color Them = Us == WHITE ? BLACK : WHITE;
  constexpr int Up = Us == WHITE ? 8 : -8;
  constexpr Bitboard Rank3 = Us == WHITE ? Rank3BB : Rank6BB;
  constexpr Bitboard Rank7 = Us == WHITE ? Rank7BB : Rank2BB;

  Bitboard pawns = pos.pieces(Us, PAWN);
  Bitboard enemies = pos.pieces(Them);
  Bitboard empty = ~pos.occupied();
  Bitboard promoting = pawns & Rank7;
  Bitboard normal = pawns & ~Rank7;

  // quiet pushes
  if (type == GEN_ALL) {
    Bitboard b1 = (Us == WHITE ? shift<NORTH>(normal) : shift<SOUTH>(normal)) & empty;
    Bitboard b2 = (Us == WHITE ? shift<NORTH>(b1 & Rank3) : shift<SOUTH>(b1 & Rank3)) & empty;
    b1 &= targetMask;
    b2 &= targetMask;
    while (b1) {
      Square to = pop_lsb(b1);
      add(out, make_move(to - Up, to, MF_QUIET));
    }
    while (b2) {
      Square to = pop_lsb(b2);
      add(out, make_move(to - 2 * Up, to, MF_DOUBLE));
    }
  }

  // promotions
  if (promoting) {
    Bitboard push = (Us == WHITE ? shift<NORTH>(promoting) : shift<SOUTH>(promoting)) & empty & targetMask;
    Bitboard capL = (Us == WHITE ? shift<NORTH_WEST>(promoting) : shift<SOUTH_WEST>(promoting)) & enemies & targetMask;
    Bitboard capR = (Us == WHITE ? shift<NORTH_EAST>(promoting) : shift<SOUTH_EAST>(promoting)) & enemies & targetMask;
    constexpr int dL = Us == WHITE ? 7 : -9;
    constexpr int dR = Us == WHITE ? 9 : -7;
    while (push) {
      Square to = pop_lsb(push);
      add(out, make_move(to - Up, to, MF_PROMO + 3));
      if (type == GEN_ALL) {
        add(out, make_move(to - Up, to, MF_PROMO + 2));
        add(out, make_move(to - Up, to, MF_PROMO + 1));
        add(out, make_move(to - Up, to, MF_PROMO + 0));
      }
    }
    while (capL) {
      Square to = pop_lsb(capL);
      add(out, make_move(to - dL, to, MF_PROMO_CAP + 3));
      if (type == GEN_ALL) {
        add(out, make_move(to - dL, to, MF_PROMO_CAP + 2));
        add(out, make_move(to - dL, to, MF_PROMO_CAP + 1));
        add(out, make_move(to - dL, to, MF_PROMO_CAP + 0));
      }
    }
    while (capR) {
      Square to = pop_lsb(capR);
      add(out, make_move(to - dR, to, MF_PROMO_CAP + 3));
      if (type == GEN_ALL) {
        add(out, make_move(to - dR, to, MF_PROMO_CAP + 2));
        add(out, make_move(to - dR, to, MF_PROMO_CAP + 1));
        add(out, make_move(to - dR, to, MF_PROMO_CAP + 0));
      }
    }
  }

  // captures
  Bitboard capL = (Us == WHITE ? shift<NORTH_WEST>(normal) : shift<SOUTH_WEST>(normal)) & enemies & targetMask;
  Bitboard capR = (Us == WHITE ? shift<NORTH_EAST>(normal) : shift<SOUTH_EAST>(normal)) & enemies & targetMask;
  constexpr int dL = Us == WHITE ? 7 : -9;
  constexpr int dR = Us == WHITE ? 9 : -7;
  while (capL) {
    Square to = pop_lsb(capL);
    add(out, make_move(to - dL, to, MF_CAPTURE));
  }
  while (capR) {
    Square to = pop_lsb(capR);
    add(out, make_move(to - dR, to, MF_CAPTURE));
  }

  // en passant (legality verified by pos.legal)
  Square ep = pos.ep_square();
  if (ep != SQ_NONE) {
    Bitboard attackers = PawnAttacks[Them][ep] & pos.pieces(Us, PAWN);
    while (attackers) {
      Square from = pop_lsb(attackers);
      add(out, make_move(from, ep, MF_EP));
    }
  }
}

void gen_piece_moves(const Position& pos, ExtMove*& out, Color us, PieceType pt, Bitboard targetMask, GenType type) {
  Bitboard bb = pos.pieces(us, pt);
  Bitboard occ = pos.occupied();
  Bitboard enemies = pos.pieces(~us);
  while (bb) {
    Square from = pop_lsb(bb);
    Bitboard att = attacks_bb(pt, from, occ) & targetMask;
    Bitboard caps = att & enemies;
    while (caps) add(out, make_move(from, pop_lsb(caps), MF_CAPTURE));
    if (type == GEN_ALL) {
      Bitboard quiets = att & ~occ;
      while (quiets) add(out, make_move(from, pop_lsb(quiets), MF_QUIET));
    }
  }
}

}  // namespace

int generate(const Position& pos, ExtMove* list, GenType type) {
  ExtMove* out = list;
  Color us = pos.side_to_move();
  Bitboard friends = pos.pieces(us);
  Bitboard targets = ~friends & ~pos.pieces(KING);  // never "capture" a king

  bool inCheck = pos.in_check();
  Bitboard checkers = pos.checkers();
  Bitboard evasionMask = ~0ULL;
  if (inCheck && !more_than_one(checkers)) {
    Square cs = lsb(checkers);
    evasionMask = BetweenBB[pos.king_square(us)][cs] | sq_bb(cs);
    // an en-passant capture can also remove a checking pawn: handled specially below
  }
  GenType effective = (inCheck) ? GEN_ALL : type;

  if (!(inCheck && more_than_one(checkers))) {
    Bitboard mask = targets & evasionMask;
    if (us == WHITE)
      gen_pawns<WHITE>(pos, out, effective, mask);
    else
      gen_pawns<BLACK>(pos, out, effective, mask);
    for (PieceType pt : {KNIGHT, BISHOP, ROOK, QUEEN}) gen_piece_moves(pos, out, us, pt, mask, effective);
  }
  // king moves (always)
  {
    Square ksq = pos.king_square(us);
    Bitboard att = KingAttacks[ksq] & targets;
    Bitboard caps = att & pos.pieces(~us);
    while (caps) add(out, make_move(ksq, pop_lsb(caps), MF_CAPTURE));
    if (effective == GEN_ALL) {
      Bitboard quiets = att & ~pos.occupied();
      while (quiets) add(out, make_move(ksq, pop_lsb(quiets), MF_QUIET));
      if (!inCheck) {
        int cr = pos.castling_rights();
        Bitboard occ = pos.occupied();
        if (us == WHITE) {
          if ((cr & WHITE_OO) && !(occ & (sq_bb(SQ_F1) | sq_bb(SQ_G1)))) add(out, make_move(SQ_E1, SQ_G1, MF_KCASTLE));
          if ((cr & WHITE_OOO) && !(occ & (sq_bb(SQ_D1) | sq_bb(SQ_C1) | sq_bb(1)))) add(out, make_move(SQ_E1, SQ_C1, MF_QCASTLE));
        } else {
          if ((cr & BLACK_OO) && !(occ & (sq_bb(SQ_F8) | sq_bb(SQ_G8)))) add(out, make_move(SQ_E8, SQ_G8, MF_KCASTLE));
          if ((cr & BLACK_OOO) && !(occ & (sq_bb(SQ_D8) | sq_bb(SQ_C8) | sq_bb(57)))) add(out, make_move(SQ_E8, SQ_C8, MF_QCASTLE));
        }
      }
    }
  }

  // Filter to fully legal moves.
  ExtMove* cur = list;
  for (ExtMove* it = list; it != out; ++it)
    if (pos.legal(it->move)) *cur++ = *it;
  return int(cur - list);
}

uint64_t perft(Position& pos, int depth) {
  ExtMove list[256];
  int n = generate(pos, list, GEN_ALL);
  if (depth == 1) return n;
  uint64_t nodes = 0;
  for (int i = 0; i < n; ++i) {
    pos.do_move(list[i].move);
    nodes += perft(pos, depth - 1);
    pos.undo_move(list[i].move);
  }
  return nodes;
}

Move parse_move(const Position& pos, const std::string& s) {
  if (s.size() < 4) return MOVE_NONE;
  MoveList ml(pos);
  for (int i = 0; i < ml.size(); ++i)
    if (move_str(ml.moves[i].move) == s) return ml.moves[i].move;
  return MOVE_NONE;
}

}  // namespace heretic
