#include "position.h"

#include <cctype>
#include <sstream>

namespace heretic {

const char* Position::StartFEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

namespace Zobrist {
Key psq[PIECE_NB][64];
Key castling[16];
Key epFile[8];
Key side;

void init() {
  static bool done = false;
  if (done) return;
  done = true;
  PRNG rng(1070372);
  for (int p = 0; p < PIECE_NB; ++p)
    for (int s = 0; s < 64; ++s) psq[p][s] = rng.rand64();
  for (int c = 0; c < 16; ++c) castling[c] = rng.rand64();
  for (int f = 0; f < 8; ++f) epFile[f] = rng.rand64();
  side = rng.rand64();
}
}  // namespace Zobrist

namespace {
int CastlingMask[64];
struct InitTables {
  InitTables() {
    for (int i = 0; i < 64; ++i) CastlingMask[i] = 15;
    CastlingMask[SQ_A1] &= ~WHITE_OOO;
    CastlingMask[SQ_E1] &= ~(WHITE_OO | WHITE_OOO);
    CastlingMask[SQ_H1] &= ~WHITE_OO;
    CastlingMask[SQ_A8] &= ~BLACK_OOO;
    CastlingMask[SQ_E8] &= ~(BLACK_OO | BLACK_OOO);
    CastlingMask[SQ_H8] &= ~BLACK_OO;
  }
} initTables;
}  // namespace

void Position::compute_keys(StateInfo& s) const {
  s.key = 0;
  s.pawnKey = 0;
  s.nonPawnKey[0] = s.nonPawnKey[1] = 0;
  for (Square sq = 0; sq < 64; ++sq)
    if (board[sq]) hash_piece(s, board[sq], sq);
  if (stm == BLACK) s.key ^= Zobrist::side;
  s.key ^= Zobrist::castling[s.castling];
  if (s.ep != SQ_NONE) s.key ^= Zobrist::epFile[file_of(s.ep)];
}

void Position::set_check_info(StateInfo& s) const {
  for (Color c : {WHITE, BLACK}) {
    Square ksq = king_square(c);
    Bitboard snipers = ((rook_attacks(ksq, 0) & (pieces(ROOK) | pieces(QUEEN))) |
                        (bishop_attacks(ksq, 0) & (pieces(BISHOP) | pieces(QUEEN)))) &
                       pieces(~c);
    Bitboard blockers = 0;
    Bitboard occ = occupied() ^ snipers;
    while (snipers) {
      Square sn = pop_lsb(snipers);
      Bitboard b = BetweenBB[ksq][sn] & occupied();
      if (b && !more_than_one(b)) blockers |= b;
    }
    (void)occ;
    s.blockers[c] = blockers;
  }
}

void Position::set_fen(const std::string& fen) {
  std::memset(board, 0, sizeof(board));
  std::memset(byType, 0, sizeof(byType));
  std::memset(byColor, 0, sizeof(byColor));
  if (states.size() < 1024) states.resize(1024);
  sp = 0;
  StateInfo& s = states[0];
  std::memset(&s, 0, sizeof(s));
  s.ep = SQ_NONE;

  std::istringstream ss(fen);
  std::string placement, side, castle, ep;
  int hm = 0, fm = 1;
  ss >> placement >> side >> castle >> ep;
  if (!(ss >> hm)) hm = 0;
  if (!(ss >> fm)) fm = 1;

  int rank = 7, file = 0;
  for (char c : placement) {
    if (c == '/') {
      --rank;
      file = 0;
    } else if (c >= '1' && c <= '8') {
      file += c - '0';
    } else {
      static const std::string pcs = "PNBRQK";
      size_t idx = pcs.find(std::toupper(c));
      if (idx == std::string::npos || rank < 0 || file > 7) continue;
      Color col = std::isupper(c) ? WHITE : BLACK;
      put_raw(make_piece(col, PieceType(idx + 1)), make_square(file, rank));
      ++file;
    }
  }
  stm = (side == "b") ? BLACK : WHITE;
  s.castling = 0;
  for (char c : castle) {
    if (c == 'K') s.castling |= WHITE_OO;
    if (c == 'Q') s.castling |= WHITE_OOO;
    if (c == 'k') s.castling |= BLACK_OO;
    if (c == 'q') s.castling |= BLACK_OOO;
  }
  s.ep = SQ_NONE;
  if (ep.size() == 2 && ep[0] >= 'a' && ep[0] <= 'h' && ep[1] >= '1' && ep[1] <= '8') {
    Square e = make_square(ep[0] - 'a', ep[1] - '1');
    // only keep the ep square if an enemy-to-move pawn can actually capture
    if (PawnAttacks[~stm][e] & pieces(stm, PAWN)) s.ep = e;
  }
  s.rule50 = hm;
  s.pliesFromNull = hm;
  gamePly = std::max(2 * (fm - 1), 0) + (stm == BLACK);
  s.captured = NO_PIECE;
  s.move = MOVE_NONE;
  s.dp.n = 0;
  compute_keys(s);
  s.checkers = attackers_to(king_square(stm), occupied()) & pieces(~stm);
  set_check_info(s);
}

void Position::set_board(const Piece b[64], Color side, int castling, Square ep) {
  std::memset(board, 0, sizeof(board));
  std::memset(byType, 0, sizeof(byType));
  std::memset(byColor, 0, sizeof(byColor));
  if (states.size() < 1024) states.resize(1024);
  sp = 0;
  StateInfo& s = states[0];
  std::memset(&s, 0, sizeof(s));
  for (int sq = 0; sq < 64; ++sq)
    if (b[sq]) put_raw(b[sq], sq);
  stm = side;
  s.castling = castling;
  s.ep = ep;
  gamePly = 0;
  s.captured = NO_PIECE;
  s.move = MOVE_NONE;
  compute_keys(s);
  s.checkers = attackers_to(king_square(stm), occupied()) & pieces(~stm);
  set_check_info(s);
}

std::string Position::fen() const {
  std::string r;
  for (int rank = 7; rank >= 0; --rank) {
    int empty = 0;
    for (int file = 0; file < 8; ++file) {
      Piece p = board[make_square(file, rank)];
      if (!p) {
        ++empty;
        continue;
      }
      if (empty) r += char('0' + empty), empty = 0;
      char c = " PNBRQK"[type_of(p)];
      r += color_of(p) == WHITE ? c : char(std::tolower(c));
    }
    if (empty) r += char('0' + empty);
    if (rank) r += '/';
  }
  r += stm == WHITE ? " w " : " b ";
  int cr = st().castling;
  if (!cr)
    r += '-';
  else {
    if (cr & WHITE_OO) r += 'K';
    if (cr & WHITE_OOO) r += 'Q';
    if (cr & BLACK_OO) r += 'k';
    if (cr & BLACK_OOO) r += 'q';
  }
  r += ' ';
  r += st().ep == SQ_NONE ? "-" : square_str(st().ep);
  r += ' ' + std::to_string(st().rule50) + ' ' + std::to_string(gamePly / 2 + 1);
  return r;
}

Bitboard Position::attackers_to(Square s, Bitboard occ) const {
  return (PawnAttacks[BLACK][s] & pieces(WHITE, PAWN)) | (PawnAttacks[WHITE][s] & pieces(BLACK, PAWN)) |
         (KnightAttacks[s] & pieces(KNIGHT)) | (rook_attacks(s, occ) & (pieces(ROOK) | pieces(QUEEN))) |
         (bishop_attacks(s, occ) & (pieces(BISHOP) | pieces(QUEEN))) | (KingAttacks[s] & pieces(KING));
}

bool Position::pseudo_legal(Move m) const {
  if (m == MOVE_NONE) return false;
  Color us = stm;
  Square from = from_sq(m), to = to_sq(m);
  int f = flags_of(m);
  if (f == 6 || f == 7) return false;  // undefined flag values
  Piece pc = board[from];
  if (pc == NO_PIECE || color_of(pc) != us) return false;
  PieceType pt = type_of(pc);
  Piece tp = board[to];

  if (f == MF_EP) {
    return pt == PAWN && to == st().ep && tp == NO_PIECE && (PawnAttacks[us][from] & sq_bb(to));
  }
  if (f & 4) {
    if (tp == NO_PIECE || color_of(tp) == us || type_of(tp) == KING) return false;
  } else if (tp != NO_PIECE)
    return false;

  if (f == MF_KCASTLE || f == MF_QCASTLE) {
    if (pt != KING) return false;
    Square ksq = us == WHITE ? SQ_E1 : SQ_E8;
    if (from != ksq) return false;
    int cr = st().castling;
    if (f == MF_KCASTLE) {
      if (!(cr & (us == WHITE ? WHITE_OO : BLACK_OO)) || to != ksq + 2) return false;
      if (occupied() & (sq_bb(ksq + 1) | sq_bb(ksq + 2))) return false;
    } else {
      if (!(cr & (us == WHITE ? WHITE_OOO : BLACK_OOO)) || to != ksq - 2) return false;
      if (occupied() & (sq_bb(ksq - 1) | sq_bb(ksq - 2) | sq_bb(ksq - 3))) return false;
    }
    return true;
  }

  if (pt == PAWN) {
    Bitboard promoRank = us == WHITE ? Rank8BB : Rank1BB;
    bool isPromo = f & 8;
    if (isPromo != bool(sq_bb(to) & promoRank)) return false;
    int up = us == WHITE ? 8 : -8;
    if (f & 4) return PawnAttacks[us][from] & sq_bb(to);
    if (f == MF_DOUBLE)
      return to == from + 2 * up && rank_of(from) == (us == WHITE ? 1 : 6) && board[from + up] == NO_PIECE;
    return to == from + up;
  }
  if (f != MF_QUIET && f != MF_CAPTURE) return false;
  return attacks_bb(pt, from, occupied()) & sq_bb(to);
}

bool Position::legal(Move m) const {
  Color us = stm;
  Square from = from_sq(m), to = to_sq(m);
  int f = flags_of(m);
  Square ksq = king_square(us);
  const StateInfo& s = st();

  if (f == MF_EP) {
    Square capsq = us == WHITE ? to - 8 : to + 8;
    Bitboard occ = (occupied() ^ sq_bb(from) ^ sq_bb(capsq)) | sq_bb(to);
    return !(attackers_to(ksq, occ) & pieces(~us) & ~sq_bb(capsq));
  }
  if (f == MF_KCASTLE || f == MF_QCASTLE) {
    if (s.checkers) return false;
    int step = f == MF_KCASTLE ? 1 : -1;
    // squares the king crosses/lands on must not be attacked
    for (int i = 1; i <= 2; ++i)
      if (attacked_by(from + step * i, ~us, occupied())) return false;
    return true;
  }
  if (type_of(board[from]) == KING) return !attacked_by(to, ~us, occupied() ^ sq_bb(from));

  if (s.checkers) {
    if (more_than_one(s.checkers)) return false;
    Square cs = lsb(s.checkers);
    if (!((BetweenBB[ksq][cs] | sq_bb(cs)) & sq_bb(to))) return false;
  }
  if ((s.blockers[us] & pieces(us)) & sq_bb(from)) return aligned(from, to, ksq);
  return true;
}

bool Position::gives_check(Move m) const {
  Color us = stm, them = ~us;
  Square from = from_sq(m), to = to_sq(m);
  int f = flags_of(m);
  Square ksq = king_square(them);
  Piece pc = board[from];
  PieceType pt = type_of(pc);

  Bitboard occ = occupied() ^ sq_bb(from);
  Bitboard n = pieces(us, KNIGHT), p = pieces(us, PAWN), bq = pieces(us, BISHOP) | pieces(us, QUEEN),
           rq = pieces(us, ROOK) | pieces(us, QUEEN);
  auto removeFrom = [&](PieceType t) {
    if (t == PAWN) p &= ~sq_bb(from);
    else if (t == KNIGHT) n &= ~sq_bb(from);
    else if (t == BISHOP) bq &= ~sq_bb(from);
    else if (t == ROOK) rq &= ~sq_bb(from);
    else if (t == QUEEN) { bq &= ~sq_bb(from); rq &= ~sq_bb(from); }
  };
  auto addTo = [&](PieceType t, Square s) {
    if (t == PAWN) p |= sq_bb(s);
    else if (t == KNIGHT) n |= sq_bb(s);
    else if (t == BISHOP) bq |= sq_bb(s);
    else if (t == ROOK) rq |= sq_bb(s);
    else if (t == QUEEN) { bq |= sq_bb(s); rq |= sq_bb(s); }
  };
  removeFrom(pt);
  occ |= sq_bb(to);
  PieceType landed = pt;
  if (is_promo(m)) landed = promo_type(m);
  addTo(landed, to);
  if (f == MF_EP) {
    Square capsq = us == WHITE ? to - 8 : to + 8;
    occ ^= sq_bb(capsq);
  } else if (f == MF_KCASTLE) {
    Square rf = us == WHITE ? SQ_H1 : SQ_H8, rt = us == WHITE ? SQ_F1 : SQ_F8;
    occ ^= sq_bb(rf) | sq_bb(rt);
    rq &= ~sq_bb(rf);
    rq |= sq_bb(rt);
  } else if (f == MF_QCASTLE) {
    Square rf = us == WHITE ? SQ_A1 : SQ_A8, rt = us == WHITE ? SQ_D1 : SQ_D8;
    occ ^= sq_bb(rf) | sq_bb(rt);
    rq &= ~sq_bb(rf);
    rq |= sq_bb(rt);
  }
  if (KnightAttacks[ksq] & n) return true;
  if (PawnAttacks[them][ksq] & p) return true;
  if (bishop_attacks(ksq, occ) & bq) return true;
  if (rook_attacks(ksq, occ) & rq) return true;
  return false;
}

bool Position::see_ge(Move m, int threshold) const {
  int f = flags_of(m);
  if (is_promo_flag(f) || f == MF_KCASTLE || f == MF_QCASTLE) return 0 >= threshold;
  Square from = from_sq(m), to = to_sq(m);
  Bitboard occ = occupied();
  int captured = (f == MF_EP) ? SeeValue[PAWN] : SeeValue[type_of(board[to])];
  int swap = captured - threshold;
  if (swap < 0) return false;
  swap = SeeValue[type_of(board[from])] - swap;
  if (swap <= 0) return true;

  occ ^= sq_bb(from) ^ sq_bb(to);
  if (f == MF_EP) occ ^= sq_bb(stm == WHITE ? to - 8 : to + 8);
  Color side = stm;
  Bitboard attackers = attackers_to(to, occ) & occ;
  int res = 1;
  while (true) {
    side = ~side;
    attackers &= occ;
    Bitboard sa = attackers & pieces(side);
    if (!sa) break;
    // pinned pieces cannot recapture if the pinner is still on the board
    if (st().blockers[~side] & occ) {
      // approximate: only treat pins against the original king configuration
      Bitboard pinnersExist = 0;
      (void)pinnersExist;
    }
    res ^= 1;
    Bitboard bb;
    if ((bb = sa & pieces(PAWN))) {
      if ((swap = SeeValue[PAWN] - swap) < res) break;
      occ ^= bb & -bb;
      attackers |= bishop_attacks(to, occ) & (pieces(BISHOP) | pieces(QUEEN));
    } else if ((bb = sa & pieces(KNIGHT))) {
      if ((swap = SeeValue[KNIGHT] - swap) < res) break;
      occ ^= bb & -bb;
    } else if ((bb = sa & pieces(BISHOP))) {
      if ((swap = SeeValue[BISHOP] - swap) < res) break;
      occ ^= bb & -bb;
      attackers |= bishop_attacks(to, occ) & (pieces(BISHOP) | pieces(QUEEN));
    } else if ((bb = sa & pieces(ROOK))) {
      if ((swap = SeeValue[ROOK] - swap) < res) break;
      occ ^= bb & -bb;
      attackers |= rook_attacks(to, occ) & (pieces(ROOK) | pieces(QUEEN));
    } else if ((bb = sa & pieces(QUEEN))) {
      if ((swap = SeeValue[QUEEN] - swap) < res) break;
      occ ^= bb & -bb;
      attackers |= (bishop_attacks(to, occ) & (pieces(BISHOP) | pieces(QUEEN))) |
                   (rook_attacks(to, occ) & (pieces(ROOK) | pieces(QUEEN)));
    } else {
      return (attackers & ~pieces(side)) ? res ^ 1 : res;
    }
  }
  return bool(res);
}

void Position::do_move(Move m) {
  if ((size_t)sp + 2 >= states.size()) states.resize(states.size() * 2);
  const StateInfo& o = states[sp];
  StateInfo& n = states[sp + 1];
  n = o;

  Color us = stm, them = ~us;
  Square from = from_sq(m), to = to_sq(m);
  int f = flags_of(m);
  Piece pc = board[from];
  PieceType pt = type_of(pc);
  Piece captured = (f == MF_EP) ? make_piece(them, PAWN) : board[to];

  n.dp.n = 0;
  n.captured = captured;
  n.move = m;
  n.rule50 = o.rule50 + 1;
  n.pliesFromNull = o.pliesFromNull + 1;
  n.key ^= Zobrist::side;
  if (o.ep != SQ_NONE) n.key ^= Zobrist::epFile[file_of(o.ep)];
  n.ep = SQ_NONE;

  int newCr = o.castling & CastlingMask[from] & CastlingMask[to];
  n.key ^= Zobrist::castling[o.castling] ^ Zobrist::castling[newCr];
  n.castling = newCr;

  if (captured) {
    Square capsq = (f == MF_EP) ? (us == WHITE ? to - 8 : to + 8) : to;
    remove_raw(capsq);
    hash_piece(n, captured, capsq);
    n.rule50 = 0;
    n.dp.pc[n.dp.n] = captured;
    n.dp.from[n.dp.n] = capsq;
    n.dp.to[n.dp.n] = SQ_NONE;
    ++n.dp.n;
  }

  if (is_promo_flag(f)) {
    Piece pp = make_piece(us, promo_type(m));
    remove_raw(from);
    put_raw(pp, to);
    hash_piece(n, pc, from);
    hash_piece(n, pp, to);
    n.dp.pc[n.dp.n] = pc;
    n.dp.from[n.dp.n] = from;
    n.dp.to[n.dp.n] = SQ_NONE;
    ++n.dp.n;
    n.dp.pc[n.dp.n] = pp;
    n.dp.from[n.dp.n] = SQ_NONE;
    n.dp.to[n.dp.n] = to;
    ++n.dp.n;
  } else {
    move_raw(from, to);
    hash_piece(n, pc, from);
    hash_piece(n, pc, to);
    n.dp.pc[n.dp.n] = pc;
    n.dp.from[n.dp.n] = from;
    n.dp.to[n.dp.n] = to;
    ++n.dp.n;
  }

  if (f == MF_KCASTLE || f == MF_QCASTLE) {
    Square rf, rt;
    if (f == MF_KCASTLE) {
      rf = us == WHITE ? SQ_H1 : SQ_H8;
      rt = us == WHITE ? SQ_F1 : SQ_F8;
    } else {
      rf = us == WHITE ? SQ_A1 : SQ_A8;
      rt = us == WHITE ? SQ_D1 : SQ_D8;
    }
    Piece rook = make_piece(us, ROOK);
    move_raw(rf, rt);
    hash_piece(n, rook, rf);
    hash_piece(n, rook, rt);
    n.dp.pc[n.dp.n] = rook;
    n.dp.from[n.dp.n] = rf;
    n.dp.to[n.dp.n] = rt;
    ++n.dp.n;
  }

  if (pt == PAWN) {
    n.rule50 = 0;
    if (f == MF_DOUBLE) {
      Square e = (from + to) / 2;
      if (PawnAttacks[us][e] & pieces(them, PAWN)) {
        n.ep = e;
        n.key ^= Zobrist::epFile[file_of(e)];
      }
    }
  }

  stm = them;
  ++sp;
  ++gamePly;
  n.checkers = attackers_to(king_square(them), occupied()) & pieces(us);
  set_check_info(n);
}

void Position::undo_move(Move m) {
  const StateInfo& n = states[sp];
  Color them = stm;  // side that was NOT the mover == side to move now
  Color us = ~them;
  Square from = from_sq(m), to = to_sq(m);
  int f = flags_of(m);

  if (f == MF_KCASTLE || f == MF_QCASTLE) {
    Square rf, rt;
    if (f == MF_KCASTLE) {
      rf = us == WHITE ? SQ_H1 : SQ_H8;
      rt = us == WHITE ? SQ_F1 : SQ_F8;
    } else {
      rf = us == WHITE ? SQ_A1 : SQ_A8;
      rt = us == WHITE ? SQ_D1 : SQ_D8;
    }
    move_raw(rt, rf);
  }
  if (is_promo_flag(f)) {
    remove_raw(to);
    put_raw(make_piece(us, PAWN), from);
  } else {
    move_raw(to, from);
  }
  if (n.captured) {
    Square capsq = (f == MF_EP) ? (us == WHITE ? to - 8 : to + 8) : to;
    put_raw(n.captured, capsq);
  }
  stm = us;
  --sp;
  --gamePly;
}

void Position::do_null() {
  if ((size_t)sp + 2 >= states.size()) states.resize(states.size() * 2);
  const StateInfo& o = states[sp];
  StateInfo& n = states[sp + 1];
  n = o;
  n.key ^= Zobrist::side;
  if (o.ep != SQ_NONE) n.key ^= Zobrist::epFile[file_of(o.ep)];
  n.ep = SQ_NONE;
  n.rule50 = o.rule50 + 1;
  n.pliesFromNull = 0;
  n.captured = NO_PIECE;
  n.move = MOVE_NONE;
  n.dp.n = 0;
  stm = ~stm;
  ++sp;
  ++gamePly;
  n.checkers = 0;  // null move is never played in check
  set_check_info(n);
}

void Position::undo_null() {
  stm = ~stm;
  --sp;
  --gamePly;
}

bool Position::insufficient_material() const {
  if (pieces(PAWN) | pieces(ROOK) | pieces(QUEEN)) return false;
  int minors = popcount(pieces(KNIGHT) | pieces(BISHOP));
  return minors <= 1;
}

int Position::repetition_count() const {
  const StateInfo& s = st();
  int end = std::min(s.rule50, s.pliesFromNull);
  int cnt = 0;
  for (int i = 4; i <= end && i <= sp; i += 2)
    if (states[sp - i].key == s.key) ++cnt;
  return cnt;
}

bool Position::is_draw(int ply) const {
  const StateInfo& s = st();
  if (s.rule50 > 99) return true;  // callers handle the checkmate-on-move-100 corner case
  int end = std::min(s.rule50, s.pliesFromNull);
  int beforeRoot = 0;
  for (int i = 4; i <= end && i <= sp; i += 2) {
    if (states[sp - i].key == s.key) {
      if (i <= ply) return true;
      if (++beforeRoot >= 2) return true;
    }
  }
  return insufficient_material();
}

Key Position::key_after(Move m) const {
  Key k = st().key ^ Zobrist::side;
  if (m == MOVE_NONE) return k;
  Square from = from_sq(m), to = to_sq(m);
  Piece pc = board[from];
  k ^= Zobrist::psq[pc][from] ^ Zobrist::psq[pc][to];
  if (board[to]) k ^= Zobrist::psq[board[to]][to];
  return k;
}

}  // namespace heretic
