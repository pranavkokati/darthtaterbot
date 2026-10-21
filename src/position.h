// Board position: bitboards, make/unmake, hashing, legality, SEE
#pragma once
#include "bitboard.h"
#include "types.h"

namespace heretic {

namespace Zobrist {
extern Key psq[PIECE_NB][64];
extern Key castling[16];
extern Key epFile[8];
extern Key side;
void init();
}  // namespace Zobrist

// Pieces touched by the last move, for incremental NNUE accumulator updates.
struct DirtyPiece {
  int n;
  Piece pc[3];
  Square from[3];  // SQ_NONE if the piece was added
  Square to[3];    // SQ_NONE if the piece was removed
};

struct StateInfo {
  Key key;
  Key pawnKey;
  Key nonPawnKey[2];
  int castling;
  Square ep;
  int rule50;
  int pliesFromNull;
  Piece captured;
  Bitboard checkers;
  Bitboard blockers[2];  // pieces (either colour) blocking a slider attack on king of given colour
  Move move;
  DirtyPiece dp;
};

constexpr int SeeValue[PT_NB] = {0, 100, 320, 330, 500, 900, 0};

class Position {
 public:
  static const char* StartFEN;

  Position() {
    bitboards_init();  // idempotent; tables must exist before any attack lookup
    Zobrist::init();
    set_fen(StartFEN);
  }
  void set_fen(const std::string& fen);
  // Fast setup from a raw piece array (used by the tuner/trainer; skips history and key computation)
  void set_board(const Piece b[64], Color side, int castling, Square ep);
  std::string fen() const;

  Color side_to_move() const { return stm; }
  Piece piece_on(Square s) const { return board[s]; }
  Bitboard occupied() const { return byColor[WHITE] | byColor[BLACK]; }
  Bitboard pieces(PieceType pt) const { return byType[pt]; }
  Bitboard pieces(Color c) const { return byColor[c]; }
  Bitboard pieces(Color c, PieceType pt) const { return byType[pt] & byColor[c]; }
  Bitboard pieces(PieceType a, PieceType b) const { return byType[a] | byType[b]; }
  Square king_square(Color c) const { return lsb(pieces(c, KING)); }
  int count(Color c, PieceType pt) const { return popcount(pieces(c, pt)); }
  int castling_rights() const { return st().castling; }
  Square ep_square() const { return st().ep; }
  int rule50() const { return st().rule50; }
  int game_ply() const { return gamePly; }
  Key key() const { return st().key; }
  Key pawn_key() const { return st().pawnKey; }
  Key non_pawn_key(Color c) const { return st().nonPawnKey[c]; }
  Bitboard checkers() const { return st().checkers; }
  bool in_check() const { return st().checkers != 0; }
  const StateInfo& state() const { return st(); }
  const StateInfo& state_at(int back) const { return states[sp - back]; }
  int state_depth() const { return sp; }

  Bitboard attackers_to(Square s, Bitboard occ) const;
  Bitboard attackers_to(Square s) const { return attackers_to(s, occupied()); }
  bool attacked_by(Square s, Color by, Bitboard occ) const { return attackers_to(s, occ) & pieces(by); }

  bool pseudo_legal(Move m) const;
  bool legal(Move m) const;  // assumes pseudo_legal
  bool gives_check(Move m) const;
  bool see_ge(Move m, int threshold) const;

  void do_move(Move m);
  void undo_move(Move m);
  void do_null();
  void undo_null();

  bool is_draw(int ply) const;        // repetition, 50 move rule, insufficient material
  bool has_non_pawn_material(Color c) const { return pieces(c) & ~pieces(PAWN) & ~pieces(KING); }
  bool insufficient_material() const;
  Key key_after(Move m) const;  // approximate: hash after a move (for TT prefetch / loop detection)

  // How many times the current position occurred before (via same key in history)
  int repetition_count() const;

  Move last_move() const { return st().move; }
  void clear_history() {  // keep position, drop history before it
    states[0] = states[sp];
    sp = 0;
  }

 private:
  const StateInfo& st() const { return states[sp]; }
  StateInfo& st() { return states[sp]; }
  void compute_keys(StateInfo& s) const;
  void set_check_info(StateInfo& s) const;
  void put_raw(Piece pc, Square s) {
    board[s] = pc;
    byType[type_of(pc)] |= sq_bb(s);
    byColor[color_of(pc)] |= sq_bb(s);
  }
  void remove_raw(Square s) {
    Piece pc = board[s];
    board[s] = NO_PIECE;
    byType[type_of(pc)] &= ~sq_bb(s);
    byColor[color_of(pc)] &= ~sq_bb(s);
  }
  void move_raw(Square f, Square t) {
    Piece pc = board[f];
    Bitboard fb = sq_bb(f) | sq_bb(t);
    board[f] = NO_PIECE;
    board[t] = pc;
    byType[type_of(pc)] ^= fb;
    byColor[color_of(pc)] ^= fb;
  }
  void hash_piece(StateInfo& s, Piece pc, Square sq) const {
    Key k = Zobrist::psq[pc][sq];
    s.key ^= k;
    if (type_of(pc) == PAWN)
      s.pawnKey ^= k;
    else
      s.nonPawnKey[color_of(pc)] ^= k;
  }

  Piece board[64];
  Bitboard byType[PT_NB];
  Bitboard byColor[COLOR_NB];
  Color stm;
  int gamePly;
  std::vector<StateInfo> states;
  int sp;
};

}  // namespace heretic
