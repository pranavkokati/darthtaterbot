#include "eval.h"

#include <cmath>
#include <cstdio>

#include "nnue.h"

namespace heretic {

EvalParams P;

namespace {

Bitboard ForwardFile[2][64];
Bitboard PassedMask[2][64];
Bitboard AdjacentFiles[8];

constexpr int PhaseW[7] = {0, 0, 1, 1, 2, 4, 0};

int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

#include "tuned_params.inc"

}  // namespace

static void set_defaults() {
  static const int matMg[7] = {0, 82, 337, 365, 477, 1025, 0};
  static const int matEg[7] = {0, 94, 281, 297, 512, 936, 0};
  std::memset(&P, 0, sizeof(P));
  for (int pt = 0; pt < 7; ++pt) {
    P.mat[0][pt] = matMg[pt];
    P.mat[1][pt] = matEg[pt];
  }
  static const int pawnRankMg[8] = {0, 0, 2, 6, 12, 25, 45, 0};
  static const int pawnRankEg[8] = {0, 0, 5, 12, 25, 45, 80, 0};
  for (int s = 0; s < 64; ++s) {
    int f = file_of(s), r = rank_of(s);
    int cen = std::min(f, 7 - f) + std::min(r, 7 - r);  // 0..6
    int fcen = std::min(f, 7 - f);                      // 0..3
    // pawns
    P.pst[0][PAWN][s] = pawnRankMg[r] + (r >= 1 && r <= 4 ? (fcen == 3 ? 8 : fcen == 2 ? 3 : fcen == 0 ? -4 : 0) : 0);
    P.pst[1][PAWN][s] = pawnRankEg[r];
    P.pst[0][KNIGHT][s] = (cen - 3) * 9;
    P.pst[1][KNIGHT][s] = (cen - 3) * 7;
    P.pst[0][BISHOP][s] = (cen - 3) * 4;
    P.pst[1][BISHOP][s] = (cen - 3) * 3;
    P.pst[0][ROOK][s] = (fcen >= 2 ? 3 : 0);
    P.pst[1][ROOK][s] = 0;
    P.pst[0][QUEEN][s] = (cen - 3) * 2;
    P.pst[1][QUEEN][s] = (cen - 3) * 4;
    int adf = std::abs(2 * f - 7);  // 1,3,5,7
    P.pst[0][KING][s] = adf * 5 - 25 * std::min(r, 3);
    P.pst[1][KING][s] = (cen - 3) * 12;
  }
  for (int i = 0; i < 9; ++i) P.mobN[0][i] = (i - 4) * 4, P.mobN[1][i] = (i - 4) * 4;
  for (int i = 0; i < 14; ++i) P.mobB[0][i] = (i - 6) * 3, P.mobB[1][i] = (i - 6) * 4;
  for (int i = 0; i < 15; ++i) P.mobR[0][i] = (i - 7) * 2, P.mobR[1][i] = (i - 7) * 4;
  for (int i = 0; i < 28; ++i) P.mobQ[0][i] = (i - 13) * 1, P.mobQ[1][i] = (i - 13) * 2;
  P.bishopPair[0] = 25; P.bishopPair[1] = 45;
  P.rookOpen[0] = 25; P.rookOpen[1] = 10;
  P.rookSemi[0] = 12; P.rookSemi[1] = 8;
  P.rook7th[0] = 15; P.rook7th[1] = 25;
  P.knightOutpost[0] = 18; P.knightOutpost[1] = 10;
  P.doubledP[0] = -10; P.doubledP[1] = -22;
  P.isolatedP[0] = -10; P.isolatedP[1] = -14;
  P.backwardP[0] = -8; P.backwardP[1] = -10;
  static const int conn[8] = {0, 0, 3, 5, 8, 14, 22, 0};
  static const int passMg[8] = {0, 2, 5, 12, 30, 60, 100, 0};
  static const int passEg[8] = {0, 8, 14, 28, 55, 100, 160, 0};
  for (int r = 0; r < 8; ++r) {
    P.connectedP[0][r] = conn[r];
    P.connectedP[1][r] = conn[r] / 2 + 2 * (r >= 3);
    P.passed[0][r] = passMg[r];
    P.passed[1][r] = passEg[r];
  }
  P.passedKingDist[1] = 4;
  P.shield[0] = 14; P.shield[1] = 8; P.shield[2] = 3; P.shield[3] = -16;
  for (int i = 0; i < 32; ++i) P.kingAtk[i] = std::min(i * i / 3, 420);
  P.threatPawn[0] = 28; P.threatPawn[1] = 18;
  P.threatMinorRook[0] = 14; P.threatMinorRook[1] = 10;
  P.hanging[0] = 12; P.hanging[1] = 8;
  P.tempo = 14;
}

void eval_init() {
  static bool done = false;
  if (!done) {
    done = true;
    set_defaults();
    if (TUNED_COUNT == (int)(sizeof(EvalParams) / sizeof(int)))
      std::memcpy(&P, TUNED, sizeof(EvalParams));
    for (int f = 0; f < 8; ++f)
      AdjacentFiles[f] = (f > 0 ? file_bb(f - 1) : 0) | (f < 7 ? file_bb(f + 1) : 0);
    for (Square s = 0; s < 64; ++s) {
      int f = file_of(s), r = rank_of(s);
      Bitboard fw = 0, bw = 0;
      for (int rr = r + 1; rr < 8; ++rr) fw |= rank_bb(rr);
      for (int rr = r - 1; rr >= 0; --rr) bw |= rank_bb(rr);
      ForwardFile[WHITE][s] = file_bb(f) & fw;
      ForwardFile[BLACK][s] = file_bb(f) & bw;
      PassedMask[WHITE][s] = (file_bb(f) | AdjacentFiles[f]) & fw;
      PassedMask[BLACK][s] = (file_bb(f) | AdjacentFiles[f]) & bw;
    }
  }
}

int eval_param_count() { return int(sizeof(EvalParams) / sizeof(int)); }
int* eval_param_ptr() { return reinterpret_cast<int*>(&P); }

std::string eval_param_name(int idx) {
  char buf[64];
  // offsets in order of declaration
  int off = 0;
  auto in = [&](int n) { bool r = idx >= off && idx < off + n; if (!r) off += n; return r; };
  if (in(14)) { snprintf(buf, sizeof buf, "mat[%d][%d]", (idx - off) / 7, (idx - off) % 7); return buf; }
  if (in(2 * 7 * 64)) { int k = idx - off; snprintf(buf, sizeof buf, "pst[%d][%d][%d]", k / 448, (k / 64) % 7, k % 64); return buf; }
  if (in(18)) return "mobN";
  if (in(28)) return "mobB";
  if (in(30)) return "mobR";
  if (in(56)) return "mobQ";
  return "misc[" + std::to_string(idx) + "]";
}

bool eval_save_params(const std::string& path) {
  FILE* f = fopen(path.c_str(), "w");
  if (!f) return false;
  int n = eval_param_count();
  fprintf(f, "%d\n", n);
  for (int i = 0; i < n; ++i) fprintf(f, "%d%c", eval_param_ptr()[i], (i % 16 == 15) ? '\n' : ' ');
  fclose(f);
  return true;
}

bool eval_load_params(const std::string& path) {
  FILE* f = fopen(path.c_str(), "r");
  if (!f) return false;
  int n = 0;
  if (fscanf(f, "%d", &n) != 1 || n != eval_param_count()) { fclose(f); return false; }
  for (int i = 0; i < n; ++i)
    if (fscanf(f, "%d", &eval_param_ptr()[i]) != 1) { fclose(f); return false; }
  fclose(f);
  return true;
}

Value evaluate_hce(const Position& pos) {
  const EvalParams& p = P;
  int mg[2] = {0, 0}, eg[2] = {0, 0};
  int phase = 0;
  const Bitboard occ = pos.occupied();
  Bitboard pawns[2] = {pos.pieces(WHITE, PAWN), pos.pieces(BLACK, PAWN)};
  Bitboard pawnAtt[2] = {shift<NORTH_WEST>(pawns[0]) | shift<NORTH_EAST>(pawns[0]),
                         shift<SOUTH_WEST>(pawns[1]) | shift<SOUTH_EAST>(pawns[1])};
  Bitboard kingZone[2];
  Square ksq[2] = {pos.king_square(WHITE), pos.king_square(BLACK)};
  for (int c = 0; c < 2; ++c) {
    Bitboard z = KingAttacks[ksq[c]] | sq_bb(ksq[c]);
    z |= (c == WHITE) ? shift<NORTH>(z) : shift<SOUTH>(z);
    kingZone[c] = z;
  }
  int atkUnits[2] = {0, 0}, atkCount[2] = {0, 0};
  Bitboard attacked[2] = {pawnAtt[0] | KingAttacks[ksq[0]], pawnAtt[1] | KingAttacks[ksq[1]]};
  Bitboard minorAtt[2] = {0, 0};
  int npm[2] = {0, 0};

  for (int ci = 0; ci < 2; ++ci) {
    Color c = Color(ci), them = ~c;
    Bitboard ownPawns = pawns[ci], theirPawns = pawns[them];
    Bitboard mobArea = ~pos.pieces(c) & ~pawnAtt[them];

    // ---- pawns
    Bitboard bb = ownPawns;
    while (bb) {
      Square s = pop_lsb(bb);
      int f = file_of(s), r = relative_rank(c, s);
      int idx = c == WHITE ? s : (s ^ 56);
      mg[ci] += p.mat[0][PAWN] + p.pst[0][PAWN][idx];
      eg[ci] += p.mat[1][PAWN] + p.pst[1][PAWN][idx];
      bool passed = !(PassedMask[c][s] & theirPawns);
      bool doubled = ForwardFile[c][s] & ownPawns;
      bool isolated = !(AdjacentFiles[f] & ownPawns);
      bool supported = PawnAttacks[them][s] & ownPawns;
      bool phalanx = (shift<EAST>(sq_bb(s)) | shift<WEST>(sq_bb(s))) & ownPawns;
      if (passed && !doubled) {
        mg[ci] += p.passed[0][r];
        eg[ci] += p.passed[1][r];
        int dOwn = std::max(std::abs(file_of(ksq[ci]) - f), std::abs(rank_of(ksq[ci]) - rank_of(s)));
        int dTheirs = std::max(std::abs(file_of(ksq[them]) - f), std::abs(rank_of(ksq[them]) - rank_of(s)));
        eg[ci] += p.passedKingDist[1] * (dTheirs - dOwn) * r / 6;
      }
      if (doubled) { mg[ci] += p.doubledP[0]; eg[ci] += p.doubledP[1]; }
      if (isolated) { mg[ci] += p.isolatedP[0]; eg[ci] += p.isolatedP[1]; }
      else if (!supported && !phalanx) {
        // backward: stop square attacked by enemy pawn and no friendly pawn can ever support
        Square stop = c == WHITE ? s + 8 : s - 8;
        Bitboard behindAdj = AdjacentFiles[f] & ownPawns & ~PassedMask[c][s] & ~ForwardFile[c][s];
        if ((PawnAttacks[c][stop] & theirPawns) && !(behindAdj & ~ForwardFile[c][s]) ) {
          mg[ci] += p.backwardP[0]; eg[ci] += p.backwardP[1];
        }
      }
      if (supported || phalanx) { mg[ci] += p.connectedP[0][r]; eg[ci] += p.connectedP[1][r]; }
    }

    // ---- knights
    bb = pos.pieces(c, KNIGHT);
    while (bb) {
      Square s = pop_lsb(bb);
      int idx = c == WHITE ? s : (s ^ 56);
      mg[ci] += p.mat[0][KNIGHT] + p.pst[0][KNIGHT][idx];
      eg[ci] += p.mat[1][KNIGHT] + p.pst[1][KNIGHT][idx];
      phase += PhaseW[KNIGHT];
      npm[ci] += SeeValue[KNIGHT];
      Bitboard a = KnightAttacks[s];
      attacked[ci] |= a;
      minorAtt[ci] |= a;
      int m = popcount(a & mobArea);
      mg[ci] += p.mobN[0][m];
      eg[ci] += p.mobN[1][m];
      int r = relative_rank(c, s);
      if (r >= 3 && r <= 5 && (PawnAttacks[them][s] & ownPawns) && !(PassedMask[c][s] & AdjacentFiles[file_of(s)] & theirPawns)) {
        mg[ci] += p.knightOutpost[0];
        eg[ci] += p.knightOutpost[1];
      }
      if (a & kingZone[them]) { atkUnits[ci] += 2; ++atkCount[ci]; }
    }
    // ---- bishops
    bb = pos.pieces(c, BISHOP);
    if (more_than_one(bb)) { mg[ci] += p.bishopPair[0]; eg[ci] += p.bishopPair[1]; }
    while (bb) {
      Square s = pop_lsb(bb);
      int idx = c == WHITE ? s : (s ^ 56);
      mg[ci] += p.mat[0][BISHOP] + p.pst[0][BISHOP][idx];
      eg[ci] += p.mat[1][BISHOP] + p.pst[1][BISHOP][idx];
      phase += PhaseW[BISHOP];
      npm[ci] += SeeValue[BISHOP];
      Bitboard a = bishop_attacks(s, occ ^ pos.pieces(c, QUEEN));
      attacked[ci] |= a;
      minorAtt[ci] |= a;
      int m = popcount(a & mobArea);
      mg[ci] += p.mobB[0][m];
      eg[ci] += p.mobB[1][m];
      if (a & kingZone[them]) { atkUnits[ci] += 2; ++atkCount[ci]; }
    }
    // ---- rooks
    bb = pos.pieces(c, ROOK);
    while (bb) {
      Square s = pop_lsb(bb);
      int idx = c == WHITE ? s : (s ^ 56);
      mg[ci] += p.mat[0][ROOK] + p.pst[0][ROOK][idx];
      eg[ci] += p.mat[1][ROOK] + p.pst[1][ROOK][idx];
      phase += PhaseW[ROOK];
      npm[ci] += SeeValue[ROOK];
      Bitboard a = rook_attacks(s, occ ^ pos.pieces(c, ROOK) ^ pos.pieces(c, QUEEN));
      attacked[ci] |= a;
      int m = popcount(a & mobArea);
      mg[ci] += p.mobR[0][m];
      eg[ci] += p.mobR[1][m];
      Bitboard fileBB = file_bb(file_of(s));
      if (!(fileBB & ownPawns)) {
        if (!(fileBB & theirPawns)) { mg[ci] += p.rookOpen[0]; eg[ci] += p.rookOpen[1]; }
        else { mg[ci] += p.rookSemi[0]; eg[ci] += p.rookSemi[1]; }
      }
      if (relative_rank(c, s) == 6 && (relative_rank(c, ksq[them]) == 7 || (pawns[them] & rank_bb(c == WHITE ? 6 : 1)))) {
        mg[ci] += p.rook7th[0]; eg[ci] += p.rook7th[1];
      }
      if (a & kingZone[them]) { atkUnits[ci] += 3; ++atkCount[ci]; }
    }
    // ---- queens
    bb = pos.pieces(c, QUEEN);
    while (bb) {
      Square s = pop_lsb(bb);
      int idx = c == WHITE ? s : (s ^ 56);
      mg[ci] += p.mat[0][QUEEN] + p.pst[0][QUEEN][idx];
      eg[ci] += p.mat[1][QUEEN] + p.pst[1][QUEEN][idx];
      phase += PhaseW[QUEEN];
      npm[ci] += SeeValue[QUEEN];
      Bitboard a = queen_attacks(s, occ);
      attacked[ci] |= a;
      int m = popcount(a & mobArea);
      mg[ci] += p.mobQ[0][m];
      eg[ci] += p.mobQ[1][m];
      if (a & kingZone[them]) { atkUnits[ci] += 5; ++atkCount[ci]; }
    }
    // ---- king
    {
      int idx = c == WHITE ? ksq[ci] : (ksq[ci] ^ 56);
      mg[ci] += p.pst[0][KING][idx];
      eg[ci] += p.pst[1][KING][idx];
      // pawn shield
      int kf = file_of(ksq[ci]), kr = relative_rank(c, ksq[ci]);
      if (kr <= 1) {
        for (int f = std::max(0, kf - 1); f <= std::min(7, kf + 1); ++f) {
          Bitboard fp = file_bb(f) & ownPawns;
          int best = 3;  // missing
          while (fp) {
            Square ps = pop_lsb(fp);
            int d = relative_rank(c, ps) - kr;
            if (d >= 1 && d - 1 < best) best = d - 1;
          }
          mg[ci] += p.shield[std::min(best, 3)];
        }
      }
    }
  }

  // ---- king danger + threats (needs attack info for both sides)
  for (int ci = 0; ci < 2; ++ci) {
    Color c = Color(ci), them = ~c;
    if (atkCount[ci] >= 2) mg[ci] += p.kingAtk[std::min(atkUnits[ci], 31)];   // attacker (c) gets bonus
    // threats by c on them
    Bitboard theirNonPawn = pos.pieces(them) & ~pos.pieces(PAWN) & ~pos.pieces(KING);
    int tp = popcount(pawnAtt[ci] & theirNonPawn);
    mg[ci] += tp * p.threatPawn[0];
    eg[ci] += tp * p.threatPawn[1];
    Bitboard theirRQ = pos.pieces(them, ROOK) | pos.pieces(them, QUEEN);
    int tm = popcount(minorAtt[ci] & theirRQ);
    mg[ci] += tm * p.threatMinorRook[0];
    eg[ci] += tm * p.threatMinorRook[1];
    Bitboard hang = theirNonPawn & attacked[ci] & ~attacked[them];
    int hn = popcount(hang);
    mg[ci] += hn * p.hanging[0];
    eg[ci] += hn * p.hanging[1];
  }

  int mgS = mg[WHITE] - mg[BLACK];
  int egS = eg[WHITE] - eg[BLACK];

  // endgame scaling
  int scale = 64;
  {
    Color strong = egS > 0 ? WHITE : BLACK;
    Color weak = ~strong;
    if (!pawns[strong] && npm[strong] - npm[weak] <= SeeValue[BISHOP]) scale = 8;
    else if (npm[0] == SeeValue[BISHOP] && npm[1] == SeeValue[BISHOP] && pos.count(WHITE, BISHOP) == 1 &&
             pos.count(BLACK, BISHOP) == 1) {
      Bitboard wb = pos.pieces(WHITE, BISHOP), bbp = pos.pieces(BLACK, BISHOP);
      bool wLight = (wb & 0x55AA55AA55AA55AAULL) != 0;
      bool bLight = (bbp & 0x55AA55AA55AA55AAULL) != 0;
      if (wLight != bLight) scale = 32;
    }
  }
  if (phase > 24) phase = 24;
  int score = (mgS * phase + egS * (24 - phase) * scale / 64) / 24;
  if (pos.side_to_move() == BLACK) score = -score;
  score += p.tempo;
  // 50-move damping
  score = score * (200 - pos.rule50()) / 200;
  return clampi(score, -VALUE_MATE_IN_MAX_PLY + 1, VALUE_MATE_IN_MAX_PLY - 1);
}

Value evaluate(const Position& pos) {
  if (nnue::loaded()) return nnue::evaluate_position(pos);
  return evaluate_hce(pos);
}

}  // namespace heretic
