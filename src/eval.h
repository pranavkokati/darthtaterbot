// Handcrafted evaluation (tapered, Texel-tunable)
#pragma once
#include <string>

#include "position.h"

namespace heretic {

// All members are ints so the struct can be treated as a flat parameter vector by the tuner.
struct EvalParams {
  int mat[2][7];        // [mg/eg][piece type]
  int pst[2][7][64];    // [mg/eg][piece type][square from white's view]
  int mobN[2][9];
  int mobB[2][14];
  int mobR[2][15];
  int mobQ[2][28];
  int bishopPair[2];
  int rookOpen[2];
  int rookSemi[2];
  int rook7th[2];
  int knightOutpost[2];
  int doubledP[2];
  int isolatedP[2];
  int backwardP[2];
  int connectedP[2][8];  // by relative rank
  int passed[2][8];      // by relative rank
  int passedKingDist[2]; // eg: (enemy king dist - own king dist) per rank weight
  int shield[4];         // pawn-shield bonus by distance 1,2,3, missing
  int kingAtk[32];       // king danger by attack units
  int threatPawn[2];     // pawn attacks a non-pawn piece
  int threatMinorRook[2];
  int hanging[2];        // undefended piece attacked by enemy
  int tempo;
};

extern EvalParams P;

void eval_init();                                  // set defaults (and apply any tuned table)
bool eval_load_params(const std::string& path);    // load flat int dump
bool eval_save_params(const std::string& path);
int eval_param_count();
int* eval_param_ptr();
std::string eval_param_name(int idx);

// Side-to-move relative score in centipawn-like units.
Value evaluate_hce(const Position& pos);

// Public entry point used by search: dispatches to NNUE if loaded, else HCE.
Value evaluate(const Position& pos);

}  // namespace heretic
