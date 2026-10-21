// Heretic NNUE: (768 -> H) x 2 perspectives -> SCReLU -> 1, float inference.
//
// Feature index for perspective P:  ((piece_color != P) * 6 + (type-1)) * 64 + (sq ^ (P == BLACK ? 56 : 0))
// so every network sees the board "from the mover's side" (colour- and rank-flipped for Black).
#include "nnue.h"

#include "eval.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace heretic {
namespace nnue {

namespace {
constexpr int FEATURES = 768;
constexpr int MAX_H = 1024;
constexpr float CP_PER_UNIT = 173.7f;  // output units -> centipawns (inverse of ln(10)/400)

struct Net {
  int H = 0;
  std::vector<float> w1;  // FEATURES x H
  std::vector<float> b1;  // H
  std::vector<float> w2;  // 2H
  float b2 = 0;
  bool residual = false;  // HNN2: network output is a correction added to the handcrafted evaluation
};
Net net;
bool isLoaded = false;

inline int feature(Color persp, Piece pc, Square sq) {
  int rel = color_of(pc) != persp;
  return (rel * 6 + (type_of(pc) - 1)) * 64 + (persp == BLACK ? (sq ^ 56) : sq);
}
}  // namespace

bool loaded() { return isLoaded; }

void unload() {
  isLoaded = false;
  net = Net();
}

bool load(const std::string& path) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) return false;
  char magic[4];
  int32_t H = 0, F = 0;
  if (fread(magic, 1, 4, f) != 4 || (std::memcmp(magic, "HNN1", 4) != 0 && std::memcmp(magic, "HNN2", 4) != 0) || fread(&H, 4, 1, f) != 1 ||
      fread(&F, 4, 1, f) != 1 || F != FEATURES || H <= 0 || H > MAX_H) {
    fclose(f);
    return false;
  }
  Net n;
  n.H = H;
  n.residual = std::memcmp(magic, "HNN2", 4) == 0;
  n.w1.resize(size_t(FEATURES) * H);
  n.b1.resize(H);
  n.w2.resize(2 * size_t(H));
  bool ok = fread(n.w1.data(), 4, n.w1.size(), f) == n.w1.size() && fread(n.b1.data(), 4, n.b1.size(), f) == n.b1.size() &&
            fread(n.w2.data(), 4, n.w2.size(), f) == n.w2.size() && fread(&n.b2, 4, 1, f) == 1;
  fclose(f);
  if (!ok) return false;
  net = std::move(n);
  isLoaded = true;
  return true;
}

Value evaluate_position(const Position& pos) {
  const int H = net.H;
  alignas(64) float accS[MAX_H];
  alignas(64) float accN[MAX_H];
  const Color us = pos.side_to_move(), them = ~us;
  std::memcpy(accS, net.b1.data(), sizeof(float) * H);
  std::memcpy(accN, net.b1.data(), sizeof(float) * H);
  Bitboard occ = pos.occupied();
  while (occ) {
    Square s = pop_lsb(occ);
    Piece pc = pos.piece_on(s);
    const float* rs = &net.w1[size_t(feature(us, pc, s)) * H];
    const float* rn = &net.w1[size_t(feature(them, pc, s)) * H];
    for (int j = 0; j < H; ++j) {
      accS[j] += rs[j];
      accN[j] += rn[j];
    }
  }
  float out = net.b2;
  const float* w2s = net.w2.data();
  const float* w2n = net.w2.data() + H;
  for (int j = 0; j < H; ++j) {
    float a = accS[j] < 0.f ? 0.f : (accS[j] > 1.f ? 1.f : accS[j]);
    float b = accN[j] < 0.f ? 0.f : (accN[j] > 1.f ? 1.f : accN[j]);
    out += a * a * w2s[j] + b * b * w2n[j];
  }
  int cp = int(std::lround(out * CP_PER_UNIT));
  if (net.residual) cp += evaluate_hce(pos);
  // same 50-move damping as the HCE so search heuristics behave consistently
  cp = cp * (200 - pos.rule50()) / 200;
  if (cp > VALUE_MATE_IN_MAX_PLY - 1) cp = VALUE_MATE_IN_MAX_PLY - 1;
  if (cp < -VALUE_MATE_IN_MAX_PLY + 1) cp = -VALUE_MATE_IN_MAX_PLY + 1;
  return cp;
}

}  // namespace nnue
}  // namespace heretic
