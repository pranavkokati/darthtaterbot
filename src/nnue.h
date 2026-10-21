// NNUE-style evaluation (stub header; implementation added once the network trainer exists)
#pragma once
#include <string>

#include "position.h"

namespace heretic {
namespace nnue {

bool loaded();
bool load(const std::string& path);
void unload();
Value evaluate_position(const Position& pos);

}  // namespace nnue
}  // namespace heretic
