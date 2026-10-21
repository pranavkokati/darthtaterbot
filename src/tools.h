// Offline tooling commands (data generation, tuning, tests) reachable from the UCI loop
#pragma once
#include <iostream>
#include <sstream>

#include "search.h"

namespace heretic {

// Returns true if `cmd` was a recognised tool command.
bool tools_command(const std::string& cmd, std::istringstream& args, Engine& eng);

}  // namespace heretic
