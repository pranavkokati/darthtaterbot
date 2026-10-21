CXX      ?= g++
ARCH     ?= native
CXXFLAGS ?= -std=c++17 -O3 -march=$(ARCH) -fno-plt -flto -DNDEBUG -Wall -Wextra -Wno-unused-parameter
LDFLAGS  ?= -pthread -flto
SRC       = $(filter-out src/perft_main.cpp,$(wildcard src/*.cpp))
OBJ       = $(patsubst src/%.cpp,build/%.o,$(SRC))

all: build/heretic build/perft

build/heretic: $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

build/perft: src/perft_main.cpp src/bitboard.cpp src/position.cpp src/movegen.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

build/%.o: src/%.cpp $(wildcard src/*.h) src/tuned_params.inc
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -c -o $@ $<

test: all
	python3 tests/run_perft.py ./build/perft 5
	./build/perft fuzz 500

clean:
	rm -rf build

.PHONY: all test clean
