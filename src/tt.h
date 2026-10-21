// Lockless transposition table (16-byte entries, 4 per bucket, XOR-validated for Lazy SMP)
#pragma once
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "types.h"

namespace heretic {

enum Bound : int { BOUND_NONE = 0, BOUND_UPPER = 1, BOUND_LOWER = 2, BOUND_EXACT = 3 };

struct TTData {
  Move move;
  Value value;
  Value eval;
  int depth;
  int bound;
  bool pv;
};

class TranspositionTable {
 public:
  TranspositionTable() : table(nullptr), clusters(0), gen(0) { resize(16); }
  ~TranspositionTable() { free_mem(); }
  TranspositionTable(const TranspositionTable&) = delete;
  TranspositionTable& operator=(const TranspositionTable&) = delete;

  void resize(size_t mb) {
    free_mem();
    clusters = std::max<size_t>(1, mb * 1024 * 1024 / sizeof(Cluster));
    table = static_cast<Cluster*>(std::aligned_alloc(64, clusters * sizeof(Cluster)));
    clear();
  }
  void clear() {
    std::memset((void*)table, 0, clusters * sizeof(Cluster));
    gen = 0;
  }
  void new_search() { gen = uint8_t((gen + 1) & 31); }

  bool probe(Key key, TTData& out) const {
    const Cluster& c = table[index(key)];
    for (int i = 0; i < 4; ++i) {
      uint64_t d = c.e[i].data;
      if (d && (c.e[i].keyXor ^ d) == key) {
        out.move = Move(d & 0xFFFF);
        out.value = int16_t((d >> 16) & 0xFFFF);
        out.eval = int16_t((d >> 32) & 0xFFFF);
        out.depth = int((d >> 48) & 0xFF) - 16;
        uint8_t fl = uint8_t(d >> 56);
        out.bound = fl & 3;
        out.pv = fl & 4;
        return true;
      }
    }
    out.move = MOVE_NONE; out.value = VALUE_NONE; out.eval = VALUE_NONE; out.depth = -16; out.bound = BOUND_NONE; out.pv = false;
    return false;
  }

  void store(Key key, Value value, bool pv, int bound, int depth, Move m, Value eval) {
    Cluster& c = table[index(key)];
    Entry* replace = &c.e[0];
    int worst = 1 << 30;
    for (int i = 0; i < 4; ++i) {
      uint64_t d = c.e[i].data;
      if (!d || (c.e[i].keyXor ^ d) == key) {
        replace = &c.e[i];
        if (d) {
          // keep old move if we have none
          if (!m) m = Move(d & 0xFFFF);
          int oldDepth = int((d >> 48) & 0xFF) - 16;
          uint8_t oldFl = uint8_t(d >> 56);
          int oldGen = oldFl >> 3;
          if (bound != BOUND_EXACT && oldGen == gen && depth + 2 * pv + 4 <= oldDepth && (oldFl & 3)) {
            // keep deeper same-generation info, but refresh the move
            if (m) {
              uint64_t nd = (d & ~0xFFFFULL) | m;
              c.e[i].data = nd;
              c.e[i].keyXor = key ^ nd;
            }
            return;
          }
        }
        goto write;
      }
      int age = (gen - (uint8_t(c.e[i].data >> 56) >> 3)) & 31;
      int rel = int((c.e[i].data >> 48) & 0xFF) - 16 - 2 * age;
      if (rel < worst) { worst = rel; replace = &c.e[i]; }
    }
  write:
    int dd = std::max(-16, std::min(depth, 239)) + 16;
    uint8_t fl = uint8_t(bound | (pv ? 4 : 0) | (gen << 3));
    uint64_t d = uint64_t(m) | (uint64_t(uint16_t(int16_t(value))) << 16) | (uint64_t(uint16_t(int16_t(eval))) << 32) |
                 (uint64_t(dd) << 48) | (uint64_t(fl) << 56);
    replace->data = d;
    replace->keyXor = key ^ d;
  }

  int hashfull() const {
    int cnt = 0;
    size_t n = std::min<size_t>(clusters, 250);
    for (size_t i = 0; i < n; ++i)
      for (int j = 0; j < 4; ++j) {
        uint64_t d = table[i].e[j].data;
        if (d && ((uint8_t(d >> 56) >> 3) == gen)) ++cnt;
      }
    return int(cnt * 1000 / (n * 4));
  }

  void prefetch(Key key) const { __builtin_prefetch(&table[index(key)]); }

 private:
  struct Entry { uint64_t keyXor, data; };
  struct Cluster { Entry e[4]; };
  size_t index(Key key) const { return size_t((unsigned __int128)key * clusters >> 64); }
  void free_mem() { if (table) std::free(table), table = nullptr; }
  Cluster* table;
  size_t clusters;
  uint8_t gen;
};

}  // namespace heretic
