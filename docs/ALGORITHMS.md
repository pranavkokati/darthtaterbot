# Heretic — algorithms, design decisions and honest assessment

This document explains every major component of the engine, what it is derived from, what is (to my knowledge)
unusual, and what was **measured** versus merely **designed**. Confidence levels are given where a claim is not
directly backed by a measurement in this repository.

Conventions: *cp* = centipawns; *stm* = side to move; *wp(x)* = `1/(1+exp(-x·ln10/400))`, the logistic win-probability
map used throughout (eval training, MCTS values, swindle model).

---------------------------------------------------------------------------------------------------------------

## 1. Board representation and move generation  (`bitboard.*`, `position.*`, `movegen.*`)

* 64-bit **bitboards**, one per piece type and colour; a mailbox array for O(1) `piece_on`.
* **Magic bitboards** (Kannan 2007; Romstad's "fancy" layout) for sliders. Magics are *found at start-up* with a
  deterministic xorshift64* search (fixed per-rank seeds), so the repository contains no opaque constant tables.
* **Zobrist hashing**: full key, pawn key, and per-colour non-pawn keys (the last three feed correction history).
* `do_move/undo_move` over a `StateInfo` stack; **legal** move generation using pin and check masks (no
  make-then-test), and a swap-algorithm **SEE** (`see_ge`).
* **Validation** (confidence: high): the standard perft suite (startpos, Kiwipete, positions 3–6) matches the published
  node counts to depth 5–6, and a **fuzz oracle** cross-checks `generate()`, `pseudo_legal()`, `legal()`,
  `gives_check()` and make/unmake round-trips on >100 000 random-play positions (0 errors). The fuzzer found a real
  bug during development (flags 6/7 accepted as pawn captures by `pseudo_legal`), which is fixed.

## 2. Alpha-beta search  (`search.cpp`)

Principal-variation search with iterative deepening and the set of techniques that define modern engines
(Stockfish, Berserk, Ethereal, …):

| Component | Notes |
|---|---|
| Aspiration windows | delta = 10 + |avg|/128, widened by delta/3 on failure |
| Transposition table | lockless, 16-byte entries, XOR-validated, 4-way buckets, generation aging |
| Pruning | reverse futility, razoring, null move (with verification at depth), ProbCut, futility, LMP, SEE pruning, IIR |
| Extensions | check, singular extension |
| LMR | log-log table, adjusted by history, PV-ness, improving, cut-node |
| Move ordering | TT move, SEE-sorted captures, killers, counter-move, butterfly + continuation (1/2/4 ply) + capture history |
| **Correction history** | static-eval error learned online, keyed by pawn structure and by each colour's non-pawn keys (the Stockfish ≥ 16 idea) |
| Lazy SMP | helper threads share the TT and skip alternate iterations |
| Time management | `optimum = avail/mtg + 0.75·inc`, stretched/shrunk by best-move stability and score drops, hard cap at `0.4·avail` |
| Draw handling | repetition, 50-move, insufficient material, contempt with ±1 parity jitter so the engine does not see a perfectly symmetric draw value |

Confidence that each component is *correctly implemented*: moderate-high (UCI test + hundreds of self-play and
cross-engine games without crash, illegal move, or time forfeit). Confidence that each is *individually tuned*:
low — no component has been SPRT-tuned in isolation; I have not run Fishtest-scale parameter searches.

## 3. Evaluation  (`eval.*`)

### 3.1 Handcrafted evaluation (HCE) — the shipped default

Tapered mg/eg evaluation: material, piece-square tables, mobility (N/B/R/Q), bishop pair, rook on (semi-)open
files and 7th, knight outposts, pawn structure (doubled, isolated, backward, connected, passed with king-distance),
king safety (pawn shield + attack-unit table), threats, hanging pieces, tempo. **All 1135 integers live in one flat
`EvalParams` struct** so that a single generic tuner can optimise everything.

### 3.2 Texel-style tuning  (`heretic tune`)

Minimise `mean (T − wp(K·eval))²` over quiet self-play positions, by coordinate descent (±step, walk while improving).
Target `T = (1−β)·result + β·wp(search score)` with β = 0.5: the result is unbiased but high-variance, the labeller's
search score is low-variance but biased; the blend regularises.

**A cautionary result, measured.** My first run split train/validation *per position*. Positions of the same game
share one result, so this leaks: validation error fell 0.0945 → 0.0898, yet the tuned parameters played **−78 Elo**
(200 games, 20 000 nodes, 95 % CI [−125, −33]) against the untuned defaults — classic memorisation of game outcomes.
After (a) splitting **by game**, (b) an L2 prior toward the starting parameters (μ = 5·10⁻⁹), (c) 8× more games and
(d) the β-blend, the same procedure gave **+143 Elo** (200 games, CI [+100, +191]). Lesson recorded in the code
comments: *never split sequential game data per position.*

### 3.3 NNUE  (`nnue.cpp`, `python/train_nnue.py`)

Architecture `768 → H (×2 perspectives) → SCReLU → 1`, float inference, pure-numpy trainer (no GPU or PyTorch
required). Two modes: absolute, and **residual** (`HNN2`): the network output is *added to* the HCE score, so training
starts exactly at the HCE and learns only a correction. Findings so far (see §8): on ≈ 220 k self-play positions a
H = 64 net **predicts the labels better than the HCE does** (validation BCE 0.4596 vs 0.4884) yet **plays worse** at
equal node counts (−168 Elo residual, −301 Elo absolute). Interpretation (confidence: moderate): distilling a
depth-8 search of the *same* HCE teaches the net to anticipate that search's tactics; inside a search that already
does those tactics the scale/margin interaction is harmful. Real NNUE gains need orders of magnitude more data and
a stronger teacher than I could generate on two CPU cores; the NNUE path is therefore shipped as an *experimental*
option, not the default.

## 4. Self-play data pipeline  (`heretic datagen`)

Random 8–12-ply openings filtered to |eval| ≤ 350 cp at depth 6; fixed-depth self-play; adjudication on 5 consecutive
plies ≥ 1200 cp or 30 plies of |eval| ≤ 6 after ply 100; 1/48 random-move chance before ply 40. Records are 40-byte
`PackedPos` (nibble-packed board, stm, flags, search score, result). Check positions and positions whose best move is
noisy are excluded from training.

---------------------------------------------------------------------------------------------------------------

## 5. The "heresies" (non-standard components)

I want to be precise about novelty. **None of these is a claimed invention**; each combines published ideas in a way
that I am not aware of in mainstream engines. Confidence in that "not aware" statement: moderate (I did a literature
and engine-source survey, but engine codebases are enormous and engines like ShashChess, Brainfish and
Komodo have many undocumented features).

### 5.1 Quantal-response swindling  (`QuantalSwindle`, `OpponentElo`, `OpponentTime`)

*Idea.* Minimax assumes a perfect opponent. Against a fallible one, a slightly inferior move that sets problems can
have a higher *expected* result. Model the opponent as a **logit quantal-response** player (McKelvey & Palfrey 1995,
*Games and Economic Behavior* 10(1), 6–38): it chooses reply *i* with probability ∝ `exp(s_i / T)`, where `s_i` is that
reply's value for the opponent and **T is the rationality temperature**. Opponent-model search itself is old (Iida et
al. 1993; Carmel & Markovitch, AAAI 1996); the specific recipe below is, as far as I know, not in mainstream engines.

*Algorithm (final design).*
1. Run the **normal, unmodified search** with its full budget.
2. If the position is not clearly won (≤ +250 cp), run a separate **probe engine** (own TT) for ~25 % of the move's
   optimum time with MultiPV = 4 to get candidate scores on a common footing. Keep candidates within
   `SwindleMaxLoss` (30 cp) of the probe's best; if the main search's move is not among them, trust the main search.
3. For each candidate, probe all of the opponent's legal replies (full width, shallow). With reply values `s_i`
   (opponent's point of view), the *swindle bonus* is
   `E[wp(−s)] − wp(−max_i s_i)` where the expectation uses softmax weights `exp(s_i/T)`; i.e. the expected gain
   from the opponent not finding the best reply, measured by the *same* probe so shallow-search noise cancels.
4. Candidate utility `U = wp(probe score) + bonus`; deviate from the main search's move only if `U` is higher by
   ≥ `SwindleMinGain` (3 percentage points).
5. **Temperature** `T = max(25, 40 + 0.12·(2400 − Elo_opp))` cp, up to 2× when the opponent's clock is under 10 s.

*Status and measured cost (3 s + 0.1 s, 2 cores, vs. the same engine without the option; Elo, 95 % CI).*

| Design | Result |
|---|---|
| v0: MultiPV 3 in the main search + 60 % budget, 60 cp loss cap | −576 [−876, −467] (100 g), later −352 (60 g) |
| v1: same, 30 cp cap | −325 [−453, −243] (60 g) |
| v1 with deviation disabled (overhead only) | −215 [−344, −124] (40 g) |
| **v2 (final, probe design)** | **−199 [−300, −121]** (60 g) |
| v2 with deviation disabled (probe time only) | −108 [−196, −31] (40 g) |

Interpretation (confidence: moderate-high): the early designs were dominated by *search overhead* (MultiPV and a
cut main-search budget), which is why the design was changed; in the final design roughly half of the remaining cost is
pure extra clock usage (the probe phase) and the other half is the cost of deviating from the engine's preferred move.
**Against equally strong, non-fallible engines the swindle is a net loss, as theory predicts** — a quantal model is
wrong for an engine opponent. Its intended habitat is human opponents, where I could not measure it; the bot therefore
enables it **only against non-BOT opponents** and it is **off by default**. I do not claim it gains Elo against humans.

### 5.2 Opening variety without a book  (`VarietyPlies`, `VarietyMargin`)

For the first N plies sample among root moves within `VarietyMargin` cp of the best, weighted `exp(−Δ/8)`. This makes
the bot non-repetitive on a public server (where deterministic engines get exploited by opening preparation) at a
bounded evaluation cost.

### 5.3 Gumbel/PUCT Monte-Carlo tree search mode  (`SearchMode 1`)

An AlphaZero-flavoured search that needs no trained policy network: priors from a softmax over a cheap move
heuristic (MVV, SEE sign, checks, promotions, history), values from the win probability of the quiescence-searched
evaluation, PUCT below the root and — at the root — **Gumbel-Top-k sampling with Sequential Halving** (Danihelka,
Guez, Schrittwieser & Silver, *Policy improvement by planning with Gumbel*, ICLR 2022). Gumbel root selection is
designed to give a guaranteed policy improvement even with very few simulations. Confirmed working: finds mates
and tactics in a few thousand simulations and respects time controls. **Measured: 0 wins, 0 draws, 60 losses** against the alpha-beta mode (3 s + 0.1 s) — it is far weaker (priors
are heuristic rather than learned, and each leaf is only a quiescence-searched static eval); it is shipped as a research option and a stylistic
alternative, and it is the natural place to plug in a learned policy network.

### 5.4 Experience memory in the bot  (`python/lichess_bot.py`)

Per-position `(EPD, move) → [W, D, L]` is recorded in `data/experience.json`; at the root of the first 30 moves the
bot refuses a move that has *only* lost from that exact position before, provided an alternative within 80 cp exists.
(Experience files exist in other engines — Brainfish, ShashChess; this is a simplified variant tied to the bot.)

### 5.5 Opponent-aware dynamic contempt in the bot

`Contempt = clamp((my_rating − opp_rating)/8, −20, 60)` cp: play for a win against weaker players, accept draws
against stronger ones.

---------------------------------------------------------------------------------------------------------------

## 6. Lichess Bot API bridge  (`python/lichess_bot.py`)

Implemented against the official OpenAPI specification (`lichess-org/api`): `GET /api/stream/event`,
`GET /api/bot/game/stream/{id}`, `POST /api/bot/game/{id}/move/{move}`, `/draw/{yes|no}`, `/claim-victory`, `/chat`,
`POST /api/challenge/{id}/accept|decline`, `POST /api/bot/account/upgrade`.

* NDJSON streams with reconnect/back-off; HTTP 429 → 60 s sleep as the API documents.
* Challenge policy (variants, speeds, rated/casual, bot/human, per-opponent rate limit, concurrency) with correct
  decline reasons.
* Per-game thread; time passed as integer ms clocks to the UCI engine; `OpponentElo/OpponentTime/Contempt`
  re-configured every move.
* Draw offers accepted only when our score ≤ threshold in a long game; `opponentGone` → `claim-victory` timer.
* Token only from the environment; **account upgrade to BOT is irreversible and only happens with `--upgrade`.**

**Testing status (be precise):** no real Lichess bot token was available in the development environment, so the
bridge has **not played on lichess.org**. It was tested end-to-end against `tests/mock_lichess.py`, a local server
speaking the same chunked-NDJSON protocol: challenge accept/decline, game stream, 12 legal bot moves against a random
opponent, chat, game end, and experience recording all passed. Differences between the mock and production that I
cannot rule out: rate-limit behaviour, edge-case event types, and real network jitter.

---------------------------------------------------------------------------------------------------------------

## 7. Methodology for strength claims

* Matches by `scripts/match.py`: paired colour-reversed openings from `data/openings.txt` (150 positions pre-filtered
  to ±45 cp with Stockfish 16 depth 10), adjudication, Elo with 95 % CI from the trinomial score distribution, LOS,
  and optional Wald **SPRT**. Fixed-node matches isolate algorithmic quality from CPU contention; the development
  machine has only 2 cores, so time-controlled results are noisier.
* Every number reported in §8 is copied from a log in `results/`.

## 8. Results

See `README.md` ("Measured results") — kept in one place to avoid drift.

## 9. Known limitations / things I did not do

* No pondering, no tablebases (Syzygy), no opening book. The first two are straightforward additions; the last is
  replaced by §5.2.
* NNUE inference recomputes the accumulator from scratch (no incremental update) and is scalar float.
* No fishtest-scale tuning of search constants.
* The Lichess bridge has not been run against the live service.
* Swindle and MCTS strength effects are unmeasured/likely negative-to-neutral against engines of equal strength.

## References

1. Silver et al., *A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play*,
   Science 362(6419):1140–1144, 2018.
2. Danihelka, Guez, Schrittwieser, Silver, *Policy improvement by planning with Gumbel*, ICLR 2022.
3. McKelvey & Palfrey, *Quantal response equilibria for normal form games*, Games and Economic Behavior 10(1):6–38, 1995.
4. Carmel & Markovitch, *Incorporating opponent models into adversary search*, AAAI 1996.
5. Iida, Uiterwijk, van den Herik, Herschberg, *Potential applications of opponent-model search*, ICCA Journal, 1993
   (confidence in exact citation details: moderate).
6. Nasu, *Efficiently Updatable Neural-Network-based Evaluation Function for computer Shogi* (2018); Stockfish 12
   release notes (2020) for NNUE in chess.
7. Wald, *Sequential Tests of Statistical Hypotheses*, Annals of Mathematical Statistics 16(2), 1945 (SPRT).
8. Österlund, "Texel's tuning method" (Chess Programming Wiki).
9. Lichess Bot API: <https://lichess.org/api#tag/Bot> and the OpenAPI spec at `github.com/lichess-org/api`.
