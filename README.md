# Heretic — a from-scratch chess engine and Lichess bot

C++17 UCI engine (bitboards + magic attacks, PVS alpha-beta with Lazy SMP, a Texel-tuned handcrafted evaluation,
an experimental NNUE trainer/inference path, a Gumbel/PUCT MCTS mode) plus a Python Lichess Bot API bridge, a
self-play data pipeline, a Texel tuner and a match/SPRT harness. Author: Pranav Kokati. Everything here was written
and measured in one session on a **2-core** machine; claims below are limited to what those measurements support.
Algorithm details, references and negative results: [`docs/ALGORITHMS.md`](docs/ALGORITHMS.md).

## Build and run

```bash
make                     # build/heretic (UCI engine), build/perft
make test                # perft suite (startpos, Kiwipete, pos 3-6) + 500-iteration fuzz oracle
python3 tests/test_uci.py ./build/heretic        # legality/time-control smoke test of every mode
python3 tests/mock_lichess.py 8765 24 &           # offline end-to-end test of the Lichess bridge
LICHESS_URL=http://127.0.0.1:8765 LICHESS_TOKEN=mock-token python3 python/lichess_bot.py
```
Requires g++ ≥ 9, Python ≥ 3.11 with `python-chess`, `requests`, `numpy`.

UCI options: `Hash`, `Threads`, `MultiPV`, `Contempt`, `Move Overhead`, `EvalParams` (flat parameter file),
`NNUEFile`, and the "heresies" `QuantalSwindle`, `OpponentElo`, `OpponentTime`, `SwindleMaxLoss`, `SwindleMinGain`,
`VarietyPlies`, `VarietyMargin`, `SearchMode` (0 = alpha-beta, 1 = Gumbel/PUCT MCTS).
Tool commands (typed into the engine): `bench N`, `perft D`, `datagen`, `tune`, `hceeval`.

## Playing on Lichess

1. Create a **new** Lichess account that has played **no games**, and an API token with the `bot:play` scope.
2. `export LICHESS_TOKEN=...; python3 python/lichess_bot.py --config config.toml --upgrade`
   (**`--upgrade` converts the account to a BOT irreversibly**; omit it once upgraded).
3. Copy `config.example.toml` to `config.toml` to set the engine path, hash/threads and challenge policy.

**Not yet verified:** the bridge has been tested only against `tests/mock_lichess.py` (a protocol-faithful local mock),
not against lichess.org, because no bot token was available during development.

## Measured results

Conditions: 2 CPU cores, 1 engine thread unless stated, paired colour-reversed openings (`data/openings.txt`),
adjudication on, Elo with 95 % CI from the trinomial score distribution (`scripts/match.py`). Raw logs: `results/`.

| Experiment | Result | Notes |
|---|---|---|
| Perft + fuzz | all suites pass, 101 215-position fuzz: 0 errors | correctness of move generation |
| Texel-tuned HCE vs hand-set defaults | **+143 Elo** [+100, +191], 200 g, 20 k nodes | game-wise split + L2 prior + β-blend |
| …same tuner with per-position split (the mistake) | −78 Elo [−125, −33] | overfitting via leakage; see ALGORITHMS §3.2 |
| A further tuning round (v3 vs v2) | −2 Elo [−42, +39] | converged; v2 shipped |
| **Heretic vs Stockfish 16, `UCI_Elo=2800`** | **−24 Elo [−86, +35], 46.5 %**, 100 g, 3 s+0.1 s | ≈ SF-"2800"; see caveat |
| Heretic vs SF `UCI_Elo=3190` | −191 [−330, −92], 30 g | |
| Heretic vs SF `UCI_Elo=2400` | +223 [+134, +350], 60 g | |
| Heretic vs SF `UCI_Elo=2200` | +407 [+274, +985], 40 g | |
| Lazy SMP, 2 threads vs 1 | **+101 Elo** [+41, +169], 60 g, 3 s+0.1 s | |
| MCTS mode vs alpha-beta | 0 – 60 (−∞) | research option only |
| Quantal swindle vs plain (final design) | −199 [−300, −121], 60 g | **costs Elo vs engines; off by default** |
| Residual NNUE (H=64, 220 k positions) vs HCE | −168 [−294, −74], 40 g, 20 k nodes | not shipped; see below |

**Strength caveat (read before quoting a rating).** `UCI_Elo` in Stockfish is calibrated against specific
conditions (CCRL-style, longer time controls, its own hardware) and weakens play by a randomised-move mechanism, so
"≈ 2750–2800 vs SF-2800" is **not** a CCRL/FIDE/Lichess rating. My honest estimate for the Lichess blitz rating is
"strong-master-range, order of 2600–3000 depending on hardware/time control", confidence **low**; it can only be
settled by playing rated games. Single-thread, 3 s + 0.1 s on a shared 2-core machine is far below the hardware a
Lichess-hosted bot would use.

### What did *not* work (kept deliberately)
* **NNUE by distillation.** A H=64 net trained on ≈ 220 k labelled positions predicts the labels better than the HCE
  (BCE 0.4596 vs 0.4884) yet plays 170–300 Elo worse at equal nodes. The net learns the labelling search's tactics and
  mis-scales the evaluation. Beating a *tuned* HCE needs far more data and a stronger teacher than two cores allow.
  The path (`train_nnue.py`, `HNN1/HNN2` loader, `hceeval`) is kept and tested for correctness (engine output equals the
  numpy forward pass exactly) but is **experimental**.
* **Quantal swindle** costs Elo against engine opponents (table above). It is documented as a human-opponent option.
* **MCTS mode** is far weaker than alpha-beta without a learned policy/value network.

## Repository map
`src/` engine · `python/lichess_bot.py` bridge · `python/train_nnue.py` trainer · `scripts/match.py` SPRT/Elo harness ·
`scripts/make_openings.py` · `scripts/run_eval_suite.sh` · `scripts/selfplay_loop.sh` ·
`tests/` perft, UCI and mock-Lichess tests · `tuned/` tuning logs and parameter files · `results/` match logs.

## Reproducing the strength numbers
`scripts/run_eval_suite.sh` reruns the Stockfish calibration and quirk matches; fixed-node A/B tests use
`python3 scripts/match.py --a A --b B --nodes 20000 --games 200`. The tuned parameters are embedded in
`src/tuned_params.inc` (generated by `heretic tune`).

## Bullet play and challenging opponents

Nothing about timing is a fixed number of seconds. The bridge measures the real round-trip of every move it posts and sets
`Move Overhead` from it (recent max blended with the mean, +30 %); the engine budgets from the remaining clock, the
increment, and an expected-moves-to-go that shrinks as the game proceeds, and it thins its thinking smoothly once the clock
holds fewer than ~150 lag-lengths. The shape of the curve is controlled by UCI options (`TimeMtgBase`, `TimeMtgDecay`,
`TimeMtgFloor`, `TimeIncPct`, `TimeMaxPct`) that are calibrated by self-play. **Run the bot on the machine with the best
connection to Lichess (your laptop), not in a proxied cloud sandbox** — every extra 100 ms of lag is a large fraction of a
bullet move. `scripts/run_bot.sh` restarts the bot automatically if the network drops.

```
export LICHESS_TOKEN=lip_...   # then: scripts/run_bot.sh   (scopes: bot:play, challenge:read, challenge:write)
python3 python/lichess_bot.py --challenge SomeUser --time 60 --inc 0       # 1+0, casual
python3 python/lichess_bot.py --challenge SomeUser --time 120 --inc 1 --rated
```

Status: the Lichess bridge is verified against a mock server and a few live games; long-run live behaviour is still being validated.

Time-knob calibration (self-play, 10+0.1, 30 games each, same binary, 2 threads): `TimeMtgBase=30` scored 36.7 % against 45
(−95 Elo, CI −187…−14); `TimeMtgBase=65` scored 40.0 % against 45 (−70 Elo, CI −169…+18). Both directions lose to the default
of 45, so it is kept. These samples are small; treat them as a sanity check, not a tuned optimum.
