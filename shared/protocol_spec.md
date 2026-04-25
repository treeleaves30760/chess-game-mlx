# Engine ↔ GUI Protocol Specification

The engine speaks two protocols simultaneously over **stdin/stdout**:

1. **Standard UCI** (chess) or **USI** (shogi) — for compatibility with any stock chess/shogi GUI (cutechess, Arena, ShogiGUI, ShogiHome). See references at the end.
2. **Chess_Game_mlx JSON-RPC extension** — for features the standard protocols don't cover, most importantly **multi-ponder** and **single-side mode**.

The GUI can use either. Third-party GUIs ignore the JSON lines; our own GUI uses both.

---

## 1. Line discrimination

The engine sends **newline-delimited** messages on stdout. Discrimination:

- If the line starts with `{` → it's a JSON-RPC extension message.
- Otherwise → it's a standard UCI/USI `info` / `bestmove` / `readyok` / etc. line.

On stdin, the same rule applies.

This keeps standard-GUI compatibility perfect (they never parse lines starting with `{`).

---

## 2. Standard UCI (chess)

Implemented in `engine/src/protocol/uci_loop.cpp`. Supported commands:

| Command | Notes |
|---|---|
| `uci` | Reply with engine ID, options, `uciok` |
| `isready` | Reply `readyok` |
| `ucinewgame` | Reset engine state (tree + TT) |
| `position {startpos\|fen <FEN>} [moves <m1> <m2> ...]` | Set current position |
| `go [wtime N] [btime N] [winc N] [binc N] [movestogo N] [depth N] [nodes N] [movetime N] [infinite]` | Start search |
| `stop` | Halt current search, emit `bestmove` |
| `ponderhit` | Transition from ponder to normal search |
| `quit` | Exit |

### 2.1 Options (via `setoption name X value Y`)

| Option | Type | Default | Description |
|---|---|---|---|
| `MultiPV` | spin 1..10 | 1 | Number of principal variations to return |
| `Hash` | spin 16..16384 | 512 | Transposition table MiB (when enabled) |
| `Threads` | spin 1..16 | 4 | CPU threads for MCTS |
| `NN_Weights` | string | (empty) | Path to `.safetensors` file loaded at init |
| `Ponder` | check | true | Enable standard single-line ponder |

### 2.2 `info` output

Every ~250 ms during search we emit:
```
info depth <d> seldepth <sd> multipv <k> score cp <v> nodes <n> nps <nps> time <ms> pv <m1> <m2> ...
```
with one line per multipv (k = 1..MultiPV).

`score` is in **centipawns from white's POV** (positive = white better) regardless of side-to-move. Our internal value is `[-1, +1]` — we map via `cp = round(400 * tan(value * π / 2))` (the standard AlphaZero-style conversion, clipped to [-∞, +∞]).

---

## 3. Standard USI (shogi)

Same structure as UCI but with:
- `usi` / `usiok` instead of `uci` / `uciok`
- `usinewgame` instead of `ucinewgame`
- Move notation: USI (e.g. `7g7f`, `P*5f` for drops)
- Position: `sfen <SFEN> moves ...`

Same options as UCI (with `USI_MultiPV` as an alias for compatibility).

---

## 4. JSON-RPC extension

### 4.1 Wire format

Requests (client → engine):
```json
{"jsonrpc":"2.0","method":"<name>","params":{...},"id":<integer>}
```

Responses:
```json
{"jsonrpc":"2.0","result":{...},"id":<integer>}
```

Notifications (engine → client, no `id`):
```json
{"jsonrpc":"2.0","method":"<event_name>","params":{...}}
```

### 4.2 Methods (client → engine)

#### `set_side`
Lock engine to analyse one colour only.
```json
{"method":"set_side","params":{"me":"white"}}
// or "black" / "sente" / "gote"
```
After `set_side`, the engine auto-enters multi-ponder whenever it is the opponent's turn.

#### `start_multi_ponder`
Begin 5-tree background search with default budget [0.40, 0.20, 0.15, 0.15, 0.10].
```json
{"method":"start_multi_ponder","params":{"k":5,"budget_weights":[0.4,0.2,0.15,0.15,0.1]}}
```
Must be called when it is the opponent's turn. Safe to call before opponent makes a move (the engine uses its policy network to predict opponent's top-k).

#### `opponent_played`
Inform engine what the opponent actually played. If the move matches one of the k pre-computed trees, the engine promotes that tree's root and reports `ponder_hit`. Otherwise, all trees are dropped and the engine starts fresh.
```json
{"method":"opponent_played","params":{"move":"e7e5"}}
```

#### `get_eval_bar`
Instant snapshot of the current root evaluation (for UI eval bar refresh).
```json
{"method":"get_eval_bar","params":{}}
// → {"result":{"score_cp":35,"win_prob":0.58,"depth":24}}
```

#### `get_top_moves`
Return current top-k moves with scores. Works whether or not a search is active.
```json
{"method":"get_top_moves","params":{"k":3}}
// → {"result":{"moves":[{"uci":"Nf3","cp":35,"pv":["Nf3","Nc6"]},...]}}
```

### 4.3 Notifications (engine → client)

#### `policy_preview`
Emitted shortly after `start_multi_ponder` — the engine's initial top-k policy over opponent's moves.
```json
{"method":"policy_preview","params":{"moves":[{"uci":"e7e5","prob":0.32},...]}}
```

#### `ponder_progress`
Emitted every ~500 ms during multi-ponder.
```json
{"method":"ponder_progress","params":{
    "trees":[
        {"tree":0,"opponent_move":"e7e5","depth":22,"score_cp":35,"pv":["Nf3","Nc6","Bb5"]},
        {"tree":1,"opponent_move":"c7c5","depth":18,"score_cp":15,"pv":["Nf3","d6"]},
        ...
    ]
}}
```

#### `ponder_hit`
Sent by engine when `opponent_played` matches one of the pre-computed trees.
```json
{"method":"ponder_hit","params":{"tree":0,"instant_bestmove":"Nf3","score_cp":35}}
```
After this, a normal `bestmove` line follows within the same response.

#### `ponder_miss`
Sent when opponent's move was not in the top-k. All pre-computed work is discarded; engine begins a fresh search.
```json
{"method":"ponder_miss","params":{"trees_discarded":5}}
```

---

## 5. Lifecycle example

Standard single-side + multi-ponder flow:

```
GUI→   uci
ENG→   id name chess_engine ...
ENG→   uciok
GUI→   setoption name NN_Weights value /.../chess_40m.safetensors
GUI→   setoption name MultiPV value 3
GUI→   {"method":"set_side","params":{"me":"white"}}
ENG→   readyok
GUI→   ucinewgame
GUI→   position startpos
GUI→   go wtime 600000 btime 600000 winc 5000 binc 5000

... engine searches as white ...

ENG→   info depth 24 multipv 1 score cp 35 pv e2e4 ...
ENG→   info depth 24 multipv 2 score cp 30 pv d2d4 ...
ENG→   info depth 24 multipv 3 score cp 22 pv c2c4 ...
ENG→   bestmove e2e4

... user plays e2e4 via GUI ...

GUI→   position startpos moves e2e4
GUI→   {"method":"start_multi_ponder","params":{"k":5}}

ENG→   {"method":"policy_preview","params":{"moves":[{"uci":"e7e5","prob":0.32},{"uci":"c7c5","prob":0.24},...]}}
ENG→   {"method":"ponder_progress","params":{"trees":[...]}}       # every 500 ms

... opponent plays e7e5 ...

GUI→   {"method":"opponent_played","params":{"move":"e7e5"}}
ENG→   {"method":"ponder_hit","params":{"tree":0,"instant_bestmove":"Nf3","score_cp":35}}
ENG→   bestmove Nf3
```

---

## 6. Implementation notes

- Engine runs **one persistent subprocess** per game — the GUI does NOT spawn a new process per move.
- Engine is single-binary-per-game (separate `chess_engine` and `shogi_engine` executables, different protocol).
- All state (tree, TT, NN backend) is preserved across `go` calls within a `ucinewgame` session.
- JSON-RPC is line-delimited JSON (LDJSON), **not** JSON-RPC-2.0 over HTTP or WebSocket — it's just a newline-framed form.

---

## References

- UCI: <https://www.chessprogramming.org/UCI>
- Standard USI: <http://hgm.nubati.net/usi.html>
- Multi-PV: <https://talkchess.com/forum3/viewtopic.php?t=69469>
- JSON-RPC 2.0 (we follow the message shape but not transport): <https://www.jsonrpc.org/specification>
