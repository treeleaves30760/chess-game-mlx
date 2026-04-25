#pragma once

// ----------------------------------------------------------------------------
// engine/include/protocol/json_rpc_ext.hpp
//
// JSON-RPC extension handler for the Chess_Game_mlx protocol extension (§4 of
// protocol_spec.md).
//
// Handles:
//   Requests:  set_side, start_multi_ponder, opponent_played, get_eval_bar,
//              get_top_moves
//   Notifications (engine→client): policy_preview, ponder_progress,
//              ponder_hit, ponder_miss  (emitted by MultiPonderManager)
//
// Usage pattern:
//   JsonRpcDispatcher disp(backend, mcts_config);
//   disp.set_current_position(pos);          // call after every 'position'
//   disp.dispatch(line);                     // call when line[0]=='{' on stdin
//
// The dispatcher owns one MultiPonderManager and integrates it with the UCI
// loop state via mutable position/MCTS references set by the loop owner.
// ----------------------------------------------------------------------------

#include "nn/backend.hpp"
#include "search/mcts.hpp"

#include <nlohmann_json/json.hpp>

#include <functional>
#include <memory>
#include <mutex>
#include <string>

namespace chess_mlx::protocol {

using json = nlohmann::json;

// ---------------------------------------------------------------------------
// JsonRpcDispatcher
//
// Game-agnostic core: handles JSON line parsing and dispatch.
// The game-specific move parse/emit lambdas are injected at construction time
// so the same object can serve both chess (UCI) and shogi (USI) loops.
// ---------------------------------------------------------------------------
class JsonRpcDispatcher {
public:
    // Move-string → opaque move ID (string key for matching).
    // For chess: UCI string "e7e5" → "e7e5"
    // For shogi: USI string "7g7f" → "7g7f"
    using MoveParser   = std::function<std::string(const std::string&)>;

    // Emit a line to stdout.
    using Emitter      = std::function<void(const std::string&)>;

    // Callback invoked when set_side is processed.
    using SideCallback = std::function<void(const std::string&)>;  // "white"/"black"/"sente"/"gote"

    // Callback invoked when start_multi_ponder is requested.
    // Receives k and budget_weights.
    using PonderStartCallback = std::function<void(int k, std::vector<double>)>;

    // Callback invoked when opponent_played is processed.
    // Returns hit index (0..k-1) or -1 for miss.
    using OpponentPlayedCallback = std::function<int(const std::string& move_str)>;

    // Callback for get_eval_bar.
    using EvalBarCallback = std::function<json()>;

    // Callback for get_top_moves.
    using TopMovesCallback = std::function<json(int k)>;

    JsonRpcDispatcher() = default;

    // Wire callbacks (call before dispatch()).
    void set_side_callback          (SideCallback cb)           { side_cb_   = std::move(cb); }
    void set_ponder_start_callback  (PonderStartCallback cb)    { pstart_cb_ = std::move(cb); }
    void set_opponent_played_callback(OpponentPlayedCallback cb){ opp_cb_    = std::move(cb); }
    void set_eval_bar_callback      (EvalBarCallback cb)        { eval_cb_   = std::move(cb); }
    void set_top_moves_callback     (TopMovesCallback cb)       { topm_cb_   = std::move(cb); }
    void set_emitter                (Emitter em)                { emitter_   = std::move(em); }

    // Process one incoming JSON-RPC line.
    // Returns false if the engine should shut down ("quit" method, if ever used).
    bool dispatch(const std::string& line);

    // Emit a JSON notification to the client (thread-safe).
    void emit(const json& notif);

private:
    json  error_response(int id, int code, const std::string& msg);
    json  ok_response   (int id, const json& result);

    void  emit_line(const std::string& line);

    SideCallback             side_cb_;
    PonderStartCallback      pstart_cb_;
    OpponentPlayedCallback   opp_cb_;
    EvalBarCallback          eval_cb_;
    TopMovesCallback         topm_cb_;
    Emitter                  emitter_;

    std::mutex               out_mtx_;
};

// ---------------------------------------------------------------------------
// Legacy free-function entry point (Phase 1 stub replacement).
// This is kept for link compatibility; the real implementation delegates to
// a global JsonRpcDispatcher created in uci_loop.cpp.
// ---------------------------------------------------------------------------
bool process_json_rpc_line(const std::string& line);

} // namespace chess_mlx::protocol
