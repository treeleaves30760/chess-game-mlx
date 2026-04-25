// SPDX-License-Identifier: MIT
// engine/src/protocol/json_rpc_ext.cpp
//
// JSON-RPC extension handler — full Phase 6 implementation.

#include "protocol/json_rpc_ext.hpp"

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace chess_mlx::protocol {

// ---------------------------------------------------------------------------
// JsonRpcDispatcher::emit_line — write a JSON string + newline to stdout
// under a mutex so multi-ponder progress emitter and search threads don't race.
// ---------------------------------------------------------------------------
void JsonRpcDispatcher::emit_line(const std::string& line) {
    if (emitter_) {
        emitter_(line);
    } else {
        std::lock_guard<std::mutex> lk(out_mtx_);
        std::cout << line << '\n' << std::flush;
    }
}

void JsonRpcDispatcher::emit(const json& notif) {
    emit_line(notif.dump());
}

// ---------------------------------------------------------------------------
// JsonRpcDispatcher::dispatch — parse and route one JSON-RPC line.
// ---------------------------------------------------------------------------
bool JsonRpcDispatcher::dispatch(const std::string& line) {
    json req;
    try {
        req = json::parse(line);
    } catch (const json::parse_error& e) {
        // Malformed JSON — send parse-error response (no id available).
        json err;
        err["jsonrpc"] = "2.0";
        err["error"]["code"]    = -32700;
        err["error"]["message"] = std::string("Parse error: ") + e.what();
        err["id"]               = nullptr;
        emit_line(err.dump());
        return true;
    }

    // Extract id (may be absent for notifications, but clients always send one).
    const json id_field = req.value("id", json(nullptr));
    const int  id_int   = id_field.is_number_integer()
                          ? id_field.get<int>() : 0;

    const std::string method = req.value("method", std::string{});
    const json params = req.value("params", json::object());

    // -------------------------------------------------------------------
    // Dispatch table
    // -------------------------------------------------------------------

    if (method == "set_side") {
        const std::string side = params.value("me", std::string{});
        if (side.empty()) {
            emit_line(error_response(id_int, -32602,
                "set_side: missing 'me' param").dump());
            return true;
        }
        if (side_cb_) side_cb_(side);
        emit_line(ok_response(id_int, json{{"ok", true}, {"side", side}}).dump());

    } else if (method == "start_multi_ponder") {
        const int k = params.value("k", 5);
        std::vector<double> weights;
        if (params.contains("budget_weights") && params["budget_weights"].is_array()) {
            for (const auto& w : params["budget_weights"]) {
                weights.push_back(w.get<double>());
            }
        } else {
            // Default budget weights.
            weights = {0.40, 0.20, 0.15, 0.15, 0.10};
        }
        if (pstart_cb_) pstart_cb_(k, weights);
        emit_line(ok_response(id_int,
            json{{"ok", true}, {"k", k}}).dump());

    } else if (method == "opponent_played") {
        const std::string move_str = params.value("move", std::string{});
        if (move_str.empty()) {
            emit_line(error_response(id_int, -32602,
                "opponent_played: missing 'move' param").dump());
            return true;
        }
        int hit = -1;
        if (opp_cb_) hit = opp_cb_(move_str);
        json res;
        res["hit"]        = (hit >= 0);
        res["tree_index"] = hit;
        emit_line(ok_response(id_int, res).dump());

    } else if (method == "get_eval_bar") {
        json eval_result = json::object();
        if (eval_cb_) eval_result = eval_cb_();
        emit_line(ok_response(id_int, eval_result).dump());

    } else if (method == "get_top_moves") {
        const int k = params.value("k", 3);
        json moves_result = json::object();
        if (topm_cb_) moves_result = topm_cb_(k);
        emit_line(ok_response(id_int, moves_result).dump());

    } else {
        // Unknown method.
        emit_line(error_response(id_int, -32601,
            std::string("Method not found: '") + method + "'").dump());
    }

    return true;  // continue running
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
json JsonRpcDispatcher::error_response(int id, int code, const std::string& msg) {
    json r;
    r["jsonrpc"]        = "2.0";
    r["error"]["code"]  = code;
    r["error"]["message"] = msg;
    r["id"]             = id;
    return r;
}

json JsonRpcDispatcher::ok_response(int id, const json& result) {
    json r;
    r["jsonrpc"] = "2.0";
    r["result"]  = result;
    r["id"]      = id;
    return r;
}

// ---------------------------------------------------------------------------
// Legacy free-function entry point — now a no-op stub kept for link compat.
// The real handling is done through UciLoop's embedded JsonRpcDispatcher.
// ---------------------------------------------------------------------------
bool process_json_rpc_line(const std::string& /*line*/) {
    return true;
}

} // namespace chess_mlx::protocol
