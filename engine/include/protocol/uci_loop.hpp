// SPDX-License-Identifier: MIT
// engine/include/protocol/uci_loop.hpp
//
// UCI I/O loop for the chess engine.  The implementation is templated on
// the MCTS instance so that chess and shogi can share the protocol layer
// (shogi variant lives in usi_loop.hpp).
//
// IMPORTANT LICENSE NOTE: This header MUST NOT include any chess-specific
// or shogi-specific headers, since both the chess_engine (MIT) and
// shogi_engine (GPL v3) pull it in.  The game-specific traits
// (ChessUciTraits / ShogiUsiTraits) are defined in chess_uci.hpp /
// shogi_usi.hpp respectively.
//
// Phase 6 changes:
//   * Line discriminator: if line[0] == '{' → route to JsonRpcDispatcher.
//   * Embed JsonRpcDispatcher + MultiPonderManager<Traits>.
//   * Wire set_side, start_multi_ponder, opponent_played, get_eval_bar,
//     get_top_moves callbacks.
//   * On search completion in single-side mode, auto-start multi-ponder.

#pragma once

#include "nn/backend.hpp"
#include "search/mcts.hpp"
#include "search/multi_ponder.hpp"
#include "protocol/json_rpc_ext.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace chess_mlx::protocol {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Convert our internal value [-1, +1] to centipawns via the standard
// AlphaZero-style conversion: cp = round(400 * tan(v * pi / 2)).
// Saturate to [-32000, +32000].
inline int value_to_cp(float v) {
    if (v >  0.999f) v =  0.999f;
    if (v < -0.999f) v = -0.999f;
    const double cp = 400.0 * std::tan(static_cast<double>(v) * M_PI * 0.5);
    const int clipped = static_cast<int>(std::round(
        std::max(-32000.0, std::min(32000.0, cp))));
    return clipped;
}

// Split a command-line string on whitespace.
inline std::vector<std::string> tokenise(std::string_view s) {
    std::vector<std::string> out;
    std::size_t i = 0;
    while (i < s.size()) {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) ++i;
        const std::size_t start = i;
        while (i < s.size() && s[i] != ' ' && s[i] != '\t') ++i;
        if (i > start) out.emplace_back(s.substr(start, i - start));
    }
    return out;
}

// ---------------------------------------------------------------------------
// UciLoop<ProtocolTraits> — templated driver
//
// ProtocolTraits provides game-specific hooks (engine_id_name, init_string,
// ok_string, newgame_keyword, start_position, parse_position_command,
// move_to_string, stm_is_white).  See protocol/chess_uci_traits.hpp and
// protocol/shogi_usi_traits.hpp.
// ---------------------------------------------------------------------------
template <typename ProtocolTraits>
class UciLoop {
public:
    using Traits   = typename ProtocolTraits::Traits;
    using Position = typename Traits::Position;
    using Move     = typename Traits::Move;
    using MCTS     = chess_mlx::search::MCTS<Traits>;
    using MPM      = chess_mlx::search::MultiPonderManager<Traits>;

    UciLoop(std::shared_ptr<nn::NNBackend> backend,
            typename MCTS::Config          cfg)
        : backend_(backend)
        , mcts_(backend, std::move(cfg))
        , mponder_mgr_(backend,
                       /*threads_per_tree=*/1,
                       backend->input_size(),
                       backend->policy_size())
    {
        // Wire JSON-RPC dispatcher callbacks.
        json_rpc_.set_emitter([this](const std::string& line) {
            out_line(line);
        });
        json_rpc_.set_side_callback([this](const std::string& side) {
            handle_set_side(side);
        });
        json_rpc_.set_ponder_start_callback(
            [this](int k, std::vector<double> weights) {
                handle_start_multi_ponder(k, std::move(weights));
            });
        json_rpc_.set_opponent_played_callback(
            [this](const std::string& move_str) -> int {
                return handle_opponent_played(move_str);
            });
        json_rpc_.set_eval_bar_callback([this]() -> json {
            return handle_get_eval_bar();
        });
        json_rpc_.set_top_moves_callback([this](int k) -> json {
            return handle_get_top_moves(k);
        });
    }

    // Main loop — reads from stdin until "quit".  Returns 0 on clean exit.
    int run() {
        std::ios::sync_with_stdio(false);
        std::cin.tie(nullptr);

        std::string line;
        while (std::getline(std::cin, line)) {
            if (line.empty()) continue;

            // Phase 6: line discriminator per spec §1.
            if (!line.empty() && line[0] == '{') {
                json_rpc_.dispatch(line);
                continue;
            }

            const auto tokens = tokenise(line);
            if (tokens.empty()) continue;
            const std::string& cmd = tokens[0];

            if (cmd == ProtocolTraits::init_string()) {
                emit_id();
            } else if (cmd == "isready") {
                out_line("readyok");
            } else if (cmd == "setoption") {
                handle_setoption(tokens);
            } else if (cmd == ProtocolTraits::newgame_keyword()) {
                mcts_.reset();
                mponder_mgr_.stop();
                current_pos_ = ProtocolTraits::start_position();
                pos_set_ = true;
            } else if (cmd == "position") {
                handle_position(line);
            } else if (cmd == "go") {
                handle_go(tokens);
            } else if (cmd == "stop") {
                mcts_.stop();
                mponder_mgr_.stop();
                wait_for_search();
            } else if (cmd == "ponderhit") {
                // Standard single-tree ponderhit — not implemented yet.
            } else if (cmd == "quit") {
                mcts_.stop();
                mponder_mgr_.stop();
                wait_for_search();
                break;
            }
            // else: ignore unknown commands
        }
        return 0;
    }

private:
    // -----------------------------------------------------------------------
    // Output
    // -----------------------------------------------------------------------
    void out_line(std::string s) {
        std::lock_guard<std::mutex> lk(out_mtx_);
        std::cout << s << '\n' << std::flush;
    }

    void emit_id() {
        std::lock_guard<std::mutex> lk(out_mtx_);
        std::cout << "id name "   << ProtocolTraits::engine_id_name() << "\n";
        std::cout << "id author Chess_Game_mlx\n";
        std::cout << "option name MultiPV type spin default 1 min 1 max 10\n";
        std::cout << "option name Threads type spin default 1 min 1 max 16\n";
        std::cout << "option name Hash type spin default 512 min 16 max 16384\n";
        std::cout << "option name NN_Weights type string default \"\"\n";
        std::cout << ProtocolTraits::ok_string() << "\n" << std::flush;
    }

    // -----------------------------------------------------------------------
    // UCI command handlers
    // -----------------------------------------------------------------------
    void handle_setoption(const std::vector<std::string>& tokens) {
        auto name_it = std::find(tokens.begin(), tokens.end(), "name");
        auto value_it = std::find(tokens.begin(), tokens.end(), "value");
        if (name_it == tokens.end() || name_it + 1 == tokens.end()) return;

        std::string name = *(name_it + 1);
        std::string value;
        if (value_it != tokens.end() && value_it + 1 != tokens.end()) {
            for (auto it = value_it + 1; it != tokens.end(); ++it) {
                if (!value.empty()) value += ' ';
                value += *it;
            }
        }

        if (name == "MultiPV" || name == "USI_MultiPV") {
            try { mcts_.set_multipv(std::stoi(value)); } catch (...) {}
        } else if (name == "Threads") {
            try { mcts_.set_threads(std::stoi(value)); } catch (...) {}
        } else if (name == "NN_Weights") {
            out_line("info string setoption NN_Weights = " + value
                     + " (requires engine restart to take effect)");
        }
    }

    void handle_position(const std::string& line) {
        // A new position command invalidates any in-flight search, so halt it
        // before updating current_pos_. Otherwise the running search thread
        // keeps emitting info lines for the *old* position, confusing the GUI.
        mcts_.stop();
        mponder_mgr_.stop();
        wait_for_search();
        try {
            std::string_view sv = line;
            if (sv.substr(0, 9) == "position ") sv.remove_prefix(9);
            current_pos_ = ProtocolTraits::parse_position_command(sv);
            pos_set_ = true;
        } catch (const std::exception& e) {
            out_line(std::string("info string bad position: ") + e.what());
        }
    }

    void handle_go(const std::vector<std::string>& tokens) {
        // Stop any ongoing multi-ponder so resources are freed for the search.
        mponder_mgr_.stop();
        // Halt any currently-running search so `wait_for_search()` below can
        // actually join — otherwise a previous `go infinite` would pin us
        // forever waiting for its thread. search() resets stop_ internally
        // at the start, so this is safe.
        mcts_.stop();

        chess_mlx::search::TimeControl tc;
        tc.white_to_move = ProtocolTraits::stm_is_white(current_pos_);
        root_stm_is_white_ = tc.white_to_move;

        for (std::size_t i = 1; i < tokens.size(); ++i) {
            const auto& t = tokens[i];
            auto take_int = [&](int& dst) {
                if (i + 1 < tokens.size()) {
                    try { dst = std::stoi(tokens[++i]); } catch (...) {}
                }
            };
            if      (t == "wtime")     take_int(tc.wtime_ms);
            else if (t == "btime")     take_int(tc.btime_ms);
            else if (t == "winc")      take_int(tc.winc_ms);
            else if (t == "binc")      take_int(tc.binc_ms);
            else if (t == "movestogo") take_int(tc.movestogo);
            else if (t == "depth")     take_int(tc.max_depth);
            else if (t == "nodes")     take_int(tc.max_nodes);
            else if (t == "movetime")  take_int(tc.movetime_ms);
            else if (t == "infinite")  tc.infinite = true;
            else if (t == "ponder")    { /* ignore */ }
            else if (t == "searchmoves") { break; }
        }

        mcts_.set_progress_callback(
            [this](const chess_mlx::search::SearchResult<Traits>& r) {
                emit_info(r);
            });

        // Check if a promoted MCTS is available from a ponder hit.
        auto promoted = mponder_mgr_.take_promoted_mcts();

        wait_for_search();
        search_running_ = true;
        const Position search_pos = current_pos_;

        if (promoted) {
            // Use the promoted MCTS (already warmed up at this position).
            // Transfer ownership to a search thread.
            auto* promoted_raw = promoted.release();
            promoted_raw->set_progress_callback(
                [this](const chess_mlx::search::SearchResult<Traits>& r) {
                    emit_info(r);
                });
            search_thread_ = std::thread([this, search_pos, tc, promoted_raw]() {
                std::unique_ptr<MCTS> owned(promoted_raw);
                auto result = owned->search(search_pos, tc);
                emit_info(result);
                std::string s = "bestmove ";
                s += ProtocolTraits::move_to_string(result.best_move);
                out_line(s);
                search_running_ = false;
                last_result_ = result;
                // Auto-start multi-ponder if in single-side mode.
                auto_start_ponder(search_pos, result);
            });
        } else {
            search_thread_ = std::thread([this, search_pos, tc]() {
                auto result = mcts_.search(search_pos, tc);
                emit_info(result);
                // Debug: report effective batch size.
                {
                    const double avg_bs = mcts_.batcher_avg_batch_size();
                    const auto total_b  = mcts_.batcher_total_batches();
                    const auto total_r  = mcts_.batcher_total_requests();
                    if (total_b > 0) {
                        std::ostringstream oss;
                        oss << "info string batcher batches=" << total_b
                            << " requests=" << total_r
                            << " avg_batch=" << std::fixed;
                        oss.precision(1);
                        oss << avg_bs;
                        out_line(oss.str());
                    }
                }
                std::string s = "bestmove ";
                s += ProtocolTraits::move_to_string(result.best_move);
                out_line(s);
                search_running_ = false;
                last_result_ = result;
                // Auto-start multi-ponder if in single-side mode.
                auto_start_ponder(search_pos, result);
            });
        }
    }

    void wait_for_search() {
        if (search_thread_.joinable()) search_thread_.join();
    }

    void emit_info(const chess_mlx::search::SearchResult<Traits>& r) {
        const int multipv_n = static_cast<int>(r.top_pvs.size());
        const std::uint64_t nps =
            (r.elapsed_ms > 0)
                ? (r.nodes * 1000ULL / std::max<std::uint64_t>(1, r.elapsed_ms))
                : 0;
        for (int k = 0; k < multipv_n; ++k) {
            const auto& pv = r.top_pvs[static_cast<std::size_t>(k)];
            // Skip info lines with no PV — they carry no actionable best-move
            // information and the GUI can't render an arrow for an empty PV
            // (it ends up dropping the slot, which is what causes top-1 to
            // vanish from the board during the first few MCTS iterations).
            if (pv.empty()) continue;
            std::ostringstream oss;
            oss << "info";
            oss << " depth "    << std::max(1, r.seldepth);
            oss << " seldepth " << r.seldepth;
            // Always emit the multipv tag so each slot has a stable identity,
            // even when only one PV exists. Without the tag, the parser
            // defaults to multipv=1, which causes single-PV updates to
            // clobber multi-PV state from earlier in the same search.
            oss << " multipv " << (k + 1);
            const float stm_v = r.top_values[static_cast<std::size_t>(k)];
            const float white_v = root_stm_is_white_ ? stm_v : -stm_v;
            const int cp = value_to_cp(white_v);
            oss << " score cp " << cp;
            oss << " nodes "    << r.nodes;
            oss << " nps "      << nps;
            oss << " time "     << r.elapsed_ms;
            oss << " pv";
            for (const auto& m : pv) {
                oss << " " << ProtocolTraits::move_to_string(m);
            }
            out_line(oss.str());
        }
    }

    // -----------------------------------------------------------------------
    // JSON-RPC method handlers (called by JsonRpcDispatcher callbacks)
    // -----------------------------------------------------------------------

    void handle_set_side(const std::string& side) {
        my_side_ = side;
        single_side_mode_ = !side.empty();
        out_line("info string set_side = " + side);
    }

    void handle_start_multi_ponder(int k, std::vector<double> weights) {
        if (!pos_set_) {
            out_line("info string start_multi_ponder: no position set");
            return;
        }
        // Start multi-ponder from current position (opponent's turn).
        mponder_mgr_.start(
            current_pos_, k, weights,
            [this](const json& notif) {
                out_line(notif.dump());
            });
    }

    int handle_opponent_played(const std::string& move_str) {
        // Parse the move string into a Move.
        Move mv{};
        try {
            mv = ProtocolTraits::parse_move(current_pos_, move_str);
        } catch (...) {
            out_line("info string opponent_played: bad move '" + move_str + "'");
            return -1;
        }
        // Advance our position tracker.
        Traits::apply(current_pos_, mv);
        // Delegate to the multi-ponder manager.
        return mponder_mgr_.opponent_played(mv);
    }

    json handle_get_eval_bar() {
        // Snapshot the last search result.
        json res;
        if (!last_result_.top_values.empty()) {
            const float stm_v = last_result_.top_values[0];
            const float white_v = root_stm_is_white_.load()
                                  ? stm_v : -stm_v;
            res["score_cp"] = value_to_cp(white_v);
            // Win probability from logistic transform of cp.
            res["win_prob"] = 1.0 / (1.0 + std::exp(-static_cast<double>(white_v) * 4.0));
            res["depth"]    = last_result_.seldepth;
        } else {
            res["score_cp"] = 0;
            res["win_prob"] = 0.5;
            res["depth"]    = 0;
        }
        return res;
    }

    json handle_get_top_moves(int k) {
        json res;
        res["moves"] = json::array();
        const int n = std::min(k, static_cast<int>(last_result_.top_pvs.size()));
        for (int i = 0; i < n; ++i) {
            json entry;
            const std::size_t si = static_cast<std::size_t>(i);
            entry["uci"] = last_result_.top_pvs[si].empty()
                           ? ""
                           : ProtocolTraits::move_to_string(last_result_.top_pvs[si][0]);
            entry["cp"]  = (si < last_result_.top_values.size())
                           ? value_to_cp(last_result_.top_values[si]) : 0;
            entry["pv"]  = json::array();
            for (const auto& mv : last_result_.top_pvs[si]) {
                entry["pv"].push_back(ProtocolTraits::move_to_string(mv));
            }
            res["moves"].push_back(entry);
        }
        return res;
    }

    // Auto-start multi-ponder after a search completes (single-side mode).
    void auto_start_ponder(const Position& pos_before_our_move,
                           const chess_mlx::search::SearchResult<Traits>& result) {
        if (!single_side_mode_ || Traits::is_null_move(result.best_move)) return;

        // Build position after our best move (that's now the opponent's turn).
        Position after_our_move = pos_before_our_move;
        Traits::apply(after_our_move, result.best_move);
        {
            std::lock_guard<std::mutex> lk(out_mtx_);
            current_pos_ = after_our_move;
        }

        // Start multi-ponder on that position.
        const std::vector<double> default_weights = {0.40, 0.20, 0.15, 0.15, 0.10};
        mponder_mgr_.start(
            after_our_move,
            /*k=*/5,
            default_weights,
            [this](const json& notif) {
                out_line(notif.dump());
            });
    }

    // -----------------------------------------------------------------------
    // Members
    // -----------------------------------------------------------------------
    std::shared_ptr<nn::NNBackend> backend_;
    MCTS                           mcts_;
    MPM                            mponder_mgr_;
    JsonRpcDispatcher              json_rpc_;

    Position                       current_pos_{};
    bool                           pos_set_{false};

    std::atomic<bool>              search_running_{false};
    std::thread                    search_thread_;
    std::mutex                     out_mtx_;
    std::atomic<bool>              root_stm_is_white_{true};

    // Single-side mode state.
    std::string                    my_side_;
    bool                           single_side_mode_{false};

    // Snapshot of last completed search result (for get_eval_bar etc.).
    chess_mlx::search::SearchResult<Traits> last_result_;
};

// Legacy stub entry point.
void run_uci_loop();

} // namespace chess_mlx::protocol
