#pragma once

// ----------------------------------------------------------------------------
// engine/include/chess/history_tracker.hpp
//
// HistoryTracker — maintains a fixed-size ring buffer of recent positions for
// use with the LC0 112-plane encoder.
//
// Usage:
//   HistoryTracker tracker;
//   tracker.reset();            // call on UCI "ucinewgame"
//   tracker.push(pos);          // call after each move is made
//   auto& h = tracker.history();// const ref to ordered history vector
//
// The history() vector follows the convention expected by lc0::encode_nn():
//   history[0] = position 1 ply before current (most recent prior position)
//   history[1] = position 2 plies before current
//   ...
//   history[HISTORY_PLIES-1] = position HISTORY_PLIES plies before current
//
// At most HISTORY_PLIES-1 entries are stored (slots for history positions only;
// the current position is passed separately to encode_nn).
// ----------------------------------------------------------------------------

#include <chess/chess_traits.hpp>
#include <chess/lc0_encoding.hpp>

#include <array>
#include <vector>
#include <cstddef>

namespace chess_mlx::chess::lc0 {

class HistoryTracker {
public:
    HistoryTracker() { reset(); }

    // -------------------------------------------------------------------------
    // Reset on UCI "ucinewgame" or when starting a new search from a new FEN.
    // -------------------------------------------------------------------------
    void reset() noexcept {
        head_ = 0;
        size_ = 0;
    }

    // -------------------------------------------------------------------------
    // Push a position into the ring buffer.
    // Call this AFTER the position has already been applied (i.e., push the
    // position BEFORE the current one — the position we're moving FROM).
    //
    // Example sequence:
    //   tracker.reset();
    //   // startpos = pos0
    //   tracker.push(pos0);    // store startpos as "1 ply ago" once we advance
    //   apply_move(pos0, e2e4) // pos0 becomes pos1
    //   encode_nn(pos1, tracker.history(), ...)  // history[0] = pos0
    // -------------------------------------------------------------------------
    void push(const ChessPosition& pos) noexcept {
        buf_[head_] = pos;
        head_ = (head_ + 1) % kMaxHistory;
        if (size_ < kMaxHistory) ++size_;
    }

    // -------------------------------------------------------------------------
    // Returns history as a vector ordered for encode_nn:
    //   index 0 = most recent pushed position (1 ply ago)
    //   index 1 = 2 plies ago, etc.
    //
    // Returns at most HISTORY_PLIES-1 entries (we only need 7 history positions
    // since the encoder accepts the current pos separately).
    // -------------------------------------------------------------------------
    [[nodiscard]] std::vector<ChessPosition> history() const {
        const int n = static_cast<int>(size_);
        std::vector<ChessPosition> result;
        result.reserve(static_cast<std::size_t>(n));
        // head_ points to next write slot; most recent = head_-1 (mod kMaxHistory)
        for (int i = 1; i <= n; ++i) {
            const int idx = (head_ - i + kMaxHistory) % kMaxHistory;
            result.push_back(buf_[idx]);
        }
        return result;
    }

    // -------------------------------------------------------------------------
    // Number of positions currently stored.
    // -------------------------------------------------------------------------
    [[nodiscard]] int size() const noexcept { return static_cast<int>(size_); }

private:
    static constexpr int kMaxHistory = HISTORY_PLIES - 1; // 7 prior positions

    std::array<ChessPosition, kMaxHistory> buf_;
    int head_ = 0;   ///< Next write index
    int size_ = 0;   ///< Number of valid entries
};

} // namespace chess_mlx::chess::lc0
