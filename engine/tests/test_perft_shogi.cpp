// SPDX-License-Identifier: GPL-3.0-or-later
// engine/tests/test_perft_shogi.cpp
//
// Shogi perft tests and auxiliary correctness tests.
//
// LICENSE NOTE: Links against cshogi (GPL v3).
// Only link into the shogi test binary.
//
// Perft reference counts for standard starting position:
//   perft(1) = 30
//   perft(2) = 900
//   perft(3) = 25470
//   perft(4) = 719731
//   perft(5) = 19861490

#include "shogi/shogi_traits.hpp"
#include "shogi/shogi_encoding.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <string>

using namespace shogi;

// ---------------------------------------------------------------------------
// Perft
// ---------------------------------------------------------------------------

// Recursive perft without any move-ordering or transposition table.
// Returns the number of leaf nodes at the given depth.
static std::uint64_t perft(ShogiPosition& pos, int depth) {
    if (depth == 0) return 1;

    core::MoveList<ShogiMove> ml;
    ShogiTraits::generate_legal(pos, ml);

    if (depth == 1) return static_cast<std::uint64_t>(ml.size());

    std::uint64_t nodes = 0;
    for (const ShogiMove& m : ml) {
        pos.do_move(m);
        nodes += perft(pos, depth - 1);
        pos.undo_move(m);
    }
    return nodes;
}

// ---------------------------------------------------------------------------
// Test fixtures
// ---------------------------------------------------------------------------

class ShogiPerftTest : public ::testing::Test {
protected:
    void SetUp() override {
        ShogiRuntime::ensure_initialized();
    }
};

// Starting position perft.
TEST_F(ShogiPerftTest, StartingPosition_Depth1) {
    ShogiPosition pos;
    EXPECT_EQ(perft(pos, 1), 30ULL);
}

TEST_F(ShogiPerftTest, StartingPosition_Depth2) {
    ShogiPosition pos;
    EXPECT_EQ(perft(pos, 2), 900ULL);
}

TEST_F(ShogiPerftTest, StartingPosition_Depth3) {
    ShogiPosition pos;
    auto t0 = std::chrono::steady_clock::now();
    const std::uint64_t n = perft(pos, 3);
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    EXPECT_EQ(n, 25470ULL);
    EXPECT_LT(ms, 30000.0) << "perft(3) took too long: " << ms << " ms";
}

TEST_F(ShogiPerftTest, StartingPosition_Depth4) {
    ShogiPosition pos;
    auto t0 = std::chrono::steady_clock::now();
    const std::uint64_t n = perft(pos, 4);
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    EXPECT_EQ(n, 719731ULL);
    // Skip timing assertion at depth 4 — it may be slow in debug builds.
    (void)ms;
}

// ---------------------------------------------------------------------------
// Square mapping test (encoding_spec.md §3.1 vs cshogi Square)
// ---------------------------------------------------------------------------

TEST(ShogiSquareMapping, AllSquares) {
    ShogiRuntime::ensure_initialized();
    // Spec: square_index = (9 - file) * 9 + (rank - 1)
    // where file ∈ {1..9}, rank ∈ {1..9}.
    // cshogi: makeSquare(File f, Rank r) = f*9 + r (0-indexed).
    // Expected relationship:
    //   spec_sq = (8 - cshogi_file) * 9 + cshogi_rank
    // where cshogi_file = file-1 and cshogi_rank = rank-1.
    //
    // Verify cshogi_to_spec_sq and spec_to_cshogi_sq are inverses.

    for (int file = 1; file <= 9; ++file) {
        for (int rank = 1; rank <= 9; ++rank) {
            const int spec_sq_expected = (9 - file) * 9 + (rank - 1);

            // cshogi convention: File1=0, Rank1=0.
            const File csq_file = static_cast<File>(file - 1);
            const Rank csq_rank = static_cast<Rank>(rank - 1);
            const Square csq = makeSquare(csq_file, csq_rank);

            const int spec_sq_actual = encoding::cshogi_to_spec_sq(csq);
            EXPECT_EQ(spec_sq_actual, spec_sq_expected)
                << "file=" << file << " rank=" << rank;

            // Round-trip: spec → cshogi → spec.
            const Square csq_rt = encoding::spec_to_cshogi_sq(spec_sq_expected);
            EXPECT_EQ(csq_rt, csq)
                << "file=" << file << " rank=" << rank;
        }
    }
}

// Specific corner cases documented in encoding_spec.md.
TEST(ShogiSquareMapping, SpecificSquares) {
    ShogiRuntime::ensure_initialized();
    // SQ91 (file 9, rank 1) → spec 0.
    EXPECT_EQ(encoding::cshogi_to_spec_sq(SQ91), 0);
    // SQ11 (file 1, rank 1) → spec 72.
    EXPECT_EQ(encoding::cshogi_to_spec_sq(SQ11), 72);
    // SQ19 (file 1, rank 9) → spec 80.
    EXPECT_EQ(encoding::cshogi_to_spec_sq(SQ19), 80);
    // SQ99 (file 9, rank 9) → spec 8.
    EXPECT_EQ(encoding::cshogi_to_spec_sq(SQ99), 8);
}

// ---------------------------------------------------------------------------
// Policy round-trip test
// ---------------------------------------------------------------------------
// For 10 diverse positions, verify that for every legal move m:
//   policy_idx_to_move(move_to_policy_idx(m, pos), pos) == m

static void policy_roundtrip(const std::string& sfen) {
    ShogiRuntime::ensure_initialized();
    ShogiPosition pos(sfen);
    core::MoveList<ShogiMove> ml;
    ShogiTraits::generate_legal(pos, ml);

    for (const ShogiMove& m : ml) {
        const int idx = ShogiTraits::move_to_policy_idx(m, pos);
        ASSERT_GE(idx, 0)    << "sfen=" << sfen << " move=" << m.to_usi();
        ASSERT_LT(idx, 2187) << "sfen=" << sfen << " move=" << m.to_usi();

        const ShogiMove rt = ShogiTraits::policy_idx_to_move(idx, pos);
        EXPECT_EQ(rt, m)
            << "sfen=" << sfen << " move=" << m.to_usi()
            << " idx=" << idx << " rt=" << rt.to_usi();
    }
}

TEST(ShogiPolicyMapping, RoundTripStartPos) {
    policy_roundtrip("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1");
}

TEST(ShogiPolicyMapping, RoundTripAfter1e) {
    // After 1.7g7f (▲7六歩)
    policy_roundtrip("lnsgkgsnl/1r5b1/ppppppppp/9/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL w - 2");
}

TEST(ShogiPolicyMapping, RoundTripAfter1e2e) {
    // After 1.7g7f 3c3d
    policy_roundtrip("lnsgkgsnl/1r5b1/pppppp1pp/6p2/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL b - 3");
}

TEST(ShogiPolicyMapping, RoundTripMiddlegame1) {
    // Standard Fujii System variation — 先手の飛車が動いた局面
    policy_roundtrip("ln1g1gsnl/1r4kb1/p1ppps1pp/6p2/1p7/2P6/PPBPPPPPP/3S1R3/LN1GKGSNL b Pp 15");
}

TEST(ShogiPolicyMapping, RoundTripMiddlegame2) {
    // 後手の持ち駒あり (White to move with hand pieces)
    policy_roundtrip("ln1g1gsnl/1r4kb1/p1ppps1pp/6p2/1p7/2P6/PPBPPPPPP/3S1R3/LN1GKGSNL w Pp 16");
}

TEST(ShogiPolicyMapping, RoundTripPromotion) {
    // Position where promotions are available
    policy_roundtrip("lnsgkgsnl/1r5b1/p1ppppppp/9/9/1pP6/PP1PPPPPP/1B5R1/LNSGKGSNL b - 5");
}

TEST(ShogiPolicyMapping, RoundTripDrops) {
    // Position with hand pieces for drops
    policy_roundtrip("8l/1r5b1/p1ppppppp/9/9/1p7/PPPPsPPPP/1B4R2/LNSGKGSNL b GNPp 9");
}

TEST(ShogiPolicyMapping, RoundTripInCheck) {
    // Position in check (limited evasion moves)
    policy_roundtrip("lnsgkgsn1/7b1/p1ppppppp/9/9/1p7/PPPPPPPPP/7R1/LNSGKGSNL b rL 7");
}

TEST(ShogiPolicyMapping, RoundTripBishopExchange) {
    // After bishop exchange (角交換) — promoted pieces likely
    policy_roundtrip("ln1gkgsnl/1r5b1/p1pp1pppp/4p4/1p7/2P6/PPBPPPPPP/7R1/LNSGKGSNL b S 11");
}

TEST(ShogiPolicyMapping, RoundTripEndgame) {
    // Endgame-ish position with promoted pieces
    policy_roundtrip("3+R4l/4kg3/p3ppnpp/9/3s5/1B7/P4PPPP/4G4/LN2K1SNL b GSN3Prbslnp2g 45");
}

// ---------------------------------------------------------------------------
// Encoding smoke test
// ---------------------------------------------------------------------------

TEST(ShogiEncoding, ShapeAndSideToMoveBlack) {
    ShogiRuntime::ensure_initialized();
    ShogiPosition pos; // starting position, Black to move.
    float buf[81 * 90] = {};
    ShogiTraits::encode_nn(pos, buf);

    // Channel 89 should be 1.0 for all squares (Black to move).
    for (int sq = 0; sq < 81; ++sq) {
        EXPECT_FLOAT_EQ(buf[sq * 90 + 89], 1.0f)
            << "square " << sq;
    }
}

TEST(ShogiEncoding, ShapeAndSideToMoveWhite) {
    ShogiRuntime::ensure_initialized();
    // One move in: 7g7f
    const std::string sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL w - 2";
    ShogiPosition pos(sfen);
    float buf[81 * 90] = {};
    ShogiTraits::encode_nn(pos, buf);

    // Channel 89 should be 0.0 for all squares (White to move).
    for (int sq = 0; sq < 81; ++sq) {
        EXPECT_FLOAT_EQ(buf[sq * 90 + 89], 0.0f)
            << "square " << sq;
    }
}

TEST(ShogiEncoding, StartingPositionPieceCount) {
    ShogiRuntime::ensure_initialized();
    ShogiPosition pos; // Black to move.
    float buf[81 * 90] = {};
    ShogiTraits::encode_nn(pos, buf);

    // Count own-side (Black) pawns: channel 0, should be exactly 9 squares.
    int pawn_count = 0;
    for (int sq = 0; sq < 81; ++sq) {
        if (buf[sq * 90 + 0] > 0.5f) ++pawn_count;
    }
    EXPECT_EQ(pawn_count, 9) << "Black pawns at start";

    // Count opponent (White) pawns: channel 14, should be 9.
    int opp_pawn_count = 0;
    for (int sq = 0; sq < 81; ++sq) {
        if (buf[sq * 90 + 14] > 0.5f) ++opp_pawn_count;
    }
    EXPECT_EQ(opp_pawn_count, 9) << "White pawns at start";

    // No hand pieces at start: channels 56..69 all zero.
    for (int sq = 0; sq < 81; ++sq) {
        for (int ch = 56; ch <= 69; ++ch) {
            EXPECT_FLOAT_EQ(buf[sq * 90 + ch], 0.0f)
                << "sq=" << sq << " ch=" << ch;
        }
    }
}
