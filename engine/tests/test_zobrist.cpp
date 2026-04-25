// engine/tests/test_zobrist.cpp
//
// Sanity tests for Zobrist hashing.
//
// This test file is linked only against chess_mlx_core + gtest.
// It tests the ZobristKey type alias and a minimal hand-rolled Zobrist
// implementation to verify the make/undo invariant without pulling in
// chess-library.
//
// The real chess Zobrist tests (which wrap ::chess::Board::hash()) live in
// test_encoding_chess.cpp where chess_mlx_chess is available.

#include <gtest/gtest.h>

#include <core/hash.hpp>
#include <core/game_rules.hpp>

#include <array>
#include <cstdint>
#include <random>

// ============================================================================
// Minimal toy Zobrist implementation for unit testing
// ============================================================================

// Simple position for testing: a tiny 4-square board with 2 piece types.
// Zobrist table indexed by [square][piece_type]; 0 = empty.
// Piece types: 1 = white pawn, 2 = black pawn.

static constexpr int kNumSquares = 4;
static constexpr int kNumPieces  = 3; // 0 = empty, 1 = white, 2 = black

struct ZobristTable {
    chess_mlx::ZobristKey table[kNumSquares][kNumPieces];

    ZobristTable() {
        // Deterministic PRNG for reproducible tests.
        std::mt19937_64 rng(0xDEADBEEF12345678ULL);
        for (auto& row : table) {
            for (auto& v : row) {
                v = rng();
            }
        }
    }
};

static const ZobristTable kTable;

struct TinyBoard {
    std::array<int, kNumSquares> pieces{}; // 0=empty, 1=white, 2=black
    chess_mlx::ZobristKey key{0};

    TinyBoard() {
        // Initial state: white on sq0, black on sq3.
        set(0, 1);
        set(2, 2);
    }

    void set(int sq, int piece) {
        if (pieces[sq] != 0) key ^= kTable.table[sq][pieces[sq]]; // remove old
        pieces[sq] = piece;
        if (piece != 0) key ^= kTable.table[sq][piece]; // add new
    }

    void move_piece(int from, int to) {
        // Remove captured piece if any.
        const int mover = pieces[from];
        set(from, 0);
        set(to, mover);
    }
};

// ============================================================================
// Tests
// ============================================================================

// 1. ZobristKey type is 64-bit.
TEST(ZobristType, Is64Bit) {
    EXPECT_EQ(sizeof(chess_mlx::ZobristKey), sizeof(std::uint64_t));
}

// 2. Initial hash is non-zero.
TEST(ZobristBasic, InitialHashNonZero) {
    TinyBoard b;
    EXPECT_NE(b.key, chess_mlx::ZobristKey{0});
}

// 3. Move piece changes hash.
TEST(ZobristBasic, MoveChangesHash) {
    TinyBoard b;
    const auto original = b.key;
    b.move_piece(0, 1);
    EXPECT_NE(b.key, original);
}

// 4. Move forward then back restores hash (make/undo invariant).
TEST(ZobristBasic, MakeUndoRestoresHash) {
    TinyBoard b;
    const auto original_key   = b.key;
    const auto original_pieces = b.pieces;

    b.move_piece(0, 1); // move white from sq0 to sq1
    EXPECT_NE(b.key, original_key);

    b.move_piece(1, 0); // undo: move back
    EXPECT_EQ(b.key, original_key)
        << "Hash must be identical after move+undo";
    EXPECT_EQ(b.pieces, original_pieces);
}

// 5. Two boards in the same state have the same hash.
TEST(ZobristBasic, SameStateSameHash) {
    TinyBoard a;
    TinyBoard b;
    EXPECT_EQ(a.key, b.key);

    a.move_piece(0, 1);
    b.move_piece(0, 1);
    EXPECT_EQ(a.key, b.key);
}

// 6. Two boards in different states have different hashes.
TEST(ZobristBasic, DifferentStatesHashDiffer) {
    TinyBoard a;
    TinyBoard b;

    a.move_piece(0, 1); // white to sq1
    // b unchanged (white on sq0)
    EXPECT_NE(a.key, b.key);
}

// 7. XOR composition: toggling the same piece twice restores hash.
TEST(ZobristBasic, XorToggle) {
    TinyBoard b;
    const auto h0 = b.key;

    b.set(0, 0); // remove white from sq0
    const auto h1 = b.key;
    EXPECT_NE(h0, h1);

    b.set(0, 1); // restore white to sq0
    EXPECT_EQ(b.key, h0);
}

// 8. Multi-move sequence undo (non-capturing moves only).
TEST(ZobristBasic, MultiMoveUndoSequence) {
    // Use a fresh board with only white on sq0 (no black piece to accidentally capture).
    TinyBoard b;
    // Remove black piece from sq2 to avoid captures.
    b.set(2, 0);
    const auto h0 = b.key;

    // Record state snapshots.
    std::vector<chess_mlx::ZobristKey> snapshots;
    snapshots.push_back(h0);

    // Move 1: white sq0 → sq1
    b.move_piece(0, 1);
    snapshots.push_back(b.key);

    // Move 2: white sq1 → sq2 (now empty, no capture)
    b.move_piece(1, 2);
    snapshots.push_back(b.key);

    // Move 3: white sq2 → sq3
    b.move_piece(2, 3);
    snapshots.push_back(b.key);

    // Verify all intermediate hashes were distinct.
    EXPECT_NE(snapshots[0], snapshots[1]);
    EXPECT_NE(snapshots[1], snapshots[2]);
    EXPECT_NE(snapshots[2], snapshots[3]);

    // Undo move 3.
    b.move_piece(3, 2);
    EXPECT_EQ(b.key, snapshots[2]) << "Hash must be restored after undo of move 3";

    // Undo move 2.
    b.move_piece(2, 1);
    EXPECT_EQ(b.key, snapshots[1]) << "Hash must be restored after undo of move 2";

    // Undo move 1.
    b.move_piece(1, 0);
    EXPECT_EQ(b.key, snapshots[0]) << "Hash must be restored after full undo sequence";
}
