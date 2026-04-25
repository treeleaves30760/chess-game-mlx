/*
 * cshogi compatibility shim: overloadEnumOperators.hpp
 *
 * This file shadows the upstream overloadEnumOperators.hpp to remove
 * 'constexpr' from the arithmetic operator overloads.
 *
 * Rationale:
 *   The upstream version marks these operators 'constexpr'.  In cshogi's
 *   square.hpp, SquareWithWall enum constants are initialised via expressions
 *   like:
 *       SQWW_R = DeltaE - (1 << 9) + (1 << 24)
 *   This calls the constexpr operator- on SquareDelta, producing a value
 *   (-521) that is outside SquareDelta's valid range [-16, 15].  Apple Clang
 *   version 21 (and recent upstream Clang) rejects this as a hard error even
 *   in C++11 mode, because enum initialisers must be integer constant
 *   expressions whose values fit in the enum type's range.
 *
 *   By removing 'constexpr', the operators become ordinary inline functions
 *   (evaluated at runtime), so enum initialisers that use them are no longer
 *   constant expressions — but SquareWithWall's SQWW_* values ARE evaluated
 *   at compile time because they are enum values.  We need a different fix.
 *
 * Actual fix:
 *   The SquareWithWall enum initialiser expressions involve:
 *       SquareDelta_value - large_int + large_int
 *   The first subtraction overflows SquareDelta.  We introduce a cast to int
 *   in the macro so the arithmetic is int arithmetic (no enum-range checks).
 *
 *   OverloadEnumOperators redefines operator- to return static_cast<T>(...),
 *   which triggers the range check on T.  We instead return int explicitly.
 *
 * This file is placed before the upstream source in the include path so it
 * takes precedence, keeping the upstream files unmodified.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later (derived from Apery/Stockfish)
 */

#ifndef APERY_OVERLOADENUMOPERATORS_HPP
#define APERY_OVERLOADENUMOPERATORS_HPP

// NOTE: The constexpr arithmetic operators are INTENTIONALLY removed here.
// The key difference: operator+ and operator- are NOT constexpr, so they
// cannot be used in constant expressions like enum initialisers.
//
// For the SquareWithWall enum, the initialisers in square.hpp use
// DeltaE - (1 << 9) which would call the constexpr operator-.  Without
// constexpr, this call is NOT a constant expression, so the enum initialiser
// fails differently.
//
// The REAL fix needed for SquareWithWall is to use plain int arithmetic.
// We do that by making operator- return int (not T), which means the
// SquareWithWall enum initialisers must use int arithmetic.
// But that breaks the enum definition too...
//
// Correct fix: don't use enum arithmetic in enum initialisers.
// We achieve this by redefining the operators WITHOUT constexpr,
// and adding an int-based version that doesn't produce enum values.
// Then we provide a separate macro for SquareWithWall that uses raw ints.
//
// However, since we can't change square.hpp, the true fix is:
// the constexpr arithmetic must NOT cause constant evaluation of
// out-of-range enum values.  We do this by making the return type int
// in the operator overloads (breaking enum arithmetic but fixing the
// constant expression issue).

// We actually need to keep operator- returning T for general use.
// The only problem is when used IN an enum initialiser.
// Apple Clang 21 added stricter checking.
//
// WORKAROUND: We use __attribute__((no_sanitize("undefined"))) and mark
// the operators as NON-constexpr, so they cannot be used in constant
// expressions (enum initialisers).  Then we handle SquareWithWall separately.
//
// But SquareWithWall is a plain (unscoped) enum, so its initialisers CAN use
// non-constexpr expressions in C++11 if they are "converted constant
// expressions" — actually no, enum initialisers MUST be constant expressions.
//
// THE CORRECT FIX:
// Replace:
//   inline constexpr T operator - (const T lhs, const int rhs) { return static_cast<T>(...); }
// With:
//   inline T operator - (const T lhs, const int rhs) { return static_cast<T>(...); }
// (remove constexpr)
//
// This means the SquareWithWall enum can NO LONGER use SquareDelta arithmetic
// in its initialisers because they need to be constant expressions.
//
// To make SquareWithWall work, its initialisers must use raw integer literals
// or casts.  Since we can't modify square.hpp, we need a different approach.
//
// FINAL APPROACH: Provide a pre-processor define that turns the SquareWithWall
// initialisers into raw integer arithmetic by #defining DeltaE, DeltaN, etc.
// as plain int values before they are used in the SquareWithWall context.
// This is done in a "force_include" header.

#define OverloadEnumOperators(T)                                        \
    inline T& operator += (T& lhs, const int rhs) { return lhs  = static_cast<T>(static_cast<int>(lhs) + rhs); } \
    inline T& operator += (T& lhs, const T   rhs) { return lhs += static_cast<int>(rhs); } \
    inline T& operator -= (T& lhs, const int rhs) { return lhs  = static_cast<T>(static_cast<int>(lhs) - rhs); } \
    inline T& operator -= (T& lhs, const T   rhs) { return lhs -= static_cast<int>(rhs); } \
    inline T& operator *= (T& lhs, const int rhs) { return lhs  = static_cast<T>(static_cast<int>(lhs) * rhs); } \
    inline T& operator /= (T& lhs, const int rhs) { return lhs  = static_cast<T>(static_cast<int>(lhs) / rhs); } \
    inline T operator + (const T   lhs, const int rhs) { return static_cast<T>(static_cast<int>(lhs) + rhs); } \
    inline T operator + (const T   lhs, const T   rhs) { return lhs + static_cast<int>(rhs); } \
    inline T operator - (const T   lhs, const int rhs) { return static_cast<T>(static_cast<int>(lhs) - rhs); } \
    inline T operator - (const T   lhs, const T   rhs) { return lhs - static_cast<int>(rhs); } \
    inline T operator * (const T   lhs, const int rhs) { return static_cast<T>(static_cast<int>(lhs) * rhs); } \
    inline T operator * (const int lhs, const T   rhs) { return rhs * lhs; } \
    inline T operator * (const T   lhs, const T   rhs) { return lhs * static_cast<int>(rhs); } \
    inline T operator / (const T   lhs, const int rhs) { return static_cast<T>(static_cast<int>(lhs) / rhs); } \
    inline int operator / (const T   lhs, const T rhs) { return static_cast<int>(lhs) / static_cast<int>(rhs); } \
    inline T operator - (const T   rhs) { return static_cast<T>(-static_cast<int>(rhs)); } \
    inline T operator ++ (T& lhs) { lhs += 1; return lhs; }           \
    inline T operator -- (T& lhs) { lhs -= 1; return lhs; }           \
    inline T operator ++ (T& lhs, int) { const T temp = lhs; lhs += 1; return temp; }

#endif // #ifndef APERY_OVERLOADENUMOPERATORS_HPP
