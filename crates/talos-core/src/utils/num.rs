// Copyright (c) 2026 Felix Kahe.
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

use std::hash::Hash;

use num_traits::{PrimInt, Saturating, SaturatingAdd, SaturatingMul, SaturatingSub, Signed, Zero};

/// A trait alias for numeric types that can be used in the solver.
/// This includes integer types that support various arithmetic operations
/// with both saturating and checked semantics.
/// These are usually all signed integer types `i8`, `i16`, `i32`, `i64` and `isize`.
///
/// # Note
///
/// `i128` are intentionally excluded due to performance reasons, as
/// they are significantly slower on many platforms.
pub trait SolverNumeric:
    Send
    + Sync
    + PrimInt
    + Signed
    + Zero
    + Hash
    + Saturating
    + SaturatingMul
    + SaturatingAdd
    + SaturatingSub
    + std::fmt::Debug
    + std::fmt::Display
{
    const MIN: Self;
    const MAX: Self;
    const ZERO: Self;
    const ONE: Self;
    const NEGATIVE_ONE: Self;
}

impl SolverNumeric for i64 {
    const MIN: Self = Self::MIN;
    const MAX: Self = Self::MAX;
    const ZERO: Self = 0;
    const ONE: Self = 1;
    const NEGATIVE_ONE: Self = -1;
}

impl SolverNumeric for i32 {
    const MIN: Self = Self::MIN;
    const MAX: Self = Self::MAX;
    const ZERO: Self = 0;
    const ONE: Self = 1;
    const NEGATIVE_ONE: Self = -1;
}

impl SolverNumeric for i16 {
    const MIN: Self = Self::MIN;
    const MAX: Self = Self::MAX;
    const ZERO: Self = 0;
    const ONE: Self = 1;
    const NEGATIVE_ONE: Self = -1;
}

impl SolverNumeric for i8 {
    const MIN: Self = Self::MIN;
    const MAX: Self = Self::MAX;
    const ZERO: Self = 0;
    const ONE: Self = 1;
    const NEGATIVE_ONE: Self = -1;
}

impl SolverNumeric for isize {
    const MIN: Self = Self::MIN;
    const MAX: Self = Self::MAX;
    const ZERO: Self = 0;
    const ONE: Self = 1;
    const NEGATIVE_ONE: Self = -1;
}
