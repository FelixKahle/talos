// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

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
