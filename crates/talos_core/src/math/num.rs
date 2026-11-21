// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The integer types a problem's numbers may have.

use std::hash::Hash;

use num_traits::{PrimInt, SaturatingAdd, SaturatingMul, SaturatingSub, Signed};

/// A signed integer type in which a problem states its times, durations and costs.
///
/// The solver is generic over this trait so that a problem is solved in the integer type it is stated in. It is
/// implemented for the signed primitive integers `i8`, `i16`, `i32`, `i64` and `isize`; no type wider than 64
/// bits is one. Beside the arithmetic of [`PrimInt`] and [`Signed`], a type offers saturating arithmetic and
/// names the constants the solver uses.
pub trait SolverNumeric:
    Send
    + Sync
    + PrimInt
    + Signed
    + Hash
    + SaturatingMul
    + SaturatingAdd
    + SaturatingSub
    + std::fmt::Debug
    + std::fmt::Display
{
    /// The least value of the type.
    const MIN: Self;
    /// The greatest value of the type.
    const MAX: Self;
    /// Zero.
    const ZERO: Self;
    /// One.
    const ONE: Self;
    /// Minus one.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants_agree_with_num_traits() {
        fn agree<T: SolverNumeric>() {
            assert_eq!(T::MIN, <T as num_traits::Bounded>::min_value());
            assert_eq!(T::MAX, <T as num_traits::Bounded>::max_value());
            assert_eq!(T::ZERO, T::zero());
            assert_eq!(T::ONE, T::one());
            assert_eq!(T::NEGATIVE_ONE, -T::one());
        }
        agree::<i8>();
        agree::<i16>();
        agree::<i32>();
        agree::<i64>();
        agree::<isize>();
    }
}
