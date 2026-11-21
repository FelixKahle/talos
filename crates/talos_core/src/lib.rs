// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Foundations shared by the Talos crates.
//!
//! - [`TypedIndex`] makes each kind of id its own type, tagged by a [`TypedIndexTag`].
//! - [`SolverNumeric`] names the integer types a problem's numbers may have, and [`ClosedOpenInterval`] is an
//!   interval `[start, end)` of them.

mod index;
mod math;

pub use index::{TypedIndex, TypedIndexTag};
pub use math::{ClosedOpenInterval, ClosedOpenIntervalIter, SolverNumeric};
