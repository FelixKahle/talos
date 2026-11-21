// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Foundations shared by the Talos crates.
//!
//! - [`TypedIndex`] makes each kind of id its own type, tagged by a [`TypedIndexTag`], and [`TypedIndexIter`]
//!   walks every id of a kind up to a count.
//! - [`SolverNumeric`] names the integer types a problem's numbers may have, and [`ClosedOpenInterval`] is an
//!   interval `[start, end)` of them; bounds out of order are refused as [`StartAfterEnd`].

mod index;
mod index_iter;
mod math;

pub use index::{TypedIndex, TypedIndexTag};
pub use index_iter::TypedIndexIter;
pub use math::{ClosedOpenInterval, ClosedOpenIntervalIter, SolverNumeric, StartAfterEnd};
