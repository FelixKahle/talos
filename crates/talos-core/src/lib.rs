// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Foundations shared by the Talos crates.
//!
//! - [`TypedIndex`] makes each kind of id its own type, tagged by a [`TypedIndexTag`].
//! - [`SolverNumeric`] names the integer types a problem's numbers may have, and [`ClosedOpenInterval`] is an
//!   interval `[start, end)` of them.
//! - [`RingArena`] holds a fixed set of [`Node`]s arranged in closed rings, with constant-time splicing.

mod index;
mod math;
mod ring_arena;

pub use index::{TypedIndex, TypedIndexTag};
pub use math::{ClosedOpenInterval, ClosedOpenIntervalIterator, SolverNumeric};
pub use ring_arena::{
    Node, NodeIndexTag, RingArena, RingEdgeIter, RingSequenceIter, RingSequenceRevIter,
};
