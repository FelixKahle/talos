// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A fixed set of nodes arranged in closed rings, spliced in constant time.
//!
//! - [`Node`] names a node of an arena.
//! - `Slot` is a node its arena knows to lie inside it, the proof behind every unchecked access.
//! - [`RingSequenceIter`], [`RingSequenceRevIter`] and [`RingEdgeIter`] walk along a ring.
//! - [`RingArena`] holds the rings and the steps that rearrange them.

mod node;
mod slot;

mod iter;

mod arena;

pub use arena::RingArena;
pub use iter::{RingEdgeIter, RingSequenceIter, RingSequenceRevIter};
pub use node::Node;
