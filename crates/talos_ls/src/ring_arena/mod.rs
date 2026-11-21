// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A fixed set of nodes arranged in closed rings, spliced in constant time.
//!
//! - [`Node`] names a node of an arena.
//! - `Slot` is a node its arena knows to lie inside it, the proof behind every unchecked access.
//! - [`RingSequenceIter`] walks along a ring in either direction, and [`RingLinkIter`] yields the links it passes.
//! - [`RingArena`] holds the rings and the steps that rearrange them.

mod node;
mod slot;

mod iter;

mod arena;

#[cfg(test)]
mod ring_lists;

pub(super) use arena::RingArena;
pub(super) use iter::RingSequenceIter;
pub(super) use node::Node;
