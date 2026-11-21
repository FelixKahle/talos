// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A node known to lie in its arena.

use super::node::Node;
use std::fmt;

/// A node of one arena that is known to lie inside it.
///
/// A slot is the proof behind every unchecked access of the arena: whatever column of the arena it indexes, the
/// access is in bounds.
// Invariant: the node lies below the length of the one arena the slot was made for, and that length does not change
// while the slot exists. Only the arena makes slots, through `new_unchecked`: from a node it has checked at its door
// or been promised by the caller of an `_unchecked` step, or from an entry of one of its own columns, each of which
// names one of its nodes. A slot never outlives the method or the borrow of the arena that made it, so the length
// cannot change under it, and no method of the arena holds slots of two arenas, so a slot is only ever read against
// the columns of the arena that made it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct Slot {
    node: Node,
}

impl Slot {
    /// The slot of `node`.
    ///
    /// # Safety
    ///
    /// `node` must lie in the arena the slot is used with, and that arena's length must not change while the slot
    /// exists: every later access through the slot trusts it.
    // No `debug_assert!` stands here: the slot does not know its arena; each caller asserts the bound before the call.
    #[inline]
    pub(super) const unsafe fn new_unchecked(node: Node) -> Self {
        Self { node }
    }

    /// The node the slot holds.
    #[inline]
    pub(super) const fn node(self) -> Node {
        self.node
    }
}

impl fmt::Display for Slot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.node)
    }
}
