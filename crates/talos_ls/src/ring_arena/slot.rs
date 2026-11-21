// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A node known to lie in its arena.

use super::node::Node;

/// A node of one arena that is known to lie inside it.
///
/// Only the arena makes one: from a node it has checked or been promised to lie inside it, or by reading one of its
/// own links, every one of which is a node of the arena by its invariant. A slot never leaves the method of the
/// arena that made it, and the arena's length does not change while that method runs, so the bound holds for as
/// long as the slot exists.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct Slot(pub(super) Node);
