// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The id of a node of a ring arena.

use talos_core::{TypedIndex, TypedIndexTag};

/// The tag of [`Node`].
#[derive(Debug)]
pub struct NodeIndexTag;

impl TypedIndexTag for NodeIndexTag {
    const NAME: &'static str = "Node";
}

/// A node of a [`RingArena`](super::RingArena): a position in its links.
pub type Node = TypedIndex<NodeIndexTag>;
