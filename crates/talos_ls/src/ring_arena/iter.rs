// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The walks along a ring.

use super::node::Node;
use super::slot::Slot;
use std::iter::FusedIterator;

/// The nodes of a ring from a start along the successors, up to but not including a stop node.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct RingSequenceIter<'a> {
    /// The successor of each node of the arena.
    links: &'a [Node],
    /// The node to yield next; a node of the arena.
    current: Node,
    /// The node at which the walk ends, not yielded.
    stop: Node,
    /// How many more nodes may be yielded at most; bounds the walk to one lap of the arena.
    remaining: usize,
}

impl<'a> RingSequenceIter<'a> {
    pub(super) fn new(links: &'a [Node], start: Slot, stop: Slot) -> Self {
        Self {
            links,
            current: start.0,
            stop: stop.0,
            remaining: links.len(),
        }
    }
}

impl Iterator for RingSequenceIter<'_> {
    type Item = Node;

    #[inline]
    fn next(&mut self) -> Option<Node> {
        if self.current == self.stop || self.remaining == 0 {
            return None;
        }
        let node = self.current;
        debug_assert!(node.index() < self.links.len());
        // SAFETY: `current` is the start, which the arena checked or was promised to lie in it, or a link read from
        // the arena's column, which names one of its nodes (`RingArena`).
        self.current = unsafe { *self.links.get_unchecked(node.index()) };
        self.remaining -= 1;
        Some(node)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let upper = if self.current == self.stop {
            0
        } else {
            self.remaining
        };
        (0, Some(upper))
    }
}

impl FusedIterator for RingSequenceIter<'_> {}

/// The nodes of a ring from a start along the predecessors, up to but not including a stop node.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct RingSequenceRevIter<'a> {
    /// The predecessor of each node of the arena.
    links: &'a [Node],
    /// The node to yield next; a node of the arena.
    current: Node,
    /// The node at which the walk ends, not yielded.
    stop: Node,
    /// How many more nodes may be yielded at most; bounds the walk to one lap of the arena.
    remaining: usize,
}

impl<'a> RingSequenceRevIter<'a> {
    pub(super) fn new(links: &'a [Node], start: Slot, stop: Slot) -> Self {
        Self {
            links,
            current: start.0,
            stop: stop.0,
            remaining: links.len(),
        }
    }
}

impl Iterator for RingSequenceRevIter<'_> {
    type Item = Node;

    #[inline]
    fn next(&mut self) -> Option<Node> {
        if self.current == self.stop || self.remaining == 0 {
            return None;
        }
        let node = self.current;
        debug_assert!(node.index() < self.links.len());
        // SAFETY: `current` is the start, which the arena checked or was promised to lie in it, or a link read from
        // the arena's column, which names one of its nodes (`RingArena`).
        self.current = unsafe { *self.links.get_unchecked(node.index()) };
        self.remaining -= 1;
        Some(node)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let upper = if self.current == self.stop {
            0
        } else {
            self.remaining
        };
        (0, Some(upper))
    }
}

impl FusedIterator for RingSequenceRevIter<'_> {}

/// The links `(node, successor)` of a ring from a start along the successors, up to the link into a stop node.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct RingEdgeIter<'a> {
    /// The successor of each node of the arena.
    links: &'a [Node],
    /// The node whose link to its successor is yielded next; a node of the arena.
    current: Node,
    /// The node at which the walk ends; no link into it is yielded.
    stop: Node,
    /// How many more links may be yielded at most; bounds the walk to one lap of the arena.
    remaining: usize,
}

impl<'a> RingEdgeIter<'a> {
    pub(super) fn new(links: &'a [Node], start: Slot, stop: Slot) -> Self {
        Self {
            links,
            current: start.0,
            stop: stop.0,
            remaining: links.len(),
        }
    }
}

impl Iterator for RingEdgeIter<'_> {
    type Item = (Node, Node);

    #[inline]
    fn next(&mut self) -> Option<(Node, Node)> {
        if self.current == self.stop || self.remaining == 0 {
            return None;
        }
        let from = self.current;
        debug_assert!(from.index() < self.links.len());
        // SAFETY: `current` is the start, which the arena checked or was promised to lie in it, or a link read from
        // the arena's column, which names one of its nodes (`RingArena`).
        let to = unsafe { *self.links.get_unchecked(from.index()) };
        self.current = to;
        if to == self.stop {
            return None;
        }
        self.remaining -= 1;
        Some((from, to))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let upper = if self.current == self.stop {
            0
        } else {
            self.remaining
        };
        (0, Some(upper))
    }
}

impl FusedIterator for RingEdgeIter<'_> {}
