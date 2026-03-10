// Copyright (c) 2026 Felix Kahle.
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

//! High-performance, allocation-free topological representation of closed rings.
//!
//! # Architecture
//!
//! Because the graph strictly consists of valid, closed rings, these two arrays are exact
//! mathematical inverses: `prev[next[v]] == v` for all `v`. Nodes can never enter or leave
//! the graph once initialized.

use std::iter::FusedIterator;

use crate::utils::index::{TypedIndex, TypedIndexTag};

// ----------------------------------------------------------------
// Node
// ----------------------------------------------------------------

/// Marker type for `Node` indices in the arena.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeIndexTag;

impl TypedIndexTag for NodeIndexTag {
    const NAME: &'static str = "Node";
}

/// A newtype wrapper around `usize` representing a node in the arena.
///
/// The node itself is also the index into the internal `prev` and `next` arrays
/// of the `RingArena`, which is the primary reason to the incredible
/// performance of this data structure. The `Node` type is a thin wrapper to provide
/// type safety and clarity, while still allowing for efficient conversions to and from `usize`.
pub type Node = TypedIndex<NodeIndexTag>;

// ----------------------------------------------------------------
// RingSequenceIter
// ----------------------------------------------------------------

/// Iterator over a sequence in the arena, starting from a specific node and ending at a stop node.
#[derive(Clone, PartialEq, Eq)]
pub struct RingSequenceIter<'a> {
    /// The `next` array borrowed from the owning `RingArena`.
    next_pointers: &'a [Node],
    /// Current cursor into the ring.
    current_node: Node,
    /// The node at which iteration should stop (exclusive).
    stop_node: Node,
    /// Guards against infinite loops if the graph becomes disjoint.
    remaining: usize,
}

impl<'a> Iterator for RingSequenceIter<'a> {
    type Item = Node;

    #[inline(always)]
    fn next(&mut self) -> Option<Node> {
        if self.current_node == self.stop_node || self.remaining == 0 {
            return None;
        }

        let current = self.current_node;
        self.current_node = self.next_pointers[current.get()];
        self.remaining -= 1;

        Some(current)
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.remaining))
    }
}

impl<'a> FusedIterator for RingSequenceIter<'a> {}

// ----------------------------------------------------------------
// RingSequenceRevIter
// ----------------------------------------------------------------

/// Reverse iterator over a sequence in the arena, starting from a specific node and ending at a stop node.
#[derive(Clone, PartialEq, Eq)]
pub struct RingSequenceRevIter<'a> {
    /// The `prev` array borrowed from the owning `RingArena`.
    prev_pointers: &'a [Node],
    /// Current cursor into the ring.
    current_node: Node,
    /// The node at which iteration should stop (exclusive).
    stop_node: Node,
    /// Guards against infinite loops if the graph becomes disjoint.
    remaining: usize,
}

impl<'a> Iterator for RingSequenceRevIter<'a> {
    type Item = Node;

    #[inline(always)]
    fn next(&mut self) -> Option<Node> {
        if self.current_node == self.stop_node || self.remaining == 0 {
            return None;
        }

        let current = self.current_node;
        self.current_node = self.prev_pointers[current.get()];
        self.remaining -= 1;

        Some(current)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.remaining))
    }
}

impl<'a> FusedIterator for RingSequenceRevIter<'a> {}

// ----------------------------------------------------------------
// RingEdgeIter
// ----------------------------------------------------------------

/// Iterator over the edges (adjacent node pairs) within a sequence.
#[derive(Clone, Debug)]
pub struct RingEdgeIter<'a> {
    /// The `next` array borrowed from the owning `RingArena`.
    next_pointers: &'a [Node],
    /// The left-hand side of the edge we are about to yield.
    current_node: Node,
    /// The node at which iteration should stop.
    stop_node: Node,
    /// The remaining number of edges to yield before giving up to prevent infinite loops.
    remaining: usize,
}

impl<'a> Iterator for RingEdgeIter<'a> {
    type Item = (Node, Node);

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_node == self.stop_node || self.remaining == 0 {
            return None;
        }

        let from = self.current_node;
        let to = self.next_pointers[from.get()];

        if to == self.stop_node {
            self.current_node = self.stop_node;
            return None;
        }

        self.current_node = to;
        self.remaining -= 1;

        Some((from, to))
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.remaining))
    }
}

impl<'a> FusedIterator for RingEdgeIter<'a> {}

// ----------------------------------------------------------------
// RingArena
// ----------------------------------------------------------------

/// High-performance, arena-backed collection of circular, doubly-linked lists.
///
/// # Note
///
/// Nodes can never enter or leave the graph once initialized.
/// The `prev` and `next` arrays are exact mathematical inverses: `prev[next[v]] == v` for all `v`.
///
/// # Safety and `_unchecked` Methods
///
/// For maximum performance during high-throughput local search iterations, this struct
/// exposes many `_unchecked` methods that omit bounds checking.
///
/// **Calling `_unchecked` methods is only safe if:**
/// 1. The provided indices are strictly within the bounds of the arena allocation.
/// 2. When dealing with segments (`first` to `last`), the nodes must be sequentially
///    linked in the same ring, and the `target` anchor must **not** be part of that segment.
#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct RingArena {
    /// O(1) lookup: the node immediately preceding this one.
    prev: Vec<Node>,

    /// O(1) lookup: the node immediately following this one.
    next: Vec<Node>,
}

impl RingArena {
    /// Creates a new `RingArena` from parallel slices.
    ///
    /// # Panics
    ///
    /// Panics if `prev.len() != next.len()`.
    #[inline]
    pub fn new(prev: Vec<Node>, next: Vec<Node>) -> Self {
        assert_eq!(prev.len(), next.len());

        Self { prev, next }
    }

    /// Overwrites the current arena with data from parallel slices to avoid reallocation.
    #[inline]
    pub fn overwrite_from_slices(&mut self, prev: &[Node], next: &[Node]) {
        assert_eq!(prev.len(), next.len());

        self.prev.clear();
        self.prev.extend_from_slice(prev);
        self.next.clear();
        self.next.extend_from_slice(next);
    }

    /// Overwrites the current arena with data from another arena.
    #[inline]
    pub fn overwrite_from_arena(&mut self, other: &RingArena) {
        self.prev.clone_from(&other.prev);
        self.next.clone_from(&other.next);
    }

    /// Resets the arena to a state where each node forms a self-loop (i.e., a ring of length 1).
    #[inline]
    pub fn reset_to_self_loops(&mut self, new_len: usize) {
        self.prev.clear();
        self.next.clear();
        self.prev.extend((0..new_len).map(Node::new));
        self.next.extend((0..new_len).map(Node::new));
    }

    /// Resizes the arena to exactly `new_len` nodes.
    ///
    /// If growing, new slots are initialized to `Node(0)` (not valid rings —
    /// the caller must fix them up). If shrinking, excess nodes are dropped.
    /// Existing data for indices `< min(old_len, new_len)` is preserved.
    #[inline]
    pub fn resize(&mut self, new_len: usize) {
        self.prev.resize(new_len, Node::new(0));
        self.next.resize(new_len, Node::new(0));
    }

    /// Returns mutable references to the raw `prev` and `next` arrays
    /// for direct bulk initialization.
    ///
    /// # Safety
    ///
    /// After modifying the returned slices, the caller **must** ensure that
    /// all entries form valid closed rings (i.e., `prev[next[i]] == i` and
    /// `next[prev[i]] == i` for all `i`) before calling any other method
    /// on this arena.
    #[inline(always)]
    pub unsafe fn raw_mut(&mut self) -> (&mut [Node], &mut [Node]) {
        (&mut self.prev, &mut self.next)
    }

    /// Returns the total number of nodes in the arena.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.prev.len()
    }

    /// Returns `true` if the arena has no nodes.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.prev.is_empty()
    }

    /// Returns the internal `next` pointer for the given node.
    #[inline]
    pub fn next(&self, node: Node) -> Node {
        self.next[node.get()]
    }

    /// Returns the internal `next` pointer for the given node.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `node < self.next.len()`.
    #[inline(always)]
    pub unsafe fn next_unchecked(&self, node: Node) -> Node {
        debug_assert!(node < self.next.len());

        *unsafe { self.next.get_unchecked(node.get()) }
    }

    /// Returns the internal `prev` pointer for the given node.
    #[inline]
    pub fn prev(&self, node: Node) -> Node {
        debug_assert!(node < self.prev.len());

        self.prev[node.get()]
    }

    /// Returns the internal `prev` pointer for the given node.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `node < self.prev.len()`.
    #[inline(always)]
    pub unsafe fn prev_unchecked(&self, node: Node) -> Node {
        debug_assert!(node < self.prev.len());

        *unsafe { self.prev.get_unchecked(node.get()) }
    }

    /// Returns an iterator over a sequence of nodes starting from `start_node`
    /// and ending right before it reaches `stop_node`.
    ///
    /// # Panics
    ///
    /// Panics if either `start_node` or `stop_node` is out of bounds.
    #[inline]
    pub fn sequence_iter(&self, start_node: Node, stop_node: Node) -> RingSequenceIter<'_> {
        assert!(start_node < self.next.len());
        assert!(stop_node < self.next.len());

        let remaining = self.next.len();
        RingSequenceIter {
            next_pointers: &self.next,
            current_node: start_node,
            stop_node,
            remaining,
        }
    }

    /// Returns an iterator over a sequence of nodes starting from `start_node`
    /// and ending right before it reaches `stop_node`.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `start_node` and `stop_node` are valid indices within the arena.
    /// No bounds checking is performed.
    #[inline]
    pub unsafe fn sequence_iter_unchecked(
        &self,
        start_node: Node,
        stop_node: Node,
    ) -> RingSequenceIter<'_> {
        debug_assert!(start_node < self.next.len());
        debug_assert!(stop_node < self.next.len());

        let remaining = self.next.len();
        RingSequenceIter {
            next_pointers: &self.next,
            current_node: start_node,
            remaining,
            stop_node,
        }
    }

    /// Returns a reverse iterator over a sequence of nodes starting from `start_node`
    /// and ending right before it reaches `stop_node`.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `start_node` and `stop_node` are valid indices within the arena.
    /// No bounds checking is performed.
    #[inline]
    pub unsafe fn sequence_rev_iter_unchecked(
        &self,
        start_node: Node,
        stop_node: Node,
    ) -> RingSequenceRevIter<'_> {
        debug_assert!(start_node < self.prev.len());
        debug_assert!(stop_node < self.prev.len());

        let remaining = self.prev.len();
        RingSequenceRevIter {
            prev_pointers: &self.prev,
            current_node: start_node,
            remaining,
            stop_node,
        }
    }

    /// Returns an iterator over a sequence of nodes starting from `start_node`
    /// and ending right before it reaches `stop_node`.
    ///
    /// # Panics
    ///
    /// Panics if either `start_node` or `stop_node` is out of bounds.
    #[inline]
    pub fn edge_iter(&self, start_node: Node, stop_node: Node) -> RingEdgeIter<'_> {
        assert!(start_node < self.next.len());
        assert!(stop_node < self.next.len());

        let remaining = self.next.len();
        RingEdgeIter {
            next_pointers: &self.next,
            current_node: start_node,
            stop_node,
            remaining,
        }
    }

    /// Returns an iterator over all edges (adjacent node pairs) within a sequence.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `start_node` and `stop_node` are valid indices within the arena.
    /// No bounds checking is performed.
    #[inline]
    pub unsafe fn edge_iter_unchecked(
        &self,
        start_node: Node,
        stop_node: Node,
    ) -> RingEdgeIter<'_> {
        debug_assert!(start_node < self.next.len());
        debug_assert!(stop_node < self.next.len());

        let remaining = self.next.len();
        RingEdgeIter {
            next_pointers: &self.next,
            current_node: start_node,
            stop_node,
            remaining,
        }
    }

    /// Detaches a node from its current position in the ring.
    ///
    /// After this call, the node's own `prev` and `next` slots are **stale**. The
    /// caller must reinsert it or otherwise fix them up before the graph is observed again.
    #[inline(always)]
    unsafe fn extract_node_unchecked(&mut self, node_to_extract: Node) {
        debug_assert!(node_to_extract < self.prev.len());

        let predecessor = *unsafe { self.prev.get_unchecked(node_to_extract.get()) };
        let successor = *unsafe { self.next.get_unchecked(node_to_extract.get()) };

        *unsafe { self.next.get_unchecked_mut(predecessor.get()) } = successor;
        *unsafe { self.prev.get_unchecked_mut(successor.get()) } = predecessor;
    }

    /// Inserts a node immediately following the `insertion_point` node.
    #[inline(always)]
    unsafe fn insert_node_after_unchecked(&mut self, node_to_insert: Node, insertion_point: Node) {
        debug_assert!(node_to_insert < self.prev.len());
        debug_assert!(insertion_point < self.prev.len());

        let successor_of_insertion_point =
            *unsafe { self.next.get_unchecked(insertion_point.get()) };

        *unsafe { self.next.get_unchecked_mut(insertion_point.get()) } = node_to_insert;
        *unsafe { self.prev.get_unchecked_mut(node_to_insert.get()) } = insertion_point;

        *unsafe { self.next.get_unchecked_mut(node_to_insert.get()) } =
            successor_of_insertion_point;
        *unsafe {
            self.prev
                .get_unchecked_mut(successor_of_insertion_point.get())
        } = node_to_insert;
    }

    /// Swaps the positions of two nodes in the arena.
    ///
    /// ```text
    /// Before:
    /// Ring A: ... <-> A_Prev <-> Node1 <-> A_Next <-> ...
    /// Ring B: ... <-> B_Prev <-> Node2 <-> B_Next <-> ...
    ///
    /// After:
    /// Ring A: ... <-> A_Prev <-> Node2 <-> A_Next <-> ...
    /// Ring B: ... <-> B_Prev <-> Node1 <-> B_Next <-> ...
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if either `first` or `second` is out of bounds.
    #[inline]
    pub fn swap_nodes(&mut self, first: Node, second: Node) {
        debug_assert!(first < self.len());
        debug_assert!(second < self.len());

        if first == second {
            return;
        }

        let first_prev = self.prev[first.get()];
        let first_next = self.next[first.get()];
        let second_prev = self.prev[second.get()];
        let second_next = self.next[second.get()];

        let first_is_solo = first_prev == first;
        let second_is_solo = second_prev == second;

        if first_is_solo && second_is_solo {
            // Both are self-loops — swapping is a no-op.
        } else if first_is_solo {
            self.next[second_prev.get()] = first;
            self.prev[first.get()] = second_prev;
            self.next[first.get()] = second_next;
            self.prev[second_next.get()] = first;

            self.next[second.get()] = second;
            self.prev[second.get()] = second;
        } else if second_is_solo {
            self.next[first_prev.get()] = second;
            self.prev[second.get()] = first_prev;
            self.next[second.get()] = first_next;
            self.prev[first_next.get()] = second;

            self.next[first.get()] = first;
            self.prev[first.get()] = first;
        } else if first_next == second {
            // Adjacent: first -> second
            self.next[first_prev.get()] = second;
            self.prev[second.get()] = first_prev;
            self.next[second.get()] = first;
            self.prev[first.get()] = second;
            self.next[first.get()] = second_next;
            self.prev[second_next.get()] = first;
        } else if second_next == first {
            // Adjacent: second -> first
            self.next[second_prev.get()] = first;
            self.prev[first.get()] = second_prev;
            self.next[first.get()] = second;
            self.prev[second.get()] = first;
            self.next[second.get()] = first_next;
            self.prev[first_next.get()] = second;
        } else {
            // Non-adjacent
            self.next[first_prev.get()] = second;
            self.prev[second.get()] = first_prev;
            self.next[second.get()] = first_next;
            self.prev[first_next.get()] = second;

            self.next[second_prev.get()] = first;
            self.prev[first.get()] = second_prev;
            self.next[first.get()] = second_next;
            self.prev[second_next.get()] = first;
        }
    }

    /// Swaps the positions of two nodes in the arena.
    ///
    /// ```text
    /// Before:
    /// Ring A: ... <-> A_Prev <-> Node1 <-> A_Next <-> ...
    /// Ring B: ... <-> B_Prev <-> Node2 <-> B_Next <-> ...
    ///
    /// After:
    /// Ring A: ... <-> A_Prev <-> Node2 <-> A_Next <-> ...
    /// Ring B: ... <-> B_Prev <-> Node1 <-> B_Next <-> ...
    /// ```
    ///
    /// # Safety
    ///
    /// Both `first` and `second` must be valid node indices. No bounds checking is performed.
    #[inline]
    pub unsafe fn swap_nodes_unchecked(&mut self, first: Node, second: Node) {
        debug_assert!(first < self.len());
        debug_assert!(second < self.len());

        if first == second {
            return;
        }

        let first_prev = *unsafe { self.prev.get_unchecked(first.get()) };
        let first_next = *unsafe { self.next.get_unchecked(first.get()) };
        let second_prev = *unsafe { self.prev.get_unchecked(second.get()) };
        let second_next = *unsafe { self.next.get_unchecked(second.get()) };

        let first_is_solo = first_prev == first;
        let second_is_solo = second_prev == second;

        if first_is_solo && second_is_solo {
            // Both are self-loops — swapping is a no-op.
        } else if first_is_solo {
            // First is a self-loop. Extract second from its ring, put first in second's
            // old position, make second a self-loop.
            *unsafe { self.next.get_unchecked_mut(second_prev.get()) } = first;
            *unsafe { self.prev.get_unchecked_mut(first.get()) } = second_prev;
            *unsafe { self.next.get_unchecked_mut(first.get()) } = second_next;
            *unsafe { self.prev.get_unchecked_mut(second_next.get()) } = first;

            *unsafe { self.next.get_unchecked_mut(second.get()) } = second;
            *unsafe { self.prev.get_unchecked_mut(second.get()) } = second;
        } else if second_is_solo {
            // Second is a self-loop. Extract first from its ring, put second in first's
            // old position, make first a self-loop.
            *unsafe { self.next.get_unchecked_mut(first_prev.get()) } = second;
            *unsafe { self.prev.get_unchecked_mut(second.get()) } = first_prev;
            *unsafe { self.next.get_unchecked_mut(second.get()) } = first_next;
            *unsafe { self.prev.get_unchecked_mut(first_next.get()) } = second;

            *unsafe { self.next.get_unchecked_mut(first.get()) } = first;
            *unsafe { self.prev.get_unchecked_mut(first.get()) } = first;
        } else if first_next == second {
            // Adjacent: first -> second
            *unsafe { self.next.get_unchecked_mut(first_prev.get()) } = second;
            *unsafe { self.prev.get_unchecked_mut(second.get()) } = first_prev;
            *unsafe { self.next.get_unchecked_mut(second.get()) } = first;
            *unsafe { self.prev.get_unchecked_mut(first.get()) } = second;
            *unsafe { self.next.get_unchecked_mut(first.get()) } = second_next;
            *unsafe { self.prev.get_unchecked_mut(second_next.get()) } = first;
        } else if second_next == first {
            // Adjacent: second -> first
            *unsafe { self.next.get_unchecked_mut(second_prev.get()) } = first;
            *unsafe { self.prev.get_unchecked_mut(first.get()) } = second_prev;
            *unsafe { self.next.get_unchecked_mut(first.get()) } = second;
            *unsafe { self.prev.get_unchecked_mut(second.get()) } = first;
            *unsafe { self.next.get_unchecked_mut(second.get()) } = first_next;
            *unsafe { self.prev.get_unchecked_mut(first_next.get()) } = second;
        } else {
            // Non-adjacent, neither is a self-loop
            *unsafe { self.next.get_unchecked_mut(first_prev.get()) } = second;
            *unsafe { self.prev.get_unchecked_mut(second.get()) } = first_prev;
            *unsafe { self.next.get_unchecked_mut(second.get()) } = first_next;
            *unsafe { self.prev.get_unchecked_mut(first_next.get()) } = second;

            *unsafe { self.next.get_unchecked_mut(second_prev.get()) } = first;
            *unsafe { self.prev.get_unchecked_mut(first.get()) } = second_prev;
            *unsafe { self.next.get_unchecked_mut(first.get()) } = second_next;
            *unsafe { self.prev.get_unchecked_mut(second_next.get()) } = first;
        }
    }

    /// Swaps the positions of two contiguous segments of nodes.
    ///
    /// ```text
    /// Before:
    /// Ring A: ... <-> A_Prev <-> [ A_First ... A_Last ] <-> A_Next <-> ...
    /// Ring B: ... <-> B_Prev <-> [ B_First ... B_Last ] <-> B_Next <-> ...
    ///
    /// After:
    /// Ring A: ... <-> A_Prev <-> [ B_First ... B_Last ] <-> A_Next <-> ...
    /// Ring B: ... <-> B_Prev <-> [ A_First ... A_Last ] <-> B_Next <-> ...
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if any of the indices are out of bounds.
    #[inline]
    pub fn swap_segments(&mut self, a_first: Node, a_last: Node, b_first: Node, b_last: Node) {
        debug_assert!(a_first < self.len());
        debug_assert!(a_last < self.len());
        debug_assert!(b_first < self.len());
        debug_assert!(b_last < self.len());

        if a_first == b_first {
            return;
        }

        let a_pred = self.prev[a_first.get()];
        let a_succ = self.next[a_last.get()];
        let b_pred = self.prev[b_first.get()];
        let b_succ = self.next[b_last.get()];

        if a_succ == b_first && b_succ == a_first {
            // Full-ring case
            self.next[b_last.get()] = a_first;
            self.prev[a_first.get()] = b_last;
            self.next[a_last.get()] = b_first;
            self.prev[b_first.get()] = a_last;
        } else if a_succ == b_first {
            // Adjacent: A immediately before B
            self.next[a_pred.get()] = b_first;
            self.prev[b_first.get()] = a_pred;
            self.next[b_last.get()] = a_first;
            self.prev[a_first.get()] = b_last;
            self.next[a_last.get()] = b_succ;
            self.prev[b_succ.get()] = a_last;
        } else if b_succ == a_first {
            // Adjacent: B immediately before A
            self.next[b_pred.get()] = a_first;
            self.prev[a_first.get()] = b_pred;
            self.next[a_last.get()] = b_first;
            self.prev[b_first.get()] = a_last;
            self.next[b_last.get()] = a_succ;
            self.prev[a_succ.get()] = b_last;
        } else {
            // Non-adjacent
            self.next[a_pred.get()] = b_first;
            self.prev[b_first.get()] = a_pred;
            self.next[b_last.get()] = a_succ;
            self.prev[a_succ.get()] = b_last;

            self.next[b_pred.get()] = a_first;
            self.prev[a_first.get()] = b_pred;
            self.next[a_last.get()] = b_succ;
            self.prev[b_succ.get()] = a_last;
        }
    }

    /// Swaps the positions of two contiguous segments of nodes.
    ///
    /// ```text
    /// Before:
    /// Ring A: ... <-> A_Prev <-> [ A_First ... A_Last ] <-> A_Next <-> ...
    /// Ring B: ... <-> B_Prev <-> [ B_First ... B_Last ] <-> B_Next <-> ...
    ///
    /// After:
    /// Ring A: ... <-> A_Prev <-> [ B_First ... B_Last ] <-> A_Next <-> ...
    /// Ring B: ... <-> B_Prev <-> [ A_First ... A_Last ] <-> B_Next <-> ...
    /// ```
    ///
    /// # Safety
    ///
    /// All segments must form a valid contiguous path and must not overlap.
    /// No bounds checking is performed.
    #[inline]
    pub unsafe fn swap_segments_unchecked(
        &mut self,
        a_first: Node,
        a_last: Node,
        b_first: Node,
        b_last: Node,
    ) {
        debug_assert!(a_first < self.len());
        debug_assert!(a_last < self.len());
        debug_assert!(b_first < self.len());
        debug_assert!(b_last < self.len());

        if a_first == b_first {
            return;
        }

        let a_pred = *unsafe { self.prev.get_unchecked(a_first.get()) };
        let a_succ = *unsafe { self.next.get_unchecked(a_last.get()) };
        let b_pred = *unsafe { self.prev.get_unchecked(b_first.get()) };
        let b_succ = *unsafe { self.next.get_unchecked(b_last.get()) };

        if a_succ == b_first && b_succ == a_first {
            // Full-ring case: both segments together cover the entire ring.
            // Just rotate: swap the connection point between them.
            unsafe {
                *self.next.get_unchecked_mut(b_last.get()) = a_first;
                *self.prev.get_unchecked_mut(a_first.get()) = b_last;
                *self.next.get_unchecked_mut(a_last.get()) = b_first;
                *self.prev.get_unchecked_mut(b_first.get()) = a_last;
            }
        } else if a_succ == b_first {
            // Adjacent: A immediately before B
            unsafe {
                *self.next.get_unchecked_mut(a_pred.get()) = b_first;
                *self.prev.get_unchecked_mut(b_first.get()) = a_pred;

                *self.next.get_unchecked_mut(b_last.get()) = a_first;
                *self.prev.get_unchecked_mut(a_first.get()) = b_last;

                *self.next.get_unchecked_mut(a_last.get()) = b_succ;
                *self.prev.get_unchecked_mut(b_succ.get()) = a_last;
            }
        } else if b_succ == a_first {
            // Adjacent: B immediately before A
            unsafe {
                *self.next.get_unchecked_mut(b_pred.get()) = a_first;
                *self.prev.get_unchecked_mut(a_first.get()) = b_pred;

                *self.next.get_unchecked_mut(a_last.get()) = b_first;
                *self.prev.get_unchecked_mut(b_first.get()) = a_last;

                *self.next.get_unchecked_mut(b_last.get()) = a_succ;
                *self.prev.get_unchecked_mut(a_succ.get()) = b_last;
            }
        } else {
            // Non-adjacent
            unsafe {
                *self.next.get_unchecked_mut(a_pred.get()) = b_first;
                *self.prev.get_unchecked_mut(b_first.get()) = a_pred;

                *self.next.get_unchecked_mut(b_last.get()) = a_succ;
                *self.prev.get_unchecked_mut(a_succ.get()) = b_last;

                *self.next.get_unchecked_mut(b_pred.get()) = a_first;
                *self.prev.get_unchecked_mut(a_first.get()) = b_pred;

                *self.next.get_unchecked_mut(a_last.get()) = b_succ;
                *self.prev.get_unchecked_mut(b_succ.get()) = a_last;
            }
        }
    }

    /// Relocates a single node to immediately follow the target anchor.
    ///
    /// ```text
    /// Before:
    /// Source: ... <-> Prev <-> Subject <-> Next <-> ...
    /// Target: ... <-> Anchor <-> Anchor_Next <-> ...
    ///
    /// After:
    /// Source: ... <-> Prev <-----------> Next <-> ...
    /// Target: ... <-> Anchor <-> Subject <-> Anchor_Next <-> ...
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if either `node` or `anchor` is out of bounds.
    #[inline]
    pub fn relocate_after(&mut self, node: Node, anchor: Node) {
        debug_assert!(node < self.len());
        debug_assert!(anchor < self.len());

        if node == anchor {
            return;
        }
        if self.prev[node.get()] == anchor {
            return;
        }

        // Extract node
        let predecessor = self.prev[node.get()];
        let successor = self.next[node.get()];
        self.next[predecessor.get()] = successor;
        self.prev[successor.get()] = predecessor;

        // Insert after anchor
        let successor_of_anchor = self.next[anchor.get()];
        self.next[anchor.get()] = node;
        self.prev[node.get()] = anchor;
        self.next[node.get()] = successor_of_anchor;
        self.prev[successor_of_anchor.get()] = node;
    }

    /// Relocates a single node to immediately follow the target anchor.
    ///
    /// ```text
    /// Before:
    /// Source: ... <-> Prev <-> Subject <-> Next <-> ...
    /// Target: ... <-> Anchor <-> Anchor_Next <-> ...
    ///
    /// After:
    /// Source: ... <-> Prev <-----------> Next <-> ...
    /// Target: ... <-> Anchor <-> Subject <-> Anchor_Next <-> ...
    /// ```
    ///
    /// # Safety
    ///
    /// Valid bounds must be respected. No bounds checking is performed.
    #[inline]
    pub unsafe fn relocate_after_unchecked(&mut self, node: Node, anchor: Node) {
        debug_assert!(node < self.len());
        debug_assert!(anchor < self.len());

        if node == anchor {
            return;
        }
        if *unsafe { self.prev.get_unchecked(node.get()) } == anchor {
            return;
        }
        unsafe { self.extract_node_unchecked(node) };
        unsafe { self.insert_node_after_unchecked(node, anchor) };
    }

    /// Relocates a contiguous segment of nodes to immediately follow the target anchor.
    ///
    /// ```text
    /// Before:
    /// Source: ... <-> Prev <-> [ First ... Last ] <-> Next <-> ...
    /// Target: ... <-> Anchor <-> Anchor_Next <-> ...
    ///
    /// After:
    /// Source: ... <-> Prev <--------------------> Next <-> ...
    /// Target: ... <-> Anchor <-> [ First ... Last ] <-> Anchor_Next <-> ...
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if any of the indices are out of bounds.
    #[inline]
    pub fn relocate_segment_after(&mut self, first: Node, last: Node, anchor: Node) {
        debug_assert!(first < self.len());
        debug_assert!(last < self.len());
        debug_assert!(anchor < self.len());

        if self.prev[first.get()] == anchor {
            return;
        }

        let before_segment = self.prev[first.get()];
        let after_segment = self.next[last.get()];

        self.next[before_segment.get()] = after_segment;
        self.prev[after_segment.get()] = before_segment;

        let successor_of_anchor = self.next[anchor.get()];

        self.next[anchor.get()] = first;
        self.prev[first.get()] = anchor;
        self.next[last.get()] = successor_of_anchor;
        self.prev[successor_of_anchor.get()] = last;
    }

    /// Relocates a contiguous segment of nodes to immediately follow the target anchor.
    ///
    /// ```text
    /// Before:
    /// Source: ... <-> Prev <-> [ First ... Last ] <-> Next <-> ...
    /// Target: ... <-> Anchor <-> Anchor_Next <-> ...
    ///
    /// After:
    /// Source: ... <-> Prev <--------------------> Next <-> ...
    /// Target: ... <-> Anchor <-> [ First ... Last ] <-> Anchor_Next <-> ...
    /// ```
    ///
    /// # Safety
    ///
    /// Valid bounds must be respected. No bounds checking is performed.
    #[inline]
    pub unsafe fn relocate_segment_after_unchecked(
        &mut self,
        first: Node,
        last: Node,
        anchor: Node,
    ) {
        debug_assert!(first < self.len());
        debug_assert!(last < self.len());
        debug_assert!(anchor < self.len());

        if *unsafe { self.prev.get_unchecked(first.get()) } == anchor {
            return;
        }

        let before_segment = *unsafe { self.prev.get_unchecked(first.get()) };
        let after_segment = *unsafe { self.next.get_unchecked(last.get()) };

        unsafe {
            *self.next.get_unchecked_mut(before_segment.get()) = after_segment;
            *self.prev.get_unchecked_mut(after_segment.get()) = before_segment;
        }

        let successor_of_anchor = *unsafe { self.next.get_unchecked(anchor.get()) };

        unsafe {
            *self.next.get_unchecked_mut(anchor.get()) = first;
            *self.prev.get_unchecked_mut(first.get()) = anchor;

            *self.next.get_unchecked_mut(last.get()) = successor_of_anchor;
            *self.prev.get_unchecked_mut(successor_of_anchor.get()) = last;
        }
    }

    /// Relocates a single node to immediately precede the target anchor.
    ///
    /// ```text
    /// Before:
    /// Source: ... <-> Prev <-> Subject <-> Next <-> ...
    /// Target: ... <-> Anchor_Prev <-> Anchor <-> ...
    ///
    /// After:
    /// Source: ... <-> Prev <-----------> Next <-> ...
    /// Target: ... <-> Anchor_Prev <-> Subject <-> Anchor <-> ...
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if either `node` or `anchor` is out of bounds.
    #[inline]
    pub fn relocate_before(&mut self, node: Node, anchor: Node) {
        debug_assert!(node < self.len());
        debug_assert!(anchor < self.len());

        let anchor_predecessor = self.prev[anchor.get()];
        self.relocate_after(node, anchor_predecessor);
    }

    /// Relocates a single node to immediately precede the target anchor.
    ///
    /// ```text
    /// Before:
    /// Source: ... <-> Prev <-> Subject <-> Next <-> ...
    /// Target: ... <-> Anchor_Prev <-> Anchor <-> ...
    ///
    /// After:
    /// Source: ... <-> Prev <-----------> Next <-> ...
    /// Target: ... <-> Anchor_Prev <-> Subject <-> Anchor <-> ...
    /// ```
    ///
    /// # Safety
    ///
    /// Valid bounds must be respected. No bounds checking is performed.
    #[inline]
    pub unsafe fn relocate_before_unchecked(&mut self, node: Node, anchor: Node) {
        debug_assert!(node < self.len());
        debug_assert!(anchor < self.len());

        let anchor_predecessor = *unsafe { self.prev.get_unchecked(anchor.get()) };
        unsafe { self.relocate_after_unchecked(node, anchor_predecessor) };
    }

    /// Relocates a contiguous segment of nodes to immediately precede the target anchor.
    ///
    /// ```text
    /// Before:
    /// Source: ... <-> Prev <-> [ First ... Last ] <-> Next <-> ...
    /// Target: ... <-> Anchor_Prev <-> Anchor <-> ...
    ///
    /// After:
    /// Source: ... <-> Prev <--------------------> Next <-> ...
    /// Target: ... <-> Anchor_Prev <-> [ First ... Last ] <-> Anchor <-> ...
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if any of the indices are out of bounds.
    #[inline]
    pub fn relocate_segment_before(&mut self, first: Node, last: Node, anchor: Node) {
        debug_assert!(first < self.len());
        debug_assert!(last < self.len());
        debug_assert!(anchor < self.len());

        let anchor_predecessor = self.prev[anchor.get()];
        self.relocate_segment_after(first, last, anchor_predecessor);
    }

    /// Relocates a contiguous segment of nodes to immediately precede the target anchor.
    ///
    /// ```text
    /// Before:
    /// Source: ... <-> Prev <-> [ First ... Last ] <-> Next <-> ...
    /// Target: ... <-> Anchor_Prev <-> Anchor <-> ...
    ///
    /// After:
    /// Source: ... <-> Prev <--------------------> Next <-> ...
    /// Target: ... <-> Anchor_Prev <-> [ First ... Last ] <-> Anchor <-> ...
    /// ```
    ///
    /// # Safety
    ///
    /// Valid bounds must be respected. No bounds checking is performed.
    #[inline]
    pub unsafe fn relocate_segment_before_unchecked(
        &mut self,
        first: Node,
        last: Node,
        anchor: Node,
    ) {
        debug_assert!(first < self.len());
        debug_assert!(last < self.len());
        debug_assert!(anchor < self.len());

        let anchor_predecessor = *unsafe { self.prev.get_unchecked(anchor.get()) };
        unsafe { self.relocate_segment_after_unchecked(first, last, anchor_predecessor) };
    }

    /// Reverses the order of a contiguous segment of nodes.
    ///
    /// ```text
    /// Before:
    /// ... <-> Prev <-> [ A <-> B <-> C <-> D ] <-> Next <-> ...
    ///                    ^                 ^
    ///                  first              last
    ///
    /// After:
    /// ... <-> Prev <-> [ D <-> C <-> B <-> A ] <-> Next <-> ...
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if either `segment_first` or `segment_last` is out of bounds.
    #[inline]
    pub fn reverse_segment(&mut self, first: Node, last: Node) {
        debug_assert!(first < self.len());
        debug_assert!(last < self.len());

        if first == last {
            return;
        }

        let predecessor_of_segment = self.prev[first.get()];
        let successor_of_segment = self.next[last.get()];

        let mut current_node = first;

        loop {
            let original_next = self.next[current_node.get()];
            self.prev.swap(current_node.get(), current_node.get()); // need proper swap implementation for indices
            let temp = self.prev[current_node.get()];
            self.prev[current_node.get()] = self.next[current_node.get()];
            self.next[current_node.get()] = temp;

            if current_node == last {
                break;
            }
            current_node = original_next;
        }

        self.next[first.get()] = successor_of_segment;
        self.prev[last.get()] = predecessor_of_segment;

        self.next[predecessor_of_segment.get()] = last;
        self.prev[successor_of_segment.get()] = first;
    }

    /// Reverses the order of a contiguous segment of nodes.
    ///
    /// ```text
    /// Before:
    /// ... <-> Prev <-> [ A <-> B <-> C <-> D ] <-> Next <-> ...
    ///                    ^                 ^
    ///                  first              last
    ///
    /// After:
    /// ... <-> Prev <-> [ D <-> C <-> B <-> A ] <-> Next <-> ...
    /// ```
    ///
    /// # Safety
    ///
    /// Valid bounds must be respected, and `first` / `last` must form a valid contiguous path.
    /// No bounds checking is performed.
    #[inline]
    pub unsafe fn reverse_segment_unchecked(&mut self, first: Node, last: Node) {
        debug_assert!(first < self.len());
        debug_assert!(last < self.len());

        if first == last {
            return;
        }

        let predecessor_of_segment = *unsafe { self.prev.get_unchecked(first.get()) };
        let successor_of_segment = *unsafe { self.next.get_unchecked(last.get()) };

        let prev_ptr = self.prev.as_mut_ptr();
        let next_ptr = self.next.as_mut_ptr();
        let mut current_node = first;

        loop {
            let original_next = *unsafe { self.next.get_unchecked(current_node.get()) };
            unsafe {
                std::ptr::swap(
                    prev_ptr.add(current_node.get()),
                    next_ptr.add(current_node.get()),
                );
            }
            if current_node == last {
                break;
            }
            current_node = original_next;
        }

        *unsafe { self.next.get_unchecked_mut(first.get()) } = successor_of_segment;
        *unsafe { self.prev.get_unchecked_mut(last.get()) } = predecessor_of_segment;

        *unsafe { self.next.get_unchecked_mut(predecessor_of_segment.get()) } = last;
        *unsafe { self.prev.get_unchecked_mut(successor_of_segment.get()) } = first;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A robust test fixture containing three distinct, closed rings:
    /// - Ring 0: [0 <-> 1 <-> 2 <-> 3] (4 nodes)
    /// - Ring 1: [4 <-> 5 <-> 6]       (3 nodes)
    /// - Ring 2: [7]                   (1 node, points to itself)
    fn complex_fixture() -> RingArena {
        RingArena::new(
            // prev pointers
            [
                3, 0, 1, 2, // Ring 0
                6, 4, 5, // Ring 1
                7, // Ring 2
            ]
            .map(Node::new)
            .to_vec(),
            // next pointers
            [
                1, 2, 3, 0, // Ring 0
                5, 6, 4, // Ring 1
                7, // Ring 2
            ]
            .map(Node::new)
            .to_vec(),
        )
    }

    /// Helper to verify the foundational mathematical invariant of the arena:
    /// For every node `i`, `prev[next[i]] == i` and `next[prev[i]] == i`.
    fn verify_integrity(arena: &RingArena) {
        for i in 0..arena.len() {
            let node = Node::new(i);
            let next_node = arena.next(node);
            let prev_node = arena.prev(node);
            assert_eq!(
                arena.prev(next_node),
                node,
                "Integrity failure at node {}: prev of next is not self",
                i
            );
            assert_eq!(
                arena.next(prev_node),
                node,
                "Integrity failure at node {}: next of prev is not self",
                i
            );
        }
    }

    /// Helper to extract a full ring into a Vec for easy assertions.
    /// Walks the `next` pointers until it cycles back to `start`.
    fn extract_ring(arena: &RingArena, start: usize) -> Vec<Node> {
        let start_node = Node::new(start);
        let mut result = vec![start_node];
        let mut current = arena.next(start_node);
        while current != start_node {
            result.push(current);
            current = arena.next(current);
        }
        result
    }

    #[test]
    fn test_initialization_and_integrity() {
        let arena = complex_fixture();
        assert_eq!(arena.len(), 8);
        assert!(!arena.is_empty());
        verify_integrity(&arena);

        assert_eq!(extract_ring(&arena, 0), [0, 1, 2, 3].map(Node::new));
        assert_eq!(extract_ring(&arena, 4), [4, 5, 6].map(Node::new));
        assert_eq!(extract_ring(&arena, 7), [7].map(Node::new));
    }

    // ----------------------------------------------------------------
    // Swap Nodes
    // ----------------------------------------------------------------

    #[test]
    fn test_swap_nodes_adjacent_same_ring() {
        let mut arena = complex_fixture();
        arena.swap_nodes(Node::new(1), Node::new(2));
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), [0, 2, 1, 3].map(Node::new));
    }

    #[test]
    fn test_swap_nodes_non_adjacent_same_ring() {
        let mut arena = complex_fixture();
        arena.swap_nodes(Node::new(0), Node::new(2));
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 2), [2, 1, 0, 3].map(Node::new));
    }

    #[test]
    fn test_swap_nodes_different_rings() {
        let mut arena = complex_fixture();
        arena.swap_nodes(Node::new(1), Node::new(5));
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), [0, 5, 2, 3].map(Node::new));
        assert_eq!(extract_ring(&arena, 4), [4, 1, 6].map(Node::new));
    }

    #[test]
    fn test_swap_nodes_with_single_node_ring() {
        let mut arena = complex_fixture();
        arena.swap_nodes(Node::new(2), Node::new(7));
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), [0, 1, 7, 3].map(Node::new));
        assert_eq!(extract_ring(&arena, 2), [2].map(Node::new));
    }

    #[test]
    fn test_swap_nodes_self_noop() {
        let mut arena = complex_fixture();
        arena.swap_nodes(Node::new(1), Node::new(1));
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), [0, 1, 2, 3].map(Node::new));
    }

    // ----------------------------------------------------------------
    // Swap Segments
    // ----------------------------------------------------------------

    #[test]
    fn test_swap_segments_different_rings() {
        let mut arena = complex_fixture();
        // Swap [1, 2] from Ring 0 with [5, 6] from Ring 1
        arena.swap_segments(Node::new(1), Node::new(2), Node::new(5), Node::new(6));
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), [0, 5, 6, 3].map(Node::new));
        assert_eq!(extract_ring(&arena, 4), [4, 1, 2].map(Node::new));
    }

    #[test]
    fn test_swap_segments_different_lengths() {
        let mut arena = complex_fixture();
        // Swap [1, 2, 3] from Ring 0 with [5] from Ring 1
        arena.swap_segments(Node::new(1), Node::new(3), Node::new(5), Node::new(5));
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), [0, 5].map(Node::new));
        assert_eq!(extract_ring(&arena, 4), [4, 1, 2, 3, 6].map(Node::new));
    }

    #[test]
    fn test_swap_segments_same_ring_adjacent() {
        let mut arena = complex_fixture();
        // Swap [0, 1] with [2, 3]
        arena.swap_segments(Node::new(0), Node::new(1), Node::new(2), Node::new(3));
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 2), [2, 3, 0, 1].map(Node::new));
    }

    // ----------------------------------------------------------------
    // Relocate Single Nodes
    // ----------------------------------------------------------------

    #[test]
    fn test_relocate_after_same_ring() {
        let mut arena = complex_fixture();
        // Move 1 to be after 3
        arena.relocate_after(Node::new(1), Node::new(3));
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), [0, 2, 3, 1].map(Node::new));
    }

    #[test]
    fn test_relocate_after_different_ring() {
        let mut arena = complex_fixture();
        // Move 1 from Ring 0 to after 5 in Ring 1
        arena.relocate_after(Node::new(1), Node::new(5));
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), [0, 2, 3].map(Node::new));
        assert_eq!(extract_ring(&arena, 4), [4, 5, 1, 6].map(Node::new));
    }

    #[test]
    fn test_relocate_after_noop() {
        let mut arena = complex_fixture();
        arena.relocate_after(Node::new(1), Node::new(0)); // 1 is already after 0
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), [0, 1, 2, 3].map(Node::new));
    }

    #[test]
    fn test_relocate_before_different_ring() {
        let mut arena = complex_fixture();
        // Move 1 from Ring 0 to before 5 in Ring 1 (which means after 4)
        arena.relocate_before(Node::new(1), Node::new(5));
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), [0, 2, 3].map(Node::new));
        assert_eq!(extract_ring(&arena, 4), [4, 1, 5, 6].map(Node::new));
    }

    // ----------------------------------------------------------------
    // Relocate Segments
    // ----------------------------------------------------------------

    #[test]
    fn test_relocate_segment_after_different_ring() {
        let mut arena = complex_fixture();
        // Move [1, 2] from Ring 0 to after 5 in Ring 1
        arena.relocate_segment_after(Node::new(1), Node::new(2), Node::new(5));
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), [0, 3].map(Node::new));
        assert_eq!(extract_ring(&arena, 4), [4, 5, 1, 2, 6].map(Node::new));
    }

    #[test]
    fn test_relocate_segment_before_different_ring() {
        let mut arena = complex_fixture();
        // Move [1, 2] from Ring 0 to before 5 in Ring 1 (after 4)
        arena.relocate_segment_before(Node::new(1), Node::new(2), Node::new(5));
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), [0, 3].map(Node::new));
        assert_eq!(extract_ring(&arena, 4), [4, 1, 2, 5, 6].map(Node::new));
    }

    #[test]
    fn test_relocate_segment_into_single_node_ring() {
        let mut arena = complex_fixture();
        // Move [1, 2] into the single-node ring (after 7)
        arena.relocate_segment_after(Node::new(1), Node::new(2), Node::new(7));
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), [0, 3].map(Node::new));
        assert_eq!(extract_ring(&arena, 7), [7, 1, 2].map(Node::new));
    }

    // ----------------------------------------------------------------
    // Reverse Segments
    // ----------------------------------------------------------------

    #[test]
    fn test_reverse_segment_partial_ring() {
        let mut arena = complex_fixture();
        // Reverse [1, 2] in Ring 0
        arena.reverse_segment(Node::new(1), Node::new(2));
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), [0, 2, 1, 3].map(Node::new));
    }

    #[test]
    fn test_reverse_segment_full_data_with_sentinel() {
        // Actually let's just build it correctly:
        // 4 -> 0 -> 1 -> 2 -> 3 -> 4
        // next: [1, 2, 3, 4, 0]
        // prev: [4, 0, 1, 2, 3]
        let mut arena = RingArena::new(
            [4, 0, 1, 2, 3].map(Node::new).to_vec(),
            [1, 2, 3, 4, 0].map(Node::new).to_vec(),
        );

        // Reverse all data nodes [0, 1, 2, 3], leaving sentinel 4 in place.
        arena.reverse_segment(Node::new(0), Node::new(3));
        verify_integrity(&arena);
        // Ring should now be: 4 -> 3 -> 2 -> 1 -> 0 -> 4
        assert_eq!(extract_ring(&arena, 4), [4, 3, 2, 1, 0].map(Node::new));
    }

    #[test]
    fn test_reverse_single_node_noop() {
        let mut arena = complex_fixture();
        arena.reverse_segment(Node::new(1), Node::new(1));
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), [0, 1, 2, 3].map(Node::new));
    }

    // ----------------------------------------------------------------
    // Iterators
    // ----------------------------------------------------------------

    #[test]
    fn test_sequence_iter() {
        let arena = complex_fixture();
        // Start at 1, stop at 0 (so it should yield 1, 2, 3)
        let seq: Vec<Node> = unsafe {
            arena
                .sequence_iter_unchecked(Node::new(1), Node::new(0))
                .collect()
        };
        assert_eq!(seq, [1, 2, 3].map(Node::new).to_vec());

        // Start at 4, stop at 4 (empty)
        let seq_empty: Vec<Node> = unsafe {
            arena
                .sequence_iter_unchecked(Node::new(4), Node::new(4))
                .collect()
        };
        assert!(seq_empty.is_empty());
    }

    #[test]
    fn test_sequence_rev_iter() {
        let arena = complex_fixture();
        // Start at 3, reverse stop at 0 (should yield 3, 2, 1)
        let seq: Vec<Node> = unsafe {
            arena
                .sequence_rev_iter_unchecked(Node::new(3), Node::new(0))
                .collect()
        };
        assert_eq!(seq, [3, 2, 1].map(Node::new).to_vec());
    }

    #[test]
    fn test_edge_iter() {
        let arena = complex_fixture();
        // Edges starting at 0, stopping at 3 (should yield (0,1), (1,2))
        let edges: Vec<(Node, Node)> = unsafe {
            arena
                .edge_iter_unchecked(Node::new(0), Node::new(3))
                .collect()
        };
        assert_eq!(
            edges,
            vec![(Node::new(0), Node::new(1)), (Node::new(1), Node::new(2))]
        );
    }

    #[test]
    fn test_overwrite() {
        let mut arena1 = complex_fixture();
        let arena2 = RingArena::new(
            [1, 0].map(Node::new).to_vec(),
            [1, 0].map(Node::new).to_vec(),
        ); // Two node ring

        arena1.overwrite_from_arena(&arena2);
        assert_eq!(arena1.len(), 2);
        assert_eq!(extract_ring(&arena1, 0), [0, 1].map(Node::new));

        let binding_prev = [Node::new(0)];
        let binding_next = [Node::new(0)];
        arena1.overwrite_from_slices(&binding_prev, &binding_next); // Single node ring
        assert_eq!(arena1.len(), 1);
        assert_eq!(extract_ring(&arena1, 0), [0].map(Node::new));
    }
}
