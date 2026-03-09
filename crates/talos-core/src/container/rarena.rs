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
//! This module provides the [`RingArena`], which acts as the core state representation
//! (genotype) for Local Search algorithms evaluated on sequence-based problems.
//!
//! # Architecture
//!
//! To evaluate thousands of neighborhood moves per second, operations like swapping nodes,
//! relocating segments, or reversing sequences must be `O(1)` and entirely free of heap allocations.
//!
//! `RingArena` achieves this by using a **flat-array, arena-backed doubly-linked list** design.
//! Instead of using standard pointer-based nodes, the graph maintains two parallel `Vec<usize>`
//! arrays: `prev` and `next`.
//!
//! Because the graph strictly consists of valid, closed rings, these two arrays are exact
//! mathematical inverses: `prev[next[v]] == v` for all `v`. Nodes can never enter or leave
//! the graph once initialized.

use std::iter::FusedIterator;

// ----------------------------------------------------------------
// RingSequenceIter
// ----------------------------------------------------------------

/// Iterator over a sequence in the arena, starting from a specific node and ending at a stop node.
#[derive(Clone, PartialEq, Eq)]
pub struct RingSequenceIter<'a> {
    /// The `next` array borrowed from the owning `RingArena`.
    next_pointers: &'a [usize],
    /// Current cursor into the ring.
    current_node: usize,
    /// The node at which iteration should stop (exclusive).
    stop_node: usize,
}

impl<'a> Iterator for RingSequenceIter<'a> {
    type Item = usize;

    #[inline(always)]
    fn next(&mut self) -> Option<usize> {
        if self.current_node == self.stop_node {
            return None;
        }

        let current = self.current_node;
        // SAFETY: The graph invariant guarantees `current` is a valid index.
        self.current_node = *unsafe { self.next_pointers.get_unchecked(current) };

        debug_assert!(
            self.current_node == self.stop_node || self.current_node < self.next_pointers.len(),
            "invariant violation: sequence iterator cursor out of bounds: current_node = {}, next_pointers_len = {}",
            self.current_node,
            self.next_pointers.len()
        );

        Some(current)
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
    prev_pointers: &'a [usize],
    /// Current cursor into the ring.
    current_node: usize,
    /// The node at which iteration should stop (exclusive).
    stop_node: usize,
}

impl<'a> Iterator for RingSequenceRevIter<'a> {
    type Item = usize;

    #[inline(always)]
    fn next(&mut self) -> Option<usize> {
        if self.current_node == self.stop_node {
            return None;
        }

        let current = self.current_node;
        // SAFETY: The graph invariant guarantees `current` is a valid index.
        self.current_node = *unsafe { self.prev_pointers.get_unchecked(current) };

        debug_assert!(
            self.current_node == self.stop_node || self.current_node < self.prev_pointers.len(),
            "invariant violation: reverse sequence iterator cursor out of bounds: current_node = {}, prev_pointers_len = {}",
            self.current_node,
            self.prev_pointers.len()
        );

        Some(current)
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
    next_pointers: &'a [usize],
    /// The left-hand side of the edge we are about to yield.
    current_node: usize,
    /// The node at which iteration should stop.
    stop_node: usize,
}

impl<'a> Iterator for RingEdgeIter<'a> {
    type Item = (usize, usize);

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_node == self.stop_node {
            return None;
        }

        let from = self.current_node;
        // SAFETY: The graph invariant guarantees `from` is a valid index.
        let to = *unsafe { self.next_pointers.get_unchecked(from) };

        if to == self.stop_node {
            // Stop without yielding an edge to the stop node itself
            self.current_node = self.stop_node;
            return None;
        }

        self.current_node = to;
        Some((from, to))
    }
}

impl<'a> FusedIterator for RingEdgeIter<'a> {}

// ----------------------------------------------------------------
// RingArena
// ----------------------------------------------------------------

/// High-performance, arena-backed collection of circular, doubly-linked lists.
///
/// `RingArena` acts as the primary state representation for local search algorithms,
/// supporting `O(1)` neighborhood moves (swaps, relocations, reversals) entirely
/// without heap allocations.
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
    prev: Vec<usize>,

    /// O(1) lookup: the node immediately following this one.
    next: Vec<usize>,
}

impl RingArena {
    /// Creates a new `RingArena` from parallel slices.
    ///
    /// # Panics
    ///
    /// Panics if `prev.len() != next.len()`.
    #[inline]
    pub fn new(prev: Vec<usize>, next: Vec<usize>) -> Self {
        assert_eq!(
            prev.len(),
            next.len(),
            "called `RingArena::new` with mismatched slice lengths: prev.len() = {}, next.len() = {}",
            prev.len(),
            next.len()
        );
        Self { prev, next }
    }

    /// Overwrites the current arena with data from parallel slices to avoid reallocation.
    #[inline]
    pub fn overwrite_from_slices(&mut self, prev: &[usize], next: &[usize]) {
        assert_eq!(
            prev.len(),
            next.len(),
            "called `RingArena::overwrite_from_slices` with mismatched slice lengths"
        );
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

    /// Resizes the arena to exactly `new_len` nodes.
    ///
    /// If growing, new slots are initialized to `0` (not valid rings —
    /// the caller must fix them up). If shrinking, excess nodes are dropped.
    /// Existing data for indices `< min(old_len, new_len)` is preserved.
    #[inline]
    pub fn resize(&mut self, new_len: usize) {
        self.prev.resize(new_len, 0);
        self.next.resize(new_len, 0);
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
    pub unsafe fn raw_mut(&mut self) -> (&mut [usize], &mut [usize]) {
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
    pub fn next(&self, node: usize) -> usize {
        assert!(
            node < self.next.len(),
            "called `RingArena::next` with out-of-bounds node index: node = {}, next_len = {}",
            node,
            self.next.len()
        );
        self.next[node]
    }

    /// Returns the internal `next` pointer for the given node.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `node < self.next.len()`.
    #[inline(always)]
    pub unsafe fn next_unchecked(&self, node: usize) -> usize {
        debug_assert!(
            node < self.next.len(),
            "called `RingArena::next_unchecked` with out-of-bounds node index"
        );
        *unsafe { self.next.get_unchecked(node) }
    }

    /// Returns the internal `prev` pointer for the given node.
    #[inline]
    pub fn prev(&self, node: usize) -> usize {
        assert!(
            node < self.prev.len(),
            "called `RingArena::prev` with out-of-bounds node index: node = {}, prev_len = {}",
            node,
            self.prev.len()
        );
        self.prev[node]
    }

    /// Returns the internal `prev` pointer for the given node.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `node < self.prev.len()`.
    #[inline(always)]
    pub unsafe fn prev_unchecked(&self, node: usize) -> usize {
        debug_assert!(
            node < self.prev.len(),
            "called `RingArena::prev_unchecked` with out-of-bounds node index"
        );
        *unsafe { self.prev.get_unchecked(node) }
    }

    /// Returns an iterator over a sequence of nodes starting from `start_node`
    /// and ending right before it reaches `stop_node`.
    #[inline]
    pub fn sequence_iter(&self, start_node: usize, stop_node: usize) -> RingSequenceIter<'_> {
        RingSequenceIter {
            next_pointers: &self.next,
            current_node: start_node,
            stop_node,
        }
    }

    /// Returns a reverse iterator over a sequence of nodes starting from `start_node`
    /// and ending right before it reaches `stop_node`.
    #[inline]
    pub fn sequence_rev_iter(
        &self,
        start_node: usize,
        stop_node: usize,
    ) -> RingSequenceRevIter<'_> {
        RingSequenceRevIter {
            prev_pointers: &self.prev,
            current_node: start_node,
            stop_node,
        }
    }

    /// Returns an iterator over all edges (adjacent node pairs) within a sequence.
    #[inline]
    pub fn edge_iter(&self, start_node: usize, stop_node: usize) -> RingEdgeIter<'_> {
        RingEdgeIter {
            next_pointers: &self.next,
            current_node: start_node,
            stop_node,
        }
    }

    /// Detaches a node from its current position in the ring.
    ///
    /// After this call, the node's own `prev` and `next` slots are **stale**. The
    /// caller must reinsert it or otherwise fix them up before the graph is observed again.
    #[inline(always)]
    unsafe fn extract_node_unchecked(&mut self, node_to_extract: usize) {
        debug_assert!(node_to_extract < self.prev.len(), "out-of-bounds node");

        let predecessor = *unsafe { self.prev.get_unchecked(node_to_extract) };
        let successor = *unsafe { self.next.get_unchecked(node_to_extract) };

        *unsafe { self.next.get_unchecked_mut(predecessor) } = successor;
        *unsafe { self.prev.get_unchecked_mut(successor) } = predecessor;
    }

    /// Inserts a node immediately following the `insertion_point` node.
    #[inline(always)]
    unsafe fn insert_node_after_unchecked(
        &mut self,
        node_to_insert: usize,
        insertion_point: usize,
    ) {
        debug_assert!(node_to_insert < self.prev.len(), "out-of-bounds node");
        debug_assert!(
            insertion_point < self.prev.len(),
            "out-of-bounds insertion point"
        );

        let successor_of_insertion_point = *unsafe { self.next.get_unchecked(insertion_point) };

        *unsafe { self.next.get_unchecked_mut(insertion_point) } = node_to_insert;
        *unsafe { self.prev.get_unchecked_mut(node_to_insert) } = insertion_point;

        *unsafe { self.next.get_unchecked_mut(node_to_insert) } = successor_of_insertion_point;
        *unsafe { self.prev.get_unchecked_mut(successor_of_insertion_point) } = node_to_insert;
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
    pub fn swap_nodes(&mut self, first: usize, second: usize) {
        assert!(
            first < self.len() && second < self.len(),
            "called `RingArena::swap_nodes` with out-of-bounds node"
        );
        unsafe { self.swap_nodes_unchecked(first, second) }
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
    pub unsafe fn swap_nodes_unchecked(&mut self, first: usize, second: usize) {
        debug_assert!(
            first < self.len() && second < self.len(),
            "called `RingArena::swap_nodes_unchecked` with out-of-bounds node"
        );

        if first == second {
            return;
        }

        let first_prev = *unsafe { self.prev.get_unchecked(first) };
        let first_next = *unsafe { self.next.get_unchecked(first) };
        let second_prev = *unsafe { self.prev.get_unchecked(second) };
        let second_next = *unsafe { self.next.get_unchecked(second) };

        let first_is_solo = first_prev == first;
        let second_is_solo = second_prev == second;

        if first_is_solo && second_is_solo {
            // Both are self-loops — swapping is a no-op.
        } else if first_is_solo {
            // First is a self-loop. Extract second from its ring, put first in second's
            // old position, make second a self-loop.
            *unsafe { self.next.get_unchecked_mut(second_prev) } = first;
            *unsafe { self.prev.get_unchecked_mut(first) } = second_prev;
            *unsafe { self.next.get_unchecked_mut(first) } = second_next;
            *unsafe { self.prev.get_unchecked_mut(second_next) } = first;

            *unsafe { self.next.get_unchecked_mut(second) } = second;
            *unsafe { self.prev.get_unchecked_mut(second) } = second;
        } else if second_is_solo {
            // Second is a self-loop. Extract first from its ring, put second in first's
            // old position, make first a self-loop.
            *unsafe { self.next.get_unchecked_mut(first_prev) } = second;
            *unsafe { self.prev.get_unchecked_mut(second) } = first_prev;
            *unsafe { self.next.get_unchecked_mut(second) } = first_next;
            *unsafe { self.prev.get_unchecked_mut(first_next) } = second;

            *unsafe { self.next.get_unchecked_mut(first) } = first;
            *unsafe { self.prev.get_unchecked_mut(first) } = first;
        } else if first_next == second {
            // Adjacent: first -> second
            *unsafe { self.next.get_unchecked_mut(first_prev) } = second;
            *unsafe { self.prev.get_unchecked_mut(second) } = first_prev;
            *unsafe { self.next.get_unchecked_mut(second) } = first;
            *unsafe { self.prev.get_unchecked_mut(first) } = second;
            *unsafe { self.next.get_unchecked_mut(first) } = second_next;
            *unsafe { self.prev.get_unchecked_mut(second_next) } = first;
        } else if second_next == first {
            // Adjacent: second -> first
            *unsafe { self.next.get_unchecked_mut(second_prev) } = first;
            *unsafe { self.prev.get_unchecked_mut(first) } = second_prev;
            *unsafe { self.next.get_unchecked_mut(first) } = second;
            *unsafe { self.prev.get_unchecked_mut(second) } = first;
            *unsafe { self.next.get_unchecked_mut(second) } = first_next;
            *unsafe { self.prev.get_unchecked_mut(first_next) } = second;
        } else {
            // Non-adjacent, neither is a self-loop
            *unsafe { self.next.get_unchecked_mut(first_prev) } = second;
            *unsafe { self.prev.get_unchecked_mut(second) } = first_prev;
            *unsafe { self.next.get_unchecked_mut(second) } = first_next;
            *unsafe { self.prev.get_unchecked_mut(first_next) } = second;

            *unsafe { self.next.get_unchecked_mut(second_prev) } = first;
            *unsafe { self.prev.get_unchecked_mut(first) } = second_prev;
            *unsafe { self.next.get_unchecked_mut(first) } = second_next;
            *unsafe { self.prev.get_unchecked_mut(second_next) } = first;
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
    pub fn swap_segments(
        &mut self,
        segment_a_first: usize,
        segment_a_last: usize,
        segment_b_first: usize,
        segment_b_last: usize,
    ) {
        assert!(
            segment_a_first < self.len()
                && segment_a_last < self.len()
                && segment_b_first < self.len()
                && segment_b_last < self.len(),
            "called `RingArena::swap_segments` with out-of-bounds nodes"
        );

        unsafe {
            self.swap_segments_unchecked(
                segment_a_first,
                segment_a_last,
                segment_b_first,
                segment_b_last,
            )
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
        a_first: usize,
        a_last: usize,
        b_first: usize,
        b_last: usize,
    ) {
        debug_assert!(
            a_first < self.len()
                && a_last < self.len()
                && b_first < self.len()
                && b_last < self.len(),
            "called `RingArena::swap_segments_unchecked` with out-of-bounds nodes"
        );

        if a_first == b_first {
            return;
        }

        let a_pred = *unsafe { self.prev.get_unchecked(a_first) };
        let a_succ = *unsafe { self.next.get_unchecked(a_last) };
        let b_pred = *unsafe { self.prev.get_unchecked(b_first) };
        let b_succ = *unsafe { self.next.get_unchecked(b_last) };

        if a_succ == b_first && b_succ == a_first {
            // Full-ring case: both segments together cover the entire ring.
            // Just rotate: swap the connection point between them.
            unsafe {
                *self.next.get_unchecked_mut(b_last) = a_first;
                *self.prev.get_unchecked_mut(a_first) = b_last;
                *self.next.get_unchecked_mut(a_last) = b_first;
                *self.prev.get_unchecked_mut(b_first) = a_last;
            }
        } else if a_succ == b_first {
            // Adjacent: A immediately before B
            unsafe {
                *self.next.get_unchecked_mut(a_pred) = b_first;
                *self.prev.get_unchecked_mut(b_first) = a_pred;

                *self.next.get_unchecked_mut(b_last) = a_first;
                *self.prev.get_unchecked_mut(a_first) = b_last;

                *self.next.get_unchecked_mut(a_last) = b_succ;
                *self.prev.get_unchecked_mut(b_succ) = a_last;
            }
        } else if b_succ == a_first {
            // Adjacent: B immediately before A
            unsafe {
                *self.next.get_unchecked_mut(b_pred) = a_first;
                *self.prev.get_unchecked_mut(a_first) = b_pred;

                *self.next.get_unchecked_mut(a_last) = b_first;
                *self.prev.get_unchecked_mut(b_first) = a_last;

                *self.next.get_unchecked_mut(b_last) = a_succ;
                *self.prev.get_unchecked_mut(a_succ) = b_last;
            }
        } else {
            // Non-adjacent
            unsafe {
                *self.next.get_unchecked_mut(a_pred) = b_first;
                *self.prev.get_unchecked_mut(b_first) = a_pred;

                *self.next.get_unchecked_mut(b_last) = a_succ;
                *self.prev.get_unchecked_mut(a_succ) = b_last;

                *self.next.get_unchecked_mut(b_pred) = a_first;
                *self.prev.get_unchecked_mut(a_first) = b_pred;

                *self.next.get_unchecked_mut(a_last) = b_succ;
                *self.prev.get_unchecked_mut(b_succ) = a_last;
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
    pub fn relocate_after(&mut self, node: usize, anchor: usize) {
        assert!(
            node < self.len() && anchor < self.len(),
            "called `RingArena::relocate_after` with out-of-bounds nodes"
        );
        unsafe { self.relocate_after_unchecked(node, anchor) }
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
    pub unsafe fn relocate_after_unchecked(&mut self, node: usize, anchor: usize) {
        if node == anchor {
            return;
        }
        if *unsafe { self.prev.get_unchecked(node) } == anchor {
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
    pub fn relocate_segment_after(
        &mut self,
        segment_first: usize,
        segment_last: usize,
        anchor: usize,
    ) {
        assert!(
            segment_first < self.len() && segment_last < self.len() && anchor < self.len(),
            "called `RingArena::relocate_segment_after` with out-of-bounds indices"
        );
        unsafe { self.relocate_segment_after_unchecked(segment_first, segment_last, anchor) }
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
        first: usize,
        last: usize,
        anchor: usize,
    ) {
        if *unsafe { self.prev.get_unchecked(first) } == anchor {
            return;
        }

        let before_segment = *unsafe { self.prev.get_unchecked(first) };
        let after_segment = *unsafe { self.next.get_unchecked(last) };

        unsafe {
            *self.next.get_unchecked_mut(before_segment) = after_segment;
            *self.prev.get_unchecked_mut(after_segment) = before_segment;
        }

        let successor_of_anchor = *unsafe { self.next.get_unchecked(anchor) };

        unsafe {
            *self.next.get_unchecked_mut(anchor) = first;
            *self.prev.get_unchecked_mut(first) = anchor;

            *self.next.get_unchecked_mut(last) = successor_of_anchor;
            *self.prev.get_unchecked_mut(successor_of_anchor) = last;
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
    pub fn relocate_before(&mut self, node: usize, anchor: usize) {
        assert!(
            node < self.len() && anchor < self.len(),
            "called `RingArena::relocate_before` with out-of-bounds nodes"
        );
        unsafe { self.relocate_before_unchecked(node, anchor) }
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
    pub unsafe fn relocate_before_unchecked(&mut self, node: usize, anchor: usize) {
        let anchor_predecessor = *unsafe { self.prev.get_unchecked(anchor) };
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
    pub fn relocate_segment_before(
        &mut self,
        segment_first: usize,
        segment_last: usize,
        anchor: usize,
    ) {
        assert!(
            segment_first < self.len() && segment_last < self.len() && anchor < self.len(),
            "called `RingArena::relocate_segment_before` with out-of-bounds indices"
        );
        unsafe { self.relocate_segment_before_unchecked(segment_first, segment_last, anchor) }
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
        first: usize,
        last: usize,
        anchor: usize,
    ) {
        let anchor_predecessor = *unsafe { self.prev.get_unchecked(anchor) };
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
    pub fn reverse_segment(&mut self, segment_first: usize, segment_last: usize) {
        assert!(
            segment_first < self.len() && segment_last < self.len(),
            "called `RingArena::reverse_segment` with out-of-bounds nodes"
        );
        unsafe { self.reverse_segment_unchecked(segment_first, segment_last) }
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
    pub unsafe fn reverse_segment_unchecked(&mut self, first: usize, last: usize) {
        if first == last {
            return;
        }

        let predecessor_of_segment = *unsafe { self.prev.get_unchecked(first) };
        let successor_of_segment = *unsafe { self.next.get_unchecked(last) };

        let prev_ptr = self.prev.as_mut_ptr();
        let next_ptr = self.next.as_mut_ptr();
        let mut current_node = first;

        loop {
            let original_next = *unsafe { self.next.get_unchecked(current_node) };
            unsafe {
                std::ptr::swap(prev_ptr.add(current_node), next_ptr.add(current_node));
            }
            if current_node == last {
                break;
            }
            current_node = original_next;
        }

        *unsafe { self.next.get_unchecked_mut(first) } = successor_of_segment;
        *unsafe { self.prev.get_unchecked_mut(last) } = predecessor_of_segment;

        *unsafe { self.next.get_unchecked_mut(predecessor_of_segment) } = last;
        *unsafe { self.prev.get_unchecked_mut(successor_of_segment) } = first;
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
            vec![
                3, 0, 1, 2, // Ring 0
                6, 4, 5, // Ring 1
                7, // Ring 2
            ],
            // next pointers
            vec![
                1, 2, 3, 0, // Ring 0
                5, 6, 4, // Ring 1
                7, // Ring 2
            ],
        )
    }

    /// Helper to verify the foundational mathematical invariant of the arena:
    /// For every node `i`, `prev[next[i]] == i` and `next[prev[i]] == i`.
    fn verify_integrity(arena: &RingArena) {
        for i in 0..arena.len() {
            let next_node = arena.next(i);
            let prev_node = arena.prev(i);
            assert_eq!(
                arena.prev(next_node),
                i,
                "Integrity failure at node {}: prev of next is not self",
                i
            );
            assert_eq!(
                arena.next(prev_node),
                i,
                "Integrity failure at node {}: next of prev is not self",
                i
            );
        }
    }

    /// Helper to extract a full ring into a Vec for easy assertions.
    /// Walks the `next` pointers until it cycles back to `start`.
    fn extract_ring(arena: &RingArena, start: usize) -> Vec<usize> {
        let mut result = vec![start];
        let mut current = arena.next(start);
        while current != start {
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

        assert_eq!(extract_ring(&arena, 0), vec![0, 1, 2, 3]);
        assert_eq!(extract_ring(&arena, 4), vec![4, 5, 6]);
        assert_eq!(extract_ring(&arena, 7), vec![7]);
    }

    // ----------------------------------------------------------------
    // Swap Nodes
    // ----------------------------------------------------------------

    #[test]
    fn test_swap_nodes_adjacent_same_ring() {
        let mut arena = complex_fixture();
        arena.swap_nodes(1, 2);
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), vec![0, 2, 1, 3]);
    }

    #[test]
    fn test_swap_nodes_non_adjacent_same_ring() {
        let mut arena = complex_fixture();
        arena.swap_nodes(0, 2);
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 2), vec![2, 1, 0, 3]);
    }

    #[test]
    fn test_swap_nodes_different_rings() {
        let mut arena = complex_fixture();
        arena.swap_nodes(1, 5);
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), vec![0, 5, 2, 3]);
        assert_eq!(extract_ring(&arena, 4), vec![4, 1, 6]);
    }

    #[test]
    fn test_swap_nodes_with_single_node_ring() {
        let mut arena = complex_fixture();
        arena.swap_nodes(2, 7);
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), vec![0, 1, 7, 3]);
        assert_eq!(extract_ring(&arena, 2), vec![2]);
    }

    #[test]
    fn test_swap_nodes_self_noop() {
        let mut arena = complex_fixture();
        arena.swap_nodes(1, 1);
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), vec![0, 1, 2, 3]);
    }

    // ----------------------------------------------------------------
    // Swap Segments
    // ----------------------------------------------------------------

    #[test]
    fn test_swap_segments_different_rings() {
        let mut arena = complex_fixture();
        // Swap [1, 2] from Ring 0 with [5, 6] from Ring 1
        arena.swap_segments(1, 2, 5, 6);
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), vec![0, 5, 6, 3]);
        assert_eq!(extract_ring(&arena, 4), vec![4, 1, 2]);
    }

    #[test]
    fn test_swap_segments_different_lengths() {
        let mut arena = complex_fixture();
        // Swap [1, 2, 3] from Ring 0 with [5] from Ring 1
        arena.swap_segments(1, 3, 5, 5);
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), vec![0, 5]);
        assert_eq!(extract_ring(&arena, 4), vec![4, 1, 2, 3, 6]);
    }

    #[test]
    fn test_swap_segments_same_ring_adjacent() {
        let mut arena = complex_fixture();
        // Swap [0, 1] with [2, 3]
        arena.swap_segments(0, 1, 2, 3);
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 2), vec![2, 3, 0, 1]);
    }

    // ----------------------------------------------------------------
    // Relocate Single Nodes
    // ----------------------------------------------------------------

    #[test]
    fn test_relocate_after_same_ring() {
        let mut arena = complex_fixture();
        // Move 1 to be after 3
        arena.relocate_after(1, 3);
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), vec![0, 2, 3, 1]);
    }

    #[test]
    fn test_relocate_after_different_ring() {
        let mut arena = complex_fixture();
        // Move 1 from Ring 0 to after 5 in Ring 1
        arena.relocate_after(1, 5);
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), vec![0, 2, 3]);
        assert_eq!(extract_ring(&arena, 4), vec![4, 5, 1, 6]);
    }

    #[test]
    fn test_relocate_after_noop() {
        let mut arena = complex_fixture();
        arena.relocate_after(1, 0); // 1 is already after 0
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_relocate_before_different_ring() {
        let mut arena = complex_fixture();
        // Move 1 from Ring 0 to before 5 in Ring 1 (which means after 4)
        arena.relocate_before(1, 5);
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), vec![0, 2, 3]);
        assert_eq!(extract_ring(&arena, 4), vec![4, 1, 5, 6]);
    }

    // ----------------------------------------------------------------
    // Relocate Segments
    // ----------------------------------------------------------------

    #[test]
    fn test_relocate_segment_after_different_ring() {
        let mut arena = complex_fixture();
        // Move [1, 2] from Ring 0 to after 5 in Ring 1
        arena.relocate_segment_after(1, 2, 5);
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), vec![0, 3]);
        assert_eq!(extract_ring(&arena, 4), vec![4, 5, 1, 2, 6]);
    }

    #[test]
    fn test_relocate_segment_before_different_ring() {
        let mut arena = complex_fixture();
        // Move [1, 2] from Ring 0 to before 5 in Ring 1 (after 4)
        arena.relocate_segment_before(1, 2, 5);
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), vec![0, 3]);
        assert_eq!(extract_ring(&arena, 4), vec![4, 1, 2, 5, 6]);
    }

    #[test]
    fn test_relocate_segment_into_single_node_ring() {
        let mut arena = complex_fixture();
        // Move [1, 2] into the single-node ring (after 7)
        arena.relocate_segment_after(1, 2, 7);
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), vec![0, 3]);
        assert_eq!(extract_ring(&arena, 7), vec![7, 1, 2]);
    }

    // ----------------------------------------------------------------
    // Reverse Segments
    // ----------------------------------------------------------------

    #[test]
    fn test_reverse_segment_partial_ring() {
        let mut arena = complex_fixture();
        // Reverse [1, 2] in Ring 0
        arena.reverse_segment(1, 2);
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), vec![0, 2, 1, 3]);
    }

    #[test]
    fn test_reverse_segment_full_data_with_sentinel() {
        // Actually let's just build it correctly:
        // 4 -> 0 -> 1 -> 2 -> 3 -> 4
        // next: [1, 2, 3, 4, 0]
        // prev: [4, 0, 1, 2, 3]
        let mut arena = RingArena::new(vec![4, 0, 1, 2, 3], vec![1, 2, 3, 4, 0]);

        // Reverse all data nodes [0, 1, 2, 3], leaving sentinel 4 in place.
        arena.reverse_segment(0, 3);
        verify_integrity(&arena);
        // Ring should now be: 4 -> 3 -> 2 -> 1 -> 0 -> 4
        assert_eq!(extract_ring(&arena, 4), vec![4, 3, 2, 1, 0]);
    }

    #[test]
    fn test_reverse_single_node_noop() {
        let mut arena = complex_fixture();
        arena.reverse_segment(1, 1);
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), vec![0, 1, 2, 3]);
    }

    // ----------------------------------------------------------------
    // Iterators
    // ----------------------------------------------------------------

    #[test]
    fn test_sequence_iter() {
        let arena = complex_fixture();
        // Start at 1, stop at 0 (so it should yield 1, 2, 3)
        let seq: Vec<usize> = arena.sequence_iter(1, 0).collect();
        assert_eq!(seq, vec![1, 2, 3]);

        // Start at 4, stop at 4 (empty)
        let seq_empty: Vec<usize> = arena.sequence_iter(4, 4).collect();
        assert!(seq_empty.is_empty());
    }

    #[test]
    fn test_sequence_rev_iter() {
        let arena = complex_fixture();
        // Start at 3, reverse stop at 0 (should yield 3, 2, 1)
        let seq: Vec<usize> = arena.sequence_rev_iter(3, 0).collect();
        assert_eq!(seq, vec![3, 2, 1]);
    }

    #[test]
    fn test_edge_iter() {
        let arena = complex_fixture();
        // Edges starting at 0, stopping at 3 (should yield (0,1), (1,2))
        let edges: Vec<(usize, usize)> = arena.edge_iter(0, 3).collect();
        assert_eq!(edges, vec![(0, 1), (1, 2)]);
    }

    #[test]
    fn test_overwrite() {
        let mut arena1 = complex_fixture();
        let arena2 = RingArena::new(vec![1, 0], vec![1, 0]); // Two node ring

        arena1.overwrite_from_arena(&arena2);
        assert_eq!(arena1.len(), 2);
        assert_eq!(extract_ring(&arena1, 0), vec![0, 1]);

        arena1.overwrite_from_slices(&[0], &[0]); // Single node ring
        assert_eq!(arena1.len(), 1);
        assert_eq!(extract_ring(&arena1, 0), vec![0]);
    }
}
