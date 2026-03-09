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

use crate::{
    sgraph::{ScheduleGraph, ScheduleGraphNodeIndex},
    sgraphdiff::ScheduleGraphDiff,
    sgraphundo::ScheduleGraphUndoLog,
};
use talos_model::index::{BerthIndex, VesselIndex};

/// A micro-tracker that lives purely on the stack (in CPU registers).
struct EdgeDeltaTracker {
    nodes: [ScheduleGraphNodeIndex; 4],
    old_nexts: [ScheduleGraphNodeIndex; 4],
    len: usize,
}

impl EdgeDeltaTracker {
    /// Creates a new, empty tracker.
    ///
    /// All fields are initialized to zero, but `len` is set to 0 to indicate emptiness.
    #[inline(always)]
    fn new() -> Self {
        Self {
            nodes: [ScheduleGraphNodeIndex::new(0); 4],
            old_nexts: [ScheduleGraphNodeIndex::new(0); 4],
            len: 0,
        }
    }

    /// Records a node's state *before* mutation.
    #[inline(always)]
    fn track(&mut self, node: ScheduleGraphNodeIndex, graph: &ScheduleGraph) {
        // Force unrolled loop for O(1) deduplication without branching overhead.
        // Because `len` is max 4, this compiles to a flat sequence of instructions.

        if self.len > 0 && unsafe { *self.nodes.get_unchecked(0) } == node {
            return;
        }
        if self.len > 1 && unsafe { *self.nodes.get_unchecked(1) } == node {
            return;
        }
        if self.len > 2 && unsafe { *self.nodes.get_unchecked(2) } == node {
            return;
        }
        if self.len > 3 && unsafe { *self.nodes.get_unchecked(3) } == node {
            return;
        }

        unsafe {
            *self.nodes.get_unchecked_mut(self.len) = node;
            *self.old_nexts.get_unchecked_mut(self.len) = graph.raw_next_unchecked(node);
        }
        self.len += 1;
    }

    /// Emits the exact net differences by comparing the recorded state
    /// against the graph's new state.
    #[inline(always)]
    fn commit(self, graph: &ScheduleGraph, diff: &mut ScheduleGraphDiff) {
        for i in 0..self.len {
            let node = unsafe { *self.nodes.get_unchecked(i) };
            let old_nxt = unsafe { *self.old_nexts.get_unchecked(i) };
            let new_nxt = unsafe { graph.raw_next_unchecked(node) };

            if old_nxt != new_nxt {
                // The diff expects VesselIndex. It treats indices >= num_vessels as sentinels internally.
                let from = VesselIndex::new(node.get());
                let old_to = VesselIndex::new(old_nxt.get());
                let new_to = VesselIndex::new(new_nxt.get());

                diff.push_link_broken(from, old_to);
                diff.push_link_created(from, new_to);
            }
        }
    }
}

/// A mutation engine for applying topological changes to a `ScheduleGraph`.
///
/// The `Mutator` acts as a proxy to the graph, automatically recording every
/// applied mutation into a `ScheduleGraphUndoLog`. This allows complex local
/// search neighborhoods to cleanly revert their changes.
///
/// # Sentinel Protection
///
/// The mutator enforces a strict boundary: **Users must never pass Sentinel nodes
/// (Berth indices masquerading as Vessel indices) to any public method.**
/// All safe methods assert that `index < num_vessels()`, and all unchecked methods
/// debug-assert the same invariant.
#[derive(Debug)]
pub struct Mutator<'a> {
    graph: &'a mut ScheduleGraph,
    graph_undo: &'a mut ScheduleGraphUndoLog,
    graph_diff: &'a mut ScheduleGraphDiff,
}

impl<'a> Mutator<'a> {
    #[inline(always)]
    pub fn new(
        graph: &'a mut ScheduleGraph,
        graph_undo: &'a mut ScheduleGraphUndoLog,
        graph_diff: &'a mut ScheduleGraphDiff,
    ) -> Self {
        Self {
            graph,
            graph_undo,
            graph_diff,
        }
    }

    #[inline(always)]
    pub fn graph(&self) -> &ScheduleGraph {
        self.graph
    }

    #[inline(always)]
    pub fn graph_undo(&mut self) -> &mut ScheduleGraphUndoLog {
        self.graph_undo
    }

    // ----------------------------------------------------------------
    // Internal Undo Recording Helpers
    // ----------------------------------------------------------------

    /// Records the original position of a single vessel before it is relocated.
    /// Branchless: Works flawlessly whether the predecessor is a vessel or a sentinel.
    #[inline(always)]
    unsafe fn record_relocate_unchecked(&mut self, vessel: VesselIndex) {
        let node = ScheduleGraphNodeIndex::from_vessel(vessel);
        let prev = unsafe { self.graph.raw_prev_unchecked(node) };
        let original_berth = unsafe { self.graph.node_berth_unchecked(node) };
        self.graph_undo.push_relocate(node, prev, original_berth);
    }

    /// Records the original position of a segment before it is relocated.
    /// Branchless: Works flawlessly whether the predecessor is a vessel or a sentinel.
    #[inline(always)]
    unsafe fn record_relocate_segment_unchecked(&mut self, first: VesselIndex, last: VesselIndex) {
        let first_node = ScheduleGraphNodeIndex::from_vessel(first);
        let last_node = ScheduleGraphNodeIndex::from_vessel(last);
        let prev = unsafe { self.graph.raw_prev_unchecked(first_node) };
        let original_berth = unsafe { self.graph.node_berth_unchecked(first_node) };
        self.graph_undo
            .push_relocate_segment(first_node, last_node, prev, original_berth);
    }

    #[inline(always)]
    unsafe fn record_segment_reallocation_unchecked(
        &mut self,
        first: VesselIndex,
        last: VesselIndex,
        old_berth: BerthIndex,
        new_berth: BerthIndex,
    ) {
        if old_berth == new_berth {
            return;
        }
        let mut curr = ScheduleGraphNodeIndex::from_vessel(first);
        let last_node = ScheduleGraphNodeIndex::from_vessel(last);
        loop {
            if let Some(vessel) = curr.as_vessel(self.graph.num_vessels()) {
                self.graph_diff
                    .push_reallocation(vessel, old_berth, new_berth);
            }
            if curr == last_node {
                break;
            }
            curr = unsafe { self.graph.raw_next_unchecked(curr) };
        }
    }

    // ----------------------------------------------------------------
    // Checked Mutations
    // ----------------------------------------------------------------

    /// Swaps the positions of two vessels.
    ///
    /// ```text
    /// Before:
    /// Berth 0: [S] -> A -> (V1) -> B -> [S]
    /// Berth 1: [S] -> C -> (V2) -> D -> [S]
    ///
    /// After `swap_vessels(V1, V2)`:
    /// Berth 0: [S] -> A -> (V2) -> B -> [S]
    /// Berth 1: [S] -> C -> (V1) -> D -> [S]
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if either `first_vessel` or `second_vessel` is out of bounds
    /// (i.e., `>= self.graph.num_vessels()`). Sentinel nodes are not allowed.
    #[inline]
    pub fn swap_vessels(&mut self, first_vessel: VesselIndex, second_vessel: VesselIndex) {
        assert!(
            first_vessel.get() < self.graph.num_vessels()
                && second_vessel.get() < self.graph.num_vessels(),
            "Mutator::swap_vessels bounds check failed: v1 = {}, v2 = {}, num_vessels = {}",
            first_vessel.get(),
            second_vessel.get(),
            self.graph.num_vessels()
        );
        unsafe { self.swap_vessels_unchecked(first_vessel, second_vessel) }
    }

    /// Swaps the positions of two distinct segments.
    ///
    /// ```text
    /// Before:
    /// Berth 0: [S] -> A -> [V1 -> V2] -> B -> [S]
    /// Berth 1: [S] -> C -> [V3] -> D -> [S]
    ///
    /// After `swap_segments(V1, V2, V3, V3)`:
    /// Berth 0: [S] -> A -> [V3] -> B -> [S]
    /// Berth 1: [S] -> C -> [V1 -> V2] -> D -> [S]
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if any of the provided vessel indices are out of bounds
    /// (i.e., `>= self.graph.num_vessels()`). Sentinel nodes are not allowed.
    #[inline]
    pub fn swap_segments(
        &mut self,
        segment_a_first: VesselIndex,
        segment_a_last: VesselIndex,
        segment_b_first: VesselIndex,
        segment_b_last: VesselIndex,
    ) {
        assert!(
            segment_a_first.get() < self.graph.num_vessels()
                && segment_a_last.get() < self.graph.num_vessels()
                && segment_b_first.get() < self.graph.num_vessels()
                && segment_b_last.get() < self.graph.num_vessels(),
            "Mutator::swap_segments bounds check failed: a_first = {}, a_last = {}, b_first = {}, b_last = {}, num_vessels = {}",
            segment_a_first.get(),
            segment_a_last.get(),
            segment_b_first.get(),
            segment_b_last.get(),
            self.graph.num_vessels()
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

    /// Reverses the topological order of a contiguous segment of vessels.
    ///
    /// ```text
    /// Before:
    /// Berth 0: [S] -> A -> [V1 -> V2 -> V3] -> B -> [S]
    ///
    /// After `reverse_segment(V1, V3)`:
    /// Berth 0: [S] -> A -> [V3 -> V2 -> V1] -> B -> [S]
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if either `segment_first` or `segment_last` is out of bounds
    /// (i.e., `>= self.graph.num_vessels()`). Sentinel nodes are not allowed.
    #[inline]
    pub fn reverse_segment(&mut self, segment_first: VesselIndex, segment_last: VesselIndex) {
        assert!(
            segment_first.get() < self.graph.num_vessels()
                && segment_last.get() < self.graph.num_vessels(),
            "Mutator::reverse_segment bounds check failed: first = {}, last = {}, num_vessels = {}",
            segment_first.get(),
            segment_last.get(),
            self.graph.num_vessels()
        );
        unsafe { self.reverse_segment_unchecked(segment_first, segment_last) }
    }

    /// Relocates a vessel to immediately follow a specific reference vessel.
    ///
    /// ```text
    /// Before:
    /// Berth 0: [S] -> A -> (V1) -> B -> [S]
    /// Berth 1: [S] -> C -> (Ref) -> D -> [S]
    ///
    /// After `relocate_after(V1, Ref)`:
    /// Berth 0: [S] -> A -> B -> [S]
    /// Berth 1: [S] -> C -> (Ref) -> (V1) -> D -> [S]
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if either `vessel_to_move` or `insertion_anchor` is out of bounds.
    #[inline]
    pub fn relocate_after(&mut self, vessel_to_move: VesselIndex, insertion_anchor: VesselIndex) {
        assert!(
            vessel_to_move.get() < self.graph.num_vessels()
                && insertion_anchor.get() < self.graph.num_vessels(),
            "Mutator::relocate_after bounds check failed: subject = {}, anchor = {}, num_vessels = {}",
            vessel_to_move.get(),
            insertion_anchor.get(),
            self.graph.num_vessels()
        );
        unsafe { self.relocate_after_unchecked(vessel_to_move, insertion_anchor) }
    }

    /// Relocates a vessel to immediately precede a specific reference vessel.
    ///
    /// # Panics
    ///
    /// Panics if either `vessel_to_move` or `reference_vessel` is out of bounds.
    #[inline]
    pub fn relocate_before(&mut self, vessel_to_move: VesselIndex, reference_vessel: VesselIndex) {
        assert!(
            vessel_to_move.get() < self.graph.num_vessels()
                && reference_vessel.get() < self.graph.num_vessels(),
            "Mutator::relocate_before bounds check failed: subject = {}, reference = {}, num_vessels = {}",
            vessel_to_move.get(),
            reference_vessel.get(),
            self.graph.num_vessels()
        );
        unsafe { self.relocate_before_unchecked(vessel_to_move, reference_vessel) }
    }

    /// Relocates a vessel to the head (very beginning) of a target berth.
    ///
    /// ```text
    /// Before:
    /// Berth 0: [S] -> A -> (V1) -> B -> [S]
    /// Berth 1: [S] -> C -> D -> [S]
    ///
    /// After `relocate_to_head(V1, Berth 1)`:
    /// Berth 0: [S] -> A -> B -> [S]
    /// Berth 1: [S] -> (V1) -> C -> D -> [S]
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `vessel_to_move >= self.graph.num_vessels()` or
    /// if `target_berth >= self.graph.num_berths()`.
    #[inline]
    pub fn relocate_to_head(&mut self, vessel_to_move: VesselIndex, target_berth: BerthIndex) {
        assert!(
            vessel_to_move.get() < self.graph.num_vessels(),
            "Mutator::relocate_to_head bounds check failed: subject = {}, num_vessels = {}",
            vessel_to_move.get(),
            self.graph.num_vessels()
        );
        assert!(
            target_berth.get() < self.graph.num_berths(),
            "Mutator::relocate_to_head bounds check failed: target_berth = {}, num_berths = {}",
            target_berth.get(),
            self.graph.num_berths()
        );
        unsafe { self.relocate_to_head_unchecked(vessel_to_move, target_berth) }
    }

    /// Relocates a vessel to the tail (very end) of a target berth.
    ///
    /// # Panics
    ///
    /// Panics if `vessel_to_move >= self.graph.num_vessels()` or
    /// if `target_berth >= self.graph.num_berths()`.
    #[inline]
    pub fn relocate_to_tail(&mut self, vessel_to_move: VesselIndex, target_berth: BerthIndex) {
        assert!(
            vessel_to_move.get() < self.graph.num_vessels(),
            "Mutator::relocate_to_tail bounds check failed: subject = {}, num_vessels = {}",
            vessel_to_move.get(),
            self.graph.num_vessels()
        );
        assert!(
            target_berth.get() < self.graph.num_berths(),
            "Mutator::relocate_to_tail bounds check failed: target_berth = {}, num_berths = {}",
            target_berth.get(),
            self.graph.num_berths()
        );
        unsafe { self.relocate_to_tail_unchecked(vessel_to_move, target_berth) }
    }

    /// Relocates an entire segment to immediately follow a reference vessel.
    ///
    /// # Panics
    ///
    /// Panics if any of the provided vessel indices are out of bounds.
    #[inline]
    pub fn relocate_segment_after(
        &mut self,
        segment_first: VesselIndex,
        segment_last: VesselIndex,
        insertion_anchor: VesselIndex,
    ) {
        assert!(
            segment_first.get() < self.graph.num_vessels()
                && segment_last.get() < self.graph.num_vessels()
                && insertion_anchor.get() < self.graph.num_vessels(),
            "Mutator::relocate_segment_after bounds check failed: first = {}, last = {}, anchor = {}, num_vessels = {}",
            segment_first.get(),
            segment_last.get(),
            insertion_anchor.get(),
            self.graph.num_vessels()
        );
        unsafe {
            self.relocate_segment_after_unchecked(segment_first, segment_last, insertion_anchor)
        }
    }

    /// Relocates an entire segment to immediately precede a reference vessel.
    ///
    /// # Panics
    ///
    /// Panics if any of the provided vessel indices are out of bounds.
    #[inline]
    pub fn relocate_segment_before(
        &mut self,
        segment_first: VesselIndex,
        segment_last: VesselIndex,
        reference_vessel: VesselIndex,
    ) {
        assert!(
            segment_first.get() < self.graph.num_vessels()
                && segment_last.get() < self.graph.num_vessels()
                && reference_vessel.get() < self.graph.num_vessels(),
            "Mutator::relocate_segment_before bounds check failed: first = {}, last = {}, reference = {}, num_vessels = {}",
            segment_first.get(),
            segment_last.get(),
            reference_vessel.get(),
            self.graph.num_vessels()
        );
        unsafe {
            self.relocate_segment_before_unchecked(segment_first, segment_last, reference_vessel)
        }
    }

    /// Relocates an entire segment to the head of a target berth.
    ///
    /// # Panics
    ///
    /// Panics if any vessel index is out of bounds, or if the target berth is out of bounds.
    #[inline]
    pub fn relocate_segment_to_head(
        &mut self,
        segment_first: VesselIndex,
        segment_last: VesselIndex,
        target_berth: BerthIndex,
    ) {
        assert!(
            segment_first.get() < self.graph.num_vessels()
                && segment_last.get() < self.graph.num_vessels(),
            "Mutator::relocate_segment_to_head bounds check failed: first = {}, last = {}, num_vessels = {}",
            segment_first.get(),
            segment_last.get(),
            self.graph.num_vessels()
        );
        assert!(
            target_berth.get() < self.graph.num_berths(),
            "Mutator::relocate_segment_to_head bounds check failed: target_berth = {}, num_berths = {}",
            target_berth.get(),
            self.graph.num_berths()
        );
        unsafe {
            self.relocate_segment_to_head_unchecked(segment_first, segment_last, target_berth)
        }
    }

    /// Relocates an entire segment to the tail of a target berth.
    ///
    /// # Panics
    ///
    /// Panics if any vessel index is out of bounds, or if the target berth is out of bounds.
    #[inline]
    pub fn relocate_segment_to_tail(
        &mut self,
        segment_first: VesselIndex,
        segment_last: VesselIndex,
        target_berth: BerthIndex,
    ) {
        assert!(
            segment_first.get() < self.graph.num_vessels()
                && segment_last.get() < self.graph.num_vessels(),
            "Mutator::relocate_segment_to_tail bounds check failed: first = {}, last = {}, num_vessels = {}",
            segment_first.get(),
            segment_last.get(),
            self.graph.num_vessels()
        );
        assert!(
            target_berth.get() < self.graph.num_berths(),
            "Mutator::relocate_segment_to_tail bounds check failed: target_berth = {}, num_berths = {}",
            target_berth.get(),
            self.graph.num_berths()
        );
        unsafe {
            self.relocate_segment_to_tail_unchecked(segment_first, segment_last, target_berth)
        }
    }

    // ----------------------------------------------------------------
    // Unchecked Mutations
    // ----------------------------------------------------------------

    /// Swaps two vessels without checking if indices are sentinels or out of bounds.
    ///
    /// # Safety
    ///
    /// Both `first_vessel` and `second_vessel` must be strictly `< self.graph.num_vessels()`.
    /// Passing a sentinel index or out-of-bounds index will result in undefined behavior,
    /// likely manifesting as memory corruption or out-of-bounds array access.
    #[inline]
    pub unsafe fn swap_vessels_unchecked(
        &mut self,
        first_vessel: VesselIndex,
        second_vessel: VesselIndex,
    ) {
        debug_assert!(
            first_vessel.get() < self.graph.num_vessels(),
            "Unchecked bounds violation: swap_vessels_unchecked first_vessel {} >= {}",
            first_vessel.get(),
            self.graph.num_vessels()
        );
        debug_assert!(
            second_vessel.get() < self.graph.num_vessels(),
            "Unchecked bounds violation: swap_vessels_unchecked second_vessel {} >= {}",
            second_vessel.get(),
            self.graph.num_vessels()
        );

        let node_a = ScheduleGraphNodeIndex::from_vessel(first_vessel);
        let node_b = ScheduleGraphNodeIndex::from_vessel(second_vessel);

        if node_a != node_b {
            let mut tracker = EdgeDeltaTracker::new();
            tracker.track(unsafe { self.graph.raw_prev_unchecked(node_a) }, self.graph);
            tracker.track(unsafe { self.graph.raw_prev_unchecked(node_b) }, self.graph);
            tracker.track(node_a, self.graph);
            tracker.track(node_b, self.graph);

            let old_b1 = unsafe { self.graph.node_berth_unchecked(node_a) };
            let old_b2 = unsafe { self.graph.node_berth_unchecked(node_b) };

            self.graph_undo.push_swap_nodes(node_a, node_b);
            unsafe { self.graph.swap_nodes_unchecked(node_a, node_b) };

            tracker.commit(self.graph, self.graph_diff);

            let new_b1 = unsafe { self.graph.node_berth_unchecked(node_a) };
            let new_b2 = unsafe { self.graph.node_berth_unchecked(node_b) };

            if old_b1 != new_b1 {
                self.graph_diff
                    .push_reallocation(first_vessel, old_b1, new_b1);
            }
            if old_b2 != new_b2 {
                self.graph_diff
                    .push_reallocation(second_vessel, old_b2, new_b2);
            }
        }
    }

    /// Swaps two segments without bounds or sentinel checks.
    ///
    /// # Safety
    ///
    /// All provided indices must be strictly `< self.graph.num_vessels()`.
    /// Segments must be valid and non-overlapping.
    #[inline]
    pub unsafe fn swap_segments_unchecked(
        &mut self,
        segment_a_first: VesselIndex,
        segment_a_last: VesselIndex,
        segment_b_first: VesselIndex,
        segment_b_last: VesselIndex,
    ) {
        debug_assert!(
            segment_a_first.get() < self.graph.num_vessels(),
            "Unchecked bounds violation: swap_segments_unchecked a_first {} >= {}",
            segment_a_first.get(),
            self.graph.num_vessels()
        );
        debug_assert!(
            segment_a_last.get() < self.graph.num_vessels(),
            "Unchecked bounds violation: swap_segments_unchecked a_last {} >= {}",
            segment_a_last.get(),
            self.graph.num_vessels()
        );
        debug_assert!(
            segment_b_first.get() < self.graph.num_vessels(),
            "Unchecked bounds violation: swap_segments_unchecked b_first {} >= {}",
            segment_b_first.get(),
            self.graph.num_vessels()
        );
        debug_assert!(
            segment_b_last.get() < self.graph.num_vessels(),
            "Unchecked bounds violation: swap_segments_unchecked b_last {} >= {}",
            segment_b_last.get(),
            self.graph.num_vessels()
        );

        let a_first = ScheduleGraphNodeIndex::from_vessel(segment_a_first);
        let a_last = ScheduleGraphNodeIndex::from_vessel(segment_a_last);
        let b_first = ScheduleGraphNodeIndex::from_vessel(segment_b_first);
        let b_last = ScheduleGraphNodeIndex::from_vessel(segment_b_last);

        if a_first != b_first {
            let mut tracker = EdgeDeltaTracker::new();
            tracker.track(
                unsafe { self.graph.raw_prev_unchecked(a_first) },
                self.graph,
            );
            tracker.track(a_last, self.graph);
            tracker.track(
                unsafe { self.graph.raw_prev_unchecked(b_first) },
                self.graph,
            );
            tracker.track(b_last, self.graph);

            let old_b_a = unsafe { self.graph.node_berth_unchecked(a_first) };
            let old_b_b = unsafe { self.graph.node_berth_unchecked(b_first) };

            self.graph_undo
                .push_swap_segments(a_first, a_last, b_first, b_last);
            unsafe {
                self.graph
                    .swap_segments_unchecked(a_first, a_last, b_first, b_last)
            };

            tracker.commit(self.graph, self.graph_diff);

            unsafe {
                self.record_segment_reallocation_unchecked(
                    segment_a_first,
                    segment_a_last,
                    old_b_a,
                    old_b_b,
                );
                self.record_segment_reallocation_unchecked(
                    segment_b_first,
                    segment_b_last,
                    old_b_b,
                    old_b_a,
                );
            }
        }
    }

    /// Reverses a segment without bounds or sentinel checks.
    ///
    /// # Safety
    ///
    /// Indices must be strictly `< self.graph.num_vessels()`. The segment must be a valid,
    /// contiguous chain in the graph.
    #[inline]
    pub unsafe fn reverse_segment_unchecked(
        &mut self,
        segment_first: VesselIndex,
        segment_last: VesselIndex,
    ) {
        debug_assert!(
            segment_first.get() < self.graph.num_vessels(),
            "Unchecked bounds violation: reverse_segment_unchecked first {} >= {}",
            segment_first.get(),
            self.graph.num_vessels()
        );
        debug_assert!(
            segment_last.get() < self.graph.num_vessels(),
            "Unchecked bounds violation: reverse_segment_unchecked last {} >= {}",
            segment_last.get(),
            self.graph.num_vessels()
        );

        let first_node = ScheduleGraphNodeIndex::from_vessel(segment_first);
        let last_node = ScheduleGraphNodeIndex::from_vessel(segment_last);

        if first_node != last_node {
            // Reversing a segment breaks EVERY internal forward link and creates them in reverse.
            // EdgeDeltaTracker is fixed at 4 elements, so we manually map the diff here.
            let prev_first = unsafe { self.graph.raw_prev_unchecked(first_node) };
            let next_last = unsafe { self.graph.raw_next_unchecked(last_node) };

            let prev_first_v = prev_first
                .as_vessel(self.graph.num_vessels())
                .unwrap_or(VesselIndex::new(prev_first.get()));
            let next_last_v = next_last
                .as_vessel(self.graph.num_vessels())
                .unwrap_or(VesselIndex::new(next_last.get()));

            self.graph_diff
                .push_link_broken(prev_first_v, segment_first);
            self.graph_diff.push_link_broken(segment_last, next_last_v);

            let mut curr = first_node;
            while curr != last_node {
                let nxt = unsafe { self.graph.raw_next_unchecked(curr) };
                let curr_v = VesselIndex::new(curr.get());
                let nxt_v = VesselIndex::new(nxt.get());

                self.graph_diff.push_link_broken(curr_v, nxt_v);
                self.graph_diff.push_link_created(nxt_v, curr_v);
                curr = nxt;
            }

            self.graph_diff
                .push_link_created(prev_first_v, segment_last);
            self.graph_diff
                .push_link_created(segment_first, next_last_v);

            self.graph_undo.push_reverse_segment(first_node, last_node);
            unsafe { self.graph.reverse_segment_unchecked(first_node, last_node) };

            // Reversing a segment locally never shifts its Berth association,
            // so reallocations are strictly zero.
        }
    }

    /// Relocates a vessel after another without bounds checks.
    ///
    /// # Safety
    ///
    /// Indices must be strictly `< self.graph.num_vessels()`.
    #[inline]
    pub unsafe fn relocate_after_unchecked(
        &mut self,
        vessel_to_move: VesselIndex,
        insertion_anchor: VesselIndex,
    ) {
        debug_assert!(
            vessel_to_move.get() < self.graph.num_vessels(),
            "Unchecked bounds violation: relocate_after_unchecked subject {} >= {}",
            vessel_to_move.get(),
            self.graph.num_vessels()
        );
        debug_assert!(
            insertion_anchor.get() < self.graph.num_vessels(),
            "Unchecked bounds violation: relocate_after_unchecked anchor {} >= {}",
            insertion_anchor.get(),
            self.graph.num_vessels()
        );

        let subject_node = ScheduleGraphNodeIndex::from_vessel(vessel_to_move);
        let anchor_node = ScheduleGraphNodeIndex::from_vessel(insertion_anchor);
        let prev = unsafe { self.graph.raw_prev_unchecked(subject_node) };

        if subject_node != anchor_node && prev != anchor_node {
            let mut tracker = EdgeDeltaTracker::new();
            tracker.track(prev, self.graph);
            tracker.track(subject_node, self.graph);
            tracker.track(anchor_node, self.graph);

            let old_b = unsafe { self.graph.node_berth_unchecked(subject_node) };

            unsafe { self.record_relocate_unchecked(vessel_to_move) };
            unsafe {
                self.graph
                    .relocate_after_unchecked(subject_node, anchor_node)
            };

            tracker.commit(self.graph, self.graph_diff);

            let new_b = unsafe { self.graph.node_berth_unchecked(subject_node) };
            if old_b != new_b {
                self.graph_diff
                    .push_reallocation(vessel_to_move, old_b, new_b);
            }
        }
    }

    /// Relocates a vessel before another without bounds checks.
    ///
    /// # Safety
    ///
    /// Indices must be strictly `< self.graph.num_vessels()`.
    #[inline]
    pub unsafe fn relocate_before_unchecked(
        &mut self,
        vessel_to_move: VesselIndex,
        reference_vessel: VesselIndex,
    ) {
        debug_assert!(
            vessel_to_move.get() < self.graph.num_vessels(),
            "Unchecked bounds violation: relocate_before_unchecked subject {} >= {}",
            vessel_to_move.get(),
            self.graph.num_vessels()
        );
        debug_assert!(
            reference_vessel.get() < self.graph.num_vessels(),
            "Unchecked bounds violation: relocate_before_unchecked reference {} >= {}",
            reference_vessel.get(),
            self.graph.num_vessels()
        );

        let ref_node = ScheduleGraphNodeIndex::from_vessel(reference_vessel);
        let reference_predecessor = unsafe { self.graph.raw_prev_unchecked(ref_node) };
        let subject_node = ScheduleGraphNodeIndex::from_vessel(vessel_to_move);
        let prev = unsafe { self.graph.raw_prev_unchecked(subject_node) };

        if subject_node != ref_node && prev != reference_predecessor {
            let mut tracker = EdgeDeltaTracker::new();
            tracker.track(prev, self.graph);
            tracker.track(subject_node, self.graph);
            tracker.track(reference_predecessor, self.graph);

            let old_b = unsafe { self.graph.node_berth_unchecked(subject_node) };

            unsafe { self.record_relocate_unchecked(vessel_to_move) };
            unsafe { self.graph.relocate_before_unchecked(subject_node, ref_node) };

            tracker.commit(self.graph, self.graph_diff);

            let new_b = unsafe { self.graph.node_berth_unchecked(subject_node) };
            if old_b != new_b {
                self.graph_diff
                    .push_reallocation(vessel_to_move, old_b, new_b);
            }
        }
    }

    /// Relocates a vessel to the head of a berth without bounds checks.
    ///
    /// # Safety
    ///
    /// `vessel_to_move` must be strictly `< self.graph.num_vessels()`.
    /// `target_berth` must be strictly `< self.graph.num_berths()`.
    #[inline]
    pub unsafe fn relocate_to_head_unchecked(
        &mut self,
        vessel_to_move: VesselIndex,
        target_berth: BerthIndex,
    ) {
        debug_assert!(
            vessel_to_move.get() < self.graph.num_vessels(),
            "Unchecked bounds violation: relocate_to_head_unchecked subject {} >= {}",
            vessel_to_move.get(),
            self.graph.num_vessels()
        );
        debug_assert!(
            target_berth.get() < self.graph.num_berths(),
            "Unchecked bounds violation: relocate_to_head_unchecked target_berth {} >= {}",
            target_berth.get(),
            self.graph.num_berths()
        );

        let subject_node = ScheduleGraphNodeIndex::from_vessel(vessel_to_move);
        let prev = unsafe { self.graph.raw_prev_unchecked(subject_node) };
        let target_sentinel =
            ScheduleGraphNodeIndex::from_sentinel(target_berth, self.graph.num_vessels());

        let mut tracker = EdgeDeltaTracker::new();
        tracker.track(prev, self.graph);
        tracker.track(subject_node, self.graph);
        tracker.track(target_sentinel, self.graph);

        let old_b = unsafe { self.graph.node_berth_unchecked(subject_node) };

        unsafe { self.record_relocate_unchecked(vessel_to_move) };
        unsafe {
            self.graph
                .relocate_to_head_unchecked(subject_node, target_berth)
        };

        tracker.commit(self.graph, self.graph_diff);

        if old_b != target_berth {
            self.graph_diff
                .push_reallocation(vessel_to_move, old_b, target_berth);
        }
    }

    /// Relocates a vessel to the tail of a berth without bounds checks.
    ///
    /// # Safety
    ///
    /// `vessel_to_move` must be strictly `< self.graph.num_vessels()`.
    /// `target_berth` must be strictly `< self.graph.num_berths()`.
    #[inline]
    pub unsafe fn relocate_to_tail_unchecked(
        &mut self,
        vessel_to_move: VesselIndex,
        target_berth: BerthIndex,
    ) {
        debug_assert!(
            vessel_to_move.get() < self.graph.num_vessels(),
            "Unchecked bounds violation: relocate_to_tail_unchecked subject {} >= {}",
            vessel_to_move.get(),
            self.graph.num_vessels()
        );
        debug_assert!(
            target_berth.get() < self.graph.num_berths(),
            "Unchecked bounds violation: relocate_to_tail_unchecked target_berth {} >= {}",
            target_berth.get(),
            self.graph.num_berths()
        );

        let subject_node = ScheduleGraphNodeIndex::from_vessel(vessel_to_move);
        let prev = unsafe { self.graph.raw_prev_unchecked(subject_node) };
        let target_sentinel =
            ScheduleGraphNodeIndex::from_sentinel(target_berth, self.graph.num_vessels());
        let old_tail = unsafe { self.graph.raw_prev_unchecked(target_sentinel) };

        let mut tracker = EdgeDeltaTracker::new();
        tracker.track(prev, self.graph);
        tracker.track(subject_node, self.graph);
        tracker.track(old_tail, self.graph);

        let old_b = unsafe { self.graph.node_berth_unchecked(subject_node) };

        unsafe { self.record_relocate_unchecked(vessel_to_move) };
        unsafe {
            self.graph
                .relocate_to_tail_unchecked(subject_node, target_berth)
        };

        tracker.commit(self.graph, self.graph_diff);

        if old_b != target_berth {
            self.graph_diff
                .push_reallocation(vessel_to_move, old_b, target_berth);
        }
    }

    /// Relocates a segment after a vessel without bounds checks.
    ///
    /// # Safety
    ///
    /// All vessel indices must be strictly `< self.graph.num_vessels()`.
    #[inline]
    pub unsafe fn relocate_segment_after_unchecked(
        &mut self,
        segment_first: VesselIndex,
        segment_last: VesselIndex,
        insertion_anchor: VesselIndex,
    ) {
        debug_assert!(
            segment_first.get() < self.graph.num_vessels(),
            "Unchecked bounds violation: relocate_segment_after_unchecked first {} >= {}",
            segment_first.get(),
            self.graph.num_vessels()
        );
        debug_assert!(
            segment_last.get() < self.graph.num_vessels(),
            "Unchecked bounds violation: relocate_segment_after_unchecked last {} >= {}",
            segment_last.get(),
            self.graph.num_vessels()
        );
        debug_assert!(
            insertion_anchor.get() < self.graph.num_vessels(),
            "Unchecked bounds violation: relocate_segment_after_unchecked anchor {} >= {}",
            insertion_anchor.get(),
            self.graph.num_vessels()
        );

        let first_node = ScheduleGraphNodeIndex::from_vessel(segment_first);
        let last_node = ScheduleGraphNodeIndex::from_vessel(segment_last);
        let anchor_node = ScheduleGraphNodeIndex::from_vessel(insertion_anchor);

        let prev = unsafe { self.graph.raw_prev_unchecked(first_node) };
        if prev != anchor_node {
            let mut tracker = EdgeDeltaTracker::new();
            tracker.track(prev, self.graph);
            tracker.track(last_node, self.graph);
            tracker.track(anchor_node, self.graph);

            let old_b = unsafe { self.graph.node_berth_unchecked(first_node) };

            unsafe { self.record_relocate_segment_unchecked(segment_first, segment_last) };
            unsafe {
                self.graph
                    .relocate_segment_after_unchecked(first_node, last_node, anchor_node)
            };

            tracker.commit(self.graph, self.graph_diff);

            let new_b = unsafe { self.graph.node_berth_unchecked(first_node) };
            unsafe {
                self.record_segment_reallocation_unchecked(
                    segment_first,
                    segment_last,
                    old_b,
                    new_b,
                )
            };
        }
    }

    /// Relocates a segment before a vessel without bounds checks.
    ///
    /// # Safety
    ///
    /// All vessel indices must be strictly `< self.graph.num_vessels()`.
    #[inline]
    pub unsafe fn relocate_segment_before_unchecked(
        &mut self,
        segment_first: VesselIndex,
        segment_last: VesselIndex,
        reference_vessel: VesselIndex,
    ) {
        debug_assert!(
            segment_first.get() < self.graph.num_vessels(),
            "Unchecked bounds violation: relocate_segment_before_unchecked first {} >= {}",
            segment_first.get(),
            self.graph.num_vessels()
        );
        debug_assert!(
            segment_last.get() < self.graph.num_vessels(),
            "Unchecked bounds violation: relocate_segment_before_unchecked last {} >= {}",
            segment_last.get(),
            self.graph.num_vessels()
        );
        debug_assert!(
            reference_vessel.get() < self.graph.num_vessels(),
            "Unchecked bounds violation: relocate_segment_before_unchecked reference {} >= {}",
            reference_vessel.get(),
            self.graph.num_vessels()
        );

        let first_node = ScheduleGraphNodeIndex::from_vessel(segment_first);
        let last_node = ScheduleGraphNodeIndex::from_vessel(segment_last);
        let ref_node = ScheduleGraphNodeIndex::from_vessel(reference_vessel);

        let reference_predecessor = unsafe { self.graph.raw_prev_unchecked(ref_node) };
        let prev = unsafe { self.graph.raw_prev_unchecked(first_node) };

        if prev != reference_predecessor {
            let mut tracker = EdgeDeltaTracker::new();
            tracker.track(prev, self.graph);
            tracker.track(last_node, self.graph);
            tracker.track(reference_predecessor, self.graph);

            let old_b = unsafe { self.graph.node_berth_unchecked(first_node) };

            unsafe { self.record_relocate_segment_unchecked(segment_first, segment_last) };
            unsafe {
                self.graph
                    .relocate_segment_before_unchecked(first_node, last_node, ref_node)
            };

            tracker.commit(self.graph, self.graph_diff);

            let new_b = unsafe { self.graph.node_berth_unchecked(first_node) };
            unsafe {
                self.record_segment_reallocation_unchecked(
                    segment_first,
                    segment_last,
                    old_b,
                    new_b,
                )
            };
        }
    }

    /// Relocates a segment to the head of a berth without bounds checks.
    ///
    /// # Safety
    ///
    /// All vessel indices must be strictly `< self.graph.num_vessels()`.
    /// `target_berth` must be strictly `< self.graph.num_berths()`.
    #[inline]
    pub unsafe fn relocate_segment_to_head_unchecked(
        &mut self,
        segment_first: VesselIndex,
        segment_last: VesselIndex,
        target_berth: BerthIndex,
    ) {
        debug_assert!(
            segment_first.get() < self.graph.num_vessels(),
            "Unchecked bounds violation: relocate_segment_to_head_unchecked first {} >= {}",
            segment_first.get(),
            self.graph.num_vessels()
        );
        debug_assert!(
            segment_last.get() < self.graph.num_vessels(),
            "Unchecked bounds violation: relocate_segment_to_head_unchecked last {} >= {}",
            segment_last.get(),
            self.graph.num_vessels()
        );
        debug_assert!(
            target_berth.get() < self.graph.num_berths(),
            "Unchecked bounds violation: relocate_segment_to_head_unchecked target_berth {} >= {}",
            target_berth.get(),
            self.graph.num_berths()
        );

        let first_node = ScheduleGraphNodeIndex::from_vessel(segment_first);
        let last_node = ScheduleGraphNodeIndex::from_vessel(segment_last);
        let target_sentinel =
            ScheduleGraphNodeIndex::from_sentinel(target_berth, self.graph.num_vessels());

        let prev = unsafe { self.graph.raw_prev_unchecked(first_node) };

        let mut tracker = EdgeDeltaTracker::new();
        tracker.track(prev, self.graph);
        tracker.track(last_node, self.graph);
        tracker.track(target_sentinel, self.graph);

        let old_b = unsafe { self.graph.node_berth_unchecked(first_node) };

        unsafe { self.record_relocate_segment_unchecked(segment_first, segment_last) };
        unsafe {
            self.graph
                .relocate_segment_to_head_unchecked(first_node, last_node, target_berth)
        };

        tracker.commit(self.graph, self.graph_diff);

        unsafe {
            self.record_segment_reallocation_unchecked(
                segment_first,
                segment_last,
                old_b,
                target_berth,
            )
        };
    }

    /// Relocates a segment to the tail of a berth without bounds checks.
    ///
    /// # Safety
    ///
    /// All vessel indices must be strictly `< self.graph.num_vessels()`.
    /// `target_berth` must be strictly `< self.graph.num_berths()`.
    #[inline]
    pub unsafe fn relocate_segment_to_tail_unchecked(
        &mut self,
        segment_first: VesselIndex,
        segment_last: VesselIndex,
        target_berth: BerthIndex,
    ) {
        debug_assert!(
            segment_first.get() < self.graph.num_vessels(),
            "Unchecked bounds violation: relocate_segment_to_tail_unchecked first {} >= {}",
            segment_first.get(),
            self.graph.num_vessels()
        );
        debug_assert!(
            segment_last.get() < self.graph.num_vessels(),
            "Unchecked bounds violation: relocate_segment_to_tail_unchecked last {} >= {}",
            segment_last.get(),
            self.graph.num_vessels()
        );
        debug_assert!(
            target_berth.get() < self.graph.num_berths(),
            "Unchecked bounds violation: relocate_segment_to_tail_unchecked target_berth {} >= {}",
            target_berth.get(),
            self.graph.num_berths()
        );

        let first_node = ScheduleGraphNodeIndex::from_vessel(segment_first);
        let last_node = ScheduleGraphNodeIndex::from_vessel(segment_last);
        let target_sentinel =
            ScheduleGraphNodeIndex::from_sentinel(target_berth, self.graph.num_vessels());

        let prev = unsafe { self.graph.raw_prev_unchecked(first_node) };
        let old_tail = unsafe { self.graph.raw_prev_unchecked(target_sentinel) };

        let mut tracker = EdgeDeltaTracker::new();
        tracker.track(prev, self.graph);
        tracker.track(last_node, self.graph);
        tracker.track(old_tail, self.graph);

        let old_b = unsafe { self.graph.node_berth_unchecked(first_node) };

        unsafe { self.record_relocate_segment_unchecked(segment_first, segment_last) };
        unsafe {
            self.graph
                .relocate_segment_to_tail_unchecked(first_node, last_node, target_berth)
        };

        tracker.commit(self.graph, self.graph_diff);

        unsafe {
            self.record_segment_reallocation_unchecked(
                segment_first,
                segment_last,
                old_b,
                target_berth,
            )
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[inline]
    fn b(i: usize) -> BerthIndex {
        BerthIndex::new(i)
    }

    #[inline]
    fn v(i: usize) -> VesselIndex {
        VesselIndex::new(i)
    }

    /// Sets up a standard 3-berth graph for testing mutations
    /// B0: [S5] -> V0 -> V1 -> V2 -> [S5]
    /// B1: [S6] -> V3 -> V4 -> [S6]
    /// B2: [S7] -> [S7]  (empty)
    fn setup_graph() -> ScheduleGraph {
        let berths = [b(0), b(0), b(0), b(1), b(1)];
        let starts = [10, 20, 30, 10, 20];
        ScheduleGraph::from_slices(&berths, &starts, 3)
    }

    // Helper to extract diffs as tuples for easy assertion
    #[allow(clippy::type_complexity)]
    fn extract_diff_links(
        diff: &ScheduleGraphDiff,
    ) -> (
        Vec<(Option<VesselIndex>, Option<VesselIndex>)>,
        Vec<(Option<VesselIndex>, Option<VesselIndex>)>,
    ) {
        let broken = diff.broken_links().map(|e| (e.from, e.to)).collect();
        let created = diff.created_links().map(|e| (e.from, e.to)).collect();
        (broken, created)
    }

    fn extract_reallocations(
        diff: &ScheduleGraphDiff,
    ) -> Vec<(VesselIndex, BerthIndex, BerthIndex)> {
        diff.reallocations().collect()
    }

    #[test]
    fn test_mutator_swap_vessels_same_berth() {
        let mut graph = setup_graph();
        let mut undo = ScheduleGraphUndoLog::new(10);
        let mut diff = ScheduleGraphDiff::new(graph.num_vessels());

        {
            let mut mutator = Mutator::new(&mut graph, &mut undo, &mut diff);
            // Swap V0 and V2 in B0
            // Original B0: [S] -> V0 -> V1 -> V2 -> [S]
            // Expected B0: [S] -> V2 -> V1 -> V0 -> [S]
            mutator.swap_vessels(v(0), v(2));
        }

        let seq: Vec<_> = graph.vessel_sequence_iter(b(0)).collect();
        assert_eq!(seq, vec![v(2), v(1), v(0)]);

        let (broken, created) = extract_diff_links(&diff);

        // Broken links: S->V0, V0->V1, V1->V2, V2->S
        assert!(broken.contains(&(None, Some(v(0)))));
        assert!(broken.contains(&(Some(v(0)), Some(v(1)))));
        assert!(broken.contains(&(Some(v(1)), Some(v(2)))));
        assert!(broken.contains(&(Some(v(2)), None)));
        assert_eq!(broken.len(), 4);

        // Created links: S->V2, V2->V1, V1->V0, V0->S
        assert!(created.contains(&(None, Some(v(2)))));
        assert!(created.contains(&(Some(v(2)), Some(v(1)))));
        assert!(created.contains(&(Some(v(1)), Some(v(0)))));
        assert!(created.contains(&(Some(v(0)), None)));
        assert_eq!(created.len(), 4);

        // No reallocations, stayed in B0
        assert!(extract_reallocations(&diff).is_empty());
    }

    #[test]
    fn test_mutator_relocate_to_head_cross_berth_diff() {
        let mut graph = setup_graph();
        let mut undo = ScheduleGraphUndoLog::new(10);
        let mut diff = ScheduleGraphDiff::new(graph.num_vessels());

        {
            let mut mutator = Mutator::new(&mut graph, &mut undo, &mut diff);
            // Move V0 from head of B0 to head of B1
            mutator.relocate_to_head(v(0), b(1));
        }

        // B0 should now be V1 -> V2
        assert_eq!(
            graph.vessel_sequence_iter(b(0)).collect::<Vec<_>>(),
            vec![v(1), v(2)]
        );
        // B1 should now be V0 -> V3 -> V4
        assert_eq!(
            graph.vessel_sequence_iter(b(1)).collect::<Vec<_>>(),
            vec![v(0), v(3), v(4)]
        );

        let (broken, created) = extract_diff_links(&diff);

        // Broken: S5->V0, V0->V1, S6->V3
        assert!(broken.contains(&(None, Some(v(0)))));
        assert!(broken.contains(&(Some(v(0)), Some(v(1)))));
        assert!(broken.contains(&(None, Some(v(3)))));

        // Created: S5->V1, S6->V0, V0->V3
        assert!(created.contains(&(None, Some(v(1)))));
        assert!(created.contains(&(None, Some(v(0)))));
        assert!(created.contains(&(Some(v(0)), Some(v(3)))));

        // Reallocations
        let reallocs = extract_reallocations(&diff);
        assert_eq!(reallocs, vec![(v(0), b(0), b(1))]);
    }

    #[test]
    fn test_mutator_relocate_segment_after_cross_berth() {
        let mut graph = setup_graph();
        let mut undo = ScheduleGraphUndoLog::new(10);
        let mut diff = ScheduleGraphDiff::new(graph.num_vessels());

        {
            let mut mutator = Mutator::new(&mut graph, &mut undo, &mut diff);
            // Move [V0, V1] from B0 to after V3 in B1.
            mutator.relocate_segment_after(v(0), v(1), v(3));
        }

        assert_eq!(
            graph.vessel_sequence_iter(b(0)).collect::<Vec<_>>(),
            vec![v(2)]
        );
        assert_eq!(
            graph.vessel_sequence_iter(b(1)).collect::<Vec<_>>(),
            vec![v(3), v(0), v(1), v(4)]
        );

        let (broken, created) = extract_diff_links(&diff);

        // Broken: S5->V0, V1->V2, V3->V4
        assert!(broken.contains(&(None, Some(v(0)))));
        assert!(broken.contains(&(Some(v(1)), Some(v(2)))));
        assert!(broken.contains(&(Some(v(3)), Some(v(4)))));

        // Created: S5->V2, V3->V0, V1->V4
        assert!(created.contains(&(None, Some(v(2)))));
        assert!(created.contains(&(Some(v(3)), Some(v(0)))));
        assert!(created.contains(&(Some(v(1)), Some(v(4)))));

        let reallocs = extract_reallocations(&diff);
        // Order inside the diff is determined by the internal loop,
        // which iterates forward through the segment.
        assert_eq!(reallocs, vec![(v(0), b(0), b(1)), (v(1), b(0), b(1))]);
    }

    #[test]
    fn test_mutator_reverse_segment_diff() {
        let mut graph = setup_graph();
        let mut undo = ScheduleGraphUndoLog::new(10);
        let mut diff = ScheduleGraphDiff::new(graph.num_vessels());

        {
            let mut mutator = Mutator::new(&mut graph, &mut undo, &mut diff);
            // Reverse [V0, V1, V2] in B0
            mutator.reverse_segment(v(0), v(2));
        }

        assert_eq!(
            graph.vessel_sequence_iter(b(0)).collect::<Vec<_>>(),
            vec![v(2), v(1), v(0)]
        );

        let (broken, created) = extract_diff_links(&diff);

        // Broken: S->V0, V0->V1, V1->V2, V2->S
        assert!(broken.contains(&(None, Some(v(0)))));
        assert!(broken.contains(&(Some(v(0)), Some(v(1)))));
        assert!(broken.contains(&(Some(v(1)), Some(v(2)))));
        assert!(broken.contains(&(Some(v(2)), None)));

        // Created: S->V2, V2->V1, V1->V0, V0->S
        assert!(created.contains(&(None, Some(v(2)))));
        assert!(created.contains(&(Some(v(2)), Some(v(1)))));
        assert!(created.contains(&(Some(v(1)), Some(v(0)))));
        assert!(created.contains(&(Some(v(0)), None)));

        // Reversing is strictly in-place; no berth reallocations should occur
        assert!(extract_reallocations(&diff).is_empty());
    }

    #[test]
    fn test_mutator_swap_segments_cross_berth() {
        let mut graph = setup_graph();
        let mut undo = ScheduleGraphUndoLog::new(10);
        let mut diff = ScheduleGraphDiff::new(graph.num_vessels());

        {
            let mut mutator = Mutator::new(&mut graph, &mut undo, &mut diff);
            // Swap [V0, V1] from B0 with [V3] from B1
            mutator.swap_segments(v(0), v(1), v(3), v(3));
        }

        assert_eq!(
            graph.vessel_sequence_iter(b(0)).collect::<Vec<_>>(),
            vec![v(3), v(2)]
        );
        assert_eq!(
            graph.vessel_sequence_iter(b(1)).collect::<Vec<_>>(),
            vec![v(0), v(1), v(4)]
        );

        let reallocs = extract_reallocations(&diff);

        // V0 and V1 moved from B0 to B1. V3 moved from B1 to B0.
        assert!(reallocs.contains(&(v(0), b(0), b(1))));
        assert!(reallocs.contains(&(v(1), b(0), b(1))));
        assert!(reallocs.contains(&(v(3), b(1), b(0))));
        assert_eq!(reallocs.len(), 3);
    }

    #[test]
    fn test_mutator_relocate_segment_to_tail_empty_berth() {
        let mut graph = setup_graph();
        let mut undo = ScheduleGraphUndoLog::new(10);
        let mut diff = ScheduleGraphDiff::new(graph.num_vessels());

        {
            let mut mutator = Mutator::new(&mut graph, &mut undo, &mut diff);
            // Move [V3, V4] from B1 to empty B2.
            mutator.relocate_segment_to_tail(v(3), v(4), b(2));
        }

        assert!(graph.vessel_sequence_iter(b(1)).next().is_none());
        assert_eq!(
            graph.vessel_sequence_iter(b(2)).collect::<Vec<_>>(),
            vec![v(3), v(4)]
        );

        let (broken, created) = extract_diff_links(&diff);

        // Broken: S6->V3, V4->S6 (emptying B1)
        assert!(broken.contains(&(None, Some(v(3)))));
        assert!(broken.contains(&(Some(v(4)), None)));
        // Broken: S7->S7 (breaking the empty loop of B2)
        assert!(broken.contains(&(None, None)));

        // Created: S6->S6 (closing B1), S7->V3, V4->S7 (inserting into B2)
        assert!(created.contains(&(None, None)));
        assert!(created.contains(&(None, Some(v(3)))));
        assert!(created.contains(&(Some(v(4)), None)));

        let reallocs = extract_reallocations(&diff);
        assert_eq!(reallocs, vec![(v(3), b(1), b(2)), (v(4), b(1), b(2))]);
    }
}
