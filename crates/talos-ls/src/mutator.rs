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

//! High-level mutation engine for ScheduleGraph topology changes.
//!
//! This module provides Mutator, a transactional wrapper around a
//! ScheduleGraph that intercepts every topological modification and
//! automatically records two side-products:
//!
//! - An undo log (ScheduleGraphUndoLog) that captures the inverse of each
//!   operation, enabling callers to roll back an arbitrary sequence of
//!   mutations in reverse order.
//!
//! - A structural diff (ScheduleGraphDiff) that records which linked-list
//!   edges were broken or created and which vessels were reallocated between
//!   berths. Downstream consumers use this diff to determine which berths
//!   need re-decoding after a mutation.
//!
//! Mutator exposes both bounds-checked and unchecked variants of every
//! operation. The checked variants panic on out-of-bounds indices while the
//! unchecked variants use only debug assertions and skip bounds checks in
//! release builds. The supported operations are:
//!
//! - Single-vessel and segment swaps
//! - Segment reversal
//! - Relocation of a single vessel or contiguous segment to an arbitrary
//!   position (after/before a reference vessel, or to the head/tail of a
//!   berth)
//!
//! Edge diff tracking is handled by EdgeDeltaTracker, a stack-allocated
//! micro-tracker that snapshots up to four nodes' next pointers before a
//! mutation and emits the net edge changes into the diff afterward.

use crate::{
    sgraph::{ScheduleGraph, ScheduleGraphDiff},
    sgraphundo::ScheduleGraphUndoLog,
};
use talos_core::container::rarena::Node;
use talos_model::index::{BerthIndex, VesselIndex};

// ----------------------------------------------------------------
// EdgeDeltaTracker
// ----------------------------------------------------------------

/// A micro-tracker that lives purely on the stack.
///
/// Records up to 4 nodes' `next` pointers *before* a mutation, then after
/// the mutation, emits the exact net edge differences into the diff.
///
/// This is the one place that reaches through `graph.arena()` for raw
/// topology access — justified by the hot-loop performance requirement.
struct EdgeDeltaTracker {
    nodes: [Node; 4],
    old_nexts: [Node; 4],
    len: usize,
}

impl EdgeDeltaTracker {
    #[inline(always)]
    fn new() -> Self {
        Self {
            nodes: [Node::new(0); 4],
            old_nexts: [Node::new(0); 4],
            len: 0,
        }
    }

    /// Records a node's `next` pointer before mutation.
    /// Uses raw arena access for zero-overhead reads.
    #[inline(always)]
    fn track(&mut self, node: Node, graph: &ScheduleGraph) {
        // Unrolled dedup for len <= 4
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
            debug_assert!(self.len < 4);
            *self.nodes.get_unchecked_mut(self.len) = node;
            *self.old_nexts.get_unchecked_mut(self.len) = graph.next_node_unchecked(node);
        }
        self.len += 1;
        debug_assert!(self.len <= 4);
    }

    /// Compares recorded state against current state and emits diffs.
    #[inline(always)]
    fn commit(self, graph: &ScheduleGraph, diff: &mut ScheduleGraphDiff) {
        for i in 0..self.len {
            let node = unsafe { *self.nodes.get_unchecked(i) };
            let old_nxt = unsafe { *self.old_nexts.get_unchecked(i) };
            let new_nxt = unsafe { graph.next_node_unchecked(node) };

            if old_nxt != new_nxt {
                diff.push_link_broken(node, old_nxt);
                diff.push_link_created(node, new_nxt);
            }
        }
    }
}

// ----------------------------------------------------------------
// Mutator
// ----------------------------------------------------------------

/// A mutation engine for applying topological changes to a `ScheduleGraph`.
///
/// Automatically records every mutation into a `ScheduleGraphUndoLog` and
/// tracks edge/reallocation diffs. The public API uses only `VesselIndex`
/// and `BerthIndex`.
#[derive(Debug)]
pub struct Mutator<'a> {
    graph: &'a mut ScheduleGraph,
    undo: &'a mut ScheduleGraphUndoLog,
    diff: &'a mut ScheduleGraphDiff,
}

impl<'a> Mutator<'a> {
    #[inline(always)]
    pub fn new(
        graph: &'a mut ScheduleGraph,
        undo: &'a mut ScheduleGraphUndoLog,
        diff: &'a mut ScheduleGraphDiff,
    ) -> Self {
        Self { graph, undo, diff }
    }

    #[inline(always)]
    pub fn graph(&self) -> &ScheduleGraph {
        self.graph
    }

    #[inline(always)]
    pub fn undo(&mut self) -> &mut ScheduleGraphUndoLog {
        self.undo
    }

    /// Records the original position of a vessel before relocation.
    /// Uses `vessel_predecessor_unchecked` to determine if the vessel was
    /// after another vessel or at the head of its berth.
    #[inline(always)]
    unsafe fn record_relocate_unchecked(&mut self, vessel: VesselIndex) {
        match unsafe { self.graph.vessel_predecessor_unchecked(vessel) } {
            Some(pred) => {
                self.undo.push_relocate_after_vessel(vessel, pred);
            }
            None => {
                let berth = unsafe { self.graph.vessel_berth_unchecked(vessel) };
                self.undo.push_relocate_to_head(vessel, berth);
            }
        }
    }

    #[inline(always)]
    fn record_relocate(&mut self, vessel: VesselIndex) {
        match self.graph.vessel_predecessor(vessel) {
            Some(pred) => {
                self.undo.push_relocate_after_vessel(vessel, pred);
            }
            None => {
                let berth = self.graph.vessel_berth(vessel);
                self.undo.push_relocate_to_head(vessel, berth);
            }
        }
    }

    /// Records the original position of a segment before relocation.
    #[inline(always)]
    unsafe fn record_relocate_segment_unchecked(&mut self, first: VesselIndex, last: VesselIndex) {
        match unsafe { self.graph.vessel_predecessor_unchecked(first) } {
            Some(pred) => {
                self.undo
                    .push_relocate_segment_after_vessel(first, last, pred);
            }
            None => {
                let berth = unsafe { self.graph.vessel_berth_unchecked(first) };
                self.undo.push_relocate_segment_to_head(first, last, berth);
            }
        }
    }

    #[inline(always)]
    fn record_relocate_segment(&mut self, first: VesselIndex, last: VesselIndex) {
        match self.graph.vessel_predecessor(first) {
            Some(pred) => {
                self.undo
                    .push_relocate_segment_after_vessel(first, last, pred);
            }
            None => {
                let berth = self.graph.vessel_berth(first);
                self.undo.push_relocate_segment_to_head(first, last, berth);
            }
        }
    }

    /// Records reallocation diffs for every vessel in a segment that changed berths.
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
        let mut curr = self.graph.vessel_node(first);
        let last_node = self.graph.vessel_node(last);
        loop {
            self.diff
                .push_reallocation(VesselIndex::new(curr.get()), old_berth, new_berth);
            if curr == last_node {
                break;
            }
            curr = unsafe { self.graph.next_node_unchecked(curr) };
        }
    }

    #[inline(always)]
    fn record_segment_reallocation(
        &mut self,
        first: VesselIndex,
        last: VesselIndex,
        old_berth: BerthIndex,
        new_berth: BerthIndex,
    ) {
        if old_berth == new_berth {
            return;
        }

        let mut curr = self.graph.vessel_node(first);
        let last_node = self.graph.vessel_node(last);
        loop {
            self.diff
                .push_reallocation(VesselIndex::new(curr.get()), old_berth, new_berth);
            if curr == last_node {
                break;
            }
            curr = unsafe { self.graph.next_node_unchecked(curr) };
        }
    }

    /// Swaps the positions of two vessels in the schedule graph.
    ///
    /// ```text
    /// Before:
    ///   Berth X: ... <-> Prev_A <-> A <-> Next_A <-> ...
    ///   Berth Y: ... <-> Prev_B <-> B <-> Next_B <-> ...
    ///
    /// After:
    ///   Berth X: ... <-> Prev_A <-> B <-> Next_A <-> ...
    ///   Berth Y: ... <-> Prev_B <-> A <-> Next_B <-> ...
    /// ```
    ///
    /// Records the swap in the undo log, emits edge diffs for every
    /// changed `next` pointer, and emits reallocation entries if the
    /// two vessels belonged to different berths.
    ///
    /// No-op if `a == b`.
    ///
    /// # Panics
    ///
    /// Panics if either vessel is out of bounds.
    #[inline]
    pub fn swap_vessels(&mut self, a: VesselIndex, b: VesselIndex) {
        assert!(a < self.graph.num_vessels());
        assert!(b < self.graph.num_vessels());

        if a == b {
            return;
        }

        let a_node = self.graph.vessel_node(a);
        let b_node = self.graph.vessel_node(b);

        let mut tracker = EdgeDeltaTracker::new();
        tracker.track(
            unsafe { self.graph.prev_node_unchecked(a_node) },
            self.graph,
        );
        tracker.track(
            unsafe { self.graph.prev_node_unchecked(b_node) },
            self.graph,
        );
        tracker.track(a_node, self.graph);
        tracker.track(b_node, self.graph);

        let old_ba = self.graph.vessel_berth(a);
        let old_bb = self.graph.vessel_berth(b);

        self.undo.push_swap_vessels(a, b);
        self.graph.swap_vessels(a, b);

        tracker.commit(self.graph, self.diff);

        let new_ba = self.graph.vessel_berth(a);
        let new_bb = self.graph.vessel_berth(b);
        if old_ba != new_ba {
            self.diff.push_reallocation(a, old_ba, new_ba);
        }
        if old_bb != new_bb {
            self.diff.push_reallocation(b, old_bb, new_bb);
        }
    }

    /// Swaps two contiguous segments of vessels.
    ///
    /// ```text
    /// Before:
    ///   Berth X: ... <-> Prev_A <-> [ A_First ... A_Last ] <-> Next_A <-> ...
    ///   Berth Y: ... <-> Prev_B <-> [ B_First ... B_Last ] <-> Next_B <-> ...
    ///
    /// After:
    ///   Berth X: ... <-> Prev_A <-> [ B_First ... B_Last ] <-> Next_A <-> ...
    ///   Berth Y: ... <-> Prev_B <-> [ A_First ... A_Last ] <-> Next_B <-> ...
    /// ```
    ///
    /// Records the swap in the undo log, emits edge diffs, and emits
    /// reallocation entries for every vessel in each segment if the
    /// segments belonged to different berths.
    ///
    /// No-op if `a_first == b_first`.
    ///
    /// # Panics
    ///
    /// Panics if any vessel is out of bounds.
    #[inline]
    pub fn swap_segments(
        &mut self,
        a_first: VesselIndex,
        a_last: VesselIndex,
        b_first: VesselIndex,
        b_last: VesselIndex,
    ) {
        assert!(a_first < self.graph.num_vessels());
        assert!(a_last < self.graph.num_vessels());
        assert!(b_first < self.graph.num_vessels());

        if a_first == b_first {
            return;
        }

        let a_f_node = self.graph.vessel_node(a_first);
        let a_l_node = self.graph.vessel_node(a_last);
        let b_f_node = self.graph.vessel_node(b_first);
        let b_l_node = self.graph.vessel_node(b_last);

        let mut tracker = EdgeDeltaTracker::new();
        tracker.track(
            unsafe { self.graph.prev_node_unchecked(a_f_node) },
            self.graph,
        );
        tracker.track(a_l_node, self.graph);
        tracker.track(
            unsafe { self.graph.prev_node_unchecked(b_f_node) },
            self.graph,
        );
        tracker.track(b_l_node, self.graph);

        let old_ba = self.graph.vessel_berth(a_first);
        let old_bb = self.graph.vessel_berth(b_first);

        self.undo
            .push_swap_segments(a_first, a_last, b_first, b_last);
        self.graph.swap_segments(a_first, a_last, b_first, b_last);

        tracker.commit(self.graph, self.diff);

        self.record_segment_reallocation(a_first, a_last, old_ba, old_bb);
        self.record_segment_reallocation(b_first, b_last, old_bb, old_ba);
    }

    /// Reverses the internal ordering of a contiguous segment of vessels.
    ///
    /// ```text
    /// Before:
    ///   ... <-> Prev <-> [ A <-> B <-> C <-> D ] <-> Next <-> ...
    ///                      ^                 ^
    ///                    first              last
    ///
    /// After:
    ///   ... <-> Prev <-> [ D <-> C <-> B <-> A ] <-> Next <-> ...
    /// ```
    ///
    /// Records the reversal in the undo log and emits edge diffs for
    /// every internal link that was reversed, plus the two boundary
    /// links. No reallocation entries are emitted because reversal
    /// does not change berth assignments.
    ///
    /// No-op if `first == last`.
    ///
    /// # Panics
    ///
    /// Panics if either vessel is out of bounds.
    #[inline]
    pub fn reverse_segment(&mut self, first: VesselIndex, last: VesselIndex) {
        assert!(first < self.graph.num_vessels());
        assert!(last < self.graph.num_vessels());

        if first == last {
            return;
        }

        let first_node = self.graph.vessel_node(first);
        let last_node = self.graph.vessel_node(last);

        let prev_first = unsafe { self.graph.prev_node_unchecked(first_node) };
        let next_last = unsafe { self.graph.next_node_unchecked(last_node) };

        self.diff.push_link_broken(prev_first, first_node);
        self.diff.push_link_broken(last_node, next_last);

        let mut curr = first_node;
        while curr != last_node {
            let nxt = unsafe { self.graph.next_node_unchecked(curr) };
            self.diff.push_link_broken(curr, nxt);
            self.diff.push_link_created(nxt, curr);
            curr = nxt;
        }

        self.diff.push_link_created(prev_first, last_node);
        self.diff.push_link_created(first_node, next_last);

        self.undo.push_reverse_segment(first, last);
        self.graph.reverse_segment(first, last);
    }

    /// Relocates a single vessel to immediately follow another vessel.
    ///
    /// ```text
    /// Before:
    ///   Source: ... <-> Prev <-> Vessel <-> Next <-> ...
    ///   Target: ... <-> Anchor <-> Anchor_Next <-> ...
    ///
    /// After:
    ///   Source: ... <-> Prev <------------> Next <-> ...
    ///   Target: ... <-> Anchor <-> Vessel <-> Anchor_Next <-> ...
    /// ```
    ///
    /// Records the original position in the undo log, emits edge diffs,
    /// and emits a reallocation entry if the vessel changed berths.
    ///
    /// No-op if `vessel == anchor` or `vessel` already follows `anchor`.
    ///
    /// # Panics
    ///
    /// Panics if either vessel is out of bounds.
    #[inline]
    pub fn relocate_after(&mut self, vessel: VesselIndex, anchor: VesselIndex) {
        assert!(vessel < self.graph.num_vessels());
        assert!(anchor < self.graph.num_vessels());

        let v_node = self.graph.vessel_node(vessel);
        let a_node = self.graph.vessel_node(anchor);
        let prev = unsafe { self.graph.prev_node_unchecked(v_node) };

        if v_node == a_node || prev == a_node {
            return;
        }

        let mut tracker = EdgeDeltaTracker::new();
        tracker.track(prev, self.graph);
        tracker.track(v_node, self.graph);
        tracker.track(a_node, self.graph);

        let old_b = self.graph.vessel_berth(vessel);

        self.record_relocate(vessel);
        self.graph.relocate_after(vessel, anchor);

        tracker.commit(self.graph, self.diff);

        let new_b = self.graph.vessel_berth(vessel);
        if old_b != new_b {
            self.diff.push_reallocation(vessel, old_b, new_b);
        }
    }

    /// Relocates a single vessel to immediately precede another vessel.
    ///
    /// ```text
    /// Before:
    ///   Source: ... <-> Prev <-> Vessel <-> Next <-> ...
    ///   Target: ... <-> Ref_Prev <-> Reference <-> ...
    ///
    /// After:
    ///   Source: ... <-> Prev <------------> Next <-> ...
    ///   Target: ... <-> Ref_Prev <-> Vessel <-> Reference <-> ...
    /// ```
    ///
    /// Records the original position in the undo log, emits edge diffs,
    /// and emits a reallocation entry if the vessel changed berths.
    ///
    /// No-op if `vessel == reference` or `vessel` already precedes `reference`.
    ///
    /// # Panics
    ///
    /// Panics if either vessel is out of bounds.
    #[inline]
    pub fn relocate_before(&mut self, vessel: VesselIndex, reference: VesselIndex) {
        assert!(vessel < self.graph.num_vessels());
        assert!(reference < self.graph.num_vessels());

        let v_node = self.graph.vessel_node(vessel);
        let ref_node = self.graph.vessel_node(reference);
        let ref_prev = unsafe { self.graph.prev_node_unchecked(ref_node) };
        let v_prev = unsafe { self.graph.prev_node_unchecked(v_node) };

        if v_node == ref_node || v_prev == ref_prev {
            return;
        }

        let mut tracker = EdgeDeltaTracker::new();
        tracker.track(v_prev, self.graph);
        tracker.track(v_node, self.graph);
        tracker.track(ref_prev, self.graph);

        let old_b = self.graph.vessel_berth(vessel);

        self.record_relocate(vessel);
        self.graph.relocate_before(vessel, reference);

        tracker.commit(self.graph, self.diff);

        let new_b = self.graph.vessel_berth(vessel);
        if old_b != new_b {
            self.diff.push_reallocation(vessel, old_b, new_b);
        }
    }

    /// Relocates a single vessel to the head (first position) of a berth.
    ///
    /// ```text
    /// Before:
    ///   Source:  ... <-> Prev <-> Vessel <-> Next <-> ...
    ///   Target:  Sentinel <-> Old_Head <-> ...
    ///
    /// After:
    ///   Source:  ... <-> Prev <------------> Next <-> ...
    ///   Target:  Sentinel <-> Vessel <-> Old_Head <-> ...
    /// ```
    ///
    /// Records the original position in the undo log, emits edge diffs,
    /// and emits a reallocation entry if the vessel changed berths.
    ///
    /// # Panics
    ///
    /// Panics if vessel or berth is out of bounds.
    #[inline]
    pub fn relocate_to_head(&mut self, vessel: VesselIndex, berth: BerthIndex) {
        assert!(vessel < self.graph.num_vessels());
        assert!(berth < self.graph.num_berths());

        let v_node = self.graph.vessel_node(vessel);
        let head_boundary = self.graph.berth_head_boundary_node(berth);
        let prev = unsafe { self.graph.prev_node_unchecked(v_node) };

        let mut tracker = EdgeDeltaTracker::new();
        tracker.track(prev, self.graph);
        tracker.track(v_node, self.graph);
        tracker.track(head_boundary, self.graph);

        let old_b = self.graph.vessel_berth(vessel);

        self.record_relocate(vessel);
        self.graph.relocate_to_head(vessel, berth);

        tracker.commit(self.graph, self.diff);

        if old_b != berth {
            self.diff.push_reallocation(vessel, old_b, berth);
        }
    }

    /// Relocates a single vessel to the tail (last position) of a berth.
    ///
    /// ```text
    /// Before:
    ///   Source:  ... <-> Prev <-> Vessel <-> Next <-> ...
    ///   Target:  ... <-> Old_Tail <-> Sentinel
    ///
    /// After:
    ///   Source:  ... <-> Prev <------------> Next <-> ...
    ///   Target:  ... <-> Old_Tail <-> Vessel <-> Sentinel
    /// ```
    ///
    /// Records the original position in the undo log, emits edge diffs,
    /// and emits a reallocation entry if the vessel changed berths.
    ///
    /// # Panics
    ///
    /// Panics if vessel or berth is out of bounds.
    #[inline]
    pub fn relocate_to_tail(&mut self, vessel: VesselIndex, berth: BerthIndex) {
        assert!(vessel < self.graph.num_vessels());
        assert!(berth < self.graph.num_berths());

        let v_node = self.graph.vessel_node(vessel);
        let tail_boundary = self.graph.berth_tail_boundary_node(berth);
        let prev = unsafe { self.graph.prev_node_unchecked(v_node) };
        let old_tail = unsafe { self.graph.prev_node_unchecked(tail_boundary) };

        let mut tracker = EdgeDeltaTracker::new();
        tracker.track(prev, self.graph);
        tracker.track(v_node, self.graph);
        tracker.track(old_tail, self.graph);

        let old_b = self.graph.vessel_berth(vessel);

        self.record_relocate(vessel);
        self.graph.relocate_to_tail(vessel, berth);

        tracker.commit(self.graph, self.diff);

        if old_b != berth {
            self.diff.push_reallocation(vessel, old_b, berth);
        }
    }

    /// Relocates a contiguous segment of vessels to immediately follow
    /// another vessel.
    ///
    /// ```text
    /// Before:
    ///   Source: ... <-> Prev <-> [ First ... Last ] <-> Next <-> ...
    ///   Target: ... <-> Anchor <-> Anchor_Next <-> ...
    ///
    /// After:
    ///   Source: ... <-> Prev <---------------------> Next <-> ...
    ///   Target: ... <-> Anchor <-> [ First ... Last ] <-> Anchor_Next <-> ...
    /// ```
    ///
    /// Records the original position in the undo log, emits edge diffs,
    /// and emits reallocation entries for every vessel in the segment if
    /// it changed berths.
    ///
    /// No-op if the segment already follows `anchor`.
    ///
    /// # Panics
    ///
    /// Panics if any vessel is out of bounds.
    #[inline]
    pub fn relocate_segment_after(
        &mut self,
        first: VesselIndex,
        last: VesselIndex,
        anchor: VesselIndex,
    ) {
        assert!(first < self.graph.num_vessels());
        assert!(last < self.graph.num_vessels());
        assert!(anchor < self.graph.num_vessels());

        let f_node = self.graph.vessel_node(first);
        let l_node = self.graph.vessel_node(last);
        let a_node = self.graph.vessel_node(anchor);
        let prev = unsafe { self.graph.prev_node_unchecked(f_node) };

        if prev == a_node {
            return;
        }

        let mut tracker = EdgeDeltaTracker::new();
        tracker.track(prev, self.graph);
        tracker.track(l_node, self.graph);
        tracker.track(a_node, self.graph);

        let old_b = self.graph.vessel_berth(first);

        self.record_relocate_segment(first, last);
        self.graph.relocate_segment_after(first, last, anchor);

        tracker.commit(self.graph, self.diff);

        let new_b = self.graph.vessel_berth(first);
        self.record_segment_reallocation(first, last, old_b, new_b);
    }

    /// Relocates a contiguous segment of vessels to immediately precede
    /// another vessel.
    ///
    /// ```text
    /// Before:
    ///   Source: ... <-> Prev <-> [ First ... Last ] <-> Next <-> ...
    ///   Target: ... <-> Ref_Prev <-> Reference <-> ...
    ///
    /// After:
    ///   Source: ... <-> Prev <---------------------> Next <-> ...
    ///   Target: ... <-> Ref_Prev <-> [ First ... Last ] <-> Reference <-> ...
    /// ```
    ///
    /// Records the original position in the undo log, emits edge diffs,
    /// and emits reallocation entries for every vessel in the segment if
    /// it changed berths.
    ///
    /// No-op if the segment already precedes `reference`.
    ///
    /// # Panics
    ///
    /// Panics if any vessel is out of bounds.
    #[inline]
    pub fn relocate_segment_before(
        &mut self,
        first: VesselIndex,
        last: VesselIndex,
        reference: VesselIndex,
    ) {
        assert!(first < self.graph.num_vessels());
        assert!(last < self.graph.num_vessels());
        assert!(reference < self.graph.num_vessels());

        let f_node = self.graph.vessel_node(first);
        let l_node = self.graph.vessel_node(last);
        let ref_node = self.graph.vessel_node(reference);
        let ref_prev = unsafe { self.graph.prev_node_unchecked(ref_node) };
        let prev = unsafe { self.graph.prev_node_unchecked(f_node) };

        if prev == ref_prev {
            return;
        }

        let mut tracker = EdgeDeltaTracker::new();
        tracker.track(prev, self.graph);
        tracker.track(l_node, self.graph);
        tracker.track(ref_prev, self.graph);

        let old_b = self.graph.vessel_berth(first);

        self.record_relocate_segment(first, last);
        self.graph.relocate_segment_before(first, last, reference);

        tracker.commit(self.graph, self.diff);

        let new_b = self.graph.vessel_berth(first);
        self.record_segment_reallocation(first, last, old_b, new_b);
    }

    /// Relocates a contiguous segment of vessels to the head (first
    /// position) of a berth.
    ///
    /// ```text
    /// Before:
    ///   Source:  ... <-> Prev <-> [ First ... Last ] <-> Next <-> ...
    ///   Target:  Sentinel <-> Old_Head <-> ...
    ///
    /// After:
    ///   Source:  ... <-> Prev <---------------------> Next <-> ...
    ///   Target:  Sentinel <-> [ First ... Last ] <-> Old_Head <-> ...
    /// ```
    ///
    /// Records the original position in the undo log, emits edge diffs,
    /// and emits reallocation entries for every vessel in the segment if
    /// it changed berths.
    ///
    /// # Panics
    ///
    /// Panics if any vessel or berth is out of bounds.
    #[inline]
    pub fn relocate_segment_to_head(
        &mut self,
        first: VesselIndex,
        last: VesselIndex,
        berth: BerthIndex,
    ) {
        assert!(first < self.graph.num_vessels());
        assert!(last < self.graph.num_vessels());
        assert!(berth < self.graph.num_berths());

        let f_node = self.graph.vessel_node(first);
        let l_node = self.graph.vessel_node(last);
        let head_boundary = self.graph.berth_head_boundary_node(berth);
        let prev = unsafe { self.graph.prev_node_unchecked(f_node) };

        let mut tracker = EdgeDeltaTracker::new();
        tracker.track(prev, self.graph);
        tracker.track(l_node, self.graph);
        tracker.track(head_boundary, self.graph);

        let old_b = self.graph.vessel_berth(first);

        self.record_relocate_segment(first, last);
        self.graph.relocate_segment_to_head(first, last, berth);

        tracker.commit(self.graph, self.diff);

        self.record_segment_reallocation(first, last, old_b, berth);
    }

    /// Relocates a contiguous segment of vessels to the tail (last
    /// position) of a berth.
    ///
    /// ```text
    /// Before:
    ///   Source:  ... <-> Prev <-> [ First ... Last ] <-> Next <-> ...
    ///   Target:  ... <-> Old_Tail <-> Sentinel
    ///
    /// After:
    ///   Source:  ... <-> Prev <---------------------> Next <-> ...
    ///   Target:  ... <-> Old_Tail <-> [ First ... Last ] <-> Sentinel
    /// ```
    ///
    /// Records the original position in the undo log, emits edge diffs,
    /// and emits reallocation entries for every vessel in the segment if
    /// it changed berths.
    ///
    /// # Panics
    ///
    /// Panics if any vessel or berth is out of bounds.
    #[inline]
    pub fn relocate_segment_to_tail(
        &mut self,
        first: VesselIndex,
        last: VesselIndex,
        berth: BerthIndex,
    ) {
        assert!(first < self.graph.num_vessels());
        assert!(last < self.graph.num_vessels());
        assert!(berth < self.graph.num_berths());

        let f_node = self.graph.vessel_node(first);
        let l_node = self.graph.vessel_node(last);
        let tail_boundary = self.graph.berth_tail_boundary_node(berth);
        let prev = unsafe { self.graph.prev_node_unchecked(f_node) };
        let old_tail = unsafe { self.graph.prev_node_unchecked(tail_boundary) };

        let mut tracker = EdgeDeltaTracker::new();
        tracker.track(prev, self.graph);
        tracker.track(l_node, self.graph);
        tracker.track(old_tail, self.graph);

        let old_b = self.graph.vessel_berth(first);

        self.record_relocate_segment(first, last);
        self.graph.relocate_segment_to_tail(first, last, berth);

        tracker.commit(self.graph, self.diff);

        self.record_segment_reallocation(first, last, old_b, berth);
    }

    /// Swaps the positions of two vessels in the schedule graph.
    ///
    /// ```text
    /// Before:
    ///   Berth X: ... <-> Prev_A <-> A <-> Next_A <-> ...
    ///   Berth Y: ... <-> Prev_B <-> B <-> Next_B <-> ...
    ///
    /// After:
    ///   Berth X: ... <-> Prev_A <-> B <-> Next_A <-> ...
    ///   Berth Y: ... <-> Prev_B <-> A <-> Next_B <-> ...
    /// ```
    ///
    /// Unchecked version of `swap_vessels`. Uses `debug_assert!` only.
    ///
    /// # Safety
    ///
    /// Both vessels must be in bounds.
    #[inline]
    pub unsafe fn swap_vessels_unchecked(&mut self, a: VesselIndex, b: VesselIndex) {
        debug_assert!(a < self.graph.num_vessels());
        debug_assert!(b < self.graph.num_vessels());

        if a == b {
            return;
        }

        let a_node = self.graph.vessel_node(a);
        let b_node = self.graph.vessel_node(b);

        let mut tracker = EdgeDeltaTracker::new();
        tracker.track(
            unsafe { self.graph.prev_node_unchecked(a_node) },
            self.graph,
        );
        tracker.track(
            unsafe { self.graph.prev_node_unchecked(b_node) },
            self.graph,
        );
        tracker.track(a_node, self.graph);
        tracker.track(b_node, self.graph);

        let old_ba = unsafe { self.graph.vessel_berth_unchecked(a) };
        let old_bb = unsafe { self.graph.vessel_berth_unchecked(b) };

        self.undo.push_swap_vessels(a, b);
        unsafe { self.graph.swap_vessels_unchecked(a, b) };

        tracker.commit(self.graph, self.diff);

        let new_ba = unsafe { self.graph.vessel_berth_unchecked(a) };
        let new_bb = unsafe { self.graph.vessel_berth_unchecked(b) };
        if old_ba != new_ba {
            self.diff.push_reallocation(a, old_ba, new_ba);
        }
        if old_bb != new_bb {
            self.diff.push_reallocation(b, old_bb, new_bb);
        }
    }

    /// Swaps two contiguous segments of vessels.
    ///
    /// ```text
    /// Before:
    ///   Berth X: ... <-> Prev_A <-> [ A_First ... A_Last ] <-> Next_A <-> ...
    ///   Berth Y: ... <-> Prev_B <-> [ B_First ... B_Last ] <-> Next_B <-> ...
    ///
    /// After:
    ///   Berth X: ... <-> Prev_A <-> [ B_First ... B_Last ] <-> Next_A <-> ...
    ///   Berth Y: ... <-> Prev_B <-> [ A_First ... A_Last ] <-> Next_B <-> ...
    /// ```
    ///
    /// Unchecked version of `swap_segments`. Uses `debug_assert!` only.
    ///
    /// # Safety
    ///
    /// All vessels must be in bounds. Segments must be valid and non-overlapping.
    #[inline]
    pub unsafe fn swap_segments_unchecked(
        &mut self,
        a_first: VesselIndex,
        a_last: VesselIndex,
        b_first: VesselIndex,
        b_last: VesselIndex,
    ) {
        debug_assert!(a_first < self.graph.num_vessels());
        debug_assert!(a_last < self.graph.num_vessels());
        debug_assert!(b_first < self.graph.num_vessels());
        debug_assert!(b_last < self.graph.num_vessels());

        if a_first == b_first {
            return;
        }

        let a_f_node = self.graph.vessel_node(a_first);
        let a_l_node = self.graph.vessel_node(a_last);
        let b_f_node = self.graph.vessel_node(b_first);
        let b_l_node = self.graph.vessel_node(b_last);

        let mut tracker = EdgeDeltaTracker::new();
        tracker.track(
            unsafe { self.graph.prev_node_unchecked(a_f_node) },
            self.graph,
        );
        tracker.track(a_l_node, self.graph);
        tracker.track(
            unsafe { self.graph.prev_node_unchecked(b_f_node) },
            self.graph,
        );
        tracker.track(b_l_node, self.graph);

        let old_ba = unsafe { self.graph.vessel_berth_unchecked(a_first) };
        let old_bb = unsafe { self.graph.vessel_berth_unchecked(b_first) };

        self.undo
            .push_swap_segments(a_first, a_last, b_first, b_last);
        unsafe {
            self.graph
                .swap_segments_unchecked(a_first, a_last, b_first, b_last)
        };

        tracker.commit(self.graph, self.diff);

        unsafe { self.record_segment_reallocation_unchecked(a_first, a_last, old_ba, old_bb) };
        unsafe { self.record_segment_reallocation_unchecked(b_first, b_last, old_bb, old_ba) };
    }

    /// Reverses the internal ordering of a contiguous segment of vessels.
    ///
    /// ```text
    /// Before:
    ///   ... <-> Prev <-> [ A <-> B <-> C <-> D ] <-> Next <-> ...
    ///                      ^                 ^
    ///                    first              last
    ///
    /// After:
    ///   ... <-> Prev <-> [ D <-> C <-> B <-> A ] <-> Next <-> ...
    /// ```
    ///
    /// Unchecked version of `reverse_segment`. Uses `debug_assert!` only.
    ///
    /// # Safety
    ///
    /// Both vessels must be in bounds and form a valid contiguous segment.
    #[inline]
    pub unsafe fn reverse_segment_unchecked(&mut self, first: VesselIndex, last: VesselIndex) {
        debug_assert!(first < self.graph.num_vessels());
        debug_assert!(last < self.graph.num_vessels());

        if first == last {
            return;
        }

        let first_node = self.graph.vessel_node(first);
        let last_node = self.graph.vessel_node(last);

        let prev_first = unsafe { self.graph.prev_node_unchecked(first_node) };
        let next_last = unsafe { self.graph.next_node_unchecked(last_node) };

        self.diff.push_link_broken(prev_first, first_node);
        self.diff.push_link_broken(last_node, next_last);

        let mut curr = first_node;
        while curr != last_node {
            let nxt = unsafe { self.graph.next_node_unchecked(curr) };
            self.diff.push_link_broken(curr, nxt);
            self.diff.push_link_created(nxt, curr);
            curr = nxt;
        }

        self.diff.push_link_created(prev_first, last_node);
        self.diff.push_link_created(first_node, next_last);

        self.undo.push_reverse_segment(first, last);
        unsafe { self.graph.reverse_segment_unchecked(first, last) };
    }

    /// Relocates a single vessel to immediately follow another vessel.
    ///
    /// ```text
    /// Before:
    ///   Source: ... <-> Prev <-> Vessel <-> Next <-> ...
    ///   Target: ... <-> Anchor <-> Anchor_Next <-> ...
    ///
    /// After:
    ///   Source: ... <-> Prev <------------> Next <-> ...
    ///   Target: ... <-> Anchor <-> Vessel <-> Anchor_Next <-> ...
    /// ```
    ///
    /// Unchecked version of `relocate_after`. Uses `debug_assert!` only.
    ///
    /// # Safety
    ///
    /// Both vessels must be in bounds.
    #[inline]
    pub unsafe fn relocate_after_unchecked(&mut self, vessel: VesselIndex, anchor: VesselIndex) {
        debug_assert!(vessel < self.graph.num_vessels());
        debug_assert!(anchor < self.graph.num_vessels());

        let v_node = self.graph.vessel_node(vessel);
        let a_node = self.graph.vessel_node(anchor);
        let prev = unsafe { self.graph.prev_node_unchecked(v_node) };

        if v_node == a_node || prev == a_node {
            return;
        }

        let mut tracker = EdgeDeltaTracker::new();
        tracker.track(prev, self.graph);
        tracker.track(v_node, self.graph);
        tracker.track(a_node, self.graph);

        let old_b = unsafe { self.graph.vessel_berth_unchecked(vessel) };

        unsafe { self.record_relocate_unchecked(vessel) };
        unsafe { self.graph.relocate_after_unchecked(vessel, anchor) };

        tracker.commit(self.graph, self.diff);

        let new_b = unsafe { self.graph.vessel_berth_unchecked(vessel) };
        if old_b != new_b {
            self.diff.push_reallocation(vessel, old_b, new_b);
        }
    }

    /// Relocates a single vessel to immediately precede another vessel.
    ///
    /// ```text
    /// Before:
    ///   Source: ... <-> Prev <-> Vessel <-> Next <-> ...
    ///   Target: ... <-> Ref_Prev <-> Reference <-> ...
    ///
    /// After:
    ///   Source: ... <-> Prev <------------> Next <-> ...
    ///   Target: ... <-> Ref_Prev <-> Vessel <-> Reference <-> ...
    /// ```
    ///
    /// Unchecked version of `relocate_before`. Uses `debug_assert!` only.
    ///
    /// # Safety
    ///
    /// Both vessels must be in bounds.
    #[inline]
    pub unsafe fn relocate_before_unchecked(
        &mut self,
        vessel: VesselIndex,
        reference: VesselIndex,
    ) {
        debug_assert!(vessel < self.graph.num_vessels());
        debug_assert!(reference < self.graph.num_vessels());

        let v_node = self.graph.vessel_node(vessel);
        let ref_node = self.graph.vessel_node(reference);
        let ref_prev = unsafe { self.graph.prev_node_unchecked(ref_node) };
        let v_prev = unsafe { self.graph.prev_node_unchecked(v_node) };

        if v_node == ref_node || v_prev == ref_prev {
            return;
        }

        let mut tracker = EdgeDeltaTracker::new();
        tracker.track(v_prev, self.graph);
        tracker.track(v_node, self.graph);
        tracker.track(ref_prev, self.graph);

        let old_b = unsafe { self.graph.vessel_berth_unchecked(vessel) };

        unsafe { self.record_relocate_unchecked(vessel) };
        unsafe { self.graph.relocate_before_unchecked(vessel, reference) };

        tracker.commit(self.graph, self.diff);

        let new_b = unsafe { self.graph.vessel_berth_unchecked(vessel) };
        if old_b != new_b {
            self.diff.push_reallocation(vessel, old_b, new_b);
        }
    }

    /// Relocates a single vessel to the head (first position) of a berth.
    ///
    /// ```text
    /// Before:
    ///   Source:  ... <-> Prev <-> Vessel <-> Next <-> ...
    ///   Target:  Sentinel <-> Old_Head <-> ...
    ///
    /// After:
    ///   Source:  ... <-> Prev <------------> Next <-> ...
    ///   Target:  Sentinel <-> Vessel <-> Old_Head <-> ...
    /// ```
    ///
    /// Unchecked version of `relocate_to_head`. Uses `debug_assert!` only.
    ///
    /// # Safety
    ///
    /// Vessel and berth must be in bounds.
    #[inline]
    pub unsafe fn relocate_to_head_unchecked(&mut self, vessel: VesselIndex, berth: BerthIndex) {
        debug_assert!(vessel < self.graph.num_vessels());
        debug_assert!(berth < self.graph.num_berths());

        let v_node = self.graph.vessel_node(vessel);
        let head_boundary = self.graph.berth_head_boundary_node(berth);
        let prev = unsafe { self.graph.prev_node_unchecked(v_node) };

        let mut tracker = EdgeDeltaTracker::new();
        tracker.track(prev, self.graph);
        tracker.track(v_node, self.graph);
        tracker.track(head_boundary, self.graph);

        let old_b = unsafe { self.graph.vessel_berth_unchecked(vessel) };

        unsafe { self.record_relocate_unchecked(vessel) };
        unsafe { self.graph.relocate_to_head_unchecked(vessel, berth) };

        tracker.commit(self.graph, self.diff);

        if old_b != berth {
            self.diff.push_reallocation(vessel, old_b, berth);
        }
    }

    /// Relocates a single vessel to the tail (last position) of a berth.
    ///
    /// ```text
    /// Before:
    ///   Source:  ... <-> Prev <-> Vessel <-> Next <-> ...
    ///   Target:  ... <-> Old_Tail <-> Sentinel
    ///
    /// After:
    ///   Source:  ... <-> Prev <------------> Next <-> ...
    ///   Target:  ... <-> Old_Tail <-> Vessel <-> Sentinel
    /// ```
    ///
    /// Unchecked version of `relocate_to_tail`. Uses `debug_assert!` only.
    ///
    /// # Safety
    ///
    /// Vessel and berth must be in bounds.
    #[inline]
    pub unsafe fn relocate_to_tail_unchecked(&mut self, vessel: VesselIndex, berth: BerthIndex) {
        debug_assert!(vessel < self.graph.num_vessels());
        debug_assert!(berth < self.graph.num_berths());

        let v_node = self.graph.vessel_node(vessel);
        let tail_boundary = self.graph.berth_tail_boundary_node(berth);
        let prev = unsafe { self.graph.prev_node_unchecked(v_node) };
        let old_tail = unsafe { self.graph.prev_node_unchecked(tail_boundary) };

        let mut tracker = EdgeDeltaTracker::new();
        tracker.track(prev, self.graph);
        tracker.track(v_node, self.graph);
        tracker.track(old_tail, self.graph);

        let old_b = unsafe { self.graph.vessel_berth_unchecked(vessel) };

        unsafe { self.record_relocate_unchecked(vessel) };
        unsafe { self.graph.relocate_to_tail_unchecked(vessel, berth) };

        tracker.commit(self.graph, self.diff);

        if old_b != berth {
            self.diff.push_reallocation(vessel, old_b, berth);
        }
    }

    /// Relocates a contiguous segment of vessels to immediately follow
    /// another vessel.
    ///
    /// ```text
    /// Before:
    ///   Source: ... <-> Prev <-> [ First ... Last ] <-> Next <-> ...
    ///   Target: ... <-> Anchor <-> Anchor_Next <-> ...
    ///
    /// After:
    ///   Source: ... <-> Prev <---------------------> Next <-> ...
    ///   Target: ... <-> Anchor <-> [ First ... Last ] <-> Anchor_Next <-> ...
    /// ```
    ///
    /// Unchecked version of `relocate_segment_after`. Uses `debug_assert!` only.
    ///
    /// # Safety
    ///
    /// All vessels must be in bounds.
    #[inline]
    pub unsafe fn relocate_segment_after_unchecked(
        &mut self,
        first: VesselIndex,
        last: VesselIndex,
        anchor: VesselIndex,
    ) {
        debug_assert!(first < self.graph.num_vessels());
        debug_assert!(last < self.graph.num_vessels());
        debug_assert!(anchor < self.graph.num_vessels());

        let f_node = self.graph.vessel_node(first);
        let l_node = self.graph.vessel_node(last);
        let a_node = self.graph.vessel_node(anchor);
        let prev = unsafe { self.graph.prev_node_unchecked(f_node) };

        if prev == a_node {
            return;
        }

        let mut tracker = EdgeDeltaTracker::new();
        tracker.track(prev, self.graph);
        tracker.track(l_node, self.graph);
        tracker.track(a_node, self.graph);

        let old_b = unsafe { self.graph.vessel_berth_unchecked(first) };

        unsafe { self.record_relocate_segment_unchecked(first, last) };
        unsafe {
            self.graph
                .relocate_segment_after_unchecked(first, last, anchor)
        };

        tracker.commit(self.graph, self.diff);

        let new_b = unsafe { self.graph.vessel_berth_unchecked(first) };
        unsafe { self.record_segment_reallocation_unchecked(first, last, old_b, new_b) };
    }

    /// Relocates a contiguous segment of vessels to immediately precede
    /// another vessel.
    ///
    /// ```text
    /// Before:
    ///   Source: ... <-> Prev <-> [ First ... Last ] <-> Next <-> ...
    ///   Target: ... <-> Ref_Prev <-> Reference <-> ...
    ///
    /// After:
    ///   Source: ... <-> Prev <---------------------> Next <-> ...
    ///   Target: ... <-> Ref_Prev <-> [ First ... Last ] <-> Reference <-> ...
    /// ```
    ///
    /// Unchecked version of `relocate_segment_before`. Uses `debug_assert!` only.
    ///
    /// # Safety
    ///
    /// All vessels must be in bounds.
    #[inline]
    pub unsafe fn relocate_segment_before_unchecked(
        &mut self,
        first: VesselIndex,
        last: VesselIndex,
        reference: VesselIndex,
    ) {
        debug_assert!(first < self.graph.num_vessels());
        debug_assert!(last < self.graph.num_vessels());
        debug_assert!(reference < self.graph.num_vessels());

        let f_node = self.graph.vessel_node(first);
        let l_node = self.graph.vessel_node(last);
        let ref_node = self.graph.vessel_node(reference);
        let ref_prev = unsafe { self.graph.prev_node_unchecked(ref_node) };
        let prev = unsafe { self.graph.prev_node_unchecked(f_node) };

        if prev == ref_prev {
            return;
        }

        let mut tracker = EdgeDeltaTracker::new();
        tracker.track(prev, self.graph);
        tracker.track(l_node, self.graph);
        tracker.track(ref_prev, self.graph);

        let old_b = unsafe { self.graph.vessel_berth_unchecked(first) };

        unsafe { self.record_relocate_segment_unchecked(first, last) };
        unsafe {
            self.graph
                .relocate_segment_before_unchecked(first, last, reference)
        };

        tracker.commit(self.graph, self.diff);

        let new_b = unsafe { self.graph.vessel_berth_unchecked(first) };
        unsafe { self.record_segment_reallocation_unchecked(first, last, old_b, new_b) };
    }

    /// Relocates a contiguous segment of vessels to the head (first
    /// position) of a berth.
    ///
    /// ```text
    /// Before:
    ///   Source:  ... <-> Prev <-> [ First ... Last ] <-> Next <-> ...
    ///   Target:  Sentinel <-> Old_Head <-> ...
    ///
    /// After:
    ///   Source:  ... <-> Prev <---------------------> Next <-> ...
    ///   Target:  Sentinel <-> [ First ... Last ] <-> Old_Head <-> ...
    /// ```
    ///
    /// Unchecked version of `relocate_segment_to_head`. Uses `debug_assert!` only.
    ///
    /// # Safety
    ///
    /// All vessels and berth must be in bounds.
    #[inline]
    pub unsafe fn relocate_segment_to_head_unchecked(
        &mut self,
        first: VesselIndex,
        last: VesselIndex,
        berth: BerthIndex,
    ) {
        debug_assert!(first < self.graph.num_vessels());
        debug_assert!(last < self.graph.num_vessels());
        debug_assert!(berth < self.graph.num_berths());

        let f_node = self.graph.vessel_node(first);
        let l_node = self.graph.vessel_node(last);
        let head_boundary = self.graph.berth_head_boundary_node(berth);
        let prev = unsafe { self.graph.prev_node_unchecked(f_node) };

        let mut tracker = EdgeDeltaTracker::new();
        tracker.track(prev, self.graph);
        tracker.track(l_node, self.graph);
        tracker.track(head_boundary, self.graph);

        let old_b = unsafe { self.graph.vessel_berth_unchecked(first) };

        unsafe { self.record_relocate_segment_unchecked(first, last) };
        unsafe {
            self.graph
                .relocate_segment_to_head_unchecked(first, last, berth)
        };

        tracker.commit(self.graph, self.diff);

        unsafe { self.record_segment_reallocation_unchecked(first, last, old_b, berth) };
    }

    /// Relocates a contiguous segment of vessels to the tail (last
    /// position) of a berth.
    ///
    /// ```text
    /// Before:
    ///   Source:  ... <-> Prev <-> [ First ... Last ] <-> Next <-> ...
    ///   Target:  ... <-> Old_Tail <-> Sentinel
    ///
    /// After:
    ///   Source:  ... <-> Prev <---------------------> Next <-> ...
    ///   Target:  ... <-> Old_Tail <-> [ First ... Last ] <-> Sentinel
    /// ```
    ///
    /// Unchecked version of `relocate_segment_to_tail`. Uses `debug_assert!` only.
    ///
    /// # Safety
    ///
    /// All vessels and berth must be in bounds.
    #[inline]
    pub unsafe fn relocate_segment_to_tail_unchecked(
        &mut self,
        first: VesselIndex,
        last: VesselIndex,
        berth: BerthIndex,
    ) {
        debug_assert!(first < self.graph.num_vessels());
        debug_assert!(last < self.graph.num_vessels());
        debug_assert!(berth < self.graph.num_berths());

        let f_node = self.graph.vessel_node(first);
        let l_node = self.graph.vessel_node(last);
        let tail_boundary = self.graph.berth_tail_boundary_node(berth);
        let prev = unsafe { self.graph.prev_node_unchecked(f_node) };
        let old_tail = unsafe { self.graph.prev_node_unchecked(tail_boundary) };

        let mut tracker = EdgeDeltaTracker::new();
        tracker.track(prev, self.graph);
        tracker.track(l_node, self.graph);
        tracker.track(old_tail, self.graph);

        let old_b = unsafe { self.graph.vessel_berth_unchecked(first) };

        unsafe { self.record_relocate_segment_unchecked(first, last) };
        unsafe {
            self.graph
                .relocate_segment_to_tail_unchecked(first, last, berth)
        };

        tracker.commit(self.graph, self.diff);

        unsafe { self.record_segment_reallocation_unchecked(first, last, old_b, berth) };
    }
}

// ----------------------------------------------------------------
// Tests
// ----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn b(i: usize) -> BerthIndex {
        BerthIndex::new(i)
    }

    fn v(i: usize) -> VesselIndex {
        VesselIndex::new(i)
    }

    /// B0: V0 -> V1 -> V2
    /// B1: V3 -> V4
    /// B2: (empty)
    fn setup_graph() -> ScheduleGraph {
        let berths = [b(0), b(0), b(0), b(1), b(1)];
        let starts = [10, 20, 30, 10, 20];
        ScheduleGraph::from_slices(&berths, &starts, 3)
    }

    fn extract_reallocations(
        diff: &ScheduleGraphDiff,
    ) -> Vec<(VesselIndex, BerthIndex, BerthIndex)> {
        diff.reallocations().collect()
    }

    #[test]
    fn test_swap_vessels_same_berth() {
        let mut graph = setup_graph();
        let mut undo = ScheduleGraphUndoLog::new(10);
        let mut diff = ScheduleGraphDiff::new(graph.num_vessels());

        {
            let mut m = Mutator::new(&mut graph, &mut undo, &mut diff);
            m.swap_vessels(v(0), v(2));
        }

        let seq: Vec<_> = graph.vessel_sequence_iter(b(0)).collect();
        assert_eq!(seq, vec![v(2), v(1), v(0)]);
        assert!(extract_reallocations(&diff).is_empty());
    }

    #[test]
    fn test_relocate_to_head_cross_berth() {
        let mut graph = setup_graph();
        let mut undo = ScheduleGraphUndoLog::new(10);
        let mut diff = ScheduleGraphDiff::new(graph.num_vessels());

        {
            let mut m = Mutator::new(&mut graph, &mut undo, &mut diff);
            m.relocate_to_head(v(0), b(1));
        }

        assert_eq!(
            graph.vessel_sequence_iter(b(0)).collect::<Vec<_>>(),
            vec![v(1), v(2)]
        );
        assert_eq!(
            graph.vessel_sequence_iter(b(1)).collect::<Vec<_>>(),
            vec![v(0), v(3), v(4)]
        );

        let reallocs = extract_reallocations(&diff);
        assert_eq!(reallocs, vec![(v(0), b(0), b(1))]);
    }

    #[test]
    fn test_relocate_segment_after_cross_berth() {
        let mut graph = setup_graph();
        let mut undo = ScheduleGraphUndoLog::new(10);
        let mut diff = ScheduleGraphDiff::new(graph.num_vessels());

        {
            let mut m = Mutator::new(&mut graph, &mut undo, &mut diff);
            m.relocate_segment_after(v(0), v(1), v(3));
        }

        assert_eq!(
            graph.vessel_sequence_iter(b(0)).collect::<Vec<_>>(),
            vec![v(2)]
        );
        assert_eq!(
            graph.vessel_sequence_iter(b(1)).collect::<Vec<_>>(),
            vec![v(3), v(0), v(1), v(4)]
        );

        let reallocs = extract_reallocations(&diff);
        assert_eq!(reallocs, vec![(v(0), b(0), b(1)), (v(1), b(0), b(1))]);
    }

    #[test]
    fn test_reverse_segment() {
        let mut graph = setup_graph();
        let mut undo = ScheduleGraphUndoLog::new(10);
        let mut diff = ScheduleGraphDiff::new(graph.num_vessels());

        {
            let mut m = Mutator::new(&mut graph, &mut undo, &mut diff);
            m.reverse_segment(v(0), v(2));
        }

        assert_eq!(
            graph.vessel_sequence_iter(b(0)).collect::<Vec<_>>(),
            vec![v(2), v(1), v(0)]
        );
        assert!(extract_reallocations(&diff).is_empty());
    }

    #[test]
    fn test_swap_segments_cross_berth() {
        let mut graph = setup_graph();
        let mut undo = ScheduleGraphUndoLog::new(10);
        let mut diff = ScheduleGraphDiff::new(graph.num_vessels());

        {
            let mut m = Mutator::new(&mut graph, &mut undo, &mut diff);
            m.swap_segments(v(0), v(1), v(3), v(3));
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
        assert!(reallocs.contains(&(v(0), b(0), b(1))));
        assert!(reallocs.contains(&(v(1), b(0), b(1))));
        assert!(reallocs.contains(&(v(3), b(1), b(0))));
    }

    #[test]
    fn test_relocate_segment_to_tail_empty_berth() {
        let mut graph = setup_graph();
        let mut undo = ScheduleGraphUndoLog::new(10);
        let mut diff = ScheduleGraphDiff::new(graph.num_vessels());

        {
            let mut m = Mutator::new(&mut graph, &mut undo, &mut diff);
            m.relocate_segment_to_tail(v(3), v(4), b(2));
        }

        assert!(graph.vessel_sequence_iter(b(1)).next().is_none());
        assert_eq!(
            graph.vessel_sequence_iter(b(2)).collect::<Vec<_>>(),
            vec![v(3), v(4)]
        );

        let reallocs = extract_reallocations(&diff);
        assert_eq!(reallocs, vec![(v(3), b(1), b(2)), (v(4), b(1), b(2))]);
    }

    #[test]
    fn test_mutator_then_undo_rollback() {
        let mut graph = setup_graph();
        let original = graph.clone();
        let mut undo = ScheduleGraphUndoLog::new(20);
        let mut diff = ScheduleGraphDiff::new(graph.num_vessels());

        {
            let mut m = Mutator::new(&mut graph, &mut undo, &mut diff);
            m.relocate_to_head(v(0), b(2));
            m.reverse_segment(v(3), v(4));
            m.swap_vessels(v(1), v(3));
        }

        assert_ne!(graph, original);

        undo.apply_rollback(&mut graph);

        assert_eq!(graph, original);
        assert!(undo.is_empty());
    }
}
