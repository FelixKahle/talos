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
    sgraph::ScheduleGraph, sgraphdiff::ScheduleGraphDiff, sgraphundo::ScheduleGraphUndoLog,
};
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
    nodes: [usize; 4],
    old_nexts: [usize; 4],
    len: usize,
}

impl EdgeDeltaTracker {
    #[inline(always)]
    fn new() -> Self {
        Self {
            nodes: [0; 4],
            old_nexts: [0; 4],
            len: 0,
        }
    }

    /// Records a node's `next` pointer before mutation.
    /// Uses raw arena access for zero-overhead reads.
    #[inline(always)]
    fn track(&mut self, node: usize, graph: &ScheduleGraph) {
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
            *self.nodes.get_unchecked_mut(self.len) = node;
            *self.old_nexts.get_unchecked_mut(self.len) = graph.arena().next_unchecked(node);
        }
        self.len += 1;
    }

    /// Compares recorded state against current state and emits diffs.
    #[inline(always)]
    fn commit(self, graph: &ScheduleGraph, diff: &mut ScheduleGraphDiff) {
        for i in 0..self.len {
            let node = unsafe { *self.nodes.get_unchecked(i) };
            let old_nxt = unsafe { *self.old_nexts.get_unchecked(i) };
            let new_nxt = unsafe { graph.arena().next_unchecked(node) };

            if old_nxt != new_nxt {
                // Convert raw indices to VesselIndex. Indices >= num_vessels are
                // sentinels — the diff treats them as None internally.
                let from = VesselIndex::new(node);
                let old_to = VesselIndex::new(old_nxt);
                let new_to = VesselIndex::new(new_nxt);

                diff.push_link_broken(from, old_to);
                diff.push_link_created(from, new_to);
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

    // ----------------------------------------------------------------
    // Undo recording helpers
    // ----------------------------------------------------------------

    /// Records the original position of a vessel before relocation.
    /// Uses `vessel_predecessor_unchecked` to determine if the vessel was
    /// after another vessel or at the head of its berth.
    #[inline(always)]
    unsafe fn record_relocate(&mut self, vessel: VesselIndex) {
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

    /// Records the original position of a segment before relocation.
    #[inline(always)]
    unsafe fn record_relocate_segment(&mut self, first: VesselIndex, last: VesselIndex) {
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

    /// Records reallocation diffs for every vessel in a segment that changed berths.
    #[inline(always)]
    unsafe fn record_segment_reallocation(
        &mut self,
        first: VesselIndex,
        last: VesselIndex,
        old_berth: BerthIndex,
        new_berth: BerthIndex,
    ) {
        if old_berth == new_berth {
            return;
        }
        // Walk the segment via the arena's next pointers.
        let mut curr = first.get();
        let last_raw = last.get();
        loop {
            self.diff
                .push_reallocation(VesselIndex::new(curr), old_berth, new_berth);
            if curr == last_raw {
                break;
            }
            curr = unsafe { self.graph.arena().next_unchecked(curr) };
        }
    }

    // ----------------------------------------------------------------
    // Checked mutations
    // ----------------------------------------------------------------

    /// Swaps the positions of two vessels.
    ///
    /// # Panics
    ///
    /// Panics if either vessel is out of bounds.
    #[inline]
    pub fn swap_vessels(&mut self, a: VesselIndex, b: VesselIndex) {
        assert!(
            a.get() < self.graph.num_vessels() && b.get() < self.graph.num_vessels(),
            "Mutator::swap_vessels: a = {}, b = {}, num_vessels = {}",
            a.get(),
            b.get(),
            self.graph.num_vessels()
        );
        unsafe { self.swap_vessels_unchecked(a, b) }
    }

    /// Swaps two contiguous segments.
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
        assert!(
            a_first.get() < self.graph.num_vessels()
                && a_last.get() < self.graph.num_vessels()
                && b_first.get() < self.graph.num_vessels()
                && b_last.get() < self.graph.num_vessels(),
            "Mutator::swap_segments out of bounds"
        );
        unsafe { self.swap_segments_unchecked(a_first, a_last, b_first, b_last) }
    }

    /// Reverses a contiguous segment.
    ///
    /// # Panics
    ///
    /// Panics if either vessel is out of bounds.
    #[inline]
    pub fn reverse_segment(&mut self, first: VesselIndex, last: VesselIndex) {
        assert!(
            first.get() < self.graph.num_vessels() && last.get() < self.graph.num_vessels(),
            "Mutator::reverse_segment out of bounds"
        );
        unsafe { self.reverse_segment_unchecked(first, last) }
    }

    /// Relocates a vessel to immediately follow another vessel.
    ///
    /// # Panics
    ///
    /// Panics if either vessel is out of bounds.
    #[inline]
    pub fn relocate_after(&mut self, vessel: VesselIndex, anchor: VesselIndex) {
        assert!(
            vessel.get() < self.graph.num_vessels() && anchor.get() < self.graph.num_vessels(),
            "Mutator::relocate_after out of bounds"
        );
        unsafe { self.relocate_after_unchecked(vessel, anchor) }
    }

    /// Relocates a vessel to immediately precede another vessel.
    ///
    /// # Panics
    ///
    /// Panics if either vessel is out of bounds.
    #[inline]
    pub fn relocate_before(&mut self, vessel: VesselIndex, reference: VesselIndex) {
        assert!(
            vessel.get() < self.graph.num_vessels() && reference.get() < self.graph.num_vessels(),
            "Mutator::relocate_before out of bounds"
        );
        unsafe { self.relocate_before_unchecked(vessel, reference) }
    }

    /// Relocates a vessel to the head of a berth.
    ///
    /// # Panics
    ///
    /// Panics if vessel or berth is out of bounds.
    #[inline]
    pub fn relocate_to_head(&mut self, vessel: VesselIndex, berth: BerthIndex) {
        assert!(
            vessel.get() < self.graph.num_vessels() && berth.get() < self.graph.num_berths(),
            "Mutator::relocate_to_head out of bounds"
        );
        unsafe { self.relocate_to_head_unchecked(vessel, berth) }
    }

    /// Relocates a vessel to the tail of a berth.
    ///
    /// # Panics
    ///
    /// Panics if vessel or berth is out of bounds.
    #[inline]
    pub fn relocate_to_tail(&mut self, vessel: VesselIndex, berth: BerthIndex) {
        assert!(
            vessel.get() < self.graph.num_vessels() && berth.get() < self.graph.num_berths(),
            "Mutator::relocate_to_tail out of bounds"
        );
        unsafe { self.relocate_to_tail_unchecked(vessel, berth) }
    }

    /// Relocates a segment to immediately follow another vessel.
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
        assert!(
            first.get() < self.graph.num_vessels()
                && last.get() < self.graph.num_vessels()
                && anchor.get() < self.graph.num_vessels(),
            "Mutator::relocate_segment_after out of bounds"
        );
        unsafe { self.relocate_segment_after_unchecked(first, last, anchor) }
    }

    /// Relocates a segment to immediately precede another vessel.
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
        assert!(
            first.get() < self.graph.num_vessels()
                && last.get() < self.graph.num_vessels()
                && reference.get() < self.graph.num_vessels(),
            "Mutator::relocate_segment_before out of bounds"
        );
        unsafe { self.relocate_segment_before_unchecked(first, last, reference) }
    }

    /// Relocates a segment to the head of a berth.
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
        assert!(
            first.get() < self.graph.num_vessels()
                && last.get() < self.graph.num_vessels()
                && berth.get() < self.graph.num_berths(),
            "Mutator::relocate_segment_to_head out of bounds"
        );
        unsafe { self.relocate_segment_to_head_unchecked(first, last, berth) }
    }

    /// Relocates a segment to the tail of a berth.
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
        assert!(
            first.get() < self.graph.num_vessels()
                && last.get() < self.graph.num_vessels()
                && berth.get() < self.graph.num_berths(),
            "Mutator::relocate_segment_to_tail out of bounds"
        );
        unsafe { self.relocate_segment_to_tail_unchecked(first, last, berth) }
    }

    // ----------------------------------------------------------------
    // Unchecked mutations
    // ----------------------------------------------------------------

    /// # Safety
    ///
    /// Both vessels must be in bounds.
    #[inline]
    pub unsafe fn swap_vessels_unchecked(&mut self, a: VesselIndex, b: VesselIndex) {
        if a == b {
            return;
        }

        let a_raw = a.get();
        let b_raw = b.get();

        let mut tracker = EdgeDeltaTracker::new();
        tracker.track(
            unsafe { self.graph.arena().prev_unchecked(a_raw) },
            self.graph,
        );
        tracker.track(
            unsafe { self.graph.arena().prev_unchecked(b_raw) },
            self.graph,
        );
        tracker.track(a_raw, self.graph);
        tracker.track(b_raw, self.graph);

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
        if a_first == b_first {
            return;
        }

        let mut tracker = EdgeDeltaTracker::new();
        tracker.track(
            unsafe { self.graph.arena().prev_unchecked(a_first.get()) },
            self.graph,
        );
        tracker.track(a_last.get(), self.graph);
        tracker.track(
            unsafe { self.graph.arena().prev_unchecked(b_first.get()) },
            self.graph,
        );
        tracker.track(b_last.get(), self.graph);

        let old_ba = unsafe { self.graph.vessel_berth_unchecked(a_first) };
        let old_bb = unsafe { self.graph.vessel_berth_unchecked(b_first) };

        self.undo
            .push_swap_segments(a_first, a_last, b_first, b_last);
        unsafe {
            self.graph
                .swap_segments_unchecked(a_first, a_last, b_first, b_last)
        };

        tracker.commit(self.graph, self.diff);

        unsafe { self.record_segment_reallocation(a_first, a_last, old_ba, old_bb) };
        unsafe { self.record_segment_reallocation(b_first, b_last, old_bb, old_ba) };
    }

    /// # Safety
    ///
    /// Both vessels must be in bounds and form a valid contiguous segment.
    #[inline]
    pub unsafe fn reverse_segment_unchecked(&mut self, first: VesselIndex, last: VesselIndex) {
        if first == last {
            return;
        }

        let arena = self.graph.arena();
        let prev_first = unsafe { arena.prev_unchecked(first.get()) };
        let next_last = unsafe { arena.next_unchecked(last.get()) };

        let prev_first_v = VesselIndex::new(prev_first);
        let next_last_v = VesselIndex::new(next_last);

        self.diff.push_link_broken(prev_first_v, first);
        self.diff.push_link_broken(last, next_last_v);

        let mut curr = first.get();
        let last_raw = last.get();
        while curr != last_raw {
            let nxt = unsafe { arena.next_unchecked(curr) };
            let curr_v = VesselIndex::new(curr);
            let nxt_v = VesselIndex::new(nxt);
            self.diff.push_link_broken(curr_v, nxt_v);
            self.diff.push_link_created(nxt_v, curr_v);
            curr = nxt;
        }

        self.diff.push_link_created(prev_first_v, last);
        self.diff.push_link_created(first, next_last_v);

        self.undo.push_reverse_segment(first, last);
        unsafe { self.graph.reverse_segment_unchecked(first, last) };
    }

    /// # Safety
    ///
    /// Both vessels must be in bounds.
    #[inline]
    pub unsafe fn relocate_after_unchecked(&mut self, vessel: VesselIndex, anchor: VesselIndex) {
        let v_raw = vessel.get();
        let a_raw = anchor.get();
        let prev = unsafe { self.graph.arena().prev_unchecked(v_raw) };

        if v_raw == a_raw || prev == a_raw {
            return;
        }

        let mut tracker = EdgeDeltaTracker::new();
        tracker.track(prev, self.graph);
        tracker.track(v_raw, self.graph);
        tracker.track(a_raw, self.graph);

        let old_b = unsafe { self.graph.vessel_berth_unchecked(vessel) };

        unsafe { self.record_relocate(vessel) };
        unsafe { self.graph.relocate_after_unchecked(vessel, anchor) };

        tracker.commit(self.graph, self.diff);

        let new_b = unsafe { self.graph.vessel_berth_unchecked(vessel) };
        if old_b != new_b {
            self.diff.push_reallocation(vessel, old_b, new_b);
        }
    }

    /// # Safety
    ///
    /// Both vessels must be in bounds.
    #[inline]
    pub unsafe fn relocate_before_unchecked(
        &mut self,
        vessel: VesselIndex,
        reference: VesselIndex,
    ) {
        let v_raw = vessel.get();
        let ref_raw = reference.get();
        let ref_prev = unsafe { self.graph.arena().prev_unchecked(ref_raw) };
        let v_prev = unsafe { self.graph.arena().prev_unchecked(v_raw) };

        if v_raw == ref_raw || v_prev == ref_prev {
            return;
        }

        let mut tracker = EdgeDeltaTracker::new();
        tracker.track(v_prev, self.graph);
        tracker.track(v_raw, self.graph);
        tracker.track(ref_prev, self.graph);

        let old_b = unsafe { self.graph.vessel_berth_unchecked(vessel) };

        unsafe { self.record_relocate(vessel) };
        unsafe { self.graph.relocate_before_unchecked(vessel, reference) };

        tracker.commit(self.graph, self.diff);

        let new_b = unsafe { self.graph.vessel_berth_unchecked(vessel) };
        if old_b != new_b {
            self.diff.push_reallocation(vessel, old_b, new_b);
        }
    }

    /// # Safety
    ///
    /// Vessel and berth must be in bounds.
    #[inline]
    pub unsafe fn relocate_to_head_unchecked(&mut self, vessel: VesselIndex, berth: BerthIndex) {
        let v_raw = vessel.get();
        let sentinel = self.graph.sentinel(berth);
        let prev = unsafe { self.graph.arena().prev_unchecked(v_raw) };

        let mut tracker = EdgeDeltaTracker::new();
        tracker.track(prev, self.graph);
        tracker.track(v_raw, self.graph);
        tracker.track(sentinel, self.graph);

        let old_b = unsafe { self.graph.vessel_berth_unchecked(vessel) };

        unsafe { self.record_relocate(vessel) };
        unsafe { self.graph.relocate_to_head_unchecked(vessel, berth) };

        tracker.commit(self.graph, self.diff);

        if old_b != berth {
            self.diff.push_reallocation(vessel, old_b, berth);
        }
    }

    /// # Safety
    ///
    /// Vessel and berth must be in bounds.
    #[inline]
    pub unsafe fn relocate_to_tail_unchecked(&mut self, vessel: VesselIndex, berth: BerthIndex) {
        let v_raw = vessel.get();
        let sentinel = self.graph.sentinel(berth);
        let prev = unsafe { self.graph.arena().prev_unchecked(v_raw) };
        let old_tail = unsafe { self.graph.arena().prev_unchecked(sentinel) };

        let mut tracker = EdgeDeltaTracker::new();
        tracker.track(prev, self.graph);
        tracker.track(v_raw, self.graph);
        tracker.track(old_tail, self.graph);

        let old_b = unsafe { self.graph.vessel_berth_unchecked(vessel) };

        unsafe { self.record_relocate(vessel) };
        unsafe { self.graph.relocate_to_tail_unchecked(vessel, berth) };

        tracker.commit(self.graph, self.diff);

        if old_b != berth {
            self.diff.push_reallocation(vessel, old_b, berth);
        }
    }

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
        let f_raw = first.get();
        let l_raw = last.get();
        let a_raw = anchor.get();
        let prev = unsafe { self.graph.arena().prev_unchecked(f_raw) };

        if prev == a_raw {
            return;
        }

        let mut tracker = EdgeDeltaTracker::new();
        tracker.track(prev, self.graph);
        tracker.track(l_raw, self.graph);
        tracker.track(a_raw, self.graph);

        let old_b = unsafe { self.graph.vessel_berth_unchecked(first) };

        unsafe { self.record_relocate_segment(first, last) };
        unsafe {
            self.graph
                .relocate_segment_after_unchecked(first, last, anchor)
        };

        tracker.commit(self.graph, self.diff);

        let new_b = unsafe { self.graph.vessel_berth_unchecked(first) };
        unsafe { self.record_segment_reallocation(first, last, old_b, new_b) };
    }

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
        let f_raw = first.get();
        let l_raw = last.get();
        let ref_raw = reference.get();
        let ref_prev = unsafe { self.graph.arena().prev_unchecked(ref_raw) };
        let prev = unsafe { self.graph.arena().prev_unchecked(f_raw) };

        if prev == ref_prev {
            return;
        }

        let mut tracker = EdgeDeltaTracker::new();
        tracker.track(prev, self.graph);
        tracker.track(l_raw, self.graph);
        tracker.track(ref_prev, self.graph);

        let old_b = unsafe { self.graph.vessel_berth_unchecked(first) };

        unsafe { self.record_relocate_segment(first, last) };
        unsafe {
            self.graph
                .relocate_segment_before_unchecked(first, last, reference)
        };

        tracker.commit(self.graph, self.diff);

        let new_b = unsafe { self.graph.vessel_berth_unchecked(first) };
        unsafe { self.record_segment_reallocation(first, last, old_b, new_b) };
    }

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
        let f_raw = first.get();
        let l_raw = last.get();
        let sentinel = self.graph.sentinel(berth);
        let prev = unsafe { self.graph.arena().prev_unchecked(f_raw) };

        let mut tracker = EdgeDeltaTracker::new();
        tracker.track(prev, self.graph);
        tracker.track(l_raw, self.graph);
        tracker.track(sentinel, self.graph);

        let old_b = unsafe { self.graph.vessel_berth_unchecked(first) };

        unsafe { self.record_relocate_segment(first, last) };
        unsafe {
            self.graph
                .relocate_segment_to_head_unchecked(first, last, berth)
        };

        tracker.commit(self.graph, self.diff);

        unsafe { self.record_segment_reallocation(first, last, old_b, berth) };
    }

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
        let f_raw = first.get();
        let l_raw = last.get();
        let sentinel = self.graph.sentinel(berth);
        let prev = unsafe { self.graph.arena().prev_unchecked(f_raw) };
        let old_tail = unsafe { self.graph.arena().prev_unchecked(sentinel) };

        let mut tracker = EdgeDeltaTracker::new();
        tracker.track(prev, self.graph);
        tracker.track(l_raw, self.graph);
        tracker.track(old_tail, self.graph);

        let old_b = unsafe { self.graph.vessel_berth_unchecked(first) };

        unsafe { self.record_relocate_segment(first, last) };
        unsafe {
            self.graph
                .relocate_segment_to_tail_unchecked(first, last, berth)
        };

        tracker.commit(self.graph, self.diff);

        unsafe { self.record_segment_reallocation(first, last, old_b, berth) };
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
