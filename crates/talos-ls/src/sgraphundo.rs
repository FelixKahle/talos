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

//! Undo support for `ScheduleGraph` mutations.
//!
//! `UndoLog` is a **reverse stack machine** designed for high-performance
//! local search. It allows a sequence of topological mutations to be applied
//! to a schedule and then perfectly reverted in LIFO order.

use crate::sgraph::ScheduleGraph;
use std::iter::FusedIterator;
use talos_model::index::{BerthIndex, VesselIndex};

// ----------------------------------------------------------------
// UndoInstruction
// ----------------------------------------------------------------

/// Strongly-typed instructions for the undo stack machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UndoInstruction {
    /// Reverts a swap between two vessels by swapping them again.
    SwapVessels {
        vessel_one: VesselIndex,
        vessel_two: VesselIndex,
    },

    /// Reverts a swap between two segments by swapping them back.
    SwapSegments {
        segment_one_first: VesselIndex,
        segment_one_last: VesselIndex,
        segment_two_first: VesselIndex,
        segment_two_last: VesselIndex,
    },

    /// Reverts a segment reversal. The boundaries are tracked by the *original*
    /// first and last vessels, which become inverted after the forward pass.
    ReverseSegment {
        original_first: VesselIndex,
        original_last: VesselIndex,
    },

    /// Reverts a relocation by placing the subject back immediately after its original predecessor.
    RelocateAfter {
        vessel: VesselIndex,
        original_predecessor: VesselIndex,
    },

    /// Reverts a relocation by placing the subject back at the head of its original berth.
    RelocateToHead {
        vessel: VesselIndex,
        original_berth: BerthIndex,
    },

    /// Reverts a segment relocation by placing the segment back immediately after its original predecessor.
    RelocateSegmentAfter {
        segment_first: VesselIndex,
        segment_last: VesselIndex,
        original_predecessor: VesselIndex,
    },

    /// Reverts a segment relocation by placing the segment back at the head of its original berth.
    RelocateSegmentToHead {
        segment_first: VesselIndex,
        segment_last: VesselIndex,
        original_berth: BerthIndex,
    },
}

impl std::fmt::Display for UndoInstruction {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SwapVessels {
                vessel_one,
                vessel_two,
            } => {
                write!(
                    formatter,
                    "Swap Vessel V{} <-> V{}",
                    vessel_one.get(),
                    vessel_two.get()
                )
            }
            Self::SwapSegments {
                segment_one_first,
                segment_one_last,
                segment_two_first,
                segment_two_last,
            } => {
                write!(
                    formatter,
                    "Swap Segment(V{}..V{}) <-> Segment(V{}..V{})",
                    segment_one_first.get(),
                    segment_one_last.get(),
                    segment_two_first.get(),
                    segment_two_last.get()
                )
            }
            Self::ReverseSegment {
                original_first,
                original_last,
            } => {
                write!(
                    formatter,
                    "Reverse Segment(V{}..V{}) back to original",
                    original_last.get(),
                    original_first.get()
                )
            }
            Self::RelocateAfter {
                vessel,
                original_predecessor,
            } => {
                write!(
                    formatter,
                    "Relocate Vessel V{} back after Vessel V{}",
                    vessel.get(),
                    original_predecessor.get()
                )
            }
            Self::RelocateToHead {
                vessel,
                original_berth,
            } => {
                write!(
                    formatter,
                    "Relocate Vessel V{} back to head of Berth {}",
                    vessel.get(),
                    original_berth.get()
                )
            }
            Self::RelocateSegmentAfter {
                segment_first,
                segment_last,
                original_predecessor,
            } => {
                write!(
                    formatter,
                    "Relocate Segment(V{}..V{}) back after Vessel V{}",
                    segment_first.get(),
                    segment_last.get(),
                    original_predecessor.get()
                )
            }
            Self::RelocateSegmentToHead {
                segment_first,
                segment_last,
                original_berth,
            } => {
                write!(
                    formatter,
                    "Relocate Segment(V{}..V{}) back to head of Berth {}",
                    segment_first.get(),
                    segment_last.get(),
                    original_berth.get()
                )
            }
        }
    }
}

// ----------------------------------------------------------------
// UndoLogIter
// ----------------------------------------------------------------

/// Iterator over recorded `UndoInstruction`s in chronological (push) order.
#[derive(Debug, Clone)]
pub struct UndoLogIter<'a> {
    iter: std::slice::Iter<'a, UndoInstruction>,
}

impl<'a> Iterator for UndoLogIter<'a> {
    type Item = &'a UndoInstruction;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a> DoubleEndedIterator for UndoLogIter<'a> {
    #[inline(always)]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back()
    }
}

impl FusedIterator for UndoLogIter<'_> {}
impl ExactSizeIterator for UndoLogIter<'_> {}

// ----------------------------------------------------------------
// ScheduleGraphUndoLog
// ----------------------------------------------------------------

/// A high-performance, purely instruction-based undo log for `ScheduleGraph` mutations.
///
/// Because the `ScheduleGraph` is an arena-backed doubly-linked list, all topological
/// mutations have strict mathematical inverses. This log stores exactly how to execute
/// those inverses without requiring heap allocations or state cloning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScheduleGraphUndoLog {
    /// The primary instruction stack, evaluated in LIFO order upon rollback.
    stack: Vec<UndoInstruction>,
}

impl ScheduleGraphUndoLog {
    const MIN_CAP_OPS: usize = 16;
    const OPS_PER_VESSEL: usize = 8;

    /// Creates a new `UndoLog` with the given initial capacity in operations.
    #[inline]
    pub fn new(capacity: usize) -> Self {
        Self {
            stack: Vec::with_capacity(capacity),
        }
    }

    /// Creates a new `UndoLog` heuristically sized for a given number of vessels,
    /// ensuring no reallocations occur during standard local search depths.
    #[inline]
    pub fn preallocated(num_vessels: usize) -> Self {
        let capacity = num_vessels
            .saturating_mul(Self::OPS_PER_VESSEL)
            .max(Self::MIN_CAP_OPS);

        Self {
            stack: Vec::with_capacity(capacity),
        }
    }

    /// Discards all recorded operations without deallocating internal buffers.
    /// This should be called when a neighborhood move is **accepted**.
    #[inline(always)]
    pub fn clear(&mut self) {
        self.stack.clear();
    }

    /// Returns `true` if no operations are currently recorded.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.stack.is_empty()
    }

    /// Returns the number of currently recorded operations.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.stack.len()
    }

    /// Records the intent to swap two vessels.
    /// Must be called *before* the actual swap occurs.
    #[inline(always)]
    pub fn push_swap_vessels(&mut self, vessel1: VesselIndex, vessel2: VesselIndex) {
        self.stack.push(UndoInstruction::SwapVessels {
            vessel_one: vessel1,
            vessel_two: vessel2,
        });
    }

    /// Records the intent to swap two segments.
    /// Must be called *before* the actual swap occurs.
    #[inline(always)]
    pub fn push_swap_segments(
        &mut self,
        segment1_first: VesselIndex,
        segment1_last: VesselIndex,
        segment2_first: VesselIndex,
        segment2_last: VesselIndex,
    ) {
        self.stack.push(UndoInstruction::SwapSegments {
            segment_one_first: segment1_first,
            segment_one_last: segment1_last,
            segment_two_first: segment2_first,
            segment_two_last: segment2_last,
        });
    }

    /// Records the intent to reverse a contiguous segment.
    /// Must be called *before* the actual reversal occurs.
    #[inline(always)]
    pub fn push_reverse_segment(&mut self, first: VesselIndex, last: VesselIndex) {
        self.stack.push(UndoInstruction::ReverseSegment {
            original_first: first,
            original_last: last,
        });
    }

    /// Records the intent to relocate a vessel, noting its original predecessor.
    /// Must be called *before* the vessel is extracted from its origin.
    #[inline(always)]
    pub fn push_relocate_after(&mut self, subject: VesselIndex, original_predecessor: VesselIndex) {
        self.stack.push(UndoInstruction::RelocateAfter {
            vessel: subject,
            original_predecessor,
        });
    }

    /// Records the intent to relocate a vessel, noting that it originally sat at the head of a berth.
    /// Must be called *before* the vessel is extracted from its origin.
    #[inline(always)]
    pub fn push_relocate_to_head(&mut self, subject: VesselIndex, original_berth: BerthIndex) {
        self.stack.push(UndoInstruction::RelocateToHead {
            vessel: subject,
            original_berth,
        });
    }

    /// Records the intent to relocate a segment, noting its original predecessor.
    /// Must be called *before* the segment is extracted from its origin.
    #[inline(always)]
    pub fn push_relocate_segment_after(
        &mut self,
        segment_first: VesselIndex,
        segment_last: VesselIndex,
        original_predecessor: VesselIndex,
    ) {
        self.stack.push(UndoInstruction::RelocateSegmentAfter {
            segment_first,
            segment_last,
            original_predecessor,
        });
    }

    /// Records the intent to relocate a segment, noting that it originally sat at the head of a berth.
    /// Must be called *before* the segment is extracted from its origin.
    #[inline(always)]
    pub fn push_relocate_segment_to_head(
        &mut self,
        segment_first: VesselIndex,
        segment_last: VesselIndex,
        original_berth: BerthIndex,
    ) {
        self.stack.push(UndoInstruction::RelocateSegmentToHead {
            segment_first,
            segment_last,
            original_berth,
        });
    }

    /// Pops all recorded instructions in reverse (LIFO) order and applies their
    /// exact topological inverses to the `ScheduleGraph`.
    ///
    /// Once complete, the log is entirely empty, and the graph is mathematically
    /// identical to its state prior to the first logged mutation.
    #[inline(always)]
    pub fn apply_rollback(&mut self, graph: &mut ScheduleGraph) {
        while let Some(instruction) = self.stack.pop() {
            match instruction {
                UndoInstruction::SwapVessels {
                    vessel_one: vessel1,
                    vessel_two: vessel2,
                } => unsafe {
                    graph.swap_vessels_unchecked(vessel1, vessel2);
                },

                UndoInstruction::SwapSegments {
                    segment_one_first: segment1_first,
                    segment_one_last: segment1_last,
                    segment_two_first: segment2_first,
                    segment_two_last: segment2_last,
                } => unsafe {
                    graph.swap_segments_unchecked(
                        segment2_first,
                        segment2_last,
                        segment1_first,
                        segment1_last,
                    );
                },

                UndoInstruction::ReverseSegment {
                    original_first,
                    original_last,
                } => unsafe {
                    graph.reverse_segment_unchecked(original_last, original_first);
                },

                UndoInstruction::RelocateAfter {
                    vessel: subject,
                    original_predecessor: predecessor,
                } => unsafe {
                    graph.relocate_after_unchecked(subject, predecessor);
                },

                UndoInstruction::RelocateToHead {
                    vessel: subject,
                    original_berth: berth,
                } => unsafe {
                    graph.relocate_to_head_unchecked(subject, berth);
                },

                UndoInstruction::RelocateSegmentAfter {
                    segment_first,
                    segment_last,
                    original_predecessor: predecessor,
                } => unsafe {
                    graph.relocate_segment_after_unchecked(
                        segment_first,
                        segment_last,
                        predecessor,
                    );
                },

                UndoInstruction::RelocateSegmentToHead {
                    segment_first,
                    segment_last,
                    original_berth: berth,
                } => unsafe {
                    graph.relocate_segment_to_head_unchecked(segment_first, segment_last, berth);
                },
            }
        }
    }
}

impl<'a> IntoIterator for &'a ScheduleGraphUndoLog {
    type Item = &'a UndoInstruction;
    type IntoIter = UndoLogIter<'a>;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        UndoLogIter {
            iter: self.stack.iter(),
        }
    }
}

impl std::fmt::Display for ScheduleGraphUndoLog {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let len = self.stack.len();
        writeln!(formatter, "ScheduleGraphUndoLog [Instructions: {}]", len)?;

        if len == 0 {
            return write!(formatter, "  (Empty)");
        }

        for (i, instruction) in self.stack.iter().rev().enumerate() {
            // "Next" marks the very first instruction that will be popped.
            let marker = if i == 0 { "Next -> " } else { "        " };
            writeln!(formatter, "  {}{}", marker, instruction)?;
        }
        Ok(())
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
    /// B0: V0 -> V1 -> V2
    /// B1: V3 -> V4
    /// B2: (empty)
    fn setup_graph() -> ScheduleGraph {
        let berths = [b(0), b(0), b(0), b(1), b(1)];
        let starts = [10, 20, 30, 10, 20];
        ScheduleGraph::from_slices(&berths, &starts, 3)
    }

    #[test]
    fn test_rollback_swap_vessels() {
        let mut graph = setup_graph();
        let mut log = ScheduleGraphUndoLog::new(10);

        log.push_swap_vessels(v(0), v(2));
        unsafe { graph.swap_vessels_unchecked(v(0), v(2)) };

        let seq: Vec<_> = graph.vessel_sequence_iter(b(0)).collect();
        assert_eq!(seq, vec![v(2), v(1), v(0)]);

        log.apply_rollback(&mut graph);

        let restored: Vec<_> = graph.vessel_sequence_iter(b(0)).collect();
        assert_eq!(restored, vec![v(0), v(1), v(2)]);
        assert!(log.is_empty());
    }

    #[test]
    fn test_rollback_swap_segments() {
        let mut graph = setup_graph();
        let mut log = ScheduleGraphUndoLog::new(10);

        log.push_swap_segments(v(0), v(1), v(3), v(4));
        unsafe { graph.swap_segments_unchecked(v(0), v(1), v(3), v(4)) };

        let seq0: Vec<_> = graph.vessel_sequence_iter(b(0)).collect();
        assert_eq!(seq0, vec![v(3), v(4), v(2)]);

        log.apply_rollback(&mut graph);

        let restored: Vec<_> = graph.vessel_sequence_iter(b(0)).collect();
        assert_eq!(restored, vec![v(0), v(1), v(2)]);
    }

    #[test]
    fn test_rollback_reverse_segment() {
        let mut graph = setup_graph();
        let mut log = ScheduleGraphUndoLog::new(10);

        log.push_reverse_segment(v(0), v(2));
        unsafe { graph.reverse_segment_unchecked(v(0), v(2)) };

        let seq: Vec<_> = graph.vessel_sequence_iter(b(0)).collect();
        assert_eq!(seq, vec![v(2), v(1), v(0)]);

        log.apply_rollback(&mut graph);

        let restored: Vec<_> = graph.vessel_sequence_iter(b(0)).collect();
        assert_eq!(restored, vec![v(0), v(1), v(2)]);
    }

    #[test]
    fn test_rollback_relocate_within_berth() {
        let mut graph = setup_graph();
        let mut log = ScheduleGraphUndoLog::new(10);

        // Move V0 (originally at head of B0) after V2
        log.push_relocate_to_head(v(0), b(0));
        unsafe { graph.relocate_after_unchecked(v(0), v(2)) };

        let seq: Vec<_> = graph.vessel_sequence_iter(b(0)).collect();
        assert_eq!(seq, vec![v(1), v(2), v(0)]);

        log.apply_rollback(&mut graph);

        let restored: Vec<_> = graph.vessel_sequence_iter(b(0)).collect();
        assert_eq!(restored, vec![v(0), v(1), v(2)]);
    }

    #[test]
    fn test_rollback_relocate_segment_to_new_berth() {
        let mut graph = setup_graph();
        let mut log = ScheduleGraphUndoLog::new(10);

        // Move [0..1] (originally head of B0) to head of B2
        log.push_relocate_segment_to_head(v(0), v(1), b(0));
        unsafe { graph.relocate_segment_to_head_unchecked(v(0), v(1), b(2)) };

        let seq_b0: Vec<_> = graph.vessel_sequence_iter(b(0)).collect();
        let seq_b2: Vec<_> = graph.vessel_sequence_iter(b(2)).collect();

        assert_eq!(seq_b0, vec![v(2)]);
        assert_eq!(seq_b2, vec![v(0), v(1)]);

        log.apply_rollback(&mut graph);

        let restored_b0: Vec<_> = graph.vessel_sequence_iter(b(0)).collect();
        let restored_b2: Vec<_> = graph.vessel_sequence_iter(b(2)).collect();
        assert_eq!(restored_b0, vec![v(0), v(1), v(2)]);
        assert!(restored_b2.is_empty());
    }

    #[test]
    fn test_lifo_execution_order() {
        let mut graph = setup_graph();
        let original_graph = graph.clone();
        let mut log = ScheduleGraphUndoLog::new(20);

        // Op 1: Move V2 to head of empty B2 (Record: V2 was originally after V1)
        log.push_relocate_after(v(2), v(1));
        unsafe { graph.relocate_to_head_unchecked(v(2), b(2)) };

        // Op 2: Reverse [3..4] in B1
        log.push_reverse_segment(v(3), v(4));
        unsafe { graph.reverse_segment_unchecked(v(3), v(4)) };

        // Op 3: Swap V0 and V3
        log.push_swap_vessels(v(0), v(3));
        unsafe { graph.swap_vessels_unchecked(v(0), v(3)) };

        assert_ne!(graph, original_graph);

        log.apply_rollback(&mut graph);

        assert_eq!(graph, original_graph);
        assert!(log.is_empty());
    }

    #[test]
    fn test_undo_log_capacity_and_clear() {
        let mut log = ScheduleGraphUndoLog::preallocated(10);
        // 10 vessels * 8 ops_per_vessel = 80, which is > MIN_CAP_OPS (16)
        assert!(log.stack.capacity() >= 80);

        let small_log = ScheduleGraphUndoLog::preallocated(1);
        // 1 vessel * 8 ops = 8. Should clamp to MIN_CAP_OPS (16)
        assert!(small_log.stack.capacity() >= 16);

        log.push_swap_vessels(v(0), v(1));
        log.push_reverse_segment(v(2), v(3));
        assert_eq!(log.len(), 2);
        assert!(!log.is_empty());

        // Clearing should reset length but maintain capacity
        let cap_before = log.stack.capacity();
        log.clear();
        assert_eq!(log.len(), 0);
        assert!(log.is_empty());
        assert_eq!(log.stack.capacity(), cap_before);
    }

    #[test]
    fn test_undo_log_iterator() {
        let mut log = ScheduleGraphUndoLog::new(10);
        log.push_swap_vessels(v(0), v(1));
        log.push_relocate_to_head(v(2), b(1));

        let ops: Vec<_> = log.into_iter().cloned().collect();

        assert_eq!(ops.len(), 2);
        assert_eq!(
            ops[0],
            UndoInstruction::SwapVessels {
                vessel_one: v(0),
                vessel_two: v(1)
            }
        );
        assert_eq!(
            ops[1],
            UndoInstruction::RelocateToHead {
                vessel: v(2),
                original_berth: b(1)
            }
        );
    }

    #[test]
    fn test_rollback_relocate_segment_after_cross_berth() {
        let mut graph = setup_graph();
        let mut log = ScheduleGraphUndoLog::new(10);

        // Move [0..1] from B0 to after 3 in B1.
        // Original predecessor of V0 was the sentinel of B0.
        // Note: graph.raw_prev is used here conceptually; push the *intent* of restoration.
        // To restore [0..1] to head of B0, we push `RelocateSegmentToHead`
        log.push_relocate_segment_to_head(v(0), v(1), b(0));
        unsafe { graph.relocate_segment_after_unchecked(v(0), v(1), v(3)) };

        let seq_b0: Vec<_> = graph.vessel_sequence_iter(b(0)).collect();
        let seq_b1: Vec<_> = graph.vessel_sequence_iter(b(1)).collect();

        assert_eq!(seq_b0, vec![v(2)]);
        assert_eq!(seq_b1, vec![v(3), v(0), v(1), v(4)]);

        log.apply_rollback(&mut graph);

        let restored_b0: Vec<_> = graph.vessel_sequence_iter(b(0)).collect();
        let restored_b1: Vec<_> = graph.vessel_sequence_iter(b(1)).collect();

        assert_eq!(restored_b0, vec![v(0), v(1), v(2)]);
        assert_eq!(restored_b1, vec![v(3), v(4)]);
    }

    #[test]
    fn test_rollback_relocate_after_same_berth() {
        let mut graph = setup_graph();
        let mut log = ScheduleGraphUndoLog::new(10);

        // B0 is initially: V0 -> V1 -> V2
        // We want to move V2 to be after V0.
        // To revert this, we must put V2 back after V1.
        log.push_relocate_after(v(2), v(1));
        unsafe { graph.relocate_after_unchecked(v(2), v(0)) };

        let seq: Vec<_> = graph.vessel_sequence_iter(b(0)).collect();
        assert_eq!(seq, vec![v(0), v(2), v(1)]);

        log.apply_rollback(&mut graph);

        let restored: Vec<_> = graph.vessel_sequence_iter(b(0)).collect();
        assert_eq!(restored, vec![v(0), v(1), v(2)]);
    }

    #[test]
    fn test_rollback_complex_intertwined_segments() {
        let mut graph = setup_graph();
        let mut log = ScheduleGraphUndoLog::new(10);

        // B0: V0 -> V1 -> V2
        // B1: V3 -> V4

        // 1. Move V4 to head of B0.
        // Revert: Relocate V4 after V3.
        log.push_relocate_after(v(4), v(3));
        unsafe { graph.relocate_to_head_unchecked(v(4), b(0)) };
        // B0: V4 -> V0 -> V1 -> V2
        // B1: V3

        // 2. Swap segment [0, 1] with segment [3, 3] (single element).
        // Revert: Swap them back.
        log.push_swap_segments(v(0), v(1), v(3), v(3));
        unsafe { graph.swap_segments_unchecked(v(0), v(1), v(3), v(3)) };
        // B0: V4 -> V3 -> V2
        // B1: V0 -> V1

        log.apply_rollback(&mut graph);

        // Assert it perfectly unwound both operations
        let restored_b0: Vec<_> = graph.vessel_sequence_iter(b(0)).collect();
        let restored_b1: Vec<_> = graph.vessel_sequence_iter(b(1)).collect();

        assert_eq!(restored_b0, vec![v(0), v(1), v(2)]);
        assert_eq!(restored_b1, vec![v(3), v(4)]);
    }
}
