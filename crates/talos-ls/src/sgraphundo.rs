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
//! `ScheduleGraphUndoLog` is a **reverse stack machine** designed for high-performance
//! local search. It allows a sequence of topological mutations to be applied
//! to a schedule and then perfectly reverted in LIFO order.
//!
//! # Public Types
//!
//! The undo log uses only [`VesselIndex`] and [`BerthIndex`] — no internal
//! graph types leak into this API.

use crate::sgraph::ScheduleGraph;
use std::iter::FusedIterator;
use talos_model::index::{BerthIndex, VesselIndex};

// ----------------------------------------------------------------
// UndoInstruction
// ----------------------------------------------------------------

/// Strongly-typed instructions for the undo stack machine.
///
/// Each variant encodes the exact inverse of a forward mutation using only
/// domain-level types (`VesselIndex`, `BerthIndex`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UndoInstruction {
    /// Reverts a swap between two vessels by swapping them again.
    SwapVessels {
        vessel_a: VesselIndex,
        vessel_b: VesselIndex,
    },

    /// Reverts a swap between two segments by swapping them back.
    SwapSegments {
        a_first: VesselIndex,
        a_last: VesselIndex,
        b_first: VesselIndex,
        b_last: VesselIndex,
    },

    /// Reverts a segment reversal by reversing again.
    /// Boundaries are the *original* first and last, which are inverted after the forward pass.
    ReverseSegment {
        original_first: VesselIndex,
        original_last: VesselIndex,
    },

    /// Reverts a relocation where the vessel was originally after another vessel.
    RelocateAfterVessel {
        vessel: VesselIndex,
        predecessor: VesselIndex,
    },

    /// Reverts a relocation where the vessel was originally at the head of a berth.
    RelocateToHead {
        vessel: VesselIndex,
        berth: BerthIndex,
    },

    /// Reverts a segment relocation where the segment was originally after another vessel.
    RelocateSegmentAfterVessel {
        segment_first: VesselIndex,
        segment_last: VesselIndex,
        predecessor: VesselIndex,
    },

    /// Reverts a segment relocation where the segment was originally at the head of a berth.
    RelocateSegmentToHead {
        segment_first: VesselIndex,
        segment_last: VesselIndex,
        berth: BerthIndex,
    },
}

impl std::fmt::Display for UndoInstruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SwapVessels { vessel_a, vessel_b } => {
                write!(f, "Swap V{} <-> V{}", vessel_a.get(), vessel_b.get())
            }
            Self::SwapSegments {
                a_first,
                a_last,
                b_first,
                b_last,
            } => write!(
                f,
                "Swap Segment(V{}..V{}) <-> Segment(V{}..V{})",
                a_first.get(),
                a_last.get(),
                b_first.get(),
                b_last.get()
            ),
            Self::ReverseSegment {
                original_first,
                original_last,
            } => write!(
                f,
                "Reverse Segment(V{}..V{}) back",
                original_last.get(),
                original_first.get()
            ),
            Self::RelocateAfterVessel {
                vessel,
                predecessor,
            } => write!(
                f,
                "Relocate V{} back after V{}",
                vessel.get(),
                predecessor.get()
            ),
            Self::RelocateToHead { vessel, berth } => write!(
                f,
                "Relocate V{} back to head of Berth {}",
                vessel.get(),
                berth.get()
            ),
            Self::RelocateSegmentAfterVessel {
                segment_first,
                segment_last,
                predecessor,
            } => write!(
                f,
                "Relocate Segment(V{}..V{}) back after V{}",
                segment_first.get(),
                segment_last.get(),
                predecessor.get()
            ),
            Self::RelocateSegmentToHead {
                segment_first,
                segment_last,
                berth,
            } => write!(
                f,
                "Relocate Segment(V{}..V{}) back to head of Berth {}",
                segment_first.get(),
                segment_last.get(),
                berth.get()
            ),
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
/// All topological mutations on the `ScheduleGraph` have strict mathematical inverses.
/// This log stores exactly how to execute those inverses without heap allocations
/// or state cloning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScheduleGraphUndoLog {
    stack: Vec<UndoInstruction>,
}

impl ScheduleGraphUndoLog {
    const MIN_CAP_OPS: usize = 16;
    const OPS_PER_VESSEL: usize = 8;

    /// Creates a new undo log with the given initial capacity.
    #[inline]
    pub fn new(capacity: usize) -> Self {
        Self {
            stack: Vec::with_capacity(capacity),
        }
    }

    /// Creates a new undo log heuristically sized for the given number of vessels.
    #[inline]
    pub fn preallocated(num_vessels: usize) -> Self {
        let capacity = num_vessels
            .saturating_mul(Self::OPS_PER_VESSEL)
            .max(Self::MIN_CAP_OPS);
        Self {
            stack: Vec::with_capacity(capacity),
        }
    }

    /// Discards all recorded operations without deallocating.
    #[inline(always)]
    pub fn clear(&mut self) {
        self.stack.clear();
    }

    /// Returns `true` if no operations are recorded.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.stack.is_empty()
    }

    /// Returns the number of recorded operations.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.stack.len()
    }

    // ----------------------------------------------------------------
    // Push methods
    // ----------------------------------------------------------------

    /// Records a vessel swap. Must be called *before* the swap.
    #[inline(always)]
    pub fn push_swap_vessels(&mut self, a: VesselIndex, b: VesselIndex) {
        self.stack.push(UndoInstruction::SwapVessels {
            vessel_a: a,
            vessel_b: b,
        });
    }

    /// Records a segment swap. Must be called *before* the swap.
    #[inline(always)]
    pub fn push_swap_segments(
        &mut self,
        a_first: VesselIndex,
        a_last: VesselIndex,
        b_first: VesselIndex,
        b_last: VesselIndex,
    ) {
        self.stack.push(UndoInstruction::SwapSegments {
            a_first,
            a_last,
            b_first,
            b_last,
        });
    }

    /// Records a segment reversal. Must be called *before* the reversal.
    #[inline(always)]
    pub fn push_reverse_segment(&mut self, first: VesselIndex, last: VesselIndex) {
        self.stack.push(UndoInstruction::ReverseSegment {
            original_first: first,
            original_last: last,
        });
    }

    /// Records a relocation where the vessel was after another vessel.
    #[inline(always)]
    pub fn push_relocate_after_vessel(&mut self, vessel: VesselIndex, predecessor: VesselIndex) {
        self.stack.push(UndoInstruction::RelocateAfterVessel {
            vessel,
            predecessor,
        });
    }

    /// Records a relocation where the vessel was at the head of a berth.
    #[inline(always)]
    pub fn push_relocate_to_head(&mut self, vessel: VesselIndex, berth: BerthIndex) {
        self.stack
            .push(UndoInstruction::RelocateToHead { vessel, berth });
    }

    /// Records a segment relocation where the segment was after another vessel.
    #[inline(always)]
    pub fn push_relocate_segment_after_vessel(
        &mut self,
        first: VesselIndex,
        last: VesselIndex,
        predecessor: VesselIndex,
    ) {
        self.stack
            .push(UndoInstruction::RelocateSegmentAfterVessel {
                segment_first: first,
                segment_last: last,
                predecessor,
            });
    }

    /// Records a segment relocation where the segment was at the head of a berth.
    #[inline(always)]
    pub fn push_relocate_segment_to_head(
        &mut self,
        first: VesselIndex,
        last: VesselIndex,
        berth: BerthIndex,
    ) {
        self.stack.push(UndoInstruction::RelocateSegmentToHead {
            segment_first: first,
            segment_last: last,
            berth,
        });
    }

    // ----------------------------------------------------------------
    // Rollback
    // ----------------------------------------------------------------

    /// Pops all recorded instructions in LIFO order and applies their inverses.
    ///
    /// After completion, the log is empty and the graph is identical to its
    /// state prior to the first logged mutation.
    #[inline(always)]
    pub fn apply_rollback(&mut self, graph: &mut ScheduleGraph) {
        let ptr = self.stack.as_ptr();
        for i in (0..self.stack.len()).rev() {
            debug_assert!(i < self.stack.len());

            // SAFETY: `i < self.stack.len()` is guaranteed by the loop bounds.
            let instruction = unsafe { *ptr.add(i) };
            match instruction {
                UndoInstruction::SwapVessels { vessel_a, vessel_b } => unsafe {
                    graph.swap_vessels_unchecked(vessel_a, vessel_b);
                },

                UndoInstruction::SwapSegments {
                    a_first,
                    a_last,
                    b_first,
                    b_last,
                } => unsafe {
                    // Swap back: note the reversed order.
                    graph.swap_segments_unchecked(b_first, b_last, a_first, a_last);
                },

                UndoInstruction::ReverseSegment {
                    original_first,
                    original_last,
                } => unsafe {
                    // After the forward reversal, original_last is now the head
                    // and original_first is the tail. Reversing that range restores order.
                    graph.reverse_segment_unchecked(original_last, original_first);
                },

                UndoInstruction::RelocateAfterVessel {
                    vessel,
                    predecessor,
                } => unsafe {
                    graph.relocate_after_unchecked(vessel, predecessor);
                },

                UndoInstruction::RelocateToHead { vessel, berth } => unsafe {
                    graph.relocate_to_head_unchecked(vessel, berth);
                },

                UndoInstruction::RelocateSegmentAfterVessel {
                    segment_first,
                    segment_last,
                    predecessor,
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
                    berth,
                } => unsafe {
                    graph.relocate_segment_to_head_unchecked(segment_first, segment_last, berth);
                },
            }
        }
        self.stack.clear();
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
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let len = self.stack.len();
        writeln!(f, "ScheduleGraphUndoLog [Instructions: {}]", len)?;
        if len == 0 {
            return write!(f, "  (Empty)");
        }
        for (i, instruction) in self.stack.iter().rev().enumerate() {
            let marker = if i == 0 { "Next -> " } else { "        " };
            writeln!(f, "  {}{}", marker, instruction)?;
        }
        Ok(())
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

        let seq: Vec<_> = graph.vessel_sequence_iter(b(0)).collect();
        assert_eq!(seq, vec![v(3), v(4), v(2)]);

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
    fn test_rollback_relocate_after_vessel() {
        let mut graph = setup_graph();
        let mut log = ScheduleGraphUndoLog::new(10);

        // V0 is at head of B0, predecessor is sentinel -> use RelocateToHead
        // But first test the AfterVessel variant: move V2 (predecessor is V1)
        log.push_relocate_after_vessel(v(2), v(1));
        unsafe { graph.relocate_after_unchecked(v(2), v(0)) };

        let seq: Vec<_> = graph.vessel_sequence_iter(b(0)).collect();
        assert_eq!(seq, vec![v(0), v(2), v(1)]);

        log.apply_rollback(&mut graph);

        let restored: Vec<_> = graph.vessel_sequence_iter(b(0)).collect();
        assert_eq!(restored, vec![v(0), v(1), v(2)]);
    }

    #[test]
    fn test_rollback_relocate_to_head() {
        let mut graph = setup_graph();
        let mut log = ScheduleGraphUndoLog::new(10);

        // V0 is at head of B0. Move it after V3 in B1.
        log.push_relocate_to_head(v(0), b(0));
        unsafe { graph.relocate_after_unchecked(v(0), v(3)) };

        let seq_b0: Vec<_> = graph.vessel_sequence_iter(b(0)).collect();
        let seq_b1: Vec<_> = graph.vessel_sequence_iter(b(1)).collect();
        assert_eq!(seq_b0, vec![v(1), v(2)]);
        assert_eq!(seq_b1, vec![v(3), v(0), v(4)]);

        log.apply_rollback(&mut graph);

        let restored_b0: Vec<_> = graph.vessel_sequence_iter(b(0)).collect();
        let restored_b1: Vec<_> = graph.vessel_sequence_iter(b(1)).collect();
        assert_eq!(restored_b0, vec![v(0), v(1), v(2)]);
        assert_eq!(restored_b1, vec![v(3), v(4)]);
    }

    #[test]
    fn test_rollback_relocate_segment_after_vessel() {
        let mut graph = setup_graph();
        let mut log = ScheduleGraphUndoLog::new(10);

        // Move [V1, V2] (predecessor V0) to after V3 in B1.
        log.push_relocate_segment_after_vessel(v(1), v(2), v(0));
        unsafe { graph.relocate_segment_after_unchecked(v(1), v(2), v(3)) };

        let seq_b0: Vec<_> = graph.vessel_sequence_iter(b(0)).collect();
        let seq_b1: Vec<_> = graph.vessel_sequence_iter(b(1)).collect();
        assert_eq!(seq_b0, vec![v(0)]);
        assert_eq!(seq_b1, vec![v(3), v(1), v(2), v(4)]);

        log.apply_rollback(&mut graph);

        let restored_b0: Vec<_> = graph.vessel_sequence_iter(b(0)).collect();
        let restored_b1: Vec<_> = graph.vessel_sequence_iter(b(1)).collect();
        assert_eq!(restored_b0, vec![v(0), v(1), v(2)]);
        assert_eq!(restored_b1, vec![v(3), v(4)]);
    }

    #[test]
    fn test_rollback_relocate_segment_to_head() {
        let mut graph = setup_graph();
        let mut log = ScheduleGraphUndoLog::new(10);

        // Move [V0, V1] (at head of B0) to head of B2.
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
        let original = graph.clone();
        let mut log = ScheduleGraphUndoLog::new(20);

        // Op 1: Move V2 to head of empty B2 (V2 was after V1)
        log.push_relocate_after_vessel(v(2), v(1));
        unsafe { graph.relocate_to_head_unchecked(v(2), b(2)) };

        // Op 2: Reverse [V3, V4] in B1
        log.push_reverse_segment(v(3), v(4));
        unsafe { graph.reverse_segment_unchecked(v(3), v(4)) };

        // Op 3: Swap V0 and V3
        log.push_swap_vessels(v(0), v(3));
        unsafe { graph.swap_vessels_unchecked(v(0), v(3)) };

        assert_ne!(graph, original);

        log.apply_rollback(&mut graph);

        assert_eq!(graph, original);
        assert!(log.is_empty());
    }

    #[test]
    fn test_capacity_and_clear() {
        let mut log = ScheduleGraphUndoLog::preallocated(10);
        assert!(log.stack.capacity() >= 80);

        let small = ScheduleGraphUndoLog::preallocated(1);
        assert!(small.stack.capacity() >= 16);

        log.push_swap_vessels(v(0), v(1));
        log.push_reverse_segment(v(2), v(3));
        assert_eq!(log.len(), 2);

        let cap = log.stack.capacity();
        log.clear();
        assert!(log.is_empty());
        assert_eq!(log.stack.capacity(), cap);
    }

    #[test]
    fn test_iterator() {
        let mut log = ScheduleGraphUndoLog::new(10);
        log.push_swap_vessels(v(0), v(1));
        log.push_relocate_to_head(v(2), b(1));

        let ops: Vec<_> = log.into_iter().cloned().collect();
        assert_eq!(ops.len(), 2);
        assert_eq!(
            ops[0],
            UndoInstruction::SwapVessels {
                vessel_a: v(0),
                vessel_b: v(1)
            }
        );
        assert_eq!(
            ops[1],
            UndoInstruction::RelocateToHead {
                vessel: v(2),
                berth: b(1)
            }
        );
    }
}
