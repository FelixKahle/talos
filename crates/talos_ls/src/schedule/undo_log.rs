// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The record of how to undo a sequence of moves on a schedule graph, and the undoing.
//!
//! The local search makes a candidate's moves on the graph and takes them back when it rejects the candidate. The
//! log keeps the inverse of each move as a few ids, so undoing costs what the moves cost, and clearing keeps its
//! memory, so a run records its moves without allocating once the log has grown.

use super::graph::{Outside, ScheduleGraph};
use talos_model::{BerthIndex, VesselIndex};

// ----------------------------------------------------------------
// UndoInstruction
// ----------------------------------------------------------------

/// The inverse of one move, as the ids that undo it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UndoInstruction {
    /// Reverts a swap between two vessels by swapping them again.
    SwapVessels {
        vessel_a: VesselIndex,
        vessel_b: VesselIndex,
    },

    /// Reverts a swap between two segments by swapping them back.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "a move of the schedule no neighbourhood makes yet; its tests keep it sound"
        )
    )]
    SwapSegments {
        a_first: VesselIndex,
        a_last: VesselIndex,
        b_first: VesselIndex,
        b_last: VesselIndex,
    },

    /// Reverts a segment reversal by reversing again.
    /// Boundaries are the *original* first and last, which are inverted after the forward pass.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "a move of the schedule no neighbourhood makes yet; its tests keep it sound"
        )
    )]
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

impl UndoInstruction {
    /// The first vessel, or else the berth, the instruction names that is not one of `graph`'s.
    fn outside(self, graph: &ScheduleGraph) -> Option<Outside> {
        let vessel = |vessel| graph.vessel_outside(vessel);
        match self {
            Self::SwapVessels { vessel_a, vessel_b } => {
                [vessel_a, vessel_b].into_iter().find_map(vessel)
            }
            Self::SwapSegments {
                a_first,
                a_last,
                b_first,
                b_last,
            } => [a_first, a_last, b_first, b_last]
                .into_iter()
                .find_map(vessel),
            Self::ReverseSegment {
                original_first,
                original_last,
            } => [original_first, original_last].into_iter().find_map(vessel),
            Self::RelocateAfterVessel {
                vessel: moved,
                predecessor,
            } => [moved, predecessor].into_iter().find_map(vessel),
            Self::RelocateToHead {
                vessel: moved,
                berth,
            } => vessel(moved).or_else(|| graph.berth_outside(berth)),
            Self::RelocateSegmentAfterVessel {
                segment_first,
                segment_last,
                predecessor,
            } => [segment_first, segment_last, predecessor]
                .into_iter()
                .find_map(vessel),
            Self::RelocateSegmentToHead {
                segment_first,
                segment_last,
                berth,
            } => [segment_first, segment_last]
                .into_iter()
                .find_map(vessel)
                .or_else(|| graph.berth_outside(berth)),
        }
    }

    /// Applies the inverse the instruction records to `graph`.
    ///
    /// # Safety
    ///
    /// Every vessel and berth the instruction names must be one of `graph`'s.
    #[inline(always)]
    unsafe fn undo_unchecked(self, graph: &mut ScheduleGraph) {
        debug_assert!(
            self.outside(graph).is_none(),
            "called `UndoInstruction::undo_unchecked` with {self}, which names an id outside the graph"
        );
        match self {
            // SAFETY: both vessels are the graph's, the precondition of this function.
            Self::SwapVessels { vessel_a, vessel_b } => unsafe {
                graph.swap_vessels_unchecked(vessel_a, vessel_b);
            },
            // The segments trade places again: `b` now stands where `a` stood.
            Self::SwapSegments {
                a_first,
                a_last,
                b_first,
                b_last,
            } => {
                // SAFETY: the four vessels are the graph's, the precondition of this function.
                unsafe {
                    graph.swap_segments_unchecked(b_first, b_last, a_first, a_last);
                }
            }
            // The reversal left the original last vessel first; reversing that segment restores the order.
            Self::ReverseSegment {
                original_first,
                original_last,
            } => {
                // SAFETY: both vessels are the graph's, the precondition of this function.
                unsafe {
                    graph.reverse_segment_unchecked(original_last, original_first);
                }
            }
            Self::RelocateAfterVessel {
                vessel,
                predecessor,
            } => {
                // SAFETY: both vessels are the graph's, the precondition of this function.
                unsafe {
                    graph.relocate_after_unchecked(vessel, predecessor);
                }
            }
            // SAFETY: the vessel and the berth are the graph's, the precondition of this function.
            Self::RelocateToHead { vessel, berth } => unsafe {
                graph.relocate_to_head_unchecked(vessel, berth);
            },
            Self::RelocateSegmentAfterVessel {
                segment_first,
                segment_last,
                predecessor,
            } => {
                // SAFETY: the three vessels are the graph's, the precondition of this function.
                unsafe {
                    graph.relocate_segment_after_unchecked(
                        segment_first,
                        segment_last,
                        predecessor,
                    );
                }
            }
            Self::RelocateSegmentToHead {
                segment_first,
                segment_last,
                berth,
            } => {
                // SAFETY: both vessels and the berth are the graph's, the precondition of this function.
                unsafe {
                    graph.relocate_segment_to_head_unchecked(segment_first, segment_last, berth);
                }
            }
        }
    }
}

impl std::fmt::Display for UndoInstruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SwapVessels { vessel_a, vessel_b } => {
                write!(f, "swap {vessel_a} and {vessel_b} back")
            }
            Self::SwapSegments {
                a_first,
                a_last,
                b_first,
                b_last,
            } => write!(
                f,
                "swap the segments {a_first} to {a_last} and {b_first} to {b_last} back"
            ),
            Self::ReverseSegment {
                original_first,
                original_last,
            } => write!(
                f,
                "reverse the segment {original_last} to {original_first} back"
            ),
            Self::RelocateAfterVessel {
                vessel,
                predecessor,
            } => {
                write!(f, "move {vessel} back after {predecessor}")
            }
            Self::RelocateToHead { vessel, berth } => {
                write!(f, "move {vessel} back to the head of {berth}")
            }
            Self::RelocateSegmentAfterVessel {
                segment_first,
                segment_last,
                predecessor,
            } => write!(
                f,
                "move the segment {segment_first} to {segment_last} back after {predecessor}"
            ),
            Self::RelocateSegmentToHead {
                segment_first,
                segment_last,
                berth,
            } => write!(
                f,
                "move the segment {segment_first} to {segment_last} back to the head of {berth}"
            ),
        }
    }
}

// ----------------------------------------------------------------
// ScheduleGraphUndoLog
// ----------------------------------------------------------------

/// How to undo a sequence of moves on a schedule graph: the inverse of each move, the latest last.
///
/// Every move of a schedule graph has an inverse that is itself a move, named by a few of the graph's ids; the log
/// records it when the move is made, and rolling back applies the inverses, the latest first. The
/// [`Mutator`](super::Mutator) writes the log as it makes its moves and is its only writer.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct ScheduleGraphUndoLog {
    /// The inverse of each move, in the order the moves were made.
    stack: Vec<UndoInstruction>,
}

impl ScheduleGraphUndoLog {
    /// Empties the log, keeping its memory.
    #[inline(always)]
    pub(super) fn clear(&mut self) {
        self.stack.clear();
    }

    /// Whether the log records no move.
    #[cfg(test)]
    pub(super) fn is_empty(&self) -> bool {
        self.stack.is_empty()
    }

    /// The number of moves the log records.
    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.stack.len()
    }

    // ----------------------------------------------------------------
    // Push methods
    // ----------------------------------------------------------------

    /// Records that `a` and `b` were swapped; the swap is its own inverse.
    #[inline(always)]
    pub(super) fn push_swap_vessels(&mut self, a: VesselIndex, b: VesselIndex) {
        self.stack.push(UndoInstruction::SwapVessels {
            vessel_a: a,
            vessel_b: b,
        });
    }

    /// Records that the segments `a_first ..= a_last` and `b_first ..= b_last` were swapped; swapping them back
    /// is the inverse.
    #[inline(always)]
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "a move of the schedule no neighbourhood makes yet; its tests keep it sound"
        )
    )]
    pub(super) fn push_swap_segments(
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

    /// Records that the segment `first ..= last` was reversed; reversing `last ..= first` is the inverse.
    #[inline(always)]
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "a move of the schedule no neighbourhood makes yet; its tests keep it sound"
        )
    )]
    pub(super) fn push_reverse_segment(&mut self, first: VesselIndex, last: VesselIndex) {
        self.stack.push(UndoInstruction::ReverseSegment {
            original_first: first,
            original_last: last,
        });
    }

    /// Records that `vessel`, which stood directly after `predecessor`, was moved.
    #[inline(always)]
    pub(super) fn push_relocate_after_vessel(
        &mut self,
        vessel: VesselIndex,
        predecessor: VesselIndex,
    ) {
        self.stack.push(UndoInstruction::RelocateAfterVessel {
            vessel,
            predecessor,
        });
    }

    /// Records that `vessel`, which headed `berth`, was moved.
    #[inline(always)]
    pub(super) fn push_relocate_to_head(&mut self, vessel: VesselIndex, berth: BerthIndex) {
        self.stack
            .push(UndoInstruction::RelocateToHead { vessel, berth });
    }

    /// Records that the segment `first ..= last`, which stood directly after `predecessor`, was moved.
    #[inline(always)]
    pub(super) fn push_relocate_segment_after_vessel(
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

    /// Records that the segment `first ..= last`, which headed `berth`, was moved.
    #[inline(always)]
    pub(super) fn push_relocate_segment_to_head(
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

    /// Undoes every recorded move on `graph`, the latest first, and empties the log.
    ///
    /// Afterwards `graph` stands as it stood before the first recorded move, if the log holds the moves made on
    /// it since then.
    ///
    /// # Safety
    ///
    /// Every vessel and berth the log names must be one of `graph`'s, as the [`Mutator`](super::Mutator) that
    /// writes the log over its own graph keeps.
    #[inline]
    pub(super) unsafe fn apply_rollback_unchecked(&mut self, graph: &mut ScheduleGraph) {
        while let Some(instruction) = self.stack.pop() {
            debug_assert!(
                instruction.outside(graph).is_none(),
                "called `ScheduleGraphUndoLog::apply_rollback_unchecked` with {instruction}, which names an id \
                 outside the graph"
            );
            // SAFETY: every id the log names is one of the graph's, the precondition of this function.
            unsafe { instruction.undo_unchecked(graph) };
        }
    }
}

impl std::fmt::Display for ScheduleGraphUndoLog {
    /// The instructions in the order a rollback applies them, the latest first, one per line.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "undo log of {} instructions", self.stack.len())?;
        for instruction in self.stack.iter().rev() {
            write!(f, "\n  {instruction}")?;
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
        let mut log = ScheduleGraphUndoLog::default();

        log.push_swap_vessels(v(0), v(2));
        graph.swap_vessels(v(0), v(2));

        let seq: Vec<_> = graph.vessel_sequence_iter(b(0)).collect();
        assert_eq!(seq, vec![v(2), v(1), v(0)]);

        // SAFETY: the log names only vessels and berths of the graph.
        unsafe { log.apply_rollback_unchecked(&mut graph) };

        let restored: Vec<_> = graph.vessel_sequence_iter(b(0)).collect();
        assert_eq!(restored, vec![v(0), v(1), v(2)]);
        assert!(log.is_empty());
    }

    #[test]
    fn test_rollback_swap_segments() {
        let mut graph = setup_graph();
        let mut log = ScheduleGraphUndoLog::default();

        log.push_swap_segments(v(0), v(1), v(3), v(4));
        graph.swap_segments(v(0), v(1), v(3), v(4));

        let seq: Vec<_> = graph.vessel_sequence_iter(b(0)).collect();
        assert_eq!(seq, vec![v(3), v(4), v(2)]);

        // SAFETY: the log names only vessels and berths of the graph.
        unsafe { log.apply_rollback_unchecked(&mut graph) };

        let restored: Vec<_> = graph.vessel_sequence_iter(b(0)).collect();
        assert_eq!(restored, vec![v(0), v(1), v(2)]);
    }

    #[test]
    fn test_rollback_reverse_segment() {
        let mut graph = setup_graph();
        let mut log = ScheduleGraphUndoLog::default();

        log.push_reverse_segment(v(0), v(2));
        graph.reverse_segment(v(0), v(2));

        let seq: Vec<_> = graph.vessel_sequence_iter(b(0)).collect();
        assert_eq!(seq, vec![v(2), v(1), v(0)]);

        // SAFETY: the log names only vessels and berths of the graph.
        unsafe { log.apply_rollback_unchecked(&mut graph) };

        let restored: Vec<_> = graph.vessel_sequence_iter(b(0)).collect();
        assert_eq!(restored, vec![v(0), v(1), v(2)]);
    }

    #[test]
    fn test_rollback_relocate_after_vessel() {
        let mut graph = setup_graph();
        let mut log = ScheduleGraphUndoLog::default();

        // V0 is at head of B0, predecessor is sentinel -> use RelocateToHead
        // But first test the AfterVessel variant: move V2 (predecessor is V1)
        log.push_relocate_after_vessel(v(2), v(1));
        graph.relocate_after(v(2), v(0));

        let seq: Vec<_> = graph.vessel_sequence_iter(b(0)).collect();
        assert_eq!(seq, vec![v(0), v(2), v(1)]);

        // SAFETY: the log names only vessels and berths of the graph.
        unsafe { log.apply_rollback_unchecked(&mut graph) };

        let restored: Vec<_> = graph.vessel_sequence_iter(b(0)).collect();
        assert_eq!(restored, vec![v(0), v(1), v(2)]);
    }

    #[test]
    fn test_rollback_relocate_to_head() {
        let mut graph = setup_graph();
        let mut log = ScheduleGraphUndoLog::default();

        // V0 is at head of B0. Move it after V3 in B1.
        log.push_relocate_to_head(v(0), b(0));
        graph.relocate_after(v(0), v(3));

        let seq_b0: Vec<_> = graph.vessel_sequence_iter(b(0)).collect();
        let seq_b1: Vec<_> = graph.vessel_sequence_iter(b(1)).collect();
        assert_eq!(seq_b0, vec![v(1), v(2)]);
        assert_eq!(seq_b1, vec![v(3), v(0), v(4)]);

        // SAFETY: the log names only vessels and berths of the graph.
        unsafe { log.apply_rollback_unchecked(&mut graph) };

        let restored_b0: Vec<_> = graph.vessel_sequence_iter(b(0)).collect();
        let restored_b1: Vec<_> = graph.vessel_sequence_iter(b(1)).collect();
        assert_eq!(restored_b0, vec![v(0), v(1), v(2)]);
        assert_eq!(restored_b1, vec![v(3), v(4)]);
    }

    #[test]
    fn test_rollback_relocate_segment_after_vessel() {
        let mut graph = setup_graph();
        let mut log = ScheduleGraphUndoLog::default();

        // Move [V1, V2] (predecessor V0) to after V3 in B1.
        log.push_relocate_segment_after_vessel(v(1), v(2), v(0));
        graph.relocate_segment_after(v(1), v(2), v(3));

        let seq_b0: Vec<_> = graph.vessel_sequence_iter(b(0)).collect();
        let seq_b1: Vec<_> = graph.vessel_sequence_iter(b(1)).collect();
        assert_eq!(seq_b0, vec![v(0)]);
        assert_eq!(seq_b1, vec![v(3), v(1), v(2), v(4)]);

        // SAFETY: the log names only vessels and berths of the graph.
        unsafe { log.apply_rollback_unchecked(&mut graph) };

        let restored_b0: Vec<_> = graph.vessel_sequence_iter(b(0)).collect();
        let restored_b1: Vec<_> = graph.vessel_sequence_iter(b(1)).collect();
        assert_eq!(restored_b0, vec![v(0), v(1), v(2)]);
        assert_eq!(restored_b1, vec![v(3), v(4)]);
    }

    #[test]
    fn test_rollback_relocate_segment_to_head() {
        let mut graph = setup_graph();
        let mut log = ScheduleGraphUndoLog::default();

        // Move [V0, V1] (at head of B0) to head of B2.
        log.push_relocate_segment_to_head(v(0), v(1), b(0));
        graph.relocate_segment_to_head(v(0), v(1), b(2));

        let seq_b0: Vec<_> = graph.vessel_sequence_iter(b(0)).collect();
        let seq_b2: Vec<_> = graph.vessel_sequence_iter(b(2)).collect();
        assert_eq!(seq_b0, vec![v(2)]);
        assert_eq!(seq_b2, vec![v(0), v(1)]);

        // SAFETY: the log names only vessels and berths of the graph.
        unsafe { log.apply_rollback_unchecked(&mut graph) };

        let restored_b0: Vec<_> = graph.vessel_sequence_iter(b(0)).collect();
        let restored_b2: Vec<_> = graph.vessel_sequence_iter(b(2)).collect();
        assert_eq!(restored_b0, vec![v(0), v(1), v(2)]);
        assert!(restored_b2.is_empty());
    }

    #[test]
    fn test_lifo_execution_order() {
        let mut graph = setup_graph();
        let original = graph.clone();
        let mut log = ScheduleGraphUndoLog::default();

        // Op 1: Move V2 to head of empty B2 (V2 was after V1)
        log.push_relocate_after_vessel(v(2), v(1));
        graph.relocate_to_head(v(2), b(2));

        // Op 2: Reverse [V3, V4] in B1
        log.push_reverse_segment(v(3), v(4));
        graph.reverse_segment(v(3), v(4));

        // Op 3: Swap V0 and V3
        log.push_swap_vessels(v(0), v(3));
        graph.swap_vessels(v(0), v(3));

        assert_ne!(graph, original);

        // SAFETY: the log names only vessels and berths of the graph.
        unsafe { log.apply_rollback_unchecked(&mut graph) };

        assert_eq!(graph, original);
        assert!(log.is_empty());
    }

    #[test]
    fn test_clear_keeps_the_memory() {
        let mut log = ScheduleGraphUndoLog::default();
        log.push_swap_vessels(v(0), v(1));
        log.push_reverse_segment(v(2), v(3));
        assert_eq!(log.len(), 2);
        let capacity = log.stack.capacity();
        log.clear();
        assert!(log.is_empty());
        assert_eq!(log.stack.capacity(), capacity);
    }

    #[test]
    fn test_the_log_keeps_the_moves_in_their_order() {
        let mut log = ScheduleGraphUndoLog::default();
        log.push_swap_vessels(v(0), v(1));
        log.push_relocate_to_head(v(2), b(1));

        let ops = &log.stack;
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

    #[test]
    fn test_display_lists_the_instructions_latest_first() {
        let mut log = ScheduleGraphUndoLog::default();
        assert_eq!(log.to_string(), "undo log of 0 instructions");
        log.push_swap_vessels(v(0), v(1));
        log.push_relocate_to_head(v(2), b(1));
        assert_eq!(
            log.to_string(),
            "undo log of 2 instructions\n  move Vessel 2 back to the head of Berth 1\n  swap Vessel 0 and Vessel 1 back"
        );
    }
}
