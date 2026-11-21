// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The neighbourhoods that move one vessel to right after another: on its own berth, or onto another.

use super::filter::{inter_berth_shift_filter_unchecked, intra_berth_shift_filter_unchecked};
use crate::schedule::{Mutator, ScheduleGraph};
use std::iter::FusedIterator;
use talos_core::SolverNumeric;
use talos_model::{Model, VesselIndex};

/// The placements of a graph of `num_vessels` vessels: each vessel paired with every other vessel as the anchor it
/// would follow, in increasing order of the vessel and then of the anchor.
#[derive(Debug, Clone)]
struct PlacementsIter {
    num_vessels: usize,
    /// The vessel of the next placement, or `num_vessels` once every placement is given.
    vessel: usize,
    /// The anchor of the next placement, or `num_vessels` once the placements of `vessel` are given.
    anchor: usize,
}

impl PlacementsIter {
    /// The placements of a graph of `num_vessels` vessels, from the first.
    fn new(num_vessels: usize) -> Self {
        Self {
            num_vessels,
            vessel: 0,
            anchor: 0,
        }
    }

    /// Starts the placements again from the first, for the same count of vessels.
    fn restart(&mut self) {
        self.vessel = 0;
        self.anchor = 0;
    }
}

impl Iterator for PlacementsIter {
    type Item = (VesselIndex, VesselIndex);

    fn next(&mut self) -> Option<Self::Item> {
        // Each pass either gives a placement or steps past the vessel itself or to the next vessel, so the loop
        // ends within `num_vessels + 2` passes.
        for _ in 0..self.num_vessels + 2 {
            if self.vessel >= self.num_vessels {
                return None;
            }
            if self.anchor >= self.num_vessels {
                self.vessel += 1;
                self.anchor = 0;
                continue;
            }
            let placement = (VesselIndex::new(self.vessel), VesselIndex::new(self.anchor));
            self.anchor += 1;
            if placement.0 != placement.1 {
                return Some(placement);
            }
        }
        unreachable!("a placement or the end should be found within `num_vessels + 2` passes")
    }
}

impl FusedIterator for PlacementsIter {}

/// The moves of a vessel to right after another vessel of its berth, where it does not already stand and where
/// their windows overlap, in the order of the placements.
#[derive(Debug, Clone)]
pub(super) struct IntraBerthShiftOperator {
    placements: PlacementsIter,
}

impl IntraBerthShiftOperator {
    /// The operator, which proposes no move until it is prepared for a graph.
    pub(super) fn new() -> Self {
        Self {
            placements: PlacementsIter::new(0),
        }
    }

    /// Starts the neighbourhood of `graph` from its first move.
    pub(super) fn prepare(&mut self, graph: &ScheduleGraph) {
        self.placements = PlacementsIter::new(graph.num_vessels());
    }

    /// Starts the neighbourhood again from its first move, for a graph of the same vessels.
    pub(super) fn reset(&mut self) {
        self.placements.restart();
    }

    /// Makes the next move of the neighbourhood on the graph of `mutator` and answers `true`, or answers `false`
    /// and changes nothing once the neighbourhood is exhausted.
    ///
    /// # Safety
    ///
    /// The graph of `mutator` must count the vessels of the graph the operator was last prepared for, and have the
    /// vessels and berths of `model`.
    pub(super) unsafe fn next_neighbor_unchecked<T>(
        &mut self,
        model: &Model<T>,
        mutator: &mut Mutator,
    ) -> bool
    where
        T: SolverNumeric,
    {
        for (vessel, anchor) in &mut self.placements {
            let graph = mutator.graph();
            debug_assert!(
                vessel.index() < graph.num_vessels() && anchor.index() < graph.num_vessels(),
                "called `IntraBerthShiftOperator::next_neighbor_unchecked` with a graph of fewer vessels than \
                 prepared for"
            );
            // SAFETY: the placements count the vessels of the graph, the precondition of this function, so both
            // vessels are the graph's.
            let (same_berth, already_there) = unsafe {
                (
                    graph.vessel_berth_unchecked(vessel) == graph.vessel_berth_unchecked(anchor),
                    graph.vessel_predecessor_unchecked(vessel) == Some(anchor),
                )
            };
            // SAFETY: both vessels are the graph's, whose vessels are the model's, the precondition of this
            // function.
            if same_berth
                && !already_there
                && unsafe { intra_berth_shift_filter_unchecked(vessel, anchor, model) }
            {
                debug_assert!(vessel.index() < mutator.graph().num_vessels());
                // SAFETY: both vessels are the graph's, as above.
                unsafe { mutator.relocate_after_unchecked(vessel, anchor) };
                return true;
            }
        }
        false
    }
}

/// The moves of a vessel to right after a vessel of another berth that it may use, in the order of the
/// placements.
#[derive(Debug, Clone)]
pub(super) struct InterBerthShiftOperator {
    placements: PlacementsIter,
}

impl InterBerthShiftOperator {
    /// The operator, which proposes no move until it is prepared for a graph.
    pub(super) fn new() -> Self {
        Self {
            placements: PlacementsIter::new(0),
        }
    }

    /// Starts the neighbourhood of `graph` from its first move.
    pub(super) fn prepare(&mut self, graph: &ScheduleGraph) {
        self.placements = PlacementsIter::new(graph.num_vessels());
    }

    /// Starts the neighbourhood again from its first move, for a graph of the same vessels.
    pub(super) fn reset(&mut self) {
        self.placements.restart();
    }

    /// Makes the next move of the neighbourhood on the graph of `mutator` and answers `true`, or answers `false`
    /// and changes nothing once the neighbourhood is exhausted.
    ///
    /// # Safety
    ///
    /// The graph of `mutator` must count the vessels of the graph the operator was last prepared for, and have the
    /// vessels and berths of `model`.
    pub(super) unsafe fn next_neighbor_unchecked<T>(
        &mut self,
        model: &Model<T>,
        mutator: &mut Mutator,
    ) -> bool
    where
        T: SolverNumeric,
    {
        for (vessel, anchor) in &mut self.placements {
            let graph = mutator.graph();
            debug_assert!(
                vessel.index() < graph.num_vessels() && anchor.index() < graph.num_vessels(),
                "called `InterBerthShiftOperator::next_neighbor_unchecked` with a graph of fewer vessels than \
                 prepared for"
            );
            // SAFETY: the placements count the vessels of the graph, the precondition of this function, so both
            // vessels are the graph's.
            let other_berth = unsafe {
                graph.vessel_berth_unchecked(vessel) != graph.vessel_berth_unchecked(anchor)
            };
            // SAFETY: both vessels are the graph's, which has the vessels and berths of the model, the
            // precondition of this function.
            if other_berth
                && unsafe { inter_berth_shift_filter_unchecked(vessel, anchor, graph, model) }
            {
                debug_assert!(vessel.index() < mutator.graph().num_vessels());
                // SAFETY: both vessels are the graph's, as above.
                unsafe { mutator.relocate_after_unchecked(vessel, anchor) };
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_placements_pair_each_vessel_with_every_other_in_order() {
        let placements: Vec<(usize, usize)> = PlacementsIter::new(3)
            .map(|(vessel, anchor)| (vessel.index(), anchor.index()))
            .collect();
        assert_eq!(placements, [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]);
    }

    #[test]
    fn test_placements_of_fewer_than_two_vessels_are_none() {
        assert_eq!(PlacementsIter::new(0).next(), None);
        assert_eq!(PlacementsIter::new(1).next(), None);
    }

    #[test]
    fn test_placements_stay_exhausted_and_restart_from_the_first() {
        let mut placements = PlacementsIter::new(2);
        assert_eq!(placements.by_ref().count(), 2);
        assert_eq!(placements.next(), None);
        placements.restart();
        assert_eq!(
            placements.next(),
            Some((VesselIndex::new(0), VesselIndex::new(1)))
        );
    }
}
