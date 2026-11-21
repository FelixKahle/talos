// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The scan of a neighbourhood: every vessel in increasing order, each with the moves it has, gathered when its
//! turn comes.

use crate::schedule::{Mutator, ScheduleGraph};
use talos_core::SolverNumeric;
use talos_model::{Model, VesselIndex, VesselIndexIter};

/// The moves of one kind that a vessel has in a schedule: where each takes the vessel, and how it is made.
pub(super) trait VesselMoves {
    /// Where a move takes the vessel: the vessel it is exchanged with, or the place it is inserted at.
    type Target: Copy + std::fmt::Debug;

    /// Writes into `targets` the moves of `vessel` that the neighbourhood proposes in `graph`, in its order,
    /// after what `targets` held.
    ///
    /// # Safety
    ///
    /// `vessel` must be the graph's, and the graph must have the vessels and berths of `model`.
    unsafe fn gather_unchecked<T>(
        vessel: VesselIndex,
        graph: &ScheduleGraph,
        model: &Model<T>,
        targets: &mut Vec<Self::Target>,
    ) where
        T: SolverNumeric;

    /// Makes the move of `vessel` to `target` on the graph of `mutator`.
    ///
    /// # Safety
    ///
    /// `vessel` and every vessel and berth `target` names must be the graph's.
    unsafe fn make_unchecked(mutator: &mut Mutator, vessel: VesselIndex, target: Self::Target);
}

/// The scan of the neighbourhood of `K`: every vessel in increasing order, and for each the moves `K` proposes, in
/// its order, gathered from the graph when the vessel's turn comes.
///
/// The moves of a vessel are those of the graph as it stood when they were gathered, so a caller undoes each move
/// before it asks for the next, and prepares the scan anew once it keeps one.
// Invariant: `targets` holds the moves of `vessel`, each naming only vessels and berths of the graph they were
// gathered from, whose counts are those the scan was prepared for; it is empty until the first vessel's turn.
#[derive(Debug, Clone)]
pub(super) struct VesselScan<K>
where
    K: VesselMoves,
{
    /// The vessels whose turn is still to come.
    vessels: VesselIndexIter,
    /// The vessel whose moves `targets` holds.
    vessel: VesselIndex,
    /// The moves of `vessel`, in the order the scan proposes them.
    targets: Vec<K::Target>,
    /// The place in `targets` of the next move to make.
    next_target: usize,
}

impl<K> VesselScan<K>
where
    K: VesselMoves,
{
    /// The scan, which proposes no move until it is prepared for a graph.
    pub(super) fn new() -> Self {
        Self {
            vessels: VesselIndexIter::new(0),
            vessel: VesselIndex::new(0),
            targets: Vec::new(),
            next_target: 0,
        }
    }

    /// Starts the scan of the neighbourhood of `graph` from its first move.
    pub(super) fn prepare(&mut self, graph: &ScheduleGraph) {
        self.vessels = graph.vessels();
        self.targets.clear();
        // A vessel has at most a move per other vessel and per berth, so the scan gathers without allocating.
        self.targets
            .reserve(graph.num_vessels() + graph.num_berths());
        self.next_target = 0;
    }

    /// Makes the next move of the scan on the graph of `mutator` and answers `true`, or answers `false` and
    /// changes nothing once the scan is exhausted.
    ///
    /// # Safety
    ///
    /// The graph of `mutator` must count the vessels and berths of the graph the scan was last prepared for, and
    /// have the vessels and berths of `model`.
    #[inline]
    pub(super) unsafe fn next_neighbor_unchecked<T>(
        &mut self,
        model: &Model<T>,
        mutator: &mut Mutator,
    ) -> bool
    where
        T: SolverNumeric,
    {
        // Each pass makes a move or gives the next vessel its turn, so the loop ends within one pass per vessel
        // after the last move.
        loop {
            if let Some(&target) = self.targets.get(self.next_target) {
                self.next_target += 1;
                debug_assert!(
                    self.vessel.index() < mutator.graph().num_vessels(),
                    "called `VesselScan::next_neighbor_unchecked` with a graph of fewer vessels than prepared for"
                );
                // SAFETY: the vessel and its targets were read from a graph of the counts the scan was prepared for
                // (`VesselScan`), which are those of the graph, the precondition of this function.
                unsafe { K::make_unchecked(mutator, self.vessel, target) };
                return true;
            }
            let Some(vessel) = self.vessels.next() else {
                return false;
            };
            self.vessel = vessel;
            self.targets.clear();
            self.next_target = 0;
            debug_assert!(
                vessel.index() < mutator.graph().num_vessels()
                    && mutator.graph().num_vessels() == model.num_vessels()
                    && mutator.graph().num_berths() == model.num_berths(),
                "called `VesselScan::next_neighbor_unchecked` with a graph of fewer vessels than prepared for, or \
                 one whose counts are not the model's"
            );
            // SAFETY: the vessels counted at the preparation are the graph's, whose vessels and berths are the
            // model's, the precondition of this function.
            unsafe { K::gather_unchecked(vessel, mutator.graph(), model, &mut self.targets) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::neighborhoods_on_lists::{self, WINDOWS};
    use super::super::shift::InterBerthShifts;
    use super::*;

    // A scan prepared anew starts again from its first move, whatever its cursor had reached.
    #[test]
    fn test_prepare_restarts_the_scan_from_its_first_move() {
        let model = neighborhoods_on_lists::model(&WINDOWS);
        let mut scan = VesselScan::<InterBerthShifts>::new();
        scan.prepare(&neighborhoods_on_lists::start_graph());
        let mut mutator = Mutator::new(neighborhoods_on_lists::start_graph());
        let mut orders = Vec::new();
        for _ in 0..8 {
            // SAFETY: the mutator's graph, rolled back after each move, has the counts the scan was prepared for,
            // which are the model's.
            assert!(unsafe { scan.next_neighbor_unchecked(&model, &mut mutator) });
            orders.push(neighborhoods_on_lists::order_of(mutator.graph()));
            mutator.rollback();
        }
        scan.prepare(&neighborhoods_on_lists::start_graph());
        // SAFETY: the mutator's graph, rolled back to the start, has the counts the scan was prepared for anew,
        // which are the model's.
        assert!(unsafe { scan.next_neighbor_unchecked(&model, &mut mutator) });
        assert_eq!(neighborhoods_on_lists::order_of(mutator.graph()), orders[0]);
        assert_ne!(orders[0], orders[7]);
    }

    #[test]
    fn test_a_new_scan_proposes_nothing_until_it_is_prepared() {
        let model = neighborhoods_on_lists::model(&WINDOWS);
        let mut mutator = Mutator::new(neighborhoods_on_lists::start_graph());
        // SAFETY: a scan that was never prepared counts no vessel, so it reads nothing of the graph.
        let proposed = unsafe {
            VesselScan::<InterBerthShifts>::new().next_neighbor_unchecked(&model, &mut mutator)
        };
        assert!(!proposed);
        assert_eq!(mutator.graph(), &neighborhoods_on_lists::start_graph());
    }

    // An exhausted scan stays exhausted and leaves the graph as it was.
    #[test]
    fn test_an_exhausted_scan_proposes_nothing_more() {
        let model = neighborhoods_on_lists::model(&WINDOWS);
        let mut scan = VesselScan::<InterBerthShifts>::new();
        scan.prepare(&neighborhoods_on_lists::start_graph());
        let mut mutator = Mutator::new(neighborhoods_on_lists::start_graph());
        let mut moves = 0;
        // SAFETY: the mutator's graph, rolled back after each move, has the counts the scan was prepared for,
        // which are the model's.
        while moves <= 42 && unsafe { scan.next_neighbor_unchecked(&model, &mut mutator) } {
            mutator.rollback();
            moves += 1;
        }
        assert_eq!(moves, 42);
        for _ in 0..2 {
            // SAFETY: as in the loop, the graph stands at the start, of the counts prepared for.
            assert!(!unsafe { scan.next_neighbor_unchecked(&model, &mut mutator) });
        }
        assert_eq!(mutator.graph(), &neighborhoods_on_lists::start_graph());
    }
}
