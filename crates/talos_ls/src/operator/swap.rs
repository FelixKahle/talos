// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The neighbourhoods that exchange the places of two vessels: on one berth, or across two.

use super::filter::{inter_berth_swap_filter_unchecked, intra_berth_swap_filter_unchecked};
use crate::schedule::{Mutator, ScheduleGraph};
use std::iter::FusedIterator;
use talos_core::SolverNumeric;
use talos_model::{Model, VesselIndex};

/// The pairs of distinct vessels of a graph of `num_vessels` vessels, each pair once with the lower vessel first,
/// in increasing order of the first vessel and then of the second.
#[derive(Debug, Clone)]
struct VesselPairsIter {
    /// The vessels of the graph.
    num_vessels: usize,
    /// The first vessel of the next pair.
    first: usize,
    /// The second vessel of the next pair, or `num_vessels` once the pairs of `first` are given.
    second: usize,
}

impl VesselPairsIter {
    /// The pairs of a graph of `num_vessels` vessels, from the first.
    fn new(num_vessels: usize) -> Self {
        Self {
            num_vessels,
            first: 0,
            second: 1,
        }
    }
}

impl Iterator for VesselPairsIter {
    type Item = (VesselIndex, VesselIndex);

    fn next(&mut self) -> Option<Self::Item> {
        if self.second >= self.num_vessels {
            self.first += 1;
            self.second = self.first + 1;
        }
        if self.first + 1 >= self.num_vessels {
            // Every pair is given; staying here keeps the iterator fused.
            self.first = self.num_vessels.saturating_sub(1);
            self.second = self.num_vessels;
            return None;
        }
        let pair = (VesselIndex::new(self.first), VesselIndex::new(self.second));
        self.second += 1;
        Some(pair)
    }
}

impl FusedIterator for VesselPairsIter {}

/// The swaps of two vessels on one berth whose windows overlap, in the order of the vessel pairs.
#[derive(Debug, Clone)]
pub(super) struct IntraBerthSwapOperator {
    pairs: VesselPairsIter,
}

impl IntraBerthSwapOperator {
    /// The operator, which proposes no move until it is prepared for a graph.
    pub(super) fn new() -> Self {
        Self {
            pairs: VesselPairsIter::new(0),
        }
    }

    /// Starts the neighbourhood of `graph` from its first move.
    pub(super) fn prepare(&mut self, graph: &ScheduleGraph) {
        self.pairs = VesselPairsIter::new(graph.num_vessels());
    }

    /// Makes the next swap of the neighbourhood on the graph of `mutator` and answers `true`, or answers `false`
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
        for (first, second) in &mut self.pairs {
            let graph = mutator.graph();
            debug_assert!(
                first.index() < graph.num_vessels() && second.index() < graph.num_vessels(),
                "called `IntraBerthSwapOperator::next_neighbor_unchecked` with a graph of fewer vessels than \
                 prepared for"
            );
            // SAFETY: the pairs count the vessels of the graph, the precondition of this function, so both
            // vessels are the graph's.
            let same_berth = unsafe {
                graph.vessel_berth_unchecked(first) == graph.vessel_berth_unchecked(second)
            };
            if !same_berth {
                continue;
            }
            debug_assert!(
                first.index() < model.num_vessels() && second.index() < model.num_vessels()
            );
            // SAFETY: both vessels are the graph's, whose vessels are the model's, the precondition of this
            // function.
            if unsafe { intra_berth_swap_filter_unchecked(first, second, model) } {
                debug_assert!(
                    first.index() < mutator.graph().num_vessels()
                        && second.index() < mutator.graph().num_vessels()
                );
                // SAFETY: the pairs count the vessels of the graph, the precondition of this function, so both
                // vessels are the graph's.
                unsafe { mutator.swap_vessels_unchecked(first, second) };
                return true;
            }
        }
        false
    }
}

/// The swaps of two vessels on different berths that may each use the other's berth, in the order of the vessel
/// pairs.
#[derive(Debug, Clone)]
pub(super) struct InterBerthSwapOperator {
    pairs: VesselPairsIter,
}

impl InterBerthSwapOperator {
    /// The operator, which proposes no move until it is prepared for a graph.
    pub(super) fn new() -> Self {
        Self {
            pairs: VesselPairsIter::new(0),
        }
    }

    /// Starts the neighbourhood of `graph` from its first move.
    pub(super) fn prepare(&mut self, graph: &ScheduleGraph) {
        self.pairs = VesselPairsIter::new(graph.num_vessels());
    }

    /// Makes the next swap of the neighbourhood on the graph of `mutator` and answers `true`, or answers `false`
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
        for (first, second) in &mut self.pairs {
            let graph = mutator.graph();
            debug_assert!(
                first.index() < graph.num_vessels() && second.index() < graph.num_vessels(),
                "called `InterBerthSwapOperator::next_neighbor_unchecked` with a graph of fewer vessels than \
                 prepared for"
            );
            // SAFETY: the pairs count the vessels of the graph, the precondition of this function, so both
            // vessels are the graph's.
            let other_berth = unsafe {
                graph.vessel_berth_unchecked(first) != graph.vessel_berth_unchecked(second)
            };
            if !other_berth {
                continue;
            }
            debug_assert!(
                graph.num_vessels() == model.num_vessels()
                    && graph.num_berths() == model.num_berths()
            );
            // SAFETY: both vessels are the graph's, which has the vessels and berths of the model, the
            // precondition of this function.
            if unsafe { inter_berth_swap_filter_unchecked(first, second, graph, model) } {
                debug_assert!(
                    first.index() < mutator.graph().num_vessels()
                        && second.index() < mutator.graph().num_vessels()
                );
                // SAFETY: the pairs count the vessels of the graph, the precondition of this function, so both
                // vessels are the graph's.
                unsafe { mutator.swap_vessels_unchecked(first, second) };
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
    fn test_vessel_pairs_give_each_pair_once_in_order() {
        let pairs: Vec<(usize, usize)> = VesselPairsIter::new(4)
            .map(|(first, second)| (first.index(), second.index()))
            .collect();
        assert_eq!(pairs, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]);
    }

    #[test]
    fn test_vessel_pairs_of_fewer_than_two_vessels_are_none() {
        assert_eq!(VesselPairsIter::new(0).next(), None);
        assert_eq!(VesselPairsIter::new(1).next(), None);
    }

    #[test]
    fn test_vessel_pairs_stay_exhausted() {
        let mut pairs = VesselPairsIter::new(3);
        assert_eq!(pairs.by_ref().count(), 3);
        assert_eq!(pairs.next(), None);
        assert_eq!(pairs.next(), None);
    }
}
