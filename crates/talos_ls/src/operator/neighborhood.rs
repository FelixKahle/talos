// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The neighbourhoods a search explores, and the operator the engine draws its candidate moves from.

use super::{
    shift::{InterBerthShiftOperator, IntraBerthShiftOperator},
    swap::{InterBerthSwapOperator, IntraBerthSwapOperator},
};
use crate::schedule::{Mutator, ScheduleGraph};
use talos_core::SolverNumeric;
use talos_model::Model;

/// A kind of move from one schedule to its neighbours.
///
/// Each neighbourhood proposes its moves in a fixed order and skips a move its filter refuses: where the instance
/// rules it out, and on one berth wherever the move sets a vessel next to one whose window from arrival to latest
/// departure does not overlap its own. For a swap and a move to the head of a berth that refuses only infeasible
/// moves, unless both vessels take no time to serve; for a move after another vessel it may skip a move an optimum
/// needs.
/// Together the two shifts insert a vessel at every place of every berth: after any vessel, at the head of any
/// berth, and onto an empty berth.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Neighborhood {
    /// Exchange two vessels on one berth whose windows from arrival to latest departure overlap.
    IntraBerthSwap,
    /// Exchange two vessels on different berths that may each use the other's berth.
    InterBerthSwap,
    /// Move a vessel to another place of its berth: right after another vessel whose window overlaps its own, or
    /// to the head of the berth before a vessel whose window overlaps its own.
    IntraBerthShift,
    /// Move a vessel onto another berth that it may use: right after one of its vessels, or to its head, an empty
    /// berth included.
    InterBerthShift,
}

impl Neighborhood {
    /// Every neighbourhood, in the order of the variants.
    pub const ALL: [Self; 4] = [
        Self::IntraBerthSwap,
        Self::InterBerthSwap,
        Self::IntraBerthShift,
        Self::InterBerthShift,
    ];
}

/// The source of the candidate moves of a run: it makes one move on the graph at each call, is prepared for each
/// scan of the neighbourhood of the accepted schedule, and is told which of its moves the run takes.
pub trait LocalSearchOperator<T>
where
    T: SolverNumeric,
{
    /// Starts a scan of the neighbourhood of the accepted schedule, whose objective is `accepted_objective` and
    /// whose order is `graph`.
    fn prepare(&mut self, accepted_objective: T, graph: &ScheduleGraph);

    /// Makes the next move of the scan on the graph of `mutator` and answers `true`, or answers `false` and
    /// changes nothing once the scan is exhausted.
    ///
    /// # Safety
    ///
    /// The graph of `mutator` must count the vessels and berths of the graph the operator was last prepared for,
    /// and have the vessels and berths of `model`.
    unsafe fn next_neighbor_unchecked(&mut self, model: &Model<T>, mutator: &mut Mutator) -> bool;

    /// Takes note that the move it made last was kept aside as the best candidate of the scan.
    fn on_kept_aside(&mut self) {}

    /// Takes note that `accepted`, a move of the current scan, made the accepted schedule of objective
    /// `accepted_objective`.
    fn on_accepted(&mut self, _accepted: AcceptedMove, _accepted_objective: T) {}
}

/// Which move of a scan the run takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AcceptedMove {
    /// The move the operator made last.
    Last,
    /// The move it made earlier in the scan and that was kept aside.
    KeptAside,
}

/// The scan of one neighbourhood, with its place in the scan.
#[derive(Debug, Clone)]
pub(super) enum NeighborhoodOperator {
    IntraBerthSwap(IntraBerthSwapOperator),
    InterBerthSwap(InterBerthSwapOperator),
    IntraBerthShift(IntraBerthShiftOperator),
    InterBerthShift(InterBerthShiftOperator),
}

impl NeighborhoodOperator {
    /// The scan of `neighborhood`, which proposes no move until it is prepared for a graph.
    pub(super) fn new(neighborhood: Neighborhood) -> Self {
        match neighborhood {
            Neighborhood::IntraBerthSwap => Self::IntraBerthSwap(IntraBerthSwapOperator::new()),
            Neighborhood::InterBerthSwap => Self::InterBerthSwap(InterBerthSwapOperator::new()),
            Neighborhood::IntraBerthShift => Self::IntraBerthShift(IntraBerthShiftOperator::new()),
            Neighborhood::InterBerthShift => Self::InterBerthShift(InterBerthShiftOperator::new()),
        }
    }

    /// Starts the scan of the neighbourhood of `graph` from its first move.
    pub(super) fn prepare(&mut self, graph: &ScheduleGraph) {
        match self {
            Self::IntraBerthSwap(operator) => operator.prepare(graph),
            Self::InterBerthSwap(operator) => operator.prepare(graph),
            Self::IntraBerthShift(operator) => operator.prepare(graph),
            Self::InterBerthShift(operator) => operator.prepare(graph),
        }
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
        debug_assert!(
            mutator.graph().num_vessels() == model.num_vessels()
                && mutator.graph().num_berths() == model.num_berths(),
            "called `NeighborhoodOperator::next_neighbor_unchecked` with a graph and a model of different counts"
        );
        match self {
            Self::IntraBerthSwap(operator) => {
                // SAFETY: the precondition of this function is the scan's.
                unsafe { operator.next_neighbor_unchecked(model, mutator) }
            }
            Self::InterBerthSwap(operator) => {
                // SAFETY: the precondition of this function is the scan's.
                unsafe { operator.next_neighbor_unchecked(model, mutator) }
            }
            Self::IntraBerthShift(operator) => {
                // SAFETY: the precondition of this function is the scan's.
                unsafe { operator.next_neighbor_unchecked(model, mutator) }
            }
            Self::InterBerthShift(operator) => {
                // SAFETY: the precondition of this function is the scan's.
                unsafe { operator.next_neighbor_unchecked(model, mutator) }
            }
        }
    }
}
