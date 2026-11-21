// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The metaheuristics a run may follow, and the rule through which the engine asks one about each candidate.

use super::{greedy_descent::GreedyDescent, guided_local_search::GuidedLocalSearch};
use crate::schedule::{ScheduleGraph, ScheduleGraphDiff};
use talos_core::SolverNumeric;

/// The strategy that decides which candidates a run accepts and when it ends.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Metaheuristic {
    /// Accept only improving moves, and end at the first local optimum.
    GreedyDescent(GreedyDescent),
    /// Accept moves that lower the objective augmented by penalties, and penalize the features of the schedule
    /// where the search gets stuck.
    GuidedLocalSearch(GuidedLocalSearch),
}

impl Default for Metaheuristic {
    /// A guided local search with its default settings.
    fn default() -> Self {
        Self::GuidedLocalSearch(GuidedLocalSearch::default())
    }
}

/// What the engine does with a candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AcceptanceOutcome {
    /// The candidate becomes the accepted schedule, and the scan ends.
    Accept,
    /// The candidate is kept aside as the best of the scan so far, and the scan goes on from the accepted schedule.
    Buffer,
    /// The candidate is undone, and the scan goes on.
    Reject,
}

/// What the engine does once a scan of the neighbourhood is exhausted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NeighborhoodExhaustionOutcome {
    /// Scan the neighbourhood of the accepted schedule again.
    Restart,
    /// End the run.
    Terminate,
}

/// The decisions of a metaheuristic over one run, which the engine asks at each step.
///
/// The engine calls `on_start` first, with the graph it then searches; every later graph and diff it passes is of
/// that graph, whose counts of vessels and berths never change during the run.
pub trait AcceptanceRule<T>
where
    T: SolverNumeric,
{
    /// Starts a run from the schedule of `graph`, whose objective is `initial_objective`.
    fn on_start(&mut self, graph: &ScheduleGraph, initial_objective: T);

    /// The fate of a candidate of objective `candidate_objective`, which the moves of `diff` made from the accepted
    /// schedule of objective `accepted_objective`; `buffered_objective` is that of the candidate kept aside in this
    /// scan, if any.
    ///
    /// # Safety
    ///
    /// Every vessel and berth `diff` names must be of the graph `on_start` was given.
    unsafe fn decide_unchecked(
        &mut self,
        accepted_objective: T,
        buffered_objective: Option<T>,
        candidate_objective: T,
        diff: &ScheduleGraphDiff,
    ) -> AcceptanceOutcome;

    /// Takes note that the moves of `diff` made the new accepted schedule `graph`, of objective
    /// `accepted_objective`.
    ///
    /// # Safety
    ///
    /// Every vessel and berth `diff` names must be of the graph `on_start` was given.
    unsafe fn on_accept_unchecked(
        &mut self,
        accepted_objective: T,
        graph: &ScheduleGraph,
        diff: &ScheduleGraphDiff,
    );

    /// Takes note of a new best schedule, of objective `best_objective`.
    fn on_new_best(&mut self, best_objective: T);

    /// Takes note of the end of a step, a candidate decided or a scan exhausted, with the best objective so far and
    /// the accepted schedule `graph`.
    fn after_step(&mut self, best_objective: T, graph: &ScheduleGraph);

    /// Whether the candidate kept aside in the scan just exhausted becomes the accepted schedule; the rule is then
    /// told of it by [`on_accept_unchecked`](Self::on_accept_unchecked).
    fn commits_buffered(&self) -> bool;

    /// What follows the exhausted scan, from the accepted schedule `graph`.
    fn on_neighborhood_exhausted(&mut self, graph: &ScheduleGraph)
    -> NeighborhoodExhaustionOutcome;
}
