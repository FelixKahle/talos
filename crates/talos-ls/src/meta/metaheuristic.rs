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

//! Metaheuristic interface for local search control.
//!
//! This module defines the trait used to steer a local search run. It separates
//! move generation and delta-decoding from the strategic acceptance policy and
//! termination logic.

use crate::{
    exec::SearchCommand,
    sgraph::{ScheduleGraph, ScheduleGraphDiff},
};
use talos_core::utils::num::SolverNumeric;
use talos_model::{model::Model, solution::SolutionView};
use talos_search::oracle::GlobalOracle;

// ----------------------------------------------------------------
// AcceptanceOutcome
// ----------------------------------------------------------------

/// Determines how the search engine should handle the currently evaluated candidate.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AcceptanceOutcome {
    /// Accept immediately. The evaluated candidate becomes the new Master state.
    Accept,

    /// Do not move the Master state yet, but save this candidate into the Buffer
    /// as the new "Best in Neighborhood" for later evaluation.
    Buffer,

    /// Discard the candidate. The engine will instantly roll back the topological layout.
    Reject,
}

impl std::fmt::Display for AcceptanceOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AcceptanceOutcome::Accept => write!(f, "Accept"),
            AcceptanceOutcome::Buffer => write!(f, "Buffer"),
            AcceptanceOutcome::Reject => write!(f, "Reject"),
        }
    }
}

// ----------------------------------------------------------------
// TeleportTarget
// ----------------------------------------------------------------

/// Specifies which solution in the oracle pool the engine should teleport to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TeleportTarget {
    /// Teleport to the best (rank 0) solution in the pool.
    Best,

    /// Teleport to the solution at the given rank (0 = best).
    /// Falls back to the best solution if the rank is out of bounds.
    Rank(usize),
}

impl std::fmt::Display for TeleportTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TeleportTarget::Best => write!(f, "Best"),
            TeleportTarget::Rank(r) => write!(f, "Rank({})", r),
        }
    }
}

// ----------------------------------------------------------------
// NeighborhoodExhaustionOutcome
// ----------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NeighborhoodExhaustionOutcome {
    /// Restart the search with a new neighborhood exploration sequence.
    Restart,

    /// Terminate the search.
    Terminate,

    /// Signal the engine to teleport to the specified oracle solution.
    ///
    /// The engine is responsible for extracting the solution from the
    /// oracle and writing it directly into its internal buffers,
    /// avoiding an intermediate heap allocation.
    Teleport(TeleportTarget),
}

impl std::fmt::Display for NeighborhoodExhaustionOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NeighborhoodExhaustionOutcome::Restart => write!(f, "Restart"),
            NeighborhoodExhaustionOutcome::Terminate => write!(f, "Terminate"),
            NeighborhoodExhaustionOutcome::Teleport(target) => write!(f, "Teleport({})", target),
        }
    }
}

// ----------------------------------------------------------------
// Metaheuristic Trait
// ----------------------------------------------------------------

/// A trait governing the strategic acceptance logic and termination of the local search.
pub trait Metaheuristic<T, G>
where
    T: SolverNumeric,
    G: GlobalOracle<T>,
{
    /// Returns the name of the metaheuristic.
    fn name(&self) -> &str;

    /// Called at the start of the search.
    fn on_start(
        &mut self,
        model: &Model<T>,
        initial_solution: SolutionView<T>,
        graph: &ScheduleGraph,
    );

    /// Called at the end of the search.
    fn on_end(&mut self, model: &Model<T>, final_solution: SolutionView<T>, graph: &ScheduleGraph);

    /// Called when an operator has completely exhausted its neighborhood for the current state.
    /// This means the engine has completed a full cycle.
    ///
    /// **Strategic Hook:** Primary entrance for **Guided Local Search (GLS)**.
    /// GLS uses the `graph` to identify features to penalize in the `model`.
    ///
    /// Returns `NeighborhoodExhaustionOutcome` to indicate whether to restart exploration
    /// (escaping local optima) or terminate the search.
    fn on_neighbourhood_exhausted(
        &mut self,
        model: &Model<T>,
        best_solution: SolutionView<'_, T>,
        accepted_solution: SolutionView<'_, T>,
        buffered_solution: Option<SolutionView<'_, T>>,
        graph: &ScheduleGraph,
        oracle: &G,
    ) -> NeighborhoodExhaustionOutcome;

    /// Called when an operator has finished scanning a full neighborhood.
    ///
    /// **Strategic Hook:** Used by **Steepest Descent** and **LAHC** to decide
    /// if the best neighbor found (currently in the engine's buffer) should be committed.
    fn should_commit_buffered(
        &mut self,
        model: &Model<T>,
        best_solution: SolutionView<'_, T>,
        accepted_solution: SolutionView<'_, T>,
        buffered_solution: Option<SolutionView<'_, T>>,
        layout: &ScheduleGraph,
        buffer_layout: &ScheduleGraph,
    ) -> bool;

    /// Determines if the search should proceed to the next iteration.
    fn search_command(
        &mut self,
        iteration: u64,
        model: &Model<T>,
        best_solution: SolutionView<'_, T>,
        accepted_solution: SolutionView<'_, T>,
        buffered_solution: Option<SolutionView<'_, T>>,
    ) -> SearchCommand;

    /// **THE HOT PATH:** Decides the fate of a proposed neighborhood move.
    ///
    /// Receives `moved_vessels` for $O(1)$ Tabu checks and the `layout` to
    /// verify aspiration criteria or complex constraints.
    #[allow(clippy::too_many_arguments)]
    fn decide_fate(
        &mut self,
        model: &Model<T>,
        best_solution: SolutionView<'_, T>,
        accepted_solution: SolutionView<'_, T>,
        buffered_solution: Option<SolutionView<'_, T>>,
        candidate_objective: T,
        graph: &ScheduleGraph,
        graph_diff: &ScheduleGraphDiff,
    ) -> AcceptanceOutcome;

    /// Called when a move is firmly accepted into the Master state.
    ///
    /// **Strategic Hook:** Used by **Tabu Search** to update forbidden move
    /// tenures based on the vessels that actually moved.
    fn on_accept(
        &mut self,
        model: &Model<T>,
        best_solution: SolutionView<'_, T>,
        new_accepted_solution: SolutionView<'_, T>,
        buffered_solution: Option<SolutionView<'_, T>>,
        graph: &ScheduleGraph,
        graph_diff: &ScheduleGraphDiff,
    );

    /// Called when a move is rejected by `decide_fate`.
    #[allow(clippy::too_many_arguments)]
    fn on_reject(
        &mut self,
        model: &Model<T>,
        best_solution: SolutionView<'_, T>,
        new_accepted_solution: SolutionView<'_, T>,
        buffered_solution: Option<SolutionView<'_, T>>,
        candidate_objective: T,
        graph: &ScheduleGraph,
        graph_diff: &ScheduleGraphDiff,
    );

    /// Called when a newly accepted move represents a new global best solution.
    fn on_new_best(
        &mut self,
        model: &Model<T>,
        new_best: SolutionView<T>,
        graph: &ScheduleGraph,
        graph_diff: &ScheduleGraphDiff,
    );

    fn on_teleport(
        &mut self,
        model: &Model<T>,
        new_solution: SolutionView<'_, T>,
        graph: &ScheduleGraph,
    );

    /// Called at the end of every neighborhood evaluation loop.
    fn on_iteration(
        &mut self,
        iteration: u64,
        model: &Model<T>,
        best_solution: SolutionView<'_, T>,
        new_accepted_solution: SolutionView<'_, T>,
        buffered_solution: Option<SolutionView<'_, T>>,
        graph: &ScheduleGraph,
    );
}

impl<T, G> std::fmt::Debug for dyn Metaheuristic<T, G>
where
    T: SolverNumeric,
    G: GlobalOracle<T>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Metaheuristic {{ name: {} }}", self.name())
    }
}

impl<T, G> std::fmt::Display for dyn Metaheuristic<T, G>
where
    T: SolverNumeric,
    G: GlobalOracle<T>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Metaheuristic: {}", self.name())
    }
}
