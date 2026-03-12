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

//! Defines the `LocalSearchOperator` trait for neighborhood exploration in local search.
//!
//! A `LocalSearchOperator` encapsulates the logic for exploring a specific neighborhood
//! by applying reversible mutations to a `ScheduleGraph` through a `Mutator`. The operator
//! maintains internal state, such as a cursor or move list, to produce a sequence of
//! distinct neighbors without rebuilding its analysis on every step. It prepares once for
//! a new incumbent solution, generates successive neighbors through lightweight mutations,
//! and can be reset to revisit the same neighborhood without repeating expensive
//! precomputation.
//!
//! This separation keeps mutation, decoding, and scoring concerns isolated from the search
//! engine, and supports flexible composition of neighborhood operators under meta-heuristics
//! and multi-restart strategies.
//!
//! Implementations should keep `prepare` focused on extracting actionable structure from
//! the current solutions and graph, keep `next_neighbor` fast and deterministic with
//! respect to internal state, and use `reset` to restore the starting position of the
//! exploration sequence without re-analyzing the schedule.

use crate::{mutator::Mutator, sgraph::ScheduleGraph, stats::LocalSearchStatistics};
use talos_core::utils::num::SolverNumeric;
use talos_model::{model::Model, solution::SolutionView};

/// A stateful operator that explores a specific neighborhood in a local search.
///
/// Behaves similarly to an `Iterator`, but is designed for the high-performance
/// requirements of local search where mutations are applied to an external
/// `ScheduleGraph` via a `Mutator` and must be reversible.
///
/// ## Lifecycle
///
/// 1. **`prepare`** — Called when the search reaches a new incumbent solution. The
///    operator analyzes the best, accepted, and (optionally) buffered solutions together
///    with the current `ScheduleGraph` and builds its internal list of candidate moves.
/// 2. **`next_neighbor`** — Called repeatedly. Each call applies exactly one mutation to
///    the `ScheduleGraph` through the `Mutator`. Returns `true` if a mutation was applied
///    (the search engine should decode and evaluate the candidate), or `false` when the
///    neighborhood is exhausted.
/// 3. **`reset`** — Reverts the operator's internal cursor back to the start of the
///    current neighborhood *without* re-analyzing the schedule.
pub trait LocalSearchOperator<T>: Send + Sync
where
    T: SolverNumeric,
{
    /// Returns the name of the operator for logging and identification purposes.
    fn name(&self) -> &str;

    /// Prepares the operator to explore the neighborhood of a new solution.
    ///
    /// This is the "heavy lifting" phase where the operator might:
    /// - Identify bottleneck vessels in the current solutions to target for mutation.
    /// - Pre-calculate a list of vessel pairs for swapping.
    /// - Shuffle or re-order its internal move list if the operator is stochastic.
    /// - Inspect the `ScheduleGraph` topology to determine feasible moves.
    fn prepare(
        &mut self,
        best_solution: SolutionView<'_, T>,
        accepted_solution: SolutionView<'_, T>,
        buffered_solution: Option<SolutionView<'_, T>>,
        graph: &ScheduleGraph,
    );

    /// Applies the next mutation in the neighborhood sequence.
    ///
    /// The operator uses the provided `Mutator` to apply a topological change to the
    /// underlying `ScheduleGraph`. The search engine handles decoding, evaluation, and
    /// potential rollback of these changes.
    ///
    /// # Returns
    /// - `true` — A mutation was successfully applied. The search engine should now
    ///   decode and evaluate the new candidate.
    /// - `false` — No more neighbors remain in this neighborhood.
    ///
    /// # Note
    /// The operator must manage an internal cursor so that successive calls produce
    /// distinct neighbors.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// 1. The `ScheduleGraph` inside the `Mutator` has the same dimensions (vessel and
    ///    berth counts) as the graph passed to `prepare`.
    /// 2. All `VesselIndex` values produced by the operator are valid indices within
    ///    the graph.
    unsafe fn next_neighbor(
        &mut self,
        model: &Model<T>,
        best_solution: SolutionView<'_, T>,
        accepted_solution: SolutionView<'_, T>,
        buffered_solution: Option<SolutionView<'_, T>>,
        mutator: &mut Mutator,
        stats: &LocalSearchStatistics,
    ) -> bool;

    /// Resets the operator's internal state to the beginning of the neighborhood.
    ///
    /// Unlike `prepare`, `reset` should not perform expensive re-analysis of the schedule.
    /// It simply moves the internal iteration cursor back to the first potential neighbor.
    /// This is useful for multi-restart strategies or meta-heuristics that need to
    /// re-examine the same neighborhood multiple times.
    fn reset(&mut self);
}

impl<T> std::fmt::Debug for dyn LocalSearchOperator<T>
where
    T: SolverNumeric,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "LocalSearchOperator {{ name: {} }}", self.name())
    }
}

impl<T> std::fmt::Display for dyn LocalSearchOperator<T>
where
    T: SolverNumeric,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}
