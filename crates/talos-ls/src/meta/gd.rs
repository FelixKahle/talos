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

//! Greedy Descent metaheuristic for local search.
//!
//! Greedy Descent (GD) is the simplest descent-based strategy: it only accepts
//! moves that strictly improve the objective value. It never accepts equal or
//! worsening moves, making it a pure hill climber that terminates at the first
//! local optimum.
//!
//! # Selection Strategy
//!
//! The move selection strategy is configurable via
//! `SelectionStrategy`
//!
//! | Strategy             | Behaviour                                                               |
//! |----------------------|-------------------------------------------------------------------------|
//! | `FirstImprovement`   | Accept the first strictly improving move immediately.                   |
//! | `BestImprovement`    | Buffer all improving moves, commit the best one from the neighbourhood. |
//!
//! `FirstImprovement` is the default: as soon as a candidate with a strictly
//! better objective is found, it is accepted and the neighbourhood scan ends.
//!
//! `BestImprovement` (steepest descent) evaluates the full neighbourhood,
//! buffering each improving candidate that is strictly better than the current
//! buffer. When the neighbourhood is exhausted, the engine commits the best
//! improving move. If no improving move exists, the search terminates.
//!
//! # Tie-Breaking Strategy
//!
//! When using `BestImprovement`, ties between the current buffer and a new
//! candidate with the same objective are resolved by the
//! `TieBreakingStrategy`:
//!
//! | Strategy           | Behaviour                                              |
//! |--------------------|--------------------------------------------------------|
//! | `KeepFirst`        | Keep the earlier move (default).                       |
//! | `KeepLast`         | Always replace the buffer with the newer move.         |
//! | `RandomTieBreak`   | Fair coin flip — helps avoid cycling on flat surfaces. |
//!
//! # Engine Integration
//!
//! - **First-improvement mode:** `decide_fate` returns `Accept` on the first
//!   strict improvement, `Reject` otherwise. `should_commit_buffered` returns
//!   `false`. Neighbourhood exhaustion terminates the search (local optimum).
//!
//! - **Best-improvement mode:** `decide_fate` returns `Buffer` when the
//!   candidate is strictly better than the current buffer (or ties and the
//!   tie-breaking strategy decides in favour), `Reject` otherwise.
//!   `should_commit_buffered` returns `true`. Neighbourhood exhaustion is the
//!   normal end of each scan — the engine restarts for the next GD iteration.
//!   The search terminates when no improving move is found in a full scan
//!   (the buffer remains empty and exhaustion yields `Terminate`).

use crate::{
    exec::SearchCommand,
    meta::{
        metaheuristic::{AcceptanceOutcome, Metaheuristic, NeighborhoodExhaustionOutcome},
        selec::SelectionStrategy,
        tie::{KeepFirst, TieBreakingStrategy},
    },
    sgraph::{ScheduleGraph, ScheduleGraphDiff},
};
use talos_core::utils::num::SolverNumeric;
use talos_model::{model::Model, solution::SolutionView};

// ----------------------------------------------------------------
// Greedy Descent
// ----------------------------------------------------------------

/// Greedy Descent metaheuristic with configurable selection strategy.
pub struct GreedyDescent<B = KeepFirst> {
    /// Move selection strategy.
    selection: SelectionStrategy,

    /// Tie-breaking strategy for best-improvement mode.
    tie_breaking: B,
}

impl<B: std::fmt::Debug> std::fmt::Debug for GreedyDescent<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GreedyDescent")
            .field("selection", &self.selection)
            .field("tie_breaking", &self.tie_breaking)
            .finish()
    }
}

impl GreedyDescent<KeepFirst> {
    /// Creates a new Greedy Descent with `FirstImprovement` selection
    /// and `KeepFirst` tie-breaking.
    #[inline]
    pub fn new() -> Self {
        Self {
            selection: SelectionStrategy::FirstImprovement,
            tie_breaking: KeepFirst,
        }
    }
}

impl Default for GreedyDescent<KeepFirst> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<B: TieBreakingStrategy> GreedyDescent<B> {
    /// Sets the move selection strategy.
    ///
    /// - `FirstImprovement` (default): accept the first improving move.
    /// - `BestImprovement`: buffer all improving moves, commit the best.
    #[inline]
    pub fn with_selection(mut self, strategy: SelectionStrategy) -> Self {
        self.selection = strategy;
        self
    }

    /// Replaces the tie-breaking strategy.
    ///
    /// Only relevant in `BestImprovement` mode. The default is `KeepFirst`.
    #[inline]
    pub fn with_tie_breaking<B2: TieBreakingStrategy>(self, tie_breaking: B2) -> GreedyDescent<B2> {
        GreedyDescent {
            selection: self.selection,
            tie_breaking,
        }
    }

    /// Returns the current selection strategy.
    #[inline]
    pub fn selection(&self) -> SelectionStrategy {
        self.selection
    }

    /// Returns a reference to the tie-breaking strategy.
    #[inline]
    pub fn tie_breaking(&self) -> &B {
        &self.tie_breaking
    }

    /// Returns a mutable reference to the tie-breaking strategy.
    #[inline]
    pub fn tie_breaking_mut(&mut self) -> &mut B {
        &mut self.tie_breaking
    }
}

impl<T, B> Metaheuristic<T> for GreedyDescent<B>
where
    T: SolverNumeric,
    B: TieBreakingStrategy,
{
    fn name(&self) -> &str {
        "GreedyDescent"
    }

    fn on_start(
        &mut self,
        _model: &Model<T>,
        _initial_solution: SolutionView<T>,
        _graph: &ScheduleGraph,
    ) {
    }

    fn on_end(
        &mut self,
        _model: &Model<T>,
        _final_solution: SolutionView<T>,
        _graph: &ScheduleGraph,
    ) {
    }

    /// In best-improvement mode, exhaustion is the normal end of a
    /// neighbourhood scan — the engine restarts for the next iteration.
    ///
    /// In first-improvement mode, exhaustion means no improving move
    /// exists — the search has reached a local optimum and terminates.
    fn on_neighbourhood_exhausted(
        &mut self,
        _model: &Model<T>,
        _best_solution: SolutionView<'_, T>,
        _accepted_solution: SolutionView<'_, T>,
        _buffered_solution: Option<SolutionView<'_, T>>,
        _graph: &ScheduleGraph,
    ) -> NeighborhoodExhaustionOutcome {
        match self.selection {
            SelectionStrategy::BestImprovement => NeighborhoodExhaustionOutcome::Restart,
            SelectionStrategy::FirstImprovement => NeighborhoodExhaustionOutcome::Terminate,
        }
    }

    /// In best-improvement mode, always commits the buffered solution
    /// (the best improving move found in the neighbourhood scan).
    /// In first-improvement mode, nothing is ever buffered.
    fn should_commit_buffered(
        &mut self,
        _model: &Model<T>,
        _best_solution: SolutionView<'_, T>,
        _accepted_solution: SolutionView<'_, T>,
        _buffered_solution: Option<SolutionView<'_, T>>,
        _layout: &ScheduleGraph,
        _buffer_layout: &ScheduleGraph,
    ) -> bool {
        matches!(self.selection, SelectionStrategy::BestImprovement)
    }

    fn search_command(
        &mut self,
        _iteration: u64,
        _model: &Model<T>,
        _best_solution: SolutionView<'_, T>,
        _accepted_solution: SolutionView<'_, T>,
        _buffered_solution: Option<SolutionView<'_, T>>,
    ) -> SearchCommand {
        SearchCommand::Continue
    }

    /// Decides the fate of a candidate move.
    ///
    /// Only strictly improving moves ($f' < f$) are considered. Equal
    /// and worsening moves are always rejected.
    ///
    /// In **first-improvement** mode, the first strict improvement is
    /// accepted immediately.
    ///
    /// In **best-improvement** mode, a strict improvement is buffered
    /// when it is better than the current buffer, or when it ties and
    /// the tie-breaking strategy decides in favour of the new candidate.
    #[allow(clippy::too_many_arguments)]
    fn decide_fate(
        &mut self,
        _model: &Model<T>,
        _best_solution: SolutionView<'_, T>,
        accepted_solution: SolutionView<'_, T>,
        buffered_solution: Option<SolutionView<'_, T>>,
        candidate_objective: T,
        _graph: &ScheduleGraph,
        _graph_diff: &ScheduleGraphDiff,
    ) -> AcceptanceOutcome {
        let current_objective = accepted_solution.objective_value();

        // Only accept strict improvements.
        if candidate_objective >= current_objective {
            return AcceptanceOutcome::Reject;
        }

        match self.selection {
            SelectionStrategy::FirstImprovement => AcceptanceOutcome::Accept,
            SelectionStrategy::BestImprovement => match buffered_solution {
                None => AcceptanceOutcome::Buffer,
                Some(buf) => {
                    let buf_obj = buf.objective_value();
                    if candidate_objective < buf_obj
                        || (candidate_objective == buf_obj && self.tie_breaking.break_tie())
                    {
                        AcceptanceOutcome::Buffer
                    } else {
                        AcceptanceOutcome::Reject
                    }
                }
            },
        }
    }

    fn on_accept(
        &mut self,
        _model: &Model<T>,
        _best_solution: SolutionView<'_, T>,
        _new_accepted_solution: SolutionView<'_, T>,
        _buffered_solution: Option<SolutionView<'_, T>>,
        _graph: &ScheduleGraph,
        _graph_diff: &ScheduleGraphDiff,
    ) {
        // No state to update — pure descent has no memory.
    }

    #[allow(clippy::too_many_arguments)]
    fn on_reject(
        &mut self,
        _model: &Model<T>,
        _best_solution: SolutionView<'_, T>,
        _new_accepted_solution: SolutionView<'_, T>,
        _buffered_solution: Option<SolutionView<'_, T>>,
        _candidate_objective: T,
        _graph: &ScheduleGraph,
        _graph_diff: &ScheduleGraphDiff,
    ) {
    }

    fn on_new_best(
        &mut self,
        _model: &Model<T>,
        _new_best: SolutionView<T>,
        _graph: &ScheduleGraph,
        _graph_diff: &ScheduleGraphDiff,
    ) {
    }

    fn on_iteration(
        &mut self,
        _iteration: u64,
        _model: &Model<T>,
        _best_solution: SolutionView<'_, T>,
        _new_accepted_solution: SolutionView<'_, T>,
        _buffered_solution: Option<SolutionView<'_, T>>,
        _graph: &ScheduleGraph,
    ) {
    }
}

// ----------------------------------------------------------------
// Tests
// ----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::meta::tie::{KeepLast, RandomTieBreak};

    #[test]
    fn test_greedy_descent_new_defaults() {
        let gd = GreedyDescent::new();
        assert_eq!(gd.selection(), SelectionStrategy::FirstImprovement);
    }

    #[test]
    fn test_greedy_descent_default_trait() {
        let gd = GreedyDescent::default();
        assert_eq!(gd.selection(), SelectionStrategy::FirstImprovement);
    }

    #[test]
    fn test_greedy_descent_with_selection() {
        let gd = GreedyDescent::new().with_selection(SelectionStrategy::BestImprovement);
        assert_eq!(gd.selection(), SelectionStrategy::BestImprovement);
    }

    #[test]
    fn test_greedy_descent_with_tie_breaking_keep_last() {
        let mut gd = GreedyDescent::new()
            .with_selection(SelectionStrategy::BestImprovement)
            .with_tie_breaking(KeepLast);
        assert!(gd.tie_breaking_mut().break_tie());
    }

    #[test]
    fn test_greedy_descent_with_tie_breaking_random() {
        let _gd = GreedyDescent::new()
            .with_selection(SelectionStrategy::BestImprovement)
            .with_tie_breaking(RandomTieBreak::new(rand::rng()));
    }

    #[test]
    fn test_greedy_descent_tie_breaking_accessor() {
        let mut gd = GreedyDescent::new();
        // KeepFirst always returns false.
        assert!(!gd.tie_breaking_mut().break_tie());
    }

    #[test]
    fn test_greedy_descent_debug_does_not_panic() {
        let gd = GreedyDescent::new();
        let s = format!("{:?}", gd);
        assert!(s.contains("GreedyDescent"));
        assert!(s.contains("FirstImprovement"));
        assert!(s.contains("KeepFirst"));
    }

    #[test]
    fn test_greedy_descent_debug_best_improvement() {
        let gd = GreedyDescent::new().with_selection(SelectionStrategy::BestImprovement);
        let s = format!("{:?}", gd);
        assert!(s.contains("BestImprovement"));
    }

    #[test]
    fn test_greedy_descent_name() {
        let gd = GreedyDescent::new();
        assert_eq!(Metaheuristic::<i32>::name(&gd), "GreedyDescent");
    }

    #[test]
    fn test_should_commit_buffered_first_improvement() {
        let gd = GreedyDescent::new().with_selection(SelectionStrategy::FirstImprovement);
        assert!(!matches!(
            gd.selection(),
            SelectionStrategy::BestImprovement
        ));
    }

    #[test]
    fn test_should_commit_buffered_best_improvement() {
        let gd = GreedyDescent::new().with_selection(SelectionStrategy::BestImprovement);
        assert!(matches!(gd.selection(), SelectionStrategy::BestImprovement));
    }
}
