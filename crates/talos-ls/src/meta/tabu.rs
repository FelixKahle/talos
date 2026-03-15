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

//! Tabu Search metaheuristic for local search.
//!
//! Tabu Search (TS) escapes local optima by maintaining a short-term memory
//! of recently reversed attributes and forbidding the search from undoing
//! them for a configurable number of iterations (the *tabu tenure*).
//!
//! # Feature Decomposition
//!
//! This implementation tracks two independent attribute classes via `TabuMemory`:
//!
//! | Attribute           | Memory                              | Semantics                               |
//! |---------------------|-------------------------------------|-----------------------------------------|
//! | **Edge (Sequence)** | `edge[A * V + B] = expire_iter`     | Vessel A directly preceding Vessel B    |
//! | **Berth (Assignment)** | `berth[V * B + X] = expire_iter` | Vessel V assigned to Berth X         |
//!
//! When a move is accepted, the *broken* edges and the *old* berth assignments
//! from the `ScheduleGraphDiff` are marked as tabu — forbidding the solver
//! from reverting them. When evaluating a candidate, the *created* edges and
//! *new* berth assignments are checked against the tabu memory.
//!
//! # Aspiration Criterion
//!
//! A tabu move can still be accepted if it satisfies the `AspirationCriterion`.
//! The default `BestObjectiveAspiration` overrides the tabu status when the
//! candidate objective is strictly better than the global best — the standard
//! aspiration rule from the literature.
//!
//! # Tenure Strategies
//!
//! The tabu tenure (how many iterations a move stays forbidden) is controlled
//! via the `TenureStrategy` trait:
//!
//! | Strategy       | Behavior                                           |
//! |----------------|----------------------------------------------------|
//! | `FixedTenure`  | Constant tenure for every move.                    |
//! | `RandomTenure` | Uniformly sampled from a configurable range.       |
//!
//! # Selection Strategy
//!
//! The move selection strategy is configurable via `SelectionStrategy`:
//!
//! | Strategy             | Behaviour                                                      |
//! |----------------------|----------------------------------------------------------------|
//! | `BestImprovement`    | Buffer all admissible moves, commit the best one (default).    |
//! | `FirstImprovement`   | Accept the first admissible move immediately.                  |
//!
//! `BestImprovement` is the standard choice from the literature: the engine
//! evaluates the full neighbourhood each iteration, and the metaheuristic
//! only buffers a candidate when it is strictly better than the current
//! buffer. When the neighbourhood is exhausted, the engine commits the
//! best admissible move and calls `on_accept` with the saved buffered
//! diff, so the tabu memory is updated correctly. The engine then
//! restarts the scan for the next TS iteration.
//!
//! `FirstImprovement` short-circuits the scan: the first admissible move
//! is accepted immediately and `on_accept` records the tabu attributes
//! from the still-live diff.
//!
//! # Tie-Breaking Strategy
//!
//! When using `BestImprovement`, ties between the current buffer and a
//! new candidate with the same objective are resolved by the
//! `TieBreakingStrategy` trait:
//!
//! | Strategy           | Behaviour                                              |
//! |--------------------|--------------------------------------------------------|
//! | `KeepFirst`        | Keep the earlier move (default).                       |
//! | `KeepLast`         | Always replace the buffer with the newer move.         |
//! | `RandomTieBreak`   | Fair coin flip — helps avoid cycling on flat surfaces. |
//!
//! # Engine Integration
//!
//! The behaviour on neighbourhood exhaustion is configurable via
//! `with_exhaustion_outcome` (default: `Restart`). In best-improvement
//! mode, exhaustion is the normal termination of each neighbourhood scan
//! — the engine restarts a fresh scan every iteration.

use crate::{
    exec::SearchCommand,
    meta::{
        metaheuristic::{
            AcceptanceOutcome, Metaheuristic, NeighborhoodExhaustionOutcome, TeleportTarget,
        },
        teleport::{NoTeleport, TeleportPolicy, should_attempt_teleport},
        tie::{KeepFirst, TieBreakingStrategy},
    },
    sgraph::{ScheduleGraph, ScheduleGraphDiff},
};
use rand::{Rng, RngExt};
use std::marker::PhantomData;
use talos_core::utils::num::SolverNumeric;
use talos_model::{model::Model, solution::SolutionView};
use talos_search::oracle::GlobalOracle;

// ----------------------------------------------------------------
// Tenure Strategy
// ----------------------------------------------------------------

/// Controls how long a move attribute remains tabu.
pub trait TenureStrategy: std::fmt::Debug {
    /// Returns the tenure (in iterations) for the next accepted move.
    fn tenure(&mut self) -> u64;
}

/// Constant tabu tenure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FixedTenure {
    value: u64,
}

impl FixedTenure {
    /// Creates a new fixed tenure strategy.
    ///
    /// # Panics
    ///
    /// Panics if `value` is zero.
    #[inline]
    pub fn new(value: u64) -> Self {
        assert!(value > 0, "FixedTenure: tenure must be > 0, got {}", value);
        Self { value }
    }
}

impl TenureStrategy for FixedTenure {
    #[inline]
    fn tenure(&mut self) -> u64 {
        self.value
    }
}

/// Uniformly random tabu tenure sampled from $[\text{min}, \text{max}]$.
#[derive(Debug)]
pub struct RandomTenure<R> {
    min: u64,
    max: u64,
    rng: R,
}

impl<R: Rng> RandomTenure<R> {
    /// Creates a new random tenure strategy.
    ///
    /// # Panics
    ///
    /// Panics if `min > max` or `min == 0`.
    #[inline]
    pub fn new(min: u64, max: u64, rng: R) -> Self {
        assert!(min > 0, "RandomTenure: min must be > 0, got {}", min);
        assert!(
            min <= max,
            "RandomTenure: min must be <= max, got {} > {}",
            min,
            max
        );
        Self { min, max, rng }
    }
}

impl<R: Rng + std::fmt::Debug> TenureStrategy for RandomTenure<R> {
    #[inline]
    fn tenure(&mut self) -> u64 {
        if self.min == self.max {
            self.min
        } else {
            self.min + self.rng.random_range(0..=(self.max - self.min))
        }
    }
}

// ----------------------------------------------------------------
// Aspiration Criterion
// ----------------------------------------------------------------

/// Decides whether a tabu move should be accepted anyway.
///
/// The criterion receives the candidate objective and the current global
/// best. If it returns `true`, the tabu status is overridden and the move
/// is accepted.
pub trait AspirationCriterion: std::fmt::Debug {
    /// Returns `true` if the tabu status should be overridden.
    ///
    /// # Arguments
    ///
    /// * `candidate_objective` — The objective value of the candidate move.
    /// * `best_objective` — The current global best objective value.
    fn aspires<T: SolverNumeric>(&self, candidate_objective: T, best_objective: T) -> bool;
}

/// Overrides tabu status when the candidate strictly improves the global best.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct BestObjectiveAspiration;

impl AspirationCriterion for BestObjectiveAspiration {
    #[inline]
    fn aspires<T: SolverNumeric>(&self, candidate_objective: T, best_objective: T) -> bool {
        candidate_objective < best_objective
    }
}

// ----------------------------------------------------------------
// Tabu Memory
// ----------------------------------------------------------------

/// Short-term memory tracking forbidden move attributes.
///
/// Stores expiration iterations for two attribute classes:
/// - **Edge**: vessel-to-vessel sequence links.
/// - **Berth**: vessel-to-berth assignments.
///
/// All operations are $O(1)$ array lookups.
#[derive(Debug, Clone)]
pub struct TabuMemory {
    /// Flat `num_vessels * num_vessels` array. `edge[a * num_vessels + b]`
    /// stores the iteration at which the edge $a \to b$ ceases to be tabu.
    edge: Vec<u64>,

    /// Flat `num_vessels * num_berths` array. `berth[v * num_berths + b]`
    /// stores the iteration at which assigning vessel $v$ to berth $b$
    /// ceases to be tabu.
    berth: Vec<u64>,

    num_vessels: usize,
    num_berths: usize,
}

impl TabuMemory {
    /// Creates a new tabu memory for the given problem dimensions.
    #[inline]
    pub fn new(num_vessels: usize, num_berths: usize) -> Self {
        Self {
            edge: vec![0; num_vessels * num_vessels],
            berth: vec![0; num_vessels * num_berths],
            num_vessels,
            num_berths,
        }
    }

    /// Resets all tabu expiration counters to zero.
    #[inline]
    pub fn clear(&mut self) {
        self.edge.fill(0);
        self.berth.fill(0);
    }

    /// Records the broken edges and old berth assignments from the diff as tabu.
    ///
    /// This forbids the search from *reverting* the accepted move.
    #[inline]
    pub fn record(&mut self, diff: &ScheduleGraphDiff, expire_iter: u64) {
        // SAFETY: all flat indices are bounded by allocation sizes.
        // Edge indices: a < num_vessels, b < num_vessels => idx < num_vessels^2.
        // Berth indices: vessel < num_vessels, berth < num_berths => idx < num_vessels*num_berths.
        unsafe {
            // Forbid re-creating the broken edges.
            for edge in diff.broken_links() {
                let a = edge.from.get();
                let b = edge.to.get();
                if a < self.num_vessels && b < self.num_vessels {
                    *self.edge.get_unchecked_mut(a * self.num_vessels + b) = expire_iter;
                }
            }

            // Forbid returning vessels to their old berths.
            for (vessel, old_berth, _new_berth) in diff.reallocations() {
                *self
                    .berth
                    .get_unchecked_mut(vessel.get() * self.num_berths + old_berth.get()) =
                    expire_iter;
            }
        }
    }

    /// Returns `true` if any created edge or new berth assignment in the diff
    /// is currently tabu at the given iteration.
    #[inline]
    pub fn is_tabu(&self, diff: &ScheduleGraphDiff, current_iter: u64) -> bool {
        // SAFETY: all flat indices are bounded by allocation sizes (same as `record`).
        unsafe {
            // Check created edges.
            for edge in diff.created_links() {
                let a = edge.from.get();
                let b = edge.to.get();
                if a < self.num_vessels
                    && b < self.num_vessels
                    && *self.edge.get_unchecked(a * self.num_vessels + b) > current_iter
                {
                    return true;
                }
            }

            // Check new berth assignments.
            for (vessel, _old_berth, new_berth) in diff.reallocations() {
                if *self
                    .berth
                    .get_unchecked(vessel.get() * self.num_berths + new_berth.get())
                    > current_iter
                {
                    return true;
                }
            }
        }

        false
    }
}

// ----------------------------------------------------------------
// Selection Strategy
// ----------------------------------------------------------------

/// Controls how the search selects among admissible moves in a
/// neighbourhood scan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SelectionStrategy {
    /// Evaluate the full neighbourhood, buffer every admissible move,
    /// and commit the best one. This is the standard Tabu Search
    /// behaviour from the literature.
    BestImprovement,

    /// Accept the first admissible move encountered and skip the
    /// rest of the neighbourhood.
    FirstImprovement,
}

// ----------------------------------------------------------------
// Tabu Search
// ----------------------------------------------------------------

/// Tabu Search metaheuristic with pluggable tenure and aspiration strategies.
///
/// See the [module-level documentation](self) for algorithmic details.
pub struct TabuSearch<T, S, A = BestObjectiveAspiration, B = KeepFirst, Tp = NoTeleport>
where
    T: Copy,
{
    /// The tenure strategy controlling how long moves stay forbidden.
    tenure: S,

    /// The aspiration criterion for overriding tabu status.
    aspiration: A,

    /// Short-term memory of forbidden attributes.
    memory: TabuMemory,

    /// Current TS iteration counter (incremented in `on_accept`).
    iteration: u64,

    /// What to do when the neighbourhood is exhausted.
    exhaustion_outcome: NeighborhoodExhaustionOutcome,

    /// Move selection strategy.
    selection: SelectionStrategy,

    /// Tie-breaking strategy for best-improvement mode.
    tie_breaking: B,

    /// Teleport policy controlling oracle-based solution injection.
    teleport_policy: Tp,

    /// Marker to keep `T` as a type parameter.
    _marker: PhantomData<T>,
}

impl<
    T: std::fmt::Debug,
    S: std::fmt::Debug,
    A: std::fmt::Debug,
    B: std::fmt::Debug,
    Tp: std::fmt::Debug,
> std::fmt::Debug for TabuSearch<T, S, A, B, Tp>
where
    T: Copy,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TabuSearch")
            .field("tenure", &self.tenure)
            .field("aspiration", &self.aspiration)
            .field("selection", &self.selection)
            .field("tie_breaking", &self.tie_breaking)
            .field("iteration", &self.iteration)
            .field("teleport_policy", &self.teleport_policy)
            .finish()
    }
}

impl<T, S> TabuSearch<T, S, BestObjectiveAspiration, KeepFirst>
where
    T: Copy,
    S: TenureStrategy,
{
    /// Creates a new Tabu Search with default `BestObjectiveAspiration`
    /// and `KeepFirst` tie-breaking.
    ///
    /// # Arguments
    ///
    /// * `tenure` — The tenure strategy (e.g., `FixedTenure::new(10)`).
    /// * `num_vessels` — Number of vessels in the problem.
    /// * `num_berths` — Number of berths in the problem.
    #[inline]
    pub fn new(tenure: S, num_vessels: usize, num_berths: usize) -> Self {
        Self {
            tenure,
            aspiration: BestObjectiveAspiration,
            memory: TabuMemory::new(num_vessels, num_berths),
            iteration: 0,
            exhaustion_outcome: NeighborhoodExhaustionOutcome::Restart,
            selection: SelectionStrategy::BestImprovement,
            tie_breaking: KeepFirst,
            teleport_policy: NoTeleport,
            _marker: PhantomData,
        }
    }
}

impl<T, S, A> TabuSearch<T, S, A, KeepFirst>
where
    T: Copy,
    S: TenureStrategy,
    A: AspirationCriterion,
{
    /// Creates a new Tabu Search with a custom aspiration criterion
    /// and default `KeepFirst` tie-breaking.
    ///
    /// # Arguments
    ///
    /// * `tenure` — The tenure strategy.
    /// * `aspiration` — The aspiration criterion.
    /// * `num_vessels` — Number of vessels in the problem.
    /// * `num_berths` — Number of berths in the problem.
    #[inline]
    pub fn with_aspiration(
        tenure: S,
        aspiration: A,
        num_vessels: usize,
        num_berths: usize,
    ) -> Self {
        Self {
            tenure,
            aspiration,
            memory: TabuMemory::new(num_vessels, num_berths),
            iteration: 0,
            exhaustion_outcome: NeighborhoodExhaustionOutcome::Restart,
            selection: SelectionStrategy::BestImprovement,
            tie_breaking: KeepFirst,
            teleport_policy: NoTeleport,
            _marker: PhantomData,
        }
    }
}

impl<T, S, A, B, Tp> TabuSearch<T, S, A, B, Tp>
where
    T: Copy,
    S: TenureStrategy,
    A: AspirationCriterion,
    B: TieBreakingStrategy,
    Tp: TeleportPolicy,
{
    /// Replaces the tie-breaking strategy.
    ///
    /// The default is `KeepFirst` — the first-seen move wins ties. Use
    /// `RandomTieBreak` to avoid deterministic cycling on flat landscapes.
    #[inline]
    pub fn with_tie_breaking<B2: TieBreakingStrategy>(
        self,
        tie_breaking: B2,
    ) -> TabuSearch<T, S, A, B2, Tp> {
        TabuSearch {
            tenure: self.tenure,
            aspiration: self.aspiration,
            memory: self.memory,
            iteration: self.iteration,
            exhaustion_outcome: self.exhaustion_outcome,
            selection: self.selection,
            tie_breaking,
            teleport_policy: self.teleport_policy,
            _marker: PhantomData,
        }
    }

    /// Replaces the teleport policy.
    #[inline]
    pub fn with_teleport<Tp2: TeleportPolicy>(self, policy: Tp2) -> TabuSearch<T, S, A, B, Tp2> {
        TabuSearch {
            tenure: self.tenure,
            aspiration: self.aspiration,
            memory: self.memory,
            iteration: self.iteration,
            exhaustion_outcome: self.exhaustion_outcome,
            selection: self.selection,
            tie_breaking: self.tie_breaking,
            teleport_policy: policy,
            _marker: PhantomData,
        }
    }

    /// Sets the move selection strategy.
    ///
    /// The default is `SelectionStrategy::BestImprovement` — the standard
    /// Tabu Search behaviour from the literature.
    #[inline]
    pub fn with_selection(mut self, strategy: SelectionStrategy) -> Self {
        self.selection = strategy;
        self
    }

    /// Returns the current selection strategy.
    #[inline]
    pub fn selection(&self) -> SelectionStrategy {
        self.selection
    }

    /// Sets the behaviour when the neighbourhood is exhausted.
    ///
    /// The default is `NeighborhoodExhaustionOutcome::Restart`, which is
    /// the standard choice for Tabu Search — the engine re-scans the
    /// neighbourhood on each iteration.
    #[inline]
    pub fn with_exhaustion_outcome(mut self, outcome: NeighborhoodExhaustionOutcome) -> Self {
        self.exhaustion_outcome = outcome;
        self
    }

    /// Returns the current exhaustion outcome setting.
    #[inline]
    pub fn exhaustion_outcome(&self) -> NeighborhoodExhaustionOutcome {
        self.exhaustion_outcome
    }

    /// Returns a reference to the tenure strategy.
    #[inline]
    pub fn tenure(&self) -> &S {
        &self.tenure
    }

    /// Returns a mutable reference to the tenure strategy.
    #[inline]
    pub fn tenure_mut(&mut self) -> &mut S {
        &mut self.tenure
    }

    /// Returns a reference to the aspiration criterion.
    #[inline]
    pub fn aspiration(&self) -> &A {
        &self.aspiration
    }

    /// Returns a mutable reference to the aspiration criterion.
    #[inline]
    pub fn aspiration_mut(&mut self) -> &mut A {
        &mut self.aspiration
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

    /// Returns a reference to the tabu memory.
    #[inline]
    pub fn memory(&self) -> &TabuMemory {
        &self.memory
    }

    /// Returns the current TS iteration counter.
    ///
    /// This counts accepted moves, not engine-level candidate evaluations.
    #[inline]
    pub fn current_iteration(&self) -> u64 {
        self.iteration
    }
}

impl<T, S, A, B, Tp, G> Metaheuristic<T, G> for TabuSearch<T, S, A, B, Tp>
where
    T: SolverNumeric,
    S: TenureStrategy,
    A: AspirationCriterion,
    B: TieBreakingStrategy,
    Tp: TeleportPolicy,
    G: GlobalOracle<T>,
{
    fn name(&self) -> &str {
        "TabuSearch"
    }

    fn on_start(
        &mut self,
        _model: &Model<T>,
        _initial_solution: SolutionView<T>,
        _graph: &ScheduleGraph,
    ) {
        self.memory.clear();
        self.iteration = 0;
        self.teleport_policy.on_start();
    }

    fn on_end(
        &mut self,
        _model: &Model<T>,
        _final_solution: SolutionView<T>,
        _graph: &ScheduleGraph,
    ) {
    }

    /// Returns the configured exhaustion outcome.
    fn on_neighbourhood_exhausted(
        &mut self,
        _model: &Model<T>,
        _best_solution: SolutionView<'_, T>,
        _accepted_solution: SolutionView<'_, T>,
        _buffered_solution: Option<SolutionView<'_, T>>,
        _graph: &ScheduleGraph,
        oracle: &G,
    ) -> NeighborhoodExhaustionOutcome {
        if should_attempt_teleport(
            &mut self.teleport_policy,
            oracle,
            _best_solution.objective_value(),
        ) {
            return NeighborhoodExhaustionOutcome::Teleport(TeleportTarget::Best);
        }

        self.exhaustion_outcome
    }

    /// In best-improvement mode, always commits the buffered solution.
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
    /// In **best-improvement** mode, an admissible move is buffered when
    /// it is strictly better than the current buffer, or when it ties
    /// and the `TieBreakingStrategy` decides in favour of the new
    /// candidate.
    ///
    /// In **first-improvement** mode, the first admissible move is
    /// accepted immediately.
    ///
    /// A move is *inadmissible* (rejected) when it is tabu and the
    /// aspiration criterion does not override the tabu status.
    #[allow(clippy::too_many_arguments)]
    fn decide_fate(
        &mut self,
        _model: &Model<T>,
        best_solution: SolutionView<'_, T>,
        _accepted_solution: SolutionView<'_, T>,
        buffered_solution: Option<SolutionView<'_, T>>,
        candidate_objective: T,
        _graph: &ScheduleGraph,
        graph_diff: &ScheduleGraphDiff,
    ) -> AcceptanceOutcome {
        let best_objective = best_solution.objective_value();

        // Check admissibility: not tabu, or aspiration overrides.
        let tabu = self.memory.is_tabu(graph_diff, self.iteration);
        let admissible = !tabu || self.aspiration.aspires(candidate_objective, best_objective);

        if !admissible {
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

    /// Records the accepted move's reversed attributes into the tabu memory
    /// and advances the TS iteration counter.
    ///
    /// Called by the engine both for direct accepts (first-improvement) and
    /// when committing the buffer (best-improvement). In the latter case
    /// the engine passes the saved buffered diff.
    ///
    /// The TS iteration counter is advanced here — not in `on_iteration` —
    /// because a TS iteration corresponds to one accepted move, whereas
    /// the engine calls `on_iteration` once per candidate evaluation.
    fn on_accept(
        &mut self,
        _model: &Model<T>,
        _best_solution: SolutionView<'_, T>,
        _new_accepted_solution: SolutionView<'_, T>,
        _buffered_solution: Option<SolutionView<'_, T>>,
        _graph: &ScheduleGraph,
        graph_diff: &ScheduleGraphDiff,
    ) {
        let tenure = self.tenure.tenure();
        let expire_iter = self.iteration.saturating_add(tenure);
        self.memory.record(graph_diff, expire_iter);

        // Advance after recording so the tenure is measured from the
        // current iteration, not the next one.
        self.iteration = self.iteration.saturating_add(1);
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
        // No-op.
    }

    fn on_new_best(
        &mut self,
        _model: &Model<T>,
        _new_best: SolutionView<T>,
        _graph: &ScheduleGraph,
        _graph_diff: &ScheduleGraphDiff,
    ) {
        self.teleport_policy.on_improvement();
    }

    fn on_teleport(
        &mut self,
        _model: &Model<T>,
        _new_solution: SolutionView<'_, T>,
        _graph: &ScheduleGraph,
    ) {
        self.teleport_policy.on_teleport();
        self.memory.clear();
        self.iteration = 0;
    }

    /// No-op. The engine calls this once per candidate evaluation, but TS
    /// iterations are defined as accepted moves. The TS iteration counter
    /// is therefore advanced in `on_accept` instead.
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
    use talos_search::oracle::NoOracle;

    #[test]
    fn test_fixed_tenure_returns_constant() {
        let mut t = FixedTenure::new(7);
        assert_eq!(t.tenure(), 7);
        assert_eq!(t.tenure(), 7);
        assert_eq!(t.tenure(), 7);
    }

    #[test]
    #[should_panic(expected = "tenure must be > 0")]
    fn test_fixed_tenure_zero_panics() {
        FixedTenure::new(0);
    }

    #[test]
    fn test_fixed_tenure_debug() {
        let t = FixedTenure::new(5);
        let s = format!("{:?}", t);
        assert!(s.contains("FixedTenure"));
        assert!(s.contains("5"));
    }

    #[test]
    fn test_fixed_tenure_clone_eq() {
        let a = FixedTenure::new(10);
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn test_random_tenure_equal_bounds_returns_constant() {
        let mut t = RandomTenure::new(5, 5, rand::rng());
        for _ in 0..20 {
            assert_eq!(t.tenure(), 5);
        }
    }

    #[test]
    fn test_random_tenure_within_range() {
        let mut t = RandomTenure::new(3, 10, rand::rng());
        for _ in 0..100 {
            let v = t.tenure();
            assert!((3..=10).contains(&v), "tenure {} out of [3, 10]", v);
        }
    }

    #[test]
    #[should_panic(expected = "min must be > 0")]
    fn test_random_tenure_zero_min_panics() {
        RandomTenure::new(0, 10, rand::rng());
    }

    #[test]
    #[should_panic(expected = "min must be <= max")]
    fn test_random_tenure_inverted_range_panics() {
        RandomTenure::new(10, 5, rand::rng());
    }

    #[test]
    fn test_aspiration_accepts_strict_improvement() {
        assert!(BestObjectiveAspiration.aspires(99, 100));
    }

    #[test]
    fn test_aspiration_rejects_equal() {
        assert!(!BestObjectiveAspiration.aspires(100, 100));
    }

    #[test]
    fn test_aspiration_rejects_worse() {
        assert!(!BestObjectiveAspiration.aspires(101, 100));
    }

    #[test]
    fn test_keep_first_always_false() {
        let mut k = KeepFirst;
        for _ in 0..10 {
            assert!(!k.break_tie());
        }
    }

    #[test]
    fn test_keep_last_always_true() {
        let mut k = KeepLast;
        for _ in 0..10 {
            assert!(k.break_tie());
        }
    }

    #[test]
    fn test_random_tie_break_produces_both_outcomes() {
        let mut r = RandomTieBreak::new(rand::rng());
        let mut saw_true = false;
        let mut saw_false = false;
        for _ in 0..200 {
            if r.break_tie() {
                saw_true = true;
            } else {
                saw_false = true;
            }
            if saw_true && saw_false {
                return;
            }
        }
        panic!("RandomTieBreak should produce both true and false");
    }

    #[test]
    fn test_tabu_memory_new_is_empty() {
        let mem = TabuMemory::new(5, 3);
        assert_eq!(mem.edge.len(), 25);
        assert_eq!(mem.berth.len(), 15);
        assert!(mem.edge.iter().all(|&v| v == 0));
        assert!(mem.berth.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_tabu_memory_clear() {
        let mut mem = TabuMemory::new(3, 2);
        mem.edge[4] = 99;
        mem.berth[3] = 42;
        mem.clear();
        assert!(mem.edge.iter().all(|&v| v == 0));
        assert!(mem.berth.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_selection_strategy_debug() {
        assert!(format!("{:?}", SelectionStrategy::BestImprovement).contains("BestImprovement"));
        assert!(format!("{:?}", SelectionStrategy::FirstImprovement).contains("FirstImprovement"));
    }

    #[test]
    fn test_selection_strategy_eq() {
        assert_eq!(
            SelectionStrategy::BestImprovement,
            SelectionStrategy::BestImprovement
        );
        assert_ne!(
            SelectionStrategy::BestImprovement,
            SelectionStrategy::FirstImprovement
        );
    }

    #[test]
    fn test_selection_strategy_copy() {
        let a = SelectionStrategy::BestImprovement;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn test_tabu_search_new_defaults() {
        let ts: TabuSearch<i32, _> = TabuSearch::new(FixedTenure::new(5), 10, 3);
        assert_eq!(ts.selection(), SelectionStrategy::BestImprovement);
        assert_eq!(
            ts.exhaustion_outcome(),
            NeighborhoodExhaustionOutcome::Restart
        );
        assert_eq!(ts.current_iteration(), 0);
    }

    #[test]
    fn test_tabu_search_with_selection() {
        let ts: TabuSearch<i32, _> = TabuSearch::new(FixedTenure::new(5), 10, 3)
            .with_selection(SelectionStrategy::FirstImprovement);
        assert_eq!(ts.selection(), SelectionStrategy::FirstImprovement);
    }

    #[test]
    fn test_tabu_search_with_exhaustion_outcome() {
        let ts: TabuSearch<i32, _> = TabuSearch::new(FixedTenure::new(5), 10, 3)
            .with_exhaustion_outcome(NeighborhoodExhaustionOutcome::Terminate);
        assert_eq!(
            ts.exhaustion_outcome(),
            NeighborhoodExhaustionOutcome::Terminate
        );
    }

    #[test]
    fn test_tabu_search_with_tie_breaking() {
        let mut ts: TabuSearch<i32, _, _, KeepLast> =
            TabuSearch::new(FixedTenure::new(5), 10, 3).with_tie_breaking(KeepLast);
        assert!(ts.tie_breaking_mut().break_tie());
    }

    #[test]
    fn test_tabu_search_tenure_accessor() {
        let ts: TabuSearch<i32, _> = TabuSearch::new(FixedTenure::new(7), 4, 2);
        assert_eq!(ts.tenure().value, 7);
    }

    #[test]
    fn test_tabu_search_tenure_mut_accessor() {
        let mut ts: TabuSearch<i32, _> = TabuSearch::new(FixedTenure::new(7), 4, 2);
        *ts.tenure_mut() = FixedTenure::new(12);
        assert_eq!(ts.tenure().value, 12);
    }

    #[test]
    fn test_tabu_search_aspiration_accessor() {
        let ts: TabuSearch<i32, _> = TabuSearch::new(FixedTenure::new(5), 4, 2);
        assert!(ts.aspiration().aspires(1, 2));
    }

    #[test]
    fn test_tabu_search_memory_accessor() {
        let ts: TabuSearch<i32, _> = TabuSearch::new(FixedTenure::new(5), 4, 2);
        assert_eq!(ts.memory().edge.len(), 16);
        assert_eq!(ts.memory().berth.len(), 8);
    }

    #[test]
    fn test_tabu_search_with_custom_aspiration() {
        #[derive(Debug)]
        struct AlwaysAspire;
        impl AspirationCriterion for AlwaysAspire {
            fn aspires<T: SolverNumeric>(&self, _cand: T, _best: T) -> bool {
                true
            }
        }

        let ts: TabuSearch<i32, _, AlwaysAspire> =
            TabuSearch::with_aspiration(FixedTenure::new(5), AlwaysAspire, 4, 2);
        assert!(ts.aspiration().aspires(999, 1));
    }

    #[test]
    fn test_tabu_search_debug_does_not_panic() {
        let ts: TabuSearch<i32, _> = TabuSearch::new(FixedTenure::new(5), 10, 3);
        let s = format!("{:?}", ts);
        assert!(s.contains("TabuSearch"));
        assert!(s.contains("FixedTenure"));
        assert!(s.contains("BestImprovement"));
        assert!(s.contains("KeepFirst"));
    }

    #[test]
    fn test_tabu_search_name() {
        let ts: TabuSearch<i32, _> = TabuSearch::new(FixedTenure::new(5), 4, 2);
        assert_eq!(Metaheuristic::<i32, NoOracle>::name(&ts), "TabuSearch");
    }
}
