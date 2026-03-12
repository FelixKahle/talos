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

//! Module providing composite local search operators that orchestrate multiple
//! neighborhood-specific operators under different selection strategies.
//!
//! A compound operator receives a set of `LocalSearchOperator` implementations and
//! delegates exploration to one operator at a time. This enables higher-level control
//! over how neighborhoods are visited, how effort is distributed, and how feedback from
//! previous iterations influences future choices. The design keeps each neighborhood
//! operator focused on generating reversible mutations while the compound operator
//! manages sequencing and state transitions across operators.
//!
//! The random strategy maintains stickiness within a single operator until it exhausts
//! its current neighborhood, then switches stochastically among the remaining choices.
//! The round-robin strategy cycles deterministically, advancing to the next operator
//! once the current one stops yielding neighbors, and restarting the cycle on reset.
//! The multi-armed bandit strategy ranks operators by a learned score derived from
//! observed objective improvements, blending exploitation of high-performing operators
//! with exploration to continually reassess opportunities. Its internal statistics track
//! per-operator performance and global improvements, and the ranking is updated as
//! new evidence arrives.
//!
//! These strategies allow you to balance breadth and depth of search across heterogeneous
//! neighborhoods, align operator selection with problem structure, and adapt dynamically
//! to changing effectiveness during a run, all while preserving a clear separation between
//! mutation generation and evaluation.

use crate::{
    mutator::Mutator, operator::lsoperator::LocalSearchOperator, sgraph::ScheduleGraph,
    stats::LocalSearchStatistics,
};
use rand::{Rng, RngExt, SeedableRng};
use std::cmp::Ordering;
use talos_core::utils::num::SolverNumeric;
use talos_model::{model::Model, solution::SolutionView};

// ----------------------------------------------------------------
// RandomCompoundOperator
// ----------------------------------------------------------------

/// A compound operator that selects one random sub-operator for each local search iteration.
///
/// Unlike a sequential compound operator, this operator introduces stochasticity by
/// picking a single random neighborhood strategy when `prepare` is called and sticking
/// with it until the next solution is found.
///
/// This is effective for:
/// - **Diversification**: Preventing the search from following deterministic paths.
/// - **Escape**: breaking out of local optima that a specific deterministic operator might get stuck in.
pub struct RandomCompoundOperator<'a, T, R>
where
    T: SolverNumeric,
    R: Rng,
{
    operators: Vec<Box<dyn LocalSearchOperator<T> + 'a>>,
    rng: R,
    current_index: usize,
    /// Tracks which operators have been prepared for the current solution.
    op_started: Vec<bool>,
}

impl<'a, T, R> RandomCompoundOperator<'a, T, R>
where
    T: SolverNumeric,
    R: Rng,
{
    pub fn new(operators: Vec<Box<dyn LocalSearchOperator<T> + 'a>>, rng: R) -> Self {
        assert!(!operators.is_empty(), "Operators list cannot be empty");
        let len = operators.len();
        Self {
            operators,
            rng,
            current_index: 0,
            op_started: vec![false; len],
        }
    }
}

impl<'a, T> RandomCompoundOperator<'a, T, rand::rngs::StdRng>
where
    T: SolverNumeric,
{
    #[inline]
    pub fn empty() -> Self {
        Self {
            operators: Vec::new(),
            rng: rand::rngs::StdRng::seed_from_u64(0),
            current_index: 0,
            op_started: Vec::new(),
        }
    }
}

impl<'a, T, R> LocalSearchOperator<T> for RandomCompoundOperator<'a, T, R>
where
    T: SolverNumeric,
    R: Rng + Send + Sync,
{
    fn name(&self) -> &str {
        "RandomCompoundOperator"
    }

    fn prepare(
        &mut self,
        _best_solution: SolutionView<'_, T>,
        _accepted_solution: SolutionView<'_, T>,
        _buffered_solution: Option<SolutionView<'_, T>>,
        _graph: &ScheduleGraph,
    ) {
        let n = self.operators.len();
        if n == 0 {
            return;
        }

        // Choose a random entry point for this iteration.
        self.current_index = self.rng.random_range(0..n);

        // Reset the "started" flags for the new solution state.
        self.op_started.fill(false);
    }

    unsafe fn next_neighbor(
        &mut self,
        model: &Model<T>,
        best_solution: SolutionView<'_, T>,
        accepted_solution: SolutionView<'_, T>,
        buffered_solution: Option<SolutionView<'_, T>>,
        mutator: &mut Mutator,
        stats: &LocalSearchStatistics,
    ) -> bool {
        let n = self.operators.len();
        if n == 0 {
            return false;
        }

        let start = self.current_index;
        let mut idx = start;

        loop {
            let op = unsafe { self.operators.get_unchecked_mut(idx) };

            // LAZY PREPARATION: Ensure the operator is prepared before its first call.
            if !unsafe { *self.op_started.get_unchecked(idx) } {
                op.prepare(
                    best_solution,
                    accepted_solution,
                    buffered_solution,
                    mutator.graph(),
                );
                unsafe { *self.op_started.get_unchecked_mut(idx) = true };
            }

            if unsafe {
                op.next_neighbor(
                    model,
                    best_solution,
                    accepted_solution,
                    buffered_solution,
                    mutator,
                    stats,
                )
            } {
                // Stick with this operator for as long as it yields neighbors.
                self.current_index = idx;
                return true;
            }

            // Move to next operator in a circular fashion.
            idx = (idx + 1) % n;
            if idx == start {
                return false;
            }
        }
    }

    fn reset(&mut self) {
        for op in &mut self.operators {
            op.reset();
        }
        self.current_index = 0;
        self.op_started.fill(false);
    }
}

// ----------------------------------------------------------------
// RoundRobinCompoundOperator
// ----------------------------------------------------------------

/// A compound operator that iterates through sub-operators in a fixed sequential order.
///
/// This operator provides a deterministic "Round Robin" strategy. It starts with the first
/// sub-operator and lets it run until it is exhausted (returns `false`). Then, it immediately
/// switches to the next operator in the list.
///
/// It returns `false` (exhausted) only when **all** sub-operators have been exhausted for
/// the current solution.
pub struct RoundRobinCompoundOperator<'a, T>
where
    T: SolverNumeric,
{
    operators: Vec<Box<dyn LocalSearchOperator<T> + 'a>>, // List of sub-operators
    current_index: usize,                                 // Index of the currently active operator
    op_started: Vec<bool>,                                // operator i started
}

impl<'a, T> RoundRobinCompoundOperator<'a, T>
where
    T: SolverNumeric,
{
    /// Creates a new `RoundRobinCompoundOperator`.
    pub fn new(operators: Vec<Box<dyn LocalSearchOperator<T> + 'a>>) -> Self {
        let len = operators.len();
        Self {
            operators,
            current_index: 0,
            op_started: vec![false; len],
        }
    }

    #[inline]
    pub fn empty() -> Self {
        Self {
            operators: Vec::new(),
            current_index: 0,
            op_started: Vec::new(),
        }
    }
}

impl<'a, T> LocalSearchOperator<T> for RoundRobinCompoundOperator<'a, T>
where
    T: SolverNumeric,
{
    fn name(&self) -> &str {
        "RoundRobinCompoundOperator"
    }

    fn prepare(
        &mut self,
        _best_solution: SolutionView<'_, T>,
        _accepted_solution: SolutionView<'_, T>,
        _buffered_solution: Option<SolutionView<'_, T>>,
        _graph: &ScheduleGraph,
    ) {
        self.current_index = 0;
        self.op_started.fill(false);
    }

    unsafe fn next_neighbor(
        &mut self,
        model: &Model<T>,
        best_solution: SolutionView<'_, T>,
        accepted_solution: SolutionView<'_, T>,
        buffered_solution: Option<SolutionView<'_, T>>,
        mutator: &mut Mutator,
        stats: &LocalSearchStatistics,
    ) -> bool {
        loop {
            if self.current_index >= self.operators.len() {
                return false;
            }

            debug_assert!(
                self.current_index < self.operators.len(),
                "current_index {} out of bounds for operators with len {}",
                self.current_index,
                self.operators.len()
            );
            debug_assert!(
                self.current_index < self.op_started.len(),
                "current_index {} out of bounds for op_started with len {}",
                self.current_index,
                self.op_started.len()
            );
            let op = unsafe { &mut self.operators.get_unchecked_mut(self.current_index) };
            if !*unsafe { self.op_started.get_unchecked(self.current_index) } {
                let graph = mutator.graph();
                op.prepare(best_solution, accepted_solution, buffered_solution, graph);
                unsafe { *self.op_started.get_unchecked_mut(self.current_index) = true };
            }

            if unsafe {
                op.next_neighbor(
                    model,
                    best_solution,
                    accepted_solution,
                    buffered_solution,
                    mutator,
                    stats,
                )
            } {
                return true;
            }

            self.current_index += 1;
        }
    }

    fn reset(&mut self) {
        for op in self.operators.iter_mut() {
            op.reset();
        }
        self.current_index = 0;
        self.op_started.fill(false);
    }
}

// ----------------------------------------------------------------
// BanditStats
// ----------------------------------------------------------------

/// Internal statistics for the Multi-Armed Bandit operator.
///
/// Tracks the number of samples and average improvements for each sub-operator.
/// Used to compute UCB1 scores for operator selection.
#[derive(Debug, Default)]
struct BanditStats {
    total_samples: usize,       // Total number of samples across all operators
    samples_per_op: Vec<usize>, // len = number of operators
    avg_improvements: Vec<f64>, // len = number of operators

    /// The maximum improvement seen so far across all operators.
    /// Used to normalize the exploitation term to the [0, 1] range.
    global_max_improvement: f64,
}

impl BanditStats {
    /// Creates a new `BanditStats` instance for the specified number of operators.
    #[inline]
    fn new(size: usize) -> Self {
        Self {
            total_samples: 0,
            samples_per_op: vec![0; size],
            avg_improvements: vec![0.0; size],
            // Initialize to 1.0 to prevent division by zero on the first iteration.
            // As soon as a real improvement > 1.0 is found, this will scale up.
            global_max_improvement: 1.0,
        }
    }

    /// Computes the UCB1 score for the specified operator index.
    ///
    /// # Panics
    ///
    /// In debug builds, this method will panic if `index` is not within `0..self.samples_per_op.len()`.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `index` is within `0..self.samples_per_op.len()`.
    unsafe fn get_score_unchecked(&self, index: usize, exploration_coef: f64) -> f64 {
        debug_assert!(
            index < self.samples_per_op.len(),
            "called `BanditStats::get_score_unchecked` with index out of bounds: the len is {} but the index is {}",
            self.samples_per_op.len(),
            index,
        );

        let n_i = unsafe { *self.samples_per_op.get_unchecked(index) as f64 };
        if n_i == 0.0 {
            return f64::INFINITY;
        }

        let total_n = self.total_samples as f64;

        let raw_exploitation = unsafe { *self.avg_improvements.get_unchecked(index) };
        let normalized_exploitation = raw_exploitation / self.global_max_improvement;
        let bonus = exploration_coef * ((2.0 * (1.0 + total_n).ln()) / n_i).sqrt();

        normalized_exploitation + bonus
    }

    /// Updates the average improvement for the specified operator index using
    /// the provided delta and learning rate (alpha).
    ///
    /// # Panics
    ///
    /// In debug builds, this method will panic if `index` is not within `0..self.avg_improvements.len()`.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `index` is within `0..self.avg_improvements.len()`.
    unsafe fn update_improvement_unchecked(&mut self, index: usize, delta: f64, alpha: f64) {
        debug_assert!(
            index < self.avg_improvements.len(),
            "called `BanditStats::update_improvement_unchecked` with index out of bounds: the len is {} but the index is {}",
            self.avg_improvements.len(),
            index,
        );

        // Track the global maximum improvement to maintain correct normalization scaling
        if delta > self.global_max_improvement {
            self.global_max_improvement = delta;
        }

        let current_avg = unsafe { *self.avg_improvements.get_unchecked(index) };
        let new_avg = current_avg + alpha * (delta - current_avg);
        unsafe {
            *self.avg_improvements.get_unchecked_mut(index) = new_avg;
        }
    }
}

// ----------------------------------------------------------------
// MultiArmedBanditCompoundOperator
// ----------------------------------------------------------------

/// A compound operator that selects sub-operators using a Multi-Armed Bandit (UCB1) strategy.
pub struct MultiArmedBanditCompoundOperator<'a, T>
where
    T: SolverNumeric,
{
    operators: Vec<Box<dyn LocalSearchOperator<T> + 'a>>, // List of sub-operators
    memory_coeff: f64,      // Learning rate for updating average improvements
    exploration_coeff: f64, // Coefficient for exploration bonus in UCB1

    // State
    stats: BanditStats,         // Internal statistics for operator selection
    ranked_indices: Vec<usize>, // Indices of operators sorted by UCB1 score
    active_rank_index: usize,   // Cursor in ranked_indices for the current operator
    op_started: Vec<bool>,      // operator i started
    last_obj: Option<T>,        // Objective value of the last accepted solution
}

impl<'a, T> MultiArmedBanditCompoundOperator<'a, T>
where
    T: SolverNumeric,
{
    /// Creates a new `MultiArmedBanditCompoundOperator` with the specified sub-operators
    /// and configuration parameters.
    #[inline]
    pub fn new(
        operators: Vec<Box<dyn LocalSearchOperator<T> + 'a>>,
        memory_coeff: f64,
        exploration_coeff: f64,
    ) -> Self {
        let n = operators.len();
        Self {
            operators,
            memory_coeff,
            exploration_coeff,
            stats: BanditStats::new(n),
            ranked_indices: (0..n).collect(),
            active_rank_index: 0,
            op_started: vec![false; n],
            last_obj: None,
        }
    }

    #[inline]
    pub fn empty(memory_coeff: f64, exploration_coeff: f64) -> Self {
        Self::new(vec![], memory_coeff, exploration_coeff)
    }

    #[inline]
    pub fn with_defaults(operators: Vec<Box<dyn LocalSearchOperator<T>>>) -> Self {
        // memory_coeff: 0.2 (React faster to changes in operator effectiveness)
        // exploration_coeff: 1.414 (Standard UCB1 constant)
        Self::new(operators, 0.2, std::f64::consts::SQRT_2)
    }

    fn sort_by_score(&mut self) {
        let c = self.exploration_coeff;
        let stats = &self.stats;
        self.ranked_indices.sort_by(|&a, &b| unsafe {
            stats
                .get_score_unchecked(b, c)
                .partial_cmp(&stats.get_score_unchecked(a, c))
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.cmp(&b))
        });
    }
}

impl<'a, T> LocalSearchOperator<T> for MultiArmedBanditCompoundOperator<'a, T>
where
    T: SolverNumeric,
{
    fn name(&self) -> &str {
        "MultiArmedBanditCompoundOperator"
    }

    fn prepare(
        &mut self,
        _best_solution: SolutionView<'_, T>,
        accepted_solution: SolutionView<'_, T>,
        _buffered_solution: Option<SolutionView<'_, T>>,
        _graph: &ScheduleGraph,
    ) {
        let current_obj = accepted_solution.objective_value();

        // Learning Phase: Update stats based on the last accepted move
        if let Some(prev_obj) = self.last_obj {
            let improvement =
                (prev_obj.to_f64().unwrap_or(0.0) - current_obj.to_f64().unwrap_or(0.0)).max(0.0);

            // Note: We update even if improvement is 0.0 to decay the average
            if improvement >= 0.0 {
                debug_assert!(
                    self.active_rank_index < self.ranked_indices.len(),
                    "active_rank_index {} out of bounds for ranked_indices with len {}",
                    self.active_rank_index,
                    self.ranked_indices.len()
                );

                let last_op_idx =
                    *unsafe { self.ranked_indices.get_unchecked(self.active_rank_index) };
                unsafe {
                    self.stats.update_improvement_unchecked(
                        last_op_idx,
                        improvement,
                        self.memory_coeff,
                    )
                };
            }
        }
        self.last_obj = Some(current_obj);

        // Strategy Phase: Re-rank operators and reset cursors
        self.sort_by_score();
        self.active_rank_index = 0;
        self.op_started.fill(false);
    }

    unsafe fn next_neighbor(
        &mut self,
        model: &Model<T>,
        best_solution: SolutionView<'_, T>,
        accepted_solution: SolutionView<'_, T>,
        buffered_solution: Option<SolutionView<'_, T>>,
        mutator: &mut Mutator,
        stats: &LocalSearchStatistics,
    ) -> bool {
        let num_ops = self.operators.len();
        if num_ops == 0 {
            return false;
        }

        let start_rank = self.active_rank_index;
        loop {
            debug_assert!(
                self.active_rank_index < self.ranked_indices.len(),
                "active_rank_index {} out of bounds for ranked_indices with len {}",
                self.active_rank_index,
                self.ranked_indices.len()
            );
            let op_idx = *unsafe { self.ranked_indices.get_unchecked(self.active_rank_index) };

            debug_assert!(
                op_idx < self.operators.len(),
                "op_idx {} out of bounds for operators with len {}",
                op_idx,
                self.operators.len()
            );
            let op = unsafe { &mut self.operators.get_unchecked_mut(op_idx) };

            debug_assert!(
                op_idx < self.op_started.len(),
                "op_idx {} out of bounds for op_started with len {}",
                op_idx,
                self.op_started.len()
            );

            // Lazy preparation
            if !*unsafe { self.op_started.get_unchecked(op_idx) } {
                op.prepare(
                    best_solution,
                    accepted_solution,
                    buffered_solution,
                    mutator.graph(),
                );
                // SAFETY: op_index is always in bounds due to struct invariants.
                unsafe { *self.op_started.get_unchecked_mut(op_idx) = true };
            }

            if unsafe {
                op.next_neighbor(
                    model,
                    best_solution,
                    accepted_solution,
                    buffered_solution,
                    mutator,
                    stats,
                )
            } {
                self.stats.total_samples += 1;
                debug_assert!(
                    op_idx < self.stats.samples_per_op.len(),
                    "op_idx {} out of bounds for samples_per_op with len {}",
                    op_idx,
                    self.stats.samples_per_op.len()
                );

                // SAFETY: op_index is always in bounds due to struct invariants.
                unsafe { *self.stats.samples_per_op.get_unchecked_mut(op_idx) += 1 };
                return true;
            }

            // Move to next operator in ranked order
            self.active_rank_index = (self.active_rank_index + 1) % num_ops;
            if self.active_rank_index == start_rank {
                return false;
            }
        }
    }

    fn reset(&mut self) {
        self.operators.iter_mut().for_each(|op| op.reset());
        self.active_rank_index = 0;
        self.op_started.fill(false);
    }
}
