// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Statistics reporting for local search runs.
//!
//! This module defines a lightweight container for tracking aggregate metrics
//! during a local search, including iteration count, number of candidates found,
//! number of accepted moves, and total elapsed time. The interface is optimized
//! for hot-loop usage: updates rely on saturating arithmetic to avoid overflow
//! traps and expose clear, inline methods for per-iteration and per-event
//! accounting. The resulting `LocalSearchStatistics` can be consumed by monitors,
//! metaheuristics, and result reporting to provide visibility into solver progress
//! and convergence behavior without imposing measurable overhead on the inner loop.

use std::time::Duration;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LocalSearchStatistics {
    /// Number of iterations (individual neighbors evaluated) performed by the local search.
    pub iterations: u64,

    /// Number of cycles (full neighborhood traversals) performed by the local search.
    pub cycles: u64,

    /// Total number of solutions found during the local search.
    pub total_solutions: u64,

    /// Number of accepted solutions during the local search.
    pub accepted_solutions: u64,

    /// Number of moves that were mutated but failed delta-decoding (infeasible).
    pub infeasible_moves: u64,

    /// Total time taken by the local search.
    pub time_total: Duration,
}

impl Default for LocalSearchStatistics {
    fn default() -> Self {
        Self {
            iterations: 0,
            cycles: 0,
            total_solutions: 0,
            accepted_solutions: 0,
            infeasible_moves: 0,
            time_total: Duration::ZERO,
        }
    }
}

impl LocalSearchStatistics {
    /// Called at each iteration of the local search.
    #[inline]
    pub fn on_iteration(&mut self) {
        self.iterations += 1;
    }

    /// Called at the end of each cycle (full neighborhood traversal).
    #[inline]
    pub fn on_cycle(&mut self) {
        self.cycles += 1;
    }

    /// Called when a solution is found.
    #[inline]
    pub fn on_found_solution(&mut self) {
        self.total_solutions += 1;
    }

    /// Called when a solution is accepted.
    #[inline]
    pub fn on_accepted_solution(&mut self) {
        self.accepted_solutions += 1;
    }

    /// Called when a mutation produced an infeasible candidate (delta-decode failed).
    #[inline]
    pub fn on_infeasible_move(&mut self) {
        self.infeasible_moves += 1;
    }

    /// Sets the total time taken by the local search.
    #[inline]
    pub fn set_total_time(&mut self, duration: Duration) {
        self.time_total = duration;
    }

    #[inline]
    pub fn rejected_solutions(&self) -> u64 {
        // This will never overflow since accepted_solutions <= total_solutions by definition.
        self.total_solutions - self.accepted_solutions
    }
}

impl std::fmt::Display for LocalSearchStatistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Bollard-LS-Solver Statistics:")?;
        writeln!(f, "   Iterations:           {}", self.iterations)?;
        writeln!(f, "   Cycles:               {}", self.cycles)?;
        writeln!(f, "   Total Solutions:     {}", self.total_solutions)?;
        writeln!(f, "   Accepted Solutions:  {}", self.accepted_solutions)?;
        writeln!(f, "   Rejected Solutions:  {}", self.rejected_solutions())?;
        writeln!(f, "   Infeasible Moves:   {}", self.infeasible_moves)?;
        writeln!(f, "   Total Time:         {:?}", self.time_total)?;
        Ok(())
    }
}
