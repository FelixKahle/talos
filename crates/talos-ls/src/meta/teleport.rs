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

//! Teleport policy for oracle-based solution injection.
//!
//! A [`TeleportPolicy`] controls *when* a metaheuristic should attempt to
//! teleport to a solution from the global oracle. The policy tracks
//! stagnation (time without improvement) and signals when the search
//! should jump to a globally shared solution.
//!
//! # Built-in Policies
//!
//! | Policy                | Behaviour                                            |
//! |-----------------------|------------------------------------------------------|
//! | [`NoTeleport`]        | Never teleports (default for all metaheuristics).    |
//! | [`StagnationTeleport`]| Teleports after a configurable duration without      |
//! |                       | improvement in the local best objective.              |

use std::time::{Duration, Instant};

use talos_core::utils::num::SolverNumeric;
use talos_search::oracle::GlobalOracle;

// ----------------------------------------------------------------
// TeleportPolicy Trait
// ----------------------------------------------------------------

/// Controls when a metaheuristic should teleport to an oracle solution.
///
/// Metaheuristics compose a `TeleportPolicy` and check it in
/// [`on_neighbourhood_exhausted`](super::metaheuristic::Metaheuristic::on_neighbourhood_exhausted).
/// When the policy signals `should_teleport()`, the metaheuristic queries
/// the oracle for a solution strictly better than its local best and
/// returns [`Teleport`](super::metaheuristic::NeighborhoodExhaustionOutcome::Teleport).
pub trait TeleportPolicy: std::fmt::Debug {
    /// Called when the search starts. Resets internal timers.
    fn on_start(&mut self);

    /// Called when a new best solution is found locally.
    fn on_improvement(&mut self);

    /// Called each time the neighborhood is exhausted.
    ///
    /// Exhaustion-count policies use this to increment their counter.
    /// The default is a no-op.
    fn on_exhaustion(&mut self) {}

    /// Returns `true` if the search should attempt to teleport.
    fn should_teleport(&self) -> bool;

    /// Called after a successful teleport. Resets stagnation tracking.
    fn on_teleport(&mut self);
}

// ----------------------------------------------------------------
// NoTeleport
// ----------------------------------------------------------------

/// No-op teleport policy — never triggers teleportation.
///
/// This is the default for all metaheuristics, preserving their
/// standard behaviour.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoTeleport;

impl TeleportPolicy for NoTeleport {
    #[inline]
    fn on_start(&mut self) {}
    #[inline]
    fn on_improvement(&mut self) {}
    #[inline]
    fn should_teleport(&self) -> bool {
        false
    }
    #[inline]
    fn on_teleport(&mut self) {}
}

// ----------------------------------------------------------------
// StagnationTeleport
// ----------------------------------------------------------------

/// Triggers teleportation after a configurable duration without improvement.
///
/// Once `patience` elapses since the last local improvement (or search
/// start), [`should_teleport()`](TeleportPolicy::should_teleport) returns
/// `true`. After a successful teleport the timer resets.
///
/// # Example
///
/// ```ignore
/// use std::time::Duration;
/// use talos_ls::meta::teleport::StagnationTeleport;
/// use talos_ls::meta::sa::SimulatedAnnealing;
///
/// let sa = SimulatedAnnealing::new(cooling, rng)
///     .with_teleport(StagnationTeleport::new(Duration::from_secs(10)));
/// ```
#[derive(Debug, Clone)]
pub struct StagnationTeleport {
    /// Maximum time without improvement before teleporting.
    patience: Duration,

    /// Instant of the last improvement (or search start).
    last_improvement: Instant,
}

impl StagnationTeleport {
    /// Creates a new stagnation-based teleport policy.
    ///
    /// # Arguments
    ///
    /// * `patience` — Duration without improvement that triggers teleportation.
    #[inline]
    pub fn new(patience: Duration) -> Self {
        Self {
            patience,
            last_improvement: Instant::now(),
        }
    }

    /// Returns the configured patience duration.
    #[inline]
    pub fn patience(&self) -> Duration {
        self.patience
    }
}

impl TeleportPolicy for StagnationTeleport {
    #[inline]
    fn on_start(&mut self) {
        self.last_improvement = Instant::now();
    }

    #[inline]
    fn on_improvement(&mut self) {
        self.last_improvement = Instant::now();
    }

    #[inline]
    fn should_teleport(&self) -> bool {
        self.last_improvement.elapsed() >= self.patience
    }

    #[inline]
    fn on_teleport(&mut self) {
        self.last_improvement = Instant::now();
    }
}

/// Triggers teleportation after a fixed number of neighborhood exhaustions without improvement.
#[derive(Debug, Clone)]
pub struct ExhaustionTeleport {
    limit: u64,
    current_stagnation: u64,
}

impl ExhaustionTeleport {
    pub fn new(limit: u64) -> Self {
        Self {
            limit,
            current_stagnation: 0,
        }
    }

    pub fn instant() -> Self {
        Self::new(1)
    }
}

impl TeleportPolicy for ExhaustionTeleport {
    #[inline]
    fn on_start(&mut self) {
        self.current_stagnation = 0;
    }

    #[inline]
    fn on_improvement(&mut self) {
        self.current_stagnation = 0;
    }

    #[inline]
    fn on_exhaustion(&mut self) {
        self.current_stagnation += 1;
    }

    #[inline]
    fn should_teleport(&self) -> bool {
        self.current_stagnation >= self.limit
    }

    #[inline]
    fn on_teleport(&mut self) {
        self.current_stagnation = 0;
    }
}

// ----------------------------------------------------------------
// Helper
// ----------------------------------------------------------------

/// Checks whether the metaheuristic should attempt a teleport.
///
/// Calls [`TeleportPolicy::on_exhaustion`] to advance the stagnation
/// counter, then returns `true` if:
///
/// 1. The policy says to teleport ([`TeleportPolicy::should_teleport`]).
/// 2. The oracle holds a solution strictly better than `best_objective`
///    (lock-free check).
///
/// Does *not* extract a solution or reset the policy. The engine is
/// responsible for fetching the solution directly into its internal
/// buffers and then calling [`TeleportPolicy::on_teleport`] (via
/// [`Metaheuristic::on_teleport`]) on success.
pub fn should_attempt_teleport<T, G, Tp>(policy: &mut Tp, oracle: &G, best_objective: T) -> bool
where
    T: SolverNumeric,
    G: GlobalOracle<T>,
    Tp: TeleportPolicy,
{
    policy.on_exhaustion();
    if !policy.should_teleport() {
        return false;
    }
    let Some(oracle_best) = oracle.best_objective() else {
        return false;
    };
    oracle_best < best_objective
}
