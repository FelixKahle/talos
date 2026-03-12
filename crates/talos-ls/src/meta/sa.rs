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

//! Simulated Annealing metaheuristic for local search.
//!
//! Simulated Annealing (SA) stochastically accepts worsening moves to escape
//! local optima, with a decreasing probability governed by a temperature parameter.
//! As the temperature cools, the search transitions from exploratory (accepting
//! many worsening moves) to exploitative (strict descent).
//!
//! # Acceptance Criterion
//!
//! SA uses the Metropolis criterion. For a minimization problem:
//!
//! - If $f(s') < f(s)$: accept unconditionally.
//! - If $f(s') \geq f(s)$: accept with probability
//!   $$P = \exp\!\Bigl(\frac{-(f(s') - f(s))}{T}\Bigr)$$
//!
//! where $T$ is the current temperature. When $T \to 0$, only strict improvements
//! are accepted and the search degenerates into a hill climber.
//!
//! # Engine Integration
//!
//! SA is a **first-improvement** strategy: it decides the fate of every candidate
//! immediately. It never returns `AcceptanceOutcome::Buffer` — each move is either
//! accepted or rejected on the spot. The `Metaheuristic::should_commit_buffered`
//! hook always returns `false`, and `Metaheuristic::on_neighbourhood_exhausted`
//! returns `NeighborhoodExhaustionOutcome::Terminate` (or
//! `NeighborhoodExhaustionOutcome::Restart` when reheating is enabled).
//!
//! The cooling schedule is advanced according to the configured `CoolingTrigger`:
//! per iteration (default), per cycle (neighbourhood exhaustion), or on move
//! acceptance.
//!
//! # Acceptance Criteria
//!
//! The acceptance criterion is pluggable via the `AcceptanceCriterion` trait.
//! The default is `MetropolisCriterion` ($P = \exp(-\Delta / T)$). Custom
//! criteria (e.g., threshold accepting) can be supplied via
//! `SimulatedAnnealing::with_criterion`.
//!
//! # Cooling Schedules
//!
//! Three built-in schedules are provided:
//!
//! | Schedule       | Update rule                          | Character                       |
//! |----------------|--------------------------------------|---------------------------------|
//! | Geometric      | $T_{k+1} = \alpha \cdot T_k$         | Fast early decay, slow settling |
//! | Linear         | $T_{k+1} = T_k - \delta$             | Constant decay, deadline-aware  |
//! | Logarithmic    | $T_k = C / \ln(k \cdot s + e)$       | Provably optimal, very slow     |
//!
//! All schedules expose an `is_frozen()` check: once the temperature drops below
//! a minimum threshold, the Metropolis criterion is bypassed entirely and only
//! strict improvements are accepted.
//!
//! # Reheating
//!
//! An optional reheat factor can be specified at construction. When enabled,
//! `Metaheuristic::on_neighbourhood_exhausted` multiplies the current
//! temperature by the reheat factor and returns
//! `NeighborhoodExhaustionOutcome::Restart`, allowing the search to escape a
//! basin by temporarily accepting worsening moves again. Set to `None` to
//! disable reheating (the default), in which case neighbourhood exhaustion
//! terminates the search.

use crate::{
    exec::SearchCommand,
    meta::metaheuristic::{AcceptanceOutcome, Metaheuristic, NeighborhoodExhaustionOutcome},
    sgraph::{ScheduleGraph, ScheduleGraphDiff},
};
use num_traits::ToPrimitive;
use rand::{Rng, RngExt};
use talos_core::utils::num::SolverNumeric;
use talos_model::{model::Model, solution::SolutionView};

// ──────────────────────────────────────────────────────────────
// Acceptance Criteria
// ──────────────────────────────────────────────────────────────

/// Computes the acceptance probability for a worsening move.
///
/// The trait receives the objective delta and the current temperature, and
/// returns a probability in $[0, 1]$. The SA engine flips a coin against
/// this probability to decide acceptance. Improving and equal moves bypass
/// the criterion entirely.
pub trait AcceptanceCriterion: std::fmt::Debug {
    /// Returns the probability of accepting a worsening move.
    ///
    /// # Arguments
    ///
    /// * `delta` — The objective increase ($f' - f$, always positive).
    /// * `temperature` — The current temperature from the cooling schedule.
    fn acceptance_probability(&self, delta: f64, temperature: f64) -> f64;
}

/// The classic Metropolis criterion: $P = \exp(-\Delta / T)$.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct MetropolisCriterion;

impl AcceptanceCriterion for MetropolisCriterion {
    #[inline]
    fn acceptance_probability(&self, delta: f64, temperature: f64) -> f64 {
        (-delta / temperature).exp()
    }
}

// ──────────────────────────────────────────────────────────────
// Cooling Schedules
// ──────────────────────────────────────────────────────────────

/// Defines the thermodynamics of the annealing process.
///
/// Implementors control the initial temperature, the decay function, and the
/// "frozen" threshold below which worsening moves are no longer considered.
pub trait CoolingSchedule: std::fmt::Debug {
    /// Resets the temperature to its initial state.
    fn reset(&mut self);

    /// Advances the temperature by one step.
    fn step(&mut self);

    /// Returns the current temperature.
    fn temperature(&self) -> f64;

    /// Returns `true` if the temperature has dropped below the frozen threshold.
    fn is_frozen(&self) -> bool;

    /// Multiplies the current temperature by `factor`.
    ///
    /// Used for reheating. The default implementation is a no-op for schedules
    /// that don't support mutation of their internal state (e.g., logarithmic).
    fn reheat(&mut self, _factor: f64) {}
}

// ──────────────────────────────────────────────────────────────
// Geometric Cooling
// ──────────────────────────────────────────────────────────────

/// Geometric cooling: $T_{k+1} = \alpha \cdot T_k$.
///
/// The most common schedule in the literature. Cools rapidly at first and
/// asymptotically approaches zero, allowing fine-grained settling in the
/// final phases.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeometricCooling {
    initial: f64,
    current: f64,
    alpha: f64,
    min_temp: f64,
}

impl GeometricCooling {
    /// Creates a new geometric cooling schedule.
    ///
    /// # Panics
    ///
    /// Panics if `alpha` is not in the open interval $(0, 1)$.
    #[inline]
    pub fn new(initial: f64, alpha: f64, min_temp: f64) -> Self {
        assert!(
            alpha > 0.0 && alpha < 1.0,
            "GeometricCooling: alpha must be in (0, 1), got {}",
            alpha
        );
        assert!(
            initial > 0.0,
            "GeometricCooling: initial temperature must be positive, got {}",
            initial
        );
        Self {
            initial,
            current: initial,
            alpha,
            min_temp,
        }
    }

    /// Returns the decay rate $\alpha$.
    #[inline]
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Returns the initial temperature.
    #[inline]
    pub fn initial_temperature(&self) -> f64 {
        self.initial
    }

    /// Returns the frozen threshold.
    #[inline]
    pub fn min_temperature(&self) -> f64 {
        self.min_temp
    }
}

impl CoolingSchedule for GeometricCooling {
    #[inline]
    fn reset(&mut self) {
        self.current = self.initial;
    }

    #[inline]
    fn step(&mut self) {
        self.current *= self.alpha;
    }

    #[inline]
    fn temperature(&self) -> f64 {
        self.current
    }

    #[inline]
    fn is_frozen(&self) -> bool {
        self.current <= self.min_temp
    }

    #[inline]
    fn reheat(&mut self, factor: f64) {
        self.current = (self.current * factor).min(self.initial);
    }
}

// ──────────────────────────────────────────────────────────────
// Linear Cooling
// ──────────────────────────────────────────────────────────────

/// Linear cooling: $T_{k+1} = \max(0, T_k - \delta)$.
///
/// Reduces the temperature by a fixed decrement each step. Useful when the
/// iteration budget is known in advance and you want the temperature to reach
/// zero at a predictable time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearCooling {
    initial: f64,
    current: f64,
    decrement: f64,
    min_temp: f64,
}

impl LinearCooling {
    /// Creates a new linear cooling schedule.
    #[inline]
    pub fn new(initial: f64, decrement: f64, min_temp: f64) -> Self {
        assert!(initial > 0.0, "LinearCooling: initial must be positive");
        assert!(decrement > 0.0, "LinearCooling: decrement must be positive");
        Self {
            initial,
            current: initial,
            decrement,
            min_temp,
        }
    }

    /// Constructs a linear schedule that reaches `min_temp` after exactly
    /// `total_iterations` steps.
    #[inline]
    pub fn from_budget(initial: f64, min_temp: f64, total_iterations: u64) -> Self {
        let decrement = (initial - min_temp) / total_iterations.max(1) as f64;
        Self::new(initial, decrement, min_temp)
    }
}

impl CoolingSchedule for LinearCooling {
    #[inline]
    fn reset(&mut self) {
        self.current = self.initial;
    }

    #[inline]
    fn step(&mut self) {
        self.current = (self.current - self.decrement).max(0.0);
    }

    #[inline]
    fn temperature(&self) -> f64 {
        self.current
    }

    #[inline]
    fn is_frozen(&self) -> bool {
        self.current <= self.min_temp
    }

    #[inline]
    fn reheat(&mut self, factor: f64) {
        self.current = (self.current * factor).min(self.initial);
    }
}

// ──────────────────────────────────────────────────────────────
// Logarithmic Cooling
// ──────────────────────────────────────────────────────────────

/// Logarithmic cooling: $T_k = C / \ln(k \cdot s + e)$.
///
/// Theoretically guaranteed to find the global optimum given infinite time,
/// but decays extremely slowly. Best used for fine-tuning near a known good
/// solution or as a baseline for academic comparisons.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogarithmicCooling {
    constant: f64,
    k_scale: f64,
    iteration: u64,
    min_temp: f64,
}

impl LogarithmicCooling {
    /// Creates a new logarithmic cooling schedule.
    ///
    /// # Arguments
    ///
    /// * `constant` — The numerator $C$. Controls the initial temperature magnitude.
    /// * `k_scale` — Scaling factor for iteration count. Higher = faster cooling.
    /// * `min_temp` — The frozen threshold.
    #[inline]
    pub fn new(constant: f64, k_scale: f64, min_temp: f64) -> Self {
        Self {
            constant,
            k_scale,
            iteration: 0,
            min_temp,
        }
    }
}

impl CoolingSchedule for LogarithmicCooling {
    #[inline]
    fn reset(&mut self) {
        self.iteration = 0;
    }

    #[inline]
    fn step(&mut self) {
        self.iteration = self.iteration.saturating_add(1);
    }

    #[inline]
    fn temperature(&self) -> f64 {
        let denom = (self.iteration as f64 * self.k_scale + std::f64::consts::E).ln();
        self.constant / denom
    }

    #[inline]
    fn is_frozen(&self) -> bool {
        self.temperature() <= self.min_temp
    }
}

// ──────────────────────────────────────────────────────────────
// Cooling Trigger
// ──────────────────────────────────────────────────────────────

/// Controls *when* the cooling schedule is advanced.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CoolingTrigger {
    /// Cool once per iteration (the most common choice).
    Iteration,

    /// Cool once per cycle, i.e. when the neighbourhood is exhausted.
    Cycle,

    /// Cool whenever a move is accepted.
    Acceptance,
}

impl std::fmt::Display for CoolingTrigger {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CoolingTrigger::Iteration => write!(f, "Iteration"),
            CoolingTrigger::Cycle => write!(f, "Cycle"),
            CoolingTrigger::Acceptance => write!(f, "Acceptance"),
        }
    }
}

// ──────────────────────────────────────────────────────────────
// Simulated Annealing
// ──────────────────────────────────────────────────────────────

/// Simulated Annealing metaheuristic with a pluggable cooling schedule.
///
/// See the [module-level documentation](self) for algorithmic details.
pub struct SimulatedAnnealing<R, C, A = MetropolisCriterion> {
    /// The cooling schedule managing temperature decay.
    cooling: C,

    /// Random number generator for the stochastic acceptance check.
    rng: R,

    /// The acceptance criterion used to evaluate worsening moves.
    criterion: A,

    /// Optional reheat factor. When `Some(f)`, neighbourhood exhaustion
    /// multiplies the temperature by `f` and restarts. When `None`,
    /// exhaustion terminates the search.
    reheat_factor: Option<f64>,

    /// Temperatures at or below this threshold are treated as frozen,
    /// preventing division-by-zero in the acceptance criterion.
    frozen_threshold: f64,

    /// Determines when the cooling schedule is stepped.
    cooling_trigger: CoolingTrigger,
}

impl<R: std::fmt::Debug, C: std::fmt::Debug, A: std::fmt::Debug> std::fmt::Debug
    for SimulatedAnnealing<R, C, A>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SimulatedAnnealing")
            .field("cooling", &self.cooling)
            .field("criterion", &self.criterion)
            .field("reheat_factor", &self.reheat_factor)
            .field("frozen_threshold", &self.frozen_threshold)
            .field("cooling_trigger", &self.cooling_trigger)
            .finish()
    }
}

impl<R, C> SimulatedAnnealing<R, C, MetropolisCriterion>
where
    R: Rng,
    C: CoolingSchedule,
{
    /// Creates a new SA instance with the default `MetropolisCriterion`.
    ///
    /// # Arguments
    ///
    /// * `cooling` — The cooling schedule (e.g., `GeometricCooling`).
    /// * `rng` — Random number generator for stochastic acceptance.
    #[inline]
    pub fn new(cooling: C, rng: R) -> Self {
        Self {
            cooling,
            rng,
            criterion: MetropolisCriterion,
            reheat_factor: None,
            frozen_threshold: Self::DEFAULT_FROZEN_THRESHOLD,
            cooling_trigger: CoolingTrigger::Iteration,
        }
    }
}

impl<R, C, A> SimulatedAnnealing<R, C, A>
where
    R: Rng,
    C: CoolingSchedule,
    A: AcceptanceCriterion,
{
    /// Default frozen threshold used when none is specified.
    const DEFAULT_FROZEN_THRESHOLD: f64 = 1e-12;

    /// Creates a new SA instance with a custom acceptance criterion.
    ///
    /// # Arguments
    ///
    /// * `cooling` — The cooling schedule (e.g., `GeometricCooling`).
    /// * `rng` — Random number generator for stochastic acceptance.
    /// * `criterion` — The acceptance criterion (e.g., `MetropolisCriterion`).
    #[inline]
    pub fn with_criterion(cooling: C, rng: R, criterion: A) -> Self {
        Self {
            cooling,
            rng,
            criterion,
            reheat_factor: None,
            frozen_threshold: Self::DEFAULT_FROZEN_THRESHOLD,
            cooling_trigger: CoolingTrigger::Iteration,
        }
    }

    /// Enables reheating on neighbourhood exhaustion.
    ///
    /// When the operator runs out of neighbours, the temperature is multiplied
    /// by `factor` (e.g., 2.0 to double it) and the search restarts.
    #[inline]
    pub fn with_reheat(mut self, factor: f64) -> Self {
        assert!(
            factor > 1.0,
            "SimulatedAnnealing: reheat factor must be > 1.0, got {}",
            factor
        );
        self.reheat_factor = Some(factor);
        self
    }

    /// Sets the frozen threshold below which worsening moves are always
    /// rejected (bypassing the acceptance criterion).
    #[inline]
    pub fn with_frozen_threshold(mut self, threshold: f64) -> Self {
        assert!(
            threshold >= 0.0,
            "SimulatedAnnealing: frozen threshold must be >= 0.0, got {}",
            threshold
        );
        self.frozen_threshold = threshold;
        self
    }

    /// Sets when the cooling schedule is stepped.
    #[inline]
    pub fn with_cooling_trigger(mut self, trigger: CoolingTrigger) -> Self {
        self.cooling_trigger = trigger;
        self
    }

    /// Returns the current frozen threshold.
    #[inline]
    pub fn frozen_threshold(&self) -> f64 {
        self.frozen_threshold
    }

    /// Returns the current cooling trigger.
    #[inline]
    pub fn cooling_trigger(&self) -> CoolingTrigger {
        self.cooling_trigger
    }

    /// Returns a reference to the acceptance criterion.
    #[inline]
    pub fn criterion(&self) -> &A {
        &self.criterion
    }

    /// Returns a mutable reference to the acceptance criterion.
    #[inline]
    pub fn criterion_mut(&mut self) -> &mut A {
        &mut self.criterion
    }

    /// Returns a reference to the cooling schedule.
    #[inline]
    pub fn cooling(&self) -> &C {
        &self.cooling
    }

    /// Returns a mutable reference to the cooling schedule.
    #[inline]
    pub fn cooling_mut(&mut self) -> &mut C {
        &mut self.cooling
    }

    /// Constructs a `GeometricCooling` schedule auto-tuned from the initial objective.
    ///
    /// The initial temperature $T_0$ is chosen so that a worsening move of
    /// magnitude $\Delta E = \text{sensitivity} \times f(s_0)$ is accepted with
    /// probability `initial_acceptance_prob`:
    ///
    /// $$T_0 = \frac{-\Delta E}{\ln(P_0)}$$
    ///
    /// The frozen threshold is set where that same move would be accepted with
    /// probability $10^{-4}$.
    ///
    /// # Arguments
    ///
    /// * `objective` — The initial solution's objective value.
    /// * `initial_acceptance_prob` — Target acceptance probability at $T_0$ (e.g., 0.5).
    /// * `sensitivity` — Fraction of the objective representing a "typical bad move" (e.g., 0.01).
    /// * `cooling_rate` — Geometric decay factor $\alpha$ (e.g., 0.9999).
    pub fn heuristic_geometric_params(
        objective: f64,
        initial_acceptance_prob: f64,
        sensitivity: f64,
        cooling_rate: f64,
    ) -> GeometricCooling {
        let delta_e = objective.abs() * sensitivity;
        let p = initial_acceptance_prob.clamp(0.001, 0.999);

        let t0 = -delta_e / p.ln();
        let t_min = -delta_e / (1e-4_f64).ln();

        GeometricCooling::new(t0, cooling_rate, t_min)
    }
}

// ──────────────────────────────────────────────────────────────
// Metaheuristic Implementation
// ──────────────────────────────────────────────────────────────

impl<T, R, C, A> Metaheuristic<T> for SimulatedAnnealing<R, C, A>
where
    T: SolverNumeric + ToPrimitive,
    R: Rng,
    C: CoolingSchedule,
    A: AcceptanceCriterion,
{
    fn name(&self) -> &str {
        "SimulatedAnnealing"
    }

    fn on_start(
        &mut self,
        _model: &Model<T>,
        _initial_solution: SolutionView<T>,
        _graph: &ScheduleGraph,
    ) {
        self.cooling.reset();
    }

    fn on_end(
        &mut self,
        _model: &Model<T>,
        _final_solution: SolutionView<T>,
        _graph: &ScheduleGraph,
    ) {
    }

    /// If reheating is enabled, multiplies the temperature by the reheat factor
    /// and returns `NeighborhoodExhaustionOutcome::Restart`. Otherwise returns
    /// `NeighborhoodExhaustionOutcome::Terminate`.
    ///
    /// When `CoolingTrigger::Cycle` is active, the cooling schedule is also
    /// stepped here.
    fn on_neighbourhood_exhausted(
        &mut self,
        _model: &Model<T>,
        _best_solution: SolutionView<'_, T>,
        _accepted_solution: SolutionView<'_, T>,
        _buffered_solution: Option<SolutionView<'_, T>>,
        _graph: &ScheduleGraph,
    ) -> NeighborhoodExhaustionOutcome {
        if self.cooling_trigger == CoolingTrigger::Cycle {
            self.cooling.step();
        }

        match self.reheat_factor {
            Some(factor) => {
                self.cooling.reheat(factor);
                NeighborhoodExhaustionOutcome::Restart
            }
            None => NeighborhoodExhaustionOutcome::Terminate,
        }
    }

    /// SA never buffers — it decides immediately. Always returns `false`.
    fn should_commit_buffered(
        &mut self,
        _model: &Model<T>,
        _best_solution: SolutionView<'_, T>,
        _accepted_solution: SolutionView<'_, T>,
        _buffered_solution: Option<SolutionView<'_, T>>,
        _layout: &ScheduleGraph,
        _buffer_layout: &ScheduleGraph,
    ) -> bool {
        false
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

    /// The acceptance criterion.
    ///
    /// - Strict improvement ($f' < f$): always accept.
    /// - Frozen ($T \leq$ threshold): only accept improvements.
    /// - Otherwise (equal or worsening): accept with probability from
    ///   the `AcceptanceCriterion`.
    #[allow(clippy::too_many_arguments)]
    fn decide_fate(
        &mut self,
        _model: &Model<T>,
        _best_solution: SolutionView<'_, T>,
        accepted_solution: SolutionView<'_, T>,
        _buffered_solution: Option<SolutionView<'_, T>>,
        candidate_objective: T,
        _graph: &ScheduleGraph,
        _graph_diff: &ScheduleGraphDiff,
    ) -> AcceptanceOutcome {
        let current_objective = accepted_solution.objective_value();

        // ── Strict improvement: always accept ──
        if candidate_objective < current_objective {
            return AcceptanceOutcome::Accept;
        }

        // ── Non-improving move: check temperature ──
        let temp = self.cooling.temperature();
        if temp <= self.frozen_threshold {
            return AcceptanceOutcome::Reject;
        }

        // Convert to f64 for the probability calculation.
        let cand_f64 = match candidate_objective.to_f64() {
            Some(v) if v.is_finite() => v,
            _ => return AcceptanceOutcome::Reject,
        };
        let curr_f64 = match current_objective.to_f64() {
            Some(v) if v.is_finite() => v,
            _ => return AcceptanceOutcome::Reject,
        };

        let delta = cand_f64 - curr_f64; // positive since candidate is worse
        let acceptance_prob = self.criterion.acceptance_probability(delta, temp);

        if self.rng.random_bool(acceptance_prob.clamp(0.0, 1.0)) {
            AcceptanceOutcome::Accept
        } else {
            AcceptanceOutcome::Reject
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
        if self.cooling_trigger == CoolingTrigger::Acceptance {
            self.cooling.step();
        }
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
        // No-op. Cooling is driven by the configured CoolingTrigger.
    }

    fn on_new_best(
        &mut self,
        _model: &Model<T>,
        _new_best: SolutionView<T>,
        _graph: &ScheduleGraph,
        _graph_diff: &ScheduleGraphDiff,
    ) {
        // Standard SA does not react to new bests.
        // Adaptive variants could implement reheating here.
    }

    /// Advances the cooling schedule by one step when
    /// `CoolingTrigger::Iteration` is active.
    fn on_iteration(
        &mut self,
        _iteration: u64,
        _model: &Model<T>,
        _best_solution: SolutionView<'_, T>,
        _new_accepted_solution: SolutionView<'_, T>,
        _buffered_solution: Option<SolutionView<'_, T>>,
        _graph: &ScheduleGraph,
    ) {
        if self.cooling_trigger == CoolingTrigger::Iteration {
            self.cooling.step();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── MetropolisCriterion ──────────────────────────────────

    #[test]
    fn test_metropolis_zero_delta_returns_one() {
        let p = MetropolisCriterion.acceptance_probability(0.0, 100.0);
        assert!((p - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_metropolis_large_delta_returns_near_zero() {
        let p = MetropolisCriterion.acceptance_probability(1000.0, 1.0);
        assert!(p < 1e-100);
    }

    #[test]
    fn test_metropolis_high_temperature_returns_near_one() {
        let p = MetropolisCriterion.acceptance_probability(1.0, 1e12);
        assert!((p - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_metropolis_known_value() {
        // exp(-10 / 100) = exp(-0.1) ≈ 0.9048
        let p = MetropolisCriterion.acceptance_probability(10.0, 100.0);
        assert!((p - (-0.1_f64).exp()).abs() < 1e-12);
    }

    // ── GeometricCooling ─────────────────────────────────────

    #[test]
    fn test_geometric_initial_temperature() {
        let g = GeometricCooling::new(100.0, 0.95, 1.0);
        assert!((g.temperature() - 100.0).abs() < f64::EPSILON);
        assert_eq!(g.alpha(), 0.95);
        assert_eq!(g.initial_temperature(), 100.0);
        assert_eq!(g.min_temperature(), 1.0);
    }

    #[test]
    fn test_geometric_step_decays() {
        let mut g = GeometricCooling::new(100.0, 0.9, 0.0);
        g.step();
        assert!((g.temperature() - 90.0).abs() < 1e-10);
        g.step();
        assert!((g.temperature() - 81.0).abs() < 1e-10);
    }

    #[test]
    fn test_geometric_frozen_below_min() {
        let mut g = GeometricCooling::new(2.0, 0.5, 1.0);
        assert!(!g.is_frozen());
        g.step(); // 1.0
        assert!(g.is_frozen());
        g.step(); // 0.5
        assert!(g.is_frozen());
    }

    #[test]
    fn test_geometric_reset_restores_initial() {
        let mut g = GeometricCooling::new(100.0, 0.5, 0.0);
        g.step();
        g.step();
        g.reset();
        assert!((g.temperature() - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_geometric_reheat_capped_at_initial() {
        let mut g = GeometricCooling::new(100.0, 0.5, 0.0);
        g.step(); // 50
        g.reheat(3.0); // min(150, 100) = 100
        assert!((g.temperature() - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_geometric_reheat_within_range() {
        let mut g = GeometricCooling::new(100.0, 0.5, 0.0);
        g.step(); // 50
        g.step(); // 25
        g.reheat(2.0); // 50
        assert!((g.temperature() - 50.0).abs() < 1e-10);
    }

    #[test]
    #[should_panic(expected = "alpha must be in (0, 1)")]
    fn geometric_alpha_zero_panics() {
        GeometricCooling::new(100.0, 0.0, 0.0);
    }

    #[test]
    #[should_panic(expected = "alpha must be in (0, 1)")]
    fn geometric_alpha_one_panics() {
        GeometricCooling::new(100.0, 1.0, 0.0);
    }

    #[test]
    #[should_panic(expected = "initial temperature must be positive")]
    fn geometric_zero_initial_panics() {
        GeometricCooling::new(0.0, 0.9, 0.0);
    }

    // ── LinearCooling ────────────────────────────────────────

    #[test]
    fn test_linear_step_decrements() {
        let mut l = LinearCooling::new(100.0, 10.0, 0.0);
        l.step();
        assert!((l.temperature() - 90.0).abs() < f64::EPSILON);
        for _ in 0..9 {
            l.step();
        }
        assert!((l.temperature() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_linear_does_not_go_negative() {
        let mut l = LinearCooling::new(5.0, 10.0, 0.0);
        l.step();
        assert!((l.temperature() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_linear_frozen_at_min() {
        let mut l = LinearCooling::new(100.0, 50.0, 10.0);
        assert!(!l.is_frozen());
        l.step(); // 50
        assert!(!l.is_frozen());
        l.step(); // 0, which is <= 10
        assert!(l.is_frozen());
    }

    #[test]
    fn test_linear_from_budget() {
        // After 10 steps should reach 0.
        let mut l = LinearCooling::from_budget(100.0, 0.0, 10);
        for _ in 0..10 {
            l.step();
        }
        assert!(l.temperature().abs() < 1e-10);
    }

    #[test]
    fn test_linear_reset_restores_initial() {
        let mut l = LinearCooling::new(100.0, 25.0, 0.0);
        l.step();
        l.step();
        l.reset();
        assert!((l.temperature() - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_linear_reheat_capped_at_initial() {
        let mut l = LinearCooling::new(100.0, 80.0, 0.0);
        l.step(); // 20
        l.reheat(10.0); // min(200, 100) = 100
        assert!((l.temperature() - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    #[should_panic(expected = "initial must be positive")]
    fn linear_zero_initial_panics() {
        LinearCooling::new(0.0, 1.0, 0.0);
    }

    #[test]
    #[should_panic(expected = "decrement must be positive")]
    fn linear_zero_decrement_panics() {
        LinearCooling::new(100.0, 0.0, 0.0);
    }

    // ── LogarithmicCooling ───────────────────────────────────

    #[test]
    fn test_logarithmic_initial_temperature_is_c_over_one() {
        let l = LogarithmicCooling::new(100.0, 1.0, 0.0);
        // T(0) = 100 / ln(0 * 1 + e) = 100 / 1 = 100
        assert!((l.temperature() - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_logarithmic_decreases_monotonically() {
        let mut l = LogarithmicCooling::new(100.0, 1.0, 0.0);
        let mut prev = l.temperature();
        for _ in 0..20 {
            l.step();
            let t = l.temperature();
            assert!(t < prev, "temperature should decrease: {} < {}", t, prev);
            prev = t;
        }
    }

    #[test]
    fn test_logarithmic_reset_restores_initial() {
        let mut l = LogarithmicCooling::new(100.0, 1.0, 0.0);
        for _ in 0..10 {
            l.step();
        }
        l.reset();
        assert!((l.temperature() - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_logarithmic_frozen_when_below_min() {
        let mut l = LogarithmicCooling::new(10.0, 10.0, 5.0);
        // Eventually must freeze.
        for _ in 0..1000 {
            l.step();
            if l.is_frozen() {
                return;
            }
        }
        panic!("logarithmic cooling should eventually freeze");
    }

    // ── CoolingTrigger ───────────────────────────────────────

    #[test]
    fn test_cooling_trigger_display() {
        assert_eq!(format!("{}", CoolingTrigger::Iteration), "Iteration");
        assert_eq!(format!("{}", CoolingTrigger::Cycle), "Cycle");
        assert_eq!(format!("{}", CoolingTrigger::Acceptance), "Acceptance");
    }

    // ── SimulatedAnnealing builders ──────────────────────────

    #[test]
    fn test_sa_default_trigger_is_iteration() {
        let sa = SimulatedAnnealing::new(GeometricCooling::new(100.0, 0.9, 0.0), rand::rng());
        assert_eq!(sa.cooling_trigger(), CoolingTrigger::Iteration);
    }

    #[test]
    fn test_sa_with_cooling_trigger() {
        let sa = SimulatedAnnealing::new(GeometricCooling::new(100.0, 0.9, 0.0), rand::rng())
            .with_cooling_trigger(CoolingTrigger::Acceptance);
        assert_eq!(sa.cooling_trigger(), CoolingTrigger::Acceptance);
    }

    #[test]
    fn test_sa_default_frozen_threshold() {
        let sa = SimulatedAnnealing::new(GeometricCooling::new(100.0, 0.9, 0.0), rand::rng());
        assert!((sa.frozen_threshold() - 1e-12).abs() < f64::EPSILON);
    }

    #[test]
    fn test_sa_with_frozen_threshold() {
        let sa = SimulatedAnnealing::new(GeometricCooling::new(100.0, 0.9, 0.0), rand::rng())
            .with_frozen_threshold(0.5);
        assert!((sa.frozen_threshold() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    #[should_panic(expected = "reheat factor must be > 1.0")]
    fn sa_reheat_factor_must_exceed_one() {
        SimulatedAnnealing::new(GeometricCooling::new(100.0, 0.9, 0.0), rand::rng())
            .with_reheat(0.5);
    }

    #[test]
    #[should_panic(expected = "frozen threshold must be >= 0.0")]
    fn sa_negative_frozen_threshold_panics() {
        SimulatedAnnealing::new(GeometricCooling::new(100.0, 0.9, 0.0), rand::rng())
            .with_frozen_threshold(-1.0);
    }

    #[test]
    fn test_sa_cooling_accessor() {
        let sa = SimulatedAnnealing::new(GeometricCooling::new(42.0, 0.9, 0.0), rand::rng());
        assert!((sa.cooling().temperature() - 42.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_sa_cooling_mut_accessor() {
        let mut sa = SimulatedAnnealing::new(GeometricCooling::new(100.0, 0.5, 0.0), rand::rng());
        sa.cooling_mut().step();
        assert!((sa.cooling().temperature() - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_sa_with_custom_criterion() {
        #[derive(Debug)]
        struct AlwaysAccept;
        impl AcceptanceCriterion for AlwaysAccept {
            fn acceptance_probability(&self, _delta: f64, _temp: f64) -> f64 {
                1.0
            }
        }

        let sa = SimulatedAnnealing::with_criterion(
            GeometricCooling::new(100.0, 0.9, 0.0),
            rand::rng(),
            AlwaysAccept,
        );
        assert_eq!(sa.criterion().acceptance_probability(999.0, 0.001), 1.0);
    }

    // ── heuristic_geometric_params ───────────────────────────

    #[test]
    fn test_heuristic_params_produces_valid_schedule() {
        let g = SimulatedAnnealing::<rand::rngs::ThreadRng, GeometricCooling>::heuristic_geometric_params(
            1000.0,
            0.5,
            0.01,
            0.999,
        );
        assert!(g.temperature() > 0.0);
        assert_eq!(g.alpha(), 0.999);
        assert!(!g.is_frozen());
    }

    #[test]
    fn test_heuristic_params_initial_temp_scales_with_objective() {
        let g1 = SimulatedAnnealing::<rand::rngs::ThreadRng, GeometricCooling>::heuristic_geometric_params(
            100.0, 0.5, 0.01, 0.99,
        );
        let g2 = SimulatedAnnealing::<rand::rngs::ThreadRng, GeometricCooling>::heuristic_geometric_params(
            10000.0, 0.5, 0.01, 0.99,
        );
        assert!(g2.initial_temperature() > g1.initial_temperature());
    }

    // ── Debug formatting ─────────────────────────────────────

    #[test]
    fn test_sa_debug_does_not_panic() {
        let sa = SimulatedAnnealing::new(GeometricCooling::new(100.0, 0.9, 1.0), rand::rng());
        let s = format!("{:?}", sa);
        assert!(s.contains("SimulatedAnnealing"));
    }
}
