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

use talos_ls::meta::{
    metaheuristic::NeighborhoodExhaustionOutcome, sa::CoolingTrigger, tabu::SelectionStrategy,
};

/// Move selection strategy (used by Tabu Search and Greedy Descent).
///
/// * `BestImprovement` (0) — evaluate the full neighbourhood, commit the
///   best admissible move.
/// * `FirstImprovement` (1) — accept the first admissible move.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfiSelectionStrategy {
    BestImprovement = 0,
    FirstImprovement = 1,
}

impl FfiSelectionStrategy {
    pub fn into_inner(self) -> SelectionStrategy {
        match self {
            Self::BestImprovement => SelectionStrategy::BestImprovement,
            Self::FirstImprovement => SelectionStrategy::FirstImprovement,
        }
    }
}

/// Tie-breaking strategy for best-improvement mode.
///
/// * `KeepFirst` (0) — keep the earlier move (default).
/// * `KeepLast` (1) — always replace the buffer with the newer move.
/// * `Random` (2) — fair coin flip (requires a seeded RNG).
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfiTieBreaking {
    KeepFirst = 0,
    KeepLast = 1,
    Random = 2,
}

/// Behaviour when the neighbourhood is exhausted.
///
/// * `Restart` (0) — re-scan the neighbourhood (standard for Tabu Search).
/// * `Terminate` (1) — stop the search.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfiExhaustionOutcome {
    Restart = 0,
    Terminate = 1,
}

impl FfiExhaustionOutcome {
    pub fn into_inner(self) -> NeighborhoodExhaustionOutcome {
        match self {
            Self::Restart => NeighborhoodExhaustionOutcome::Restart,
            Self::Terminate => NeighborhoodExhaustionOutcome::Terminate,
        }
    }
}

/// SA cooling schedule type.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfiCoolingSchedule {
    /// Manual geometric: provide `initial_temperature`, `alpha`,
    /// `min_temperature`.
    Geometric = 0,
    /// Heuristic geometric: provide `heuristic_objective`,
    /// `heuristic_acceptance_prob`, `heuristic_sensitivity`,
    /// `heuristic_cooling_rate`.
    GeometricHeuristic = 1,
    /// Linear: provide `initial_temperature`, `linear_decrement`,
    /// `min_temperature`.
    Linear = 2,
    /// Linear from budget: provide `initial_temperature`, `min_temperature`,
    /// `linear_budget_iterations`.
    LinearBudget = 3,
    /// Logarithmic: provide `log_constant`, `log_k_scale`,
    /// `min_temperature`.
    Logarithmic = 4,
}

/// SA cooling trigger.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfiCoolingTrigger {
    /// Cool once per iteration.
    Iteration = 0,
    /// Cool once per cycle (neighbourhood exhaustion).
    Cycle = 1,
    /// Cool on move acceptance.
    Acceptance = 2,
}

impl FfiCoolingTrigger {
    pub fn into_inner(self) -> CoolingTrigger {
        match self {
            Self::Iteration => CoolingTrigger::Iteration,
            Self::Cycle => CoolingTrigger::Cycle,
            Self::Acceptance => CoolingTrigger::Acceptance,
        }
    }
}

/// GLS penalization trigger.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfiPenalizationTrigger {
    /// Penalize on neighbourhood exhaustion (classic GLS).
    OnExhaustion = 0,
    /// Penalize after `n` non-improving iterations.
    AfterNonImprovements = 1,
    /// Penalize every `n` accepted moves.
    AfterMoves = 2,
}

/// Which metaheuristic algorithm to use (for the unified entry point).
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfiMetaheuristic {
    TabuSearch = 0,
    SimulatedAnnealing = 1,
    GuidedLocalSearch = 2,
    GreedyDescent = 3,
}

// ----------------------------------------------------------------
// Per-algorithm configuration structs
// ----------------------------------------------------------------

/// Full configuration for Tabu Search.
///
/// ## Tenure
///
/// * Fixed tenure: set `tenure_min` to the desired value and
///   `tenure_max` to the same value (or 0).
/// * Random tenure: set `tenure_min` < `tenure_max`. Requires `seed`.
///
/// ## Selection
///
/// * `selection` — `BestImprovement` (0) or `FirstImprovement` (1).
///
/// ## Tie-breaking
///
/// * `tie_breaking` — `KeepFirst` (0), `KeepLast` (1), or `Random` (2).
///   Only relevant with `BestImprovement`. `Random` uses `seed`.
///
/// ## Exhaustion
///
/// * `exhaustion_outcome` — `Restart` (0) or `Terminate` (1).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiTabuSearchConfig {
    pub num_vessels: usize,
    pub num_berths: usize,
    /// Minimum tenure. For fixed tenure, set `tenure_max` equal to this.
    pub tenure_min: u64,
    /// Maximum tenure. If `> tenure_min`, random tenure in
    /// `[tenure_min, tenure_max]`.
    pub tenure_max: u64,
    pub selection: FfiSelectionStrategy,
    pub tie_breaking: FfiTieBreaking,
    pub exhaustion_outcome: FfiExhaustionOutcome,
    /// RNG seed (used for random tenure and random tie-breaking).
    pub seed: u64,
}

/// Full configuration for Simulated Annealing.
///
/// ## Cooling schedule
///
/// Select via `schedule`:
///
/// | `schedule`           | Required fields                                    |
/// |----------------------|----------------------------------------------------|
/// | `Geometric`          | `initial_temperature`, `alpha`, `min_temperature`  |
/// | `GeometricHeuristic` | `heuristic_*` fields                               |
/// | `Linear`             | `initial_temperature`, `linear_decrement`,         |
/// |                      | `min_temperature`                                  |
/// | `LinearBudget`       | `initial_temperature`, `min_temperature`,          |
/// |                      | `linear_budget_iterations`                         |
/// | `Logarithmic`        | `log_constant`, `log_k_scale`, `min_temperature`   |
///
/// ## Reheating
///
/// Set `reheat_factor > 1.0` to enable reheating on neighbourhood
/// exhaustion. `<= 1.0` disables it.
///
/// ## Frozen threshold
///
/// Set `frozen_threshold > 0.0` to override the default (`1e-12`).
/// Set to `0.0` to use the default.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiSimulatedAnnealingConfig {
    pub schedule: FfiCoolingSchedule,
    pub cooling_trigger: FfiCoolingTrigger,
    /// RNG seed.
    pub seed: u64,

    // ── Geometric / Linear / LinearBudget ────────────────────
    pub initial_temperature: f64,
    /// Geometric decay factor $\alpha \in (0, 1)$.
    pub alpha: f64,
    /// Linear decrement per step (for `Linear` schedule).
    pub linear_decrement: f64,
    /// Total iteration budget (for `LinearBudget` schedule).
    pub linear_budget_iterations: u64,
    /// Frozen threshold. `0.0` uses the default `1e-12`.
    pub min_temperature: f64,

    // ── Logarithmic ──────────────────────────────────────────
    /// Logarithmic numerator constant $C$.
    pub log_constant: f64,
    /// Logarithmic iteration scaling factor.
    pub log_k_scale: f64,

    // ── GeometricHeuristic ───────────────────────────────────
    pub heuristic_objective: f64,
    pub heuristic_acceptance_prob: f64,
    pub heuristic_sensitivity: f64,
    pub heuristic_cooling_rate: f64,

    // ── Reheating ────────────────────────────────────────────
    /// Reheat multiplier on neighbourhood exhaustion.
    /// `<= 1.0` disables reheating.
    pub reheat_factor: f64,

    /// Custom frozen threshold. `0.0` uses the built-in default (`1e-12`).
    pub frozen_threshold: f64,
}

/// Full configuration for Guided Local Search.
///
/// ## Lambda
///
/// * Fixed: set `reactive = 0` and `lambda` to the desired weight.
/// * Reactive: set `reactive != 0` and provide `lambda` (initial),
///   `growth_factor`, `decay_factor`, `min_lambda`, `max_lambda`.
///
/// ## Penalization trigger
///
/// * `OnExhaustion` (0) — classic GLS.
/// * `AfterNonImprovements` (1) — provide `trigger_n`.
/// * `AfterMoves` (2) — provide `trigger_n`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiGuidedLocalSearchConfig {
    pub num_vessels: usize,
    pub num_berths: usize,
    /// Penalty weight $\lambda$.
    pub lambda: f64,
    /// Non-zero to use reactive lambda.
    pub reactive: i32,
    pub penalization_trigger: FfiPenalizationTrigger,
    /// Trigger threshold (for `AfterNonImprovements` / `AfterMoves`).
    pub trigger_n: u64,
    /// Reactive: growth factor (must be `> 1.0`).
    pub growth_factor: f64,
    /// Reactive: decay factor (must be in `(0.0, 1.0)`).
    pub decay_factor: f64,
    /// Reactive: minimum lambda clamp.
    pub min_lambda: f64,
    /// Reactive: maximum lambda clamp.
    pub max_lambda: f64,
}

/// Full configuration for Greedy Descent.
///
/// ## Selection
///
/// * `FirstImprovement` (1) — accept the first improving move (default).
/// * `BestImprovement` (0) — steepest descent.
///
/// ## Tie-breaking
///
/// * `KeepFirst` (0), `KeepLast` (1), `Random` (2).
///   Only relevant with `BestImprovement`. `Random` uses `seed`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiGreedyDescentConfig {
    pub selection: FfiSelectionStrategy,
    pub tie_breaking: FfiTieBreaking,
    /// RNG seed (used for random tie-breaking).
    pub seed: u64,
}

// ----------------------------------------------------------------
// Operator configuration
// ----------------------------------------------------------------

/// Compound operator selection strategy.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfiCompoundStrategy {
    RoundRobin = 0,
    Random = 1,
    MultiArmedBandit = 2,
}

/// Flat configuration for the compound operator, passed by value.
///
/// Set the `use_*` fields to non-zero to enable the corresponding
/// operator. At least one operator must be enabled.
///
/// * `seed` — used by the `Random` compound strategy.
/// * `bandit_memory_coeff` / `bandit_exploration_coeff` — used only
///   with `MultiArmedBandit`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiOperatorConfig {
    pub strategy: FfiCompoundStrategy,
    pub use_intra_berth_swap: i32,
    pub use_inter_berth_swap: i32,
    pub use_intra_berth_shift: i32,
    pub use_inter_berth_shift: i32,
    pub seed: u64,
    pub bandit_memory_coeff: f64,
    pub bandit_exploration_coeff: f64,
}

// ----------------------------------------------------------------
// Monitor configuration
// ----------------------------------------------------------------

/// Flat configuration for termination conditions, passed by value.
///
/// Set a limit field to a positive value to enable that condition.
/// A value of `0` disables it. At least one condition should be
/// enabled, otherwise the search will never terminate.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiMonitorConfig {
    /// Maximum wall-clock time in milliseconds.
    pub time_limit_millis: u64,
    /// Maximum number of iterations.
    pub iteration_limit: u64,
    /// Maximum number of accepted solutions.
    pub solution_limit: u64,
    /// Maximum number of cycles.
    pub cycle_limit: u64,
    /// Stop after this many iterations without improvement.
    pub no_improvement_iterations: u64,
    /// Stop after this many cycles without improvement.
    pub no_improvement_cycles: u64,
    /// Stop after this many milliseconds without improvement.
    pub no_improvement_time_millis: u64,
}
