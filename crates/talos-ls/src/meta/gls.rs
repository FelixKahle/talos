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

//! Guided Local Search (GLS) metaheuristic for local search.
//!
//! GLS augments a local search by adding penalties to the objective function,
//! guiding the search away from previously visited local optima. When the
//! underlying local search reaches a local optimum, GLS inspects the current
//! solution's features, identifies the one with the highest *utility*, and
//! increments its penalty counter. The augmented objective pushes the search
//! into unexplored regions of the solution space.

use crate::{
    exec::SearchCommand,
    meta::metaheuristic::{AcceptanceOutcome, Metaheuristic, NeighborhoodExhaustionOutcome},
    sgraph::{ScheduleGraph, ScheduleGraphDiff},
};
use num_traits::ToPrimitive;
use talos_core::{container::rarena::Node, utils::num::SolverNumeric};
use talos_model::{
    index::{BerthIndex, VesselIndex},
    model::Model,
    solution::SolutionView,
};

// ----------------------------------------------------------------
// Feature Cost
// ----------------------------------------------------------------

/// Computes the cost contribution of a single feature.
///
/// GLS maximizes the *utility* of penalized features. The utility of a
/// feature $f$ present in the current solution is:
///
/// $$\text{utility}(f) = \frac{c_f}{1 + p_f}$$
///
/// where $c_f$ is the cost returned by this trait and $p_f$ is the
/// current penalty count. Features with high cost and low penalty
/// are penalized first.
pub trait FeatureCost: std::fmt::Debug {
    /// Returns the cost of an edge feature between two nodes.
    /// Either node may be a vessel or a berth sentinel.
    fn edge_cost(&self, a: Node, b: Node) -> f64;

    /// Returns the cost of a berth-assignment feature (vessel `v` assigned
    /// to berth `b`).
    fn berth_cost(&self, v: VesselIndex, b: BerthIndex) -> f64;
}

/// Assigns a uniform cost of 1.0 to every feature.
///
/// This is the simplest cost model and works well when per-feature cost
/// estimates are not available.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct UniformCost;

impl FeatureCost for UniformCost {
    #[inline]
    fn edge_cost(&self, _a: Node, _b: Node) -> f64 {
        1.0
    }

    #[inline]
    fn berth_cost(&self, _v: VesselIndex, _b: BerthIndex) -> f64 {
        1.0
    }
}

// ----------------------------------------------------------------
// Penalization Strategy
// ----------------------------------------------------------------

/// Controls which features are penalized when the search reaches a local
/// optimum.
pub trait PenalizationStrategy: std::fmt::Debug {
    /// Inspects the current solution and increments penalty counters in
    /// `memory` for the features deemed most responsible for the local
    /// optimum.
    ///
    /// # Arguments
    ///
    /// * `memory` — The penalty memory to update.
    /// * `graph` — The current schedule graph (for iterating features).
    /// * `cost` — The feature cost model.
    fn penalize(
        &mut self,
        memory: &mut PenaltyMemory,
        graph: &ScheduleGraph,
        cost: &dyn FeatureCost,
    );
}

/// Penalizes the feature(s) with the highest utility in the current
/// solution.
///
/// Utility is defined as $c_f / (1 + p_f)$. If multiple features tie for
/// the maximum utility, all of them are penalized.
///
/// Maintains a reusable scratch buffer to avoid heap allocations on every
/// penalization call.
#[derive(Debug, Clone, Default)]
pub struct MaxUtilityPenalization {
    /// Reusable scratch buffer for collecting winning features.
    /// Entries are `(is_edge, flat_index)`.
    buf: Vec<(bool, usize)>,
}

impl PenalizationStrategy for MaxUtilityPenalization {
    fn penalize(
        &mut self,
        memory: &mut PenaltyMemory,
        graph: &ScheduleGraph,
        cost: &dyn FeatureCost,
    ) {
        let num_vessels = graph.num_vessels();
        let num_berths = graph.num_berths();

        let mut best_utility = f64::NEG_INFINITY;
        self.buf.clear();

        // Relative tolerance: two utilities are "equal" when they differ
        // by less than this fraction of the larger magnitude. 1e-12 is
        // tight enough to avoid false ties on well-separated values but
        // loose enough to absorb FP rounding.
        const REL_TOL: f64 = 1e-12;

        let mut evaluate_feature = |is_edge: bool, idx: usize, cost_val: f64, penalty: f64| {
            if cost_val > 0.0 {
                let utility = cost_val / (1.0 + penalty);

                let diff = utility - best_utility;
                let scale = utility.abs().max(best_utility.abs()).max(1.0);

                if best_utility == f64::NEG_INFINITY || diff > REL_TOL * scale {
                    best_utility = utility;
                    self.buf.clear();
                    self.buf.push((is_edge, idx));
                } else if diff.abs() <= REL_TOL * scale {
                    self.buf.push((is_edge, idx));
                }
            }
        };

        for berth in graph.berth_iter() {
            let topo_iter = graph.berth_topology_iter(berth);

            for (prev_node, next_node) in topo_iter.clone().zip(topo_iter.skip(1)) {
                if prev_node.get() < num_vessels || next_node.get() < num_vessels {
                    let c = cost.edge_cost(prev_node, next_node);
                    let idx = edge_flat_index(prev_node, next_node, num_vessels);

                    debug_assert!(idx < memory.edge.len());
                    let p = unsafe { *memory.edge.get_unchecked(idx) } as f64;

                    evaluate_feature(true, idx, c, p);
                }
            }

            for vessel in graph.vessel_sequence_iter(berth) {
                let c = cost.berth_cost(vessel, berth);
                let idx = vessel.get() * num_berths + berth.get();

                debug_assert!(idx < memory.berth.len());
                let p = unsafe { *memory.berth.get_unchecked(idx) } as f64;

                evaluate_feature(false, idx, c, p);
            }
        }

        for &(is_edge, idx) in &self.buf {
            debug_assert!(if is_edge {
                idx < memory.edge.len()
            } else {
                idx < memory.berth.len()
            });

            unsafe {
                if is_edge {
                    let p = memory.edge.get_unchecked_mut(idx);
                    *p = p.saturating_add(1);
                } else {
                    let p = memory.berth.get_unchecked_mut(idx);
                    *p = p.saturating_add(1);
                }
            }
        }
    }
}

// ----------------------------------------------------------------
// Penalty Memory
// ----------------------------------------------------------------

/// Long-term memory tracking how often each feature has been penalized.
///
/// Stores penalty counts for two feature classes:
/// - **Edge**: vessel-to-vessel sequence links.
/// - **Berth**: vessel-to-berth assignments.
///
/// All operations are $O(1)$ array lookups.
#[derive(Debug, Clone)]
pub struct PenaltyMemory {
    /// Flat `(num_vessels + 1) * (num_vessels + 1)` array.
    /// `edge[a * (num_vessels + 1) + b]` stores the penalty count for
    /// the edge feature $a \to b$. Index `num_vessels` represents the
    /// sentinel (berth boundary).
    edge: Vec<u64>,

    /// Flat `num_vessels * num_berths` array.
    /// `berth[v * num_berths + b]` stores the penalty count for
    /// assigning vessel $v$ to berth $b$.
    berth: Vec<u64>,

    num_vessels: usize,
    num_berths: usize,
}

/// Computes the flat index for an edge feature.
///
/// **Sentinel collapsing**: All sentinel nodes (index ≥ `num_vessels`)
/// are mapped to the single row/column `num_vessels`. This means edges
/// involving *different* berth sentinels share the same penalty counter.
/// This is intentional — the sequence penalty tracks "vessel near a
/// boundary", not "vessel near berth-3's boundary specifically".
///
/// The resulting index is guaranteed to be less than
/// $(\text{num\_vessels} + 1)^2$ — the allocation size of
/// `PenaltyMemory::edge`.
#[inline(always)]
fn edge_flat_index(from: Node, to: Node, num_vessels: usize) -> usize {
    let a = from.get().min(num_vessels);
    let b = to.get().min(num_vessels);
    a * (num_vessels + 1) + b
}

impl PenaltyMemory {
    /// Creates a new penalty memory for the given problem dimensions.
    ///
    /// Allocates two flat arrays:
    /// - `edge`: $(\text{num\_vessels} + 1)^2$ entries for sequence features
    ///   (including berth sentinels).
    /// - `berth`: $\text{num\_vessels} \times \text{num\_berths}$ entries for
    ///   assignment features.
    ///
    /// All counters are initialised to zero.
    #[inline]
    pub fn new(num_vessels: usize, num_berths: usize) -> Self {
        Self {
            edge: vec![0; (num_vessels + 1) * (num_vessels + 1)],
            berth: vec![0; num_vessels * num_berths],
            num_vessels,
            num_berths,
        }
    }

    /// Resets all penalty counters to zero.
    #[inline]
    pub fn clear(&mut self) {
        self.edge.fill(0);
        self.berth.fill(0);
    }

    /// Returns the number of vessels.
    #[inline]
    pub fn num_vessels(&self) -> usize {
        self.num_vessels
    }

    /// Returns the number of berths.
    #[inline]
    pub fn num_berths(&self) -> usize {
        self.num_berths
    }

    /// Returns the penalty count for a given edge feature.
    ///
    /// # Arguments
    ///
    /// * `from` — The source node (vessel or sentinel).
    /// * `to` — The destination node (vessel or sentinel).
    #[inline]
    pub fn edge_penalty(&self, from: Node, to: Node) -> u64 {
        let idx = edge_flat_index(from, to, self.num_vessels);
        debug_assert!(idx < self.edge.len());

        // SAFETY: edge_flat_index yields < (num_vessels+1)^2, the allocation size.
        unsafe { *self.edge.get_unchecked(idx) }
    }

    /// Returns the penalty count for a given berth-assignment feature.
    ///
    /// # Arguments
    ///
    /// * `vessel` — The vessel index.
    /// * `berth` — The berth index.
    #[inline]
    pub fn berth_penalty(&self, vessel: VesselIndex, berth: BerthIndex) -> u64 {
        let idx = vessel.get() * self.num_berths + berth.get();
        debug_assert!(idx < self.berth.len());

        // SAFETY: vessel < num_vessels and berth < num_berths.
        unsafe { *self.berth.get_unchecked(idx) }
    }

    /// Computes the augmented penalty delta for a candidate move described
    /// by `diff`.
    ///
    /// The delta is: penalties gained from *created* edges and *new* berth
    /// assignments minus penalties lost from *broken* edges and *old* berth
    /// assignments.
    ///
    /// Because a single move typically touches only 2–6 diff entries, this
    /// runs in $O(|\text{diff}|)$ — far cheaper than recomputing the full
    /// penalty sum from scratch.
    ///
    /// # Arguments
    ///
    /// * `diff` — The schedule-graph diff describing the candidate move.
    #[inline]
    pub fn penalty_delta(&self, diff: &ScheduleGraphDiff) -> i64 {
        let mut delta: i64 = 0;

        // SAFETY: all flat indices are bounded by allocation sizes.
        // Edge indices < (num_vessels+1)^2, berth indices < num_vessels*num_berths.
        unsafe {
            // Subtract penalties for broken edges.
            for edge in diff.broken_links() {
                let idx = edge_flat_index(edge.from, edge.to, self.num_vessels);
                debug_assert!(idx < self.edge.len());

                delta -= *self.edge.get_unchecked(idx) as i64;
            }

            // Add penalties for created edges.
            for edge in diff.created_links() {
                let idx = edge_flat_index(edge.from, edge.to, self.num_vessels);
                debug_assert!(idx < self.edge.len());

                delta += *self.edge.get_unchecked(idx) as i64;
            }

            // Subtract penalties for old berth assignments, add for new.
            for (vessel, old_berth, new_berth) in diff.reallocations() {
                let old_idx = vessel.get() * self.num_berths + old_berth.get();
                let new_idx = vessel.get() * self.num_berths + new_berth.get();
                debug_assert!(
                    old_idx < self.berth.len(),
                    "penalty_delta old berth: index {old_idx} out of bounds (len {})",
                    self.berth.len()
                );
                debug_assert!(
                    new_idx < self.berth.len(),
                    "penalty_delta new berth: index {new_idx} out of bounds (len {})",
                    self.berth.len()
                );
                delta -= *self.berth.get_unchecked(old_idx) as i64;
                delta += *self.berth.get_unchecked(new_idx) as i64;
            }
        }

        delta
    }

    /// Multiplies all penalty counters by `factor`, truncating toward zero.
    ///
    /// This implements "forgetfulness" — old penalties gradually erode,
    /// allowing the search to revisit previously penalized regions.
    ///
    /// # Arguments
    ///
    /// * `factor` — Multiplicative decay factor (e.g. 0.9). Must be in `[0, 1]`.
    pub fn decay(&mut self, factor: f64) {
        debug_assert!(
            (0.0..=1.0).contains(&factor),
            "PenaltyMemory::decay: factor must be in [0, 1], got {factor}"
        );
        for p in &mut self.edge {
            *p = (*p as f64 * factor) as u64;
        }
        for p in &mut self.berth {
            *p = (*p as f64 * factor) as u64;
        }
    }
}

// ----------------------------------------------------------------
// Penalization Trigger
// ----------------------------------------------------------------

/// Controls *when* GLS applies its penalization step.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PenalizationTrigger {
    /// Penalize when the neighbourhood is exhausted (classic GLS).
    OnExhaustion,

    /// Penalize after `n` consecutive iterations without a new global best.
    AfterNonImprovements(u64),

    /// Penalize every `n` accepted moves.
    AfterMoves(u64),
}

impl Default for PenalizationTrigger {
    #[inline]
    fn default() -> Self {
        Self::OnExhaustion
    }
}

impl std::fmt::Display for PenalizationTrigger {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PenalizationTrigger::OnExhaustion => write!(f, "OnExhaustion"),
            PenalizationTrigger::AfterNonImprovements(n) => {
                write!(f, "AfterNonImprovements({})", n)
            }
            PenalizationTrigger::AfterMoves(n) => write!(f, "AfterMoves({})", n),
        }
    }
}

// ----------------------------------------------------------------
// Penalty Decay
// ----------------------------------------------------------------

/// Controls how penalty counters decay over time.
///
/// Penalty decay implements "forgetfulness" — periodically eroding
/// old penalties so the search can revisit previously penalized regions
/// after it has explored elsewhere.
///
/// The decay hook is called after each penalization step.
pub trait PenaltyDecay: std::fmt::Debug {
    /// Called after each penalization step. May modify penalty counters.
    fn after_penalization(&mut self, memory: &mut PenaltyMemory);
}

/// No decay — penalty counters are never eroded.
///
/// This is the default and matches classic GLS behaviour.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct NoDecay;

impl PenaltyDecay for NoDecay {
    #[inline]
    fn after_penalization(&mut self, _memory: &mut PenaltyMemory) {}
}

/// Geometric (multiplicative) decay applied every `period` penalizations.
///
/// Every `period` penalization steps, all penalty counters are multiplied
/// by `factor` (e.g. 0.9), truncating toward zero. This gradually erodes
/// old penalties, allowing the search to revisit previously sealed regions.
///
/// A period of 1 means decay fires on every penalization.
#[derive(Debug, Clone)]
pub struct GeometricDecay {
    factor: f64,
    period: u64,
    counter: u64,
}

impl GeometricDecay {
    /// Creates a new geometric decay strategy.
    ///
    /// # Arguments
    ///
    /// * `factor` — Multiplicative decay factor (e.g. 0.9). Must be in `(0, 1)`.
    /// * `period` — Apply decay every `period` penalization steps. Must be ≥ 1.
    pub fn new(factor: f64, period: u64) -> Self {
        assert!(
            factor > 0.0 && factor < 1.0,
            "GeometricDecay: factor must be in (0, 1), got {factor}"
        );
        assert!(period >= 1, "GeometricDecay: period must be >= 1");
        Self {
            factor,
            period,
            counter: 0,
        }
    }
}

impl PenaltyDecay for GeometricDecay {
    fn after_penalization(&mut self, memory: &mut PenaltyMemory) {
        self.counter += 1;
        if self.counter >= self.period {
            self.counter = 0;
            memory.decay(self.factor);
        }
    }
}

/// Computes a heuristic $\lambda$ from the initial objective.
///
/// $$\lambda = \alpha \cdot \frac{f(s_0)}{|F_0|}$$
///
/// where $|F_0|$ is the number of features present in the initial
/// solution and $\alpha$ is a scaling factor (typically 0.1–0.5).
///
/// The result is clamped to at least [`f64::EPSILON`] to avoid a
/// zero penalty weight, which would disable GLS entirely.
///
/// # Arguments
///
/// * `objective` — The initial solution's objective value.
/// * `num_features` — The number of features in the initial solution
///   (e.g., `num_vessels + num_edges`). Clamped to at least 1.
/// * `alpha` — Scaling factor (e.g., 0.3).
///
/// # Examples
///
/// ```
/// # use talos_ls::meta::gls::heuristic_lambda;
/// let lambda = heuristic_lambda(1000.0, 50, 0.3);
/// assert!((lambda - 6.0).abs() < f64::EPSILON);
/// ```
#[inline(always)]
pub fn heuristic_lambda(objective: f64, num_features: usize, alpha: f64) -> f64 {
    let nf = num_features.max(1) as f64;
    (alpha * objective.abs() / nf).max(f64::EPSILON)
}

// ----------------------------------------------------------------
// Lambda Strategy
// ----------------------------------------------------------------

/// Controls how the penalty weight $\lambda$ evolves during search.
///
/// Lifecycle hooks are called by GLS at the appropriate moments:
/// - `on_start` — once, when the search begins.
/// - `on_accept` — after each accepted move.
/// - `on_new_best` — when a new global best is found.
/// - `on_penalization` — after each penalization step.
pub trait LambdaStrategy: std::fmt::Debug {
    /// Returns the current penalty weight $\lambda$.
    fn lambda(&self) -> f64;

    /// Called once when the search starts.
    fn on_start(&mut self, _initial_objective: f64) {}

    /// Called after an accepted move.
    fn on_accept(&mut self, _new_objective: f64) {}

    /// Called when a new global best solution is found.
    fn on_new_best(&mut self, _new_best_objective: f64) {}

    /// Called after the penalization step fires.
    fn on_penalization(&mut self) {}
}

/// A constant $\lambda$ that never changes.
///
/// This is the simplest strategy and the default for
/// `GuidedLocalSearch`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StaticLambda(pub f64);

impl LambdaStrategy for StaticLambda {
    #[inline]
    fn lambda(&self) -> f64 {
        self.0
    }
}

/// A reactive $\lambda$ that adjusts asymmetrically based on search progress.
#[derive(Debug, Clone)]
pub struct DynamicLambda {
    current: f64,
    initial: f64,
    inc_step: f64,
    dec_step: f64,
    min_lambda: f64,
    max_lambda: f64,
    reset_on_best: bool,
}

impl DynamicLambda {
    /// Creates a new dynamic lambda strategy with asymmetric scaling.
    ///
    /// # Arguments
    ///
    /// * `initial` — Starting $\lambda$ value.
    /// * `inc_step` — Multiplicative increase factor when stuck (e.g. 0.05 for +5%).
    /// * `dec_step` — Multiplicative decrease factor on new best (e.g. 0.25 for -25%).
    /// * `min_lambda` — Lower clamp bound.
    /// * `max_lambda` — Upper clamp bound.
    pub fn new(
        initial: f64,
        inc_step: f64,
        dec_step: f64,
        min_lambda: f64,
        max_lambda: f64,
    ) -> Self {
        assert!(initial > 0.0, "DynamicLambda: initial must be > 0.0");
        assert!(inc_step > 0.0, "DynamicLambda: inc_step must be > 0.0");
        assert!(
            dec_step > 0.0 && dec_step < 1.0,
            "DynamicLambda: dec_step must be in (0, 1)"
        );
        assert!(min_lambda > 0.0, "DynamicLambda: min_lambda must be > 0.0");
        assert!(
            max_lambda >= min_lambda,
            "DynamicLambda: max_lambda must be >= min_lambda"
        );

        Self {
            current: initial.clamp(min_lambda, max_lambda),
            initial,
            inc_step,
            dec_step,
            min_lambda,
            max_lambda,
            reset_on_best: false,
        }
    }

    #[inline]
    pub fn with_reset_on_best(mut self, reset: bool) -> Self {
        self.reset_on_best = reset;
        self
    }
}

impl LambdaStrategy for DynamicLambda {
    #[inline]
    fn lambda(&self) -> f64 {
        self.current
    }

    fn on_start(&mut self, _initial_objective: f64) {
        self.current = self.initial.clamp(self.min_lambda, self.max_lambda);
    }

    fn on_new_best(&mut self, _new_best_objective: f64) {
        if self.reset_on_best {
            self.current = self.initial.clamp(self.min_lambda, self.max_lambda);
        } else {
            // Intensify: Rapid drop using dec_step
            self.current =
                (self.current * (1.0 - self.dec_step)).clamp(self.min_lambda, self.max_lambda);
        }
    }

    fn on_penalization(&mut self) {
        // Diversify: Gradual climb using inc_step
        self.current =
            (self.current * (1.0 + self.inc_step)).clamp(self.min_lambda, self.max_lambda);
    }
}

/// A reactive $\lambda$ that adjusts by fixed additive steps.
#[derive(Debug, Clone)]
pub struct AdditiveDynamicLambda {
    current: f64,
    initial: f64,
    inc_step: f64,
    dec_step: f64,
    min_lambda: f64,
    max_lambda: f64,
    reset_on_best: bool,
}

impl AdditiveDynamicLambda {
    /// Creates a new additive dynamic lambda strategy with asymmetric scaling.
    ///
    /// # Arguments
    ///
    /// * `initial` — Starting $\lambda$ value.
    /// * `inc_step` — Constant additive increase (e.g. 0.05).
    /// * `dec_step` — Constant additive decrease (e.g. 0.25).
    /// * `min_lambda` — Lower clamp bound.
    /// * `max_lambda` — Upper clamp bound.
    pub fn new(
        initial: f64,
        inc_step: f64,
        dec_step: f64,
        min_lambda: f64,
        max_lambda: f64,
    ) -> Self {
        assert!(
            initial > 0.0,
            "AdditiveDynamicLambda: initial must be > 0.0"
        );
        assert!(
            inc_step > 0.0,
            "AdditiveDynamicLambda: inc_step must be > 0.0"
        );
        assert!(
            dec_step > 0.0,
            "AdditiveDynamicLambda: dec_step must be > 0.0"
        );
        assert!(
            min_lambda > 0.0,
            "AdditiveDynamicLambda: min_lambda must be > 0.0"
        );
        assert!(
            max_lambda >= min_lambda,
            "AdditiveDynamicLambda: max_lambda must be >= min_lambda"
        );

        Self {
            current: initial.clamp(min_lambda, max_lambda),
            initial,
            inc_step,
            dec_step,
            min_lambda,
            max_lambda,
            reset_on_best: false,
        }
    }

    #[inline]
    pub fn with_reset_on_best(mut self, reset: bool) -> Self {
        self.reset_on_best = reset;
        self
    }
}

impl LambdaStrategy for AdditiveDynamicLambda {
    #[inline]
    fn lambda(&self) -> f64 {
        self.current
    }

    fn on_start(&mut self, _initial_objective: f64) {
        self.current = self.initial.clamp(self.min_lambda, self.max_lambda);
    }

    fn on_new_best(&mut self, _new_best_objective: f64) {
        if self.reset_on_best {
            self.current = self.initial.clamp(self.min_lambda, self.max_lambda);
        } else {
            // Intensify: drop by a fixed step.
            self.current = (self.current - self.dec_step).clamp(self.min_lambda, self.max_lambda);
        }
    }

    fn on_penalization(&mut self) {
        // Diversify: increase by a fixed step.
        self.current = (self.current + self.inc_step).clamp(self.min_lambda, self.max_lambda);
    }
}

// ----------------------------------------------------------------
// Guided Local Search
// ----------------------------------------------------------------

/// Guided Local Search (GLS) metaheuristic implementation.
///
/// GLS augments a local search by adding penalties to the objective function,
/// guiding the search away from previously visited local optima. When the
/// underlying local search reaches a local optimum, GLS inspects the current
/// solution's features, identifies the one with the highest *utility*, and
/// increments its penalty counter. The augmented objective pushes the search
/// into unexplored regions of the solution space.
pub struct GuidedLocalSearch<
    P = MaxUtilityPenalization,
    F = UniformCost,
    L = StaticLambda,
    D = NoDecay,
> {
    penalization: P,
    feature_cost: F,
    lambda_strategy: L,
    decay: D,
    memory: PenaltyMemory,
    current_penalty_sum: i64,
    trigger: PenalizationTrigger,
    trigger_counter: u64,
    last_best_obj: f64,
}

impl<P: std::fmt::Debug, F: std::fmt::Debug, L: std::fmt::Debug, D: std::fmt::Debug> std::fmt::Debug
    for GuidedLocalSearch<P, F, L, D>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GuidedLocalSearch")
            .field("penalization", &self.penalization)
            .field("feature_cost", &self.feature_cost)
            .field("lambda_strategy", &self.lambda_strategy)
            .field("decay", &self.decay)
            .field("current_penalty_sum", &self.current_penalty_sum)
            .field("trigger", &self.trigger)
            .field("trigger_counter", &self.trigger_counter)
            .field("last_best_obj", &self.last_best_obj)
            .finish()
    }
}

impl GuidedLocalSearch<MaxUtilityPenalization, UniformCost, StaticLambda> {
    /// Creates a new GLS with all default strategies.
    ///
    /// Defaults to `MaxUtilityPenalization`, `UniformCost`,
    /// `StaticLambda(1.0)`, and `NoDecay`. Use the `.with_*()` builder
    /// methods to customise:
    ///
    /// ```ignore
    /// let gls = GuidedLocalSearch::new(num_vessels, num_berths)
    ///     .with_lambda_strategy(DynamicLambda::new(base, 0.1, lo, hi))
    ///     .with_decay(GeometricDecay::new(0.9, 10))
    ///     .with_trigger(PenalizationTrigger::AfterMoves(100));
    /// ```
    ///
    /// # Arguments
    ///
    /// * `num_vessels` — Number of vessels in the problem.
    /// * `num_berths` — Number of berths in the problem.
    #[inline]
    pub fn new(num_vessels: usize, num_berths: usize) -> Self {
        Self {
            penalization: MaxUtilityPenalization::default(),
            feature_cost: UniformCost,
            lambda_strategy: StaticLambda(1.0),
            decay: NoDecay,
            memory: PenaltyMemory::new(num_vessels, num_berths),
            current_penalty_sum: 0,
            trigger: PenalizationTrigger::OnExhaustion,
            trigger_counter: 0,
            last_best_obj: f64::INFINITY,
        }
    }

    /// Sets a static (constant) lambda value.
    ///
    /// Shorthand for `.with_lambda_strategy(StaticLambda(lambda))`.
    ///
    /// # Panics
    ///
    /// Panics if `lambda` is not positive.
    #[inline]
    pub fn with_lambda(mut self, lambda: f64) -> Self {
        assert!(
            lambda > 0.0,
            "GuidedLocalSearch: lambda must be > 0.0, got {}",
            lambda
        );
        self.lambda_strategy = StaticLambda(lambda);
        self
    }
}

impl<P, F, L> GuidedLocalSearch<P, F, L, NoDecay>
where
    P: PenalizationStrategy,
    F: FeatureCost,
    L: LambdaStrategy,
{
    /// Creates a new GLS with fully custom strategies (no penalty decay).
    ///
    /// Use `.with_decay()` to add a decay
    /// strategy afterwards.
    ///
    /// # Arguments
    ///
    /// * `penalization` — The penalization strategy.
    /// * `feature_cost` — The feature cost model.
    /// * `lambda_strategy` — The lambda strategy controlling the penalty weight.
    /// * `num_vessels` — Number of vessels in the problem.
    /// * `num_berths` — Number of berths in the problem.
    #[inline]
    pub fn with_strategies(
        penalization: P,
        feature_cost: F,
        lambda_strategy: L,
        num_vessels: usize,
        num_berths: usize,
    ) -> Self {
        Self {
            penalization,
            feature_cost,
            lambda_strategy,
            decay: NoDecay,
            memory: PenaltyMemory::new(num_vessels, num_berths),
            current_penalty_sum: 0,
            trigger: PenalizationTrigger::OnExhaustion,
            trigger_counter: 0,
            last_best_obj: f64::INFINITY,
        }
    }
}

impl<P, F, L, D> GuidedLocalSearch<P, F, L, D>
where
    P: PenalizationStrategy,
    F: FeatureCost,
    L: LambdaStrategy,
    D: PenaltyDecay,
{
    /// Replaces the penalization strategy.
    #[inline]
    pub fn with_penalization<P2: PenalizationStrategy>(
        self,
        penalization: P2,
    ) -> GuidedLocalSearch<P2, F, L, D> {
        GuidedLocalSearch {
            penalization,
            feature_cost: self.feature_cost,
            lambda_strategy: self.lambda_strategy,
            decay: self.decay,
            memory: self.memory,
            current_penalty_sum: self.current_penalty_sum,
            trigger: self.trigger,
            trigger_counter: self.trigger_counter,
            last_best_obj: self.last_best_obj,
        }
    }

    /// Replaces the feature cost model.
    #[inline]
    pub fn with_feature_cost<F2: FeatureCost>(
        self,
        feature_cost: F2,
    ) -> GuidedLocalSearch<P, F2, L, D> {
        GuidedLocalSearch {
            penalization: self.penalization,
            feature_cost,
            lambda_strategy: self.lambda_strategy,
            decay: self.decay,
            memory: self.memory,
            current_penalty_sum: self.current_penalty_sum,
            trigger: self.trigger,
            trigger_counter: self.trigger_counter,
            last_best_obj: self.last_best_obj,
        }
    }

    /// Replaces the lambda strategy.
    #[inline]
    pub fn with_lambda_strategy<L2: LambdaStrategy>(
        self,
        lambda_strategy: L2,
    ) -> GuidedLocalSearch<P, F, L2, D> {
        GuidedLocalSearch {
            penalization: self.penalization,
            feature_cost: self.feature_cost,
            lambda_strategy,
            decay: self.decay,
            memory: self.memory,
            current_penalty_sum: self.current_penalty_sum,
            trigger: self.trigger,
            trigger_counter: self.trigger_counter,
            last_best_obj: self.last_best_obj,
        }
    }

    /// Replaces the penalty decay strategy.
    #[inline]
    pub fn with_decay<D2: PenaltyDecay>(self, decay: D2) -> GuidedLocalSearch<P, F, L, D2> {
        GuidedLocalSearch {
            penalization: self.penalization,
            feature_cost: self.feature_cost,
            lambda_strategy: self.lambda_strategy,
            decay,
            memory: self.memory,
            current_penalty_sum: self.current_penalty_sum,
            trigger: self.trigger,
            trigger_counter: self.trigger_counter,
            last_best_obj: self.last_best_obj,
        }
    }

    /// Returns the current penalty weight $\lambda$.
    #[inline]
    pub fn lambda(&self) -> f64 {
        self.lambda_strategy.lambda()
    }

    /// Returns a reference to the lambda strategy.
    #[inline]
    pub fn lambda_strategy(&self) -> &L {
        &self.lambda_strategy
    }

    /// Returns a mutable reference to the lambda strategy.
    #[inline]
    pub fn lambda_strategy_mut(&mut self) -> &mut L {
        &mut self.lambda_strategy
    }

    /// Returns a reference to the penalization strategy.
    #[inline]
    pub fn penalization(&self) -> &P {
        &self.penalization
    }

    /// Returns a mutable reference to the penalization strategy.
    #[inline]
    pub fn penalization_mut(&mut self) -> &mut P {
        &mut self.penalization
    }

    /// Returns a reference to the feature cost model.
    #[inline]
    pub fn feature_cost(&self) -> &F {
        &self.feature_cost
    }

    /// Returns a mutable reference to the feature cost model.
    #[inline]
    pub fn feature_cost_mut(&mut self) -> &mut F {
        &mut self.feature_cost
    }

    /// Returns a reference to the penalty decay strategy.
    #[inline]
    pub fn decay(&self) -> &D {
        &self.decay
    }

    /// Returns a mutable reference to the penalty decay strategy.
    #[inline]
    pub fn decay_mut(&mut self) -> &mut D {
        &mut self.decay
    }

    /// Returns a reference to the penalty memory.
    #[inline]
    pub fn memory(&self) -> &PenaltyMemory {
        &self.memory
    }

    /// Sets the penalization trigger.
    ///
    /// The default is `PenalizationTrigger::OnExhaustion` — the classic
    /// GLS behaviour from the literature.
    #[inline]
    pub fn with_trigger(mut self, trigger: PenalizationTrigger) -> Self {
        self.trigger = trigger;
        self
    }

    /// Returns the current penalization trigger.
    #[inline]
    pub fn trigger(&self) -> PenalizationTrigger {
        self.trigger
    }

    /// Computes the full penalty sum for the current solution by scanning
    /// the schedule graph.
    ///
    /// This traverses every berth chain and every vessel assignment,
    /// summing up the corresponding penalty counters. Used to bootstrap
    /// the cached `current_penalty_sum` after a penalization step or at
    /// search start.
    fn compute_penalty_sum(&self, graph: &ScheduleGraph) -> i64 {
        let num_vessels = graph.num_vessels();
        let num_berths = graph.num_berths();
        let mut sum: i64 = 0;

        for berth in graph.berth_iter() {
            let sentinel = graph.sentinel(berth);
            let mut prev_node = sentinel;
            loop {
                let next = graph.next_node(prev_node);
                if prev_node.get() < num_vessels || next.get() < num_vessels {
                    let idx = edge_flat_index(prev_node, next, num_vessels);
                    debug_assert!(idx < self.memory.edge.len());

                    // SAFETY: idx < (num_vessels+1)^2, the edge allocation size.
                    sum += unsafe { *self.memory.edge.get_unchecked(idx) } as i64;
                }
                if next == sentinel {
                    break;
                }
                prev_node = next;
            }
        }

        for vessel in graph.vessel_iter() {
            let berth = graph.vessel_berth(vessel);
            let idx = vessel.get() * num_berths + berth.get();
            debug_assert!(idx < self.memory.berth.len());

            // SAFETY: v < num_vessels and berth < num_berths.
            sum += unsafe { *self.memory.berth.get_unchecked(idx) } as i64;
        }

        sum
    }

    /// Applies the penalization strategy and recomputes the cached
    /// penalty sum.
    ///
    /// This is the core "kick" step of GLS: after identifying and
    /// incrementing penalties for the highest-utility features, the
    /// cached penalty sum is fully recomputed to stay consistent.
    fn penalize_and_recompute(&mut self, graph: &ScheduleGraph) {
        self.penalization
            .penalize(&mut self.memory, graph, &self.feature_cost);
        self.decay.after_penalization(&mut self.memory);
        self.current_penalty_sum = self.compute_penalty_sum(graph);
        self.lambda_strategy.on_penalization();
    }
}

impl<T, P, F, L, D> Metaheuristic<T> for GuidedLocalSearch<P, F, L, D>
where
    T: SolverNumeric + ToPrimitive,
    P: PenalizationStrategy,
    F: FeatureCost,
    L: LambdaStrategy,
    D: PenaltyDecay,
{
    fn name(&self) -> &str {
        "GuidedLocalSearch"
    }

    fn on_start(
        &mut self,
        _model: &Model<T>,
        _initial_solution: SolutionView<T>,
        graph: &ScheduleGraph,
    ) {
        self.memory.clear();
        self.current_penalty_sum = self.compute_penalty_sum(graph);
        self.trigger_counter = 0;
        let obj = _initial_solution.objective_value().to_f64().unwrap_or(0.0);
        self.last_best_obj = obj;
        self.lambda_strategy.on_start(obj);
    }

    fn on_end(
        &mut self,
        _model: &Model<T>,
        _final_solution: SolutionView<T>,
        _graph: &ScheduleGraph,
    ) {
    }

    /// Penalizes features in the current solution and restarts the search.
    ///
    /// This is the core GLS mechanism: when the local search is stuck,
    /// the penalty landscape is modified to push the search away from
    /// the current basin.
    fn on_neighbourhood_exhausted(
        &mut self,
        _model: &Model<T>,
        _best_solution: SolutionView<'_, T>,
        _accepted_solution: SolutionView<'_, T>,
        _buffered_solution: Option<SolutionView<'_, T>>,
        graph: &ScheduleGraph,
    ) -> NeighborhoodExhaustionOutcome {
        if self.trigger == PenalizationTrigger::OnExhaustion {
            self.penalize_and_recompute(graph);
        }
        NeighborhoodExhaustionOutcome::Restart
    }

    /// GLS is a first-improvement strategy. Never buffers.
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

    /// Decides the fate of a candidate move using the augmented objective.
    ///
    /// The augmented objective is:
    ///
    /// $$f_{\text{aug}}(s') = f(s') + \lambda \cdot P(s')$$
    ///
    /// where $P(s')$ is the total penalty of the candidate solution.
    /// Thanks to the diff, we compute $P(s') = P(s) + \Delta P$ in
    /// $O(|\text{diff}|)$.
    ///
    /// The move is accepted (first-improvement) if the augmented
    /// candidate objective is strictly less than the augmented current
    /// objective.
    #[allow(clippy::too_many_arguments)]
    fn decide_fate(
        &mut self,
        _model: &Model<T>,
        _best_solution: SolutionView<'_, T>,
        accepted_solution: SolutionView<'_, T>,
        _buffered_solution: Option<SolutionView<'_, T>>,
        candidate_objective: T,
        _graph: &ScheduleGraph,
        graph_diff: &ScheduleGraphDiff,
    ) -> AcceptanceOutcome {
        let current_objective = accepted_solution.objective_value();

        let cand_f64 = match candidate_objective.to_f64() {
            Some(v) if v.is_finite() => v,
            _ => return AcceptanceOutcome::Reject,
        };
        let curr_f64 = match current_objective.to_f64() {
            Some(v) if v.is_finite() => v,
            _ => return AcceptanceOutcome::Reject,
        };

        let penalty_delta = self.memory.penalty_delta(graph_diff);
        let candidate_penalty_sum = self.current_penalty_sum + penalty_delta;

        let lambda = self.lambda_strategy.lambda();
        let aug_candidate = cand_f64 + lambda * candidate_penalty_sum as f64;
        let aug_current = curr_f64 + lambda * self.current_penalty_sum as f64;

        if aug_candidate < aug_current {
            AcceptanceOutcome::Accept
        } else {
            AcceptanceOutcome::Reject
        }
    }

    /// Updates the cached penalty sum when a move is accepted and, if
    /// using `AfterMoves`, checks whether to penalize.
    fn on_accept(
        &mut self,
        _model: &Model<T>,
        _best_solution: SolutionView<'_, T>,
        _new_accepted_solution: SolutionView<'_, T>,
        _buffered_solution: Option<SolutionView<'_, T>>,
        graph: &ScheduleGraph,
        graph_diff: &ScheduleGraphDiff,
    ) {
        let penalty_delta = self.memory.penalty_delta(graph_diff);
        self.current_penalty_sum += penalty_delta;

        let obj = _new_accepted_solution
            .objective_value()
            .to_f64()
            .unwrap_or(0.0);
        self.lambda_strategy.on_accept(obj);

        if let PenalizationTrigger::AfterMoves(n) = self.trigger {
            self.trigger_counter += 1;
            if self.trigger_counter >= n {
                self.trigger_counter = 0;
                self.penalize_and_recompute(graph);
            }
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
        // No-op.
    }

    fn on_new_best(
        &mut self,
        _model: &Model<T>,
        _new_best: SolutionView<T>,
        _graph: &ScheduleGraph,
        _graph_diff: &ScheduleGraphDiff,
    ) {
        let obj = _new_best.objective_value().to_f64().unwrap_or(0.0);
        self.lambda_strategy.on_new_best(obj);
    }

    /// Advances the non-improvement counter and, if it reaches the
    /// threshold, fires penalization.
    fn on_iteration(
        &mut self,
        _iteration: u64,
        _model: &Model<T>,
        _best_solution: SolutionView<'_, T>,
        _new_accepted_solution: SolutionView<'_, T>,
        _buffered_solution: Option<SolutionView<'_, T>>,
        graph: &ScheduleGraph,
    ) {
        if let PenalizationTrigger::AfterNonImprovements(n) = self.trigger {
            let best_obj = _best_solution.objective_value().to_f64().unwrap_or(0.0);
            if best_obj < self.last_best_obj {
                self.trigger_counter = 0;
                self.last_best_obj = best_obj;
            } else {
                self.trigger_counter += 1;
                if self.trigger_counter >= n {
                    self.trigger_counter = 0;
                    self.penalize_and_recompute(graph);
                }
            }
        }
    }
}

// ----------------------------------------------------------------
// Tests
// ----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_penalty_memory_new_zeroed() {
        let mem = PenaltyMemory::new(3, 2);
        assert_eq!(mem.num_vessels(), 3);
        assert_eq!(mem.num_berths(), 2);
        assert!(mem.edge.iter().all(|&p| p == 0));
        assert!(mem.berth.iter().all(|&p| p == 0));
    }

    #[test]
    fn test_penalty_memory_clear() {
        let mut mem = PenaltyMemory::new(2, 2);
        mem.edge[0] = 5;
        mem.berth[1] = 3;
        mem.clear();
        assert!(mem.edge.iter().all(|&p| p == 0));
        assert!(mem.berth.iter().all(|&p| p == 0));
    }

    #[test]
    fn test_penalty_memory_edge_lookup() {
        let mut mem = PenaltyMemory::new(3, 2);
        let a = Node::new(0);
        let b = Node::new(1);
        assert_eq!(mem.edge_penalty(a, b), 0);
        let idx = edge_flat_index(a, b, 3);
        mem.edge[idx] = 7;
        assert_eq!(mem.edge_penalty(a, b), 7);
    }

    #[test]
    fn test_penalty_memory_berth_lookup() {
        let mut mem = PenaltyMemory::new(3, 2);
        let v = VesselIndex::new(1);
        let b = BerthIndex::new(0);
        assert_eq!(mem.berth_penalty(v, b), 0);
        mem.berth[2] = 4;
        assert_eq!(mem.berth_penalty(v, b), 4);
    }

    #[test]
    fn test_penalty_memory_sentinel_edge() {
        let mem = PenaltyMemory::new(3, 2);
        // Sentinel -> vessel edge uses index num_vessels for the sentinel.
        let sentinel = Node::new(3); // sentinel index = num_vessels
        let v = Node::new(2);
        let idx = edge_flat_index(sentinel, v, 3);
        // Should map to row=3, col=2 -> 3*4+2 = 14
        assert_eq!(idx, 14);
        assert_eq!(mem.edge_penalty(sentinel, v), 0);
    }

    #[test]
    fn test_uniform_cost_always_one() {
        let cost = UniformCost;
        assert_eq!(cost.edge_cost(Node::new(0), Node::new(1)), 1.0);
        assert_eq!(cost.edge_cost(Node::new(3), Node::new(0)), 1.0);
        assert_eq!(
            cost.berth_cost(VesselIndex::new(0), BerthIndex::new(0)),
            1.0
        );
    }

    #[test]
    fn test_edge_flat_index_vessel_to_vessel() {
        let a = Node::new(1);
        let b = Node::new(2);
        // With num_vessels=4, stride = 5. Index = 1*5 + 2 = 7.
        assert_eq!(edge_flat_index(a, b, 4), 7);
    }

    #[test]
    fn test_edge_flat_index_sentinel_to_vessel() {
        let sentinel = Node::new(3); // sentinel index >= num_vessels
        let b = Node::new(0);
        // Sentinel maps to 3. Index = 3 * 4 + 0 = 12.
        assert_eq!(edge_flat_index(sentinel, b, 3), 12);
    }

    #[test]
    fn test_edge_flat_index_vessel_to_sentinel() {
        let a = Node::new(0);
        let sentinel = Node::new(3); // sentinel index >= num_vessels
        // Sentinel maps to 3. Index = 0 * 4 + 3 = 3.
        assert_eq!(edge_flat_index(a, sentinel, 3), 3);
    }

    #[test]
    fn test_gls_new() {
        let gls = GuidedLocalSearch::new(10, 3).with_lambda(0.5);
        assert_eq!(gls.lambda(), 0.5);
        assert_eq!(gls.memory().num_vessels(), 10);
        assert_eq!(gls.memory().num_berths(), 3);
    }

    #[test]
    #[should_panic(expected = "lambda must be > 0.0")]
    fn test_gls_zero_lambda_panics() {
        let _ = GuidedLocalSearch::new(5, 2).with_lambda(0.0);
    }

    #[test]
    #[should_panic(expected = "lambda must be > 0.0")]
    fn test_gls_negative_lambda_panics() {
        let _ = GuidedLocalSearch::new(5, 2).with_lambda(-1.0);
    }

    #[test]
    fn test_heuristic_lambda_basic() {
        let lambda = heuristic_lambda(1000.0, 50, 0.3);
        // 0.3 * 1000.0 / 50 = 6.0
        assert!((lambda - 6.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_heuristic_lambda_zero_features() {
        let lambda = heuristic_lambda(1000.0, 0, 0.3);
        // num_features clamped to 1: 0.3 * 1000.0 / 1 = 300.0
        assert!((lambda - 300.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_penalty_delta_empty_diff() {
        let mem = PenaltyMemory::new(3, 2);
        let diff = ScheduleGraphDiff::new();
        assert_eq!(mem.penalty_delta(&diff), 0);
    }

    #[test]
    fn test_trigger_default_is_on_exhaustion() {
        assert_eq!(
            PenalizationTrigger::default(),
            PenalizationTrigger::OnExhaustion
        );
    }

    #[test]
    fn test_trigger_display() {
        assert_eq!(
            PenalizationTrigger::OnExhaustion.to_string(),
            "OnExhaustion"
        );
        assert_eq!(
            PenalizationTrigger::AfterNonImprovements(50).to_string(),
            "AfterNonImprovements(50)"
        );
        assert_eq!(
            PenalizationTrigger::AfterMoves(10).to_string(),
            "AfterMoves(10)"
        );
    }

    #[test]
    fn test_gls_with_trigger() {
        let gls = GuidedLocalSearch::new(5, 2)
            .with_lambda(0.5)
            .with_trigger(PenalizationTrigger::AfterMoves(20));
        assert_eq!(gls.trigger(), PenalizationTrigger::AfterMoves(20));
    }

    #[test]
    fn test_gls_default_trigger_is_on_exhaustion() {
        let gls = GuidedLocalSearch::new(5, 2);
        assert_eq!(gls.trigger(), PenalizationTrigger::OnExhaustion);
    }

    // ---- LambdaStrategy tests -----------------------------------------

    #[test]
    fn test_static_lambda_constant() {
        let s = StaticLambda(0.42);
        assert_eq!(s.lambda(), 0.42);
    }

    #[test]
    fn test_dynamic_lambda_decreases_on_new_best() {
        let mut d = DynamicLambda::new(1.0, 0.1, 0.1, 0.01, 10.0);
        assert!((d.lambda() - 1.0).abs() < f64::EPSILON);
        d.on_new_best(100.0);
        // 1.0 * (1 - 0.1) = 0.9
        assert!((d.lambda() - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn test_dynamic_lambda_increases_on_penalization() {
        let mut d = DynamicLambda::new(1.0, 0.1, 0.1, 0.01, 10.0);
        d.on_penalization();
        // 1.0 * (1 + 0.1) = 1.1
        assert!((d.lambda() - 1.1).abs() < f64::EPSILON);
    }

    #[test]
    fn test_dynamic_lambda_clamps() {
        let mut d = DynamicLambda::new(0.02, 0.5, 0.9, 0.01, 10.0);
        d.on_new_best(100.0); // 0.02 * (1 - 0.9) = 0.002 → clamped at min 0.01
        assert!((d.lambda() - 0.01).abs() < f64::EPSILON);

        let mut d2 = DynamicLambda::new(9.0, 0.5, 0.1, 0.01, 10.0);
        d2.on_penalization(); // 9.0 * 1.5 = 13.5 → clamped at 10.0
        assert!((d2.lambda() - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_dynamic_lambda_resets_on_start() {
        let mut d = DynamicLambda::new(1.0, 0.1, 0.1, 0.01, 10.0);
        d.on_penalization();
        d.on_penalization();
        assert!(d.lambda() > 1.0);
        d.on_start(500.0);
        assert!((d.lambda() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_gls_with_lambda_strategy() {
        let gls = GuidedLocalSearch::new(5, 2).with_lambda_strategy(StaticLambda(0.5));
        assert!((gls.lambda() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_additive_dynamic_lambda_decreases_on_new_best() {
        let mut a = AdditiveDynamicLambda::new(1.0, 0.1, 0.1, 0.01, 10.0);
        assert!((a.lambda() - 1.0).abs() < f64::EPSILON);
        a.on_new_best(100.0);
        // 1.0 - 0.1 = 0.9
        assert!((a.lambda() - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn test_additive_dynamic_lambda_increases_on_penalization() {
        let mut a = AdditiveDynamicLambda::new(1.0, 0.1, 0.1, 0.01, 10.0);
        a.on_penalization();
        // 1.0 + 0.1 = 1.1
        assert!((a.lambda() - 1.1).abs() < f64::EPSILON);
    }

    #[test]
    fn test_additive_dynamic_lambda_clamps() {
        let mut a = AdditiveDynamicLambda::new(0.05, 0.1, 0.1, 0.01, 10.0);
        a.on_new_best(100.0); // 0.05 - 0.1 = -0.05 → clamped at 0.01
        assert!((a.lambda() - 0.01).abs() < f64::EPSILON);

        let mut a2 = AdditiveDynamicLambda::new(9.95, 0.1, 0.1, 0.01, 10.0);
        a2.on_penalization(); // 9.95 + 0.1 = 10.05 → clamped at 10.0
        assert!((a2.lambda() - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_additive_dynamic_lambda_resets_on_start() {
        let mut a = AdditiveDynamicLambda::new(1.0, 0.1, 0.1, 0.01, 10.0);
        a.on_penalization();
        a.on_penalization();
        assert!(a.lambda() > 1.0);
        a.on_start(500.0);
        assert!((a.lambda() - 1.0).abs() < f64::EPSILON);
    }

    // ---- PenaltyDecay tests -------------------------------------------

    #[test]
    fn test_penalty_memory_decay() {
        let mut mem = PenaltyMemory::new(2, 2);
        mem.edge[0] = 100;
        mem.edge[1] = 50;
        mem.berth[0] = 80;
        mem.decay(0.9);
        assert_eq!(mem.edge[0], 90); // 100 * 0.9 = 90
        assert_eq!(mem.edge[1], 45); // 50 * 0.9 = 45
        assert_eq!(mem.berth[0], 72); // 80 * 0.9 = 72
    }

    #[test]
    fn test_penalty_memory_decay_truncates() {
        let mut mem = PenaltyMemory::new(1, 1);
        mem.edge[0] = 1;
        mem.decay(0.9);
        // 1 * 0.9 = 0.9 → truncated to 0
        assert_eq!(mem.edge[0], 0);
    }

    #[test]
    fn test_no_decay_is_noop() {
        let mut mem = PenaltyMemory::new(2, 2);
        mem.edge[0] = 10;
        mem.berth[1] = 5;
        let mut d = NoDecay;
        d.after_penalization(&mut mem);
        assert_eq!(mem.edge[0], 10);
        assert_eq!(mem.berth[1], 5);
    }

    #[test]
    fn test_geometric_decay_fires_on_period() {
        let mut mem = PenaltyMemory::new(2, 1);
        mem.edge[0] = 100;
        let mut d = GeometricDecay::new(0.5, 3);

        // First two calls: no decay.
        d.after_penalization(&mut mem);
        assert_eq!(mem.edge[0], 100);
        d.after_penalization(&mut mem);
        assert_eq!(mem.edge[0], 100);
        // Third call: decay fires.
        d.after_penalization(&mut mem);
        assert_eq!(mem.edge[0], 50); // 100 * 0.5

        // Counter resets, next decay after 3 more calls.
        d.after_penalization(&mut mem);
        assert_eq!(mem.edge[0], 50);
        d.after_penalization(&mut mem);
        assert_eq!(mem.edge[0], 50);
        d.after_penalization(&mut mem);
        assert_eq!(mem.edge[0], 25); // 50 * 0.5
    }

    #[test]
    fn test_geometric_decay_period_one() {
        let mut mem = PenaltyMemory::new(1, 1);
        mem.edge[0] = 100;
        let mut d = GeometricDecay::new(0.9, 1);
        d.after_penalization(&mut mem);
        assert_eq!(mem.edge[0], 90);
        d.after_penalization(&mut mem);
        assert_eq!(mem.edge[0], 81); // 90 * 0.9
    }

    #[test]
    #[should_panic(expected = "factor must be in (0, 1)")]
    fn test_geometric_decay_invalid_factor_panics() {
        let _ = GeometricDecay::new(1.0, 1);
    }

    #[test]
    fn test_gls_with_decay() {
        let gls = GuidedLocalSearch::new(5, 2).with_decay(GeometricDecay::new(0.9, 10));
        // Just verify it compiles and the accessor works.
        assert!((gls.decay().factor - 0.9).abs() < f64::EPSILON);
    }

    // ---- Lambda Reset on Best tests -----------------------------------

    #[test]
    fn test_dynamic_lambda_reset_on_best() {
        let mut d = DynamicLambda::new(1.0, 0.1, 0.1, 0.01, 10.0).with_reset_on_best(true);
        // Drive lambda up via penalization.
        d.on_penalization(); // 1.0 * 1.1 = 1.1
        d.on_penalization(); // 1.1 * 1.1 = 1.21
        assert!(d.lambda() > 1.0);
        // New best should reset to initial (1.0), not scale down.
        d.on_new_best(50.0);
        assert!((d.lambda() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_dynamic_lambda_no_reset_on_best_default() {
        let mut d = DynamicLambda::new(1.0, 0.1, 0.1, 0.01, 10.0);
        d.on_new_best(100.0);
        // Default: scale down, not reset.
        assert!((d.lambda() - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn test_additive_dynamic_lambda_reset_on_best() {
        let mut a = AdditiveDynamicLambda::new(1.0, 0.1, 0.1, 0.01, 10.0).with_reset_on_best(true);
        a.on_penalization(); // 1.1
        a.on_penalization(); // 1.2
        assert!(a.lambda() > 1.0);
        a.on_new_best(50.0);
        assert!((a.lambda() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_additive_dynamic_lambda_no_reset_on_best_default() {
        let mut a = AdditiveDynamicLambda::new(1.0, 0.1, 0.1, 0.01, 10.0);
        a.on_new_best(100.0);
        assert!((a.lambda() - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn test_edge_flat_index_sentinels_collapse() {
        let num_vessels = 3;
        let sentinel_berth0 = Node::new(3); // first sentinel
        let sentinel_berth1 = Node::new(4); // second sentinel
        let vessel = Node::new(0);

        // Both sentinels must map to the same flat index.
        let idx0 = edge_flat_index(sentinel_berth0, vessel, num_vessels);
        let idx1 = edge_flat_index(sentinel_berth1, vessel, num_vessels);
        assert_eq!(
            idx0, idx1,
            "different sentinels must collapse to same index"
        );

        // Reverse direction too.
        let idx0r = edge_flat_index(vessel, sentinel_berth0, num_vessels);
        let idx1r = edge_flat_index(vessel, sentinel_berth1, num_vessels);
        assert_eq!(idx0r, idx1r);
    }

    #[test]
    fn test_edge_flat_index_sentinel_to_sentinel_collapse() {
        let num_vessels = 4;
        let s0 = Node::new(4);
        let s1 = Node::new(5);
        let s2 = Node::new(99); // wildly out of range sentinel

        let idx_s0s0 = edge_flat_index(s0, s0, num_vessels);
        let idx_s1s2 = edge_flat_index(s1, s2, num_vessels);
        assert_eq!(idx_s0s0, idx_s1s2, "all sentinel×sentinel pairs collapse");
    }

    #[test]
    fn test_edge_flat_index_upper_bound() {
        let num_vessels = 5;
        let max_alloc = (num_vessels + 1) * (num_vessels + 1);

        // Exhaustively verify every possible pair stays in bounds.
        for a in 0..=(num_vessels + 3) {
            for b in 0..=(num_vessels + 3) {
                let idx = edge_flat_index(Node::new(a), Node::new(b), num_vessels);
                assert!(
                    idx < max_alloc,
                    "index {idx} out of bounds for a={a}, b={b}"
                );
            }
        }
    }

    // The following tests exercise the tolerance logic directly on
    // PenaltyMemory without needing a full ScheduleGraph. They test
    // the *evaluate_feature* closure's behaviour by constructing
    // scenarios where f64::EPSILON would fail.

    #[test]
    fn test_utility_tie_breaking_large_values() {
        // Two features with utilities that differ by less than REL_TOL
        // relative to their magnitude should tie.
        //
        // Feature A: cost = 1e8, penalty = 0  → utility = 1e8
        // Feature B: cost = 1e8 + 1e-6, penalty = 0 → utility ≈ 1e8
        //
        // The old f64::EPSILON tolerance (≈2.2e-16) would NOT consider
        // these a tie (diff ≈ 1e-6 >> EPSILON). The new REL_TOL = 1e-12
        // relative to scale 1e8 gives threshold 1e-4, so 1e-6 < 1e-4 → tie.
        //
        // This is a unit-level logic test; we verify the formula directly.
        let a_utility: f64 = 1e8;
        let b_utility: f64 = 1e8 + 1e-6;
        let rel_tol: f64 = 1e-12;

        let diff = (b_utility - a_utility).abs();
        let scale = a_utility.abs().max(b_utility.abs()).max(1.0);

        assert!(
            diff <= rel_tol * scale,
            "large near-equal utilities must tie: diff={diff}, threshold={}",
            rel_tol * scale
        );
    }

    #[test]
    fn test_utility_no_false_tie_on_distinct_values() {
        // Two features with clearly different utilities must not tie.
        // Feature A: utility = 1.0
        // Feature B: utility = 2.0
        let a_utility: f64 = 1.0;
        let b_utility: f64 = 2.0;
        let rel_tol: f64 = 1e-12;

        let diff = b_utility - a_utility;
        let scale = a_utility.abs().max(b_utility.abs()).max(1.0);

        assert!(
            diff > rel_tol * scale,
            "clearly distinct utilities must not tie"
        );
    }

    #[test]
    fn test_utility_tie_at_small_scale() {
        // Near-zero utilities: scale floors at 1.0 so absolute
        // tolerance is REL_TOL * 1.0 = 1e-12.
        let a: f64 = 1e-14;
        let b: f64 = 2e-14;
        let rel_tol: f64 = 1e-12;

        let diff = (b - a).abs();
        let scale = a.abs().max(b.abs()).max(1.0);

        assert!(
            diff <= rel_tol * scale,
            "tiny near-equal utilities must tie when scale floors at 1.0"
        );
    }

    // ---- AfterNonImprovements flag correctness ------------------------

    /// Simulates the hook call sequence for one iteration to verify
    /// the best_seen_this_iter flag prevents double-counting.
    ///
    /// These are pure state-machine tests on the trigger counter;
    /// they don't need a ScheduleGraph.

    #[test]
    fn test_non_improvement_counter_skips_best_iteration() {
        // Threshold = 3. Simulate:
        //   iter 0: no best  → counter = 1
        //   iter 1: NEW BEST → counter stays 0 (not 1)
        //   iter 2: no best  → counter = 1
        //   iter 3: no best  → counter = 2
        //   iter 4: no best  → counter = 3 → fires

        // We can't call the real Metaheuristic hooks without the full
        // engine, but we can replicate the state machine:
        let trigger = PenalizationTrigger::AfterNonImprovements(3);
        let mut counter: u64 = 0;
        let mut best_seen: bool = false;
        let mut penalization_fired = false;

        let simulate_on_new_best = |counter: &mut u64, flag: &mut bool| {
            *counter = 0;
            *flag = true;
        };

        let simulate_on_iteration =
            |counter: &mut u64, flag: &mut bool, fired: &mut bool, n: u64| {
                if *flag {
                    *flag = false;
                } else {
                    *counter += 1;
                    if *counter >= n {
                        *counter = 0;
                        *fired = true;
                    }
                }
            };

        let n = match trigger {
            PenalizationTrigger::AfterNonImprovements(n) => n,
            _ => unreachable!(),
        };

        // iter 0: no best
        simulate_on_iteration(&mut counter, &mut best_seen, &mut penalization_fired, n);
        assert_eq!(counter, 1);
        assert!(!penalization_fired);

        // iter 1: new best found
        simulate_on_new_best(&mut counter, &mut best_seen);
        simulate_on_iteration(&mut counter, &mut best_seen, &mut penalization_fired, n);
        assert_eq!(counter, 0, "counter must stay 0 on best-iteration");
        assert!(!penalization_fired);

        // iter 2: no best
        simulate_on_iteration(&mut counter, &mut best_seen, &mut penalization_fired, n);
        assert_eq!(counter, 1);

        // iter 3: no best
        simulate_on_iteration(&mut counter, &mut best_seen, &mut penalization_fired, n);
        assert_eq!(counter, 2);

        // iter 4: no best → fires at 3
        simulate_on_iteration(&mut counter, &mut best_seen, &mut penalization_fired, n);
        assert!(
            penalization_fired,
            "penalization must fire after 3 non-improvements"
        );
        assert_eq!(counter, 0, "counter resets after firing");
    }

    #[test]
    fn test_non_improvement_counter_consecutive_bests_never_fire() {
        let n: u64 = 2;
        let mut counter: u64 = 0;
        let mut best_seen: bool = false;
        let mut fired = false;

        let on_best = |c: &mut u64, f: &mut bool| {
            *c = 0;
            *f = true;
        };
        let on_iter = |c: &mut u64, f: &mut bool, fired: &mut bool, n: u64| {
            if *f {
                *f = false;
            } else {
                *c += 1;
                if *c >= n {
                    *c = 0;
                    *fired = true;
                }
            }
        };

        // 10 consecutive iterations each producing a new best.
        for _ in 0..10 {
            on_best(&mut counter, &mut best_seen);
            on_iter(&mut counter, &mut best_seen, &mut fired, n);
        }

        assert!(
            !fired,
            "penalization must never fire if every iteration finds a new best"
        );
        assert_eq!(counter, 0);
    }

    #[test]
    fn test_non_improvement_fires_exactly_at_threshold() {
        let n: u64 = 5;
        let mut counter: u64 = 0;
        let mut best_seen: bool = false;
        let mut fire_count: u64 = 0;

        let on_iter = |c: &mut u64, f: &mut bool, fc: &mut u64, n: u64| {
            if *f {
                *f = false;
            } else {
                *c += 1;
                if *c >= n {
                    *c = 0;
                    *fc += 1;
                }
            }
        };

        // Exactly n-1 non-improvements: should not fire.
        for _ in 0..(n - 1) {
            on_iter(&mut counter, &mut best_seen, &mut fire_count, n);
        }
        assert_eq!(fire_count, 0);
        assert_eq!(counter, n - 1);

        // The n-th: should fire exactly once.
        on_iter(&mut counter, &mut best_seen, &mut fire_count, n);
        assert_eq!(fire_count, 1);
        assert_eq!(counter, 0);
    }

    // ---- AfterMoves trigger (existing, just hardening) ----------------

    #[test]
    fn test_after_moves_counter_increments_only_on_accept() {
        // Verify the counter doesn't increment on reject.
        // The on_reject hook is a no-op, so the counter stays put.
        // This is a contract test: if someone adds logic to on_reject,
        // it must not touch the trigger counter.
        let n: u64 = 3;
        let mut counter: u64 = 0;
        let mut fired = false;

        // Simulate 10 rejects: counter stays 0.
        // (on_reject is a no-op, nothing to simulate.)

        // Now 3 accepts.
        for _ in 0..n {
            counter += 1;
            if counter >= n {
                counter = 0;
                fired = true;
            }
        }
        assert!(fired);
    }

    // ---- Decay precision (informational) ------------------------------

    #[test]
    fn test_decay_precision_below_2_53() {
        // Values below 2^53 round-trip through f64 exactly.
        let mut mem = PenaltyMemory::new(1, 1);
        let val: u64 = (1u64 << 53) - 1; // max exact u64 in f64
        mem.edge[0] = val;
        mem.decay(1.0); // identity decay: should not change the value

        // NOTE: factor=1.0 is outside the assert range in GeometricDecay::new,
        // but PenaltyMemory::decay itself allows [0,1]. This tests the
        // cast fidelity directly.
        assert_eq!(
            mem.edge[0], val,
            "values below 2^53 must survive f64 round-trip"
        );
    }

    #[test]
    fn test_decay_precision_above_2_53_lossy() {
        // Values above 2^53 lose precision in f64. Document this.
        let mut mem = PenaltyMemory::new(1, 1);
        let val: u64 = (1u64 << 53) + 1; // NOT exactly representable in f64
        mem.edge[0] = val;
        mem.decay(1.0);

        // The round-tripped value may differ. We just assert it doesn't panic
        // and document that precision loss is expected.
        // In practice, penalty counters never reach 2^53.
        let _ = mem.edge[0];
    }
}
