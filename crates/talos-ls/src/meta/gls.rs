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
//!
//! # Feature Decomposition
//!
//! This implementation tracks two independent feature classes via
//! `PenaltyMemory`:
//!
//! | Feature              | Memory                              | Semantics                               |
//! |----------------------|-------------------------------------|-----------------------------------------|
//! | **Edge (Sequence)**  | `edge[A * V + B] = penalty_count`   | Vessel A directly preceding Vessel B    |
//! | **Berth (Assignment)** | `berth[V * B + X] = penalty_count`| Vessel V assigned to Berth X            |
//!
//! # Penalization Strategy
//!
//! The penalization strategy is pluggable via the `PenalizationStrategy`
//! trait. The default is `MaxUtilityPenalization`, which selects the feature
//! with the highest *utility* — the ratio of the feature's *indicator cost*
//! to one plus the number of times it has already been penalized:
//!
//! $$\text{utility}(f) = \frac{I_f \cdot c_f}{1 + p_f}$$
//!
//! where $I_f$ is 1 if the feature is present in the current solution,
//! $c_f$ is the feature's cost contribution, and $p_f$ is the current
//! penalty count.
//!
//! # Feature Cost
//!
//! The feature cost is pluggable via the `FeatureCost` trait. The default
//! is `UniformCost`, which assigns a cost of 1 to every feature — suitable
//! when the real per-edge or per-assignment cost is not available or when
//! all features should be penalized equally.
//!
//! # Augmented Objective
//!
//! When evaluating a candidate move, the engine's raw objective $f(s')$ is
//! augmented by the penalty delta computed from the diff:
//!
//! $$f_{\text{aug}}(s') = f(s') + \lambda \cdot \Delta P$$
//!
//! where $\lambda$ is a tuneable weight and $\Delta P$ is the net change in
//! penalty across broken/created edges and old/new berth assignments.
//!
//! Because the diff is small (typically 2–6 entries), computing $\Delta P$
//! is $O(|\text{diff}|)$ per candidate — very cheap.
//!
//! # Engine Integration
//!
//! GLS wraps a **first-improvement** local search. Each candidate is either
//! accepted or rejected on the spot using the augmented objective. The
//! `Metaheuristic::should_commit_buffered` hook always returns `false`.
//!
//! When the neighbourhood is exhausted (local optimum), GLS penalizes
//! features in the current solution and returns
//! `NeighborhoodExhaustionOutcome::Restart` to begin a new scan.
//!
//! # Penalization Trigger
//!
//! The *when* of penalization is configurable via `PenalizationTrigger`:
//!
//! | Trigger                        | Fires when                                           |
//! |--------------------------------|------------------------------------------------------|
//! | `OnExhaustion`                 | Neighbourhood is exhausted (classic GLS).            |
//! | `AfterNonImprovements(n)`      | `n` consecutive iterations without a new global best.|
//! | `AfterMoves(n)`                | Every `n` accepted moves.                            |
//!
//! # Lambda Tuning
//!
//! The penalty weight $\lambda$ controls how aggressively GLS diversifies.
//! A common heuristic from the literature is:
//!
//! $$\lambda = \alpha \cdot \frac{f(s^*)}{|F^*|}$$
//!
//! where $f(s^*)$ is the best known objective and $|F^*|$ is the number of
//! features present in the best solution. The `heuristic_lambda` free
//! function computes this.

use crate::{
    exec::SearchCommand,
    meta::metaheuristic::{AcceptanceOutcome, Metaheuristic, NeighborhoodExhaustionOutcome},
    sgraph::{ScheduleGraph, ScheduleGraphDiff},
};
use num_traits::ToPrimitive;
use talos_core::utils::num::SolverNumeric;
use talos_model::{
    index::{BerthIndex, VesselIndex},
    model::Model,
    solution::SolutionView,
};

// ──────────────────────────────────────────────────────────────
// Feature Cost
// ──────────────────────────────────────────────────────────────

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
    /// Returns the cost of an edge feature (vessel `a` directly preceding
    /// vessel `b`). Sentinel edges (where one end is `None`) may return 0
    /// if they should not be penalized.
    fn edge_cost(&self, a: Option<VesselIndex>, b: Option<VesselIndex>) -> f64;

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
    fn edge_cost(&self, _a: Option<VesselIndex>, _b: Option<VesselIndex>) -> f64 {
        1.0
    }

    #[inline]
    fn berth_cost(&self, _v: VesselIndex, _b: BerthIndex) -> f64 {
        1.0
    }
}

// ──────────────────────────────────────────────────────────────
// Penalization Strategy
// ──────────────────────────────────────────────────────────────

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
        // Reuse the scratch buffer — clear but keep the allocation.
        self.buf.clear();

        // Walk every berth chain: sentinel -> v1 -> v2 -> ... -> sentinel.
        for b in 0..num_berths {
            let berth = BerthIndex::new(b);
            let sentinel = graph.sentinel(berth);
            let mut prev_node = sentinel;
            loop {
                let next = graph.next_node(prev_node);
                // Determine vessel indices (None for sentinel).
                let from = if prev_node.get() < num_vessels {
                    Some(VesselIndex::new(prev_node.get()))
                } else {
                    None
                };
                let to = if next.get() < num_vessels {
                    Some(VesselIndex::new(next.get()))
                } else {
                    None
                };

                // Only consider edges where at least one end is a vessel.
                if from.is_some() || to.is_some() {
                    let c = cost.edge_cost(from, to);
                    if c > 0.0 {
                        let idx = edge_flat_index(from, to, num_vessels);
                        // SAFETY: idx < (num_vessels+1)^2, the edge allocation size.
                        let p = unsafe { *memory.edge.get_unchecked(idx) } as f64;
                        let utility = c / (1.0 + p);

                        if utility > best_utility + f64::EPSILON {
                            best_utility = utility;
                            self.buf.clear();
                            self.buf.push((true, idx));
                        } else if (utility - best_utility).abs() <= f64::EPSILON {
                            self.buf.push((true, idx));
                        }
                    }
                }

                if next == sentinel {
                    break;
                }
                prev_node = next;
            }
        }

        for v in 0..num_vessels {
            let vessel = VesselIndex::new(v);
            let berth = graph.vessel_berth(vessel);
            let c = cost.berth_cost(vessel, berth);
            if c > 0.0 {
                let idx = v * num_berths + berth.get();
                // SAFETY: idx = v * num_berths + berth < num_vessels * num_berths.
                let p = unsafe { *memory.berth.get_unchecked(idx) } as f64;
                let utility = c / (1.0 + p);

                if utility > best_utility + f64::EPSILON {
                    best_utility = utility;
                    self.buf.clear();
                    self.buf.push((false, idx));
                } else if (utility - best_utility).abs() <= f64::EPSILON {
                    self.buf.push((false, idx));
                }
            }
        }

        // Increment penalty for all winning features.
        for &(is_edge, idx) in &self.buf {
            // SAFETY: idx was computed from the scan loops above and is within bounds.
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

// ──────────────────────────────────────────────────────────────
// Penalty Memory
// ──────────────────────────────────────────────────────────────

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
    pub(crate) edge: Vec<u64>,

    /// Flat `num_vessels * num_berths` array.
    /// `berth[v * num_berths + b]` stores the penalty count for
    /// assigning vessel $v$ to berth $b$.
    pub(crate) berth: Vec<u64>,

    num_vessels: usize,
    num_berths: usize,
}

/// Computes the flat index for an edge feature, mapping `None` (sentinel)
/// to the index `num_vessels`.
#[inline]
fn edge_flat_index(
    from: Option<VesselIndex>,
    to: Option<VesselIndex>,
    num_vessels: usize,
) -> usize {
    let a = from.map_or(num_vessels, |v| v.get());
    let b = to.map_or(num_vessels, |v| v.get());
    a * (num_vessels + 1) + b
}

impl PenaltyMemory {
    /// Creates a new penalty memory for the given problem dimensions.
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
    #[inline]
    pub fn edge_penalty(&self, from: Option<VesselIndex>, to: Option<VesselIndex>) -> u64 {
        // SAFETY: edge_flat_index yields < (num_vessels+1)^2, the allocation size.
        unsafe {
            *self
                .edge
                .get_unchecked(edge_flat_index(from, to, self.num_vessels))
        }
    }

    /// Returns the penalty count for a given berth-assignment feature.
    #[inline]
    pub fn berth_penalty(&self, vessel: VesselIndex, berth: BerthIndex) -> u64 {
        // SAFETY: vessel < num_vessels and berth < num_berths.
        unsafe {
            *self
                .berth
                .get_unchecked(vessel.get() * self.num_berths + berth.get())
        }
    }

    /// Computes the augmented penalty delta for a candidate move described
    /// by `diff`.
    ///
    /// The delta is: penalties gained from *created* edges and *new* berth
    /// assignments minus penalties lost from *broken* edges and *old* berth
    /// assignments.
    #[inline]
    pub fn penalty_delta(&self, diff: &ScheduleGraphDiff) -> i64 {
        let mut delta: i64 = 0;

        // SAFETY: all flat indices are bounded by allocation sizes.
        // Edge indices < (num_vessels+1)^2, berth indices < num_vessels*num_berths.
        unsafe {
            // Subtract penalties for broken edges.
            for edge in diff.broken_links() {
                let idx = edge_flat_index(edge.from, edge.to, self.num_vessels);
                delta -= *self.edge.get_unchecked(idx) as i64;
            }

            // Add penalties for created edges.
            for edge in diff.created_links() {
                let idx = edge_flat_index(edge.from, edge.to, self.num_vessels);
                delta += *self.edge.get_unchecked(idx) as i64;
            }

            // Subtract penalties for old berth assignments, add for new.
            for (vessel, old_berth, new_berth) in diff.reallocations() {
                delta -= *self
                    .berth
                    .get_unchecked(vessel.get() * self.num_berths + old_berth.get())
                    as i64;
                delta += *self
                    .berth
                    .get_unchecked(vessel.get() * self.num_berths + new_berth.get())
                    as i64;
            }
        }

        delta
    }
}

// ──────────────────────────────────────────────────────────────
// Penalization Trigger
// ──────────────────────────────────────────────────────────────

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

// ──────────────────────────────────────────────────────────────
// Lambda Strategy
// ──────────────────────────────────────────────────────────────

/// Controls the penalty weight $\lambda$ used in the augmented objective.
///
/// Implementations range from a simple fixed value (`FixedLambda`) to
/// fully reactive auto-tuning (`ReactiveLambda`) that adjusts $\lambda$
/// based on search progress.
pub trait LambdaStrategy: std::fmt::Debug {
    /// Returns the current penalty weight.
    fn weight(&self) -> f64;

    /// Called when the engine drops a penalty (i.e. the penalization
    /// trigger fires). Reactive strategies should *increase* $\lambda$
    /// here to escape the local optimum more aggressively.
    fn on_penalize(&mut self);

    /// Called when the engine discovers a new all-time global best.
    /// Reactive strategies should *decrease* $\lambda$ here to exploit
    /// the promising region.
    fn on_new_best(&mut self);
}

/// A fixed (static) $\lambda$ that never changes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FixedLambda {
    value: f64,
}

impl FixedLambda {
    /// Creates a new fixed lambda.
    ///
    /// # Panics
    ///
    /// Panics if `value` is not positive.
    #[inline]
    pub fn new(value: f64) -> Self {
        assert!(
            value > 0.0,
            "FixedLambda: value must be > 0.0, got {}",
            value
        );
        Self { value }
    }
}

impl LambdaStrategy for FixedLambda {
    #[inline]
    fn weight(&self) -> f64 {
        self.value
    }

    #[inline]
    fn on_penalize(&mut self) {}

    #[inline]
    fn on_new_best(&mut self) {}
}

/// A reactive $\lambda$ that self-adjusts based on search progress.
///
/// - When the search is stuck (penalization fires), $\lambda$ is
///   multiplied by `growth_factor` to increase diversification pressure.
/// - When a new global best is found, $\lambda$ is multiplied by
///   `decay_factor` (< 1) to reduce diversification and exploit the area.
///
/// Clamped to `[min_lambda, max_lambda]`.
#[derive(Debug, Clone, PartialEq)]
pub struct ReactiveLambda {
    current: f64,
    growth_factor: f64,
    decay_factor: f64,
    min_lambda: f64,
    max_lambda: f64,
}

impl ReactiveLambda {
    /// Creates a new reactive lambda.
    ///
    /// # Arguments
    ///
    /// * `initial` — Starting $\lambda$ (e.g. from `heuristic_lambda`).
    /// * `growth_factor` — Multiplier when stuck (e.g. 1.2).
    /// * `decay_factor` — Multiplier on new best (e.g. 0.8).
    /// * `min_lambda` — Floor clamp.
    /// * `max_lambda` — Ceiling clamp.
    ///
    /// # Panics
    ///
    /// Panics if `initial`, `min_lambda`, or `max_lambda` are not
    /// positive, if `growth_factor <= 1.0`, if `decay_factor >= 1.0`
    /// or `<= 0.0`, or if `min_lambda > max_lambda`.
    #[inline]
    pub fn new(
        initial: f64,
        growth_factor: f64,
        decay_factor: f64,
        min_lambda: f64,
        max_lambda: f64,
    ) -> Self {
        assert!(initial > 0.0, "ReactiveLambda: initial must be > 0.0");
        assert!(
            growth_factor > 1.0,
            "ReactiveLambda: growth_factor must be > 1.0"
        );
        assert!(
            decay_factor > 0.0 && decay_factor < 1.0,
            "ReactiveLambda: decay_factor must be in (0.0, 1.0)"
        );
        assert!(min_lambda > 0.0, "ReactiveLambda: min_lambda must be > 0.0");
        assert!(
            max_lambda >= min_lambda,
            "ReactiveLambda: max_lambda must be >= min_lambda"
        );
        Self {
            current: initial.clamp(min_lambda, max_lambda),
            growth_factor,
            decay_factor,
            min_lambda,
            max_lambda,
        }
    }
}

impl LambdaStrategy for ReactiveLambda {
    #[inline]
    fn weight(&self) -> f64 {
        self.current
    }

    #[inline]
    fn on_penalize(&mut self) {
        self.current = (self.current * self.growth_factor).min(self.max_lambda);
    }

    #[inline]
    fn on_new_best(&mut self) {
        self.current = (self.current * self.decay_factor).max(self.min_lambda);
    }
}

/// Computes a heuristic $\lambda$ from the initial objective.
///
/// $$\lambda = \alpha \cdot \frac{f(s_0)}{|F_0|}$$
///
/// where $|F_0|$ is the number of features present in the initial
/// solution and $\alpha$ is a scaling factor (typically 0.1–0.5).
///
/// # Arguments
///
/// * `objective` — The initial solution's objective value.
/// * `num_features` — The number of features in the initial solution
///   (e.g., `num_vessels + num_edges`).
/// * `alpha` — Scaling factor (e.g., 0.3).
#[inline]
pub fn heuristic_lambda(objective: f64, num_features: usize, alpha: f64) -> f64 {
    let nf = num_features.max(1) as f64;
    (alpha * objective.abs() / nf).max(f64::EPSILON)
}

// ──────────────────────────────────────────────────────────────
// Guided Local Search
// ──────────────────────────────────────────────────────────────

/// Guided Local Search metaheuristic with pluggable penalization,
/// feature-cost, and lambda strategies.
///
/// See the [module-level documentation](self) for algorithmic details.
pub struct GuidedLocalSearch<P = MaxUtilityPenalization, F = UniformCost, L = FixedLambda> {
    /// The penalization strategy applied at local optima.
    penalization: P,

    /// The feature cost model.
    feature_cost: F,

    /// Long-term penalty memory.
    memory: PenaltyMemory,

    /// The lambda strategy controlling the penalty weight.
    lambda: L,

    /// Cached augmented penalty of the current accepted solution.
    /// This avoids recomputing the full penalty sum on every candidate.
    current_penalty_sum: i64,

    /// Controls when penalization is triggered.
    trigger: PenalizationTrigger,

    /// Counter for the active trigger (iterations without improvement
    /// or accepted moves, depending on the trigger variant).
    trigger_counter: u64,
}

impl<P: std::fmt::Debug, F: std::fmt::Debug, L: std::fmt::Debug> std::fmt::Debug
    for GuidedLocalSearch<P, F, L>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GuidedLocalSearch")
            .field("penalization", &self.penalization)
            .field("feature_cost", &self.feature_cost)
            .field("lambda", &self.lambda)
            .field("current_penalty_sum", &self.current_penalty_sum)
            .field("trigger", &self.trigger)
            .field("trigger_counter", &self.trigger_counter)
            .finish()
    }
}

impl GuidedLocalSearch<MaxUtilityPenalization, UniformCost, FixedLambda> {
    /// Creates a new GLS with default `MaxUtilityPenalization`, `UniformCost`,
    /// and a fixed lambda.
    ///
    /// # Arguments
    ///
    /// * `lambda` — The penalty weight. See `heuristic_lambda` for auto-tuning.
    /// * `num_vessels` — Number of vessels in the problem.
    /// * `num_berths` — Number of berths in the problem.
    #[inline]
    pub fn new(lambda: f64, num_vessels: usize, num_berths: usize) -> Self {
        Self {
            penalization: MaxUtilityPenalization::default(),
            feature_cost: UniformCost,
            memory: PenaltyMemory::new(num_vessels, num_berths),
            lambda: FixedLambda::new(lambda),
            current_penalty_sum: 0,
            trigger: PenalizationTrigger::OnExhaustion,
            trigger_counter: 0,
        }
    }
}

impl<P, F, L> GuidedLocalSearch<P, F, L>
where
    P: PenalizationStrategy,
    F: FeatureCost,
    L: LambdaStrategy,
{
    /// Creates a new GLS with fully custom strategies.
    ///
    /// # Arguments
    ///
    /// * `penalization` — The penalization strategy.
    /// * `feature_cost` — The feature cost model.
    /// * `lambda` — The lambda strategy.
    /// * `num_vessels` — Number of vessels in the problem.
    /// * `num_berths` — Number of berths in the problem.
    #[inline]
    pub fn with_strategies(
        penalization: P,
        feature_cost: F,
        lambda: L,
        num_vessels: usize,
        num_berths: usize,
    ) -> Self {
        Self {
            penalization,
            feature_cost,
            memory: PenaltyMemory::new(num_vessels, num_berths),
            lambda,
            current_penalty_sum: 0,
            trigger: PenalizationTrigger::OnExhaustion,
            trigger_counter: 0,
        }
    }

    /// Replaces the penalization strategy.
    #[inline]
    pub fn with_penalization<P2: PenalizationStrategy>(
        self,
        penalization: P2,
    ) -> GuidedLocalSearch<P2, F, L> {
        GuidedLocalSearch {
            penalization,
            feature_cost: self.feature_cost,
            memory: self.memory,
            lambda: self.lambda,
            current_penalty_sum: self.current_penalty_sum,
            trigger: self.trigger,
            trigger_counter: self.trigger_counter,
        }
    }

    /// Replaces the feature cost model.
    #[inline]
    pub fn with_feature_cost<F2: FeatureCost>(
        self,
        feature_cost: F2,
    ) -> GuidedLocalSearch<P, F2, L> {
        GuidedLocalSearch {
            penalization: self.penalization,
            feature_cost,
            memory: self.memory,
            lambda: self.lambda,
            current_penalty_sum: self.current_penalty_sum,
            trigger: self.trigger,
            trigger_counter: self.trigger_counter,
        }
    }

    /// Replaces the lambda strategy.
    #[inline]
    pub fn with_lambda<L2: LambdaStrategy>(self, lambda: L2) -> GuidedLocalSearch<P, F, L2> {
        GuidedLocalSearch {
            penalization: self.penalization,
            feature_cost: self.feature_cost,
            memory: self.memory,
            lambda,
            current_penalty_sum: self.current_penalty_sum,
            trigger: self.trigger,
            trigger_counter: self.trigger_counter,
        }
    }

    /// Returns the current penalty weight $\lambda$.
    #[inline]
    pub fn lambda(&self) -> f64 {
        self.lambda.weight()
    }

    /// Returns a reference to the lambda strategy.
    #[inline]
    pub fn lambda_strategy(&self) -> &L {
        &self.lambda
    }

    /// Returns a mutable reference to the lambda strategy.
    #[inline]
    pub fn lambda_strategy_mut(&mut self) -> &mut L {
        &mut self.lambda
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
    fn compute_penalty_sum(&self, graph: &ScheduleGraph) -> i64 {
        let num_vessels = graph.num_vessels();
        let num_berths = graph.num_berths();
        let mut sum: i64 = 0;

        // ── Edge penalties ──
        for b in 0..num_berths {
            let berth = BerthIndex::new(b);
            let sentinel = graph.sentinel(berth);
            let mut prev_node = sentinel;
            loop {
                let next = graph.next_node(prev_node);
                let from = if prev_node.get() < num_vessels {
                    Some(VesselIndex::new(prev_node.get()))
                } else {
                    None
                };
                let to = if next.get() < num_vessels {
                    Some(VesselIndex::new(next.get()))
                } else {
                    None
                };
                if from.is_some() || to.is_some() {
                    let idx = edge_flat_index(from, to, num_vessels);
                    // SAFETY: idx < (num_vessels+1)^2, the edge allocation size.
                    sum += unsafe { *self.memory.edge.get_unchecked(idx) } as i64;
                }
                if next == sentinel {
                    break;
                }
                prev_node = next;
            }
        }

        // ── Berth-assignment penalties ──
        for v in 0..num_vessels {
            let vessel = VesselIndex::new(v);
            let berth = graph.vessel_berth(vessel);
            // SAFETY: v < num_vessels and berth < num_berths.
            sum += unsafe {
                *self
                    .memory
                    .berth
                    .get_unchecked(v * num_berths + berth.get())
            } as i64;
        }

        sum
    }

    /// Applies the penalization strategy, notifies the lambda strategy,
    /// and recomputes the cached penalty sum.
    fn penalize_and_recompute(&mut self, graph: &ScheduleGraph) {
        self.penalization
            .penalize(&mut self.memory, graph, &self.feature_cost);
        self.lambda.on_penalize();
        self.current_penalty_sum = self.compute_penalty_sum(graph);
    }
}

// ──────────────────────────────────────────────────────────────
// Metaheuristic Implementation
// ──────────────────────────────────────────────────────────────

impl<T, P, F, L> Metaheuristic<T> for GuidedLocalSearch<P, F, L>
where
    T: SolverNumeric + ToPrimitive,
    P: PenalizationStrategy,
    F: FeatureCost,
    L: LambdaStrategy,
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

        let w = self.lambda.weight();
        let aug_candidate = cand_f64 + w * candidate_penalty_sum as f64;
        let aug_current = curr_f64 + w * self.current_penalty_sum as f64;

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
        self.lambda.on_new_best();
        // Reset the non-improvement counter when a new best is found.
        if matches!(self.trigger, PenalizationTrigger::AfterNonImprovements(_)) {
            self.trigger_counter = 0;
        }
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
            self.trigger_counter += 1;
            if self.trigger_counter >= n {
                self.trigger_counter = 0;
                self.penalize_and_recompute(graph);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── PenaltyMemory ────────────────────────────────────────

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
        let a = Some(VesselIndex::new(0));
        let b = Some(VesselIndex::new(1));
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
        let sentinel = None;
        let v = Some(VesselIndex::new(2));
        let idx = edge_flat_index(sentinel, v, 3);
        // Should map to row=3, col=2 -> 3*4+2 = 14
        assert_eq!(idx, 14);
        assert_eq!(mem.edge_penalty(sentinel, v), 0);
    }

    // ── UniformCost ──────────────────────────────────────────

    #[test]
    fn test_uniform_cost_always_one() {
        let cost = UniformCost;
        assert_eq!(
            cost.edge_cost(Some(VesselIndex::new(0)), Some(VesselIndex::new(1))),
            1.0
        );
        assert_eq!(cost.edge_cost(None, Some(VesselIndex::new(0))), 1.0);
        assert_eq!(
            cost.berth_cost(VesselIndex::new(0), BerthIndex::new(0)),
            1.0
        );
    }

    // ── edge_flat_index ──────────────────────────────────────

    #[test]
    fn test_edge_flat_index_vessel_to_vessel() {
        let a = Some(VesselIndex::new(1));
        let b = Some(VesselIndex::new(2));
        // With num_vessels=4, stride = 5. Index = 1*5 + 2 = 7.
        assert_eq!(edge_flat_index(a, b, 4), 7);
    }

    #[test]
    fn test_edge_flat_index_sentinel_to_vessel() {
        let b = Some(VesselIndex::new(0));
        // Sentinel maps to 3. Index = 3 * 4 + 0 = 12.
        assert_eq!(edge_flat_index(None, b, 3), 12);
    }

    #[test]
    fn test_edge_flat_index_vessel_to_sentinel() {
        let a = Some(VesselIndex::new(0));
        // Sentinel maps to 3. Index = 0 * 4 + 3 = 3.
        assert_eq!(edge_flat_index(a, None, 3), 3);
    }

    // ── GuidedLocalSearch construction ───────────────────────

    #[test]
    fn test_gls_new() {
        let gls = GuidedLocalSearch::new(0.5, 10, 3);
        assert_eq!(gls.lambda(), 0.5);
        assert_eq!(gls.memory().num_vessels(), 10);
        assert_eq!(gls.memory().num_berths(), 3);
    }

    #[test]
    #[should_panic(expected = "value must be > 0.0")]
    fn test_gls_zero_lambda_panics() {
        let _ = GuidedLocalSearch::new(0.0, 5, 2);
    }

    #[test]
    #[should_panic(expected = "value must be > 0.0")]
    fn test_gls_negative_lambda_panics() {
        let _ = GuidedLocalSearch::new(-1.0, 5, 2);
    }

    #[test]
    fn test_gls_with_lambda() {
        let gls = GuidedLocalSearch::new(0.5, 5, 2).with_lambda(FixedLambda::new(1.0));
        assert_eq!(gls.lambda(), 1.0);
    }

    // ── heuristic_lambda ─────────────────────────────────────

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

    // ── PenaltyMemory::penalty_delta ─────────────────────────

    #[test]
    fn test_penalty_delta_empty_diff() {
        let mem = PenaltyMemory::new(3, 2);
        let diff = ScheduleGraphDiff::new(3);
        assert_eq!(mem.penalty_delta(&diff), 0);
    }

    // ── PenalizationTrigger ──────────────────────────────────

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
        let gls =
            GuidedLocalSearch::new(0.5, 5, 2).with_trigger(PenalizationTrigger::AfterMoves(20));
        assert_eq!(gls.trigger(), PenalizationTrigger::AfterMoves(20));
    }

    #[test]
    fn test_gls_default_trigger_is_on_exhaustion() {
        let gls = GuidedLocalSearch::new(0.5, 5, 2);
        assert_eq!(gls.trigger(), PenalizationTrigger::OnExhaustion);
    }

    // ── LambdaStrategy ───────────────────────────────────────

    #[test]
    fn test_fixed_lambda_weight() {
        let l = FixedLambda::new(3.0);
        assert_eq!(l.weight(), 3.0);
    }

    #[test]
    #[should_panic(expected = "value must be > 0.0")]
    fn test_fixed_lambda_zero_panics() {
        let _ = FixedLambda::new(0.0);
    }

    #[test]
    fn test_fixed_lambda_unchanged_after_hooks() {
        let mut l = FixedLambda::new(2.0);
        l.on_penalize();
        l.on_new_best();
        assert_eq!(l.weight(), 2.0);
    }

    #[test]
    fn test_reactive_lambda_grows_on_penalize() {
        let mut l = ReactiveLambda::new(1.0, 1.5, 0.5, 0.1, 100.0);
        assert_eq!(l.weight(), 1.0);
        l.on_penalize();
        assert!((l.weight() - 1.5).abs() < f64::EPSILON);
        l.on_penalize();
        assert!((l.weight() - 2.25).abs() < f64::EPSILON);
    }

    #[test]
    fn test_reactive_lambda_decays_on_new_best() {
        let mut l = ReactiveLambda::new(4.0, 1.5, 0.5, 0.1, 100.0);
        l.on_new_best();
        assert!((l.weight() - 2.0).abs() < f64::EPSILON);
        l.on_new_best();
        assert!((l.weight() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_reactive_lambda_clamps_to_max() {
        let mut l = ReactiveLambda::new(90.0, 1.5, 0.5, 0.1, 100.0);
        l.on_penalize(); // 90 * 1.5 = 135, clamped to 100
        assert!((l.weight() - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_reactive_lambda_clamps_to_min() {
        let mut l = ReactiveLambda::new(0.2, 1.5, 0.5, 0.1, 100.0);
        l.on_new_best(); // 0.2 * 0.5 = 0.1, exactly min
        assert!((l.weight() - 0.1).abs() < f64::EPSILON);
        l.on_new_best(); // 0.1 * 0.5 = 0.05, clamped to 0.1
        assert!((l.weight() - 0.1).abs() < f64::EPSILON);
    }

    #[test]
    #[should_panic(expected = "growth_factor must be > 1.0")]
    fn test_reactive_lambda_bad_growth_panics() {
        let _ = ReactiveLambda::new(1.0, 0.5, 0.8, 0.1, 10.0);
    }

    #[test]
    #[should_panic(expected = "decay_factor must be in (0.0, 1.0)")]
    fn test_reactive_lambda_bad_decay_panics() {
        let _ = ReactiveLambda::new(1.0, 1.5, 1.5, 0.1, 10.0);
    }

    #[test]
    fn test_gls_with_reactive_lambda() {
        let reactive = ReactiveLambda::new(5.0, 1.2, 0.8, 0.1, 50.0);
        let gls = GuidedLocalSearch::new(5.0, 10, 3).with_lambda(reactive);
        assert_eq!(gls.lambda(), 5.0);
    }
}
