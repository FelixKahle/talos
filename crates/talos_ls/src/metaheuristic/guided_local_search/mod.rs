// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Guided local search: a descent on the objective augmented by penalties of the features of the schedules it
//! stands in when it gets stuck.
//!
//! The features of a schedule are its links (a vessel next to a vessel or to the boundary of its berth, with the
//! boundaries of all berths taken as one) and its assignments (a vessel on a berth). The search accepts a move when
//! it lowers `objective + lambda * penalties`, the penalties of the features of the schedule summed. When the
//! trigger fires, it adds one penalty to every feature of the accepted schedule of the largest utility
//! `1 / (1 + penalty)`, which pushes the search out of the basin it stands in.

use super::acceptance::{AcceptanceOutcome, AcceptanceRule, NeighborhoodExhaustionOutcome};
use crate::schedule::{LinkEnd, ScheduleGraph, ScheduleGraphDiff};
use talos_core::SolverNumeric;
use talos_model::{BerthIndex, VesselIndex};

/// The settings of a guided local search: how the penalty weight evolves, how penalties decay, and when the
/// search penalizes.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct GuidedLocalSearch {
    /// The weight of the penalties in the augmented objective, and how it evolves.
    pub lambda: Lambda,
    /// How the penalties erode over the run.
    pub decay: PenaltyDecay,
    /// When the search penalizes the features of the accepted schedule.
    pub trigger: PenalizationTrigger,
}

/// The weight `lambda` of the penalties in the augmented objective, and how it evolves over a run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Lambda {
    /// A weight that never changes.
    Static(StaticLambda),
    /// A weight multiplied up at each penalization and down at each new best.
    Dynamic(DynamicLambda),
    /// A weight stepped up at each penalization and down at each new best.
    AdditiveDynamic(AdditiveDynamicLambda),
}

impl Default for Lambda {
    /// A static weight of one.
    fn default() -> Self {
        Self::Static(StaticLambda::new(1.0))
    }
}

/// A penalty weight that never changes: positive.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StaticLambda {
    value: f64,
}

impl StaticLambda {
    /// The weight `value`.
    ///
    /// # Panics
    ///
    /// Panics if `value` is not positive.
    pub fn new(value: f64) -> Self {
        assert!(
            value > 0.0,
            "called `StaticLambda::new` with {value}, which is not positive"
        );
        Self { value }
    }

    /// The weight.
    pub fn value(self) -> f64 {
        self.value
    }
}

/// A penalty weight that grows by a factor at each penalization and shrinks by a factor at each new best, kept
/// within its bounds.
///
/// The weight starts at `initial` clamped to `[min, max]`; a penalization multiplies it by `1 + increase`, and a
/// new best multiplies it by `1 - decrease` or, with `reset_on_best`, sets it back to its start.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DynamicLambda {
    initial: f64,
    increase: f64,
    decrease: f64,
    min: f64,
    max: f64,
    reset_on_best: bool,
}

impl DynamicLambda {
    /// The weight that starts at `initial` and is kept in `[min, max]`, grows by the factor `1 + increase` at each
    /// penalization and shrinks by the factor `1 - decrease` at each new best.
    ///
    /// # Panics
    ///
    /// Panics if `initial`, `increase` or `min` is not positive, if `decrease` is not in `(0, 1)`, or if `max` is
    /// below `min`.
    pub fn new(initial: f64, increase: f64, decrease: f64, min: f64, max: f64) -> Self {
        assert!(
            initial > 0.0 && increase > 0.0 && min > 0.0,
            "called `DynamicLambda::new` with initial {initial}, increase {increase} or min {min} not positive"
        );
        assert!(
            decrease > 0.0 && decrease < 1.0,
            "called `DynamicLambda::new` with decrease {decrease}, outside (0, 1)"
        );
        assert!(
            max >= min,
            "called `DynamicLambda::new` with max {max} below min {min}"
        );
        Self {
            initial,
            increase,
            decrease,
            min,
            max,
            reset_on_best: false,
        }
    }

    /// The same weight, which at a new best is set back to its start if `reset` holds, or shrinks otherwise.
    pub fn with_reset_on_best(mut self, reset: bool) -> Self {
        self.reset_on_best = reset;
        self
    }
}

/// A penalty weight that grows by a step at each penalization and shrinks by a step at each new best, kept within
/// its bounds.
///
/// The weight starts at `initial` clamped to `[min, max]`; a penalization adds `increase`, and a new best subtracts
/// `decrease` or, with `reset_on_best`, sets it back to its start.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdditiveDynamicLambda {
    initial: f64,
    increase: f64,
    decrease: f64,
    min: f64,
    max: f64,
    reset_on_best: bool,
}

impl AdditiveDynamicLambda {
    /// The weight that starts at `initial` and is kept in `[min, max]`, grows by `increase` at each penalization
    /// and shrinks by `decrease` at each new best.
    ///
    /// # Panics
    ///
    /// Panics if `initial`, `increase`, `decrease` or `min` is not positive, or if `max` is below `min`.
    pub fn new(initial: f64, increase: f64, decrease: f64, min: f64, max: f64) -> Self {
        assert!(
            initial > 0.0 && increase > 0.0 && decrease > 0.0 && min > 0.0,
            "called `AdditiveDynamicLambda::new` with initial {initial}, increase {increase}, decrease {decrease} \
             or min {min} not positive"
        );
        assert!(
            max >= min,
            "called `AdditiveDynamicLambda::new` with max {max} below min {min}"
        );
        Self {
            initial,
            increase,
            decrease,
            min,
            max,
            reset_on_best: false,
        }
    }

    /// The same weight, which at a new best is set back to its start if `reset` holds, or shrinks otherwise.
    pub fn with_reset_on_best(mut self, reset: bool) -> Self {
        self.reset_on_best = reset;
        self
    }
}

/// How the penalties erode over a run.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum PenaltyDecay {
    /// The penalties never erode.
    #[default]
    None,
    /// Every penalty is scaled down by a factor at every so many penalizations.
    Geometric(GeometricDecay),
}

/// A decay that multiplies every penalty by `factor`, rounding toward zero, at every `period`-th penalization.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeometricDecay {
    factor: f64,
    period: u64,
}

impl GeometricDecay {
    /// The decay by `factor` at every `period`-th penalization.
    ///
    /// # Panics
    ///
    /// Panics if `factor` is not in `(0, 1)` or `period` is zero.
    pub fn new(factor: f64, period: u64) -> Self {
        assert!(
            factor > 0.0 && factor < 1.0,
            "called `GeometricDecay::new` with factor {factor}, outside (0, 1)"
        );
        assert!(
            period >= 1,
            "called `GeometricDecay::new` with a period of zero"
        );
        Self { factor, period }
    }
}

/// When the search penalizes the features of the accepted schedule.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum PenalizationTrigger {
    /// When a scan finds no move that lowers the augmented objective: the classic guided local search.
    #[default]
    OnExhaustion,
    /// After this many steps in a row without a new best, each candidate and each exhausted scan a step.
    AfterNonImprovements(u64),
    /// After every this many accepted moves.
    AfterMoves(u64),
}

impl std::fmt::Display for PenalizationTrigger {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OnExhaustion => write!(f, "on exhaustion"),
            Self::AfterNonImprovements(steps) => write!(f, "after {steps} non-improving steps"),
            Self::AfterMoves(moves) => write!(f, "after {moves} accepted moves"),
        }
    }
}

/// A penalty weight from the objective of the initial schedule: `alpha * |objective| / features`, at least the
/// machine epsilon so that the penalties always count.
///
/// `features` is the number of features penalties may fall on, at least one; `alpha` is the share of the
/// objective, typically 0.1 to 0.5, that one penalty of every feature would add.
///
/// ```
/// # use talos_ls::heuristic_lambda;
/// let lambda = heuristic_lambda(1000.0, 50, 0.3);
/// assert!((lambda - 6.0).abs() < f64::EPSILON);
/// ```
#[expect(
    clippy::as_conversions,
    reason = "a count of features stays far below 2^53, where every count is an exact `f64`"
)]
pub fn heuristic_lambda(objective: f64, features: usize, alpha: f64) -> f64 {
    let features = features.max(1) as f64;
    (alpha * objective.abs() / features).max(f64::EPSILON)
}

/// A feature of a schedule, by its place in the tables of a penalty memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Feature {
    /// The link at this index of the table of links.
    Link(usize),
    /// The assignment at this index of the table of assignments.
    Assignment(usize),
}

/// Where a feature's utility stands against the best utility found so far.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UtilityRank {
    Better,
    Tie,
    Worse,
}

/// Where `utility` stands against `best`: a tie when they differ by at most a relative `1e-12` of the larger
/// magnitude, or of one where both are smaller.
///
/// The tolerance absorbs the rounding of `1 / (1 + penalty)` while keeping utilities of different penalties apart,
/// which differ by far more.
fn rank_utility(utility: f64, best: f64) -> UtilityRank {
    const RELATIVE_TOLERANCE: f64 = 1e-12;
    let difference = utility - best;
    let scale = utility.abs().max(best.abs()).max(1.0);
    if best == f64::NEG_INFINITY || difference > RELATIVE_TOLERANCE * scale {
        UtilityRank::Better
    } else if difference.abs() <= RELATIVE_TOLERANCE * scale {
        UtilityRank::Tie
    } else {
        UtilityRank::Worse
    }
}

/// The row or column of `end` in the table of links: a vessel's own index, and `num_vessels` for the boundary of
/// every berth, which all share one row and one column.
///
/// `end` is a boundary or a vessel below `num_vessels`.
#[inline]
fn end_index(end: LinkEnd, num_vessels: usize) -> usize {
    match end {
        LinkEnd::Vessel(vessel) => {
            debug_assert!(
                vessel.index() < num_vessels,
                "called `end_index` with {vessel}, outside a table of {num_vessels} vessels"
            );
            vessel.index()
        }
        LinkEnd::Berth(_) => num_vessels,
    }
}

/// The index of the link from `from` to `to` in a table of links of `num_vessels` vessels, below
/// `(num_vessels + 1)^2`.
///
/// `from` and `to` are boundaries or vessels below `num_vessels`.
#[inline]
fn link_index(from: LinkEnd, to: LinkEnd, num_vessels: usize) -> usize {
    end_index(from, num_vessels) * (num_vessels + 1) + end_index(to, num_vessels)
}

/// Whether the link from `from` to `to` is a feature: one of its ends is a vessel. The link from the head to the
/// tail of an empty berth is none.
#[inline]
fn is_feature(from: LinkEnd, to: LinkEnd) -> bool {
    matches!(from, LinkEnd::Vessel(_)) || matches!(to, LinkEnd::Vessel(_))
}

/// The penalty count of every feature of the schedules of one instance, zero at the start of a run.
///
/// The table of links has `(num_vessels + 1)^2` entries, the row and column `num_vessels` standing for the
/// boundaries of all berths; the table of assignments has `num_vessels * num_berths`, by vessel and then berth.
#[derive(Debug, Clone, Default)]
struct PenaltyMemory {
    links: Vec<u64>,
    assignments: Vec<u64>,
    num_vessels: usize,
    num_berths: usize,
}

impl PenaltyMemory {
    /// Sizes the memory for `num_vessels` vessels and `num_berths` berths, with every count zero.
    fn reset(&mut self, num_vessels: usize, num_berths: usize) {
        self.links.clear();
        self.links.resize((num_vessels + 1) * (num_vessels + 1), 0);
        self.assignments.clear();
        self.assignments.resize(num_vessels * num_berths, 0);
        self.num_vessels = num_vessels;
        self.num_berths = num_berths;
    }

    /// The index of the assignment of `vessel` to `berth`.
    ///
    /// `vessel` and `berth` are the memory's.
    #[inline]
    fn assignment_index(&self, vessel: VesselIndex, berth: BerthIndex) -> usize {
        debug_assert!(
            vessel.index() < self.num_vessels && berth.index() < self.num_berths,
            "called `PenaltyMemory::assignment_index` with {vessel} or {berth}, outside the memory"
        );
        vessel.index() * self.num_berths + berth.index()
    }

    /// The count of the feature at `feature`.
    ///
    /// # Safety
    ///
    /// The index of `feature` must be below the length of its table.
    #[inline]
    unsafe fn count_unchecked(&self, feature: Feature) -> u64 {
        match feature {
            Feature::Link(index) => {
                debug_assert!(index < self.links.len());
                // SAFETY: the index is below the length of the table of links, the precondition of this function.
                unsafe { *self.links.get_unchecked(index) }
            }
            Feature::Assignment(index) => {
                debug_assert!(index < self.assignments.len());
                // SAFETY: the index is below the length of the table of assignments, the precondition of this
                // function.
                unsafe { *self.assignments.get_unchecked(index) }
            }
        }
    }

    /// The count of the link from `from` to `to`.
    ///
    /// # Panics
    ///
    /// Panics if `from` or `to` is a vessel outside the memory.
    #[cfg(test)]
    fn link_penalty(&self, from: LinkEnd, to: LinkEnd) -> u64 {
        let inside = |end: LinkEnd| match end {
            LinkEnd::Vessel(vessel) => vessel.index() < self.num_vessels,
            LinkEnd::Berth(_) => true,
        };
        assert!(
            inside(from) && inside(to),
            "called `PenaltyMemory::link_penalty` with a link outside a memory of {} vessels",
            self.num_vessels
        );
        self.links[link_index(from, to, self.num_vessels)]
    }

    /// The count of the assignment of `vessel` to `berth`.
    ///
    /// # Panics
    ///
    /// Panics if `vessel` or `berth` is outside the memory.
    #[cfg(test)]
    fn assignment_penalty(&self, vessel: VesselIndex, berth: BerthIndex) -> u64 {
        assert!(
            vessel.index() < self.num_vessels && berth.index() < self.num_berths,
            "called `PenaltyMemory::assignment_penalty` with {vessel} and {berth}, outside a memory of {} vessels \
             and {} berths",
            self.num_vessels,
            self.num_berths
        );
        self.assignments[self.assignment_index(vessel, berth)]
    }

    /// The change in the sum of penalties that the moves of `diff` make: the counts of the links they made and
    /// the assignments they reached, less those of the links they broke and the assignments they left.
    ///
    /// It costs one read per entry of the diff.
    ///
    /// # Safety
    ///
    /// Every vessel and berth `diff` names must be the memory's.
    #[expect(
        clippy::as_conversions,
        reason = "a penalty count grows by one per penalization, far below 2^63, so it is an exact `i64`"
    )]
    unsafe fn penalty_delta_unchecked(&self, diff: &ScheduleGraphDiff) -> i64 {
        let mut delta: i64 = 0;
        for link in diff.broken_links() {
            let index = link_index(link.from, link.to, self.num_vessels);
            // SAFETY: both ends are boundaries or vessels of the memory, the precondition of this function, so
            // the index is below `(num_vessels + 1)^2`, the length of the table of links.
            delta -= unsafe { self.count_unchecked(Feature::Link(index)) } as i64;
        }
        for link in diff.created_links() {
            let index = link_index(link.from, link.to, self.num_vessels);
            // SAFETY: as for the broken links.
            delta += unsafe { self.count_unchecked(Feature::Link(index)) } as i64;
        }
        for reallocation in diff.reallocations() {
            let left = self.assignment_index(reallocation.vessel, reallocation.from);
            let reached = self.assignment_index(reallocation.vessel, reallocation.to);
            // SAFETY: the vessel and both berths are the memory's, the precondition of this function, so both
            // indices are below `num_vessels * num_berths`, the length of the table of assignments.
            let (left, reached) = unsafe {
                (
                    self.count_unchecked(Feature::Assignment(left)),
                    self.count_unchecked(Feature::Assignment(reached)),
                )
            };
            delta -= left as i64;
            delta += reached as i64;
        }
        delta
    }

    /// The sum of the penalties of every feature of `graph`.
    ///
    /// # Panics
    ///
    /// Panics if `graph` does not count the vessels and berths of the memory.
    #[expect(
        clippy::as_conversions,
        reason = "a penalty count grows by one per penalization, far below 2^63, so it is an exact `i64`"
    )]
    fn penalty_sum(&self, graph: &ScheduleGraph) -> i64 {
        self.assert_counts_of(graph);
        let mut sum: i64 = 0;
        for berth in graph.berths() {
            let ends = graph.berth_topology_iter(berth);
            for (from, to) in ends.clone().zip(ends.skip(1)) {
                if is_feature(from, to) {
                    let index = link_index(from, to, self.num_vessels);
                    // SAFETY: both ends are boundaries or vessels of the graph, which counts the memory's
                    // vessels, as checked above.
                    sum += unsafe { self.count_unchecked(Feature::Link(index)) } as i64;
                }
            }
        }
        for vessel in graph.vessels() {
            let index = self.assignment_index(vessel, graph.vessel_berth(vessel));
            // SAFETY: the vessel and its berth are the graph's, which counts the memory's vessels and berths, as
            // checked above.
            sum += unsafe { self.count_unchecked(Feature::Assignment(index)) } as i64;
        }
        sum
    }

    /// Adds one penalty to every feature of `graph` of the largest utility `1 / (1 + penalty)`, collecting them
    /// in `winners`.
    ///
    /// # Panics
    ///
    /// Panics if `graph` does not count the vessels and berths of the memory.
    #[expect(
        clippy::as_conversions,
        reason = "a penalty count grows by one per penalization, far below 2^53, so it is an exact `f64`"
    )]
    fn penalize(&mut self, graph: &ScheduleGraph, winners: &mut Vec<Feature>) {
        self.assert_counts_of(graph);
        winners.clear();
        let mut best_utility = f64::NEG_INFINITY;
        let mut consider = |feature: Feature, penalty: u64| {
            let utility = 1.0 / (1.0 + penalty as f64);
            match rank_utility(utility, best_utility) {
                UtilityRank::Better => {
                    best_utility = utility;
                    winners.clear();
                    winners.push(feature);
                }
                UtilityRank::Tie => winners.push(feature),
                UtilityRank::Worse => {}
            }
        };
        for berth in graph.berths() {
            let ends = graph.berth_topology_iter(berth);
            for (from, to) in ends.clone().zip(ends.skip(1)) {
                if is_feature(from, to) {
                    let feature = Feature::Link(link_index(from, to, self.num_vessels));
                    // SAFETY: both ends are boundaries or vessels of the graph, which counts the memory's
                    // vessels, as checked above.
                    consider(feature, unsafe { self.count_unchecked(feature) });
                }
            }
            for vessel in graph.vessel_sequence_iter(berth) {
                let feature = Feature::Assignment(self.assignment_index(vessel, berth));
                // SAFETY: the vessel and the berth are the graph's, which counts the memory's vessels and berths,
                // as checked above.
                consider(feature, unsafe { self.count_unchecked(feature) });
            }
        }
        for &feature in winners.iter() {
            let count = match feature {
                Feature::Link(index) => &mut self.links[index],
                Feature::Assignment(index) => &mut self.assignments[index],
            };
            *count = count.saturating_add(1);
        }
    }

    /// Multiplies every count by `factor`, rounding toward zero.
    ///
    /// `factor` is in `[0, 1]`.
    #[expect(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        reason = "the product of a count and a factor in [0, 1] lies in [0, count], and rounding it toward zero is \
                  the decay"
    )]
    fn decay(&mut self, factor: f64) {
        debug_assert!(
            (0.0..=1.0).contains(&factor),
            "called `PenaltyMemory::decay` with factor {factor}, outside [0, 1]"
        );
        for count in self.links.iter_mut().chain(self.assignments.iter_mut()) {
            *count = (*count as f64 * factor) as u64;
        }
    }

    /// Panics unless `graph` counts the vessels and berths of the memory, which every unchecked read through the
    /// features of `graph` rests on.
    fn assert_counts_of(&self, graph: &ScheduleGraph) {
        assert!(
            graph.num_vessels() == self.num_vessels && graph.num_berths() == self.num_berths,
            "the graph of {} vessels and {} berths should have the counts of the penalty memory, {} and {}",
            graph.num_vessels(),
            graph.num_berths(),
            self.num_vessels,
            self.num_berths
        );
    }
}

/// The penalty weight of a run as it evolves under its [`Lambda`].
#[derive(Debug, Clone, Copy)]
struct LambdaSchedule {
    rule: Lambda,
    current: f64,
}

impl LambdaSchedule {
    /// The schedule of `rule`, at its start.
    fn new(rule: Lambda) -> Self {
        let mut schedule = Self { rule, current: 0.0 };
        schedule.restart();
        schedule
    }

    /// Sets the weight to its start.
    fn restart(&mut self) {
        self.current = match self.rule {
            Lambda::Static(lambda) => lambda.value,
            Lambda::Dynamic(lambda) => lambda.initial.clamp(lambda.min, lambda.max),
            Lambda::AdditiveDynamic(lambda) => lambda.initial.clamp(lambda.min, lambda.max),
        };
    }

    /// Lowers the weight at a new best, or sets it to its start where the rule says so.
    fn on_new_best(&mut self) {
        match self.rule {
            Lambda::Static(_) => {}
            Lambda::Dynamic(lambda) if lambda.reset_on_best => self.restart(),
            Lambda::Dynamic(lambda) => {
                self.current =
                    (self.current * (1.0 - lambda.decrease)).clamp(lambda.min, lambda.max);
            }
            Lambda::AdditiveDynamic(lambda) if lambda.reset_on_best => self.restart(),
            Lambda::AdditiveDynamic(lambda) => {
                self.current = (self.current - lambda.decrease).clamp(lambda.min, lambda.max);
            }
        }
    }

    /// Raises the weight after a penalization.
    fn on_penalization(&mut self) {
        match self.rule {
            Lambda::Static(_) => {}
            Lambda::Dynamic(lambda) => {
                self.current =
                    (self.current * (1.0 + lambda.increase)).clamp(lambda.min, lambda.max);
            }
            Lambda::AdditiveDynamic(lambda) => {
                self.current = (self.current + lambda.increase).clamp(lambda.min, lambda.max);
            }
        }
    }
}

/// The decay of the penalties of a run, with the penalizations counted since it last fired.
#[derive(Debug, Clone, Copy)]
struct DecaySchedule {
    rule: PenaltyDecay,
    penalizations: u64,
}

impl DecaySchedule {
    /// Counts a penalization, and decays `memory` at every period of the rule.
    fn after_penalization(&mut self, memory: &mut PenaltyMemory) {
        if let PenaltyDecay::Geometric(decay) = self.rule {
            self.penalizations += 1;
            if self.penalizations >= decay.period {
                self.penalizations = 0;
                memory.decay(decay.factor);
            }
        }
    }
}

/// A guided local search under way: its settings, its penalty memory, and the sum of the penalties of the
/// accepted schedule.
///
/// The penalty memory counts the vessels and berths of the graph the run searches from its start on, and the
/// penalty sum is that of the accepted schedule after every step.
#[derive(Debug, Clone)]
pub struct GuidedLocalSearchRule {
    /// The penalty of every feature.
    memory: PenaltyMemory,
    /// The features the last penalization raised, kept for its memory.
    winners: Vec<Feature>,
    /// The weight of the penalties.
    lambda: LambdaSchedule,
    /// The decay of the penalties.
    decay: DecaySchedule,
    /// When the search penalizes.
    trigger: PenalizationTrigger,
    /// The steps or accepted moves counted toward the trigger.
    trigger_counter: u64,
    /// The best objective the trigger has seen, for [`PenalizationTrigger::AfterNonImprovements`].
    last_best_objective: f64,
    penalty_sum: i64,
}

impl GuidedLocalSearchRule {
    /// The search of `settings`, which starts with the run.
    pub fn new(settings: &GuidedLocalSearch) -> Self {
        Self {
            memory: PenaltyMemory::default(),
            winners: Vec::new(),
            lambda: LambdaSchedule::new(settings.lambda),
            decay: DecaySchedule {
                rule: settings.decay,
                penalizations: 0,
            },
            trigger: settings.trigger,
            trigger_counter: 0,
            last_best_objective: f64::INFINITY,
            penalty_sum: 0,
        }
    }

    /// Penalizes the features of `graph`, the accepted schedule, of the largest utility, decays the penalties as
    /// the rule says, and raises the weight.
    fn penalize(&mut self, graph: &ScheduleGraph) {
        self.memory.penalize(graph, &mut self.winners);
        self.decay.after_penalization(&mut self.memory);
        self.penalty_sum = self.memory.penalty_sum(graph);
        self.lambda.on_penalization();
    }
}

impl<T> AcceptanceRule<T> for GuidedLocalSearchRule
where
    T: SolverNumeric,
{
    fn on_start(&mut self, graph: &ScheduleGraph, initial_objective: T) {
        self.memory.reset(graph.num_vessels(), graph.num_berths());
        self.penalty_sum = self.memory.penalty_sum(graph);
        self.trigger_counter = 0;
        self.decay.penalizations = 0;
        self.last_best_objective = initial_objective.to_f64().unwrap_or(0.0);
        self.lambda.restart();
    }

    #[expect(
        clippy::as_conversions,
        reason = "a sum of penalty counts stays far below 2^53, so it is an exact `f64`"
    )]
    unsafe fn decide_unchecked(
        &mut self,
        accepted_objective: T,
        _buffered_objective: Option<T>,
        candidate_objective: T,
        diff: &ScheduleGraphDiff,
    ) -> AcceptanceOutcome {
        let (Some(candidate), Some(accepted)) =
            (candidate_objective.to_f64(), accepted_objective.to_f64())
        else {
            return AcceptanceOutcome::Reject;
        };
        if !candidate.is_finite() || !accepted.is_finite() {
            return AcceptanceOutcome::Reject;
        }
        // SAFETY: the diff names only vessels and berths of the graph the run searches, which the memory counts
        // from the start of the run on, the precondition of this function.
        let candidate_penalty_sum =
            self.penalty_sum + unsafe { self.memory.penalty_delta_unchecked(diff) };
        let lambda = self.lambda.current;
        let augmented_candidate = candidate + lambda * candidate_penalty_sum as f64;
        let augmented_accepted = accepted + lambda * self.penalty_sum as f64;
        if augmented_candidate < augmented_accepted {
            AcceptanceOutcome::Accept
        } else {
            AcceptanceOutcome::Reject
        }
    }

    unsafe fn on_accept_unchecked(
        &mut self,
        _accepted_objective: T,
        graph: &ScheduleGraph,
        diff: &ScheduleGraphDiff,
    ) {
        // SAFETY: the diff names only vessels and berths of the graph the run searches, which the memory counts
        // from the start of the run on, the precondition of this function.
        self.penalty_sum += unsafe { self.memory.penalty_delta_unchecked(diff) };
        if let PenalizationTrigger::AfterMoves(moves) = self.trigger {
            self.trigger_counter += 1;
            if self.trigger_counter >= moves {
                self.trigger_counter = 0;
                self.penalize(graph);
            }
        }
    }

    fn on_new_best(&mut self, _best_objective: T) {
        self.lambda.on_new_best();
    }

    fn after_step(&mut self, best_objective: T, graph: &ScheduleGraph) {
        if let PenalizationTrigger::AfterNonImprovements(steps) = self.trigger {
            let best = best_objective.to_f64().unwrap_or(0.0);
            if best < self.last_best_objective {
                self.trigger_counter = 0;
                self.last_best_objective = best;
            } else {
                self.trigger_counter += 1;
                if self.trigger_counter >= steps {
                    self.trigger_counter = 0;
                    self.penalize(graph);
                }
            }
        }
    }

    fn commits_buffered(&mut self) -> bool {
        false
    }

    fn on_neighborhood_exhausted(
        &mut self,
        graph: &ScheduleGraph,
    ) -> NeighborhoodExhaustionOutcome {
        if self.trigger == PenalizationTrigger::OnExhaustion {
            self.penalize(graph);
        }
        NeighborhoodExhaustionOutcome::Restart
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schedule::Mutator;

    fn vessel(index: usize) -> VesselIndex {
        VesselIndex::new(index)
    }

    /// The memory of `num_vessels` vessels and `num_berths` berths, every count zero.
    fn memory(num_vessels: usize, num_berths: usize) -> PenaltyMemory {
        let mut memory = PenaltyMemory::default();
        memory.reset(num_vessels, num_berths);
        memory
    }

    fn berth(index: usize) -> BerthIndex {
        BerthIndex::new(index)
    }

    /// V0 and V1 on B0 in that order, V2 on B1.
    fn three_vessels_on_two_berths() -> ScheduleGraph {
        ScheduleGraph::from_slices(&[berth(0), berth(0), berth(1)], &[0_i64, 1, 0], 2)
    }

    fn started(settings: GuidedLocalSearch, graph: &ScheduleGraph) -> GuidedLocalSearchRule {
        let mut rule = GuidedLocalSearchRule::new(&settings);
        AcceptanceRule::<i64>::on_start(&mut rule, graph, 100);
        rule
    }

    #[test]
    fn test_rank_utility_ties_within_the_relative_tolerance_and_ranks_the_rest() {
        assert_eq!(rank_utility(0.5, f64::NEG_INFINITY), UtilityRank::Better);
        assert_eq!(rank_utility(1e8 + 1e-6, 1e8), UtilityRank::Tie);
        assert_eq!(rank_utility(0.5, 0.5 + 1e-13), UtilityRank::Tie);
        assert_eq!(rank_utility(1.0, 0.5), UtilityRank::Better);
        assert_eq!(rank_utility(0.5, 1.0), UtilityRank::Worse);
    }

    #[test]
    fn test_link_index_gives_every_boundary_one_row_and_column() {
        let boundary = |index| LinkEnd::Berth(berth(index));
        let first = LinkEnd::Vessel(vessel(1));
        assert_eq!(link_index(first, LinkEnd::Vessel(vessel(2)), 4), 7);
        assert_eq!(
            link_index(boundary(0), first, 4),
            link_index(boundary(3), first, 4)
        );
        assert_eq!(link_index(first, boundary(0), 4), 9);
        assert_eq!(link_index(boundary(0), boundary(1), 4), 24);
    }

    #[test]
    fn test_link_index_stays_inside_the_table() {
        let num_vessels = 5;
        let ends: Vec<LinkEnd> = (0..num_vessels)
            .map(|v| LinkEnd::Vessel(vessel(v)))
            .chain([LinkEnd::Berth(berth(7))])
            .collect();
        for &from in &ends {
            for &to in &ends {
                assert!(link_index(from, to, num_vessels) < (num_vessels + 1) * (num_vessels + 1));
            }
        }
    }

    #[test]
    fn test_penalty_memory_starts_at_zero_and_resets_to_new_counts() {
        let mut memory = memory(3, 2);
        assert_eq!(memory.links.len(), 16);
        assert_eq!(memory.assignments.len(), 6);
        memory.links[5] = 3;
        memory.reset(4, 1);
        assert_eq!(memory.links, vec![0; 25]);
        assert_eq!(memory.assignments, vec![0; 4]);
    }

    #[test]
    fn test_penalty_memory_reads_the_counts_of_links_and_assignments() {
        let mut memory = memory(3, 2);
        let (from, to) = (LinkEnd::Vessel(vessel(0)), LinkEnd::Vessel(vessel(2)));
        memory.links[link_index(from, to, 3)] = 7;
        let index = memory.assignment_index(vessel(2), berth(1));
        memory.assignments[index] = 4;
        assert_eq!(memory.link_penalty(from, to), 7);
        assert_eq!(memory.link_penalty(to, from), 0);
        assert_eq!(memory.assignment_penalty(vessel(2), berth(1)), 4);
        assert_eq!(memory.assignment_penalty(vessel(1), berth(1)), 0);
    }

    // A berth beyond the memory must be refused, not read as another vessel's row.
    #[test]
    #[should_panic(
        expected = "called `PenaltyMemory::assignment_penalty` with Vessel 0 and Berth 3, outside a memory of 2 \
                    vessels and 2 berths"
    )]
    fn test_penalty_memory_assignment_penalty_refuses_a_berth_outside() {
        let _ = memory(2, 2).assignment_penalty(vessel(0), berth(3));
    }

    // A vessel beyond the memory must be refused, not read as the boundary of a berth.
    #[test]
    #[should_panic(
        expected = "called `PenaltyMemory::link_penalty` with a link outside a memory of 2 vessels"
    )]
    fn test_penalty_memory_link_penalty_refuses_a_vessel_outside() {
        let _ = memory(2, 2).link_penalty(LinkEnd::Vessel(vessel(2)), LinkEnd::Vessel(vessel(0)));
    }

    #[test]
    #[should_panic(
        expected = "the graph of 3 vessels and 2 berths should have the counts of the penalty memory"
    )]
    fn test_penalty_sum_refuses_a_graph_of_other_counts() {
        let _ = memory(2, 2).penalty_sum(&three_vessels_on_two_berths());
    }

    #[test]
    fn test_penalty_delta_counts_made_and_reached_less_broken_and_left() {
        let graph = three_vessels_on_two_berths();
        let mut memory = memory(3, 2);
        for count in memory.links.iter_mut() {
            *count = 1;
        }
        let index = memory.assignment_index(vessel(1), berth(0));
        memory.assignments[index] = 5;
        let index = memory.assignment_index(vessel(1), berth(1));
        memory.assignments[index] = 2;
        let mut mutator = Mutator::new(graph);
        // V1 moves from after V0 on B0 to after V2 on B1: it breaks V0-V1 and V1-tail and V2-tail, and makes
        // V0-tail, V2-V1 and V1-tail: as many links, each of count one.
        mutator.relocate_after(vessel(1), vessel(2));
        // SAFETY: the diff names vessels and berths of a graph of the memory's counts.
        let delta = unsafe { memory.penalty_delta_unchecked(mutator.diff()) };
        assert_eq!(delta, 2 - 5);
    }

    #[test]
    fn test_penalty_sum_counts_every_link_with_a_vessel_and_every_assignment() {
        let graph = three_vessels_on_two_berths();
        let mut memory = memory(3, 2);
        for count in memory.links.iter_mut().chain(memory.assignments.iter_mut()) {
            *count = 1;
        }
        // B0: head-V0, V0-V1, V1-tail; B1: head-V2, V2-tail; and three assignments.
        assert_eq!(memory.penalty_sum(&graph), 5 + 3);
    }

    #[test]
    fn test_penalize_raises_every_feature_of_the_largest_utility() {
        let graph = three_vessels_on_two_berths();
        let mut memory = memory(3, 2);
        let mut winners = Vec::new();
        memory.penalize(&graph, &mut winners);
        // Every feature is unpenalized: all eight tie and all are raised.
        assert_eq!(winners.len(), 8);
        assert_eq!(memory.penalty_sum(&graph), 8);
        // Now the assignment of V2 alone is lower than the rest.
        let index = memory.assignment_index(vessel(2), berth(1));
        memory.assignments[index] = 0;
        memory.penalize(&graph, &mut winners);
        assert_eq!(
            winners,
            [Feature::Assignment(
                memory.assignment_index(vessel(2), berth(1))
            )]
        );
        assert_eq!(memory.assignment_penalty(vessel(2), berth(1)), 1);
    }

    #[test]
    fn test_decay_rounds_every_count_toward_zero() {
        let mut memory = memory(1, 1);
        memory.links[0] = 10;
        memory.links[1] = 3;
        memory.assignments[0] = 1;
        memory.decay(0.5);
        assert_eq!(memory.links[..2], [5, 1]);
        assert_eq!(memory.assignments[0], 0);
    }

    #[test]
    fn test_decay_by_one_keeps_a_count_below_two_to_the_fifty_third() {
        let mut memory = memory(1, 1);
        memory.links[0] = (1_u64 << 53) - 1;
        memory.decay(1.0);
        assert_eq!(memory.links[0], (1_u64 << 53) - 1);
    }

    #[test]
    fn test_heuristic_lambda_shares_the_objective_among_the_features() {
        assert!((heuristic_lambda(1000.0, 50, 0.3) - 6.0).abs() < 1e-12);
        assert!((heuristic_lambda(-1000.0, 0, 0.3) - 300.0).abs() < 1e-12);
        assert_eq!(heuristic_lambda(0.0, 10, 0.3), f64::EPSILON);
    }

    #[test]
    #[should_panic(expected = "called `StaticLambda::new` with 0, which is not positive")]
    fn test_static_lambda_refuses_zero() {
        let _ = StaticLambda::new(0.0);
    }

    #[test]
    #[should_panic(expected = "called `DynamicLambda::new` with decrease 1, outside (0, 1)")]
    fn test_dynamic_lambda_refuses_a_decrease_of_one() {
        let _ = DynamicLambda::new(1.0, 0.1, 1.0, 0.5, 2.0);
    }

    #[test]
    #[should_panic(expected = "called `AdditiveDynamicLambda::new` with max 1 below min 2")]
    fn test_additive_dynamic_lambda_refuses_max_below_min() {
        let _ = AdditiveDynamicLambda::new(1.0, 0.1, 0.1, 2.0, 1.0);
    }

    #[test]
    #[should_panic(expected = "called `GeometricDecay::new` with factor 1, outside (0, 1)")]
    fn test_geometric_decay_refuses_a_factor_of_one() {
        let _ = GeometricDecay::new(1.0, 1);
    }

    #[test]
    fn test_static_lambda_never_changes() {
        let mut schedule = LambdaSchedule::new(Lambda::Static(StaticLambda::new(2.5)));
        schedule.on_penalization();
        schedule.on_new_best();
        assert_eq!(schedule.current, 2.5);
    }

    #[test]
    fn test_dynamic_lambda_multiplies_within_its_bounds() {
        let mut schedule =
            LambdaSchedule::new(Lambda::Dynamic(DynamicLambda::new(1.0, 0.5, 0.5, 0.4, 2.0)));
        schedule.on_penalization();
        assert_eq!(schedule.current, 1.5);
        schedule.on_penalization();
        assert_eq!(schedule.current, 2.0);
        schedule.on_new_best();
        assert_eq!(schedule.current, 1.0);
        schedule.on_new_best();
        schedule.on_new_best();
        assert_eq!(schedule.current, 0.4);
        schedule.restart();
        assert_eq!(schedule.current, 1.0);
    }

    #[test]
    fn test_dynamic_lambda_starts_clamped_and_resets_on_best_where_asked() {
        let lambda = DynamicLambda::new(5.0, 0.5, 0.5, 0.4, 2.0).with_reset_on_best(true);
        let mut schedule = LambdaSchedule::new(Lambda::Dynamic(lambda));
        assert_eq!(schedule.current, 2.0);
        schedule.current = 0.5;
        schedule.on_new_best();
        assert_eq!(schedule.current, 2.0);
    }

    #[test]
    fn test_additive_dynamic_lambda_steps_within_its_bounds() {
        let mut schedule = LambdaSchedule::new(Lambda::AdditiveDynamic(
            AdditiveDynamicLambda::new(1.0, 0.25, 0.5, 0.5, 1.5),
        ));
        schedule.on_penalization();
        assert_eq!(schedule.current, 1.25);
        schedule.on_penalization();
        schedule.on_penalization();
        assert_eq!(schedule.current, 1.5);
        schedule.on_new_best();
        schedule.on_new_best();
        schedule.on_new_best();
        assert_eq!(schedule.current, 0.5);
    }

    #[test]
    fn test_geometric_decay_fires_at_every_period() {
        let mut memory = memory(1, 1);
        memory.links[0] = 100;
        let mut decay = DecaySchedule {
            rule: PenaltyDecay::Geometric(GeometricDecay::new(0.5, 3)),
            penalizations: 0,
        };
        decay.after_penalization(&mut memory);
        decay.after_penalization(&mut memory);
        assert_eq!(memory.links[0], 100);
        decay.after_penalization(&mut memory);
        assert_eq!(memory.links[0], 50);
        decay.after_penalization(&mut memory);
        decay.after_penalization(&mut memory);
        decay.after_penalization(&mut memory);
        assert_eq!(memory.links[0], 25);
    }

    #[test]
    fn test_no_decay_keeps_the_counts() {
        let mut memory = memory(1, 1);
        memory.links[0] = 100;
        let mut decay = DecaySchedule {
            rule: PenaltyDecay::None,
            penalizations: 0,
        };
        decay.after_penalization(&mut memory);
        assert_eq!(memory.links[0], 100);
    }

    // A search of three vessels on two berths, set up for nothing smaller: its memory must count the graph it
    // searches, or a move of the last vessel reads past the tables.
    #[test]
    fn test_start_sizes_the_memory_for_the_graph_searched() {
        let graph = three_vessels_on_two_berths();
        let rule = started(GuidedLocalSearch::default(), &graph);
        assert_eq!((rule.memory.num_vessels, rule.memory.num_berths), (3, 2));
        assert_eq!(rule.penalty_sum, 0);
    }

    #[test]
    fn test_decide_accepts_only_a_lower_augmented_objective() {
        let graph = three_vessels_on_two_berths();
        let settings = GuidedLocalSearch {
            lambda: Lambda::Static(StaticLambda::new(10.0)),
            ..GuidedLocalSearch::default()
        };
        let mut rule = started(settings, &graph);
        let mut mutator = Mutator::new(graph);
        mutator.relocate_after(vessel(1), vessel(2));
        let decide =
            |rule: &mut GuidedLocalSearchRule, candidate: i64, diff: &ScheduleGraphDiff| {
                // SAFETY: the diff names vessels and berths of the graph the rule started on.
                unsafe { AcceptanceRule::<i64>::decide_unchecked(rule, 100, None, candidate, diff) }
            };
        assert_eq!(
            decide(&mut rule, 99, mutator.diff()),
            AcceptanceOutcome::Accept
        );
        assert_eq!(
            decide(&mut rule, 100, mutator.diff()),
            AcceptanceOutcome::Reject
        );
        // One penalty on the assignment V1 reaches weighs 10: the move must now gain more than 10.
        let index = rule.memory.assignment_index(vessel(1), berth(1));
        rule.memory.assignments[index] = 1;
        assert_eq!(
            decide(&mut rule, 91, mutator.diff()),
            AcceptanceOutcome::Reject
        );
        assert_eq!(
            decide(&mut rule, 89, mutator.diff()),
            AcceptanceOutcome::Accept
        );
    }

    #[test]
    fn test_on_exhaustion_trigger_penalizes_and_restarts() {
        let graph = three_vessels_on_two_berths();
        let mut rule = started(GuidedLocalSearch::default(), &graph);
        let outcome = AcceptanceRule::<i64>::on_neighborhood_exhausted(&mut rule, &graph);
        assert_eq!(outcome, NeighborhoodExhaustionOutcome::Restart);
        assert_eq!(rule.penalty_sum, 8);
    }

    #[test]
    fn test_non_improvement_trigger_fires_after_steps_without_a_new_best() {
        let graph = three_vessels_on_two_berths();
        let settings = GuidedLocalSearch {
            trigger: PenalizationTrigger::AfterNonImprovements(3),
            ..GuidedLocalSearch::default()
        };
        let mut rule = started(settings, &graph);
        let step = |rule: &mut GuidedLocalSearchRule, best: i64| {
            AcceptanceRule::<i64>::after_step(rule, best, &graph)
        };
        step(&mut rule, 100);
        step(&mut rule, 100);
        // A new best resets the count.
        step(&mut rule, 90);
        step(&mut rule, 90);
        step(&mut rule, 90);
        assert_eq!(rule.penalty_sum, 0);
        step(&mut rule, 90);
        assert_eq!(rule.penalty_sum, 8);
        assert_eq!(rule.trigger_counter, 0);
        // Exhaustion does not penalize under this trigger.
        let _ = AcceptanceRule::<i64>::on_neighborhood_exhausted(&mut rule, &graph);
        assert_eq!(rule.penalty_sum, 8);
    }

    #[test]
    fn test_moves_trigger_fires_at_every_so_many_accepted_moves() {
        let graph = three_vessels_on_two_berths();
        let settings = GuidedLocalSearch {
            trigger: PenalizationTrigger::AfterMoves(2),
            ..GuidedLocalSearch::default()
        };
        let mut rule = started(settings, &graph);
        let empty = ScheduleGraphDiff::new();
        // SAFETY: an empty diff names no vessel or berth.
        unsafe { AcceptanceRule::<i64>::on_accept_unchecked(&mut rule, 90, &graph, &empty) };
        assert_eq!(rule.penalty_sum, 0);
        // SAFETY: as above.
        unsafe { AcceptanceRule::<i64>::on_accept_unchecked(&mut rule, 80, &graph, &empty) };
        assert_eq!(rule.penalty_sum, 8);
    }

    #[test]
    fn test_accepting_a_move_adds_its_penalty_delta() {
        let graph = three_vessels_on_two_berths();
        let mut rule = started(GuidedLocalSearch::default(), &graph);
        let index = rule.memory.assignment_index(vessel(1), berth(1));
        rule.memory.assignments[index] = 4;
        let mut mutator = Mutator::new(graph);
        mutator.relocate_after(vessel(1), vessel(2));
        mutator.accept();
        // SAFETY: the diff names vessels and berths of the graph the rule started on.
        unsafe {
            AcceptanceRule::<i64>::on_accept_unchecked(
                &mut rule,
                90,
                mutator.graph(),
                mutator.diff(),
            )
        };
        assert_eq!(rule.penalty_sum, 4);
        assert_eq!(rule.penalty_sum, rule.memory.penalty_sum(mutator.graph()));
    }

    #[test]
    fn test_trigger_display_names_the_rule() {
        assert_eq!(
            PenalizationTrigger::OnExhaustion.to_string(),
            "on exhaustion"
        );
        assert_eq!(
            PenalizationTrigger::AfterNonImprovements(5).to_string(),
            "after 5 non-improving steps"
        );
        assert_eq!(
            PenalizationTrigger::AfterMoves(3).to_string(),
            "after 3 accepted moves"
        );
    }
}
