// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A guided local search under way: the rule the engine asks during a run.

use super::{
    penalty_memory::{Feature, PenaltyMemory, real_sum},
    schedules::{DecaySchedule, LambdaSchedule},
    settings::{GuidedLocalSearch, PenalizationTrigger},
};
use crate::{
    metaheuristic::acceptance::{AcceptanceOutcome, AcceptanceRule, NeighborhoodExhaustionOutcome},
    schedule::{ScheduleGraph, ScheduleGraphDiff},
};
use std::num::NonZeroU64;
use talos_core::SolverNumeric;

/// The trigger of a run's penalizations, with what it has counted.
#[derive(Debug, Clone, Copy, PartialEq)]
enum TriggerState {
    /// Penalize at every exhausted scan.
    OnExhaustion,
    /// Penalize once `steps` steps in a row passed without a new best.
    AfterNonImprovements {
        /// The steps in a row that make the trigger fire.
        steps: NonZeroU64,
        /// The steps since the last new best the trigger saw, or since it last fired.
        count: u64,
        /// The best objective the trigger has seen.
        best: f64,
    },
    /// Penalize after every `moves` accepted moves, and at every exhausted scan.
    AfterMoves {
        /// The accepted moves that make the trigger fire.
        moves: NonZeroU64,
        /// The accepted moves since the trigger last fired, or since the start.
        count: u64,
    },
}

impl TriggerState {
    /// The trigger of `trigger`, before the start of a run.
    fn new(trigger: PenalizationTrigger) -> Self {
        match trigger {
            PenalizationTrigger::OnExhaustion => Self::OnExhaustion,
            PenalizationTrigger::AfterNonImprovements(steps) => Self::AfterNonImprovements {
                steps,
                count: 0,
                best: f64::INFINITY,
            },
            PenalizationTrigger::AfterMoves(moves) => Self::AfterMoves { moves, count: 0 },
        }
    }

    /// Starts counting at the start of a run from a schedule of `initial_objective`.
    fn restart(&mut self, initial_objective: f64) {
        match self {
            Self::OnExhaustion => {}
            Self::AfterNonImprovements { count, best, .. } => {
                *count = 0;
                *best = initial_objective;
            }
            Self::AfterMoves { count, .. } => *count = 0,
        }
    }

    /// Counts an accepted move, and whether the trigger fires.
    fn on_accept(&mut self) -> bool {
        match self {
            Self::AfterMoves { moves, count } => {
                *count += 1;
                let fires = *count >= moves.get();
                if fires {
                    *count = 0;
                }
                fires
            }
            Self::OnExhaustion | Self::AfterNonImprovements { .. } => false,
        }
    }

    /// Counts a step after which the best objective is `best_objective`, and whether the trigger fires.
    fn after_step(&mut self, best_objective: f64) -> bool {
        match self {
            Self::AfterNonImprovements { steps, count, best } => {
                if best_objective < *best {
                    *count = 0;
                    *best = best_objective;
                    return false;
                }
                *count += 1;
                let fires = *count >= steps.get();
                if fires {
                    *count = 0;
                }
                fires
            }
            Self::OnExhaustion | Self::AfterMoves { .. } => false,
        }
    }

    /// Whether an exhausted scan penalizes.
    fn penalizes_at_exhaustion(&self) -> bool {
        match self {
            // An exhausted scan accepted no move, so a count of accepted moves cannot reach its threshold from
            // there: without a penalty every later scan would repeat it.
            Self::OnExhaustion | Self::AfterMoves { .. } => true,
            // Every candidate of the next scan is a step without a new best, so the count fires within a few scans.
            Self::AfterNonImprovements { .. } => false,
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
    /// When the search penalizes, with what it has counted.
    trigger: TriggerState,
    /// The sum of the penalties of the features of the accepted schedule.
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
            trigger: TriggerState::new(settings.trigger),
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
        // A schedule has a link with a vessel at each end of each berth and between consecutive vessels, at most
        // `num_vessels + num_berths`, and an assignment per vessel: every feature a penalization can raise.
        self.winners.clear();
        self.winners
            .reserve(2 * graph.num_vessels() + graph.num_berths());
        self.penalty_sum = self.memory.penalty_sum(graph);
        self.trigger
            .restart(initial_objective.to_f64().unwrap_or(0.0));
        self.decay.penalizations = 0;
        self.lambda.restart();
    }

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
        debug_assert!(self.memory.covers(diff));
        // SAFETY: the diff names only vessels and berths of the graph the run searches, which the memory counts
        // from the start of the run on, the precondition of this function.
        let candidate_penalty_sum =
            self.penalty_sum + unsafe { self.memory.penalty_delta_unchecked(diff) };
        let lambda = self.lambda.current;
        let augmented_candidate = candidate + lambda * real_sum(candidate_penalty_sum);
        let augmented_accepted = accepted + lambda * real_sum(self.penalty_sum);
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
        debug_assert!(self.memory.covers(diff));
        // SAFETY: the diff names only vessels and berths of the graph the run searches, which the memory counts
        // from the start of the run on, the precondition of this function.
        self.penalty_sum += unsafe { self.memory.penalty_delta_unchecked(diff) };
        if self.trigger.on_accept() {
            self.penalize(graph);
        }
    }

    fn on_new_best(&mut self, _best_objective: T) {
        self.lambda.on_new_best();
    }

    fn after_step(&mut self, best_objective: T, graph: &ScheduleGraph) {
        if let TriggerState::AfterNonImprovements { .. } = self.trigger
            && self
                .trigger
                .after_step(best_objective.to_f64().unwrap_or(0.0))
        {
            self.penalize(graph);
        }
    }

    fn commits_buffered(&self) -> bool {
        false
    }

    fn on_neighborhood_exhausted(
        &mut self,
        graph: &ScheduleGraph,
    ) -> NeighborhoodExhaustionOutcome {
        if self.trigger.penalizes_at_exhaustion() {
            self.penalize(graph);
        }
        NeighborhoodExhaustionOutcome::Restart
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        metaheuristic::guided_local_search::settings::{Lambda, StaticLambda},
        schedule::Mutator,
    };
    use talos_model::{BerthIndex, VesselIndex};

    fn vessel(index: usize) -> VesselIndex {
        VesselIndex::new(index)
    }

    fn berth(index: usize) -> BerthIndex {
        BerthIndex::new(index)
    }

    /// V0 and V1 on B0 in that order, V2 on B1.
    fn three_vessels_on_two_berths() -> ScheduleGraph {
        ScheduleGraph::from_slices(&[berth(0), berth(0), berth(1)], &[0_i64, 1, 0], 2)
    }

    /// `count` as a count that is not zero.
    fn nonzero(count: u64) -> NonZeroU64 {
        NonZeroU64::new(count).expect("the count should not be zero")
    }

    fn started(settings: GuidedLocalSearch, graph: &ScheduleGraph) -> GuidedLocalSearchRule {
        let mut rule = GuidedLocalSearchRule::new(&settings);
        AcceptanceRule::<i64>::on_start(&mut rule, graph, 100);
        rule
    }

    // A search of three vessels on two berths, set up for nothing smaller: its memory must count the graph it
    // searches, or a move of the last vessel reads past the tables.
    #[test]
    fn test_start_sizes_the_memory_for_the_graph_searched() {
        let graph = three_vessels_on_two_berths();
        let rule = started(GuidedLocalSearch::default(), &graph);
        assert_eq!(rule.memory.counts(), (3, 2));
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
        rule.memory.set_assignment_penalty(vessel(1), berth(1), 1);
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
            trigger: PenalizationTrigger::AfterNonImprovements(nonzero(3)),
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
        assert!(matches!(
            rule.trigger,
            TriggerState::AfterNonImprovements { count: 0, .. }
        ));
        // Exhaustion does not penalize under this trigger.
        let _ = AcceptanceRule::<i64>::on_neighborhood_exhausted(&mut rule, &graph);
        assert_eq!(rule.penalty_sum, 8);
    }

    #[test]
    fn test_moves_trigger_fires_at_every_so_many_accepted_moves() {
        let graph = three_vessels_on_two_berths();
        let settings = GuidedLocalSearch {
            trigger: PenalizationTrigger::AfterMoves(nonzero(2)),
            ..GuidedLocalSearch::default()
        };
        let mut rule = started(settings, &graph);
        let empty = ScheduleGraphDiff::new();
        let mut accept = |objective: i64| {
            // SAFETY: an empty diff names no vessel or berth.
            unsafe {
                AcceptanceRule::<i64>::on_accept_unchecked(&mut rule, objective, &graph, &empty)
            };
            rule.penalty_sum
        };
        assert_eq!(accept(90), 0);
        assert_eq!(accept(80), 8);
    }

    #[test]
    fn test_trigger_state_counts_as_its_trigger_says() {
        let three = nonzero(3);
        let mut after_steps = TriggerState::new(PenalizationTrigger::AfterNonImprovements(three));
        after_steps.restart(100.0);
        assert_eq!(
            [100.0, 100.0, 90.0, 90.0, 90.0, 90.0].map(|best| after_steps.after_step(best)),
            [false, false, false, false, false, true]
        );
        assert!(!after_steps.on_accept() && !after_steps.penalizes_at_exhaustion());

        let mut after_moves = TriggerState::new(PenalizationTrigger::AfterMoves(three));
        assert_eq!(
            [(); 6].map(|()| after_moves.on_accept()),
            [false, false, true, false, false, true]
        );
        assert!(!after_moves.after_step(0.0) && after_moves.penalizes_at_exhaustion());

        let mut on_exhaustion = TriggerState::new(PenalizationTrigger::OnExhaustion);
        assert!(!on_exhaustion.on_accept() && !on_exhaustion.after_step(0.0));
        assert!(on_exhaustion.penalizes_at_exhaustion());
    }

    #[test]
    fn test_moves_trigger_penalizes_at_an_exhausted_scan() {
        let graph = three_vessels_on_two_berths();
        let settings = GuidedLocalSearch {
            trigger: PenalizationTrigger::AfterMoves(nonzero(1000)),
            ..GuidedLocalSearch::default()
        };
        let mut rule = started(settings, &graph);
        let outcome = AcceptanceRule::<i64>::on_neighborhood_exhausted(&mut rule, &graph);
        assert_eq!(outcome, NeighborhoodExhaustionOutcome::Restart);
        assert_eq!(rule.penalty_sum, 8);
    }

    #[test]
    fn test_accepting_a_move_adds_its_penalty_delta() {
        let graph = three_vessels_on_two_berths();
        let mut rule = started(GuidedLocalSearch::default(), &graph);
        rule.memory.set_assignment_penalty(vessel(1), berth(1), 4);
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
}
