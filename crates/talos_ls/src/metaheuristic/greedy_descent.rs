// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Greedy descent: a hill climber that accepts only moves that strictly lower the objective, and ends at the first
//! local optimum.

use super::{
    acceptance::{AcceptanceOutcome, AcceptanceRule, NeighborhoodExhaustionOutcome},
    move_selection::MoveSelection,
    tie_breaking::TieBreaker,
};
use crate::schedule::{ScheduleGraph, ScheduleGraphDiff};
use talos_core::SolverNumeric;

/// The settings of a greedy descent.
///
/// With [`MoveSelection::FirstImprovement`] the descent takes the first strictly improving move of each scan; with
/// [`MoveSelection::BestImprovement`] it keeps the best strictly improving move of the scan aside, ties decided by
/// its tie-breaking, and takes it once the scan is exhausted. Either ends the run at the first scan that finds no
/// improving move.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct GreedyDescent {
    /// Which of the improving moves of a scan the descent takes.
    pub selection: MoveSelection,
}

/// What a best-improvement scan has done so far.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scan {
    /// The scan committed no move: it found no improving one.
    NothingCommitted,
    /// The scan committed the best improving move it kept aside.
    Committed,
}

/// A greedy descent under way.
#[derive(Debug, Clone)]
pub enum GreedyDescentRule {
    /// A descent that takes the first improving move of each scan.
    FirstImprovement,
    /// A descent that takes the best improving move of each scan.
    BestImprovement {
        /// The tie-breaking between two moves of equal objective, with its generator where it draws.
        tie_breaker: TieBreaker,
        /// What the current scan has done.
        scan: Scan,
    },
}

impl GreedyDescentRule {
    /// The descent of `settings`, whose random tie-breaking draws from a generator seeded with `seed`.
    pub fn new(settings: &GreedyDescent, seed: u64) -> Self {
        match settings.selection {
            MoveSelection::FirstImprovement => Self::FirstImprovement,
            MoveSelection::BestImprovement(tie_breaking) => Self::BestImprovement {
                tie_breaker: TieBreaker::new(tie_breaking, seed),
                scan: Scan::NothingCommitted,
            },
        }
    }
}

impl<T> AcceptanceRule<T> for GreedyDescentRule
where
    T: SolverNumeric,
{
    fn on_start(&mut self, _graph: &ScheduleGraph, _initial_objective: T) {
        if let Self::BestImprovement { scan, .. } = self {
            *scan = Scan::NothingCommitted;
        }
    }

    unsafe fn decide_unchecked(
        &mut self,
        accepted_objective: T,
        buffered_objective: Option<T>,
        candidate_objective: T,
        _diff: &ScheduleGraphDiff,
    ) -> AcceptanceOutcome {
        if candidate_objective >= accepted_objective {
            return AcceptanceOutcome::Reject;
        }
        match (self, buffered_objective) {
            (Self::FirstImprovement, _) => AcceptanceOutcome::Accept,
            (Self::BestImprovement { .. }, None) => AcceptanceOutcome::Buffer,
            (Self::BestImprovement { tie_breaker, .. }, Some(buffered)) => {
                if candidate_objective < buffered
                    || (candidate_objective == buffered && tie_breaker.prefers_newer())
                {
                    AcceptanceOutcome::Buffer
                } else {
                    AcceptanceOutcome::Reject
                }
            }
        }
    }

    unsafe fn on_accept_unchecked(
        &mut self,
        _accepted_objective: T,
        _graph: &ScheduleGraph,
        _diff: &ScheduleGraphDiff,
    ) {
        // A best-improvement descent accepts only the move it kept aside, at the end of a scan.
        if let Self::BestImprovement { scan, .. } = self {
            *scan = Scan::Committed;
        }
    }

    fn on_new_best(&mut self, _best_objective: T) {}

    fn after_step(&mut self, _best_objective: T, _graph: &ScheduleGraph) {}

    fn commits_buffered(&self) -> bool {
        matches!(self, Self::BestImprovement { .. })
    }

    fn on_neighborhood_exhausted(
        &mut self,
        _graph: &ScheduleGraph,
    ) -> NeighborhoodExhaustionOutcome {
        match self {
            Self::BestImprovement { scan, .. } if *scan == Scan::Committed => {
                *scan = Scan::NothingCommitted;
                NeighborhoodExhaustionOutcome::Restart
            }
            Self::BestImprovement { .. } | Self::FirstImprovement => {
                NeighborhoodExhaustionOutcome::Terminate
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metaheuristic::tie_breaking::TieBreaking;

    fn rule(selection: MoveSelection) -> GreedyDescentRule {
        GreedyDescentRule::new(&GreedyDescent { selection }, 0)
    }

    fn best(tie_breaking: TieBreaking) -> MoveSelection {
        MoveSelection::BestImprovement(tie_breaking)
    }

    fn decide(
        rule: &mut GreedyDescentRule,
        buffered: Option<i64>,
        candidate: i64,
    ) -> AcceptanceOutcome {
        // SAFETY: an empty diff names no vessel or berth.
        unsafe {
            AcceptanceRule::<i64>::decide_unchecked(
                rule,
                10,
                buffered,
                candidate,
                &ScheduleGraphDiff::new(),
            )
        }
    }

    #[test]
    fn test_descent_rejects_every_move_that_does_not_improve() {
        for selection in [MoveSelection::FirstImprovement, best(TieBreaking::KeepLast)] {
            let mut descent = rule(selection);
            assert_eq!(decide(&mut descent, None, 10), AcceptanceOutcome::Reject);
            assert_eq!(decide(&mut descent, None, 11), AcceptanceOutcome::Reject);
        }
    }

    #[test]
    fn test_first_improvement_accepts_the_first_improving_move() {
        let mut descent = rule(MoveSelection::FirstImprovement);
        assert_eq!(decide(&mut descent, None, 9), AcceptanceOutcome::Accept);
        assert!(!AcceptanceRule::<i64>::commits_buffered(&descent));
    }

    #[test]
    fn test_best_improvement_keeps_the_best_aside_and_breaks_ties_by_its_rule() {
        let mut first = rule(best(TieBreaking::KeepFirst));
        assert_eq!(decide(&mut first, None, 9), AcceptanceOutcome::Buffer);
        assert_eq!(decide(&mut first, Some(8), 7), AcceptanceOutcome::Buffer);
        assert_eq!(decide(&mut first, Some(8), 9), AcceptanceOutcome::Reject);
        assert_eq!(decide(&mut first, Some(8), 8), AcceptanceOutcome::Reject);
        let mut last = rule(best(TieBreaking::KeepLast));
        assert_eq!(decide(&mut last, Some(8), 8), AcceptanceOutcome::Buffer);
    }

    #[test]
    fn test_best_improvement_restarts_after_a_commit_and_ends_after_a_scan_without_one() {
        let graph = ScheduleGraph::from_slices::<i64>(&[], &[], 0);
        let mut descent = rule(best(TieBreaking::KeepFirst));
        assert!(AcceptanceRule::<i64>::commits_buffered(&descent));
        // SAFETY: an empty diff names no vessel or berth.
        unsafe {
            AcceptanceRule::<i64>::on_accept_unchecked(
                &mut descent,
                9,
                &graph,
                &ScheduleGraphDiff::new(),
            );
        }
        let exhausted = AcceptanceRule::<i64>::on_neighborhood_exhausted(&mut descent, &graph);
        assert_eq!(exhausted, NeighborhoodExhaustionOutcome::Restart);
        let exhausted = AcceptanceRule::<i64>::on_neighborhood_exhausted(&mut descent, &graph);
        assert_eq!(exhausted, NeighborhoodExhaustionOutcome::Terminate);
    }

    #[test]
    fn test_first_improvement_ends_at_the_first_exhausted_scan() {
        let graph = ScheduleGraph::from_slices::<i64>(&[], &[], 0);
        let mut descent = rule(MoveSelection::FirstImprovement);
        let exhausted = AcceptanceRule::<i64>::on_neighborhood_exhausted(&mut descent, &graph);
        assert_eq!(exhausted, NeighborhoodExhaustionOutcome::Terminate);
    }
}
