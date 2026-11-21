// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! What a portfolio answers, and why a portfolio refuses to start.

use crate::{outcome::LocalSearchOutcome, parameters::InitialSolutionError};
use talos_core::SolverNumeric;

/// What a portfolio found: the outcome of each member's run, in the order of the members, and among them the best.
///
/// The best member is the one whose best schedule has the lowest objective, the lowest index among those that tie,
/// so that the answer does not depend on which member ended first. Each outcome is what a lone
/// [`Engine::run`](crate::Engine::run) of the member answers, unless a deadline ended the run.
// Invariant: `outcomes` is not empty.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PortfolioOutcome<T> {
    /// The outcome of each member's run, in the order of the members.
    outcomes: Vec<LocalSearchOutcome<T>>,
}

impl<T> PortfolioOutcome<T>
where
    T: SolverNumeric,
{
    /// The outcome of a portfolio whose members' runs answered `outcomes`, in the order of the members.
    ///
    /// `outcomes` is not empty.
    pub(super) fn new(outcomes: Vec<LocalSearchOutcome<T>>) -> Self {
        debug_assert!(
            !outcomes.is_empty(),
            "called `PortfolioOutcome::new` without an outcome"
        );
        Self { outcomes }
    }

    /// The outcome of each member's run, in the order of the members; never empty.
    pub fn outcomes(&self) -> &[LocalSearchOutcome<T>] {
        &self.outcomes
    }

    /// The index of the best member: the one whose best schedule has the lowest objective, the lowest index among
    /// those that tie. It costs one pass over the members.
    #[expect(
        clippy::missing_panics_doc,
        reason = "the outcomes are never empty, the invariant of the type, so the minimum exists"
    )]
    pub fn best_member(&self) -> usize {
        // Of several equal minima, `min_by_key` answers the first, the member of lowest index.
        self.outcomes
            .iter()
            .enumerate()
            .min_by_key(|(_, outcome)| outcome.solution.objective_value())
            .map(|(member, _)| member)
            .expect("a portfolio outcome should hold an outcome per member, of at least one member")
    }

    /// The outcome of the best member, as [`best_member`](Self::best_member) names it.
    pub fn best(&self) -> &LocalSearchOutcome<T> {
        &self.outcomes[self.best_member()]
    }

    /// The outcome of each member's run, in the order of the members; never empty.
    pub fn into_outcomes(self) -> Vec<LocalSearchOutcome<T>> {
        self.outcomes
    }
}

/// Why a portfolio refused to start; no member runs once it is refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PortfolioError {
    /// The portfolio has no member.
    NoMember,
    /// The parameters of a member name no neighbourhood to explore; of several such members, the one of lowest
    /// index.
    NoNeighborhood {
        /// The index of the member.
        member: usize,
    },
    /// The initial schedule is not one a run can start from, as a lone [`Engine::run`](crate::Engine::run)
    /// refuses it.
    InitialSolution(InitialSolutionError),
}

impl From<InitialSolutionError> for PortfolioError {
    fn from(error: InitialSolutionError) -> Self {
        Self::InitialSolution(error)
    }
}

// A refused initial schedule is the whole story of the refusal, so its message is the refusal's own.
impl std::fmt::Display for PortfolioError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoMember => f.write_str("the portfolio has no member"),
            Self::NoNeighborhood { member } => write!(
                f,
                "the parameters of member {member} name no neighborhood to explore"
            ),
            Self::InitialSolution(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for PortfolioError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{statistics::LocalSearchStatistics, termination::TerminationReason};
    use talos_model::{BerthIndex, Solution, SolutionView};

    /// An outcome of one vessel whose best schedule has `objective`, told apart by `iterations`.
    fn outcome(objective: i64, iterations: u64) -> LocalSearchOutcome<i64> {
        LocalSearchOutcome {
            solution: Solution::from(SolutionView::new(&[BerthIndex::new(0)], &[0], objective)),
            termination_reason: TerminationReason::IterationLimitReached,
            statistics: LocalSearchStatistics {
                iterations,
                ..LocalSearchStatistics::default()
            },
        }
    }

    #[test]
    fn test_the_best_member_has_the_lowest_objective_and_the_lowest_index_among_ties() {
        let portfolio = PortfolioOutcome::new(vec![
            outcome(30, 0),
            outcome(10, 1),
            outcome(20, 2),
            outcome(10, 3),
        ]);
        assert_eq!(portfolio.best_member(), 1);
        assert_eq!(portfolio.best(), &outcome(10, 1));
        assert_eq!(portfolio.outcomes().len(), 4);

        let alone = PortfolioOutcome::new(vec![outcome(5, 0)]);
        assert_eq!(alone.best_member(), 0);
        let tied = PortfolioOutcome::new(vec![outcome(7, 0), outcome(7, 1)]);
        assert_eq!(tied.best(), &outcome(7, 0));
        let last = PortfolioOutcome::new(vec![outcome(9, 0), outcome(8, 1), outcome(4, 2)]);
        assert_eq!(last.best_member(), 2);
    }

    #[test]
    fn test_the_outcomes_keep_the_order_of_the_members() {
        let portfolio = PortfolioOutcome::new(vec![outcome(3, 0), outcome(1, 1), outcome(2, 2)]);
        assert_eq!(
            portfolio.into_outcomes(),
            vec![outcome(3, 0), outcome(1, 1), outcome(2, 2)]
        );
    }

    #[test]
    fn test_errors_display_their_values() {
        assert_eq!(
            PortfolioError::NoMember.to_string(),
            "the portfolio has no member"
        );
        assert_eq!(
            PortfolioError::NoNeighborhood { member: 3 }.to_string(),
            "the parameters of member 3 name no neighborhood to explore"
        );
        assert_eq!(
            PortfolioError::from(InitialSolutionError::InfeasibleOrder).to_string(),
            InitialSolutionError::InfeasibleOrder.to_string()
        );
    }
}
