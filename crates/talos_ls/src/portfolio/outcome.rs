// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! What a portfolio answers, and why a portfolio refuses to start.

use crate::{outcome::LocalSearchOutcome, parameters::LocalSearchError};
use talos_core::SolverNumeric;

/// What a portfolio found: the outcome of each member's run, in the order of the members, and among them the best.
///
/// The best member is the one whose best schedule has the lowest objective, the lowest index among those that tie,
/// so that the answer does not depend on which member ended first. Each outcome is what a lone
/// [`Engine::run`](crate::Engine::run) of the member answers, unless a deadline ended the run.
// Invariant: `members` is not empty.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PortfolioOutcome<T> {
    /// The outcome of each member's run, in the order of the members.
    members: Vec<LocalSearchOutcome<T>>,
}

impl<T> PortfolioOutcome<T>
where
    T: SolverNumeric,
{
    /// The outcome of a portfolio whose members' runs answered `runs`, in the order of the members, or the refusal
    /// of the member of lowest index among those refused.
    ///
    /// `runs` is not empty.
    ///
    /// # Errors
    ///
    /// [`PortfolioError::Member`] with the index and the error of the first refused run.
    pub(super) fn from_runs(
        runs: Vec<Result<LocalSearchOutcome<T>, LocalSearchError>>,
    ) -> Result<Self, PortfolioError> {
        debug_assert!(
            !runs.is_empty(),
            "called `PortfolioOutcome::from_runs` without a run"
        );
        let members = runs
            .into_iter()
            .enumerate()
            .map(|(member, run)| run.map_err(|error| PortfolioError::Member { member, error }))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { members })
    }

    /// The outcome of each member's run, in the order of the members; never empty.
    pub fn members(&self) -> &[LocalSearchOutcome<T>] {
        &self.members
    }

    /// The index of the best member: the one whose best schedule has the lowest objective, the lowest index among
    /// those that tie. It costs one pass over the members.
    pub fn best_member(&self) -> usize {
        // A later member replaces the best only with a strictly lower objective, so a tie keeps the lower index.
        self.members
            .iter()
            .map(|outcome| outcome.solution.objective_value())
            .enumerate()
            .fold(None, |best, (member, objective)| match best {
                Some((_, lowest)) if lowest <= objective => best,
                _ => Some((member, objective)),
            })
            .map_or(0, |(member, _)| member)
    }

    /// The outcome of the best member, as [`best_member`](Self::best_member) names it.
    pub fn best(&self) -> &LocalSearchOutcome<T> {
        &self.members[self.best_member()]
    }

    /// The outcome of each member's run, in the order of the members; never empty.
    pub fn into_members(self) -> Vec<LocalSearchOutcome<T>> {
        self.members
    }
}

/// Why a portfolio refused to start.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PortfolioError {
    /// The portfolio has no member.
    NoMember,
    /// The run of a member was refused, as a lone [`Engine::run`](crate::Engine::run) of the member refuses it; of
    /// several refused members, the one of lowest index.
    Member {
        /// The index of the member.
        member: usize,
        /// Why its run was refused.
        error: LocalSearchError,
    },
}

impl std::fmt::Display for PortfolioError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoMember => f.write_str("the portfolio has no member"),
            Self::Member { member, .. } => write!(f, "the run of member {member} was refused"),
        }
    }
}

impl std::error::Error for PortfolioError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::NoMember => None,
            Self::Member { error, .. } => Some(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{statistics::LocalSearchStatistics, termination::TerminationReason};
    use std::error::Error;
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
        let portfolio = PortfolioOutcome::from_runs(vec![
            Ok(outcome(30, 0)),
            Ok(outcome(10, 1)),
            Ok(outcome(20, 2)),
            Ok(outcome(10, 3)),
        ])
        .expect("no run was refused");
        assert_eq!(portfolio.best_member(), 1);
        assert_eq!(portfolio.best(), &outcome(10, 1));
        assert_eq!(portfolio.members().len(), 4);

        let alone =
            PortfolioOutcome::from_runs(vec![Ok(outcome(5, 0))]).expect("no run was refused");
        assert_eq!(alone.best_member(), 0);
        let tied = PortfolioOutcome::from_runs(vec![Ok(outcome(7, 0)), Ok(outcome(7, 1))])
            .expect("no run was refused");
        assert_eq!(tied.best(), &outcome(7, 0));
    }

    #[test]
    fn test_the_members_keep_the_order_of_the_runs() {
        let runs = vec![Ok(outcome(3, 0)), Ok(outcome(1, 1)), Ok(outcome(2, 2))];
        let portfolio = PortfolioOutcome::from_runs(runs).expect("no run was refused");
        assert_eq!(
            portfolio.into_members(),
            vec![outcome(3, 0), outcome(1, 1), outcome(2, 2)]
        );
    }

    #[test]
    fn test_the_refusal_of_the_lowest_refused_member_is_the_answer() {
        let runs = vec![
            Ok(outcome(3, 0)),
            Err(LocalSearchError::InfeasibleInitialSolution),
            Err(LocalSearchError::NoNeighborhood),
        ];
        let refusal = PortfolioOutcome::from_runs(runs).expect_err("member 1 was refused");
        assert_eq!(
            refusal,
            PortfolioError::Member {
                member: 1,
                error: LocalSearchError::InfeasibleInitialSolution
            }
        );
        assert_eq!(refusal.to_string(), "the run of member 1 was refused");
        assert_eq!(
            refusal.source().map(ToString::to_string),
            Some(LocalSearchError::InfeasibleInitialSolution.to_string())
        );
    }
}
