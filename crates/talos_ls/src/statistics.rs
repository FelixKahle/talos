// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The work a run counts, by kind of step.

/// The work a run has done, counted by kind of step.
///
/// Every candidate is an iteration; a feasible one is also a solution, and is then accepted, kept aside or
/// rejected; an infeasible one is an infeasible move. A cycle is a scan of the neighbourhood run to exhaustion.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct LocalSearchStatistics {
    /// The candidate moves made, feasible or not.
    pub iterations: u64,
    /// The scans of the neighbourhood that ran to exhaustion.
    pub cycles: u64,
    /// The candidates that decoded to a feasible schedule.
    pub total_solutions: u64,
    /// The candidates that became the accepted schedule, at once or kept aside and committed at the end of a scan.
    pub accepted_solutions: u64,
    /// The candidates that decoded to no feasible schedule.
    pub infeasible_moves: u64,
}

impl LocalSearchStatistics {
    /// The feasible candidates that did not become the accepted schedule: the total solutions less the accepted
    /// ones, or zero where the counts, set by hand, have more accepted than total solutions.
    pub fn rejected_solutions(&self) -> u64 {
        self.total_solutions.saturating_sub(self.accepted_solutions)
    }
}

impl std::fmt::Display for LocalSearchStatistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "iterations:          {}", self.iterations)?;
        writeln!(f, "cycles:              {}", self.cycles)?;
        writeln!(f, "total solutions:     {}", self.total_solutions)?;
        writeln!(f, "accepted solutions:  {}", self.accepted_solutions)?;
        writeln!(f, "rejected solutions:  {}", self.rejected_solutions())?;
        write!(f, "infeasible moves:    {}", self.infeasible_moves)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rejected_solutions_are_the_feasible_ones_not_accepted() {
        let statistics = LocalSearchStatistics {
            total_solutions: 10,
            accepted_solutions: 3,
            ..LocalSearchStatistics::default()
        };
        assert_eq!(statistics.rejected_solutions(), 7);
    }

    // The fields are public, so a caller may write more accepted than total solutions; the difference must not
    // underflow.
    #[test]
    fn test_rejected_solutions_of_counts_set_by_hand_are_never_below_zero() {
        let statistics = LocalSearchStatistics {
            total_solutions: 1,
            accepted_solutions: 3,
            ..LocalSearchStatistics::default()
        };
        assert_eq!(statistics.rejected_solutions(), 0);
    }
}
