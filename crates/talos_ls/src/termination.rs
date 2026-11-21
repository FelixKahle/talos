// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Why a run ended, and the command by which a monitor ends one.

/// The reason a run ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TerminationReason {
    /// The run reached its time limit.
    TimeLimitReached,
    /// The run decoded as many feasible candidates as its limit allows.
    SolutionLimitReached,
    /// The run made as many candidate moves as its limit allows.
    IterationLimitReached,
    /// The run exhausted as many scans of the neighbourhood as its limit allows.
    CycleLimitReached,
    /// The run made as many candidate moves since its last new best as its limit allows.
    MaxNonImprovingIterations,
    /// The run exhausted as many scans since its last new best as its limit allows.
    MaxNonImprovingCycles,
    /// The run went as long since its last new best as its limit allows.
    MaxNonImprovingTime,
    /// A monitor found the objective it was waiting for.
    TargetObjectiveReached,
    /// The metaheuristic stood at a local optimum and ended the run, or a scan found no move at all.
    NeighborhoodExhausted,
    /// A monitor was told from outside to stop the run.
    Interrupted,
    /// A monitor gave up on the run.
    Aborted,
}

impl std::fmt::Display for TerminationReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let reason = match self {
            Self::TimeLimitReached => "time limit reached",
            Self::SolutionLimitReached => "solution limit reached",
            Self::IterationLimitReached => "iteration limit reached",
            Self::CycleLimitReached => "cycle limit reached",
            Self::MaxNonImprovingIterations => "non-improving iteration limit reached",
            Self::MaxNonImprovingCycles => "non-improving cycle limit reached",
            Self::MaxNonImprovingTime => "non-improving time limit reached",
            Self::TargetObjectiveReached => "target objective reached",
            Self::NeighborhoodExhausted => "neighborhood exhausted",
            Self::Interrupted => "interrupted",
            Self::Aborted => "aborted",
        };
        f.write_str(reason)
    }
}

/// What a monitor tells the engine when it is asked, before each scan and at each checkpoint of a run.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum SearchCommand {
    /// Go on with the run.
    #[default]
    Continue,
    /// End the run for this reason.
    Terminate(TerminationReason),
}

impl std::fmt::Display for SearchCommand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Continue => f.write_str("continue"),
            Self::Terminate(reason) => write!(f, "terminate: {reason}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_command_displays_its_reason() {
        assert_eq!(SearchCommand::Continue.to_string(), "continue");
        assert_eq!(
            SearchCommand::Terminate(TerminationReason::TimeLimitReached).to_string(),
            "terminate: time limit reached"
        );
    }
}
