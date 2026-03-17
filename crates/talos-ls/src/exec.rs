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

// ----------------------------------------------------------------
// TerminationReason
// ----------------------------------------------------------------

/// The exact reason a local search run terminated.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminationReason {
    /// The predefined time limit was exceeded.
    TimeLimitReached,
    /// The predefined solution limit was exceeded.
    SolutionLimitReached,
    /// The absolute iteration limit was reached.
    IterationLimitReached,
    /// The absolute cycle (full neighborhood traversal) limit was reached.
    CycleLimitReached,
    /// The search ran for too many iterations without finding a new best solution.
    MaxNonImprovingIterations,
    /// The search ran for too many cycles without finding a new best solution.
    MaxNonImprovingCycles,
    /// The search ran for too long without finding a new best solution.
    MaxNonImprovingTime,
    /// A target objective value was hit.
    TargetObjectiveReached,
    /// The neighborhood was completely exhausted without finding improvements (Local Optimum).
    NeighborhoodExhausted,
    /// The search was interrupted externally (e.g., Ctrl+C or a thread signal).
    Interrupted,
    /// An algorithmic constraint was violated or a fatal error occurred (replace generic 'Other').
    Aborted,
}

impl std::fmt::Display for TerminationReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TerminationReason::TimeLimitReached => write!(f, "Time limit reached"),
            TerminationReason::IterationLimitReached => write!(f, "Iteration limit reached"),
            TerminationReason::CycleLimitReached => write!(f, "Cycle limit reached"),
            TerminationReason::SolutionLimitReached => write!(f, "Solution limit reached"),
            TerminationReason::MaxNonImprovingIterations => {
                write!(f, "Max non-improving iterations reached")
            }
            TerminationReason::MaxNonImprovingCycles => {
                write!(f, "Max non-improving cycles reached")
            }
            TerminationReason::MaxNonImprovingTime => {
                write!(f, "Max non-improving time reached")
            }
            TerminationReason::TargetObjectiveReached => write!(f, "Target objective reached"),
            TerminationReason::NeighborhoodExhausted => {
                write!(f, "Neighborhood exhausted (Local Optimum)")
            }
            TerminationReason::Interrupted => write!(f, "Search interrupted"),
            TerminationReason::Aborted => write!(f, "Search aborted"),
        }
    }
}

// ----------------------------------------------------------------
// SearchCommand
// ----------------------------------------------------------------

/// Command returned by monitors or metaheuristics to control flow.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub enum SearchCommand {
    #[default]
    Continue,
    Terminate(TerminationReason),
}

impl std::fmt::Debug for SearchCommand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SearchCommand::Continue => write!(f, "Continue"),
            SearchCommand::Terminate(reason) => write!(f, "Terminate: {}", reason),
        }
    }
}

impl std::fmt::Display for SearchCommand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SearchCommand::Continue => write!(f, "Continue"),
            SearchCommand::Terminate(reason) => write!(f, "Terminate: {}", reason),
        }
    }
}
