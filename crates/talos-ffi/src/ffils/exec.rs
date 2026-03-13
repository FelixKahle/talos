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

use talos_ls::exec::TerminationReason;

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfiTerminationReason {
    TimeLimitReached = 0,
    SolutionLimitReached = 1,
    IterationLimitReached = 2,
    CycleLimitReached = 3,
    MaxNonImprovingIterations = 4,
    MaxNonImprovingCycles = 5,
    MaxNonImprovingTime = 6,
    TargetObjectiveReached = 7,
    NeighborhoodExhausted = 8,
    Interrupted = 9,
    Aborted = 10,
}

impl From<TerminationReason> for FfiTerminationReason {
    fn from(r: TerminationReason) -> Self {
        match r {
            TerminationReason::TimeLimitReached => Self::TimeLimitReached,
            TerminationReason::SolutionLimitReached => Self::SolutionLimitReached,
            TerminationReason::IterationLimitReached => Self::IterationLimitReached,
            TerminationReason::CycleLimitReached => Self::CycleLimitReached,
            TerminationReason::MaxNonImprovingIterations => Self::MaxNonImprovingIterations,
            TerminationReason::MaxNonImprovingCycles => Self::MaxNonImprovingCycles,
            TerminationReason::MaxNonImprovingTime => Self::MaxNonImprovingTime,
            TerminationReason::TargetObjectiveReached => Self::TargetObjectiveReached,
            TerminationReason::NeighborhoodExhausted => Self::NeighborhoodExhausted,
            TerminationReason::Interrupted => Self::Interrupted,
            TerminationReason::Aborted => Self::Aborted,
        }
    }
}
