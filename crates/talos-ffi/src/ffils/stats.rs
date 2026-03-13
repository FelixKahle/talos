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

use talos_ls::stats::LocalSearchStatistics;

/// C-compatible mirror of [`LocalSearchStatistics`].
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FfiLocalSearchStatistics {
    pub iterations: u64,
    pub cycles: u64,
    pub total_solutions: u64,
    pub accepted_solutions: u64,
    pub infeasible_moves: u64,
    /// Total elapsed time in nanoseconds.
    pub time_total_nanos: u64,
}

impl From<&LocalSearchStatistics> for FfiLocalSearchStatistics {
    fn from(s: &LocalSearchStatistics) -> Self {
        Self {
            iterations: s.iterations,
            cycles: s.cycles,
            total_solutions: s.total_solutions,
            accepted_solutions: s.accepted_solutions,
            infeasible_moves: s.infeasible_moves,
            time_total_nanos: s.time_total.as_nanos() as u64,
        }
    }
}
