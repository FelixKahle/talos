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

use talos_model::solution::{Solution, SolutionView};

pub trait GlobalOracle<T> {
    fn try_push_solution(&self, solution: &Solution<T>) -> bool;
    fn try_push_solution_view(&self, solution: SolutionView<'_, T>) -> bool;
    fn best_objective(&self) -> Option<T>;
    fn with_best<F, R>(&self, f: F) -> Option<R>
    where
        F: FnOnce(&Solution<T>) -> R;

    /// Returns the number of solutions currently held in the pool.
    fn pool_len(&self) -> usize;

    /// Applies `f` to the solution at the given rank (0 = best).
    /// Returns `None` if the rank is out of bounds or the pool is empty.
    fn with_ranked<F, R>(&self, rank: usize, f: F) -> Option<R>
    where
        F: FnOnce(&Solution<T>) -> R;
}

pub struct NoOracle;

impl<T> GlobalOracle<T> for NoOracle {
    fn try_push_solution(&self, _solution: &Solution<T>) -> bool {
        false
    }

    fn try_push_solution_view(&self, _solution: SolutionView<'_, T>) -> bool {
        false
    }

    fn best_objective(&self) -> Option<T> {
        None
    }

    fn with_best<F, R>(&self, _f: F) -> Option<R>
    where
        F: FnOnce(&Solution<T>) -> R,
    {
        None
    }

    fn pool_len(&self) -> usize {
        0
    }

    fn with_ranked<F, R>(&self, _rank: usize, _f: F) -> Option<R>
    where
        F: FnOnce(&Solution<T>) -> R,
    {
        None
    }
}
