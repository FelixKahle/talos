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

//! A bounded pool of solutions for the DBAP.
//!
//! `SolutionPool` maintains up to `max_size` solutions, ordered by objective
//! value (ascending — best first). When the pool is full, a new solution is
//! only inserted if it improves on the current worst.

use talos_model::solution::{Solution, SolutionView};

/// A bounded, sorted pool of `Solution`s.
///
/// Solutions are kept in ascending order of objective value (best first).
/// When the pool reaches its capacity (`max_size`), inserting a solution
/// whose objective is worse than or equal to the current worst is a no-op;
/// otherwise the worst solution is evicted.
///
/// # Panics
///
/// `new` panics if `max_size` is zero.
pub struct SolutionPool<T> {
    solutions: Vec<Solution<T>>,
    max_size: usize,
}

impl<T> SolutionPool<T> {
    /// Creates an empty pool that can hold at most `max_size` solutions.
    ///
    /// # Panics
    ///
    /// Panics if `max_size == 0`.
    #[inline]
    pub fn new(max_size: usize) -> Self {
        assert!(max_size > 0, "SolutionPool max_size must be > 0");

        Self {
            solutions: Vec::with_capacity(max_size),
            max_size,
        }
    }

    /// Returns the maximum number of solutions the pool can hold.
    #[inline]
    pub fn max_size(&self) -> usize {
        self.max_size
    }

    /// Returns the current number of solutions in the pool.
    #[inline]
    pub fn len(&self) -> usize {
        self.solutions.len()
    }

    /// Returns `true` if the pool contains no solutions.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.solutions.is_empty()
    }

    /// Returns `true` if the pool has reached its capacity.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.solutions.len() >= self.max_size
    }

    /// Returns the best (lowest objective) solution, if any.
    #[inline]
    pub fn best(&self) -> Option<&Solution<T>> {
        self.solutions.first()
    }

    /// Returns the worst (highest objective) solution, if any.
    #[inline]
    pub fn worst(&self) -> Option<&Solution<T>> {
        self.solutions.last()
    }

    /// Returns a reference to the solution at the given rank (0 = best).
    #[inline]
    pub fn get(&self, index: usize) -> Option<&Solution<T>> {
        self.solutions.get(index)
    }

    /// Returns a slice over all solutions, ordered best-to-worst.
    #[inline]
    pub fn as_slice(&self) -> &[Solution<T>] {
        &self.solutions
    }

    /// Returns an iterator over all solutions, ordered best-to-worst.
    #[inline]
    pub fn iter(&self) -> std::slice::Iter<'_, Solution<T>> {
        self.solutions.iter()
    }

    /// Removes all solutions from the pool.
    #[inline]
    pub fn clear(&mut self) {
        self.solutions.clear();
    }

    /// Attempts to insert `solution` into the pool.
    ///
    /// Returns `true` if the solution was inserted, `false` if it was rejected
    /// (i.e. the pool is full and the solution is not better than the worst).
    pub fn try_push_solution(&mut self, solution: &Solution<T>) -> bool
    where
        T: Copy + Ord,
    {
        let obj = solution.objective_value();

        // If full, reject if not better than the worst.
        if self.is_full() {
            // SAFETY: pool is full ⇒ last() is Some.
            let worst_obj = self.solutions.last().unwrap().objective_value();
            if obj >= worst_obj {
                return false;
            }
            self.solutions.pop();
        }

        // Binary search for the insertion point (ascending order).
        let pos = self
            .solutions
            .binary_search_by(|s| s.objective_value().cmp(&obj))
            .unwrap_or_else(|e| e);

        self.solutions.insert(pos, solution.clone());
        true
    }

    pub fn try_push_solution_view(&mut self, solution: SolutionView<'_, T>) -> bool
    where
        T: Copy + Ord,
    {
        let obj = solution.objective_value();

        // If full, reject if not better than the worst.
        if self.is_full() {
            // SAFETY: pool is full ⇒ last() is Some.
            let worst_obj = self.solutions.last().unwrap().objective_value();
            if obj >= worst_obj {
                return false;
            }
            self.solutions.pop();
        }

        // Binary search for the insertion point (ascending order).
        let pos = self
            .solutions
            .binary_search_by(|s| s.objective_value().cmp(&obj))
            .unwrap_or_else(|e| e);

        self.solutions.insert(pos, solution.to_owned_solution());
        true
    }
}

impl<'a, T> IntoIterator for &'a SolutionPool<T> {
    type Item = &'a Solution<T>;
    type IntoIter = std::slice::Iter<'a, Solution<T>>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<T> IntoIterator for SolutionPool<T> {
    type Item = Solution<T>;
    type IntoIter = std::vec::IntoIter<Solution<T>>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.solutions.into_iter()
    }
}
