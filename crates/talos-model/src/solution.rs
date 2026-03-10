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

//! Solution representation for the Dynamic Berth Allocation Problem (DBAP).
//!
//! This module provides two related types:
//! - `Solution`: an owned solution (stores `Vec`s for berths and start times).
//! - `SolutionView`: a lightweight borrowed view into solution slices.
//!
//! The representation is **SoA (Structure of Arrays)** for cache locality:
//! - `berths[i]` and `start_times[i]` refer to the same vessel `VesselIndex(i)`.
//!
//! ## Invariants
//! - For both `Solution` and `SolutionView`, `berths.len() == start_times.len()`.
//! - Indices passed via `VesselIndex` must refer to `0..num_vessels()`.
//!
//! ## Complexity
//! - All getters are `O(1)`.
//! - Iteration yields assignments in vessel-index order with `O(n)` total time.
//!
//! ## Safety
//! This module provides `_unchecked` accessors for hot loops. These are only
//! sound if the caller guarantees that the provided indices are in bounds.
//!
//! ## Examples
//!
//! Creating and inspecting a solution:
//! ```rust
//! use talos_model::solution::Solution;
//! use talos_model::assignment::Assignment;
//! use talos_model::index::{BerthIndex, VesselIndex};
//!
//! let berths = vec![BerthIndex::new(0), BerthIndex::new(1)];
//! let starts = vec![10_i64, 20_i64];
//! let objective = 123_i64;
//!
//! let sol = Solution::new(berths, starts, objective);
//! assert_eq!(sol.num_vessels(), 2);
//!
//! let a0 = sol.assignment_for_vessel(VesselIndex::new(0));
//! assert_eq!(a0, Assignment::new(10, BerthIndex::new(0)));
//! ```
//!
//! Using a view without allocating:
//! ```rust
//! use talos_model::solution::SolutionView;
//! use talos_model::index::BerthIndex;
//!
//! let berths = [BerthIndex::new(0)];
//! let starts = [0_i64];
//! let view = SolutionView::new(&berths, &starts, 7);
//! assert_eq!(view.num_vessels(), 1);
//! assert_eq!(view.objective_value(), 7);
//! ```

use crate::{
    assignment::Assignment,
    index::{BerthIndex, VesselIndex},
};
use std::{hash::Hash, iter::FusedIterator};

// ----------------------------------------------------------------
// SolutionIter
// ----------------------------------------------------------------

/// Iterator over all vessel assignments in a `Solution`.
///
/// Yields `Assignment<T>` values in order of increasing vessel index.
#[derive(Debug, Clone)]
pub struct SolutionIter<'a, T> {
    berths: std::slice::Iter<'a, BerthIndex>,
    start_times: std::slice::Iter<'a, T>,
}

impl<'a, T> SolutionIter<'a, T> {
    #[inline]
    fn new(solution: &'a Solution<T>) -> Self {
        debug_assert_eq!(solution.berths.len(), solution.start_times.len());

        SolutionIter {
            berths: solution.berths.iter(),
            start_times: solution.start_times.iter(),
        }
    }
}

impl<'a, T> Iterator for SolutionIter<'a, T>
where
    T: Copy,
{
    type Item = Assignment<T>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let berth = self.berths.next()?;
        let start_time = self.start_times.next()?;

        Some(Assignment::new(*start_time, *berth))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        debug_assert_eq!(self.berths.len(), self.start_times.len());

        self.berths.size_hint()
    }
}

impl<'a, T> ExactSizeIterator for SolutionIter<'a, T> where T: Copy {}
impl<'a, T> FusedIterator for SolutionIter<'a, T> where T: Copy {}

// ----------------------------------------------------------------
// Solution
// ----------------------------------------------------------------

/// Represents a solution of the Dynamic Berth Allocation Problem (DBAP),
/// i.e., a complete schedule of all vessels to berths and start times.
///
/// The solution stores, for each vessel, its assigned berth and start time,
/// as well as the associated objective value of the schedule.
///
/// # Data Layout
///
/// This type uses a SoA (Structure of Arrays) layout:
/// - `berths[v]` is the berth assigned to vessel `v`
/// - `start_times[v]` is the start time assigned to vessel `v`
///
/// This improves cache locality when processing large problems vessel-by-vessel.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Solution<T> {
    // Invariant: berths.len() == start_times.len()
    berths: Vec<BerthIndex>, // len = num_vessels
    start_times: Vec<T>,     // len = num_vessels
    objective_value: T,
}

impl<T> std::fmt::Debug for Solution<T>
where
    T: std::fmt::Debug + Copy,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        // Wrapper to debug-print assignments without allocating.
        struct Assignments<'a, U>(&'a Solution<U>);

        impl<'a, U: std::fmt::Debug + Copy> std::fmt::Debug for Assignments<'a, U> {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.debug_list().entries(self.0.iter()).finish()
            }
        }

        f.debug_struct("Solution")
            .field("objective_value", &self.objective_value)
            .field("num_vessels", &self.num_vessels())
            .field("assignments", &Assignments(self))
            .finish()
    }
}

impl<T> std::fmt::Display for Solution<T>
where
    T: std::fmt::Display + Copy,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        writeln!(
            f,
            "Solution (Vessels: {}, Objective: {})",
            self.num_vessels(),
            self.objective_value
        )?;

        for (i, assignment) in self.iter().enumerate() {
            writeln!(f, "  Vessel {}: {}", i, assignment)?;
        }

        Ok(())
    }
}

impl<T> Solution<T> {
    /// Creates a new `Solution`.
    ///
    /// # Panics
    ///
    /// Panics if `berths` and `start_times` have different lengths.
    #[inline]
    pub fn new(berths: Vec<BerthIndex>, start_times: Vec<T>, objective_value: T) -> Self {
        assert_eq!(
            berths.len(),
            start_times.len(),
            "called `Solution::new` with mismatched lengths: berths = {}, start_times = {}",
            berths.len(),
            start_times.len()
        );

        Self {
            berths,
            start_times,
            objective_value,
        }
    }

    /// Overwrites this owned solution with data from a `SolutionView`.
    ///
    /// This method reuses existing allocations by clearing and extending the
    /// internal vectors.
    #[inline]
    pub fn overwrite_from_solution_view(&mut self, view: SolutionView<'_, T>)
    where
        T: Copy,
    {
        self.objective_value = view.objective_value();
        self.berths.clear();
        self.berths.extend_from_slice(view.berths());
        self.start_times.clear();
        self.start_times.extend_from_slice(view.start_times());
    }

    /// Constructs a `Solution` by copying from slices.
    ///
    /// # Panics
    ///
    /// Panics if `berths` and `start_times` have different lengths.
    #[inline]
    pub fn from_slices(berths: &[BerthIndex], start_times: &[T], objective_value: T) -> Self
    where
        T: Copy,
    {
        assert_eq!(
            berths.len(),
            start_times.len(),
            "called `Solution::from_slices` with mismatched lengths: berths = {}, start_times = {}",
            berths.len(),
            start_times.len()
        );

        Self {
            berths: berths.to_vec(),
            start_times: start_times.to_vec(),
            objective_value,
        }
    }

    /// Returns the number of vessels in this solution.
    #[inline]
    pub fn num_vessels(&self) -> usize {
        self.berths.len()
    }

    /// Returns the assignment for the given vessel.
    ///
    /// # Panics (debug only)
    ///
    /// Panics in debug builds if `vessel` is out of bounds.
    #[inline]
    pub fn assignment_for_vessel(&self, vessel: VesselIndex) -> Assignment<T>
    where
        T: Copy,
    {
        debug_assert!(vessel < self.berths.len());

        Assignment::new(self.start_times[vessel.get()], self.berths[vessel.get()])
    }

    /// Returns the assignment for a vessel without bounds checks.
    ///
    /// # Safety
    ///
    /// The caller must ensure `vessel < self.num_vessels()`.
    #[inline]
    pub unsafe fn assignment_for_vessel_unchecked(&self, vessel: VesselIndex) -> Assignment<T>
    where
        T: Copy,
    {
        debug_assert!(vessel < self.berths.len());

        unsafe {
            Assignment::new(
                *self.start_times.get_unchecked(vessel.get()),
                *self.berths.get_unchecked(vessel.get()),
            )
        }
    }

    /// Sets the assignment for a vessel.
    ///
    /// # Panics (debug only)
    ///
    /// Panics in debug builds if `vessel` is out of bounds.
    #[inline]
    pub fn set_assignment_for_vessel(&mut self, vessel: VesselIndex, assignment: Assignment<T>) {
        debug_assert!(vessel < self.berths.len());

        self.start_times[vessel.get()] = assignment.start_time;
        self.berths[vessel.get()] = assignment.berth;
    }

    /// Returns the berth assigned to the given vessel.
    ///
    /// # Panics (debug only)
    ///
    /// Panics in debug builds if `vessel` is out of bounds.
    #[inline]
    pub fn berth_for_vessel(&self, vessel: VesselIndex) -> BerthIndex {
        debug_assert!(vessel < self.berths.len());

        self.berths[vessel.get()]
    }

    /// Returns the berth assigned to a vessel without bounds checks.
    ///
    /// # Safety
    ///
    /// The caller must ensure `vessel < self.num_vessels()`.
    #[inline]
    pub unsafe fn berth_for_vessel_unchecked(&self, vessel: VesselIndex) -> BerthIndex {
        debug_assert!(vessel < self.berths.len());

        *unsafe { self.berths.get_unchecked(vessel.get()) }
    }

    /// Returns a reference to the start time for the given vessel.
    ///
    /// # Panics (debug only)
    ///
    /// Panics in debug builds if `vessel` is out of bounds.
    #[inline]
    pub fn start_time_for_vessel(&self, vessel: VesselIndex) -> &T {
        debug_assert!(vessel < self.start_times.len());

        &self.start_times[vessel.get()]
    }

    /// Returns the start time for a vessel without bounds checks.
    ///
    /// # Safety
    ///
    /// The caller must ensure `vessel < self.num_vessels()`.
    #[inline]
    pub unsafe fn start_time_for_vessel_unchecked(&self, vessel: VesselIndex) -> &T {
        debug_assert!(vessel < self.start_times.len());

        unsafe { self.start_times.get_unchecked(vessel.get()) }
    }

    /// Returns all start times (read-only).
    #[inline]
    pub fn start_times(&self) -> &[T] {
        &self.start_times
    }

    /// Returns all berths (read-only).
    #[inline]
    pub fn berths(&self) -> &[BerthIndex] {
        &self.berths
    }

    /// Returns the objective value.
    #[inline]
    pub fn objective_value(&self) -> T
    where
        T: Copy,
    {
        self.objective_value
    }

    /// Sets the objective value.
    #[inline]
    pub fn set_objective_value(&mut self, value: T) {
        self.objective_value = value;
    }

    /// Mutable access to the objective value.
    #[inline]
    pub fn objective_value_mut(&mut self) -> &mut T {
        &mut self.objective_value
    }

    /// Returns an iterator over all assignments in this solution.
    ///
    /// The iterator yields `Assignment` values for vessels `0..num_vessels()`
    /// in order of their vessel index.
    #[inline]
    pub fn iter(&self) -> SolutionIter<'_, T>
    where
        T: Copy,
    {
        SolutionIter::new(self)
    }

    /// Returns a borrowed view into this solution.
    #[inline]
    pub fn as_view(&self) -> SolutionView<'_, T>
    where
        T: Copy,
    {
        SolutionView::from_solution(self)
    }
}

impl<'a, T> IntoIterator for &'a Solution<T>
where
    T: Copy,
{
    type Item = Assignment<T>;
    type IntoIter = SolutionIter<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T> From<&'a Solution<T>> for SolutionView<'a, T>
where
    T: Copy,
{
    #[inline]
    fn from(val: &'a Solution<T>) -> Self {
        val.as_view()
    }
}

// ----------------------------------------------------------------
// SolutionViewIter
// ----------------------------------------------------------------

/// Iterator over assignments in a `SolutionView`.
#[derive(Debug, Clone)]
pub struct SolutionViewIter<'a, T> {
    berths: std::slice::Iter<'a, BerthIndex>,
    start_times: std::slice::Iter<'a, T>,
}

impl<'a, T> SolutionViewIter<'a, T> {
    #[inline]
    fn new(view: &'a SolutionView<T>) -> Self {
        debug_assert_eq!(view.berths.len(), view.start_times.len());

        SolutionViewIter {
            berths: view.berths.iter(),
            start_times: view.start_times.iter(),
        }
    }
}

impl<'a, T> Iterator for SolutionViewIter<'a, T>
where
    T: Copy,
{
    type Item = Assignment<T>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let berth = self.berths.next()?;
        let start_time = self.start_times.next()?;
        Some(Assignment::new(*start_time, *berth))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        debug_assert_eq!(self.berths.len(), self.start_times.len());

        self.berths.size_hint()
    }
}

impl<'a, T> ExactSizeIterator for SolutionViewIter<'a, T> where T: Copy {}
impl<'a, T> FusedIterator for SolutionViewIter<'a, T> where T: Copy {}

// ----------------------------------------------------------------
// SolutionView
// ----------------------------------------------------------------

/// A borrowed view of a solution.
///
/// This type does not allocate and is intended for hot-path access.
#[derive(Debug, Clone, Copy)]
pub struct SolutionView<'a, T> {
    berths: &'a [BerthIndex],
    start_times: &'a [T],
    objective_value: T,
}

impl<'a, T> SolutionView<'a, T> {
    /// Constructs a new `SolutionView`.
    ///
    /// # Panics
    ///
    /// Panics if `berths` and `start_times` have different lengths.
    #[inline]
    pub fn new(berths: &'a [BerthIndex], start_times: &'a [T], objective_value: T) -> Self {
        assert_eq!(
            berths.len(),
            start_times.len(),
            "violated invariant of `SolutionView`: berths and start_times must have the same length, but got berths.len() = {} and start_times.len() = {}",
            berths.len(),
            start_times.len()
        );

        Self {
            berths,
            start_times,
            objective_value,
        }
    }

    /// Constructs a view from an owned `Solution`.
    #[inline]
    pub fn from_solution(solution: &'a Solution<T>) -> Self
    where
        T: Copy,
    {
        Self {
            berths: solution.berths(),
            start_times: solution.start_times(),
            objective_value: solution.objective_value(),
        }
    }

    /// Number of vessels in this view.
    #[inline]
    pub fn num_vessels(&self) -> usize {
        self.berths.len()
    }

    /// Returns berth assignments slice.
    #[inline]
    pub fn berths(&self) -> &'a [BerthIndex] {
        self.berths
    }

    /// Returns the berth assigned to the given vessel.
    ///
    /// # Panics (debug only)
    ///
    /// Panics in debug builds if `vessel` is out of bounds.
    #[inline]
    pub fn berth_for_vessel(&self, vessel: VesselIndex) -> BerthIndex {
        debug_assert!(vessel < self.berths.len());

        self.berths[vessel.get()]
    }

    /// Returns the berth assigned to a vessel without bounds checks.
    ///
    /// # Safety
    ///
    /// The caller must ensure `vessel < self.num_vessels()`.
    #[inline]
    pub unsafe fn berth_for_vessel_unchecked(&self, vessel: VesselIndex) -> BerthIndex {
        debug_assert!(vessel < self.berths.len());

        *unsafe { self.berths.get_unchecked(vessel.get()) }
    }

    /// Returns start times slice.
    #[inline]
    pub fn start_times(&self) -> &'a [T] {
        self.start_times
    }

    /// Returns the start time for the given vessel.
    ///
    /// # Panics (debug only)
    ///
    /// Panics in debug builds if `vessel` is out of bounds.
    #[inline]
    pub fn start_time_for_vessel(&self, vessel: VesselIndex) -> &T {
        debug_assert!(vessel < self.start_times.len());

        &self.start_times[vessel.get()]
    }

    /// Returns the start time for a vessel without bounds checks.
    ///
    /// # Safety
    ///
    /// The caller must ensure `vessel < self.num_vessels()`.
    #[inline]
    pub unsafe fn start_time_for_vessel_unchecked(&self, vessel: VesselIndex) -> &T {
        debug_assert!(vessel < self.start_times.len());

        unsafe { self.start_times.get_unchecked(vessel.get()) }
    }

    /// Returns the objective value.
    #[inline]
    pub fn objective_value(&self) -> T
    where
        T: Copy,
    {
        self.objective_value
    }

    /// Creates an owned `Solution` by copying the underlying slices.
    #[inline]
    pub fn to_owned_solution(&self) -> Solution<T>
    where
        T: Copy,
    {
        Solution::new(
            self.berths.to_vec(),
            self.start_times.to_vec(),
            self.objective_value,
        )
    }

    /// Returns an iterator over all assignments in this view.
    #[inline]
    pub fn iter(&self) -> SolutionViewIter<'_, T>
    where
        T: Copy,
    {
        SolutionViewIter::new(self)
    }
}

impl<'a, T> IntoIterator for &'a SolutionView<'a, T>
where
    T: Copy,
{
    type Item = Assignment<T>;
    type IntoIter = SolutionViewIter<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T> From<SolutionView<'a, T>> for Solution<T>
where
    T: Copy,
{
    #[inline]
    fn from(val: SolutionView<'a, T>) -> Self {
        val.to_owned_solution()
    }
}

impl<'a, T> From<&SolutionView<'a, T>> for Solution<T>
where
    T: Copy,
{
    #[inline]
    fn from(val: &SolutionView<'a, T>) -> Self {
        val.to_owned_solution()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::panic::{AssertUnwindSafe, catch_unwind};

    fn b(i: usize) -> BerthIndex {
        BerthIndex::new(i)
    }

    fn v(i: usize) -> VesselIndex {
        VesselIndex::new(i)
    }

    #[test]
    fn test_solution_new_panics_on_mismatched_lengths() {
        let res = catch_unwind(AssertUnwindSafe(|| {
            let _ = Solution::new(vec![b(0)], vec![10_i64, 20_i64], 0_i64);
        }));
        assert!(res.is_err());
    }

    #[test]
    fn test_solution_from_slices_panics_on_mismatched_lengths() {
        let berths = [b(0)];
        let starts = [10_i64, 20_i64];
        let res = catch_unwind(AssertUnwindSafe(|| {
            let _ = Solution::from_slices(&berths, &starts, 0_i64);
        }));
        assert!(res.is_err());
    }

    #[test]
    fn test_solution_getters_and_setters() {
        let mut sol = Solution::new(vec![b(1), b(0)], vec![10_i64, 20_i64], 123_i64);

        assert_eq!(sol.num_vessels(), 2);
        assert_eq!(sol.berths(), &[b(1), b(0)]);
        assert_eq!(sol.start_times(), &[10, 20]);
        assert_eq!(sol.objective_value(), 123);

        sol.set_objective_value(7);
        assert_eq!(sol.objective_value(), 7);
        *sol.objective_value_mut() = 9;
        assert_eq!(sol.objective_value(), 9);

        let a0 = sol.assignment_for_vessel(v(0));
        assert_eq!(a0, Assignment::new(10, b(1)));

        sol.set_assignment_for_vessel(v(0), Assignment::new(111, b(0)));
        assert_eq!(sol.assignment_for_vessel(v(0)), Assignment::new(111, b(0)));
        assert_eq!(sol.berth_for_vessel(v(0)), b(0));
        assert_eq!(*sol.start_time_for_vessel(v(0)), 111);
    }

    #[test]
    fn test_solution_iterates_in_vessel_order() {
        let sol = Solution::new(vec![b(1), b(0)], vec![10_i64, 20_i64], 0_i64);
        let items: Vec<_> = sol.iter().collect();
        assert_eq!(
            items,
            vec![Assignment::new(10, b(1)), Assignment::new(20, b(0))]
        );
    }

    #[test]
    fn test_into_iterator_for_solution_ref() {
        let sol = Solution::new(vec![b(0)], vec![5_i64], 0_i64);
        let items: Vec<_> = (&sol).into_iter().collect();
        assert_eq!(items, vec![Assignment::new(5, b(0))]);
    }

    #[test]
    fn test_solution_view_new_panics_on_mismatched_lengths() {
        let berths = [b(0)];
        let starts = [10_i64, 20_i64];
        let res = catch_unwind(AssertUnwindSafe(|| {
            let _ = SolutionView::new(&berths, &starts, 0_i64);
        }));
        assert!(res.is_err());
    }

    #[test]
    fn test_solution_view_from_solution_and_to_owned_roundtrip() {
        let sol = Solution::new(vec![b(1), b(0)], vec![10_i64, 20_i64], 77_i64);
        let view = sol.as_view();

        assert_eq!(view.num_vessels(), 2);
        assert_eq!(view.berths(), sol.berths());
        assert_eq!(view.start_times(), sol.start_times());
        assert_eq!(view.objective_value(), 77);

        let owned = view.to_owned_solution();
        assert_eq!(owned, sol);
    }

    #[test]
    fn test_solution_view_iter() {
        let sol = Solution::new(vec![b(1), b(0)], vec![10_i64, 20_i64], 0_i64);
        let view = sol.as_view();
        let items: Vec<_> = view.iter().collect();

        assert_eq!(
            items,
            vec![Assignment::new(10, b(1)), Assignment::new(20, b(0))]
        );
    }

    #[test]
    fn test_into_iterator_for_solution_view_ref() {
        let sol = Solution::new(vec![b(0)], vec![5_i64], 0_i64);
        let view = sol.as_view();
        let items: Vec<_> = (&view).into_iter().collect();
        assert_eq!(items, vec![Assignment::new(5, b(0))]);
    }

    #[test]
    fn test_debug_and_display_do_not_panic() {
        let sol = Solution::new(vec![b(0), b(1)], vec![5_i64, 6_i64], 42_i64);

        let _ = format!("{:?}", sol);
        let _ = format!("{}", sol);
    }
}
