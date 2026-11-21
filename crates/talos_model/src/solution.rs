// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A schedule of every vessel, owned as a [`Solution`] or borrowed as a [`SolutionView`].

use crate::assignment::Assignment;
use crate::index::{BerthIndex, VesselIndex};
use std::fmt;
use std::iter::{FusedIterator, Zip};
use std::slice;
use typed_index_collections::{TiSlice, TiVec};

/// A schedule of every vessel: the berth each is served at, the time each service starts, and the objective value
/// of the schedule.
///
/// The berths and the start times are tables keyed by the vessel, one entry per vessel each. A solution does not
/// know its model: whether its berths are those of a model and its times feasible is what
/// [`Model::check_feasibility`](crate::Model::check_feasibility) answers.
/// [`SolutionView`] is the same schedule borrowed from tables another owner keeps.
///
/// # Examples
///
/// ```rust
/// use talos_model::{Assignment, BerthIndex, Solution, VesselIndex};
///
/// let solution = Solution::new(vec![BerthIndex::new(1), BerthIndex::new(0)], vec![10, 20], 30);
///
/// assert_eq!(solution.num_vessels(), 2);
/// assert_eq!(solution.assigned_berths()[VesselIndex::new(0)], BerthIndex::new(1));
/// assert_eq!(
///     solution.iter().last(),
///     Some(Assignment { start_time: 20, berth: BerthIndex::new(0) })
/// );
/// ```
// Invariant: `berths` and `start_times` have the same length, the number of vessels.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Solution<T> {
    berths: TiVec<VesselIndex, BerthIndex>,
    start_times: TiVec<VesselIndex, T>,
    objective_value: T,
}

impl<T> Solution<T> {
    /// The schedule that serves vessel `v` at `berths[v]` from `start_times[v]`, of objective value
    /// `objective_value`.
    ///
    /// # Panics
    ///
    /// Panics if `berths` and `start_times` differ in length.
    #[track_caller]
    pub fn new(berths: Vec<BerthIndex>, start_times: Vec<T>, objective_value: T) -> Self {
        match Self::try_new(berths, start_times, objective_value) {
            Ok(solution) => solution,
            Err(error) => panic!("called `Solution::new` with {error}"),
        }
    }

    /// The schedule that serves vessel `v` at `berths[v]` from `start_times[v]`, of objective value
    /// `objective_value`, or `Err` if the two tables are not one entry per vessel each.
    ///
    /// # Errors
    ///
    /// [`SolutionLengthMismatch`] if `berths` and `start_times` differ in length.
    pub fn try_new(
        berths: Vec<BerthIndex>,
        start_times: Vec<T>,
        objective_value: T,
    ) -> Result<Self, SolutionLengthMismatch> {
        check_lengths(&berths, &start_times)?;
        Ok(Self {
            berths: TiVec::from(berths),
            start_times: TiVec::from(start_times),
            objective_value,
        })
    }

    /// The number of vessels.
    #[inline]
    pub fn num_vessels(&self) -> usize {
        self.berths.len()
    }

    /// The berth each vessel is assigned to.
    #[inline]
    pub fn assigned_berths(&self) -> &TiSlice<VesselIndex, BerthIndex> {
        &self.berths
    }

    /// The time each vessel's service starts.
    #[inline]
    pub fn start_times(&self) -> &TiSlice<VesselIndex, T> {
        &self.start_times
    }

    /// The objective value of the schedule.
    #[inline]
    pub fn objective_value(&self) -> T
    where
        T: Copy,
    {
        self.objective_value
    }

    /// The assignment of each vessel, in the order of the vessels.
    #[inline]
    pub fn iter(&self) -> SolutionIter<'_, T>
    where
        T: Copy,
    {
        SolutionIter::new(&self.berths, &self.start_times)
    }

    /// The schedule, borrowed.
    #[inline]
    pub fn as_view(&self) -> SolutionView<'_, T>
    where
        T: Copy,
    {
        SolutionView {
            berths: &self.berths,
            start_times: &self.start_times,
            objective_value: self.objective_value,
        }
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
    fn from(solution: &'a Solution<T>) -> Self {
        solution.as_view()
    }
}

impl<T> From<SolutionView<'_, T>> for Solution<T>
where
    T: Copy,
{
    /// The schedule of `view`, copied into tables of its own.
    #[inline]
    fn from(view: SolutionView<'_, T>) -> Self {
        Self {
            berths: view.berths.to_owned(),
            start_times: view.start_times.to_owned(),
            objective_value: view.objective_value,
        }
    }
}

/// A schedule of every vessel, borrowed from tables another owner keeps.
///
/// It answers what a [`Solution`] answers and costs no allocation to make, which suits a solver handing out its
/// current schedule. [`Solution::from`] copies it into a schedule of its own.
///
/// # Examples
///
/// ```rust
/// use talos_model::{BerthIndex, Solution, SolutionView};
///
/// let berths = [BerthIndex::new(0)];
/// let start_times = [4];
/// let view = SolutionView::new(&berths, &start_times, 7);
///
/// assert_eq!(view.objective_value(), 7);
/// assert_eq!(Solution::from(view), Solution::new(vec![BerthIndex::new(0)], vec![4], 7));
/// ```
// Invariant: `berths` and `start_times` have the same length, the number of vessels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SolutionView<'a, T> {
    berths: &'a TiSlice<VesselIndex, BerthIndex>,
    start_times: &'a TiSlice<VesselIndex, T>,
    objective_value: T,
}

impl<'a, T> SolutionView<'a, T> {
    /// The schedule that serves vessel `v` at `berths[v]` from `start_times[v]`, of objective value
    /// `objective_value`.
    ///
    /// # Panics
    ///
    /// Panics if `berths` and `start_times` differ in length.
    #[inline]
    #[track_caller]
    pub fn new(berths: &'a [BerthIndex], start_times: &'a [T], objective_value: T) -> Self {
        match Self::try_new(berths, start_times, objective_value) {
            Ok(view) => view,
            Err(error) => panic!("called `SolutionView::new` with {error}"),
        }
    }

    /// The schedule that serves vessel `v` at `berths[v]` from `start_times[v]`, of objective value
    /// `objective_value`, or `Err` if the two tables are not one entry per vessel each.
    ///
    /// # Errors
    ///
    /// [`SolutionLengthMismatch`] if `berths` and `start_times` differ in length.
    #[inline]
    pub fn try_new(
        berths: &'a [BerthIndex],
        start_times: &'a [T],
        objective_value: T,
    ) -> Result<Self, SolutionLengthMismatch> {
        check_lengths(berths, start_times)?;
        Ok(Self {
            berths: TiSlice::from_ref(berths),
            start_times: TiSlice::from_ref(start_times),
            objective_value,
        })
    }

    /// The number of vessels.
    #[inline]
    pub fn num_vessels(&self) -> usize {
        self.berths.len()
    }

    /// The berth each vessel is assigned to.
    #[inline]
    pub fn assigned_berths(&self) -> &'a TiSlice<VesselIndex, BerthIndex> {
        self.berths
    }

    /// The time each vessel's service starts.
    #[inline]
    pub fn start_times(&self) -> &'a TiSlice<VesselIndex, T> {
        self.start_times
    }

    /// The objective value of the schedule.
    #[inline]
    pub fn objective_value(&self) -> T
    where
        T: Copy,
    {
        self.objective_value
    }

    /// The assignment of each vessel, in the order of the vessels.
    #[inline]
    pub fn iter(&self) -> SolutionIter<'a, T>
    where
        T: Copy,
    {
        SolutionIter::new(self.berths, self.start_times)
    }
}

impl<'a, T> IntoIterator for SolutionView<'a, T>
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

impl<'a, T> IntoIterator for &SolutionView<'a, T>
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

/// The assignment of each vessel of a schedule, in the order of the vessels.
#[derive(Clone, Debug)]
pub struct SolutionIter<'a, T> {
    entries: Zip<slice::Iter<'a, BerthIndex>, slice::Iter<'a, T>>,
}

impl<'a, T> SolutionIter<'a, T> {
    /// The walk over the entries of `berths` and `start_times`, which have one per vessel each.
    #[inline]
    fn new(
        berths: &'a TiSlice<VesselIndex, BerthIndex>,
        start_times: &'a TiSlice<VesselIndex, T>,
    ) -> Self {
        debug_assert_eq!(
            berths.len(),
            start_times.len(),
            "called `SolutionIter::new` with {} berths but {} start times",
            berths.len(),
            start_times.len()
        );
        Self {
            entries: berths.iter().zip(start_times.iter()),
        }
    }
}

impl<T> Iterator for SolutionIter<'_, T>
where
    T: Copy,
{
    type Item = Assignment<T>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.entries
            .next()
            .map(|(&berth, &start_time)| Assignment { start_time, berth })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.entries.size_hint()
    }
}

impl<T> DoubleEndedIterator for SolutionIter<'_, T>
where
    T: Copy,
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.entries
            .next_back()
            .map(|(&berth, &start_time)| Assignment { start_time, berth })
    }
}

impl<T> ExactSizeIterator for SolutionIter<'_, T> where T: Copy {}

impl<T> FusedIterator for SolutionIter<'_, T> where T: Copy {}

/// Refuses tables of berths and start times that are not one entry per vessel each.
///
/// # Errors
///
/// [`SolutionLengthMismatch`] if `berths` and `start_times` differ in length.
fn check_lengths<T>(
    berths: &[BerthIndex],
    start_times: &[T],
) -> Result<(), SolutionLengthMismatch> {
    if berths.len() == start_times.len() {
        Ok(())
    } else {
        Err(SolutionLengthMismatch {
            berths: berths.len(),
            start_times: start_times.len(),
        })
    }
}

/// Tables of berths and start times that make no schedule, because they differ in length.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SolutionLengthMismatch {
    /// The number of berths given.
    pub berths: usize,
    /// The number of start times given.
    pub start_times: usize,
}

impl fmt::Display for SolutionLengthMismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} berths but {} start times, where each vessel needs one of each",
            self.berths, self.start_times
        )
    }
}

impl std::error::Error for SolutionLengthMismatch {}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn berth(index: usize) -> BerthIndex {
        BerthIndex::new(index)
    }

    fn assignment(start_time: i64, berth_index: usize) -> Assignment<i64> {
        Assignment {
            start_time,
            berth: berth(berth_index),
        }
    }

    #[test]
    fn test_solution_answers_its_tables_by_vessel() {
        let solution = Solution::new(vec![berth(1), berth(0)], vec![10i64, 20], 123);
        let second = VesselIndex::new(1);
        assert_eq!(solution.num_vessels(), 2);
        assert_eq!(solution.assigned_berths()[second], berth(0));
        assert_eq!(solution.start_times()[second], 20);
        assert_eq!(solution.objective_value(), 123);
    }

    #[test]
    fn test_try_new_refuses_tables_of_different_lengths() {
        assert_eq!(
            Solution::try_new(vec![berth(0)], vec![10i64, 20], 0),
            Err(SolutionLengthMismatch {
                berths: 1,
                start_times: 2
            })
        );
        let berths = [berth(0), berth(1)];
        assert_eq!(
            SolutionView::try_new(&berths, &[10i64], 0),
            Err(SolutionLengthMismatch {
                berths: 2,
                start_times: 1
            })
        );
    }

    #[test]
    #[should_panic(expected = "called `Solution::new` with 1 berths but 2 start times")]
    fn test_new_panics_on_tables_of_different_lengths() {
        let _ = Solution::new(vec![berth(0)], vec![10i64, 20], 0);
    }

    #[test]
    #[should_panic(expected = "called `SolutionView::new` with 2 berths but 0 start times")]
    fn test_view_new_panics_on_tables_of_different_lengths() {
        let _ = SolutionView::<i64>::new(&[berth(0), berth(1)], &[], 0);
    }

    #[test]
    fn test_check_lengths_accepts_equal_lengths_only() {
        assert_eq!(check_lengths::<i64>(&[], &[]), Ok(()));
        assert_eq!(check_lengths(&[berth(0)], &[1i64]), Ok(()));
        assert!(check_lengths(&[berth(0)], &[1i64, 2]).is_err());
    }

    #[test]
    fn test_iter_yields_each_vessels_assignment_in_vessel_order() {
        let solution = Solution::new(vec![berth(1), berth(0)], vec![10i64, 20], 0);
        let assignments: Vec<_> = solution.iter().collect();
        assert_eq!(assignments, [assignment(10, 1), assignment(20, 0)]);
        assert_eq!(
            (&solution).into_iter().rev().collect::<Vec<_>>(),
            [assignment(20, 0), assignment(10, 1)]
        );
    }

    #[test]
    fn test_view_borrows_the_same_schedule_and_copies_back_to_it() {
        let solution = Solution::new(vec![berth(1), berth(0)], vec![10i64, 20], 77);
        let view = solution.as_view();
        assert_eq!(view.num_vessels(), 2);
        assert_eq!(view.assigned_berths(), solution.assigned_berths());
        assert_eq!(view.start_times(), solution.start_times());
        assert_eq!(view.objective_value(), 77);
        assert_eq!(
            view.iter().collect::<Vec<_>>(),
            solution.iter().collect::<Vec<_>>()
        );
        assert_eq!(Solution::from(view), solution);
    }

    #[test]
    fn test_view_over_slices_reads_them_by_vessel() {
        let (berths, start_times) = ([berth(2)], [5i64]);
        let view = SolutionView::new(&berths, &start_times, 9);
        assert_eq!(view.into_iter().collect::<Vec<_>>(), [assignment(5, 2)]);
        assert_eq!((&view).into_iter().len(), 1);
    }

    #[test]
    fn test_length_mismatch_states_both_lengths() {
        let error = SolutionLengthMismatch {
            berths: 3,
            start_times: 4,
        };
        assert_eq!(
            error.to_string(),
            "3 berths but 4 start times, where each vessel needs one of each"
        );
    }

    proptest! {
        #[test]
        fn test_iter_pairs_the_tables_entry_by_entry(
            entries in prop::collection::vec((0usize..10, any::<i64>()), 0..20),
        ) {
            let berths: Vec<BerthIndex> = entries.iter().map(|&(index, _)| berth(index)).collect();
            let start_times: Vec<i64> = entries.iter().map(|&(_, time)| time).collect();
            let solution = Solution::new(berths, start_times, 0);

            let expected: Vec<Assignment<i64>> =
                entries.iter().map(|&(index, time)| assignment(time, index)).collect();
            prop_assert_eq!(solution.iter().len(), expected.len());
            prop_assert_eq!(solution.iter().collect::<Vec<_>>(), expected.clone());
            let mut backwards: Vec<_> = solution.iter().rev().collect();
            backwards.reverse();
            prop_assert_eq!(backwards, expected);
        }
    }
}
