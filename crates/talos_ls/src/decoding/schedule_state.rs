// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A decoded schedule: where and when each vessel is served, and what each berth costs.

use crate::schedule::{ScheduleGraph, TouchedBerths};
use talos_core::SolverNumeric;
use talos_model::{BerthIndex, SolutionView, VesselIndex};
use typed_index_collections::TiVec;

/// The schedule a [`ScheduleGraph`] stands for once decoded: each vessel's berth and start, each berth's cost, and
/// the objective, their sum.
///
/// The engine keeps several of them (the accepted schedule, the candidate, the best found) and copies between them
/// in place, so that a run allocates only when it starts.
// Invariant: `assigned_berths` and `start_times` have the same length, the number of vessels. Unchecked reads by a
// vessel rest on it: a vessel below `num_vessels` indexes both columns.
#[derive(Debug, PartialEq, Eq)]
pub struct ScheduleState<T> {
    /// The berth each vessel is served at, per vessel.
    assigned_berths: TiVec<VesselIndex, BerthIndex>,
    /// The time each vessel's service starts, per vessel.
    start_times: TiVec<VesselIndex, T>,
    /// The sum of the costs of the vessels served at each berth, per berth.
    berth_costs: TiVec<BerthIndex, T>,
    /// The sum of the berths' costs.
    objective: T,
}

impl<T> Clone for ScheduleState<T>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        Self {
            assigned_berths: self.assigned_berths.clone(),
            start_times: self.start_times.clone(),
            berth_costs: self.berth_costs.clone(),
            objective: self.objective.clone(),
        }
    }

    // Written out so that overwriting a state reuses the buffers of its columns, which their own `clone_from`
    // would replace.
    fn clone_from(&mut self, source: &Self) {
        self.assigned_berths
            .raw
            .clone_from(&source.assigned_berths.raw);
        self.start_times.raw.clone_from(&source.start_times.raw);
        self.berth_costs.raw.clone_from(&source.berth_costs.raw);
        self.objective.clone_from(&source.objective);
    }
}

impl<T> std::fmt::Display for ScheduleState<T>
where
    T: SolverNumeric + std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "schedule of objective {}", self.objective)?;
        for (berth, cost) in self.berth_costs.iter_enumerated() {
            writeln!(f, "  {berth} costs {cost}")?;
        }
        for (vessel, (berth, start)) in self
            .assigned_berths
            .iter_enumerated()
            .zip(&self.start_times)
            .map(|((vessel, berth), start)| (vessel, (berth, start)))
        {
            writeln!(f, "  {vessel} at {berth} from {start}")?;
        }
        Ok(())
    }
}

impl<T> ScheduleState<T>
where
    T: SolverNumeric,
{
    /// The schedule that serves vessel `v` at `assigned_berths[v]` from `start_times[v]`, with the given berth
    /// costs and objective.
    ///
    /// # Panics
    ///
    /// Panics if `assigned_berths` and `start_times` differ in length.
    #[track_caller]
    pub fn new(
        assigned_berths: Vec<BerthIndex>,
        start_times: Vec<T>,
        berth_costs: Vec<T>,
        objective: T,
    ) -> Self {
        assert!(
            assigned_berths.len() == start_times.len(),
            "called `ScheduleState::new` with {} assigned berths and {} start times",
            assigned_berths.len(),
            start_times.len()
        );
        Self {
            assigned_berths: TiVec::from(assigned_berths),
            start_times: TiVec::from(start_times),
            berth_costs: TiVec::from(berth_costs),
            objective,
        }
    }

    /// Gives the state `num_vessels` vessels and `num_berths` berths, reusing its memory; what it held is no longer
    /// a schedule until it is overwritten.
    #[inline]
    pub fn resize(&mut self, num_vessels: usize, num_berths: usize) {
        self.assigned_berths.resize(num_vessels, BerthIndex::new(0));
        self.start_times.resize(num_vessels, T::ZERO);
        self.berth_costs.resize(num_berths, T::ZERO);
        self.objective = T::ZERO;
    }

    /// Makes the state a copy of `other`, reusing its memory.
    #[inline]
    pub fn overwrite_from_state(&mut self, other: &Self) {
        self.clone_from(other);
    }

    /// The number of vessels.
    #[inline]
    pub fn num_vessels(&self) -> usize {
        self.assigned_berths.len()
    }

    /// The number of berths.
    #[inline]
    pub fn num_berths(&self) -> usize {
        self.berth_costs.len()
    }

    /// The objective value, the sum of the berths' costs.
    #[inline]
    pub fn objective(&self) -> T {
        self.objective
    }

    /// Sets the objective value.
    #[inline]
    pub fn set_objective(&mut self, objective: T) {
        self.objective = objective;
    }

    /// The berth `vessel` is served at.
    ///
    /// # Panics
    ///
    /// Panics if `vessel` is not one of the state's vessels.
    #[inline]
    #[track_caller]
    pub fn vessel_berth(&self, vessel: VesselIndex) -> BerthIndex {
        self.assigned_berths[vessel]
    }

    /// The berth `vessel` is served at, for a vessel the caller knows to be the state's.
    ///
    /// # Safety
    ///
    /// `vessel` must be one of the state's vessels, the condition [`vessel_berth`](Self::vessel_berth) checks.
    #[inline]
    pub unsafe fn vessel_berth_unchecked(&self, vessel: VesselIndex) -> BerthIndex {
        debug_assert!(
            vessel.index() < self.num_vessels(),
            "called `ScheduleState::vessel_berth_unchecked` with {vessel}, outside a state of {} vessels",
            self.num_vessels()
        );
        // SAFETY: `vessel` is one of the state's vessels, the precondition of this function.
        *unsafe { self.assigned_berths.get_unchecked(vessel) }
    }

    /// The time `vessel`'s service starts.
    ///
    /// # Panics
    ///
    /// Panics if `vessel` is not one of the state's vessels.
    #[inline]
    #[track_caller]
    pub fn vessel_start(&self, vessel: VesselIndex) -> T {
        self.start_times[vessel]
    }

    /// The time `vessel`'s service starts, for a vessel the caller knows to be the state's.
    ///
    /// # Safety
    ///
    /// `vessel` must be one of the state's vessels, the condition [`vessel_start`](Self::vessel_start) checks.
    #[inline]
    pub unsafe fn vessel_start_unchecked(&self, vessel: VesselIndex) -> T {
        debug_assert!(
            vessel.index() < self.num_vessels(),
            "called `ScheduleState::vessel_start_unchecked` with {vessel}, outside a state of {} vessels",
            self.num_vessels()
        );
        // SAFETY: `vessel` is one of the state's vessels, the precondition of this function.
        *unsafe { self.start_times.get_unchecked(vessel) }
    }

    /// Serves `vessel` at `berth` from `start`.
    ///
    /// # Panics
    ///
    /// Panics if `vessel` is not one of the state's vessels.
    #[inline]
    #[track_caller]
    pub fn set_vessel(&mut self, vessel: VesselIndex, berth: BerthIndex, start: T) {
        self.assigned_berths[vessel] = berth;
        self.start_times[vessel] = start;
    }

    /// Serves `vessel` at `berth` from `start`, for a vessel the caller knows to be the state's.
    ///
    /// # Safety
    ///
    /// `vessel` must be one of the state's vessels, the condition [`set_vessel`](Self::set_vessel) checks.
    #[inline]
    pub unsafe fn set_vessel_unchecked(
        &mut self,
        vessel: VesselIndex,
        berth: BerthIndex,
        start: T,
    ) {
        debug_assert!(
            vessel.index() < self.num_vessels(),
            "called `ScheduleState::set_vessel_unchecked` with {vessel}, outside a state of {} vessels",
            self.num_vessels()
        );
        // SAFETY: `vessel` is one of the state's vessels, the precondition of this function, and so indexes both
        // vessel columns by the invariant of `ScheduleState`.
        unsafe {
            *self.assigned_berths.get_unchecked_mut(vessel) = berth;
            *self.start_times.get_unchecked_mut(vessel) = start;
        }
    }

    /// The cost of `berth`.
    ///
    /// # Panics
    ///
    /// Panics if `berth` is not one of the state's berths.
    #[inline]
    #[track_caller]
    pub fn berth_cost(&self, berth: BerthIndex) -> T {
        self.berth_costs[berth]
    }

    /// The cost of `berth`, for a berth the caller knows to be the state's.
    ///
    /// # Safety
    ///
    /// `berth` must be one of the state's berths, the condition [`berth_cost`](Self::berth_cost) checks.
    #[inline]
    pub unsafe fn berth_cost_unchecked(&self, berth: BerthIndex) -> T {
        debug_assert!(
            berth.index() < self.num_berths(),
            "called `ScheduleState::berth_cost_unchecked` with {berth}, outside a state of {} berths",
            self.num_berths()
        );
        // SAFETY: `berth` is one of the state's berths, the precondition of this function.
        *unsafe { self.berth_costs.get_unchecked(berth) }
    }

    /// Sets the cost of `berth`.
    ///
    /// # Panics
    ///
    /// Panics if `berth` is not one of the state's berths.
    #[inline]
    #[track_caller]
    pub fn set_berth_cost(&mut self, berth: BerthIndex, cost: T) {
        self.berth_costs[berth] = cost;
    }

    /// Sets the cost of `berth`, for a berth the caller knows to be the state's.
    ///
    /// # Safety
    ///
    /// `berth` must be one of the state's berths, the condition [`set_berth_cost`](Self::set_berth_cost) checks.
    #[inline]
    pub unsafe fn set_berth_cost_unchecked(&mut self, berth: BerthIndex, cost: T) {
        debug_assert!(
            berth.index() < self.num_berths(),
            "called `ScheduleState::set_berth_cost_unchecked` with {berth}, outside a state of {} berths",
            self.num_berths()
        );
        // SAFETY: `berth` is one of the state's berths, the precondition of this function.
        *unsafe { self.berth_costs.get_unchecked_mut(berth) } = cost;
    }

    /// The schedule as a borrowed solution.
    #[inline]
    pub fn as_solution_view(&self) -> SolutionView<'_, T> {
        SolutionView::new(
            &self.assigned_berths.raw,
            &self.start_times.raw,
            self.objective,
        )
    }

    /// Takes from `candidate` the vessels and costs of every berth `touched` marks, and its objective.
    ///
    /// Afterwards the state is `candidate` on the touched berths and as it was on the others: each vessel of a
    /// touched berth of `graph` is served at that berth from its start in `candidate`. The cost is the number of
    /// touched berths plus the vessels on them.
    ///
    /// # Safety
    ///
    /// The state, `candidate` and `touched` must have the vessels and berths of `graph`: each must count as many
    /// vessels as `graph` does, where it counts vessels, and as many berths.
    #[inline]
    pub unsafe fn patch_from_delta_unchecked(
        &mut self,
        candidate: &Self,
        touched: &TouchedBerths,
        graph: &ScheduleGraph,
    ) {
        debug_assert!(
            self.num_vessels() == graph.num_vessels()
                && candidate.num_vessels() == graph.num_vessels()
                && self.num_berths() == graph.num_berths()
                && candidate.num_berths() == graph.num_berths()
                && touched.num_berths() == graph.num_berths(),
            "called `ScheduleState::patch_from_delta_unchecked` with states, touched berths and a graph of \
             different counts"
        );
        for berth in touched.iter_touched_berths() {
            // SAFETY: every berth `touched` yields is below its count (`TouchedBerths`), which is the states', a
            // precondition of this function.
            unsafe { self.set_berth_cost_unchecked(berth, candidate.berth_cost_unchecked(berth)) };
            // SAFETY: every berth `touched` yields is below its count (`TouchedBerths`), which is the graph's, a
            // precondition of this function.
            let sequence = unsafe { graph.vessel_sequence_iter_unchecked(berth) };
            for vessel in sequence {
                // SAFETY: the walk yields only vessels of the graph (`VesselSequenceIter`), which are the states',
                // a precondition of this function.
                unsafe {
                    let start = candidate.vessel_start_unchecked(vessel);
                    self.set_vessel_unchecked(vessel, berth, start);
                }
            }
        }
        self.objective = candidate.objective;
    }
}

impl<'a, T> From<&'a ScheduleState<T>> for SolutionView<'a, T>
where
    T: SolverNumeric,
{
    #[inline]
    fn from(state: &'a ScheduleState<T>) -> Self {
        state.as_solution_view()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn berth(index: usize) -> BerthIndex {
        BerthIndex::new(index)
    }

    fn vessel(index: usize) -> VesselIndex {
        VesselIndex::new(index)
    }

    /// V0 at B0 from 10, V1 at B1 from 20, V2 at B0 from 30; B0 costs 100, B1 200.
    fn three_vessels() -> ScheduleState<i64> {
        ScheduleState::new(
            vec![berth(0), berth(1), berth(0)],
            vec![10, 20, 30],
            vec![100, 200],
            300,
        )
    }

    #[test]
    fn test_new_counts_its_vessels_and_berths() {
        let state = three_vessels();
        assert_eq!(state.num_vessels(), 3);
        assert_eq!(state.num_berths(), 2);
        assert_eq!(state.objective(), 300);
    }

    #[test]
    #[should_panic(
        expected = "called `ScheduleState::new` with 1 assigned berths and 2 start times"
    )]
    fn test_new_refuses_columns_of_different_lengths() {
        let _ = ScheduleState::new(vec![berth(0)], vec![10, 20], vec![100], 300);
    }

    #[test]
    fn test_set_vessel_changes_only_that_vessel() {
        let mut state = three_vessels();
        state.set_vessel(vessel(1), berth(0), 25);
        assert_eq!(state.vessel_berth(vessel(1)), berth(0));
        assert_eq!(state.vessel_start(vessel(1)), 25);
        assert_eq!(state.vessel_start(vessel(0)), 10);
        assert_eq!(state.vessel_berth(vessel(2)), berth(0));
    }

    #[test]
    fn test_set_berth_cost_changes_only_that_berth() {
        let mut state = three_vessels();
        state.set_berth_cost(berth(0), 150);
        assert_eq!(state.berth_cost(berth(0)), 150);
        assert_eq!(state.berth_cost(berth(1)), 200);
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn test_vessel_start_refuses_a_vessel_outside() {
        let _ = three_vessels().vessel_start(vessel(3));
    }

    #[test]
    fn test_overwrite_from_state_makes_an_equal_state() {
        let mut state = three_vessels();
        let other = ScheduleState::new(vec![berth(2); 4], vec![1, 2, 3, 4], vec![0, 0, 9], 9);
        state.overwrite_from_state(&other);
        assert_eq!(state, other);
    }

    // The target counts more vessels and berths than the source, so its buffers hold the copy without growing.
    #[test]
    fn test_overwrite_from_state_reuses_the_buffers() {
        let source = three_vessels();
        let mut state = ScheduleState::new(vec![berth(0); 5], vec![0; 5], vec![0; 4], 0);
        let buffers = (
            state.assigned_berths.as_ptr(),
            state.start_times.as_ptr(),
            state.berth_costs.as_ptr(),
        );
        state.overwrite_from_state(&source);
        assert_eq!(state, source);
        assert_eq!(
            (
                state.assigned_berths.as_ptr(),
                state.start_times.as_ptr(),
                state.berth_costs.as_ptr()
            ),
            buffers
        );
    }

    #[test]
    fn test_solution_view_shows_the_columns() {
        let state = three_vessels();
        let view = state.as_solution_view();
        assert_eq!(view.assigned_berths().raw, [berth(0), berth(1), berth(0)]);
        assert_eq!(view.start_times().raw, [10, 20, 30]);
        assert_eq!(view.objective_value(), 300);
    }

    // The candidate moved V2 from B0 to B1, so both berths are touched; V0 keeps its berth but its start in the
    // candidate is taken too, as it stands on a touched berth.
    #[test]
    fn test_patch_from_delta_takes_the_touched_berths_of_the_candidate() {
        let graph = ScheduleGraph::from_slices(&[berth(0), berth(1), berth(1)], &[0, 0, 1], 3);
        let mut accepted = ScheduleState::new(
            vec![berth(0), berth(1), berth(0)],
            vec![10, 20, 30],
            vec![100, 200, 7],
            307,
        );
        let candidate = ScheduleState::new(
            vec![berth(0), berth(1), berth(1)],
            vec![11, 21, 31],
            vec![40, 50, 99],
            97,
        );
        let mut touched = TouchedBerths::new(3);
        touched.touch(berth(0));
        touched.touch(berth(1));
        // SAFETY: the states, the touched berths and the graph all have three vessels and three berths.
        unsafe { accepted.patch_from_delta_unchecked(&candidate, &touched, &graph) };
        assert_eq!(accepted.vessel_berth(vessel(2)), berth(1));
        assert_eq!(accepted.vessel_start(vessel(2)), 31);
        assert_eq!(accepted.vessel_start(vessel(0)), 11);
        assert_eq!(accepted.berth_cost(berth(0)), 40);
        assert_eq!(accepted.berth_cost(berth(1)), 50);
        assert_eq!(accepted.berth_cost(berth(2)), 7);
        assert_eq!(accepted.objective(), 97);
    }

    proptest! {
        #[test]
        fn test_checked_reads_and_writes_match_their_unchecked_twins(
            starts in proptest::collection::vec(-1000_i64..1000, 1..12),
            pick in any::<proptest::sample::Index>(),
            start in -1000_i64..1000,
        ) {
            let num_vessels = starts.len();
            let berths: Vec<_> = (0..num_vessels).map(|index| berth(index % 3)).collect();
            let costs = vec![1, 2, 3];
            let mut checked = ScheduleState::new(berths.clone(), starts.clone(), costs.clone(), 6);
            let mut unchecked = checked.clone();
            let chosen = vessel(pick.index(num_vessels));
            let chosen_berth = berth(pick.index(3));
            checked.set_vessel(chosen, chosen_berth, start);
            checked.set_berth_cost(chosen_berth, start);
            // SAFETY: `chosen` is below the vessel count and `chosen_berth` below the berth count of 3.
            unsafe {
                unchecked.set_vessel_unchecked(chosen, chosen_berth, start);
                unchecked.set_berth_cost_unchecked(chosen_berth, start);
            }
            prop_assert_eq!(&checked, &unchecked);
            for index in 0..num_vessels {
                // SAFETY: `index` is below the vessel count.
                unsafe {
                    prop_assert_eq!(checked.vessel_start(vessel(index)), unchecked.vessel_start_unchecked(vessel(index)));
                    prop_assert_eq!(checked.vessel_berth(vessel(index)), unchecked.vessel_berth_unchecked(vessel(index)));
                }
            }
            // SAFETY: `chosen_berth` is below the berth count of 3.
            prop_assert_eq!(checked.berth_cost(chosen_berth), unsafe { unchecked.berth_cost_unchecked(chosen_berth) });
            let outside = vessel(num_vessels);
            prop_assert!(std::panic::catch_unwind(|| checked.vessel_start(outside)).is_err());
        }
    }
}
