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

//! Phenotype representation for the local-search scheduling loop.
//!
//! While `ScheduleGraph` encodes the *topology* (genotype) — which vessel
//! follows which on each berth — `ScheduleState` captures the *evaluated
//! result* (phenotype): the concrete berth assignment, start time, sequence
//! position, per-berth cost, and global objective for every vessel.

use crate::{sgraph::ScheduleGraph, tberth::TouchedBerths};
use talos_core::utils::num::SolverNumeric;
use talos_model::{
    assignment::Assignment,
    index::{BerthIndex, VesselIndex},
    solution::SolutionView,
};

/// An encapsulated "Phenotype" of the scheduling problem.
///
/// Groups all the loosely related parallel arrays into a single cohesive
/// memory block. This dramatically improves cache locality. The `positions`
/// array here is critical: it maintains $O(1)$ lookup speeds for the `Mutator`
/// intent tracker without polluting the hot paths of `ScheduleGraph`.
#[derive(Debug, Clone)]
pub struct ScheduleState<T> {
    berths: Vec<BerthIndex>, // len = num_vessels
    starts: Vec<T>,          // len = num_vessels
    positions: Vec<usize>,   // len = num_vessels
    costs: Vec<T>,           // len = num_berths
    objective: T,            // The objective for this state
}

impl<T: SolverNumeric> PartialEq for ScheduleState<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        if self.objective != other.objective {
            return false;
        }

        self.berths == other.berths
            && self.starts == other.starts
            && self.positions == other.positions
            && self.costs == other.costs
    }
}

impl<T: SolverNumeric> Eq for ScheduleState<T> {}

impl<T> std::fmt::Display for ScheduleState<T>
where
    T: SolverNumeric + std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "ScheduleState [Objective: {}]", self.objective)?;

        write!(f, "Berth Costs: [")?;
        for (i, cost) in self.costs.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "B{}: {}", i, cost)?;
        }
        writeln!(f, "]")?;

        for i in 0..self.berths.len() {
            writeln!(
                f,
                "  V{:<3} -> B{:<2} | start: {:<5} | pos: {:<3}",
                i,
                self.berths[i].get(),
                self.starts[i],
                self.positions[i],
            )?;
        }

        Ok(())
    }
}

impl<T: SolverNumeric> ScheduleState<T> {
    /// Creates a new `ScheduleState` from the given vectors.
    ///
    /// # Panics
    ///
    /// Panics if the lengths of `berths`, `starts`, and `positions` do not match.
    pub fn new(
        berths: Vec<BerthIndex>,
        starts: Vec<T>,
        positions: Vec<usize>,
        costs: Vec<T>,
        objective: T,
    ) -> Self {
        assert!(
            berths.len() == starts.len() && berths.len() == positions.len(),
            "called `ScheduleState::new` with mismatched vector lengths: berths.len() = {}, starts.len() = {}, positions.len() = {}",
            berths.len(),
            starts.len(),
            positions.len()
        );

        Self {
            berths,
            starts,
            positions,
            costs,
            objective,
        }
    }
    /// Creates a new `ScheduleState` from the given slices.
    ///
    /// # Panics
    ///
    /// Panics if the lengths of `berths`, `starts`, and `positions` do not match.
    #[inline]
    pub fn from_slices(
        berths: &[BerthIndex],
        starts: &[T],
        positions: &[usize],
        costs: &[T],
        objective: T,
    ) -> Self {
        assert!(
            berths.len() == starts.len() && berths.len() == positions.len(),
            "called `ScheduleState::from_slices` with mismatched slice lengths: berths.len() = {}, starts.len() = {}, positions.len() = {}",
            berths.len(),
            starts.len(),
            positions.len()
        );

        Self {
            berths: berths.to_vec(),
            starts: starts.to_vec(),
            positions: positions.to_vec(),
            costs: costs.to_vec(),
            objective,
        }
    }

    /// Overwrites the current state with new data, reusing existing heap allocations.
    ///
    /// This method is designed for high-performance state resets. By calling `.clear()`
    /// and `.extend()` on the internal vectors, it attempts to reuse the capacity of
    /// previously allocated memory buffers to minimize heap pressure during
    /// high-frequency local search restarts.
    ///
    /// # Panics
    ///
    /// Panics if the lengths of the provided `berths`, `starts`, and `positions`
    /// vectors do not match.
    #[inline]
    pub fn overwrite_from_slices(
        &mut self,
        berths: &[BerthIndex],
        starts: &[T],
        positions: &[usize],
        costs: &[T],
        objective: T,
    ) {
        assert!(
            berths.len() == starts.len() && berths.len() == positions.len(),
            "called `ScheduleState::overwrite_from_slices` with mismatched slice lengths: berths.len() = {}, starts.len() = {}, positions.len() = {}",
            berths.len(),
            starts.len(),
            positions.len()
        );

        self.berths.clear();
        self.berths.extend(berths);

        self.starts.clear();
        self.starts.extend(starts);

        self.positions.clear();
        self.positions.extend(positions);

        self.costs.clear();
        self.costs.extend(costs);

        self.objective = objective;
    }

    /// Overwrites the current state with data from another `ScheduleState`, reusing existing heap allocations.
    #[inline]
    pub fn overwrite_from_state(&mut self, other: &Self) {
        self.overwrite_from_slices(
            &other.berths,
            &other.starts,
            &other.positions,
            &other.costs,
            other.objective,
        );
    }

    /// Returns the number of vessels represented in this state.
    #[inline]
    pub fn num_vessels(&self) -> usize {
        self.berths.len()
    }

    /// Returns the number of berths represented in this state.
    #[inline]
    pub fn num_berths(&self) -> usize {
        self.costs.len()
    }

    /// Returns the objective value of this state.
    #[inline]
    pub fn objective(&self) -> T {
        self.objective
    }

    /// Updates the objective value of this state.
    #[inline]
    pub fn set_objective(&mut self, objective: T) {
        self.objective = objective;
    }

    /// Set the berth, start time, and position for a specific vessel.
    ///
    /// # Panics
    ///
    /// Panics if the `vessel` index is out of bounds,
    /// meaning `vessel >= self.num_vessels()`.
    #[inline]
    pub fn set_vessel(
        &mut self,
        vessel: VesselIndex,
        berth: BerthIndex,
        start: T,
        position: usize,
    ) {
        debug_assert!(vessel < self.berths.len());

        let v_idx = vessel.get();
        self.berths[v_idx] = berth;
        self.starts[v_idx] = start;
        self.positions[v_idx] = position;
    }

    /// Set the berth, start time, and position for a specific vessel without any bounds checks.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the `vessel` is in bounds,
    /// meaning `vessel < self.num_vessels()`.
    #[inline]
    pub unsafe fn set_vessel_unchecked(
        &mut self,
        vessel: VesselIndex,
        berth: BerthIndex,
        start: T,
        position: usize,
    ) {
        debug_assert!(vessel < self.berths.len());

        let v_idx = vessel.get();
        *unsafe { self.berths.get_unchecked_mut(v_idx) } = berth;
        *unsafe { self.starts.get_unchecked_mut(v_idx) } = start;
        *unsafe { self.positions.get_unchecked_mut(v_idx) } = position;
    }

    /// Returns the berth assigned to a specific vessel.
    ///
    /// # Panics
    ///
    /// Panics if the `vessel` index is out of bounds,
    /// meaning `vessel >= self.num_vessels()`.
    #[inline]
    pub fn vessel_berth(&self, vessel: VesselIndex) -> BerthIndex {
        debug_assert!(vessel < self.berths.len());

        self.berths[vessel.get()]
    }

    /// Returns the berth assigned to a specific vessel without any bounds checks.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the `vessel` is in bounds,
    /// meaning `vessel < self.num_vessels()`.
    #[inline]
    pub unsafe fn vessel_berth_unchecked(&self, vessel: VesselIndex) -> BerthIndex {
        debug_assert!(vessel < self.berths.len());

        *unsafe { self.berths.get_unchecked(vessel.get()) }
    }

    /// Returns the start time assigned to a specific vessel.
    ///
    /// # Panics
    ///
    /// Panics if the `vessel` index is out of bounds,
    /// meaning `vessel >= self.num_vessels()`.
    #[inline]
    pub fn vessel_start(&self, vessel: VesselIndex) -> T {
        debug_assert!(vessel < self.starts.len());

        self.starts[vessel.get()]
    }

    /// Returns the start time assigned to a specific vessel without any bounds checks.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the `vessel` is in bounds,
    /// meaning `vessel < self.num_vessels()`.
    #[inline]
    pub unsafe fn vessel_start_unchecked(&self, vessel: VesselIndex) -> T {
        debug_assert!(vessel < self.starts.len());

        *unsafe { self.starts.get_unchecked(vessel.get()) }
    }

    /// Returns the cost of the given berth.
    ///
    /// # Panics
    ///
    /// Panics if the `berth` index is out of bounds,
    /// meaning `berth >= self.num_berths()`.
    #[inline]
    pub fn berth_cost(&self, berth: BerthIndex) -> T {
        debug_assert!(berth < self.costs.len());

        self.costs[berth.get()]
    }

    /// Returns the cost of the given berth without any bounds checks.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the `berth` is in bounds,
    /// meaning `berth < self.num_berths()`.
    #[inline]
    pub unsafe fn berth_cost_unchecked(&self, berth: BerthIndex) -> T {
        debug_assert!(berth < self.costs.len());

        *unsafe { self.costs.get_unchecked(berth.get()) }
    }

    /// Sets the cost of the given berth.
    ///
    /// # Panics
    ///
    /// Panics if the `berth` index is out of bounds,
    /// meaning `berth >= self.num_berths()`.
    #[inline]
    pub fn set_berth_cost(&mut self, berth: BerthIndex, cost: T) {
        debug_assert!(berth < self.costs.len());

        self.costs[berth.get()] = cost;
    }

    /// Sets the cost of the given berth without any bounds checks.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the `berth` is in bounds,
    /// meaning `berth < self.num_berths()`.
    #[inline]
    pub unsafe fn set_berth_cost_unchecked(&mut self, berth: BerthIndex, cost: T) {
        debug_assert!(berth < self.costs.len());

        *unsafe { self.costs.get_unchecked_mut(berth.get()) } = cost;
    }

    /// Returns the position of the given vessel in its berth's sequence.
    ///
    /// # Panics
    ///
    /// Panics if the `vessel` index is out of bounds,
    /// meaning `vessel >= self.num_vessels()`.
    #[inline]
    pub fn vessel_position(&self, vessel: VesselIndex) -> usize {
        debug_assert!(vessel < self.positions.len());

        self.positions[vessel.get()]
    }

    /// Returns the position of the given vessel in its berth's sequence without any bounds checks.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the `vessel` is in bounds,
    /// meaning `vessel < self.num_vessels()`.
    #[inline]
    pub unsafe fn vessel_position_unchecked(&self, vessel: VesselIndex) -> usize {
        debug_assert!(vessel < self.positions.len());

        *unsafe { self.positions.get_unchecked(vessel.get()) }
    }

    /// Returns an `Assignment` for the given vessel, which includes both the berth and start time.
    ///
    /// # Panics
    /// Panics if the `vessel` index is out of bounds,
    /// meaning `vessel >= self.num_vessels()`.
    #[inline]
    pub fn vessel_assignment(&self, vessel: VesselIndex) -> Assignment<T> {
        debug_assert!(vessel < self.berths.len());

        Assignment::new(self.vessel_start(vessel), self.vessel_berth(vessel))
    }

    /// Returns an `Assignment` for the given vessel without any bounds checks.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the `vessel` is in bounds,
    /// meaning `vessel < self.num_vessels()`.
    #[inline]
    pub unsafe fn vessel_assignment_unchecked(&self, vessel: VesselIndex) -> Assignment<T> {
        debug_assert!(vessel < self.berths.len());

        Assignment::new(unsafe { self.vessel_start_unchecked(vessel) }, unsafe {
            self.vessel_berth_unchecked(vessel)
        })
    }

    /// Creates a `SolutionView` that borrows from this state.
    #[inline]
    pub fn as_solution_view(&self) -> SolutionView<'_, T> {
        SolutionView::new(&self.berths, &self.starts, self.objective)
    }

    /// Incrementally updates this baseline state using data from a newly evaluated candidate state.
    ///
    /// Instead of performing a full array copy (which is expensive and causes cache thrashing),
    /// this method acts as a highly targeted "commit". It only synchronizes the physical pathways
    /// (berths and their assigned vessels) that were altered by a neighborhood move, as indicated
    /// by the `touched` array.
    ///
    /// It walks the pure-topology `graph` for the dirtied berths, instantly updating the `$O(1)$`
    /// lookup mappings (`berth`, `start`, and `position`) for the affected vessels, along with
    /// the localized berth `cost`. Finally, it updates the global `objective` score.
    ///
    /// # Safety
    ///
    /// Calling this method is highly `unsafe` as it entirely bypasses bounds checking in the hot loop.
    /// To avoid Undefined Behavior, the caller must strictly guarantee the following dimensional alignments:
    ///
    /// 1. **Berth Dimensions**: `touched.len()`, `self.costs.len()`, and `cand.costs.len()` must all be
    ///    equal to or greater than `graph.num_berths()`.
    /// 2. **Vessel Dimensions**: `self.berths.len()`, `self.starts.len()`, and `self.positions.len()`
    ///    (as well as their equivalents in `cand`) must all be equal to or greater than `graph.num_vessels()`.
    /// 3. **Graph Integrity**: The `graph` must be structurally valid and must only yield vessel indices
    ///    that fit within the bounds of the state arrays.
    #[inline]
    pub unsafe fn patch_from_delta_unchecked(
        &mut self,
        cand: &Self,
        touched: &TouchedBerths,
        graph: &ScheduleGraph,
    ) {
        debug_assert_eq!(self.costs.len(), cand.costs.len());
        debug_assert_eq!(self.costs.len(), graph.num_berths());
        debug_assert!(graph.num_vessels() <= self.berths.len());

        for berth_idx in touched.iter_touched_berths() {
            let b = berth_idx.get();

            unsafe {
                *self.costs.get_unchecked_mut(b) = *cand.costs.get_unchecked(b);
            }

            let sequence = unsafe { graph.vessel_sequence_iter_unchecked(berth_idx) };
            for (pos, vessel) in sequence.enumerate() {
                let v_idx = vessel.get();
                unsafe {
                    *self.berths.get_unchecked_mut(v_idx) = berth_idx;
                    *self.starts.get_unchecked_mut(v_idx) = *cand.starts.get_unchecked(v_idx);
                    *self.positions.get_unchecked_mut(v_idx) = pos;
                }
            }
        }

        self.objective = cand.objective;
    }
}

impl<'a, T> From<&'a ScheduleState<T>> for SolutionView<'a, T>
where
    T: SolverNumeric,
{
    #[inline]
    fn from(val: &'a ScheduleState<T>) -> Self {
        val.as_solution_view()
    }
}

#[cfg(test)]
mod test_schedule_state {
    use super::*;
    use talos_model::index::{BerthIndex, VesselIndex};

    // Helper function to create a basic state for testing
    fn create_default_state() -> ScheduleState<i32> {
        ScheduleState::new(
            vec![BerthIndex::new(0), BerthIndex::new(1), BerthIndex::new(0)], // berths
            vec![10, 20, 30],                                                 // starts
            vec![0, 0, 1],                                                    // positions
            vec![100, 200],                                                   // costs
            300,                                                              // objective
        )
    }

    #[test]
    fn test_initialization_and_dimensions() {
        let state = create_default_state();

        assert_eq!(state.num_vessels(), 3);
        assert_eq!(state.num_berths(), 2);
        assert_eq!(state.objective(), 300);
    }

    #[test]
    #[should_panic(expected = "mismatched vector lengths")]
    fn test_new_panics_on_length_mismatch() {
        ScheduleState::new(
            vec![BerthIndex::new(0)],
            vec![10, 20], // Mismatched length
            vec![0],
            vec![100],
            300,
        );
    }

    #[test]
    fn test_vessel_getters_and_setters() {
        let mut state = create_default_state();
        let v1 = VesselIndex::new(1);

        // Check initial values
        assert_eq!(state.vessel_berth(v1), BerthIndex::new(1));
        assert_eq!(state.vessel_start(v1), 20);
        assert_eq!(state.vessel_position(v1), 0);

        // Update vessel
        state.set_vessel(v1, BerthIndex::new(2), 25, 1);

        // Verify updates
        assert_eq!(state.vessel_berth(v1), BerthIndex::new(2));
        assert_eq!(state.vessel_start(v1), 25);
        assert_eq!(state.vessel_position(v1), 1);

        // Check Assignment creation
        let assignment = state.vessel_assignment(v1);
        assert_eq!(assignment.berth, BerthIndex::new(2));
        assert_eq!(assignment.start_time, 25);
    }

    #[test]
    fn test_berth_cost_getters_and_setters() {
        let mut state = create_default_state();
        let b0 = BerthIndex::new(0);

        assert_eq!(state.berth_cost(b0), 100);

        state.set_berth_cost(b0, 150);
        assert_eq!(state.berth_cost(b0), 150);
    }

    #[test]
    fn test_overwrite_from_slices() {
        let mut state = create_default_state();

        let new_berths = [BerthIndex::new(2), BerthIndex::new(2), BerthIndex::new(2)];
        let new_starts = [5, 5, 5];
        let new_positions = [0, 1, 2];
        let new_costs = [0, 0, 500];
        let new_objective = 500;

        // Overwrite existing buffers
        state.overwrite_from_slices(
            &new_berths,
            &new_starts,
            &new_positions,
            &new_costs,
            new_objective,
        );

        assert_eq!(state.objective(), 500);
        assert_eq!(state.num_berths(), 3);
        assert_eq!(state.vessel_start(VesselIndex::new(0)), 5);
        assert_eq!(state.berth_cost(BerthIndex::new(2)), 500);
    }

    #[test]
    fn test_overwrite_from_state_and_equality() {
        let mut state1 = create_default_state();
        let state2 = ScheduleState::new(
            vec![BerthIndex::new(1), BerthIndex::new(1), BerthIndex::new(1)],
            vec![50, 60, 70],
            vec![0, 1, 2],
            vec![0, 999],
            999,
        );

        // Initially they should not be equal
        assert_ne!(state1, state2);

        // Overwrite state1 with state2's data
        state1.overwrite_from_state(&state2);

        // Now they should be perfectly equal
        assert_eq!(state1, state2);
        assert_eq!(state1.objective(), 999);
    }

    #[test]
    fn test_unchecked_operations_maintain_integrity() {
        let mut state = create_default_state();
        let v0 = VesselIndex::new(0);
        let b1 = BerthIndex::new(1);

        unsafe {
            state.set_vessel_unchecked(v0, b1, 99, 5);
            state.set_berth_cost_unchecked(b1, 777);
        }

        assert_eq!(unsafe { state.vessel_start_unchecked(v0) }, 99);
        assert_eq!(unsafe { state.vessel_position_unchecked(v0) }, 5);
        assert_eq!(unsafe { state.berth_cost_unchecked(b1) }, 777);
    }
}
