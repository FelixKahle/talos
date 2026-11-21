// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The cost of serving one vessel at one berth from one start, which the decoder sums over a schedule.
//!
//! Each cost is an unchecked function, called through a cost type by the decoder, with a checked twin the tests
//! hold it against; a cost is `None` where the vessel may not use the berth or would leave after its latest
//! departure.

use talos_core::SolverNumeric;
use talos_model::{BerthIndex, Model, VesselIndex};

/// The weighted completion time of `vessel` served at `berth` from `start_time`: its weight times the time its
/// service ends, or `None` if it may not use `berth` or would end after its latest departure.
///
/// The weighted completion time differs from the weighted turnaround time by the weighted arrival times, a sum
/// fixed by the model, so both rank schedules alike; this one leaves out a subtraction per vessel.
///
/// # Panics
///
/// Panics if `vessel` or `berth` is not one of `model`'s.
#[cfg(test)]
#[track_caller]
pub fn calculate_weighted_completion_time<T>(
    model: &Model<T>,
    vessel: VesselIndex,
    berth: BerthIndex,
    start_time: T,
) -> Option<T>
where
    T: SolverNumeric,
{
    assert!(
        vessel.index() < model.num_vessels() && berth.index() < model.num_berths(),
        "called `calculate_weighted_completion_time` with {vessel} or {berth}, outside a model of {} vessels and {} \
         berths",
        model.num_vessels(),
        model.num_berths()
    );
    // SAFETY: `vessel` and `berth` are the model's, as the check above found.
    unsafe { calculate_weighted_completion_time_unchecked(model, vessel, berth, start_time) }
}

/// The weighted completion time of `vessel` served at `berth` from `start_time`, for a vessel and a berth the
/// caller knows to be `model`'s.
///
/// # Safety
///
/// `vessel` and `berth` must be `model`'s, the condition [`calculate_weighted_completion_time`] checks.
// Called by its cost type once per vessel of every berth the decoder walks; inlined, its reads of the model fuse
// with the decoder's own.
#[inline(always)]
pub unsafe fn calculate_weighted_completion_time_unchecked<T>(
    model: &Model<T>,
    vessel: VesselIndex,
    berth: BerthIndex,
    start_time: T,
) -> Option<T>
where
    T: SolverNumeric,
{
    debug_assert!(
        vessel.index() < model.num_vessels() && berth.index() < model.num_berths(),
        "called `calculate_weighted_completion_time_unchecked` with {vessel} or {berth}, outside a model of {} \
         vessels and {} berths",
        model.num_vessels(),
        model.num_berths()
    );
    // SAFETY: `vessel` and `berth` are the model's, the precondition of this function.
    let processing_time = unsafe { model.vessel_processing_time_unchecked(vessel, berth) }?;
    let completion_time = start_time + processing_time;
    // SAFETY: `vessel` is the model's, the precondition of this function.
    let deadline = unsafe { model.vessel_latest_departure_time_unchecked(vessel) };
    if completion_time > deadline {
        return None;
    }
    // SAFETY: `vessel` is the model's, the precondition of this function.
    let weight = unsafe { model.vessel_weight_unchecked(vessel) };
    Some(completion_time * weight)
}

/// The weighted turnaround time of `vessel` served at `berth` from `start_time`: its weight times the time from
/// its arrival to the end of its service, or `None` if it may not use `berth` or would end after its latest
/// departure.
///
/// # Panics
///
/// Panics if `vessel` or `berth` is not one of `model`'s.
#[cfg(test)]
#[track_caller]
pub fn calculate_weighted_turnaround_time<T>(
    model: &Model<T>,
    vessel: VesselIndex,
    berth: BerthIndex,
    start_time: T,
) -> Option<T>
where
    T: SolverNumeric,
{
    assert!(
        vessel.index() < model.num_vessels() && berth.index() < model.num_berths(),
        "called `calculate_weighted_turnaround_time` with {vessel} or {berth}, outside a model of {} vessels and {} \
         berths",
        model.num_vessels(),
        model.num_berths()
    );
    // SAFETY: `vessel` and `berth` are the model's, as the check above found.
    unsafe { calculate_weighted_turnaround_time_unchecked(model, vessel, berth, start_time) }
}

/// The weighted turnaround time of `vessel` served at `berth` from `start_time`, for a vessel and a berth the
/// caller knows to be `model`'s.
///
/// # Safety
///
/// `vessel` and `berth` must be `model`'s, the condition [`calculate_weighted_turnaround_time`] checks.
// Called by its cost type once per vessel of every berth the decoder walks; inlined, its reads of the model fuse
// with the decoder's own.
#[inline(always)]
pub unsafe fn calculate_weighted_turnaround_time_unchecked<T>(
    model: &Model<T>,
    vessel: VesselIndex,
    berth: BerthIndex,
    start_time: T,
) -> Option<T>
where
    T: SolverNumeric,
{
    debug_assert!(
        vessel.index() < model.num_vessels() && berth.index() < model.num_berths(),
        "called `calculate_weighted_turnaround_time_unchecked` with {vessel} or {berth}, outside a model of {} \
         vessels and {} berths",
        model.num_vessels(),
        model.num_berths()
    );
    // SAFETY: `vessel` and `berth` are the model's, the precondition of this function.
    let processing_time = unsafe { model.vessel_processing_time_unchecked(vessel, berth) }?;
    let completion_time = start_time + processing_time;
    // SAFETY: `vessel` is the model's, the precondition of this function.
    let deadline = unsafe { model.vessel_latest_departure_time_unchecked(vessel) };
    if completion_time > deadline {
        return None;
    }
    // SAFETY: `vessel` is the model's, the precondition of this function.
    let (arrival_time, weight) = unsafe {
        (
            model.vessel_arrival_time_unchecked(vessel),
            model.vessel_weight_unchecked(vessel),
        )
    };
    Some((completion_time - arrival_time) * weight)
}

/// The objective a run minimises: the sum over the vessels of one cost of each vessel's service.
///
/// Both rank schedules alike, since they differ by the weighted arrival times, a sum the model fixes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum Objective {
    /// The sum of the weighted turnaround times, as [`calculate_weighted_turnaround_time`] gives each.
    #[default]
    WeightedTurnaroundTime,
    /// The sum of the weighted completion times, as [`calculate_weighted_completion_time`] gives each.
    WeightedCompletionTime,
}

/// The cost of one vessel's service, which the decoder sums over a schedule.
///
/// The decoder calls it only with a vessel and a berth of the model it decodes against, which its own
/// precondition proves; an implementation reads the model unchecked through them.
pub trait VesselCost<T>
where
    T: SolverNumeric,
{
    /// The cost of `vessel` served at `berth` from `start_time`, or `None` where that service is not allowed.
    ///
    /// # Safety
    ///
    /// `vessel` and `berth` must be `model`'s.
    unsafe fn cost_unchecked(
        model: &Model<T>,
        vessel: VesselIndex,
        berth: BerthIndex,
        start_time: T,
    ) -> Option<T>;
}

/// The weighted turnaround time of a vessel, as the cost the decoder sums for
/// [`Objective::WeightedTurnaroundTime`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct TurnaroundCost;

impl<T> VesselCost<T> for TurnaroundCost
where
    T: SolverNumeric,
{
    // The decoder calls the cost once per vessel of every berth it decodes; inlined into its loop, the reads of
    // the model fuse with the decoder's own.
    #[inline(always)]
    unsafe fn cost_unchecked(
        model: &Model<T>,
        vessel: VesselIndex,
        berth: BerthIndex,
        start_time: T,
    ) -> Option<T> {
        debug_assert!(
            vessel.index() < model.num_vessels() && berth.index() < model.num_berths(),
            "called `TurnaroundCost::cost_unchecked` with {vessel} or {berth}, outside the model"
        );
        // SAFETY: `vessel` and `berth` are the model's, the precondition of this function.
        unsafe { calculate_weighted_turnaround_time_unchecked(model, vessel, berth, start_time) }
    }
}

/// The weighted completion time of a vessel, as the cost the decoder sums for
/// [`Objective::WeightedCompletionTime`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct CompletionCost;

impl<T> VesselCost<T> for CompletionCost
where
    T: SolverNumeric,
{
    // As for the turnaround time: inlined into the decoder's loop.
    #[inline(always)]
    unsafe fn cost_unchecked(
        model: &Model<T>,
        vessel: VesselIndex,
        berth: BerthIndex,
        start_time: T,
    ) -> Option<T> {
        debug_assert!(
            vessel.index() < model.num_vessels() && berth.index() < model.num_berths(),
            "called `CompletionCost::cost_unchecked` with {vessel} or {berth}, outside the model"
        );
        // SAFETY: `vessel` and `berth` are the model's, the precondition of this function.
        unsafe { calculate_weighted_completion_time_unchecked(model, vessel, berth, start_time) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use talos_core::ClosedOpenInterval;

    /// V0 arrives at 10, must leave by 50 and weighs 2; V1 arrives at 0, must leave by 100 and weighs 3. V0 takes
    /// 5 at B0 and may not use B1; V1 takes 7 at either.
    fn two_vessels() -> Model<i64> {
        Model::new(
            vec![10, 0],
            vec![50, 100],
            vec![2, 3],
            vec![Some(5), None, Some(7), Some(7)],
            vec![
                vec![ClosedOpenInterval::new(0, 1000)],
                vec![ClosedOpenInterval::new(0, 1000)],
            ],
        )
    }

    fn vessel(index: usize) -> VesselIndex {
        VesselIndex::new(index)
    }

    fn berth(index: usize) -> BerthIndex {
        BerthIndex::new(index)
    }

    #[test]
    fn test_completion_time_weighs_the_end_of_service() {
        // V0 from 20 ends at 25, weighing 2.
        assert_eq!(
            calculate_weighted_completion_time(&two_vessels(), vessel(0), berth(0), 20),
            Some(50)
        );
    }

    #[test]
    fn test_turnaround_time_weighs_the_time_since_arrival() {
        // V0 from 20 ends at 25, 15 after its arrival at 10, weighing 2.
        assert_eq!(
            calculate_weighted_turnaround_time(&two_vessels(), vessel(0), berth(0), 20),
            Some(30)
        );
    }

    #[test]
    fn test_a_berth_the_vessel_may_not_use_has_no_cost() {
        let model = two_vessels();
        assert_eq!(
            calculate_weighted_completion_time(&model, vessel(0), berth(1), 20),
            None
        );
        assert_eq!(
            calculate_weighted_turnaround_time(&model, vessel(0), berth(1), 20),
            None
        );
    }

    #[test]
    fn test_a_service_that_ends_after_the_latest_departure_has_no_cost() {
        let model = two_vessels();
        // V0 from 45 ends at 50, its latest departure; from 46 it ends after it.
        assert_eq!(
            calculate_weighted_completion_time(&model, vessel(0), berth(0), 45),
            Some(100)
        );
        assert_eq!(
            calculate_weighted_completion_time(&model, vessel(0), berth(0), 46),
            None
        );
        assert_eq!(
            calculate_weighted_turnaround_time(&model, vessel(0), berth(0), 46),
            None
        );
    }

    #[test]
    #[should_panic(
        expected = "called `calculate_weighted_completion_time` with Vessel 2 or Berth 0, outside a model of 2 \
                    vessels and 2 berths"
    )]
    fn test_completion_time_refuses_a_vessel_outside_the_model() {
        let _ = calculate_weighted_completion_time(&two_vessels(), vessel(2), berth(0), 0);
    }

    #[test]
    #[should_panic(
        expected = "called `calculate_weighted_turnaround_time` with Vessel 0 or Berth 2, outside a model of 2 \
                    vessels and 2 berths"
    )]
    fn test_turnaround_time_refuses_a_berth_outside_the_model() {
        let _ = calculate_weighted_turnaround_time(&two_vessels(), vessel(0), berth(2), 0);
    }

    proptest! {
        #[test]
        fn test_checked_costs_refuse_every_vessel_and_berth_outside_the_model(
            vessel_index in 0_usize..8,
            berth_index in 0_usize..8,
        ) {
            prop_assume!(vessel_index >= 2 || berth_index >= 2);
            let model = two_vessels();
            let (vessel, berth) = (vessel(vessel_index), berth(berth_index));
            let refused =
                std::panic::catch_unwind(|| calculate_weighted_completion_time(&model, vessel, berth, 0));
            prop_assert!(refused.is_err());
            let refused =
                std::panic::catch_unwind(|| calculate_weighted_turnaround_time(&model, vessel, berth, 0));
            prop_assert!(refused.is_err());
        }

        #[test]
        fn test_checked_costs_match_their_unchecked_twins(
            vessel_index in 0_usize..2,
            berth_index in 0_usize..2,
            start in -100_i64..200,
        ) {
            let model = two_vessels();
            let (vessel, berth) = (vessel(vessel_index), berth(berth_index));
            // SAFETY: both vessels and both berths are the model's.
            unsafe {
                prop_assert_eq!(
                    calculate_weighted_completion_time(&model, vessel, berth, start),
                    calculate_weighted_completion_time_unchecked(&model, vessel, berth, start)
                );
                prop_assert_eq!(
                    calculate_weighted_turnaround_time(&model, vessel, berth, start),
                    calculate_weighted_turnaround_time_unchecked(&model, vessel, berth, start)
                );
            }
        }
        #[test]
        fn test_cost_types_match_their_cost_functions(
            vessel_index in 0_usize..2,
            berth_index in 0_usize..2,
            start in -100_i64..200,
        ) {
            let model = two_vessels();
            let (vessel, berth) = (vessel(vessel_index), berth(berth_index));
            // SAFETY: both vessels and both berths are the model's.
            unsafe {
                prop_assert_eq!(
                    <TurnaroundCost as VesselCost<i64>>::cost_unchecked(&model, vessel, berth, start),
                    calculate_weighted_turnaround_time(&model, vessel, berth, start)
                );
                prop_assert_eq!(
                    <CompletionCost as VesselCost<i64>>::cost_unchecked(&model, vessel, berth, start),
                    calculate_weighted_completion_time(&model, vessel, berth, start)
                );
            }
        }
    }
}
