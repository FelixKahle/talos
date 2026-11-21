// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The decoder: from the order of the vessels on each berth to the schedule that serves each as early as it can.
//!
//! Each berth is decoded on its own and left-justified: its vessels are served in their order, each from the
//! earliest time that is no earlier than its arrival and the end of the service before it, at which its service
//! fits an opening of the berth by [`Model::service_fits_opening`]. A [`VesselCost`] prices each service; a berth
//! whose vessel fits no opening, or whose service the cost refuses, makes the schedule infeasible.
//!
//! [`decode_full_unchecked`] decodes every berth; [`decode_unchecked`] decodes only the berths a move touched and
//! carries the others' costs over from the accepted schedule.

use super::schedule_state::ScheduleState;
use crate::objective::VesselCost;
use crate::schedule::{ScheduleGraph, TouchedBerths, VesselSequenceIter};
use talos_core::SolverNumeric;
use talos_model::{BerthIndex, Model};

/// The earliest start no earlier than `earliest` at which a service of `processing_time` fits an opening of
/// `berth`, searching the openings from the one at `first_opening` on, and the opening it fits; or `None` if it
/// fits none of them.
///
/// The openings are ordered and apart, so the first that admits the service gives the earliest start, and a later
/// service on the berth, which starts no earlier, can resume the search from that opening.
///
/// # Safety
///
/// `berth` must be one of `model`'s, and `first_opening` at most its count of openings.
#[inline(always)]
unsafe fn earliest_start_unchecked<T>(
    model: &Model<T>,
    berth: BerthIndex,
    earliest: T,
    processing_time: T,
    first_opening: usize,
) -> Option<(T, usize)>
where
    T: SolverNumeric,
{
    debug_assert!(
        berth.index() < model.num_berths(),
        "called `earliest_start_unchecked` with {berth}, outside a model of {} berths",
        model.num_berths()
    );
    // SAFETY: `berth` is the model's, the precondition of this function.
    let openings = unsafe { model.berth_opening_times_unchecked(berth) };
    debug_assert!(
        first_opening <= openings.len(),
        "called `earliest_start_unchecked` with opening {first_opening} of a berth of {} openings",
        openings.len()
    );
    // SAFETY: `first_opening` is at most the count of openings, the precondition of this function.
    let rest = unsafe { openings.get_unchecked(first_opening..) };
    rest.iter().enumerate().find_map(|(offset, &opening)| {
        Model::earliest_start_in_opening(opening, earliest, processing_time)
            .map(|start| (start, first_opening + offset))
    })
}

/// Serves the vessels of `sequence` on `berth` in their order, each as early as it fits, writes each one's berth
/// and start into `candidate`, and answers the sum of their costs; or `None` if a vessel fits no opening or the
/// cost refuses it, with `candidate` then partly written.
///
/// # Safety
///
/// `berth` must be one of `model`'s, and every vessel `sequence` yields one of `model`'s and of `candidate`'s.
#[inline(always)]
unsafe fn decode_berth_unchecked<T, C>(
    berth: BerthIndex,
    sequence: VesselSequenceIter<'_>,
    candidate: &mut ScheduleState<T>,
    model: &Model<T>,
) -> Option<T>
where
    T: SolverNumeric,
    C: VesselCost<T>,
{
    debug_assert!(
        berth.index() < model.num_berths(),
        "called `decode_berth_unchecked` with {berth}, outside a model of {} berths",
        model.num_berths()
    );
    let mut free_from = T::ZERO;
    let mut opening = 0;
    let mut berth_cost = T::ZERO;
    for vessel in sequence {
        debug_assert!(
            vessel.index() < model.num_vessels() && vessel.index() < candidate.num_vessels(),
            "called `decode_berth_unchecked` with {vessel}, outside the model or the candidate"
        );
        // SAFETY: `berth` and every vessel of the sequence are the model's, the precondition of this function.
        let (arrival, processing_time) = unsafe {
            (
                model.vessel_arrival_time_unchecked(vessel),
                model.vessel_processing_time_unchecked(vessel, berth)?,
            )
        };
        // SAFETY: `berth` is the model's, the precondition of this function, and `opening` is 0 or an opening the
        // last search found, so at most the berth's count of openings.
        let (start, found) = unsafe {
            earliest_start_unchecked(
                model,
                berth,
                free_from.max(arrival),
                processing_time,
                opening,
            )
        }?;
        // SAFETY: `berth` and every vessel of the sequence are the model's, the precondition of this function.
        berth_cost = berth_cost + unsafe { C::cost_unchecked(model, vessel, berth, start) }?;
        free_from = start + processing_time;
        opening = found;
        // SAFETY: every vessel of the sequence is the candidate's, the precondition of this function.
        unsafe { candidate.set_vessel_unchecked(vessel, berth, start) };
    }
    Some(berth_cost)
}

/// Decodes every berth of `graph` into `candidate`: each vessel's berth and start, each berth's cost and their
/// sum as the objective; or `None` if a vessel fits no opening or the cost refuses it, with `candidate` then
/// partly written.
///
/// # Safety
///
/// `graph`, `candidate` and `model` must have the same vessels and the same berths: each must count as many of
/// each as the others.
#[inline]
pub unsafe fn decode_full_unchecked<T, C>(
    graph: &ScheduleGraph,
    candidate: &mut ScheduleState<T>,
    model: &Model<T>,
) -> Option<()>
where
    T: SolverNumeric,
    C: VesselCost<T>,
{
    debug_assert!(
        graph.num_vessels() == model.num_vessels()
            && candidate.num_vessels() == model.num_vessels()
            && graph.num_berths() == model.num_berths()
            && candidate.num_berths() == model.num_berths(),
        "called `decode_full_unchecked` with a graph, a candidate and a model of different counts"
    );
    let mut objective = T::ZERO;
    for berth in model.berths() {
        // SAFETY: `berth` is the model's, whose berths are the graph's, a precondition of this function.
        let sequence = unsafe { graph.vessel_sequence_iter_unchecked(berth) };
        // SAFETY: `berth` is the model's; the walk yields only vessels of the graph (`VesselSequenceIter`), whose
        // vessels are the model's and the candidate's, a precondition of this function.
        let berth_cost =
            unsafe { decode_berth_unchecked::<T, C>(berth, sequence, candidate, model) }?;
        // SAFETY: `berth` is the model's, whose berths are the candidate's, a precondition of this function.
        unsafe { candidate.set_berth_cost_unchecked(berth, berth_cost) };
        objective = objective + berth_cost;
    }
    candidate.set_objective(objective);
    Some(())
}

/// Decodes only the berths `touched` marks into `candidate`, and sets its objective to that of `accepted` with the
/// touched berths' costs exchanged for their new ones; or `None` if a vessel fits no opening or the cost
/// refuses it.
///
/// `candidate` is written only on the touched berths and their vessels; elsewhere it holds what it held, which
/// [`ScheduleState::patch_from_delta_unchecked`] with the same `touched` never reads.
///
/// # Safety
///
/// `touched`, `graph`, `candidate`, `accepted` and `model` must have the same berths, and the last four the same
/// vessels: each must count as many of each as the others.
#[inline]
pub unsafe fn decode_unchecked<T, C>(
    touched: &TouchedBerths,
    graph: &ScheduleGraph,
    candidate: &mut ScheduleState<T>,
    accepted: &ScheduleState<T>,
    model: &Model<T>,
) -> Option<()>
where
    T: SolverNumeric,
    C: VesselCost<T>,
{
    debug_assert!(
        touched.num_berths() == model.num_berths()
            && graph.num_berths() == model.num_berths()
            && candidate.num_berths() == model.num_berths()
            && accepted.num_berths() == model.num_berths()
            && graph.num_vessels() == model.num_vessels()
            && candidate.num_vessels() == model.num_vessels()
            && accepted.num_vessels() == model.num_vessels(),
        "called `decode_unchecked` with touched berths, a graph, states and a model of different counts"
    );
    let mut objective = accepted.objective();
    for berth in touched.iter_touched_berths() {
        // SAFETY: every berth `touched` yields is below its count (`TouchedBerths`), which is the graph's, a
        // precondition of this function.
        let sequence = unsafe { graph.vessel_sequence_iter_unchecked(berth) };
        // SAFETY: `berth` is below the count of `touched`, which is the model's; the walk yields only vessels of the
        // graph (`VesselSequenceIter`), whose vessels are the model's and the candidate's, a precondition of this
        // function.
        let berth_cost =
            unsafe { decode_berth_unchecked::<T, C>(berth, sequence, candidate, model) }?;
        // SAFETY: `berth` is below the count of `touched`, which is that of both states, a precondition of this
        // function.
        let accepted_cost = unsafe {
            candidate.set_berth_cost_unchecked(berth, berth_cost);
            accepted.berth_cost_unchecked(berth)
        };
        objective = objective - accepted_cost + berth_cost;
    }
    candidate.set_objective(objective);
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::objective::CompletionCost;
    use proptest::prelude::*;
    use talos_core::ClosedOpenInterval;
    use talos_model::VesselIndex;

    // Helper to build a valid 2-vessel, 2-berth model for testing.
    fn build_test_model() -> Model<i64> {
        let arrivals = vec![10, 20]; // V0 arrives at 10, V1 arrives at 20
        let departures = vec![100, 100];
        let weights = vec![1, 1];

        // Matrix: [v0b0, v0b1, v1b0, v1b1]
        let processing = vec![Some(5), Some(5), Some(10), Some(10)];

        let opening = vec![
            vec![
                ClosedOpenInterval::new(0, 15),
                ClosedOpenInterval::new(25, 100),
            ], // B0 has a gap!
            vec![ClosedOpenInterval::new(0, 100)], // B1 is always open
        ];

        Model::new(arrivals, departures, weights, processing, opening)
    }

    /// The start time of a vessel as its cost, for easy arithmetic.
    #[derive(Debug)]
    struct StartTimeCost;

    impl VesselCost<i64> for StartTimeCost {
        unsafe fn cost_unchecked(
            _model: &Model<i64>,
            _vessel: VesselIndex,
            _berth: BerthIndex,
            start_time: i64,
        ) -> Option<i64> {
            Some(start_time)
        }
    }

    // B0 is open on [0, 15) and [25, 100).
    #[test]
    fn test_earliest_start_is_the_first_that_fits_an_opening() {
        let model = build_test_model();
        let b0 = BerthIndex::new(0);
        // SAFETY: B0 is the model's, and every first opening is at most its count of 2.
        unsafe {
            // From 5 a service of 5 fits [0, 15).
            assert_eq!(earliest_start_unchecked(&model, b0, 5, 5, 0), Some((5, 0)));
            // From 12 it would end at 17, past 15, so it starts at 25 in the second opening.
            assert_eq!(
                earliest_start_unchecked(&model, b0, 12, 5, 0),
                Some((25, 1))
            );
            // A search that resumes at the second opening does not look back at the first.
            assert_eq!(earliest_start_unchecked(&model, b0, 0, 5, 1), Some((25, 1)));
            // A service of 200 fits no opening, nor does any service once no opening is left.
            assert_eq!(earliest_start_unchecked(&model, b0, 0, 200, 0), None);
            assert_eq!(earliest_start_unchecked(&model, b0, 0, 1, 2), None);
        }
    }

    /// The schedule `decode_full_unchecked` makes of `graph`, with the start time as each vessel's cost.
    fn decoded(model: &Model<i64>, graph: &ScheduleGraph) -> ScheduleState<i64> {
        let num_vessels = model.num_vessels();
        let mut candidate = ScheduleState::new(
            vec![BerthIndex::new(0); num_vessels],
            vec![0; num_vessels],
            vec![0; model.num_berths()],
            0,
        );
        // SAFETY: the graph and the state are of the model's vessels and berths, as each test builds them.
        let decoded =
            unsafe { decode_full_unchecked::<_, StartTimeCost>(graph, &mut candidate, model) };
        assert_eq!(decoded, Some(()));
        candidate
    }

    // Openings given as [20, 30) and [0, 10): the first vessel (length 5) fits at 0, and the second (length 8),
    // free from 5, no longer fits [0, 10) and starts at 20.
    #[test]
    fn test_decode_places_vessels_in_openings_given_out_of_order() {
        let model = Model::new(
            vec![0, 0],
            vec![100, 100],
            vec![1, 1],
            vec![Some(5), Some(8)],
            vec![vec![
                ClosedOpenInterval::new(20, 30),
                ClosedOpenInterval::new(0, 10),
            ]],
        );
        let graph = ScheduleGraph::from_slices(&[BerthIndex::new(0); 2], &[0, 1], 1);
        let state = decoded(&model, &graph);
        assert_eq!(state.vessel_start(VesselIndex::new(0)), 0);
        assert_eq!(state.vessel_start(VesselIndex::new(1)), 20);
        assert_eq!(model.check_feasibility(state.as_solution_view()), Ok(()));
    }

    // Openings [0, 10) and [10, 20) touch, so the berth is open on [0, 20): the first vessel (length 15) fits at
    // 0, and the second (length 0), free from 15, starts at 15.
    #[test]
    fn test_decode_places_a_vessel_across_touching_openings() {
        let model = Model::new(
            vec![0, 0],
            vec![100, 100],
            vec![1, 1],
            vec![Some(15), Some(0)],
            vec![vec![
                ClosedOpenInterval::new(0, 10),
                ClosedOpenInterval::new(10, 20),
            ]],
        );
        let graph = ScheduleGraph::from_slices(&[BerthIndex::new(0); 2], &[0, 1], 1);
        let state = decoded(&model, &graph);
        assert_eq!(state.vessel_start(VesselIndex::new(0)), 0);
        assert_eq!(state.vessel_start(VesselIndex::new(1)), 15);
        assert_eq!(model.check_feasibility(state.as_solution_view()), Ok(()));
    }

    // A vessel of length 0 arriving at 10, with openings [0, 10) and [20, 30): the berth is closed at 10, so the
    // service starts at 20.
    #[test]
    fn test_decode_starts_a_zero_length_service_where_the_berth_is_open() {
        let model = Model::new(
            vec![10],
            vec![100],
            vec![1],
            vec![Some(0)],
            vec![vec![
                ClosedOpenInterval::new(0, 10),
                ClosedOpenInterval::new(20, 30),
            ]],
        );
        let graph = ScheduleGraph::from_slices(&[BerthIndex::new(0)], &[0], 1);
        let state = decoded(&model, &graph);
        assert_eq!(state.vessel_start(VesselIndex::new(0)), 20);
        assert_eq!(model.check_feasibility(state.as_solution_view()), Ok(()));
    }

    #[test]
    fn test_decode_berth_serves_each_vessel_as_early_as_it_fits() {
        let model = build_test_model();

        // Setup a sequence: V0 -> V1
        let graph =
            ScheduleGraph::from_slices(&[BerthIndex::new(0), BerthIndex::new(0)], &[0, 0], 2);
        let b0 = BerthIndex::new(0);
        // SAFETY: B0 is the graph's.
        let sequence = unsafe { graph.vessel_sequence_iter_unchecked(b0) };

        // Create buffer with garbage data
        let mut candidate = ScheduleState::new(
            vec![BerthIndex::new(99); 2],
            vec![9999; 2],
            vec![9999; 2],
            9999,
        );

        // SAFETY: B0 and both vessels are the model's and the candidate's.
        let berth_cost = unsafe {
            decode_berth_unchecked::<_, StartTimeCost>(b0, sequence, &mut candidate, &model)
        };

        // Expected logic:
        // V0: arrives 10, free 0. B0 intervals: [0, 15), [25, 100)
        // -> V0 starts at 10 (fits in [0, 15)), takes 5. Free time becomes 15. Cost = 10.
        // V1: arrives 20, free 15. Needs 10.
        // -> Min start max(20, 15) = 20. But interval 0 ends at 15.
        // -> Moves to interval 1 [25, 100). Starts at 25, takes 10. Free time 35. Cost = 25.
        // Total Berth Cost = 10 + 25 = 35.

        assert_eq!(berth_cost, Some(35));

        // Verify buffer got overwritten correctly
        assert_eq!(candidate.vessel_start(VesselIndex::new(0)), 10);
        assert_eq!(candidate.vessel_start(VesselIndex::new(1)), 25);
        assert_eq!(candidate.vessel_berth(VesselIndex::new(0)), b0);
        assert_eq!(candidate.vessel_berth(VesselIndex::new(1)), b0);
    }

    #[test]
    fn test_decode_full_sums_the_costs_of_every_berth() {
        let model = build_test_model();
        // V0 -> B0, V1 -> B1
        let graph =
            ScheduleGraph::from_slices(&[BerthIndex::new(0), BerthIndex::new(1)], &[0, 0], 2);

        let mut candidate = ScheduleState::new(
            vec![BerthIndex::new(99); 2],
            vec![9999; 2],
            vec![9999; 2],
            9999,
        );

        // SAFETY: the graph, the candidate and the model have two vessels and two berths.
        let res =
            unsafe { decode_full_unchecked::<_, StartTimeCost>(&graph, &mut candidate, &model) };
        assert!(res.is_some());

        // B0 gets V0 (starts 10). B0 cost = 10.
        // B1 gets V1 (arrives 20, B1 interval [0, 100)). B1 cost = 20.
        // Total = 30.

        assert_eq!(candidate.objective(), 30);
        assert_eq!(candidate.berth_cost(BerthIndex::new(0)), 10);
        assert_eq!(candidate.berth_cost(BerthIndex::new(1)), 20);
    }

    #[test]
    fn test_decode_exchanges_the_costs_of_the_touched_berths() {
        let model = build_test_model();

        // Let's pretend our ACCEPTED state had V0 and V1 on B1
        // Accepted Costs: B0 = 0, B1 = 50. Total = 50.
        let accepted = ScheduleState::new(
            vec![BerthIndex::new(1), BerthIndex::new(1)],
            vec![10, 20],
            vec![0, 50],
            50,
        );

        // Candidate Graph moves V0 to B0. V1 stays on B1.
        let candidate_graph =
            ScheduleGraph::from_slices(&[BerthIndex::new(0), BerthIndex::new(1)], &[0, 0], 2);

        // Create the touched mask indicating B0 and B1 were touched
        let mut touched = TouchedBerths::new(2);
        touched.touch(BerthIndex::new(0));
        touched.touch(BerthIndex::new(1)); // Both touched because V0 moved B1 -> B0

        // Create candidate_buffer filled with GARBAGE
        let mut candidate_buffer = ScheduleState::new(
            vec![BerthIndex::new(99); 2],
            vec![9999; 2],
            vec![9999; 2],
            9999,
        );

        // Run partial decode!
        // SAFETY: the touched berths, the graph, the states and the model have two vessels and two berths.
        let res = unsafe {
            decode_unchecked::<_, StartTimeCost>(
                &touched,
                &candidate_graph,
                &mut candidate_buffer,
                &accepted,
                &model,
            )
        };
        assert!(res.is_some());

        // Evaluation:
        // B0 evaluates V0 -> start 10. B0 cost = 10.
        // B1 evaluates V1 -> start 20. B1 cost = 20.
        // Delta Evaluation: Total = Accepted(50) - Old B0(0) + New B0(10) - Old B1(50) + New B1(20) = 30.

        assert_eq!(candidate_buffer.objective(), 30);
        assert_eq!(candidate_buffer.berth_cost(BerthIndex::new(0)), 10);
        assert_eq!(candidate_buffer.berth_cost(BerthIndex::new(1)), 20);

        // Check that vessel 0 and 1 got properly decoded and their garbage was overwritten
        assert_eq!(candidate_buffer.vessel_start(VesselIndex::new(0)), 10);
        assert_eq!(candidate_buffer.vessel_start(VesselIndex::new(1)), 20);
    }

    #[test]
    fn test_decode_leaves_the_untouched_berths_unwritten() {
        let model = build_test_model();

        // Accepted state where B1 has cost 500
        let accepted = ScheduleState::new(
            vec![BerthIndex::new(0), BerthIndex::new(1)],
            vec![10, 20],
            vec![10, 500],
            510, // 10 + 500
        );

        // We only modify B0 in the graph (pretend V0's sequence changed internally).
        let candidate_graph =
            ScheduleGraph::from_slices(&[BerthIndex::new(0), BerthIndex::new(1)], &[0, 0], 2);

        let mut touched = TouchedBerths::new(2);
        touched.touch(BerthIndex::new(0)); // ONLY B0 IS TOUCHED

        // Candidate buffer is full of garbage!
        let mut candidate_buffer = ScheduleState::new(
            vec![BerthIndex::new(99); 2],
            vec![9999; 2],
            vec![9999; 2],
            9999,
        );

        // SAFETY: the touched berths, the graph, the states and the model have two vessels and two berths.
        let res = unsafe {
            decode_unchecked::<_, StartTimeCost>(
                &touched,
                &candidate_graph,
                &mut candidate_buffer,
                &accepted,
                &model,
            )
        };
        assert!(res.is_some());

        // B0 gets decoded -> cost 10.
        // B1 is UNTOUCHED, so it is skipped.
        // Delta Math: 510 (Accepted) - 10 (Old B0) + 10 (New B0) = 510.

        assert_eq!(candidate_buffer.objective(), 510);
        assert_eq!(candidate_buffer.berth_cost(BerthIndex::new(0)), 10);

        // V0 should be safely overwritten because B0 was touched
        assert_eq!(candidate_buffer.vessel_start(VesselIndex::new(0)), 10);

        // **CRITICAL TEST**: B1 was untouched. Its data in candidate_buffer should STILL BE GARBAGE.
        // This proves we aren't wasting cycles writing to arrays we don't need to.
        assert_eq!(candidate_buffer.berth_cost(BerthIndex::new(1)), 9999);
        assert_eq!(candidate_buffer.vessel_start(VesselIndex::new(1)), 9999);
    }

    /// The schedule a left-justified placement by brute force gives the vessels of `order` on each berth: each one
    /// from the first time, scanned one by one from the later of its arrival and the end of the service before
    /// it, at which its service fits an opening; and the sum of their weighted completion times. `None` if a
    /// vessel fits nowhere before `horizon`, may not use its berth, or ends after its latest departure.
    fn brute_force(
        model: &Model<i64>,
        order: &[Vec<VesselIndex>],
        horizon: i64,
    ) -> Option<(Vec<i64>, i64)> {
        let mut starts = vec![0; model.num_vessels()];
        let mut objective = 0;
        for (berth, vessels) in model.berths().zip(order) {
            let mut free_from = 0;
            for &vessel in vessels {
                let processing_time = model.vessel_processing_time(vessel, berth)?;
                let start =
                    (free_from.max(model.vessel_arrival_time(vessel))..horizon).find(|&time| {
                        model.berth_opening_times(berth).iter().any(|&opening| {
                            Model::service_fits_opening(opening, time, processing_time)
                        })
                    })?;
                let end = start + processing_time;
                if end > model.vessel_latest_departure_time(vessel) {
                    return None;
                }
                starts[vessel.index()] = start;
                objective += end * model.vessel_weight(vessel);
                free_from = end;
            }
        }
        Some((starts, objective))
    }

    /// A model of up to five vessels on up to three berths, with times below 40 and openings that may touch,
    /// overlap or leave gaps, and the berth and the place in its order of each vessel.
    fn small_instance() -> impl Strategy<Value = (Model<i64>, Vec<BerthIndex>)> {
        (1_usize..=5, 1_usize..=3).prop_flat_map(|(num_vessels, num_berths)| {
            (
                proptest::collection::vec((0_i64..20, 0_i64..60, 1_i64..4), num_vessels),
                proptest::collection::vec(
                    proptest::option::weighted(0.85, 0_i64..8),
                    num_vessels * num_berths,
                ),
                proptest::collection::vec(
                    proptest::collection::vec((0_i64..40, 0_i64..15), 1..4),
                    num_berths,
                ),
                proptest::collection::vec(0..num_berths, num_vessels),
            )
                .prop_map(|(vessels, processing_times, openings, berths)| {
                    let model = Model::new(
                        vessels.iter().map(|&(arrival, _, _)| arrival).collect(),
                        vessels
                            .iter()
                            .map(|&(arrival, slack, _)| arrival + slack)
                            .collect(),
                        vessels.iter().map(|&(_, _, weight)| weight).collect(),
                        processing_times,
                        openings
                            .into_iter()
                            .map(|berth| {
                                berth
                                    .into_iter()
                                    .map(|(start, length)| {
                                        ClosedOpenInterval::new(start, start + length)
                                    })
                                    .collect()
                            })
                            .collect(),
                    );
                    (model, berths.into_iter().map(BerthIndex::new).collect())
                })
        })
    }

    proptest! {
        #[test]
        fn test_decode_matches_a_brute_force_left_justified_placement(
            (model, berths) in small_instance(),
            order_keys in proptest::collection::vec(0_i64..100, 5),
        ) {
            let keys = &order_keys[..model.num_vessels()];
            let graph = ScheduleGraph::from_slices(&berths, keys, model.num_berths());
            let order: Vec<Vec<VesselIndex>> =
                graph.berths().map(|berth| graph.vessel_sequence_iter(berth).collect()).collect();
            let mut candidate = ScheduleState::new(
                vec![BerthIndex::new(0); model.num_vessels()],
                vec![0; model.num_vessels()],
                vec![0; model.num_berths()],
                0,
            );
            // SAFETY: the graph, the candidate and the model have the same vessels and berths, as built above.
            let decoded = unsafe {
                decode_full_unchecked::<_, CompletionCost>(&graph, &mut candidate, &model)
            };
            let expected = brute_force(&model, &order, 200);
            prop_assert_eq!(decoded.is_some(), expected.is_some());
            if let Some((starts, objective)) = expected {
                prop_assert_eq!(&candidate.as_solution_view().start_times().raw, &starts);
                prop_assert_eq!(candidate.objective(), objective);
                prop_assert_eq!(model.check_feasibility(candidate.as_solution_view()), Ok(()));
            }
        }
    }
}
