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

//! Schedule graph decoding routines.
//!
//! This module translates a ScheduleGraph — which encodes the assignment of
//! vessels to berths and their relative ordering — into concrete start times,
//! positions, and cost values stored in a ScheduleState.
//!
//! Two entry points are provided:
//!
//! - `decode_full_unchecked` decodes every berth in the graph from scratch,
//!   producing a complete ScheduleState with all vessel assignments and a
//!   total objective value.
//!
//! - `decode_unchecked` performs a delta decode, re-evaluating only the berths
//!   marked in a TouchedBerths mask while carrying forward untouched berth
//!   costs from a previously accepted ScheduleState.
//!
//! Both functions rely on `decode_berth` to iterate through the vessel sequence
//! on a single berth, calling `find_earliest_start_unchecked` to place each
//! vessel into the earliest feasible opening-time interval that satisfies the
//! vessel's arrival time, processing duration, and the berth's free-time
//! constraint. A caller-supplied evaluator closure computes the per-vessel cost
//! contribution.
//!
//! All public functions in this module are marked `unsafe` because they bypass
//! bounds checks on berth and vessel indices for performance. Callers must
//! uphold the invariants documented on each function.

use crate::{
    sgraph::{ScheduleGraph, VesselSequenceIter},
    state::ScheduleState,
    tberth::TouchedBerths,
};
use talos_core::utils::num::SolverNumeric;
use talos_model::{
    index::{BerthIndex, VesselIndex},
    model::Model,
};

/// Finds the earliest feasible start time for a vessel on a given berth,
/// searching forward through the berth's opening time intervals starting
/// from `cached_index`.
///
/// Returns `Some((start_time, interval_index))` if the vessel can be scheduled,
/// or `None` if no feasible interval exists. The returned `interval_index` can
/// be passed as `cached_index` to the next call to avoid re-scanning earlier
/// intervals.
///
/// # Safety
///
/// - `berth_index.get()` must be `< model.num_berths()`.
/// - `cached_index` must be `<= model.berth_opening_times_unchecked(berth_index).len()`.
#[inline]
unsafe fn find_earliest_start_unchecked<T>(
    model: &Model<T>,
    berth_index: BerthIndex,
    vessel_arrival: T,
    duration: T,
    berth_free_time: T,
    cached_index: usize,
) -> Option<(T, usize)>
where
    T: SolverNumeric,
{
    debug_assert!(berth_index.get() < model.num_berths());

    let min_start = if berth_free_time > vessel_arrival {
        berth_free_time
    } else {
        vessel_arrival
    };

    let intervals = unsafe { model.berth_opening_times_unchecked(berth_index) };

    for (offset, interval) in intervals[cached_index..].iter().enumerate() {
        if interval.end() <= min_start {
            continue;
        }
        let actual_start = if interval.start() > min_start {
            interval.start()
        } else {
            min_start
        };

        if actual_start + duration <= interval.end() {
            return Some((actual_start, cached_index + offset));
        }
    }
    None
}

/// Decodes a single berth's vessel sequence into `candidate_buffer`, computing start times,
/// positions, and the berth's total cost contribution.
///
/// Returns `Some(berth_cost)` on success, or `None` if any vessel cannot be
/// feasibly scheduled (no valid opening interval or the evaluator returns `None`).
///
/// # Safety
///
/// - `berth.get()` must be `< model.num_berths()`.
/// - Every `VesselIndex` yielded by `sequence` must be `< candidate_buffer.num_vessels()`
///   and `< model.num_vessels()`.
/// - `candidate_buffer` must have sufficient capacity for all vessels in the sequence.
#[inline(always)]
unsafe fn decode_berth<'a, T, F>(
    berth: BerthIndex,
    sequence: VesselSequenceIter<'a>,
    candidate_buffer: &mut ScheduleState<T>,
    model: &Model<T>,
    evaluator: &F,
) -> Option<T>
where
    T: SolverNumeric,
    F: Fn(&Model<T>, VesselIndex, BerthIndex, T) -> Option<T>,
{
    let mut free_time = T::ZERO;
    let mut interval_idx = 0;
    let mut berth_cost = T::ZERO;

    for (position, vessel) in sequence.enumerate() {
        let arrival = unsafe { model.vessel_arrival_time_unchecked(vessel) };
        let pt = unsafe { model.vessel_processing_time_unchecked(vessel, berth) }.into_option()?;

        let (start, next_idx) = unsafe {
            find_earliest_start_unchecked(model, berth, arrival, pt, free_time, interval_idx)
        }?;

        let cost_delta = evaluator(model, vessel, berth, start)?;

        free_time = start + pt;
        interval_idx = next_idx;
        berth_cost = berth_cost + cost_delta;

        unsafe {
            candidate_buffer.set_vessel_unchecked(vessel, berth, start, position);
        }
    }

    Some(berth_cost)
}

/// Performs a full decode of every berth in the schedule graph, writing all vessel
/// assignments, berth costs, and the total objective into `candidate_buffer`.
///
/// Returns `Some(())` on success, or `None` if any vessel cannot be feasibly
/// scheduled.
///
/// # Safety
///
/// - `graph.num_berths()` must equal `model.num_berths()`.
/// - `graph.num_vessels()` must be `<= candidate_buffer.num_vessels()` and `<= model.num_vessels()`.
/// - `candidate_buffer.num_berths()` must be `>= model.num_berths()`.
/// - The `graph` must be structurally valid and only yield vessel indices within
///   the bounds of both `candidate_buffer` and `model`.
#[inline]
pub unsafe fn decode_full_unchecked<T, F>(
    graph: &ScheduleGraph,
    candidate_buffer: &mut ScheduleState<T>,
    model: &Model<T>,
    evaluator: F,
) -> Option<()>
where
    T: SolverNumeric,
    F: Fn(&Model<T>, VesselIndex, BerthIndex, T) -> Option<T>,
{
    let mut total_obj = T::ZERO;

    for b in 0..model.num_berths() {
        let berth = BerthIndex::new(b);
        let sequence = unsafe { graph.vessel_sequence_iter_unchecked(berth) };

        let berth_cost =
            unsafe { decode_berth(berth, sequence, candidate_buffer, model, &evaluator) }?;

        unsafe { candidate_buffer.set_berth_cost_unchecked(berth, berth_cost) };
        total_obj = total_obj + berth_cost;
    }

    candidate_buffer.set_objective(total_obj);
    Some(())
}

/// Decodes only the berths marked as `true` in `touched`, writing their vessel
/// assignments and berth costs into `candidate_buffer`. The total objective is computed by
/// summing touched berth costs from `candidate_buffer` and untouched berth costs from `accepted`.
///
/// Returns `Some(())` on success, or `None` if any vessel on a touched berth
/// cannot be feasibly scheduled.
///
/// # Note
///
/// **Important invariant:** After this call, `candidate_buffer`'s berth costs and vessel data
/// for untouched berths are *not* updated and may contain stale or uninitialized
/// values. Callers must not read `candidate_buffer` data for untouched berths directly. The
/// intended consumption path is `ScheduleState::patch_from_delta_unchecked`,
/// which respects the same `touched` mask.
///
/// # Safety
///
/// - `touched.len()` must equal `model.num_berths()`.
/// - `canditate_graph.num_berths()` must equal `model.num_berths()`.
/// - `canditate_graph.num_vessels()` must be `<= candidate_buffer.num_vessels()`, `<= accepted.num_vessels()`,
///   and `<= model.num_vessels()`.
/// - `candidate_buffer.num_berths()` and `accepted.num_berths()` must be `>= model.num_berths()`.
/// - The `canditate_graph` must be structurally valid and only yield vessel indices within
///   the bounds of `candidate_buffer`, `accepted`, and `model`.
#[inline]
pub unsafe fn decode_unchecked<T, F>(
    touched: &TouchedBerths,
    canditate_graph: &ScheduleGraph,
    candidate_buffer: &mut ScheduleState<T>,
    accepted: &ScheduleState<T>,
    model: &Model<T>,
    evaluator: F,
) -> Option<()>
where
    T: SolverNumeric,
    F: Fn(&Model<T>, VesselIndex, BerthIndex, T) -> Option<T>,
{
    debug_assert_eq!(
        touched.num_berths(),
        model.num_berths(),
        "called `decode_unchecked` with mismatched touched length and model berths: touched.num_berths() = {}, model.num_berths() = {}",
        touched.num_berths(),
        model.num_berths()
    );

    let mut total_obj = accepted.objective();
    for berth in touched.iter_touched_berths() {
        let sequence = unsafe { canditate_graph.vessel_sequence_iter_unchecked(berth) };
        let berth_cost =
            unsafe { decode_berth(berth, sequence, candidate_buffer, model, &evaluator) }?;
        unsafe { candidate_buffer.set_berth_cost_unchecked(berth, berth_cost) };
        let old_cost = unsafe { accepted.berth_cost_unchecked(berth) };
        total_obj = total_obj - old_cost + berth_cost;
    }

    candidate_buffer.set_objective(total_obj);
    Some(())
}

#[cfg(test)]
mod test_decoder {
    use super::*;
    use talos_core::math::interval::ClosedOpenInterval;
    use talos_model::model::ProcessingTime;

    // Helper to build a valid 2-vessel, 2-berth model for testing.
    fn build_test_model() -> Model<i64> {
        let num_vessels = 2;
        let num_berths = 2;

        let arrivals = vec![10, 20]; // V0 arrives at 10, V1 arrives at 20
        let departures = vec![100, 100];
        let weights = vec![1, 1];

        // Matrix: [v0b0, v0b1, v1b0, v1b1]
        let processing = vec![
            ProcessingTime::some(5),
            ProcessingTime::some(5),
            ProcessingTime::some(10),
            ProcessingTime::some(10),
        ];

        let opening = vec![
            vec![
                ClosedOpenInterval::new(0, 15),
                ClosedOpenInterval::new(25, 100),
            ], // B0 has a gap!
            vec![ClosedOpenInterval::new(0, 100)], // B1 is always open
        ];

        Model::new(
            num_vessels,
            num_berths,
            arrivals,
            departures,
            weights,
            processing,
            opening,
        )
    }

    // Evaluator that simply uses the start time as the cost for easy math
    fn mock_evaluator(
        _model: &Model<i64>,
        _vessel: VesselIndex,
        _berth: BerthIndex,
        start_time: i64,
    ) -> Option<i64> {
        Some(start_time)
    }

    #[test]
    fn test_find_earliest_start_unchecked() {
        let model = build_test_model();
        let b0 = BerthIndex::new(0);

        unsafe {
            // Test 1: Fits perfectly in the first interval
            // Arrives at 5, needs 5. B0 open [0, 15). Free time 0.
            // Starts at 5, ends at 10.
            let res1 = find_earliest_start_unchecked(&model, b0, 5, 5, 0, 0);
            assert_eq!(res1, Some((5, 0))); // start time 5, interval index 0

            // Test 2: Pushed by free time
            // Arrives at 0, needs 5. Free time is 10.
            // Starts at 10, ends at 15.
            let res2 = find_earliest_start_unchecked(&model, b0, 0, 5, 10, 0);
            assert_eq!(res2, Some((10, 0)));

            // Test 3: Interval gap skip
            // Arrives at 12, needs 5. First interval [0, 15) is too tight (ends 15, needs 17).
            // Must skip to second interval [25, 100).
            let res3 = find_earliest_start_unchecked(&model, b0, 12, 5, 0, 0);
            assert_eq!(res3, Some((25, 1))); // starts at 25 (start of 2nd interval), index 1

            // Test 4: Does not fit anywhere
            // Needs 200 time, max interval is 75.
            let res4 = find_earliest_start_unchecked(&model, b0, 0, 200, 0, 0);
            assert_eq!(res4, None);
        }
    }

    #[test]
    fn test_decode_berth_success() {
        let model = build_test_model();

        // Setup a sequence: V0 -> V1
        let graph =
            ScheduleGraph::from_slices(&[BerthIndex::new(0), BerthIndex::new(0)], &[0, 0], 2);
        let b0 = BerthIndex::new(0);
        let sequence = unsafe { graph.vessel_sequence_iter_unchecked(b0) };

        // Create buffer with garbage data
        let mut candidate = ScheduleState::new(
            vec![BerthIndex::new(99); 2],
            vec![9999; 2],
            vec![99; 2],
            vec![9999; 2],
            9999,
        );

        let berth_cost =
            unsafe { decode_berth(b0, sequence, &mut candidate, &model, &mock_evaluator) };

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
        assert_eq!(candidate.vessel_position(VesselIndex::new(0)), 0);
        assert_eq!(candidate.vessel_position(VesselIndex::new(1)), 1);
        assert_eq!(candidate.vessel_berth(VesselIndex::new(0)), b0);
        assert_eq!(candidate.vessel_berth(VesselIndex::new(1)), b0);
    }

    #[test]
    fn test_decode_full_unchecked() {
        let model = build_test_model();
        // V0 -> B0, V1 -> B1
        let graph =
            ScheduleGraph::from_slices(&[BerthIndex::new(0), BerthIndex::new(1)], &[0, 0], 2);

        let mut candidate = ScheduleState::new(
            vec![BerthIndex::new(99); 2],
            vec![9999; 2],
            vec![99; 2],
            vec![9999; 2],
            9999,
        );

        let res = unsafe { decode_full_unchecked(&graph, &mut candidate, &model, mock_evaluator) };
        assert!(res.is_some());

        // B0 gets V0 (starts 10). B0 cost = 10.
        // B1 gets V1 (arrives 20, B1 interval [0, 100)). B1 cost = 20.
        // Total = 30.

        assert_eq!(candidate.objective(), 30);
        assert_eq!(candidate.berth_cost(BerthIndex::new(0)), 10);
        assert_eq!(candidate.berth_cost(BerthIndex::new(1)), 20);
    }

    #[test]
    fn test_decode_unchecked_with_delta_and_garbage_data() {
        let model = build_test_model();

        // Let's pretend our ACCEPTED state had V0 and V1 on B1
        // Accepted Costs: B0 = 0, B1 = 50. Total = 50.
        let accepted = ScheduleState::new(
            vec![BerthIndex::new(1), BerthIndex::new(1)],
            vec![10, 20],
            vec![0, 1],
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
            vec![99; 2],
            vec![9999; 2],
            9999,
        );

        // Run partial decode!
        let res = unsafe {
            decode_unchecked(
                &touched,
                &candidate_graph,
                &mut candidate_buffer,
                &accepted,
                &model,
                mock_evaluator,
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
    fn test_decode_unchecked_ignores_untouched_garbage() {
        let model = build_test_model();

        // Accepted state where B1 has cost 500
        let accepted = ScheduleState::new(
            vec![BerthIndex::new(0), BerthIndex::new(1)],
            vec![10, 20],
            vec![0, 0],
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
            vec![99; 2],
            vec![9999; 2],
            9999,
        );

        let res = unsafe {
            decode_unchecked(
                &touched,
                &candidate_graph,
                &mut candidate_buffer,
                &accepted,
                &model,
                mock_evaluator,
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
}
