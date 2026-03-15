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

use std::collections::BTreeSet;
use talos_core::math::interval::ClosedOpenInterval;
use talos_core::utils::num::SolverNumeric;
use talos_model::{
    index::{BerthIndex, VesselIndex},
    model::Model,
    solution::Solution,
};

// ----------------------------------------------------------------
// Earliest-Deadline-First (EDF) greedy construction heuristic
// ----------------------------------------------------------------

/// Returns `true` if the interval `[start, finish)` overlaps any interval in `occupancy`.
#[inline]
fn overlaps_occupancy<T>(start: T, finish: T, occupancy: &[ClosedOpenInterval<T>]) -> bool
where
    T: SolverNumeric + PartialOrd + Copy,
{
    // A simple overlap check: max(start1, start2) < min(end1, end2)
    // Here, we just check if start < occ.end && occ.start < finish
    occupancy
        .iter()
        .any(|occ| start < occ.end() && occ.start() < finish)
}

/// Returns `true` if `stay` is entirely contained within at least one of the
/// opening-time windows for the given berth.
#[inline]
fn fits_in_opening_window<T>(
    model: &Model<T>,
    berth: BerthIndex,
    stay: ClosedOpenInterval<T>,
) -> bool
where
    T: SolverNumeric + PartialOrd + Copy,
{
    model
        .berth_opening_times(berth)
        .iter()
        .any(|window| window.start() <= stay.start() && window.end() >= stay.end())
}

/// Collects and returns the sorted, deduplicated set of candidate start times
/// for scheduling `vessel` on `berth`.
fn candidate_start_times<T>(
    model: &Model<T>,
    vessel: VesselIndex,
    berth: BerthIndex,
    occupancy: &[ClosedOpenInterval<T>],
) -> Vec<T>
where
    T: SolverNumeric + Ord + Copy,
{
    let arrival = model.vessel_arrival_time(vessel);

    // BTreeSet automatically deduplicates and keeps elements sorted.
    let mut candidates = BTreeSet::new();
    candidates.insert(arrival);

    for occ in occupancy {
        if occ.end() >= arrival {
            candidates.insert(occ.end());
        }
    }

    for window in model.berth_opening_times(berth) {
        if window.start() >= arrival {
            candidates.insert(window.start());
        }
    }

    candidates.into_iter().collect()
}

/// Generates a feasible schedule using an Earliest-Deadline-First (EDF) greedy heuristic.
///
/// Returns `Some(Solution)` if a feasible schedule could be found, or `None` if the
/// instance is too constrained for this greedy strategy.
pub fn generate_greedy_edf_schedule<T>(model: &Model<T>) -> Option<Solution<T>>
where
    T: SolverNumeric
        + Ord
        + Copy
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + Default,
{
    let num_vessels = model.num_vessels();
    let num_berths = model.num_berths();

    // Arrays to build our final Solution
    let mut final_berths = vec![BerthIndex::new(0); num_vessels];
    let mut final_starts = vec![T::default(); num_vessels];

    // Per-berth list of already-scheduled (start, end) intervals
    let mut berth_occupancy: Vec<Vec<ClosedOpenInterval<T>>> = vec![Vec::new(); num_berths];

    // 1. Sort vessels in EDF order
    let mut sorted_vessels: Vec<VesselIndex> = (0..num_vessels).map(VesselIndex::new).collect();
    sorted_vessels.sort_unstable_by(|&v1, &v2| {
        let dep1 = model.vessel_latest_departure_time(v1);
        let dep2 = model.vessel_latest_departure_time(v2);
        let arr1 = model.vessel_arrival_time(v1);
        let arr2 = model.vessel_arrival_time(v2);
        let w1 = model.vessel_weight(v1);
        let w2 = model.vessel_weight(v2);

        // Sort keys: Deadline -> Arrival -> Descending Weight -> Vessel Index
        dep1.cmp(&dep2)
            .then(arr1.cmp(&arr2))
            .then(w2.cmp(&w1)) // w2 compared to w1 gives descending order
            .then(v1.get().cmp(&v2.get()))
    });

    // 2. Greedily insert each vessel
    for vessel in sorted_vessels {
        let arrival = model.vessel_arrival_time(vessel);
        let deadline = model.vessel_latest_departure_time(vessel);
        let weight = model.vessel_weight(vessel);

        let mut best_berth: Option<BerthIndex> = None;
        let mut best_start: Option<T> = None;

        // We use a boolean flag or Option instead of `typemax` to keep things generic
        let mut best_cost: Option<T> = None;

        for (b, occupancy) in berth_occupancy.iter().enumerate().take(num_berths) {
            let berth = BerthIndex::new(b);
            let pt_opt = model.vessel_processing_time(vessel, berth);

            if pt_opt.is_none() {
                continue; // Forbidden assignment
            }
            let pt = pt_opt.unwrap();

            let candidates = candidate_start_times(model, vessel, berth, occupancy);

            for t in candidates {
                let finish = t + pt;

                // Constraint 1: Hard deadline
                if finish > deadline {
                    break; // Candidates are sorted; later ones only get worse
                }

                // Constraint 2: Must fit inside a berth opening window
                let stay = ClosedOpenInterval::new(t, finish);
                if !fits_in_opening_window(model, berth, stay) {
                    continue;
                }

                // Constraint 3: No overlap with vessels already on this berth
                if overlaps_occupancy(t, finish, &berth_occupancy[b]) {
                    continue;
                }

                // Feasible placement found — evaluate its cost.
                let cost = weight * (finish - arrival);

                if best_cost.is_none_or(|bc| cost < bc) {
                    best_cost = Some(cost);
                    best_start = Some(t);
                    best_berth = Some(berth);
                }

                break; // Earliest valid time on this berth is always cheapest; no need to look later
            }
        }

        // Infeasible: no berth could accommodate this vessel.
        let assigned_berth = best_berth?;
        let assigned_start = best_start?;

        // Record assignment
        final_berths[vessel.get()] = assigned_berth;
        final_starts[vessel.get()] = assigned_start;

        // Update occupancy
        let pt = model
            .vessel_processing_time(vessel, assigned_berth)
            .unwrap();
        berth_occupancy[assigned_berth.get()]
            .push(ClosedOpenInterval::new(assigned_start, assigned_start + pt));
        // Keep occupancies sorted by start time
        berth_occupancy[assigned_berth.get()].sort_unstable_by_key(|occ| occ.start());
    }

    // 3. Compute total objective function
    let mut total_objective = T::default();
    for vessel in model.vessel_iter() {
        let berth = final_berths[vessel.get()];
        let start = final_starts[vessel.get()];

        let weight = model.vessel_weight(vessel);
        let pt = model.vessel_processing_time(vessel, berth).unwrap();
        let arrival = model.vessel_arrival_time(vessel);

        total_objective = total_objective + (weight * (start + pt - arrival));
    }

    Some(Solution::new(final_berths, final_starts, total_objective))
}
