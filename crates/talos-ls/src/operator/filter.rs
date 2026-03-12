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

use crate::sgraph::ScheduleGraph;
use talos_core::utils::num::SolverNumeric;
use talos_model::{index::VesselIndex, model::Model, solution::SolutionView};

/// Filter for `IntraBerthSwapOperator`.
///
/// Rejects a swap when the two vessels' feasible time windows
/// `[arrival_time, latest_departure_time)` are disjoint. If vessel A's
/// latest departure is at or before vessel B's arrival (or vice-versa),
/// their relative order on the berth is forced by the time windows alone,
/// so swapping them can only produce an infeasible or dominated schedule.
///
/// This is an optimum-preserving filter: any swap that could improve the
/// objective involves two vessels whose time windows overlap.
#[inline]
pub fn intra_berth_swap_filter<T: SolverNumeric>(
    v_a: VesselIndex,
    v_b: VesselIndex,
    _solution: SolutionView<'_, T>,
    _graph: &ScheduleGraph,
    model: &Model<T>,
) -> bool {
    // Time windows overlap iff neither is entirely before the other.
    //   overlap ⟺ arrival_a < deadline_b  ∧  arrival_b < deadline_a
    let arrival_a = model.vessel_arrival_time(v_a);
    let deadline_a = model.vessel_latest_departure_time(v_a);
    let arrival_b = model.vessel_arrival_time(v_b);
    let deadline_b = model.vessel_latest_departure_time(v_b);

    arrival_a < deadline_b && arrival_b < deadline_a
}

/// Filter for `InterBerthSwapOperator`.
///
/// Rejects a swap when at least one vessel cannot be processed at the
/// other vessel's current berth (i.e. the processing time is `None`).
/// Such a swap would always be infeasible.
///
/// This is an optimum-preserving filter: every feasible inter-berth swap
/// requires both vessels to be compatible with each other's berths.
#[inline]
pub fn inter_berth_swap_filter<T: SolverNumeric>(
    v_a: VesselIndex,
    v_b: VesselIndex,
    _solution: SolutionView<'_, T>,
    graph: &ScheduleGraph,
    model: &Model<T>,
) -> bool {
    let berth_a = graph.vessel_berth(v_a);
    let berth_b = graph.vessel_berth(v_b);

    // v_a must be allowed on berth_b, and v_b must be allowed on berth_a.
    model.vessel_allowed_on_berth(v_a, berth_b) && model.vessel_allowed_on_berth(v_b, berth_a)
}

/// Filter for `IntraBerthShiftOperator`.
///
/// Rejects a shift when the vessel being moved and the anchor vessel have
/// disjoint feasible time windows `[arrival_time, latest_departure_time)`.
/// If the two windows do not overlap, the relative ordering is forced and
/// relocating the vessel after the anchor can only produce an infeasible
/// or dominated schedule.
///
/// This is an optimum-preserving filter: any shift that could improve the
/// objective involves two vessels whose time windows overlap.
#[inline]
pub fn intra_berth_shift_filter<T: SolverNumeric>(
    v: VesselIndex,
    anchor: VesselIndex,
    _solution: SolutionView<'_, T>,
    _graph: &ScheduleGraph,
    model: &Model<T>,
) -> bool {
    let arrival_v = model.vessel_arrival_time(v);
    let deadline_v = model.vessel_latest_departure_time(v);
    let arrival_anchor = model.vessel_arrival_time(anchor);
    let deadline_anchor = model.vessel_latest_departure_time(anchor);

    arrival_v < deadline_anchor && arrival_anchor < deadline_v
}

/// Filter for `InterBerthShiftOperator`.
///
/// Rejects a shift when the vessel being moved cannot be processed at the
/// anchor vessel's berth (i.e. the processing time is `None`). Such a
/// move would always be infeasible.
///
/// This is an optimum-preserving filter: every feasible inter-berth shift
/// requires the vessel to be compatible with the target berth.
#[inline]
pub fn inter_berth_shift_filter<T: SolverNumeric>(
    v: VesselIndex,
    anchor: VesselIndex,
    _solution: SolutionView<'_, T>,
    graph: &ScheduleGraph,
    model: &Model<T>,
) -> bool {
    let berth_anchor = graph.vessel_berth(anchor);
    model.vessel_allowed_on_berth(v, berth_anchor)
}
