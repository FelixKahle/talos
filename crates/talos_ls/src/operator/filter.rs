// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Neighborhood filter functions for local-search operators.
//!
//! Each filter is designed to be passed to the corresponding operator's
//! constructor (e.g. `IntraBerthSwapOperator::new(intra_berth_swap_filter_unchecked)`).
//! They prune moves that are guaranteed to be infeasible or dominated,
//! without ever discarding a move that could lead to the optimum.

use crate::sgraph::ScheduleGraph;
use talos_core::SolverNumeric;
use talos_model::{Model, SolutionView, VesselIndex};

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
///
/// # Safety
///
/// The caller must ensure that `v_a` and `v_b` are valid vessel indices
/// in `0..model.num_vessels()`.
#[inline]
pub unsafe fn intra_berth_swap_filter_unchecked<T>(
    v_a: VesselIndex,
    v_b: VesselIndex,
    _solution: SolutionView<'_, T>,
    _graph: &ScheduleGraph,
    model: &Model<T>,
) -> bool
where
    T: SolverNumeric,
{
    debug_assert!(v_a.index() < model.num_vessels());
    debug_assert!(v_b.index() < model.num_vessels());

    // Time windows overlap iff neither is entirely before the other.
    //   overlap ⟺ arrival_a < deadline_b  ∧  arrival_b < deadline_a
    let arrival_a = unsafe { model.vessel_arrival_time_unchecked(v_a) };
    let deadline_a = unsafe { model.vessel_latest_departure_time_unchecked(v_a) };
    let arrival_b = unsafe { model.vessel_arrival_time_unchecked(v_b) };
    let deadline_b = unsafe { model.vessel_latest_departure_time_unchecked(v_b) };

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
///
/// # Safety
///
/// The caller must ensure that `v_a` and `v_b` are valid vessel indices
/// in `0..graph.num_vessels()`, and that `graph` has the vessels and the
/// berths of `model`.
#[inline]
pub unsafe fn inter_berth_swap_filter_unchecked<T>(
    v_a: VesselIndex,
    v_b: VesselIndex,
    _solution: SolutionView<'_, T>,
    graph: &ScheduleGraph,
    model: &Model<T>,
) -> bool
where
    T: SolverNumeric,
{
    debug_assert!(v_a.index() < graph.num_vessels());
    debug_assert!(v_b.index() < graph.num_vessels());

    let berth_a = unsafe { graph.vessel_berth_unchecked(v_a) };
    let berth_b = unsafe { graph.vessel_berth_unchecked(v_b) };

    // v_a must be allowed on berth_b, and v_b must be allowed on berth_a.
    debug_assert!(
        v_a.index() < model.num_vessels()
            && v_b.index() < model.num_vessels()
            && berth_a.index() < model.num_berths()
            && berth_b.index() < model.num_berths()
    );
    // SAFETY: the vessels are of `graph`, as the caller guarantees, and their berths are berths of `graph`;
    // `graph` has the vessels and the berths of `model`, as the caller guarantees.
    unsafe {
        model
            .vessel_processing_time_unchecked(v_a, berth_b)
            .is_some()
            && model
                .vessel_processing_time_unchecked(v_b, berth_a)
                .is_some()
    }
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
///
/// # Safety
///
/// The caller must ensure that `v` and `anchor` are valid vessel indices
/// in `0..model.num_vessels()`.
#[inline]
pub unsafe fn intra_berth_shift_filter_unchecked<T>(
    v: VesselIndex,
    anchor: VesselIndex,
    _solution: SolutionView<'_, T>,
    _graph: &ScheduleGraph,
    model: &Model<T>,
) -> bool
where
    T: SolverNumeric,
{
    debug_assert!(anchor.index() < model.num_vessels());
    debug_assert!(v.index() < model.num_vessels());

    let arrival_v = unsafe { model.vessel_arrival_time_unchecked(v) };
    let deadline_v = unsafe { model.vessel_latest_departure_time_unchecked(v) };
    let arrival_anchor = unsafe { model.vessel_arrival_time_unchecked(anchor) };
    let deadline_anchor = unsafe { model.vessel_latest_departure_time_unchecked(anchor) };

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
///
/// # Safety
///
/// The caller must ensure that `v` and `anchor` are valid vessel indices
/// in `0..graph.num_vessels()`, and that `graph` has the vessels and the
/// berths of `model`.
#[inline]
pub unsafe fn inter_berth_shift_filter_unchecked<T>(
    v: VesselIndex,
    anchor: VesselIndex,
    _solution: SolutionView<'_, T>,
    graph: &ScheduleGraph,
    model: &Model<T>,
) -> bool
where
    T: SolverNumeric,
{
    debug_assert!(anchor.index() < graph.num_vessels());
    let berth_anchor = unsafe { graph.vessel_berth_unchecked(anchor) };
    debug_assert!(v.index() < model.num_vessels() && berth_anchor.index() < model.num_berths());
    // SAFETY: `v` is a vessel of `graph` and `berth_anchor` a berth of it, as the caller guarantees for the
    // vessels; `graph` has the vessels and the berths of `model`, as the caller guarantees.
    unsafe { model.vessel_processing_time_unchecked(v, berth_anchor) }.is_some()
}
