// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

use talos_core::math::num::SolverNumeric;
use talos_model::{
    index::{BerthIndex, VesselIndex},
    model::Model,
};

/// Calculates the weighted completion time cost for a vessel assigned to a berth at a given start time.
///
/// # Mathematical Note
///
/// This function calculates **Weighted Completion Time** ($C_j \times w_j$) rather than
/// strict **Weighted Flow Time** ($(C_j - r_j) \times w_j$).
///
/// Since the term $\sum (r_j \times w_j)$ is constant for a given problem instance, minimizing
/// Weighted Completion Time yields the **exact same optimal schedule** as minimizing Weighted
/// Flow Time. Excluding the subtraction of the arrival time $r_j$ allows the solver to perform
/// fewer arithmetic operations in the hot path.
///
/// # Panics
///
/// In debug builds, this function will panic if `vessel_index` is not within `0..model.num_vessels()`
/// or if `berth_index` is not within `0..model.num_berths()`.
#[inline(always)]
pub fn calculate_weighted_completion_time<T>(
    model: &Model<T>,
    vessel_index: VesselIndex,
    berth_index: BerthIndex,
    start_time: T,
) -> Option<T>
where
    T: SolverNumeric,
{
    debug_assert!(
        vessel_index.index() < model.num_vessels(),
        "called `calculate_weighted_completion_time` with vessel index out of bounds: the len is {} but the index is {}",
        model.num_vessels(),
        vessel_index.index()
    );

    debug_assert!(
        berth_index.index() < model.num_berths(),
        "called `calculate_weighted_completion_time` with berth index out of bounds: the len is {} but the index is {}",
        model.num_berths(),
        berth_index.index()
    );

    let deadline = model.vessel_latest_departure_time(vessel_index);
    let pt_opt = model.vessel_processing_time(vessel_index, berth_index);

    if pt_opt.is_none() {
        return None;
    }

    let pt = pt_opt.unwrap_unchecked();
    let completion_time = start_time + pt;

    if completion_time > deadline {
        return None;
    }

    let weight = model.vessel_weight(vessel_index);
    Some(completion_time * weight)
}

/// Calculates the weighted completion time cost for a vessel assigned to a berth at a given start time.
///
/// # Mathematical Note
///
/// This function calculates **Weighted Completion Time** ($C_j \times w_j$) rather than
/// strict **Weighted Flow Time** ($(C_j - r_j) \times w_j$).
///
/// Since the term $\sum (r_j \times w_j)$ is constant for a given problem instance, minimizing
/// Weighted Completion Time yields the **exact same optimal schedule** as minimizing Weighted
/// Flow Time. Excluding the subtraction of the arrival time $r_j$ allows the solver to perform
/// fewer arithmetic operations in the hot path.
///
/// # Panics
///
/// In debug builds, this function will panic if `vessel_index` is not within `0..model.num_vessels()`
/// or if `berth_index` is not within `0..model.num_berths()`.
///
/// # Safety
///
/// The caller must ensure that `vessel_index` is within `0..model.num_vessels()` and
/// `berth_index` is within `0..model.num_berths()`.
#[inline(always)]
pub unsafe fn calculate_weighted_completion_time_unchecked<T>(
    model: &Model<T>,
    vessel_index: VesselIndex,
    berth_index: BerthIndex,
    start_time: T,
) -> Option<T>
where
    T: SolverNumeric,
{
    debug_assert!(
        vessel_index.index() < model.num_vessels(),
        "called `calculate_weighted_completion_time_unchecked` with vessel index out of bounds: the len is {} but the index is {}",
        model.num_vessels(),
        vessel_index.index()
    );

    debug_assert!(
        berth_index.index() < model.num_berths(),
        "called `calculate_weighted_completion_time_unchecked` with berth index out of bounds: the len is {} but the index is {}",
        model.num_berths(),
        berth_index.index()
    );

    let deadline = unsafe { model.vessel_latest_departure_time_unchecked(vessel_index) };
    let pt_opt = unsafe { model.vessel_processing_time_unchecked(vessel_index, berth_index) };

    if pt_opt.is_none() {
        return None;
    }

    let pt = pt_opt.unwrap_unchecked();
    let completion_time = start_time + pt;

    if completion_time > deadline {
        return None;
    }

    let weight = unsafe { model.vessel_weight_unchecked(vessel_index) };
    Some(completion_time * weight)
}

/// Calculates the weighted total turnaround time (or flow time) cost for a vessel assigned to a berth at a given start time.
///
/// The turnaround time is defined as the completion time minus the arrival time of the vessel.
///
/// # Panics
///
/// In debug builds, this function will panic if `vessel_index` is not within `0..model.num_vessels()`
/// or if `berth_index` is not within `0..model.num_berths()`.
#[inline(always)]
pub fn calculate_weighted_turnaround_time<T>(
    model: &Model<T>,
    vessel_index: VesselIndex,
    berth_index: BerthIndex,
    start_time: T,
) -> Option<T>
where
    T: SolverNumeric,
{
    debug_assert!(
        vessel_index.index() < model.num_vessels(),
        "called `calculate_weighted_turnaround_time` with vessel index out of bounds: the len is {} but the index is {}",
        model.num_vessels(),
        vessel_index.index()
    );

    debug_assert!(
        berth_index.index() < model.num_berths(),
        "called `calculate_weighted_turnaround_time` with berth index out of bounds: the len is {} but the index is {}",
        model.num_berths(),
        berth_index.index()
    );

    let deadline = model.vessel_latest_departure_time(vessel_index);
    let pt_opt = model.vessel_processing_time(vessel_index, berth_index);

    if pt_opt.is_none() {
        return None;
    }

    let pt = pt_opt.unwrap_unchecked();
    let completion_time = start_time + pt;

    if completion_time > deadline {
        return None;
    }

    let arrival_time = model.vessel_arrival_time(vessel_index);
    let weight = model.vessel_weight(vessel_index);

    let turnaround_time = completion_time - arrival_time;

    Some(turnaround_time * weight)
}

/// Calculates the weighted total turnaround time (or flow time) cost for a vessel assigned to a berth at a given start time.
///
/// The turnaround time is defined as the completion time minus the arrival time of the vessel.
///
/// # Panics
///
/// In debug builds, this function will panic if `vessel_index` is not within `0..model.num_vessels()`
/// or if `berth_index` is not within `0..model.num_berths()`.
///
/// # Safety
///
/// The caller must ensure that `vessel_index` is within `0..model.num_vessels()` and
/// `berth_index` is within `0..model.num_berths()`.
#[inline(always)]
pub unsafe fn calculate_weighted_turnaround_time_unchecked<T>(
    model: &Model<T>,
    vessel_index: VesselIndex,
    berth_index: BerthIndex,
    start_time: T,
) -> Option<T>
where
    T: SolverNumeric,
{
    debug_assert!(
        vessel_index.index() < model.num_vessels(),
        "called `calculate_weighted_turnaround_time_unchecked` with vessel index out of bounds: the len is {} but the index is {}",
        model.num_vessels(),
        vessel_index.index()
    );

    debug_assert!(
        berth_index.index() < model.num_berths(),
        "called `calculate_weighted_turnaround_time_unchecked` with berth index out of bounds: the len is {} but the index is {}",
        model.num_berths(),
        berth_index.index()
    );

    let deadline = unsafe { model.vessel_latest_departure_time_unchecked(vessel_index) };
    let pt_opt = unsafe { model.vessel_processing_time_unchecked(vessel_index, berth_index) };

    if pt_opt.is_none() {
        return None;
    }

    let pt = pt_opt.unwrap_unchecked();
    let completion_time = start_time + pt;

    if completion_time > deadline {
        return None;
    }

    let arrival_time = unsafe { model.vessel_arrival_time_unchecked(vessel_index) };
    let weight = unsafe { model.vessel_weight_unchecked(vessel_index) };

    let turnaround_time = completion_time - arrival_time;

    Some(turnaround_time * weight)
}
