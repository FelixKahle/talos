// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The tests that let a neighbourhood skip a move no optimum needs.
//!
//! Each neighbourhood proposes only the moves its filter lets through. A filter refuses a move only where the
//! instance alone rules it out (the vessel may not use the berth it would reach) or forces the order it would
//! reverse (two vessels whose windows from arrival to latest departure do not overlap), so it never discards a
//! move an optimum could need.

use crate::schedule::ScheduleGraph;
use talos_core::SolverNumeric;
use talos_model::{Model, VesselIndex};

/// Whether the windows of `first` and `second`, each from its arrival to its latest departure, overlap.
///
/// Where they do not, the earlier window's vessel is served first on any berth in every feasible schedule, so a
/// move that puts the other before it is either infeasible or no better.
///
/// # Safety
///
/// `first` and `second` must be `model`'s.
#[inline]
unsafe fn windows_overlap_unchecked<T>(
    first: VesselIndex,
    second: VesselIndex,
    model: &Model<T>,
) -> bool
where
    T: SolverNumeric,
{
    debug_assert!(
        first.index() < model.num_vessels() && second.index() < model.num_vessels(),
        "called `windows_overlap_unchecked` with {first} or {second}, outside a model of {} vessels",
        model.num_vessels()
    );
    // SAFETY: both vessels are the model's, the precondition of this function.
    let (first_arrival, first_deadline, second_arrival, second_deadline) = unsafe {
        (
            model.vessel_arrival_time_unchecked(first),
            model.vessel_latest_departure_time_unchecked(first),
            model.vessel_arrival_time_unchecked(second),
            model.vessel_latest_departure_time_unchecked(second),
        )
    };
    first_arrival < second_deadline && second_arrival < first_deadline
}

/// Whether swapping `first` and `second`, which share a berth, may help: their windows overlap.
///
/// # Safety
///
/// `first` and `second` must be `model`'s.
#[inline]
pub(super) unsafe fn intra_berth_swap_filter_unchecked<T>(
    first: VesselIndex,
    second: VesselIndex,
    model: &Model<T>,
) -> bool
where
    T: SolverNumeric,
{
    debug_assert!(
        first.index() < model.num_vessels() && second.index() < model.num_vessels(),
        "called `intra_berth_swap_filter_unchecked` with {first} or {second}, outside the model"
    );
    // SAFETY: both vessels are the model's, the precondition of this function.
    unsafe { windows_overlap_unchecked(first, second, model) }
}

/// Whether swapping `first` and `second`, which lie on different berths, may help: each may use the other's
/// berth.
///
/// # Safety
///
/// `first` and `second` must be vessels of `graph`, and `graph` must have the vessels and berths of `model`.
#[inline]
pub(super) unsafe fn inter_berth_swap_filter_unchecked<T>(
    first: VesselIndex,
    second: VesselIndex,
    graph: &ScheduleGraph,
    model: &Model<T>,
) -> bool
where
    T: SolverNumeric,
{
    debug_assert!(
        first.index() < graph.num_vessels() && second.index() < graph.num_vessels(),
        "called `inter_berth_swap_filter_unchecked` with {first} or {second}, outside the graph"
    );
    // SAFETY: both vessels are the graph's, the precondition of this function.
    let (first_berth, second_berth) = unsafe {
        (
            graph.vessel_berth_unchecked(first),
            graph.vessel_berth_unchecked(second),
        )
    };
    debug_assert!(
        first.index() < model.num_vessels()
            && second.index() < model.num_vessels()
            && first_berth.index() < model.num_berths()
            && second_berth.index() < model.num_berths(),
        "called `inter_berth_swap_filter_unchecked` with a graph whose vessels or berths are not the model's"
    );
    // SAFETY: the vessels are the graph's and their berths are berths of the graph, whose vessels and berths are
    // the model's, the precondition of this function.
    unsafe {
        model
            .vessel_processing_time_unchecked(first, second_berth)
            .is_some()
            && model
                .vessel_processing_time_unchecked(second, first_berth)
                .is_some()
    }
}

/// Whether moving `vessel` right after `anchor`, on the berth they share, may help: their windows overlap.
///
/// # Safety
///
/// `vessel` and `anchor` must be `model`'s.
#[inline]
pub(super) unsafe fn intra_berth_shift_filter_unchecked<T>(
    vessel: VesselIndex,
    anchor: VesselIndex,
    model: &Model<T>,
) -> bool
where
    T: SolverNumeric,
{
    debug_assert!(
        vessel.index() < model.num_vessels() && anchor.index() < model.num_vessels(),
        "called `intra_berth_shift_filter_unchecked` with {vessel} or {anchor}, outside the model"
    );
    // SAFETY: both vessels are the model's, the precondition of this function.
    unsafe { windows_overlap_unchecked(vessel, anchor, model) }
}

/// Whether moving `vessel` right after `anchor`, onto the berth of `anchor`, may help: `vessel` may use that
/// berth.
///
/// # Safety
///
/// `vessel` and `anchor` must be vessels of `graph`, and `graph` must have the vessels and berths of `model`.
#[inline]
pub(super) unsafe fn inter_berth_shift_filter_unchecked<T>(
    vessel: VesselIndex,
    anchor: VesselIndex,
    graph: &ScheduleGraph,
    model: &Model<T>,
) -> bool
where
    T: SolverNumeric,
{
    debug_assert!(
        vessel.index() < graph.num_vessels() && anchor.index() < graph.num_vessels(),
        "called `inter_berth_shift_filter_unchecked` with {vessel} or {anchor}, outside the graph"
    );
    // SAFETY: `anchor` is the graph's, the precondition of this function.
    let anchor_berth = unsafe { graph.vessel_berth_unchecked(anchor) };
    debug_assert!(
        vessel.index() < model.num_vessels() && anchor_berth.index() < model.num_berths(),
        "called `inter_berth_shift_filter_unchecked` with a graph whose vessels or berths are not the model's"
    );
    // SAFETY: `vessel` is the graph's and `anchor_berth` a berth of the graph, whose vessels and berths are the
    // model's, the precondition of this function.
    unsafe { model.vessel_processing_time_unchecked(vessel, anchor_berth) }.is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    use talos_core::ClosedOpenInterval;
    use talos_model::BerthIndex;

    /// V0 has the window [0, 10), V1 [5, 20) and V2 [20, 30). V0 may use only B0; V1 and V2 may use either.
    fn three_vessels() -> Model<i64> {
        Model::new(
            vec![0, 5, 20],
            vec![10, 20, 30],
            vec![1, 1, 1],
            vec![Some(1), None, Some(1), Some(1), Some(1), Some(1)],
            vec![
                vec![ClosedOpenInterval::new(0, 100)],
                vec![ClosedOpenInterval::new(0, 100)],
            ],
        )
    }

    fn vessel(index: usize) -> VesselIndex {
        VesselIndex::new(index)
    }

    #[test]
    fn test_intra_berth_filters_pass_overlapping_windows_and_refuse_disjoint_ones() {
        let model = three_vessels();
        // SAFETY: every vessel named is the model's.
        unsafe {
            assert!(intra_berth_swap_filter_unchecked(
                vessel(0),
                vessel(1),
                &model
            ));
            assert!(intra_berth_shift_filter_unchecked(
                vessel(1),
                vessel(0),
                &model
            ));
            assert!(!intra_berth_swap_filter_unchecked(
                vessel(0),
                vessel(2),
                &model
            ));
            assert!(!intra_berth_shift_filter_unchecked(
                vessel(2),
                vessel(0),
                &model
            ));
            // [5, 20) and [20, 30) touch but do not overlap.
            assert!(!intra_berth_swap_filter_unchecked(
                vessel(1),
                vessel(2),
                &model
            ));
        }
    }

    #[test]
    fn test_inter_berth_filters_refuse_a_berth_the_vessel_may_not_use() {
        let model = three_vessels();
        // V0 and V2 on B0, V1 on B1.
        let berths = [BerthIndex::new(0), BerthIndex::new(1), BerthIndex::new(0)];
        let graph = ScheduleGraph::from_slices(&berths, &[0_i64, 0, 1], 2);
        // SAFETY: every vessel named is the graph's, and the graph has the vessels and berths of the model.
        unsafe {
            assert!(!inter_berth_swap_filter_unchecked(
                vessel(0),
                vessel(1),
                &graph,
                &model
            ));
            assert!(inter_berth_swap_filter_unchecked(
                vessel(2),
                vessel(1),
                &graph,
                &model
            ));
            assert!(!inter_berth_shift_filter_unchecked(
                vessel(0),
                vessel(1),
                &graph,
                &model
            ));
            assert!(inter_berth_shift_filter_unchecked(
                vessel(1),
                vessel(0),
                &graph,
                &model
            ));
        }
    }
}
