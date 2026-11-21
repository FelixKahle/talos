// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The tests by which a neighbourhood skips moves its instance makes unpromising.
//!
//! Each neighbourhood proposes only the moves its filter lets through. The filters between berths refuse a move
//! only where the instance rules it out: a vessel may not use the berth it would reach. The filters on one berth
//! refuse every move that sets a vessel next to one whose window from arrival to latest departure does not overlap
//! its own. That is exact for a swap and for a move to the head of a berth, which would reverse the order the
//! windows force, unless both vessels take no time to serve; it is conservative for a move after another vessel:
//! moving a vessel right after one whose window lies wholly before its own keeps the forced order, and may still be
//! a move an optimum needs, which such a neighbourhood then does not propose.

use talos_core::SolverNumeric;
use talos_model::{BerthIndex, Model, VesselIndex};

/// Whether the windows of `first` and `second`, each from its arrival to its latest departure, overlap.
///
/// Where they do not, a schedule that serves the later window's vessel before the other on one berth is
/// infeasible, unless both take no time to serve.
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
    // Were the windows disjoint, say the latest departure of `first` at or before the arrival of `second`, then
    // `first` served after `second` on one berth would end no earlier than that arrival plus the times both take
    // to serve, past its latest departure if either takes any time. Only where both take none may both orders be
    // feasible.
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

/// Whether moving `vessel` right after `anchor`, on the berth they share, is proposed: their windows overlap.
///
/// A shift after an anchor whose window lies wholly before the vessel's own keeps the order the windows force and
/// is refused all the same; the filter is conservative.
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

/// Whether moving `vessel` to the head of the berth it stands on, before `head`, the vessel heading it, may help:
/// their windows overlap.
///
/// From a feasible schedule, the filter refuses only moves that are infeasible, unless both vessels take no time
/// to serve.
///
/// # Safety
///
/// `vessel` and `head` must be `model`'s.
#[inline]
pub(super) unsafe fn intra_berth_shift_to_head_filter_unchecked<T>(
    vessel: VesselIndex,
    head: VesselIndex,
    model: &Model<T>,
) -> bool
where
    T: SolverNumeric,
{
    debug_assert!(
        vessel.index() < model.num_vessels() && head.index() < model.num_vessels(),
        "called `intra_berth_shift_to_head_filter_unchecked` with {vessel} or {head}, outside the model"
    );
    // In a feasible schedule `vessel` stands after `head`, so were the windows disjoint, that of `head` would lie
    // wholly before that of `vessel` wherever either takes time to serve (`windows_overlap_unchecked`), and
    // moving `vessel` before `head` would serve them in the order that leaves one late.
    // SAFETY: both vessels are the model's, the precondition of this function.
    unsafe { windows_overlap_unchecked(vessel, head, model) }
}

/// Whether a move that takes `vessel` onto `berth`, another berth than its own, may help: `vessel` may use
/// `berth`. A swap across two berths is proposed where each vessel may use the other's berth.
///
/// # Safety
///
/// `vessel` and `berth` must be `model`'s.
#[inline]
pub(super) unsafe fn inter_berth_filter_unchecked<T>(
    vessel: VesselIndex,
    berth: BerthIndex,
    model: &Model<T>,
) -> bool
where
    T: SolverNumeric,
{
    debug_assert!(
        vessel.index() < model.num_vessels() && berth.index() < model.num_berths(),
        "called `inter_berth_filter_unchecked` with {vessel} or {berth}, outside the model"
    );
    // SAFETY: the vessel and the berth are the model's, the precondition of this function.
    unsafe { model.vessel_processing_time_unchecked(vessel, berth) }.is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    use talos_core::ClosedOpenInterval;

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
    fn test_the_filter_between_berths_refuses_a_berth_the_vessel_may_not_use() {
        let model = three_vessels();
        // SAFETY: every vessel and berth named is the model's.
        unsafe {
            assert!(inter_berth_filter_unchecked(
                vessel(0),
                BerthIndex::new(0),
                &model
            ));
            assert!(!inter_berth_filter_unchecked(
                vessel(0),
                BerthIndex::new(1),
                &model
            ));
            assert!(inter_berth_filter_unchecked(
                vessel(1),
                BerthIndex::new(0),
                &model
            ));
            assert!(inter_berth_filter_unchecked(
                vessel(2),
                BerthIndex::new(1),
                &model
            ));
        }
    }

    #[test]
    fn test_the_filter_of_a_move_to_the_head_passes_overlapping_windows_and_refuses_disjoint_ones()
    {
        let model = three_vessels();
        // SAFETY: every vessel named is the model's.
        unsafe {
            assert!(intra_berth_shift_to_head_filter_unchecked(
                vessel(1),
                vessel(0),
                &model
            ));
            // V2 [20, 30) before V0 [0, 10) would start V0 at 21 at the earliest, past its latest departure.
            assert!(!intra_berth_shift_to_head_filter_unchecked(
                vessel(2),
                vessel(0),
                &model
            ));
        }
    }
}
