// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The neighbourhoods that exchange the places of two vessels: on one berth, or across two.
//!
//! Each pair of vessels is proposed once, from the vessel of lower index, and its partners in increasing order of
//! their index. On one berth a vessel walks only its own berth; across berths it reads the berth of every vessel of
//! higher index, a plain pass over a column.

use super::filter::{inter_berth_filter_unchecked, intra_berth_swap_filter_unchecked};
use super::scan::VesselMoves;
use crate::schedule::{Mutator, ScheduleGraph};
use talos_core::SolverNumeric;
use talos_model::{Model, VesselIndex};

/// The swaps of a vessel with each vessel of higher index on its berth whose window overlaps its own, in
/// increasing order of that index.
///
/// A berth of `k` vessels holds `k * (k - 1) / 2` such pairs before the filter; each move touches one berth. The
/// swaps of neighbours are this neighbourhood's alone: the shifts on one berth leave them out.
#[derive(Debug, Clone, Copy)]
pub(super) struct IntraBerthSwaps;

impl VesselMoves for IntraBerthSwaps {
    type Target = VesselIndex;

    #[inline]
    unsafe fn gather_unchecked<T>(
        vessel: VesselIndex,
        graph: &ScheduleGraph,
        model: &Model<T>,
        targets: &mut Vec<VesselIndex>,
    ) where
        T: SolverNumeric,
    {
        debug_assert!(
            vessel.index() < graph.num_vessels() && graph.num_vessels() == model.num_vessels(),
            "called `IntraBerthSwaps::gather_unchecked` with {vessel} outside the graph, or a graph whose vessels \
             are not the model's"
        );
        // SAFETY: `vessel` is the graph's, the precondition of this function.
        let berth = unsafe { graph.vessel_berth_unchecked(vessel) };
        debug_assert!(
            berth.index() < graph.num_berths(),
            "called `IntraBerthSwaps::gather_unchecked` with a graph that puts {vessel} on {berth}, outside its {} \
             berths",
            graph.num_berths()
        );
        // SAFETY: `berth` was read from the graph's column of berths, so it is one of its berths (`ScheduleGraph`).
        let berth_mates = unsafe { graph.vessel_sequence_iter_unchecked(berth) };
        let first = targets.len();
        for partner in berth_mates.filter(|&partner| partner > vessel) {
            debug_assert!(
                partner.index() < model.num_vessels(),
                "called `IntraBerthSwaps::gather_unchecked` with a graph whose {partner} is outside a model of {} \
                 vessels",
                model.num_vessels()
            );
            // SAFETY: `vessel` is the graph's by the precondition of this function, and the walk yields only the
            // graph's vessels (`VesselSequenceIter`); the graph's vessels are the model's, the precondition of this
            // function.
            if unsafe { intra_berth_swap_filter_unchecked(vessel, partner, model) } {
                targets.push(partner);
            }
        }
        // The walk gives the berth's order; the scan's order is that of the index.
        targets[first..].sort_unstable();
    }

    #[inline(always)] // Called once per proposed move, and only forwards to one move.
    unsafe fn make_unchecked(mutator: &mut Mutator, vessel: VesselIndex, partner: VesselIndex) {
        debug_assert!(
            vessel.index() < mutator.graph().num_vessels()
                && partner.index() < mutator.graph().num_vessels(),
            "called `IntraBerthSwaps::make_unchecked` with {vessel} or {partner}, outside a graph of {} vessels",
            mutator.graph().num_vessels()
        );
        // SAFETY: both vessels are the graph's, the precondition of this function.
        unsafe { mutator.swap_vessels_unchecked(vessel, partner) }
    }
}

/// The swaps of a vessel with each vessel of higher index on another berth, where each may use the other's berth,
/// in increasing order of that index.
///
/// A schedule of `n` vessels holds `n * (n - 1) / 2`, less the sum of `k * (k - 1) / 2` over its berths of `k`
/// vessels, such pairs before the filter; each move touches two berths.
#[derive(Debug, Clone, Copy)]
pub(super) struct InterBerthSwaps;

impl VesselMoves for InterBerthSwaps {
    type Target = VesselIndex;

    #[inline]
    unsafe fn gather_unchecked<T>(
        vessel: VesselIndex,
        graph: &ScheduleGraph,
        model: &Model<T>,
        targets: &mut Vec<VesselIndex>,
    ) where
        T: SolverNumeric,
    {
        debug_assert!(
            vessel.index() < graph.num_vessels()
                && graph.num_vessels() == model.num_vessels()
                && graph.num_berths() == model.num_berths(),
            "called `InterBerthSwaps::gather_unchecked` with {vessel} outside the graph, or a graph whose counts are \
             not the model's"
        );
        // SAFETY: `vessel` is the graph's, the precondition of this function.
        let own = unsafe { graph.vessel_berth_unchecked(vessel) };
        for partner in (vessel.index() + 1..graph.num_vessels()).map(VesselIndex::new) {
            debug_assert!(
                partner.index() < graph.num_vessels(),
                "called `InterBerthSwaps::gather_unchecked` with {partner}, beyond the graph's {} vessels",
                graph.num_vessels()
            );
            // SAFETY: `partner` is below the graph's count of vessels, the bound of the walk.
            let berth = unsafe { graph.vessel_berth_unchecked(partner) };
            if berth == own {
                continue;
            }
            debug_assert!(
                partner.index() < model.num_vessels()
                    && berth.index() < model.num_berths()
                    && own.index() < model.num_berths(),
                "called `InterBerthSwaps::gather_unchecked` with a graph whose {partner}, {berth} or {own} is \
                 outside the model"
            );
            // SAFETY: both vessels are the graph's and both berths were read from its column of berths
            // (`ScheduleGraph`); the graph's vessels and berths are the model's, the precondition of this function.
            let allowed = unsafe {
                inter_berth_filter_unchecked(vessel, berth, model)
                    && inter_berth_filter_unchecked(partner, own, model)
            };
            if allowed {
                targets.push(partner);
            }
        }
    }

    #[inline(always)] // Called once per proposed move, and only forwards to one move.
    unsafe fn make_unchecked(mutator: &mut Mutator, vessel: VesselIndex, partner: VesselIndex) {
        debug_assert!(
            vessel.index() < mutator.graph().num_vessels()
                && partner.index() < mutator.graph().num_vessels(),
            "called `InterBerthSwaps::make_unchecked` with {vessel} or {partner}, outside a graph of {} vessels",
            mutator.graph().num_vessels()
        );
        // SAFETY: both vessels are the graph's, the precondition of this function.
        unsafe { mutator.swap_vessels_unchecked(vessel, partner) }
    }
}

#[cfg(test)]
mod tests {
    use super::super::reference::{self, OVERLAPPING_WINDOWS, WINDOWS};
    use super::*;

    // Each pair on one berth once, from the vessel of lower index, the partners in increasing order; refused only
    // where the windows of V0 and V4 miss each other: ten pairs on B0 less that one, and none on the berths of one
    // vessel.
    #[test]
    fn test_the_intra_berth_scan_proposes_every_swap_on_one_berth_once_in_order() {
        let moves = reference::moves_of::<IntraBerthSwaps>(&reference::model(&WINDOWS));
        assert_eq!(moves, reference::intra_berth_swaps(&WINDOWS));
        assert_eq!(moves.len(), 9);
        let all = reference::moves_of::<IntraBerthSwaps>(&reference::model(&OVERLAPPING_WINDOWS));
        assert_eq!(all, reference::intra_berth_swaps(&OVERLAPPING_WINDOWS));
        assert_eq!(all.len(), 10);
    }

    // Each pair across two berths once, from the vessel of lower index, the partners in increasing order, where each
    // may use the other's berth: V0 to V4 with V6, and V5 with V6; V5 may not use B0, and the empty berths hold no
    // partner.
    #[test]
    fn test_the_inter_berth_scan_proposes_every_swap_across_berths_once_in_order() {
        let moves = reference::moves_of::<InterBerthSwaps>(&reference::model(&WINDOWS));
        assert_eq!(moves, reference::inter_berth_swaps());
        assert_eq!(moves.len(), 5 + 1);
    }
}
