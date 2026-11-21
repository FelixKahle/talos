// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The neighbourhoods that insert one vessel at another place: on its own berth, or on another.
//!
//! A place is where a link of a berth starts: right after a vessel, or at the head of a berth. Every place of
//! every berth is so reachable, the head of a berth and an empty berth included. A vessel's places are proposed
//! in increasing order of the vessel they follow and then of the berth whose head they are. On its own berth a
//! vessel walks only that berth; onto other berths it reads the berth of every other vessel, a plain pass over a
//! column.

use super::filter::{
    inter_berth_filter_unchecked, intra_berth_shift_filter_unchecked,
    intra_berth_shift_to_head_filter_unchecked,
};
use super::scan::VesselMoves;
use crate::schedule::{LinkEnd, Mutator, ScheduleGraph};
use talos_core::SolverNumeric;
use talos_model::{Model, VesselIndex};

/// Moves `vessel` to the place `anchor` names: right after a vessel, or at the head of a berth.
///
/// # Safety
///
/// `vessel` and the vessel or berth of `anchor` must be the graph's of `mutator`.
#[inline(always)] // Called once per proposed move of both shift scans, and only forwards to one move.
unsafe fn insert_unchecked(mutator: &mut Mutator, vessel: VesselIndex, anchor: LinkEnd) {
    match anchor {
        LinkEnd::Vessel(anchor) => {
            debug_assert!(
                vessel.index() < mutator.graph().num_vessels()
                    && anchor.index() < mutator.graph().num_vessels(),
                "called `insert_unchecked` with {vessel} or {anchor}, outside a graph of {} vessels",
                mutator.graph().num_vessels()
            );
            // SAFETY: both vessels are the graph's, the precondition of this function.
            unsafe { mutator.relocate_after_unchecked(vessel, anchor) }
        }
        LinkEnd::Berth(berth) => {
            debug_assert!(
                vessel.index() < mutator.graph().num_vessels()
                    && berth.index() < mutator.graph().num_berths(),
                "called `insert_unchecked` with {vessel} or {berth}, outside a graph of {} vessels and {} berths",
                mutator.graph().num_vessels(),
                mutator.graph().num_berths()
            );
            // SAFETY: the vessel and the berth are the graph's, the precondition of this function.
            unsafe { mutator.relocate_to_head_unchecked(vessel, berth) }
        }
    }
}

/// Whether `vessel` and the vessel or berth of `anchor` are the graph's.
fn names_only_ids_of(graph: &ScheduleGraph, vessel: VesselIndex, anchor: LinkEnd) -> bool {
    let anchor_inside = match anchor {
        LinkEnd::Vessel(anchor) => anchor.index() < graph.num_vessels(),
        LinkEnd::Berth(berth) => berth.index() < graph.num_berths(),
    };
    vessel.index() < graph.num_vessels() && anchor_inside
}

/// The places of a vessel's own berth that an insertion on that berth leaves out: the one it holds, and the two
/// that would only exchange it with a neighbour.
///
/// Inserting a vessel at the place it holds, right after its predecessor or at the head it holds, changes nothing.
/// Inserting it right after its successor, or at the place before its predecessor, exchanges it with that
/// neighbour: the swap of two neighbours, which [`IntraBerthSwaps`](super::swap::IntraBerthSwaps) proposes, under a
/// filter that refuses only infeasible swaps unless both take no time to serve. Every other place moves the vessel
/// by at least two places, a rotation of three vessels or more that no other place and no swap makes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct NearPlaces {
    /// The place the vessel holds: right after its predecessor, or the head of its berth.
    held: LinkEnd,
    /// The place before its predecessor, or `None` if it heads its berth.
    before_predecessor: Option<LinkEnd>,
    /// Its successor, or `None` if it is the tail of its berth.
    successor: Option<VesselIndex>,
}

impl NearPlaces {
    /// The near places of `vessel`.
    ///
    /// # Safety
    ///
    /// `vessel` must be the graph's.
    #[inline(always)] // Called once per vessel of the scan on one berth, whose gathering is its only caller.
    unsafe fn of_unchecked(vessel: VesselIndex, graph: &ScheduleGraph) -> Self {
        debug_assert!(
            vessel.index() < graph.num_vessels(),
            "called `NearPlaces::of_unchecked` with {vessel}, outside a graph of {} vessels",
            graph.num_vessels()
        );
        // SAFETY: `vessel` is the graph's, the precondition of this function.
        let (berth, predecessor, successor) = unsafe {
            (
                graph.vessel_berth_unchecked(vessel),
                graph.vessel_predecessor_unchecked(vessel),
                graph.vessels_after_unchecked(vessel).next(),
            )
        };
        let place_after =
            |vessel: Option<VesselIndex>| vessel.map_or(LinkEnd::Berth(berth), LinkEnd::Vessel);
        let before_predecessor = predecessor.map(|predecessor| {
            debug_assert!(
                predecessor.index() < graph.num_vessels(),
                "called `NearPlaces::of_unchecked` with a graph whose {predecessor} is outside its {} vessels",
                graph.num_vessels()
            );
            // SAFETY: the predecessor was read from the graph's links, which name only its vessels
            // (`ScheduleGraph`).
            place_after(unsafe { graph.vessel_predecessor_unchecked(predecessor) })
        });
        Self {
            held: place_after(predecessor),
            before_predecessor,
            successor,
        }
    }

    /// Whether inserting the vessel at `place` is left out.
    #[inline(always)]
    fn contains(&self, place: LinkEnd) -> bool {
        place == self.held
            || Some(place) == self.before_predecessor
            || self
                .successor
                .is_some_and(|successor| place == LinkEnd::Vessel(successor))
    }
}

/// The insertions of a vessel at another place of its own berth, at least two places from where it stands: right
/// after each vessel of the berth whose window overlaps its own, in increasing order of that vessel, then at the
/// head before a vessel whose window overlaps its own.
///
/// A berth of `k` vessels offers its vessels `(k - 1) * (k - 2)` such places in all, the head among them, so the
/// neighbourhood holds at most the sum of that over the berths before its filter; each move touches one berth.
#[derive(Debug, Clone, Copy)]
pub(super) struct IntraBerthShifts;

impl VesselMoves for IntraBerthShifts {
    type Target = LinkEnd;

    #[inline]
    unsafe fn gather_unchecked<T>(
        vessel: VesselIndex,
        graph: &ScheduleGraph,
        model: &Model<T>,
        targets: &mut Vec<LinkEnd>,
    ) where
        T: SolverNumeric,
    {
        debug_assert!(
            vessel.index() < graph.num_vessels() && graph.num_vessels() == model.num_vessels(),
            "called `IntraBerthShifts::gather_unchecked` with {vessel} outside the graph, or a graph whose vessels \
             are not the model's"
        );
        // SAFETY: `vessel` is the graph's, the precondition of this function.
        let berth = unsafe { graph.vessel_berth_unchecked(vessel) };
        // SAFETY: `vessel` is the graph's, the precondition of this function.
        let near = unsafe { NearPlaces::of_unchecked(vessel, graph) };
        debug_assert!(
            berth.index() < graph.num_berths(),
            "called `IntraBerthShifts::gather_unchecked` with a graph that puts {vessel} on {berth}, outside its {} \
             berths",
            graph.num_berths()
        );
        // SAFETY: `berth` was read from the graph's column of berths, so it is one of its berths (`ScheduleGraph`).
        let berth_mates = unsafe { graph.vessel_sequence_iter_unchecked(berth) };
        let first = targets.len();
        let mut head = None;
        for anchor in berth_mates {
            head = head.or(Some(anchor));
            if anchor == vessel || near.contains(LinkEnd::Vessel(anchor)) {
                continue;
            }
            debug_assert!(
                anchor.index() < model.num_vessels(),
                "called `IntraBerthShifts::gather_unchecked` with a graph whose {anchor} is outside a model of {} \
                 vessels",
                model.num_vessels()
            );
            // SAFETY: `vessel` is the graph's by the precondition of this function, and the walk yields only the
            // graph's vessels (`VesselSequenceIter`); the graph's vessels are the model's, the precondition of this
            // function.
            if unsafe { intra_berth_shift_filter_unchecked(vessel, anchor, model) } {
                targets.push(LinkEnd::Vessel(anchor));
            }
        }
        // The walk gives the berth's order; the scan's order is that of the index.
        targets[first..].sort_unstable_by_key(|place| match place {
            LinkEnd::Vessel(anchor) => anchor.index(),
            LinkEnd::Berth(_) => usize::MAX,
        });
        // The berth holds `vessel`, so it has a head.
        if let Some(head) = head
            && !near.contains(LinkEnd::Berth(berth))
        {
            debug_assert!(
                head.index() < model.num_vessels(),
                "called `IntraBerthShifts::gather_unchecked` with a graph whose head {head} of {berth} is outside a \
                 model of {} vessels",
                model.num_vessels()
            );
            // SAFETY: `vessel` is the graph's by the precondition of this function, and the walk yields only the
            // graph's vessels (`VesselSequenceIter`); the graph's vessels are the model's, the precondition of this
            // function.
            if unsafe { intra_berth_shift_to_head_filter_unchecked(vessel, head, model) } {
                targets.push(LinkEnd::Berth(berth));
            }
        }
    }

    #[inline(always)] // Called once per proposed move, and only forwards to one move.
    unsafe fn make_unchecked(mutator: &mut Mutator, vessel: VesselIndex, anchor: LinkEnd) {
        debug_assert!(
            names_only_ids_of(mutator.graph(), vessel, anchor),
            "called `IntraBerthShifts::make_unchecked` with {vessel} or {anchor}, outside the graph"
        );
        // SAFETY: the vessel and the place are the graph's, the precondition of this function.
        unsafe { insert_unchecked(mutator, vessel, anchor) }
    }
}

/// The insertions of a vessel on each other berth that it may use: right after each vessel of such a berth, in
/// increasing order of that vessel, then at the head of each such berth, an empty one included, in increasing
/// order of the berth.
///
/// A vessel on a berth of `k` vessels has `n - k` vessels and `m - 1` heads of other berths to go to, in a schedule
/// of `n` vessels and `m` berths, so the neighbourhood holds at most `n * (n - 1)`, less the sum of `k * (k - 1)`
/// over the berths, plus `n * (m - 1)` moves before its filter; each move touches two berths.
#[derive(Debug, Clone, Copy)]
pub(super) struct InterBerthShifts;

impl VesselMoves for InterBerthShifts {
    type Target = LinkEnd;

    #[inline]
    unsafe fn gather_unchecked<T>(
        vessel: VesselIndex,
        graph: &ScheduleGraph,
        model: &Model<T>,
        targets: &mut Vec<LinkEnd>,
    ) where
        T: SolverNumeric,
    {
        debug_assert!(
            vessel.index() < graph.num_vessels()
                && graph.num_vessels() == model.num_vessels()
                && graph.num_berths() == model.num_berths(),
            "called `InterBerthShifts::gather_unchecked` with {vessel} outside the graph, or a graph whose counts \
             are not the model's"
        );
        // SAFETY: `vessel` is the graph's, the precondition of this function.
        let own = unsafe { graph.vessel_berth_unchecked(vessel) };
        for anchor in graph.vessels() {
            debug_assert!(
                anchor.index() < graph.num_vessels(),
                "called `InterBerthShifts::gather_unchecked` with {anchor}, beyond the graph's {} vessels",
                graph.num_vessels()
            );
            // SAFETY: `anchor` is below the graph's count of vessels, the bound of the walk.
            let berth = unsafe { graph.vessel_berth_unchecked(anchor) };
            debug_assert!(
                berth.index() < model.num_berths(),
                "called `InterBerthShifts::gather_unchecked` with a graph that puts {anchor} on {berth}, outside the \
                 model"
            );
            // SAFETY: `vessel` is the graph's by the precondition of this function and `berth` was read from its
            // column of berths (`ScheduleGraph`); the graph's vessels and berths are the model's, the precondition of
            // this function.
            if berth != own && unsafe { inter_berth_filter_unchecked(vessel, berth, model) } {
                targets.push(LinkEnd::Vessel(anchor));
            }
        }
        for berth in graph.berths() {
            debug_assert!(
                berth.index() < model.num_berths(),
                "called `InterBerthShifts::gather_unchecked` with a graph of more berths than the model"
            );
            // SAFETY: `vessel` is the graph's by the precondition of this function and `berth` one of its berths,
            // and the graph's vessels and berths are the model's, the precondition of this function.
            if berth != own && unsafe { inter_berth_filter_unchecked(vessel, berth, model) } {
                targets.push(LinkEnd::Berth(berth));
            }
        }
    }

    #[inline(always)] // Called once per proposed move, and only forwards to one move.
    unsafe fn make_unchecked(mutator: &mut Mutator, vessel: VesselIndex, anchor: LinkEnd) {
        debug_assert!(
            names_only_ids_of(mutator.graph(), vessel, anchor),
            "called `InterBerthShifts::make_unchecked` with {vessel} or {anchor}, outside the graph"
        );
        // SAFETY: the vessel and the place are the graph's, the precondition of this function.
        unsafe { insert_unchecked(mutator, vessel, anchor) }
    }
}

#[cfg(test)]
mod tests {
    use super::super::reference::{self, OVERLAPPING_WINDOWS, WINDOWS};
    use super::*;
    use talos_model::BerthIndex;

    fn vessel(index: usize) -> VesselIndex {
        VesselIndex::new(index)
    }

    /// The near places of `index` in the start of the reference.
    fn near_places_of(index: usize) -> NearPlaces {
        let graph = reference::start_graph();
        // SAFETY: every vessel of the reference is the graph's.
        unsafe { NearPlaces::of_unchecked(vessel(index), &graph) }
    }

    // On B0: V0 -> V1 -> V2 -> V3 -> V4, and V5 alone on B1.
    #[test]
    fn test_the_near_places_are_the_held_one_and_those_of_a_swap_with_a_neighbour() {
        let head = LinkEnd::Berth(BerthIndex::new(0));
        let after = |index| LinkEnd::Vessel(vessel(index));
        let expected = [
            (0, head, None, Some(1)),
            (1, after(0), Some(head), Some(2)),
            (2, after(1), Some(after(0)), Some(3)),
            (4, after(3), Some(after(2)), None),
        ];
        for (index, held, before_predecessor, successor) in expected {
            let near = near_places_of(index);
            assert_eq!(
                near,
                NearPlaces {
                    held,
                    before_predecessor,
                    successor: successor.map(vessel)
                },
                "the near places of V{index}"
            );
        }
        let alone = near_places_of(5);
        assert!(alone.contains(LinkEnd::Berth(BerthIndex::new(1))));
        assert!(!alone.contains(LinkEnd::Vessel(vessel(4))));
        let second = near_places_of(1);
        for place in [head, after(0), after(2)] {
            assert!(second.contains(place), "V1 should leave out {place}");
        }
        assert!(!second.contains(after(3)));
    }

    // At least two places along the own berth, the head last, once each in the order of the anchors; refused
    // only where V4 would precede or follow V0. By hand: V0 after V2 and V3; V1 after V3 and V4; V2 to the head and
    // after V4; V3 to the head and after V0; V4 after V1. The berths of one vessel offer none.
    #[test]
    fn test_the_intra_berth_scan_proposes_every_far_insertion_on_the_own_berth_once_in_order() {
        let moves = reference::moves_of::<IntraBerthShifts>(&reference::model(&WINDOWS));
        assert_eq!(moves, reference::intra_berth_shifts(&WINDOWS));
        assert_eq!(moves.len(), 9);
        let all = reference::moves_of::<IntraBerthShifts>(&reference::model(&OVERLAPPING_WINDOWS));
        assert_eq!(all, reference::intra_berth_shifts(&OVERLAPPING_WINDOWS));
        // A berth of five vessels offers (5 - 1) * (5 - 2) such places in all.
        assert_eq!(all.len(), 12);
    }

    // Every place of every other berth the vessel may use, once each, after the anchors in increasing order and then
    // at the heads, the empty B2 and B4 included: six for V0, V1, V2 and V4, five for V3, which may not use B2, four
    // for V5, which may not use B0, and nine for V6, which may not use B4.
    #[test]
    fn test_the_inter_berth_scan_proposes_every_insertion_on_another_allowed_berth_once_in_order() {
        let moves = reference::moves_of::<InterBerthShifts>(&reference::model(&WINDOWS));
        assert_eq!(moves, reference::inter_berth_shifts());
        assert_eq!(moves.len(), 4 * 6 + 5 + 4 + 9);
        assert!(
            moves.contains(&vec![vec![1, 2, 3, 4], vec![5], vec![0], vec![6], vec![]]),
            "V0 should reach the empty B2"
        );
    }
}
