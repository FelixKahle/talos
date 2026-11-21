// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The neighbourhoods that insert one vessel at another place: on its own berth, or on another.
//!
//! A place is where a link of a berth starts: right after a vessel, or at the head of a berth. Every place of
//! every berth is so reachable, the head of a berth and an empty berth included.

use super::filter::{
    inter_berth_shift_filter_unchecked, intra_berth_shift_filter_unchecked,
    intra_berth_shift_to_head_filter_unchecked,
};
use crate::schedule::{LinkEnd, Mutator, ScheduleGraph};
use std::iter::FusedIterator;
use talos_core::SolverNumeric;
use talos_model::{BerthIndex, Model, VesselIndex};

/// The placements of a graph of `num_vessels` vessels and `num_berths` berths: each vessel paired with every place
/// it could be inserted at, right after every other vessel and then at the head of every berth, in increasing
/// order of the vessel, then of the place, the vessels before the berths.
#[derive(Debug, Clone)]
struct PlacementsIter {
    /// The vessels of the graph.
    num_vessels: usize,
    /// The berths of the graph.
    num_berths: usize,
    /// The vessel of the next placement, or `num_vessels` once every placement is given.
    vessel: usize,
    /// The place of the next placement: the vessel of this index below `num_vessels`, the head of berth
    /// `place - num_vessels` from there, and `num_vessels + num_berths` once the placements of `vessel` are given.
    place: usize,
}

impl PlacementsIter {
    /// The placements of a graph of `num_vessels` vessels and `num_berths` berths, from the first.
    fn new(num_vessels: usize, num_berths: usize) -> Self {
        Self {
            num_vessels,
            num_berths,
            vessel: 0,
            place: 0,
        }
    }
}

impl Iterator for PlacementsIter {
    type Item = (VesselIndex, LinkEnd);

    fn next(&mut self) -> Option<Self::Item> {
        let num_places = self.num_vessels + self.num_berths;
        // Each pass either gives a placement or steps past the vessel itself or to the next vessel, so the loop
        // ends within `num_places + 2` passes.
        for _ in 0..num_places + 2 {
            if self.vessel >= self.num_vessels {
                return None;
            }
            if self.place >= num_places {
                self.vessel += 1;
                self.place = 0;
                continue;
            }
            let place = self.place;
            self.place += 1;
            if place == self.vessel {
                continue;
            }
            let anchor = match place.checked_sub(self.num_vessels) {
                None => LinkEnd::Vessel(VesselIndex::new(place)),
                Some(berth) => LinkEnd::Berth(BerthIndex::new(berth)),
            };
            return Some((VesselIndex::new(self.vessel), anchor));
        }
        unreachable!("a placement or the end should be found within `num_places + 2` passes")
    }
}

impl FusedIterator for PlacementsIter {}

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
                    && anchor.index() < mutator.graph().num_vessels()
            );
            // SAFETY: both vessels are the graph's, the precondition of this function.
            unsafe { mutator.relocate_after_unchecked(vessel, anchor) }
        }
        LinkEnd::Berth(berth) => {
            debug_assert!(
                vessel.index() < mutator.graph().num_vessels()
                    && berth.index() < mutator.graph().num_berths()
            );
            // SAFETY: the vessel and the berth are the graph's, the precondition of this function.
            unsafe { mutator.relocate_to_head_unchecked(vessel, berth) }
        }
    }
}

/// The insertions of a vessel at another place of its own berth, right after another vessel whose window overlaps
/// its own or at the head before a vessel whose window overlaps its own, in the order of the placements.
///
/// A berth of `k` vessels offers each of them `k - 1` places, so the neighbourhood holds at most the sum of
/// `k * (k - 1)` over the berths before its filter; each move touches one berth.
#[derive(Debug, Clone)]
pub(super) struct IntraBerthShiftOperator {
    placements: PlacementsIter,
}

impl IntraBerthShiftOperator {
    /// The operator, which proposes no move until it is prepared for a graph.
    pub(super) fn new() -> Self {
        Self {
            placements: PlacementsIter::new(0, 0),
        }
    }

    /// Starts the neighbourhood of `graph` from its first move.
    pub(super) fn prepare(&mut self, graph: &ScheduleGraph) {
        self.placements = PlacementsIter::new(graph.num_vessels(), graph.num_berths());
    }

    /// Makes the next move of the neighbourhood on the graph of `mutator` and answers `true`, or answers `false`
    /// and changes nothing once the neighbourhood is exhausted.
    ///
    /// # Safety
    ///
    /// The graph of `mutator` must count the vessels and berths of the graph the operator was last prepared for,
    /// and have the vessels and berths of `model`.
    pub(super) unsafe fn next_neighbor_unchecked<T>(
        &mut self,
        model: &Model<T>,
        mutator: &mut Mutator,
    ) -> bool
    where
        T: SolverNumeric,
    {
        for (vessel, anchor) in &mut self.placements {
            // SAFETY: the placements count the vessels and berths of the graph, which are the model's, the
            // precondition of this function.
            if unsafe { Self::proposes_unchecked(vessel, anchor, mutator.graph(), model) } {
                // SAFETY: the placements count the vessels and berths of the graph, the precondition of this
                // function.
                unsafe { insert_unchecked(mutator, vessel, anchor) };
                return true;
            }
        }
        false
    }

    /// Whether the neighbourhood proposes inserting `vessel` at `anchor`: a place of its own berth it does not
    /// already hold, beside a vessel whose window overlaps its own.
    ///
    /// # Safety
    ///
    /// `vessel` and the vessel or berth of `anchor` must be the graph's, and the graph must have the vessels and
    /// berths of `model`.
    #[inline(always)] // Called once per placement of the scan, and the scan's loop is its only caller.
    unsafe fn proposes_unchecked<T>(
        vessel: VesselIndex,
        anchor: LinkEnd,
        graph: &ScheduleGraph,
        model: &Model<T>,
    ) -> bool
    where
        T: SolverNumeric,
    {
        debug_assert!(
            vessel.index() < graph.num_vessels()
                && graph.num_vessels() == model.num_vessels()
                && graph.num_berths() == model.num_berths(),
            "called `IntraBerthShiftOperator::proposes_unchecked` with {vessel} outside the graph, or a graph \
             whose counts are not the model's"
        );
        // SAFETY: `vessel` is the graph's, the precondition of this function.
        let (berth, predecessor) = unsafe {
            (
                graph.vessel_berth_unchecked(vessel),
                graph.vessel_predecessor_unchecked(vessel),
            )
        };
        match anchor {
            LinkEnd::Vessel(anchor) => {
                debug_assert!(anchor.index() < graph.num_vessels());
                // SAFETY: `anchor` is the graph's, the precondition of this function.
                let anchor_berth = unsafe { graph.vessel_berth_unchecked(anchor) };
                if anchor_berth != berth || predecessor == Some(anchor) {
                    return false;
                }
                debug_assert!(anchor.index() < model.num_vessels());
                // SAFETY: both vessels are the graph's, whose vessels are the model's, the precondition of this
                // function.
                unsafe { intra_berth_shift_filter_unchecked(vessel, anchor, model) }
            }
            LinkEnd::Berth(target) => {
                if target != berth || predecessor.is_none() {
                    return false;
                }
                debug_assert!(berth.index() < graph.num_berths());
                // SAFETY: `berth` was read from the graph's column of berths, so it is one of its berths
                // (`ScheduleGraph`).
                let head = unsafe { graph.vessel_sequence_iter_unchecked(berth) }.next();
                // The berth holds `vessel`, so it has a head; the walk yields only vessels of the graph, whose
                // vessels are the model's, the precondition of this function.
                head.is_some_and(|head| {
                    debug_assert!(head.index() < model.num_vessels());
                    // SAFETY: both vessels are the model's, as above.
                    unsafe { intra_berth_shift_to_head_filter_unchecked(vessel, head, model) }
                })
            }
        }
    }
}

/// The insertions of a vessel on another berth that it may use, right after a vessel of that berth or at its
/// head, in the order of the placements.
///
/// A vessel on a berth of `k` vessels has `n - k` vessels and `m - 1` heads of other berths to go to, in a schedule
/// of `n` vessels and `m` berths, so the neighbourhood holds at most `n * (n - 1)`, less the sum of `k * (k - 1)`
/// over the berths, plus `n * (m - 1)` moves before its filter; each move touches two berths.
#[derive(Debug, Clone)]
pub(super) struct InterBerthShiftOperator {
    placements: PlacementsIter,
}

impl InterBerthShiftOperator {
    /// The operator, which proposes no move until it is prepared for a graph.
    pub(super) fn new() -> Self {
        Self {
            placements: PlacementsIter::new(0, 0),
        }
    }

    /// Starts the neighbourhood of `graph` from its first move.
    pub(super) fn prepare(&mut self, graph: &ScheduleGraph) {
        self.placements = PlacementsIter::new(graph.num_vessels(), graph.num_berths());
    }

    /// Makes the next move of the neighbourhood on the graph of `mutator` and answers `true`, or answers `false`
    /// and changes nothing once the neighbourhood is exhausted.
    ///
    /// # Safety
    ///
    /// The graph of `mutator` must count the vessels and berths of the graph the operator was last prepared for,
    /// and have the vessels and berths of `model`.
    pub(super) unsafe fn next_neighbor_unchecked<T>(
        &mut self,
        model: &Model<T>,
        mutator: &mut Mutator,
    ) -> bool
    where
        T: SolverNumeric,
    {
        for (vessel, anchor) in &mut self.placements {
            // SAFETY: the placements count the vessels and berths of the graph, which are the model's, the
            // precondition of this function.
            if unsafe { Self::proposes_unchecked(vessel, anchor, mutator.graph(), model) } {
                // SAFETY: the placements count the vessels and berths of the graph, the precondition of this
                // function.
                unsafe { insert_unchecked(mutator, vessel, anchor) };
                return true;
            }
        }
        false
    }

    /// Whether the neighbourhood proposes inserting `vessel` at `anchor`: a place of another berth, which
    /// `vessel` may use.
    ///
    /// # Safety
    ///
    /// `vessel` and the vessel or berth of `anchor` must be the graph's, and the graph must have the vessels and
    /// berths of `model`.
    #[inline(always)] // Called once per placement of the scan, and the scan's loop is its only caller.
    unsafe fn proposes_unchecked<T>(
        vessel: VesselIndex,
        anchor: LinkEnd,
        graph: &ScheduleGraph,
        model: &Model<T>,
    ) -> bool
    where
        T: SolverNumeric,
    {
        debug_assert!(
            vessel.index() < graph.num_vessels()
                && graph.num_vessels() == model.num_vessels()
                && graph.num_berths() == model.num_berths(),
            "called `InterBerthShiftOperator::proposes_unchecked` with {vessel} outside the graph, or a graph \
             whose counts are not the model's"
        );
        let target = match anchor {
            LinkEnd::Vessel(anchor) => {
                debug_assert!(anchor.index() < graph.num_vessels());
                // SAFETY: `anchor` is the graph's, the precondition of this function.
                unsafe { graph.vessel_berth_unchecked(anchor) }
            }
            LinkEnd::Berth(berth) => berth,
        };
        // SAFETY: `vessel` is the graph's, the precondition of this function.
        if unsafe { graph.vessel_berth_unchecked(vessel) } == target {
            return false;
        }
        debug_assert!(target.index() < model.num_berths());
        // SAFETY: `vessel` is the graph's and `target` one of its berths, read from its column of berths or the
        // precondition of this function, and the graph's vessels and berths are the model's.
        unsafe { inter_berth_shift_filter_unchecked(vessel, target, model) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use talos_core::ClosedOpenInterval;

    fn vessel(index: usize) -> LinkEnd {
        LinkEnd::Vessel(VesselIndex::new(index))
    }

    fn head(index: usize) -> LinkEnd {
        LinkEnd::Berth(BerthIndex::new(index))
    }

    #[test]
    fn test_placements_pair_each_vessel_with_every_other_vessel_and_every_head_in_order() {
        let placements: Vec<(usize, LinkEnd)> = PlacementsIter::new(3, 2)
            .map(|(moved, anchor)| (moved.index(), anchor))
            .collect();
        let expected = [
            (0, vessel(1)),
            (0, vessel(2)),
            (0, head(0)),
            (0, head(1)),
            (1, vessel(0)),
            (1, vessel(2)),
            (1, head(0)),
            (1, head(1)),
            (2, vessel(0)),
            (2, vessel(1)),
            (2, head(0)),
            (2, head(1)),
        ];
        assert_eq!(placements, expected);
    }

    #[test]
    fn test_placements_of_one_vessel_are_the_heads_and_of_none_are_none() {
        let placements: Vec<_> = PlacementsIter::new(1, 2).collect();
        assert_eq!(
            placements,
            [
                (VesselIndex::new(0), head(0)),
                (VesselIndex::new(0), head(1))
            ]
        );
        assert_eq!(PlacementsIter::new(0, 3).next(), None);
        assert_eq!(PlacementsIter::new(1, 0).next(), None);
    }

    #[test]
    fn test_placements_stay_exhausted() {
        let mut placements = PlacementsIter::new(2, 1);
        assert_eq!(placements.by_ref().count(), 4);
        assert_eq!(placements.next(), None);
        assert_eq!(placements.next(), None);
    }

    /// The vessels of each berth of `graph` in order.
    fn order_of(graph: &ScheduleGraph) -> Vec<Vec<usize>> {
        graph
            .berths()
            .map(|berth| {
                graph
                    .vessel_sequence_iter(berth)
                    .map(VesselIndex::index)
                    .collect()
            })
            .collect()
    }

    /// The order of `berths` with `moved` taken out and inserted at `anchor`.
    fn inserted(berths: &[Vec<usize>], moved: usize, anchor: LinkEnd) -> Vec<Vec<usize>> {
        let mut berths = berths.to_vec();
        for vessels in &mut berths {
            vessels.retain(|&other| other != moved);
        }
        match anchor {
            LinkEnd::Vessel(anchor) => {
                for vessels in &mut berths {
                    if let Some(place) = vessels.iter().position(|&other| other == anchor.index()) {
                        vessels.insert(place + 1, moved);
                    }
                }
            }
            LinkEnd::Berth(berth) => berths[berth.index()].insert(0, moved),
        }
        berths
    }

    /// The window of each vessel of [`five_vessels`], from its arrival to its latest departure.
    const WINDOWS: [(i64, i64); 5] = [(0, 400), (0, 1000), (0, 1000), (0, 1000), (500, 600)];

    /// Five vessels of the windows [`WINDOWS`] on four berths open on [0, 1000), each taking 10 where it may
    /// berth; V3 may not use B2.
    fn five_vessels() -> Model<i64> {
        let mut processing = vec![Some(10); 20];
        processing[3 * 4 + 2] = None;
        Model::new(
            WINDOWS.map(|(arrival, _)| arrival).to_vec(),
            WINDOWS.map(|(_, departure)| departure).to_vec(),
            vec![1; 5],
            processing,
            vec![vec![ClosedOpenInterval::new(0, 1000)]; 4],
        )
    }

    /// Whether the windows of `a` and `b` overlap.
    fn overlap(a: usize, b: usize) -> bool {
        WINDOWS[a].0 < WINDOWS[b].1 && WINDOWS[b].0 < WINDOWS[a].1
    }

    /// B0: V0 -> V1 -> V2 -> V4, B1: V3, B2 and B3 empty. V4's window misses V0's alone.
    fn five_vessels_graph() -> ScheduleGraph {
        let berths = [0, 0, 0, 1, 0].map(BerthIndex::new);
        ScheduleGraph::from_slices(&berths, &[0_i64, 10, 20, 0, 500], 4)
    }

    /// Every move of a scan of `next` from a fresh mutator of the graph, as the order each makes; each is undone
    /// before the next, as a run undoes a rejected candidate.
    fn moves_of(mut next: impl FnMut(&mut Mutator) -> bool) -> Vec<Vec<Vec<usize>>> {
        let mut mutator = Mutator::new(five_vessels_graph());
        let mut moves = Vec::new();
        // The scan holds at most one move per placement, 5 * (5 - 1 + 4) of them.
        for _ in 0..=40 {
            if !next(&mut mutator) {
                return moves;
            }
            moves.push(order_of(mutator.graph()));
            mutator.rollback();
        }
        panic!("the scan should end within its 40 placements")
    }

    /// The orders of the insertions of the placements that `keep` keeps, in the order of the placements.
    fn expected_moves(keep: impl Fn(usize, LinkEnd) -> bool) -> Vec<Vec<Vec<usize>>> {
        let start = order_of(&five_vessels_graph());
        PlacementsIter::new(5, 4)
            .filter(|&(moved, anchor)| keep(moved.index(), anchor))
            .map(|(moved, anchor)| inserted(&start, moved.index(), anchor))
            .collect()
    }

    /// The berth of each vessel of the start, and whether a vessel heads its berth there.
    const START_BERTHS: [usize; 5] = [0, 0, 0, 1, 0];
    const START_PREDECESSORS: [Option<usize>; 5] = [None, Some(0), Some(1), None, Some(2)];

    // Every placement on the vessel's own berth that changes the order, once, in the order of the placements, the
    // head included; refused only where V4's window misses that of V0, the anchor or the head it would precede.
    #[test]
    fn test_the_intra_berth_scan_proposes_every_insertion_on_the_own_berth_once_in_order() {
        let model = five_vessels();
        let mut operator = IntraBerthShiftOperator::new();
        operator.prepare(&five_vessels_graph());
        // SAFETY: the mutator's graph has the counts the operator was prepared for, which are the model's.
        let moves =
            moves_of(|mutator| unsafe { operator.next_neighbor_unchecked(&model, mutator) });
        let expected = expected_moves(|moved, anchor| {
            let berth = START_BERTHS[moved];
            match anchor {
                LinkEnd::Vessel(anchor) => {
                    let anchor = anchor.index();
                    START_BERTHS[anchor] == berth
                        && START_PREDECESSORS[moved] != Some(anchor)
                        && overlap(moved, anchor)
                }
                LinkEnd::Berth(target) => {
                    target.index() == berth
                        && START_PREDECESSORS[moved].is_some()
                        && overlap(moved, 0)
                }
            }
        });
        assert_eq!(moves, expected);
        assert!(
            moves.iter().any(|order| order[0][0] == 1),
            "V1 should reach the head of B0"
        );
        assert!(
            !moves.iter().any(|order| order[0][0] == 4),
            "V4 should not be set before V0"
        );
        assert!(
            !moves.iter().any(|order| order[0] == [1, 2, 4, 0]),
            "V0 should not be set after V4"
        );
    }

    // Every placement on another berth the vessel may use, once, in the order of the placements: after a vessel of
    // B1, and at the heads of B1, of the empty B2 and B3, and of B0 for V3, which may not use B2.
    #[test]
    fn test_the_inter_berth_scan_proposes_every_insertion_on_another_allowed_berth_once_in_order() {
        let model = five_vessels();
        let mut operator = InterBerthShiftOperator::new();
        operator.prepare(&five_vessels_graph());
        // SAFETY: the mutator's graph has the counts the operator was prepared for, which are the model's.
        let moves =
            moves_of(|mutator| unsafe { operator.next_neighbor_unchecked(&model, mutator) });
        let expected = expected_moves(|moved, anchor| {
            let target = match anchor {
                LinkEnd::Vessel(anchor) => START_BERTHS[anchor.index()],
                LinkEnd::Berth(target) => target.index(),
            };
            target != START_BERTHS[moved] && !(moved == 3 && target == 2)
        });
        assert_eq!(moves, expected);
        assert_eq!(moves.len(), 4 * (1 + 3) + (4 + 2));
        assert!(
            moves.iter().any(|order| order[2] == [0]),
            "V0 should reach the empty B2"
        );
    }

    // A scan prepared anew starts again from its first move, whatever the cursor had reached.
    #[test]
    fn test_prepare_restarts_the_scan_from_its_first_move() {
        let model = five_vessels();
        let mut operator = InterBerthShiftOperator::new();
        operator.prepare(&five_vessels_graph());
        let mut mutator = Mutator::new(five_vessels_graph());
        // SAFETY: the mutator's graph has the counts the operator was prepared for, which are the model's.
        let first = unsafe { operator.next_neighbor_unchecked(&model, &mut mutator) };
        let first_order = order_of(mutator.graph());
        mutator.rollback();
        for _ in 0..5 {
            // SAFETY: as above.
            assert!(unsafe { operator.next_neighbor_unchecked(&model, &mut mutator) });
            mutator.rollback();
        }
        operator.prepare(&five_vessels_graph());
        // SAFETY: as above.
        let again = unsafe { operator.next_neighbor_unchecked(&model, &mut mutator) };
        assert!(first && again);
        assert_eq!(order_of(mutator.graph()), first_order);
    }

    #[test]
    fn test_a_new_operator_proposes_nothing_until_it_is_prepared() {
        let model = five_vessels();
        let mut mutator = Mutator::new(five_vessels_graph());
        // SAFETY: an operator that was never prepared counts no vessel, so it reads nothing of the graph.
        unsafe {
            assert!(!IntraBerthShiftOperator::new().next_neighbor_unchecked(&model, &mut mutator));
            assert!(!InterBerthShiftOperator::new().next_neighbor_unchecked(&model, &mut mutator));
        }
        assert_eq!(mutator.graph(), &five_vessels_graph());
    }
}
