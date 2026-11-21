// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The moves of the local search on a schedule graph, each recorded as it is made.

use super::diff::ScheduleGraphDiff;
use super::graph::ScheduleGraph;
use super::touched_berths::TouchedBerths;
use super::undo_log::ScheduleGraphUndoLog;
use crate::ring_arena::Node;
use talos_model::{BerthIndex, VesselIndex};

// ----------------------------------------------------------------
// LinkSnapshot
// ----------------------------------------------------------------

/// The successors of up to `N` nodes before a move, from which the links the move changed are read after it.
#[derive(Clone, Copy, Debug)]
struct LinkSnapshot<const N: usize> {
    /// The nodes whose outgoing link a move may change; one may stand more than once.
    nodes: [Node; N],
    /// The successor of each node before the move, per node of `nodes`.
    successors: [Node; N],
}

impl<const N: usize> LinkSnapshot<N> {
    /// The successors of `nodes` in `graph` as it stands.
    ///
    /// # Safety
    ///
    /// Every node must be a vessel's node or a berth's sentinel of `graph`.
    #[inline(always)]
    unsafe fn new_unchecked(graph: &ScheduleGraph, nodes: [Node; N]) -> Self {
        // SAFETY: every node is the graph's, the precondition of this function.
        let successors = nodes.map(|node| unsafe { graph.next_node_unchecked(node) });
        Self { nodes, successors }
    }

    /// Records in `diff` every link of the snapshot's nodes that differs in `graph` now, once per node.
    ///
    /// # Safety
    ///
    /// Every node of the snapshot must be a vessel's node or a berth's sentinel of `graph`, which holds when it is
    /// the graph the snapshot was taken of, with its counts unchanged.
    #[inline(always)]
    unsafe fn commit_unchecked(&self, graph: &ScheduleGraph, diff: &mut ScheduleGraphDiff) {
        for (index, (&node, &before)) in self.nodes.iter().zip(&self.successors).enumerate() {
            if self.nodes[..index].contains(&node) {
                continue;
            }
            // SAFETY: `node` is the graph's, the precondition of this function.
            let after = unsafe { graph.next_node_unchecked(node) };
            if before != after {
                let from = graph.link_end(node);
                diff.push_link_broken(from, graph.link_end(before));
                diff.push_link_created(from, graph.link_end(after));
            }
        }
    }
}

// ----------------------------------------------------------------
// Move
// ----------------------------------------------------------------

/// A safe move of a [`Mutator`], as its refusal names it.
#[cfg(test)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Move {
    SwapVessels,
    RelocateAfter,
    RelocateToHead,
}

#[cfg(test)]
impl std::fmt::Display for Move {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::SwapVessels => "swap_vessels",
            Self::RelocateAfter => "relocate_after",
            Self::RelocateToHead => "relocate_to_head",
        };
        write!(f, "`Mutator::{name}`")
    }
}

// ----------------------------------------------------------------
// Mutator
// ----------------------------------------------------------------

/// A [`ScheduleGraph`] and the moves of the local search on it, each recorded as it is made.
///
/// Every move is recorded three ways: its inverse in an undo log, so that a rejected candidate is undone; the
/// links it broke and made and the vessels it moved between berths in a [`ScheduleGraphDiff`], which the
/// metaheuristics read; and the berths whose order it changed among the [`TouchedBerths`], which the decoder
/// decodes again. A move that changes nothing records nothing where its doc says so.
///
/// The mutator owns the graph and its records together, so that they always speak of one graph: the touched
/// berths count its berths, and the undo log names only its vessels and berths. A run keeps one mutator, makes a
/// candidate's moves, and then either [`accept`](Self::accept)s them or [`rollback`](Self::rollback)s them.
///
/// Each move is made through its `_unchecked` form, by the operator whose loop proves that its vessels are the
/// graph's; its safe form, which checks them in every build, is the reference the tests check it against.
// Invariant: `touched` counts the berths of `graph`, and every vessel and berth `undo` names is one of `graph`'s.
// The unchecked marks of a berth and the unchecked rollback rest on these. Only the moves write `undo`, each with
// ids of the graph, and every step that changes the graph's counts (`overwrite_*`) empties `undo` and sizes
// `touched` anew with it.
#[derive(Clone, Debug)]
pub struct Mutator {
    /// The graph the moves change.
    graph: ScheduleGraph,
    /// The inverse of every move made since the moves were last accepted or rolled back, the latest last.
    undo: ScheduleGraphUndoLog,
    /// The links and berths the moves changed since the diff was last cleared.
    diff: ScheduleGraphDiff,
    /// The berths whose order the moves changed since they were last accepted or rolled back.
    touched: TouchedBerths,
}

impl Mutator {
    /// The mutator of `graph`, with empty records.
    #[inline]
    pub fn new(graph: ScheduleGraph) -> Self {
        let touched = TouchedBerths::new(graph.num_berths());
        Self {
            graph,
            undo: ScheduleGraphUndoLog::default(),
            diff: ScheduleGraphDiff::new(),
            touched,
        }
    }

    /// The graph the moves change.
    #[inline(always)]
    pub fn graph(&self) -> &ScheduleGraph {
        &self.graph
    }

    /// The links and berths the moves changed since the diff was last cleared.
    #[inline(always)]
    pub fn diff(&self) -> &ScheduleGraphDiff {
        &self.diff
    }

    /// The berths whose order the moves changed since they were last accepted or rolled back; they count the
    /// berths of the graph.
    #[inline(always)]
    pub fn touched(&self) -> &TouchedBerths {
        &self.touched
    }

    /// Empties the diff, keeping its memory.
    #[inline(always)]
    pub fn clear_diff(&mut self) {
        self.diff.clear();
    }

    /// Keeps the moves made since they were last accepted or rolled back: forgets how to undo them and which
    /// berths they touched, at a cost of the touched berths. The diff stays until it is cleared.
    #[inline]
    pub fn accept(&mut self) {
        self.undo.clear();
        self.touched.reset();
    }

    /// Undoes every move made since the moves were last accepted or rolled back, the latest first, and unmarks
    /// the berths they touched. Afterwards the graph stands as it stood then; the diff stays until it is cleared.
    #[inline]
    pub fn rollback(&mut self) {
        // SAFETY: every id the log names is one of the graph's, by the invariant of `Mutator`.
        unsafe { self.undo.apply_rollback_unchecked(&mut self.graph) };
        self.touched.reset();
    }

    /// Makes the graph the one [`ScheduleGraph::from_slices`] builds from the same arguments and empties the
    /// records, reusing the memory of both.
    ///
    /// # Panics
    ///
    /// Panics if [`ScheduleGraph::overwrite_from_slices`] refuses the arguments; the mutator is then left
    /// unchanged.
    #[track_caller]
    pub fn overwrite_from_slices<T>(
        &mut self,
        berths: &[BerthIndex],
        start_times: &[T],
        num_berths: usize,
    ) where
        T: Ord,
    {
        self.graph
            .overwrite_from_slices(berths, start_times, num_berths);
        self.empty_records();
    }

    /// Makes the graph a copy of `other` and empties the records, reusing the memory of both.
    #[inline]
    pub fn overwrite_from_graph(&mut self, other: &ScheduleGraph) {
        self.graph.overwrite_from_graph(other);
        self.empty_records();
    }

    /// Empties the records and sizes the touched berths for the graph's berths, as a change of the graph's counts
    /// asks.
    #[inline]
    fn empty_records(&mut self) {
        self.undo.clear();
        self.diff.clear();
        self.touched.resize(self.graph.num_berths());
    }

    /// Panics with the refusal of `step` if one of `vessels` or `berths` is not one of the graph's.
    #[inline]
    #[track_caller]
    #[cfg(test)]
    fn check<const V: usize, const B: usize>(
        &self,
        step: Move,
        vessels: [VesselIndex; V],
        berths: [BerthIndex; B],
    ) {
        if let Some(refusal) = self.graph.refusal(step, vessels, berths) {
            refusal.panic();
        }
    }

    /// Exchanges the places of `a` and `b`, on one berth or across two; nothing changes and nothing is recorded if
    /// they are one vessel.
    ///
    /// ```text
    /// before:  ... prev_a -> a -> next_a ...      ... prev_b -> b -> next_b ...
    /// after:   ... prev_a -> b -> next_a ...      ... prev_b -> a -> next_b ...
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if any of `a` and `b` is not one of the graph's.
    #[inline]
    #[track_caller]
    #[cfg(test)]
    pub fn swap_vessels(&mut self, a: VesselIndex, b: VesselIndex) {
        self.check(Move::SwapVessels, [a, b], []);
        // SAFETY: `a` and `b` are the graph's, as the check above found.
        unsafe { self.swap_vessels_unchecked(a, b) }
    }

    /// The move [`swap_vessels`](Self::swap_vessels) makes, for ids the caller knows to be the graph's.
    ///
    /// # Safety
    ///
    /// `a` and `b` must be the graph's, the condition [`swap_vessels`](Self::swap_vessels) checks.
    #[inline]
    pub unsafe fn swap_vessels_unchecked(&mut self, a: VesselIndex, b: VesselIndex) {
        debug_assert!(
            a.index() < self.graph.num_vessels() && b.index() < self.graph.num_vessels(),
            "called `Mutator::swap_vessels_unchecked` with {a} or {b}, outside a graph of {} vessels and {} berths",
            self.graph.num_vessels(),
            self.graph.num_berths()
        );
        if a == b {
            return;
        }
        let (node_a, node_b) = (self.graph.vessel_node(a), self.graph.vessel_node(b));
        // SAFETY: both vessels are the graph's, the precondition of this function.
        let (berth_a, berth_b) = unsafe {
            (
                self.graph.vessel_berth_unchecked(a),
                self.graph.vessel_berth_unchecked(b),
            )
        };
        // SAFETY: both vessels are the graph's, the precondition of this function, so their nodes lie in its arena
        // (`ScheduleGraph`).
        let (prev_a, prev_b) = unsafe {
            (
                self.graph.prev_node_unchecked(node_a),
                self.graph.prev_node_unchecked(node_b),
            )
        };
        // SAFETY: the vessels' nodes are the graph's, as the vessels are, and so are the nodes read from the
        // graph's links (`ScheduleGraph`).
        let links =
            unsafe { LinkSnapshot::new_unchecked(&self.graph, [prev_a, prev_b, node_a, node_b]) };

        self.undo.push_swap_vessels(a, b);
        // SAFETY: both berths were read from the graph's column of berths, so they are its berths
        // (`ScheduleGraph`).
        unsafe { self.touch_both_unchecked(berth_a, berth_b) };
        // SAFETY: both vessels are the graph's, the precondition of this function.
        unsafe { self.graph.swap_vessels_unchecked(a, b) };
        // SAFETY: the nodes of `links` are the graph's, read from it before the swap, which keeps its counts.
        unsafe { links.commit_unchecked(&self.graph, &mut self.diff) };
        // SAFETY: both vessels are the graph's, the precondition of this function.
        unsafe {
            self.record_berth_change_unchecked(a, berth_a);
            self.record_berth_change_unchecked(b, berth_b);
        }
    }

    /// Moves `vessel` to directly after `anchor`, on its berth or another; nothing changes and nothing is recorded if
    /// `vessel` is `anchor` or already follows it.
    ///
    /// ```text
    /// before:  ... prev -> vessel -> next ...      ... anchor -> anchor_next ...
    /// after:   ... prev -> next ...                ... anchor -> vessel -> anchor_next ...
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if any of `vessel` and `anchor` is not one of the graph's.
    #[inline]
    #[track_caller]
    #[cfg(test)]
    pub fn relocate_after(&mut self, vessel: VesselIndex, anchor: VesselIndex) {
        self.check(Move::RelocateAfter, [vessel, anchor], []);
        // SAFETY: `vessel` and `anchor` are the graph's, as the check above found.
        unsafe { self.relocate_after_unchecked(vessel, anchor) }
    }

    /// The move [`relocate_after`](Self::relocate_after) makes, for ids the caller knows to be the graph's.
    ///
    /// # Safety
    ///
    /// `vessel` and `anchor` must be the graph's, the condition [`relocate_after`](Self::relocate_after) checks.
    #[inline]
    pub unsafe fn relocate_after_unchecked(&mut self, vessel: VesselIndex, anchor: VesselIndex) {
        debug_assert!(
            vessel.index() < self.graph.num_vessels() && anchor.index() < self.graph.num_vessels(),
            "called `Mutator::relocate_after_unchecked` with {vessel} or {anchor}, outside a graph of {} vessels and \
             {} berths",
            self.graph.num_vessels(),
            self.graph.num_berths()
        );
        let (vessel_node, anchor_node) = (
            self.graph.vessel_node(vessel),
            self.graph.vessel_node(anchor),
        );
        // SAFETY: `vessel` is the graph's, the precondition of this function, so its node lies in its arena
        // (`ScheduleGraph`).
        let vessel_prev = unsafe { self.graph.prev_node_unchecked(vessel_node) };
        if vessel_node == anchor_node || vessel_prev == anchor_node {
            return;
        }
        // SAFETY: both vessels are the graph's, the precondition of this function.
        let (from, to) = unsafe {
            (
                self.graph.vessel_berth_unchecked(vessel),
                self.graph.vessel_berth_unchecked(anchor),
            )
        };
        // SAFETY: the vessels' nodes are the graph's, as the vessels are, and so are the nodes read from the
        // graph's links (`ScheduleGraph`).
        let links = unsafe {
            LinkSnapshot::new_unchecked(&self.graph, [vessel_prev, vessel_node, anchor_node])
        };

        // SAFETY: both berths were read from the graph's column of berths, so they are its berths
        // (`ScheduleGraph`).
        unsafe { self.touch_both_unchecked(from, to) };
        // SAFETY: `vessel` is the graph's, the precondition of this function.
        unsafe { self.record_position_unchecked(vessel) };
        // SAFETY: both vessels are the graph's, the precondition of this function.
        unsafe { self.graph.relocate_after_unchecked(vessel, anchor) };
        // SAFETY: the nodes of `links` are the graph's, read from it before the move, which keeps its counts.
        unsafe { links.commit_unchecked(&self.graph, &mut self.diff) };
        // SAFETY: `vessel` is the graph's, the precondition of this function.
        unsafe { self.record_berth_change_unchecked(vessel, from) };
    }

    /// Moves `vessel` to the head of `berth`, its own or another, empty or not; nothing changes and nothing is
    /// recorded if `vessel` already heads `berth`.
    ///
    /// ```text
    /// before:  ... prev -> vessel -> next ...      head(berth) -> old_head ...
    /// after:   ... prev -> next ...                head(berth) -> vessel -> old_head ...
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `vessel` or `berth` is not one of the graph's.
    #[inline]
    #[track_caller]
    #[cfg(test)]
    pub fn relocate_to_head(&mut self, vessel: VesselIndex, berth: BerthIndex) {
        self.check(Move::RelocateToHead, [vessel], [berth]);
        // SAFETY: `vessel` and `berth` are the graph's, as the check above found.
        unsafe { self.relocate_to_head_unchecked(vessel, berth) }
    }

    /// The move [`relocate_to_head`](Self::relocate_to_head) makes, for ids the caller knows to be the graph's.
    ///
    /// # Safety
    ///
    /// `vessel` and `berth` must be the graph's, the condition [`relocate_to_head`](Self::relocate_to_head) checks.
    #[inline]
    pub unsafe fn relocate_to_head_unchecked(&mut self, vessel: VesselIndex, berth: BerthIndex) {
        debug_assert!(
            vessel.index() < self.graph.num_vessels() && berth.index() < self.graph.num_berths(),
            "called `Mutator::relocate_to_head_unchecked` with {vessel} or {berth}, outside a graph of {} vessels \
             and {} berths",
            self.graph.num_vessels(),
            self.graph.num_berths()
        );
        let (vessel_node, sentinel) = (
            self.graph.vessel_node(vessel),
            self.graph.sentinel_of(berth),
        );
        // SAFETY: `vessel` is the graph's, the precondition of this function, so its node lies in its arena
        // (`ScheduleGraph`).
        let vessel_prev = unsafe { self.graph.prev_node_unchecked(vessel_node) };
        if vessel_prev == sentinel {
            return;
        }
        // SAFETY: `vessel` is the graph's, the precondition of this function.
        let from = unsafe { self.graph.vessel_berth_unchecked(vessel) };
        // SAFETY: the vessel's node and the berth's sentinel are the graph's, as the vessel and the berth are, and
        // so is the node read from the graph's links (`ScheduleGraph`).
        let links = unsafe {
            LinkSnapshot::new_unchecked(&self.graph, [vessel_prev, vessel_node, sentinel])
        };

        // SAFETY: `from` was read from the graph's column of berths and `berth` is the graph's, the precondition of
        // this function (`ScheduleGraph`).
        unsafe { self.touch_both_unchecked(from, berth) };
        // SAFETY: `vessel` is the graph's, the precondition of this function.
        unsafe { self.record_position_unchecked(vessel) };
        // SAFETY: the vessel and the berth are the graph's, the precondition of this function.
        unsafe { self.graph.relocate_to_head_unchecked(vessel, berth) };
        // SAFETY: the nodes of `links` are the graph's, read from it before the move, which keeps its counts.
        unsafe { links.commit_unchecked(&self.graph, &mut self.diff) };
        // SAFETY: `vessel` is the graph's, the precondition of this function.
        unsafe { self.record_berth_change_unchecked(vessel, from) };
    }

    /// Marks `a` and `b` touched.
    ///
    /// # Safety
    ///
    /// `a` and `b` must be the graph's.
    #[inline(always)]
    unsafe fn touch_both_unchecked(&mut self, a: BerthIndex, b: BerthIndex) {
        debug_assert!(
            a.index() < self.graph.num_berths() && b.index() < self.graph.num_berths(),
            "called `Mutator::touch_both_unchecked` with {a} or {b}, outside a graph of {} berths",
            self.graph.num_berths()
        );
        // SAFETY: both berths are the graph's, the precondition of this function, and the touched berths count the
        // graph's berths by the invariant of `Mutator`.
        unsafe {
            self.touched.touch_unchecked(a);
            self.touched.touch_unchecked(b);
        }
    }

    /// Records in the undo log where `vessel` stands before it is moved: after the vessel before it, or at the
    /// head of its berth.
    ///
    /// # Safety
    ///
    /// `vessel` must be the graph's.
    #[inline(always)]
    unsafe fn record_position_unchecked(&mut self, vessel: VesselIndex) {
        debug_assert!(
            vessel.index() < self.graph.num_vessels(),
            "called `Mutator::record_position_unchecked` with {vessel}, outside a graph of {} vessels",
            self.graph.num_vessels()
        );
        // SAFETY: `vessel` is the graph's, the precondition of this function.
        let (predecessor, berth) = unsafe {
            (
                self.graph.vessel_predecessor_unchecked(vessel),
                self.graph.vessel_berth_unchecked(vessel),
            )
        };
        match predecessor {
            Some(predecessor) => self.undo.push_relocate_after_vessel(vessel, predecessor),
            None => self.undo.push_relocate_to_head(vessel, berth),
        }
    }

    /// Records in the diff `vessel` as moved from `from`, if a move has just taken it to another berth.
    ///
    /// # Safety
    ///
    /// `vessel` must be the graph's.
    #[inline(always)]
    unsafe fn record_berth_change_unchecked(&mut self, vessel: VesselIndex, from: BerthIndex) {
        debug_assert!(
            vessel.index() < self.graph.num_vessels(),
            "called `Mutator::record_berth_change_unchecked` with {vessel}, outside a graph of {} vessels",
            self.graph.num_vessels()
        );
        // SAFETY: `vessel` is the graph's, the precondition of this function.
        let to = unsafe { self.graph.vessel_berth_unchecked(vessel) };
        if from != to {
            self.diff.push_reallocation(vessel, from, to);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::diff::Link;
    use super::*;
    use proptest::prelude::*;
    use std::collections::{BTreeSet, HashMap};

    fn b(i: usize) -> BerthIndex {
        BerthIndex::new(i)
    }

    fn v(i: usize) -> VesselIndex {
        VesselIndex::new(i)
    }

    /// B0: V0 -> V1 -> V2
    /// B1: V3 -> V4
    /// B2: (empty)
    fn setup_graph() -> ScheduleGraph {
        let berths = [b(0), b(0), b(0), b(1), b(1)];
        let starts = [10, 20, 30, 10, 20];
        ScheduleGraph::from_slices(&berths, &starts, 3)
    }

    /// The mutator of the graph of `setup_graph`.
    fn setup() -> Mutator {
        Mutator::new(setup_graph())
    }

    /// The vessels of `berth`, from its head to its tail.
    fn vessels_of(mutator: &Mutator, berth: usize) -> Vec<VesselIndex> {
        mutator.graph().vessel_sequence_iter(b(berth)).collect()
    }

    fn extract_reallocations(
        diff: &ScheduleGraphDiff,
    ) -> Vec<(VesselIndex, BerthIndex, BerthIndex)> {
        diff.reallocations()
            .map(|moved| (moved.vessel, moved.from, moved.to))
            .collect()
    }

    #[test]
    fn test_new_counts_the_berths_of_the_graph_and_records_nothing() {
        let mutator = setup();
        assert_eq!(mutator.touched().num_berths(), 3);
        assert_eq!(mutator.touched().iter_touched_berths().count(), 0);
        assert!(mutator.undo.is_empty());
        assert_eq!(mutator.diff(), &ScheduleGraphDiff::new());
    }

    #[test]
    fn test_swap_vessels_same_berth() {
        let mut mutator = setup();
        mutator.swap_vessels(v(0), v(2));
        assert_eq!(vessels_of(&mutator, 0), [v(2), v(1), v(0)]);
        assert!(extract_reallocations(mutator.diff()).is_empty());
    }

    #[test]
    #[should_panic(
        expected = "called `Mutator::swap_vessels` with Vessel 5, outside a graph of 5 vessels"
    )]
    fn test_swap_vessels_refuses_a_vessel_outside_the_graph() {
        setup().swap_vessels(v(0), v(5));
    }

    #[test]
    fn test_rollback_restores_the_graph_and_unmarks_the_touched_berths() {
        let mut mutator = setup();
        mutator.relocate_after(v(0), v(4));
        mutator.relocate_after(v(3), v(2));
        mutator.swap_vessels(v(1), v(3));
        assert_ne!(mutator.graph(), &setup_graph());
        let diff = mutator.diff().clone();

        mutator.rollback();

        assert_eq!(mutator.graph(), &setup_graph());
        assert!(mutator.undo.is_empty());
        assert_eq!(mutator.touched().iter_touched_berths().count(), 0);
        assert_eq!(mutator.diff(), &diff);
    }

    #[test]
    fn test_accept_keeps_the_moves_and_forgets_how_to_undo_them() {
        let mut mutator = setup();
        mutator.swap_vessels(v(0), v(3));
        let moved = mutator.graph().clone();
        mutator.accept();
        assert!(mutator.undo.is_empty());
        assert_eq!(mutator.touched().iter_touched_berths().count(), 0);
        mutator.rollback();
        assert_eq!(mutator.graph(), &moved);
    }

    #[test]
    fn test_overwrite_empties_the_records_and_counts_the_berths_of_the_new_graph() {
        let mut mutator = setup();
        mutator.swap_vessels(v(0), v(3));
        mutator.overwrite_from_slices(&[b(0), b(1)], &[0, 0], 4);
        assert!(mutator.undo.is_empty());
        assert_eq!(mutator.diff(), &ScheduleGraphDiff::new());
        assert_eq!(mutator.touched().num_berths(), 4);
        assert_eq!(mutator.touched().iter_touched_berths().count(), 0);

        mutator.swap_vessels(v(0), v(1));
        mutator.overwrite_from_graph(&setup_graph());
        assert!(mutator.undo.is_empty());
        assert_eq!(mutator.touched().num_berths(), 3);
        assert_eq!(mutator.graph(), &setup_graph());
    }

    #[test]
    fn test_clear_diff_keeps_the_undo_log_and_the_touched_berths() {
        let mut mutator = setup();
        mutator.swap_vessels(v(0), v(3));
        mutator.clear_diff();
        assert_eq!(mutator.diff(), &ScheduleGraphDiff::new());
        assert_eq!(mutator.undo.len(), 1);
        assert_eq!(mutator.touched().iter_touched_berths().count(), 2);
    }

    /// A move of the reference test, its vessels and berths named by numbers that `Harness::apply` reads modulo
    /// their counts.
    #[derive(Clone, Copy, Debug)]
    enum Change {
        SwapVessels(usize, usize),
        After(usize, usize),
        ToHead(usize, usize),
    }

    fn change() -> impl Strategy<Value = Change> {
        let id = || 0_usize..16;
        prop_oneof![
            (id(), id()).prop_map(|(a, b)| Change::SwapVessels(a, b)),
            (id(), id()).prop_map(|(a, b)| Change::After(a, b)),
            (id(), id()).prop_map(|(a, berth)| Change::ToHead(a, berth)),
        ]
    }

    /// The berth and the place of `vessel` in the reference.
    fn locate(berths: &[Vec<usize>], vessel: usize) -> (usize, usize) {
        berths
            .iter()
            .enumerate()
            .find_map(|(berth, vessels)| {
                vessels
                    .iter()
                    .position(|&v| v == vessel)
                    .map(|place| (berth, place))
            })
            .expect("every vessel should stand on one berth of the reference")
    }

    /// The reference with `vessel` taken out.
    fn take(berths: &mut [Vec<usize>], vessel: usize) {
        let (berth, place) = locate(berths, vessel);
        berths[berth].remove(place);
    }

    /// A mutator moved through checked moves, a second through unchecked ones, and the reference: the vessels of
    /// each berth in order, as a plain list.
    struct Harness {
        checked: Mutator,
        unchecked: Mutator,
        reference: Vec<Vec<usize>>,
    }

    impl Harness {
        fn new(berths: &[usize], num_berths: usize) -> Self {
            let ids: Vec<_> = berths.iter().map(|&berth| b(berth)).collect();
            let starts: Vec<usize> = (0..berths.len()).collect();
            let graph = ScheduleGraph::from_slices(&ids, &starts, num_berths);
            let mut reference = vec![Vec::new(); num_berths];
            for (vessel, &berth) in berths.iter().enumerate() {
                reference[berth].push(vessel);
            }
            Self {
                checked: Mutator::new(graph.clone()),
                unchecked: Mutator::new(graph),
                reference,
            }
        }

        /// Applies `change` to the reference and to both graphs, and answers the berth the moved vessel leaves
        /// and the berth it reaches, for a swap the berths of the two vessels.
        fn apply(&mut self, change: Change) -> [usize; 2] {
            let n = self.checked.graph().num_vessels();
            let (checked, unchecked) = (&mut self.checked, &mut self.unchecked);
            let reference = &mut self.reference;
            match change {
                Change::SwapVessels(a, c) => {
                    let (a, c) = (a % n, c % n);
                    let ((berth_a, place_a), (berth_c, place_c)) =
                        (locate(reference, a), locate(reference, c));
                    reference[berth_a][place_a] = c;
                    reference[berth_c][place_c] = a;
                    checked.swap_vessels(v(a), v(c));
                    // SAFETY: both vessels are below the count.
                    unsafe { unchecked.swap_vessels_unchecked(v(a), v(c)) };
                    [berth_a, berth_c]
                }
                Change::After(a, c) => {
                    let (a, c) = (a % n, c % n);
                    let ends = [locate(reference, a).0, locate(reference, c).0];
                    if a != c {
                        take(reference, a);
                        let (berth, place) = locate(reference, c);
                        reference[berth].insert(place + 1, a);
                    }
                    checked.relocate_after(v(a), v(c));
                    // SAFETY: both vessels are below the count.
                    unsafe { unchecked.relocate_after_unchecked(v(a), v(c)) };
                    ends
                }
                Change::ToHead(a, berth) => {
                    let (a, berth) = (a % n, berth % reference.len());
                    let ends = [locate(reference, a).0, berth];
                    take(reference, a);
                    reference[berth].insert(0, a);
                    checked.relocate_to_head(v(a), b(berth));
                    // SAFETY: the vessel and the berth are below their counts.
                    unsafe { unchecked.relocate_to_head_unchecked(v(a), b(berth)) };
                    ends
                }
            }
        }

        /// Applies `change` after clearing the records, and checks the graphs against the reference: the order of
        /// every berth, the touched berths, the vessels moved between berths, the links the diff records against
        /// those that changed, and the unchecked records against the checked ones.
        ///
        /// A move touches exactly the berths it changed and the two it names, the one the vessel leaves and the
        /// one it reaches; a move that changes nothing, such as one onto the place a vessel holds, touches none.
        fn step(&mut self, change: Change) -> Result<(), TestCaseError> {
            let before = order_of(self.checked.graph());
            let links_before = links_of(self.checked.graph());
            for mutator in [&mut self.checked, &mut self.unchecked] {
                mutator.clear_diff();
                mutator.touched.reset();
            }
            let ends = self.apply(change);
            let after = order_of(self.checked.graph());
            prop_assert_eq!(&after, &self.reference, "after {:?}", change);
            prop_assert_eq!(
                &order_of(self.unchecked.graph()),
                &after,
                "unchecked, after {:?}",
                change
            );
            let changed: BTreeSet<usize> = (0..after.len())
                .filter(|&berth| before[berth] != after[berth])
                .collect();
            let mut expected = changed.clone();
            if !changed.is_empty() {
                expected.extend(ends);
            }
            let touched: BTreeSet<usize> = self
                .checked
                .touched()
                .iter_touched_berths()
                .map(BerthIndex::index)
                .collect();
            prop_assert_eq!(
                touched,
                expected,
                "{:?} changed berths {:?}",
                change,
                changed
            );
            let moved: Vec<_> = self.checked.diff().reallocations().collect();
            for vessel in 0..self.checked.graph().num_vessels() {
                let (from, to) = (locate(&before, vessel).0, locate(&after, vessel).0);
                let recorded: Vec<_> = moved
                    .iter()
                    .filter(|recorded| recorded.vessel == v(vessel))
                    .map(|moved| (moved.from, moved.to))
                    .collect();
                let expected: Vec<_> = if from == to {
                    vec![]
                } else {
                    vec![(b(from), b(to))]
                };
                prop_assert_eq!(recorded, expected, "{:?} and vessel {}", change, vessel);
            }
            prop_assert_eq!(
                links_after_diff(links_before, self.checked.diff()),
                links_of(self.checked.graph()),
                "the links before {:?}, less the broken and with the created, should be those after",
                change
            );
            prop_assert_eq!(
                self.unchecked.diff(),
                self.checked.diff(),
                "after {:?}",
                change
            );
            prop_assert_eq!(
                &self.unchecked.undo,
                &self.checked.undo,
                "after {:?}",
                change
            );
            prop_assert_eq!(
                self.unchecked.touched(),
                self.checked.touched(),
                "after {:?}",
                change
            );
            Ok(())
        }

        /// Rolls both mutators back and checks that each graph stands as `start`.
        fn rollback_to(&mut self, start: &ScheduleGraph) -> Result<(), TestCaseError> {
            self.checked.rollback();
            self.unchecked.rollback();
            prop_assert_eq!(self.checked.graph(), start);
            prop_assert_eq!(self.unchecked.graph(), start);
            prop_assert_eq!(&order_of(self.checked.graph()), &order_of(start));
            Ok(())
        }
    }

    /// The vessels of each berth of `graph` in order, and its berth and count records agree with them.
    fn order_of(graph: &ScheduleGraph) -> Vec<Vec<usize>> {
        graph
            .berths()
            .map(|berth| {
                let vessels: Vec<usize> = graph
                    .vessel_sequence_iter(berth)
                    .map(VesselIndex::index)
                    .collect();
                assert_eq!(
                    graph.vessel_count(berth),
                    vessels.len(),
                    "the count of {berth}"
                );
                for &vessel in &vessels {
                    assert_eq!(
                        graph.vessel_berth(v(vessel)),
                        berth,
                        "the berth of vessel {vessel}"
                    );
                }
                vessels
            })
            .collect()
    }

    /// The links of `graph`, each counted as often as it stands, the boundaries of empty berths included.
    fn links_of(graph: &ScheduleGraph) -> HashMap<Link, usize> {
        let mut links = HashMap::new();
        for berth in graph.berths() {
            let ends: Vec<_> = graph.berth_topology_iter(berth).collect();
            for pair in ends.windows(2) {
                *links
                    .entry(Link {
                        from: pair[0],
                        to: pair[1],
                    })
                    .or_default() += 1;
            }
        }
        links
    }

    /// `links` with every link `diff` records as broken taken out once and every created one added.
    fn links_after_diff(
        mut links: HashMap<Link, usize>,
        diff: &ScheduleGraphDiff,
    ) -> HashMap<Link, usize> {
        for link in diff.broken_links() {
            let count = links
                .get_mut(&link)
                .expect("a broken link should have stood before the move");
            *count -= 1;
            if *count == 0 {
                links.remove(&link);
            }
        }
        for link in diff.created_links() {
            *links.entry(link).or_default() += 1;
        }
        links
    }

    fn some_schedule() -> impl Strategy<Value = (Vec<usize>, usize)> {
        (1_usize..4).prop_flat_map(|num_berths| {
            (
                proptest::collection::vec(0..num_berths, 1..9),
                Just(num_berths),
            )
        })
    }

    proptest! {
        #[test]
        fn test_moves_match_the_model_of_berths_as_lists_and_undo_restores_the_start(
            (berths, num_berths) in some_schedule(),
            changes in proptest::collection::vec(change(), 0..16),
        ) {
            let mut harness = Harness::new(&berths, num_berths);
            let start = harness.checked.graph().clone();
            for &change in &changes {
                harness.step(change)?;
            }
            harness.rollback_to(&start)?;
        }
    }

    /// B0: V0 -> V1 -> V2, B1: V3, B2: (empty), B3: V4: a berth of several vessels, two of one and an empty one.
    const MIXED_BERTHS: [usize; 5] = [0, 0, 0, 1, 3];

    // Every move to a head, of every vessel to every berth: onto an empty berth, off a berth it leaves empty, to
    // the head of its own berth from any place, and onto the head it already holds, which must record nothing.
    #[test]
    fn test_every_move_to_a_head_matches_the_model_and_undoes_to_the_start() {
        for vessel in 0..MIXED_BERTHS.len() {
            for berth in 0..4 {
                let mut harness = Harness::new(&MIXED_BERTHS, 4);
                let start = harness.checked.graph().clone();
                let outcome = harness
                    .step(Change::ToHead(vessel, berth))
                    .and_then(|()| harness.rollback_to(&start));
                if let Err(failure) = outcome {
                    panic!("vessel {vessel} to the head of berth {berth}: {failure}");
                }
            }
        }
    }

    #[test]
    fn test_relocate_to_head_onto_an_empty_berth_leaves_the_vessels_berth_and_links_it_to_both_ends()
     {
        let mut mutator = setup();
        mutator.relocate_to_head(v(1), b(2));
        assert_eq!(vessels_of(&mutator, 0), [v(0), v(2)]);
        assert_eq!(vessels_of(&mutator, 2), [v(1)]);
        assert_eq!(extract_reallocations(mutator.diff()), [(v(1), b(0), b(2))]);
        let created: Vec<_> = mutator
            .diff()
            .created_links()
            .map(|link| link.to_string())
            .collect();
        assert_eq!(
            created,
            [
                "Vessel 0 -> Vessel 2",
                "Vessel 1 -> the boundary of Berth 2",
                "the boundary of Berth 2 -> Vessel 1"
            ]
        );
        let touched: Vec<_> = mutator.touched().iter_touched_berths().collect();
        assert_eq!(touched, [b(0), b(2)]);
    }

    #[test]
    fn test_relocate_to_head_of_the_head_it_holds_records_nothing() {
        let mut mutator = setup();
        mutator.relocate_to_head(v(3), b(1));
        assert_eq!(mutator.graph(), &setup_graph());
        assert!(mutator.undo.is_empty());
        assert_eq!(mutator.diff(), &ScheduleGraphDiff::new());
        assert_eq!(mutator.touched().iter_touched_berths().count(), 0);
    }

    /// A safe move of the mutator, as the refusal test calls it.
    #[derive(Clone, Copy, Debug)]
    enum Call {
        SwapVessels,
        RelocateAfter,
        RelocateToHead,
    }

    impl Call {
        const ALL: [Self; 3] = [Self::SwapVessels, Self::RelocateAfter, Self::RelocateToHead];

        /// The number of vessels and of berths the move takes.
        fn arity(self) -> (usize, usize) {
            match self {
                Self::SwapVessels | Self::RelocateAfter => (2, 0),
                Self::RelocateToHead => (1, 1),
            }
        }

        /// Makes the move with the first vessels of `vessels` and the berth of `berths`.
        fn make(self, mutator: &mut Mutator, vessels: [VesselIndex; 2], berths: [BerthIndex; 1]) {
            let [a, c] = vessels;
            let [berth] = berths;
            match self {
                Self::SwapVessels => mutator.swap_vessels(a, c),
                Self::RelocateAfter => mutator.relocate_after(a, c),
                Self::RelocateToHead => mutator.relocate_to_head(a, berth),
            }
        }
    }

    /// The message of the panic `payload` carries.
    fn message_of(payload: &(dyn std::any::Any + Send)) -> String {
        payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| {
                payload
                    .downcast_ref::<&str>()
                    .map(|text| (*text).to_owned())
            })
            .unwrap_or_default()
    }

    proptest! {
        // Every id but one is the graph's; the one outside stands at any place the move takes an id.
        #[test]
        fn test_safe_moves_refuse_every_id_outside_and_record_nothing(
            call in proptest::sample::select(Call::ALL.to_vec()),
            inside in proptest::array::uniform2(0_usize..5),
            berth_inside in 0_usize..3,
            place in 0_usize..2,
            beyond in 0_usize..32,
        ) {
            let (num_vessels, num_berths) = call.arity();
            let place = place % (num_vessels + num_berths);
            let mut vessels = inside.map(v);
            let mut berths = [b(berth_inside)];
            let outside = if place < num_vessels {
                vessels[place] = v(5 + beyond);
                vessels[place].to_string()
            } else {
                berths[0] = b(3 + beyond);
                berths[0].to_string()
            };
            let mut mutator = setup();
            let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                call.make(&mut mutator, vessels, berths);
            }));
            let message = message_of(refused.expect_err("the move should refuse an id outside").as_ref());
            prop_assert!(message.starts_with("called `Mutator::"), "{}", message);
            prop_assert!(message.contains(&format!("with {outside}, outside a graph")), "{}", message);
            prop_assert_eq!(mutator.graph(), &setup_graph());
            prop_assert!(mutator.undo.is_empty());
            prop_assert_eq!(mutator.diff(), &ScheduleGraphDiff::new());
            prop_assert_eq!(mutator.touched().iter_touched_berths().len(), 0);
        }
    }
}
