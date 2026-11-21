// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The moves of the local search on a schedule graph, each recorded as it is made.

use super::diff::ScheduleGraphDiff;
use super::graph::ScheduleGraph;
use super::touched_berths::TouchedBerths;
use super::undo_log::ScheduleGraphUndoLog;
use crate::ring_arena::Node;
use std::fmt;
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
                diff.push_link_broken(node, before);
                diff.push_link_created(node, after);
            }
        }
    }
}

// ----------------------------------------------------------------
// Move
// ----------------------------------------------------------------

/// A safe move of a [`Mutator`], as its refusal names it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Move {
    SwapVessels,
    SwapSegments,
    ReverseSegment,
    RelocateAfter,
    RelocateBefore,
    RelocateToHead,
    RelocateToTail,
    RelocateSegmentAfter,
    RelocateSegmentBefore,
    RelocateSegmentToHead,
    RelocateSegmentToTail,
}

impl fmt::Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::SwapVessels => "swap_vessels",
            Self::SwapSegments => "swap_segments",
            Self::ReverseSegment => "reverse_segment",
            Self::RelocateAfter => "relocate_after",
            Self::RelocateBefore => "relocate_before",
            Self::RelocateToHead => "relocate_to_head",
            Self::RelocateToTail => "relocate_to_tail",
            Self::RelocateSegmentAfter => "relocate_segment_after",
            Self::RelocateSegmentBefore => "relocate_segment_before",
            Self::RelocateSegmentToHead => "relocate_segment_to_head",
            Self::RelocateSegmentToTail => "relocate_segment_to_tail",
        };
        write!(f, "`Mutator::{name}`")
    }
}

// ----------------------------------------------------------------
// Mutator
// ----------------------------------------------------------------

/// The moves of the local search on a [`ScheduleGraph`], each recorded as it is made.
///
/// Every move is recorded three ways: its inverse in an undo log, so that a rejected candidate is undone; the
/// links it broke and made and the vessels it moved between berths in a diff, which the metaheuristics read; and
/// the berths whose order it changed among the touched berths, which the decoder decodes again. A move that
/// changes nothing records nothing where its doc says so.
///
/// Each move comes as a safe form, which checks in every build that its vessels and berths are the graph's, and an
/// `_unchecked` twin for the operator whose loop proves that they are.
// Invariant: `touched` counts as many berths as `graph`. Unchecked marks of a berth rest on it; `new` checks it,
// and neither count can change while the mutator holds both borrows.
#[derive(Debug)]
pub struct Mutator<'a> {
    /// The graph the moves change.
    graph: &'a mut ScheduleGraph,
    /// The inverse of every move made, the latest last.
    undo: &'a mut ScheduleGraphUndoLog,
    /// The links and berths the moves changed.
    diff: &'a mut ScheduleGraphDiff,
    /// The berths whose order the moves changed.
    touched: &'a mut TouchedBerths,
}

impl<'a> Mutator<'a> {
    /// The mutator that makes moves on `graph` and records them in `undo`, `diff` and `touched`.
    ///
    /// # Panics
    ///
    /// Panics if `touched` does not count as many berths as `graph`.
    #[inline]
    #[track_caller]
    pub fn new(
        graph: &'a mut ScheduleGraph,
        undo: &'a mut ScheduleGraphUndoLog,
        diff: &'a mut ScheduleGraphDiff,
        touched: &'a mut TouchedBerths,
    ) -> Self {
        assert!(
            touched.num_berths() == graph.num_berths(),
            "called `Mutator::new` with touched berths of {} berths for a graph of {}",
            touched.num_berths(),
            graph.num_berths()
        );
        Self {
            graph,
            undo,
            diff,
            touched,
        }
    }

    /// The graph the moves change.
    #[inline(always)]
    pub fn graph(&self) -> &ScheduleGraph {
        self.graph
    }

    /// Panics with the refusal of `step` if one of `vessels` or `berths` is not one of the graph's.
    #[inline]
    #[track_caller]
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
        // SAFETY: both vessels are the graph's, the precondition of this function, so their nodes and the nodes
        // read before them are the graph's, and so are their berths.
        let (links, berth_a, berth_b) = unsafe {
            let nodes = [
                self.graph.prev_node_unchecked(node_a),
                self.graph.prev_node_unchecked(node_b),
                node_a,
                node_b,
            ];
            let berths = (
                self.graph.vessel_berth_unchecked(a),
                self.graph.vessel_berth_unchecked(b),
            );
            (
                LinkSnapshot::new_unchecked(self.graph, nodes),
                berths.0,
                berths.1,
            )
        };
        self.undo.push_swap_vessels(a, b);
        // SAFETY: both vessels and both berths are the graph's, as above.
        unsafe {
            self.touch_unchecked(berth_a);
            self.touch_unchecked(berth_b);
            self.graph.swap_vessels_unchecked(a, b);
            links.commit_unchecked(self.graph, self.diff);
            self.record_berth_change_unchecked(a, a, berth_a);
            self.record_berth_change_unchecked(b, b, berth_b);
        }
    }

    /// Exchanges the places of the segments `a_first ..= a_last` and `b_first ..= b_last`, on one berth or across
    /// two; nothing changes and nothing is recorded if both start at one vessel.
    ///
    /// ```text
    /// before:  ... prev_a -> [a_first ... a_last] -> next_a ...   ... prev_b -> [b_first ... b_last] -> next_b ...
    /// after:   ... prev_a -> [b_first ... b_last] -> next_a ...   ... prev_b -> [a_first ... a_last] -> next_b ...
    /// ```
    ///
    /// # Preconditions
    ///
    /// `a_last` follows `a_first` on its berth, `b_last` follows `b_first`, and the segments share no vessel. Otherwise
    /// the order breaks, though no access leaves the graph; a debug build checks it.
    ///
    /// # Panics
    ///
    /// Panics if any of `a_first`, `a_last`, `b_first` and `b_last` is not one of the graph's.
    #[inline]
    #[track_caller]
    pub fn swap_segments(
        &mut self,
        a_first: VesselIndex,
        a_last: VesselIndex,
        b_first: VesselIndex,
        b_last: VesselIndex,
    ) {
        self.check(Move::SwapSegments, [a_first, a_last, b_first, b_last], []);
        // SAFETY: `a_first`, `a_last`, `b_first` and `b_last` are the graph's, as the check above found.
        unsafe { self.swap_segments_unchecked(a_first, a_last, b_first, b_last) }
    }

    /// The move [`swap_segments`](Self::swap_segments) makes, for ids the caller knows to be the graph's.
    ///
    /// # Preconditions
    ///
    /// Those of [`swap_segments`](Self::swap_segments).
    ///
    /// # Safety
    ///
    /// `a_first`, `a_last`, `b_first` and `b_last` must be the graph's, the condition [`swap_segments`](Self::swap_segments) checks.
    #[inline]
    pub unsafe fn swap_segments_unchecked(
        &mut self,
        a_first: VesselIndex,
        a_last: VesselIndex,
        b_first: VesselIndex,
        b_last: VesselIndex,
    ) {
        debug_assert!(
            a_first.index() < self.graph.num_vessels()
                && a_last.index() < self.graph.num_vessels()
                && b_first.index() < self.graph.num_vessels()
                && b_last.index() < self.graph.num_vessels(),
            "called `Mutator::swap_segments_unchecked` with {a_first} or {a_last} or {b_first} or {b_last}, outside a graph of {} vessels and {} berths",
            self.graph.num_vessels(),
            self.graph.num_berths()
        );
        if a_first == b_first {
            return;
        }
        let (a_first_node, b_first_node) = (
            self.graph.vessel_node(a_first),
            self.graph.vessel_node(b_first),
        );
        let (a_last_node, b_last_node) = (
            self.graph.vessel_node(a_last),
            self.graph.vessel_node(b_last),
        );
        // SAFETY: the four vessels are the graph's, the precondition of this function, so their nodes and the nodes
        // read before them are the graph's, and so are their berths.
        let (links, berth_a, berth_b) = unsafe {
            let nodes = [
                self.graph.prev_node_unchecked(a_first_node),
                a_last_node,
                self.graph.prev_node_unchecked(b_first_node),
                b_last_node,
            ];
            let berths = (
                self.graph.vessel_berth_unchecked(a_first),
                self.graph.vessel_berth_unchecked(b_first),
            );
            (
                LinkSnapshot::new_unchecked(self.graph, nodes),
                berths.0,
                berths.1,
            )
        };
        self.undo
            .push_swap_segments(a_first, a_last, b_first, b_last);
        // SAFETY: the four vessels and both berths are the graph's, as above.
        unsafe {
            self.touch_unchecked(berth_a);
            self.touch_unchecked(berth_b);
            self.graph
                .swap_segments_unchecked(a_first, a_last, b_first, b_last);
            links.commit_unchecked(self.graph, self.diff);
            self.record_berth_change_unchecked(a_first, a_last, berth_a);
            self.record_berth_change_unchecked(b_first, b_last, berth_b);
        }
    }

    /// Reverses the order of the segment `first ..= last` on its berth, at a cost of its length; nothing changes
    /// and nothing is recorded if it is one vessel.
    ///
    /// ```text
    /// before:  ... prev -> [a -> b -> c -> d] -> next ...
    /// after:   ... prev -> [d -> c -> b -> a] -> next ...
    /// ```
    ///
    /// # Preconditions
    ///
    /// `last` follows `first` on its berth. Otherwise the order breaks, though no access leaves the graph; a debug
    /// build checks it.
    ///
    /// # Panics
    ///
    /// Panics if any of `first` and `last` is not one of the graph's.
    #[inline]
    #[track_caller]
    pub fn reverse_segment(&mut self, first: VesselIndex, last: VesselIndex) {
        self.check(Move::ReverseSegment, [first, last], []);
        // SAFETY: `first` and `last` are the graph's, as the check above found.
        unsafe { self.reverse_segment_unchecked(first, last) }
    }

    /// The move [`reverse_segment`](Self::reverse_segment) makes, for ids the caller knows to be the graph's.
    ///
    /// # Preconditions
    ///
    /// Those of [`reverse_segment`](Self::reverse_segment).
    ///
    /// # Safety
    ///
    /// `first` and `last` must be the graph's, the condition [`reverse_segment`](Self::reverse_segment) checks.
    #[inline]
    pub unsafe fn reverse_segment_unchecked(&mut self, first: VesselIndex, last: VesselIndex) {
        debug_assert!(
            first.index() < self.graph.num_vessels() && last.index() < self.graph.num_vessels(),
            "called `Mutator::reverse_segment_unchecked` with {first} or {last}, outside a graph of {} vessels and {} berths",
            self.graph.num_vessels(),
            self.graph.num_berths()
        );
        if first == last {
            return;
        }
        let (first_node, last_node) = (self.graph.vessel_node(first), self.graph.vessel_node(last));
        // SAFETY: both vessels are the graph's, the precondition of this function, so their nodes and their berth
        // are the graph's.
        let (before_first, after_last, berth) = unsafe {
            (
                self.graph.prev_node_unchecked(first_node),
                self.graph.next_node_unchecked(last_node),
                self.graph.vessel_berth_unchecked(first),
            )
        };
        // SAFETY: `berth` is the graph's, as above.
        unsafe { self.touch_unchecked(berth) };
        self.diff.push_link_broken(before_first, first_node);
        self.diff.push_link_broken(last_node, after_last);
        // SAFETY: both vessels are the graph's, the precondition of this function.
        let segment = unsafe { self.graph.segment_unchecked(first, last) };
        let mut previous = first_node;
        for vessel in segment.skip(1) {
            let node = self.graph.vessel_node(vessel);
            self.diff.push_link_broken(previous, node);
            self.diff.push_link_created(node, previous);
            previous = node;
        }
        self.diff.push_link_created(before_first, last_node);
        self.diff.push_link_created(first_node, after_last);
        self.undo.push_reverse_segment(first, last);
        // SAFETY: both vessels are the graph's, the precondition of this function.
        unsafe { self.graph.reverse_segment_unchecked(first, last) };
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
            "called `Mutator::relocate_after_unchecked` with {vessel} or {anchor}, outside a graph of {} vessels and {} berths",
            self.graph.num_vessels(),
            self.graph.num_berths()
        );
        let vessel_node = self.graph.vessel_node(vessel);
        let anchor_node = self.graph.vessel_node(anchor);
        // SAFETY: `vessel` is the graph's, the precondition of this function, so its node is.
        if vessel_node == anchor_node
            || unsafe { self.graph.prev_node_unchecked(vessel_node) } == anchor_node
        {
            return;
        }
        // SAFETY: the vessels are the graph's, the precondition of this function, so the nodes named here and
        // the nodes read from the graph are the graph's, and so are the berths.
        let (links, from, to) = unsafe {
            let nodes = [
                self.graph.prev_node_unchecked(vessel_node),
                vessel_node,
                anchor_node,
            ];
            (
                LinkSnapshot::new_unchecked(self.graph, nodes),
                self.graph.vessel_berth_unchecked(vessel),
                self.graph.vessel_berth_unchecked(anchor),
            )
        };
        // SAFETY: the vessel and both berths are the graph's, as above.
        unsafe {
            self.touch_unchecked(from);
            self.touch_unchecked(to);
            self.record_position_unchecked(vessel, vessel);
            self.graph.relocate_after_unchecked(vessel, anchor);
            links.commit_unchecked(self.graph, self.diff);
            self.record_berth_change_unchecked(vessel, vessel, from);
        }
    }

    /// Moves `vessel` to directly before `reference`, on its berth or another; nothing changes and nothing is recorded
    /// if `vessel` is `reference` or already precedes it.
    ///
    /// ```text
    /// before:  ... prev -> vessel -> next ...      ... reference_prev -> reference ...
    /// after:   ... prev -> next ...                ... reference_prev -> vessel -> reference ...
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if any of `vessel` and `reference` is not one of the graph's.
    #[inline]
    #[track_caller]
    pub fn relocate_before(&mut self, vessel: VesselIndex, reference: VesselIndex) {
        self.check(Move::RelocateBefore, [vessel, reference], []);
        // SAFETY: `vessel` and `reference` are the graph's, as the check above found.
        unsafe { self.relocate_before_unchecked(vessel, reference) }
    }

    /// The move [`relocate_before`](Self::relocate_before) makes, for ids the caller knows to be the graph's.
    ///
    /// # Safety
    ///
    /// `vessel` and `reference` must be the graph's, the condition [`relocate_before`](Self::relocate_before) checks.
    #[inline]
    pub unsafe fn relocate_before_unchecked(
        &mut self,
        vessel: VesselIndex,
        reference: VesselIndex,
    ) {
        debug_assert!(
            vessel.index() < self.graph.num_vessels()
                && reference.index() < self.graph.num_vessels(),
            "called `Mutator::relocate_before_unchecked` with {vessel} or {reference}, outside a graph of {} vessels and {} berths",
            self.graph.num_vessels(),
            self.graph.num_berths()
        );
        let vessel_node = self.graph.vessel_node(vessel);
        let reference_node = self.graph.vessel_node(reference);
        // SAFETY: `reference` is the graph's, the precondition of this function, so its node is.
        let reference_prev = unsafe { self.graph.prev_node_unchecked(reference_node) };
        if vessel_node == reference_node || vessel_node == reference_prev {
            return;
        }
        // SAFETY: the vessels are the graph's, the precondition of this function, so the nodes named here and
        // the nodes read from the graph are the graph's, and so are the berths.
        let (links, from, to) = unsafe {
            let nodes = [
                self.graph.prev_node_unchecked(vessel_node),
                vessel_node,
                reference_prev,
            ];
            (
                LinkSnapshot::new_unchecked(self.graph, nodes),
                self.graph.vessel_berth_unchecked(vessel),
                self.graph.vessel_berth_unchecked(reference),
            )
        };
        // SAFETY: the vessel and both berths are the graph's, as above.
        unsafe {
            self.touch_unchecked(from);
            self.touch_unchecked(to);
            self.record_position_unchecked(vessel, vessel);
            self.graph.relocate_before_unchecked(vessel, reference);
            links.commit_unchecked(self.graph, self.diff);
            self.record_berth_change_unchecked(vessel, vessel, from);
        }
    }

    /// Moves `vessel` to the head of `berth`.
    ///
    /// ```text
    /// before:  ... prev -> vessel -> next ...      sentinel -> old_head ...
    /// after:   ... prev -> next ...                sentinel -> vessel -> old_head ...
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if any of `vessel` and `berth` is not one of the graph's.
    #[inline]
    #[track_caller]
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
            "called `Mutator::relocate_to_head_unchecked` with {vessel} or {berth}, outside a graph of {} vessels and {} berths",
            self.graph.num_vessels(),
            self.graph.num_berths()
        );
        let vessel_node = self.graph.vessel_node(vessel);
        let sentinel = self.graph.sentinel_of(berth);
        // SAFETY: the vessel and the berth are the graph's, the precondition of this function, so the nodes named here and
        // the nodes read from the graph are the graph's, and so are the berths.
        let (links, from, to) = unsafe {
            let nodes = [
                self.graph.prev_node_unchecked(vessel_node),
                vessel_node,
                sentinel,
            ];
            (
                LinkSnapshot::new_unchecked(self.graph, nodes),
                self.graph.vessel_berth_unchecked(vessel),
                berth,
            )
        };
        // SAFETY: the vessel and both berths are the graph's, as above.
        unsafe {
            self.touch_unchecked(from);
            self.touch_unchecked(to);
            self.record_position_unchecked(vessel, vessel);
            self.graph.relocate_to_head_unchecked(vessel, berth);
            links.commit_unchecked(self.graph, self.diff);
            self.record_berth_change_unchecked(vessel, vessel, from);
        }
    }

    /// Moves `vessel` to the tail of `berth`.
    ///
    /// ```text
    /// before:  ... prev -> vessel -> next ...      ... old_tail -> sentinel
    /// after:   ... prev -> next ...                ... old_tail -> vessel -> sentinel
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if any of `vessel` and `berth` is not one of the graph's.
    #[inline]
    #[track_caller]
    pub fn relocate_to_tail(&mut self, vessel: VesselIndex, berth: BerthIndex) {
        self.check(Move::RelocateToTail, [vessel], [berth]);
        // SAFETY: `vessel` and `berth` are the graph's, as the check above found.
        unsafe { self.relocate_to_tail_unchecked(vessel, berth) }
    }

    /// The move [`relocate_to_tail`](Self::relocate_to_tail) makes, for ids the caller knows to be the graph's.
    ///
    /// # Safety
    ///
    /// `vessel` and `berth` must be the graph's, the condition [`relocate_to_tail`](Self::relocate_to_tail) checks.
    #[inline]
    pub unsafe fn relocate_to_tail_unchecked(&mut self, vessel: VesselIndex, berth: BerthIndex) {
        debug_assert!(
            vessel.index() < self.graph.num_vessels() && berth.index() < self.graph.num_berths(),
            "called `Mutator::relocate_to_tail_unchecked` with {vessel} or {berth}, outside a graph of {} vessels and {} berths",
            self.graph.num_vessels(),
            self.graph.num_berths()
        );
        let vessel_node = self.graph.vessel_node(vessel);
        let sentinel = self.graph.sentinel_of(berth);
        // SAFETY: the vessel and the berth are the graph's, the precondition of this function, so the nodes named here and
        // the nodes read from the graph are the graph's, and so are the berths.
        let (links, from, to) = unsafe {
            let nodes = [
                self.graph.prev_node_unchecked(vessel_node),
                vessel_node,
                self.graph.prev_node_unchecked(sentinel),
            ];
            (
                LinkSnapshot::new_unchecked(self.graph, nodes),
                self.graph.vessel_berth_unchecked(vessel),
                berth,
            )
        };
        // SAFETY: the vessel and both berths are the graph's, as above.
        unsafe {
            self.touch_unchecked(from);
            self.touch_unchecked(to);
            self.record_position_unchecked(vessel, vessel);
            self.graph.relocate_to_tail_unchecked(vessel, berth);
            links.commit_unchecked(self.graph, self.diff);
            self.record_berth_change_unchecked(vessel, vessel, from);
        }
    }

    /// Moves the segment `first ..= last` to directly after `anchor`, on its berth or another; nothing changes and
    /// nothing is recorded if it already follows `anchor`.
    ///
    /// ```text
    /// before:  ... prev -> [first ... last] -> next ...      ... anchor -> anchor_next ...
    /// after:   ... prev -> next ...                          ... anchor -> [first ... last] -> anchor_next ...
    /// ```
    ///
    /// # Preconditions
    ///
    /// `last` follows `first` on its berth, and `anchor` is not a vessel of the segment. Otherwise the order breaks, though no access leaves the
    /// graph; a debug build checks it.
    ///
    /// # Panics
    ///
    /// Panics if any of `first`, `last` and `anchor` is not one of the graph's.
    #[inline]
    #[track_caller]
    pub fn relocate_segment_after(
        &mut self,
        first: VesselIndex,
        last: VesselIndex,
        anchor: VesselIndex,
    ) {
        self.check(Move::RelocateSegmentAfter, [first, last, anchor], []);
        // SAFETY: `first`, `last` and `anchor` are the graph's, as the check above found.
        unsafe { self.relocate_segment_after_unchecked(first, last, anchor) }
    }

    /// The move [`relocate_segment_after`](Self::relocate_segment_after) makes, for ids the caller knows to be the graph's.
    ///
    /// # Preconditions
    ///
    /// Those of [`relocate_segment_after`](Self::relocate_segment_after).
    ///
    /// # Safety
    ///
    /// `first`, `last` and `anchor` must be the graph's, the condition [`relocate_segment_after`](Self::relocate_segment_after) checks.
    #[inline]
    pub unsafe fn relocate_segment_after_unchecked(
        &mut self,
        first: VesselIndex,
        last: VesselIndex,
        anchor: VesselIndex,
    ) {
        debug_assert!(
            first.index() < self.graph.num_vessels()
                && last.index() < self.graph.num_vessels()
                && anchor.index() < self.graph.num_vessels(),
            "called `Mutator::relocate_segment_after_unchecked` with {first} or {last} or {anchor}, outside a graph of {} vessels and {} berths",
            self.graph.num_vessels(),
            self.graph.num_berths()
        );
        let (first_node, last_node) = (self.graph.vessel_node(first), self.graph.vessel_node(last));
        let anchor_node = self.graph.vessel_node(anchor);
        // SAFETY: `first` is the graph's, the precondition of this function, so its node is.
        if unsafe { self.graph.prev_node_unchecked(first_node) } == anchor_node {
            return;
        }
        // SAFETY: the vessels are the graph's, the precondition of this function, so the nodes named here and the
        // nodes read from the graph are the graph's, and so are the berths.
        let (links, from, to) = unsafe {
            let nodes = [
                self.graph.prev_node_unchecked(first_node),
                last_node,
                anchor_node,
            ];
            (
                LinkSnapshot::new_unchecked(self.graph, nodes),
                self.graph.vessel_berth_unchecked(first),
                self.graph.vessel_berth_unchecked(anchor),
            )
        };
        // SAFETY: the vessels and both berths are the graph's, as above.
        unsafe {
            self.touch_unchecked(from);
            self.touch_unchecked(to);
            self.record_position_unchecked(first, last);
            self.graph
                .relocate_segment_after_unchecked(first, last, anchor);
            links.commit_unchecked(self.graph, self.diff);
            self.record_berth_change_unchecked(first, last, from);
        }
    }

    /// Moves the segment `first ..= last` to directly before `reference`, on its berth or another; nothing changes and
    /// nothing is recorded if it already precedes `reference`.
    ///
    /// ```text
    /// before:  ... prev -> [first ... last] -> next ...      ... reference_prev -> reference ...
    /// after:   ... prev -> next ...                          ... reference_prev -> [first ... last] -> reference ...
    /// ```
    ///
    /// # Preconditions
    ///
    /// `last` follows `first` on its berth, and `reference` is not a vessel of the segment. Otherwise the order breaks, though no access leaves the
    /// graph; a debug build checks it.
    ///
    /// # Panics
    ///
    /// Panics if any of `first`, `last` and `reference` is not one of the graph's.
    #[inline]
    #[track_caller]
    pub fn relocate_segment_before(
        &mut self,
        first: VesselIndex,
        last: VesselIndex,
        reference: VesselIndex,
    ) {
        self.check(Move::RelocateSegmentBefore, [first, last, reference], []);
        // SAFETY: `first`, `last` and `reference` are the graph's, as the check above found.
        unsafe { self.relocate_segment_before_unchecked(first, last, reference) }
    }

    /// The move [`relocate_segment_before`](Self::relocate_segment_before) makes, for ids the caller knows to be the graph's.
    ///
    /// # Preconditions
    ///
    /// Those of [`relocate_segment_before`](Self::relocate_segment_before).
    ///
    /// # Safety
    ///
    /// `first`, `last` and `reference` must be the graph's, the condition [`relocate_segment_before`](Self::relocate_segment_before) checks.
    #[inline]
    pub unsafe fn relocate_segment_before_unchecked(
        &mut self,
        first: VesselIndex,
        last: VesselIndex,
        reference: VesselIndex,
    ) {
        debug_assert!(
            first.index() < self.graph.num_vessels()
                && last.index() < self.graph.num_vessels()
                && reference.index() < self.graph.num_vessels(),
            "called `Mutator::relocate_segment_before_unchecked` with {first} or {last} or {reference}, outside a graph of {} vessels and {} berths",
            self.graph.num_vessels(),
            self.graph.num_berths()
        );
        let (first_node, last_node) = (self.graph.vessel_node(first), self.graph.vessel_node(last));
        let reference_node = self.graph.vessel_node(reference);
        // SAFETY: `reference` is the graph's, the precondition of this function, so its node is.
        let reference_prev = unsafe { self.graph.prev_node_unchecked(reference_node) };
        if last_node == reference_prev {
            return;
        }
        // SAFETY: the vessels are the graph's, the precondition of this function, so the nodes named here and the
        // nodes read from the graph are the graph's, and so are the berths.
        let (links, from, to) = unsafe {
            let nodes = [
                self.graph.prev_node_unchecked(first_node),
                last_node,
                reference_prev,
            ];
            (
                LinkSnapshot::new_unchecked(self.graph, nodes),
                self.graph.vessel_berth_unchecked(first),
                self.graph.vessel_berth_unchecked(reference),
            )
        };
        // SAFETY: the vessels and both berths are the graph's, as above.
        unsafe {
            self.touch_unchecked(from);
            self.touch_unchecked(to);
            self.record_position_unchecked(first, last);
            self.graph
                .relocate_segment_before_unchecked(first, last, reference);
            links.commit_unchecked(self.graph, self.diff);
            self.record_berth_change_unchecked(first, last, from);
        }
    }

    /// Moves the segment `first ..= last` to the head of `berth`.
    ///
    /// ```text
    /// before:  ... prev -> [first ... last] -> next ...      sentinel -> old_head ...
    /// after:   ... prev -> next ...                          sentinel -> [first ... last] -> old_head ...
    /// ```
    ///
    /// # Preconditions
    ///
    /// `last` follows `first` on its berth, and `berth` is not the segment's own berth with the segment at its head. Otherwise the order breaks, though no access leaves the
    /// graph; a debug build checks it.
    ///
    /// # Panics
    ///
    /// Panics if any of `first`, `last` and `berth` is not one of the graph's.
    #[inline]
    #[track_caller]
    pub fn relocate_segment_to_head(
        &mut self,
        first: VesselIndex,
        last: VesselIndex,
        berth: BerthIndex,
    ) {
        self.check(Move::RelocateSegmentToHead, [first, last], [berth]);
        // SAFETY: `first`, `last` and `berth` are the graph's, as the check above found.
        unsafe { self.relocate_segment_to_head_unchecked(first, last, berth) }
    }

    /// The move [`relocate_segment_to_head`](Self::relocate_segment_to_head) makes, for ids the caller knows to be the graph's.
    ///
    /// # Preconditions
    ///
    /// Those of [`relocate_segment_to_head`](Self::relocate_segment_to_head).
    ///
    /// # Safety
    ///
    /// `first`, `last` and `berth` must be the graph's, the condition [`relocate_segment_to_head`](Self::relocate_segment_to_head) checks.
    #[inline]
    pub unsafe fn relocate_segment_to_head_unchecked(
        &mut self,
        first: VesselIndex,
        last: VesselIndex,
        berth: BerthIndex,
    ) {
        debug_assert!(
            first.index() < self.graph.num_vessels()
                && last.index() < self.graph.num_vessels()
                && berth.index() < self.graph.num_berths(),
            "called `Mutator::relocate_segment_to_head_unchecked` with {first} or {last} or {berth}, outside a graph of {} vessels and {} berths",
            self.graph.num_vessels(),
            self.graph.num_berths()
        );
        let (first_node, last_node) = (self.graph.vessel_node(first), self.graph.vessel_node(last));
        let sentinel = self.graph.sentinel_of(berth);
        // SAFETY: the vessels and the berth are the graph's, the precondition of this function, so the nodes named here and the
        // nodes read from the graph are the graph's, and so are the berths.
        let (links, from, to) = unsafe {
            let nodes = [
                self.graph.prev_node_unchecked(first_node),
                last_node,
                sentinel,
            ];
            (
                LinkSnapshot::new_unchecked(self.graph, nodes),
                self.graph.vessel_berth_unchecked(first),
                berth,
            )
        };
        // SAFETY: the vessels and both berths are the graph's, as above.
        unsafe {
            self.touch_unchecked(from);
            self.touch_unchecked(to);
            self.record_position_unchecked(first, last);
            self.graph
                .relocate_segment_to_head_unchecked(first, last, berth);
            links.commit_unchecked(self.graph, self.diff);
            self.record_berth_change_unchecked(first, last, from);
        }
    }

    /// Moves the segment `first ..= last` to the tail of `berth`.
    ///
    /// ```text
    /// before:  ... prev -> [first ... last] -> next ...      ... old_tail -> sentinel
    /// after:   ... prev -> next ...                          ... old_tail -> [first ... last] -> sentinel
    /// ```
    ///
    /// # Preconditions
    ///
    /// `last` follows `first` on its berth. Otherwise the order breaks, though no access leaves the
    /// graph; a debug build checks it.
    ///
    /// # Panics
    ///
    /// Panics if any of `first`, `last` and `berth` is not one of the graph's.
    #[inline]
    #[track_caller]
    pub fn relocate_segment_to_tail(
        &mut self,
        first: VesselIndex,
        last: VesselIndex,
        berth: BerthIndex,
    ) {
        self.check(Move::RelocateSegmentToTail, [first, last], [berth]);
        // SAFETY: `first`, `last` and `berth` are the graph's, as the check above found.
        unsafe { self.relocate_segment_to_tail_unchecked(first, last, berth) }
    }

    /// The move [`relocate_segment_to_tail`](Self::relocate_segment_to_tail) makes, for ids the caller knows to be the graph's.
    ///
    /// # Preconditions
    ///
    /// Those of [`relocate_segment_to_tail`](Self::relocate_segment_to_tail).
    ///
    /// # Safety
    ///
    /// `first`, `last` and `berth` must be the graph's, the condition [`relocate_segment_to_tail`](Self::relocate_segment_to_tail) checks.
    #[inline]
    pub unsafe fn relocate_segment_to_tail_unchecked(
        &mut self,
        first: VesselIndex,
        last: VesselIndex,
        berth: BerthIndex,
    ) {
        debug_assert!(
            first.index() < self.graph.num_vessels()
                && last.index() < self.graph.num_vessels()
                && berth.index() < self.graph.num_berths(),
            "called `Mutator::relocate_segment_to_tail_unchecked` with {first} or {last} or {berth}, outside a graph of {} vessels and {} berths",
            self.graph.num_vessels(),
            self.graph.num_berths()
        );
        let (first_node, last_node) = (self.graph.vessel_node(first), self.graph.vessel_node(last));
        let sentinel = self.graph.sentinel_of(berth);
        // SAFETY: the vessels and the berth are the graph's, the precondition of this function, so the nodes named here and the
        // nodes read from the graph are the graph's, and so are the berths.
        let (links, from, to) = unsafe {
            let nodes = [
                self.graph.prev_node_unchecked(first_node),
                last_node,
                self.graph.prev_node_unchecked(sentinel),
            ];
            (
                LinkSnapshot::new_unchecked(self.graph, nodes),
                self.graph.vessel_berth_unchecked(first),
                berth,
            )
        };
        // SAFETY: the vessels and both berths are the graph's, as above.
        unsafe {
            self.touch_unchecked(from);
            self.touch_unchecked(to);
            self.record_position_unchecked(first, last);
            self.graph
                .relocate_segment_to_tail_unchecked(first, last, berth);
            links.commit_unchecked(self.graph, self.diff);
            self.record_berth_change_unchecked(first, last, from);
        }
    }

    /// Marks `berth` touched.
    ///
    /// # Safety
    ///
    /// `berth` must be one of the graph's.
    #[inline(always)]
    unsafe fn touch_unchecked(&mut self, berth: BerthIndex) {
        debug_assert!(
            berth.index() < self.graph.num_berths(),
            "called `Mutator::touch_unchecked` with {berth}, outside a graph of {} berths",
            self.graph.num_berths()
        );
        // SAFETY: `berth` is the graph's, the precondition of this function, and the touched berths count as many
        // as the graph by the invariant of `Mutator`.
        unsafe { self.touched.touch_unchecked(berth) };
    }

    /// Records in the undo log where the segment `first ..= last` stands before it is moved: after the vessel
    /// before it, or at the head of its berth.
    ///
    /// # Safety
    ///
    /// `first` and `last` must be the graph's.
    #[inline(always)]
    unsafe fn record_position_unchecked(&mut self, first: VesselIndex, last: VesselIndex) {
        debug_assert!(
            first.index() < self.graph.num_vessels() && last.index() < self.graph.num_vessels(),
            "called `Mutator::record_position_unchecked` with {first} or {last}, outside a graph of {} vessels",
            self.graph.num_vessels()
        );
        // SAFETY: `first` is the graph's, the precondition of this function.
        let (predecessor, berth) = unsafe {
            (
                self.graph.vessel_predecessor_unchecked(first),
                self.graph.vessel_berth_unchecked(first),
            )
        };
        match (predecessor, first == last) {
            (Some(predecessor), true) => self.undo.push_relocate_after_vessel(first, predecessor),
            (None, true) => self.undo.push_relocate_to_head(first, berth),
            (Some(predecessor), false) => {
                self.undo
                    .push_relocate_segment_after_vessel(first, last, predecessor);
            }
            (None, false) => self.undo.push_relocate_segment_to_head(first, last, berth),
        }
    }

    /// Records in the diff every vessel of the segment `first ..= last` as moved from `from`, if a move has just
    /// taken the segment to another berth.
    ///
    /// # Preconditions
    ///
    /// `last` follows `first` on its berth. Otherwise the wrong vessels are recorded, and a debug build panics; the
    /// walk ends after at most the graph's count of nodes.
    ///
    /// # Safety
    ///
    /// `first` and `last` must be the graph's.
    #[inline(always)]
    unsafe fn record_berth_change_unchecked(
        &mut self,
        first: VesselIndex,
        last: VesselIndex,
        from: BerthIndex,
    ) {
        debug_assert!(
            first.index() < self.graph.num_vessels() && last.index() < self.graph.num_vessels(),
            "called `Mutator::record_berth_change_unchecked` with {first} or {last}, outside a graph of {} vessels",
            self.graph.num_vessels()
        );
        // SAFETY: `first` is the graph's, the precondition of this function.
        let to = unsafe { self.graph.vessel_berth_unchecked(first) };
        if from == to {
            return;
        }
        // SAFETY: `first` and `last` are the graph's, the precondition of this function.
        let segment = unsafe { self.graph.segment_unchecked(first, last) };
        for vessel in segment {
            self.diff.push_reallocation(vessel, from, to);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

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

    fn extract_reallocations(
        diff: &ScheduleGraphDiff,
    ) -> Vec<(VesselIndex, BerthIndex, BerthIndex)> {
        diff.reallocations().collect()
    }

    #[test]
    fn test_swap_vessels_same_berth() {
        let mut graph = setup_graph();
        let mut undo = ScheduleGraphUndoLog::new(10);
        let mut diff = ScheduleGraphDiff::new();

        {
            let mut touched = TouchedBerths::new(graph.num_berths());
            let mut m = Mutator::new(&mut graph, &mut undo, &mut diff, &mut touched);
            m.swap_vessels(v(0), v(2));
        }

        let seq: Vec<_> = graph.vessel_sequence_iter(b(0)).collect();
        assert_eq!(seq, vec![v(2), v(1), v(0)]);
        assert!(extract_reallocations(&diff).is_empty());
    }

    #[test]
    fn test_relocate_to_head_cross_berth() {
        let mut graph = setup_graph();
        let mut undo = ScheduleGraphUndoLog::new(10);
        let mut diff = ScheduleGraphDiff::new();

        {
            let mut touched = TouchedBerths::new(graph.num_berths());
            let mut m = Mutator::new(&mut graph, &mut undo, &mut diff, &mut touched);
            m.relocate_to_head(v(0), b(1));
        }

        assert_eq!(
            graph.vessel_sequence_iter(b(0)).collect::<Vec<_>>(),
            vec![v(1), v(2)]
        );
        assert_eq!(
            graph.vessel_sequence_iter(b(1)).collect::<Vec<_>>(),
            vec![v(0), v(3), v(4)]
        );

        let reallocs = extract_reallocations(&diff);
        assert_eq!(reallocs, vec![(v(0), b(0), b(1))]);
    }

    #[test]
    fn test_relocate_segment_after_cross_berth() {
        let mut graph = setup_graph();
        let mut undo = ScheduleGraphUndoLog::new(10);
        let mut diff = ScheduleGraphDiff::new();

        {
            let mut touched = TouchedBerths::new(graph.num_berths());
            let mut m = Mutator::new(&mut graph, &mut undo, &mut diff, &mut touched);
            m.relocate_segment_after(v(0), v(1), v(3));
        }

        assert_eq!(
            graph.vessel_sequence_iter(b(0)).collect::<Vec<_>>(),
            vec![v(2)]
        );
        assert_eq!(
            graph.vessel_sequence_iter(b(1)).collect::<Vec<_>>(),
            vec![v(3), v(0), v(1), v(4)]
        );

        let reallocs = extract_reallocations(&diff);
        assert_eq!(reallocs, vec![(v(0), b(0), b(1)), (v(1), b(0), b(1))]);
    }

    #[test]
    fn test_reverse_segment() {
        let mut graph = setup_graph();
        let mut undo = ScheduleGraphUndoLog::new(10);
        let mut diff = ScheduleGraphDiff::new();

        {
            let mut touched = TouchedBerths::new(graph.num_berths());
            let mut m = Mutator::new(&mut graph, &mut undo, &mut diff, &mut touched);
            m.reverse_segment(v(0), v(2));
        }

        assert_eq!(
            graph.vessel_sequence_iter(b(0)).collect::<Vec<_>>(),
            vec![v(2), v(1), v(0)]
        );
        assert!(extract_reallocations(&diff).is_empty());
    }

    #[test]
    fn test_swap_segments_cross_berth() {
        let mut graph = setup_graph();
        let mut undo = ScheduleGraphUndoLog::new(10);
        let mut diff = ScheduleGraphDiff::new();

        {
            let mut touched = TouchedBerths::new(graph.num_berths());
            let mut m = Mutator::new(&mut graph, &mut undo, &mut diff, &mut touched);
            m.swap_segments(v(0), v(1), v(3), v(3));
        }

        assert_eq!(
            graph.vessel_sequence_iter(b(0)).collect::<Vec<_>>(),
            vec![v(3), v(2)]
        );
        assert_eq!(
            graph.vessel_sequence_iter(b(1)).collect::<Vec<_>>(),
            vec![v(0), v(1), v(4)]
        );

        let reallocs = extract_reallocations(&diff);
        assert!(reallocs.contains(&(v(0), b(0), b(1))));
        assert!(reallocs.contains(&(v(1), b(0), b(1))));
        assert!(reallocs.contains(&(v(3), b(1), b(0))));
    }

    #[test]
    fn test_relocate_segment_to_tail_empty_berth() {
        let mut graph = setup_graph();
        let mut undo = ScheduleGraphUndoLog::new(10);
        let mut diff = ScheduleGraphDiff::new();

        {
            let mut touched = TouchedBerths::new(graph.num_berths());
            let mut m = Mutator::new(&mut graph, &mut undo, &mut diff, &mut touched);
            m.relocate_segment_to_tail(v(3), v(4), b(2));
        }

        assert!(graph.vessel_sequence_iter(b(1)).next().is_none());
        assert_eq!(
            graph.vessel_sequence_iter(b(2)).collect::<Vec<_>>(),
            vec![v(3), v(4)]
        );

        let reallocs = extract_reallocations(&diff);
        assert_eq!(reallocs, vec![(v(3), b(1), b(2)), (v(4), b(1), b(2))]);
    }

    #[test]
    #[should_panic(
        expected = "called `Mutator::swap_vessels` with Vessel 5, outside a graph of 5 vessels"
    )]
    fn test_swap_vessels_refuses_a_vessel_outside_the_graph() {
        let mut graph = setup_graph();
        let mut undo = ScheduleGraphUndoLog::new(10);
        let mut diff = ScheduleGraphDiff::new();
        let mut touched = TouchedBerths::new(graph.num_berths());
        Mutator::new(&mut graph, &mut undo, &mut diff, &mut touched).swap_vessels(v(0), v(5));
    }

    #[test]
    #[should_panic(
        expected = "called `Mutator::relocate_segment_before` with Vessel 7, outside a graph of 5 vessels"
    )]
    fn test_relocate_segment_before_refuses_a_vessel_outside_the_graph() {
        let mut graph = setup_graph();
        let mut undo = ScheduleGraphUndoLog::new(10);
        let mut diff = ScheduleGraphDiff::new();
        let mut touched = TouchedBerths::new(graph.num_berths());
        Mutator::new(&mut graph, &mut undo, &mut diff, &mut touched).relocate_segment_before(
            v(0),
            v(1),
            v(7),
        );
    }

    #[test]
    #[should_panic(
        expected = "called `Mutator::relocate_to_tail` with Berth 3, outside a graph of 3 berths"
    )]
    fn test_relocate_to_tail_refuses_a_berth_outside_the_graph() {
        let mut graph = setup_graph();
        let mut undo = ScheduleGraphUndoLog::new(10);
        let mut diff = ScheduleGraphDiff::new();
        let mut touched = TouchedBerths::new(graph.num_berths());
        Mutator::new(&mut graph, &mut undo, &mut diff, &mut touched).relocate_to_tail(v(0), b(3));
    }

    // V1 already stands directly before V2, and the segment [V0, V1] too: neither move changes anything, so none
    // is recorded and no berth is touched.
    #[test]
    fn test_relocating_before_the_vessel_already_followed_records_nothing() {
        let mut graph = setup_graph();
        let original = graph.clone();
        let mut undo = ScheduleGraphUndoLog::new(10);
        let mut diff = ScheduleGraphDiff::new();
        let mut touched = TouchedBerths::new(graph.num_berths());
        {
            let mut m = Mutator::new(&mut graph, &mut undo, &mut diff, &mut touched);
            m.relocate_before(v(1), v(2));
            m.relocate_segment_before(v(0), v(1), v(2));
            // SAFETY: every vessel handed over is one of the graph's.
            unsafe {
                m.relocate_before_unchecked(v(1), v(2));
                m.relocate_segment_before_unchecked(v(0), v(1), v(2));
            }
        }
        assert_eq!(graph, original);
        assert!(undo.is_empty());
        assert_eq!(touched.iter_touched_berths().count(), 0);
    }

    #[test]
    fn test_relocate_segment_to_the_tail_it_already_ends_keeps_the_graph() {
        let mut graph = setup_graph();
        let original = graph.clone();
        let mut undo = ScheduleGraphUndoLog::new(10);
        let mut diff = ScheduleGraphDiff::new();
        let mut touched = TouchedBerths::new(graph.num_berths());
        Mutator::new(&mut graph, &mut undo, &mut diff, &mut touched).relocate_segment_to_tail(
            v(3),
            v(4),
            b(1),
        );
        assert_eq!(graph, original);
        assert_eq!(diff.link_broken_count(), 0);
        undo.apply_rollback(&mut graph);
        assert_eq!(graph, original);
    }

    #[test]
    fn test_mutator_then_undo_rollback() {
        let mut graph = setup_graph();
        let original = graph.clone();
        let mut undo = ScheduleGraphUndoLog::new(20);
        let mut diff = ScheduleGraphDiff::new();
        {
            let mut touched = TouchedBerths::new(graph.num_berths());
            let mut m = Mutator::new(&mut graph, &mut undo, &mut diff, &mut touched);
            m.relocate_to_head(v(0), b(2));
            m.reverse_segment(v(3), v(4));
            m.swap_vessels(v(1), v(3));
        }

        assert_ne!(graph, original);

        undo.apply_rollback(&mut graph);

        assert_eq!(graph, original);
        assert!(undo.is_empty());
    }

    /// A move of the reference test, its vessels and berths named by numbers that `Harness::apply` reads modulo
    /// the counts, and its segments by a first vessel and how many vessels after it they reach.
    #[derive(Clone, Copy, Debug)]
    enum Change {
        SwapVessels(usize, usize),
        SwapSegments(usize, usize, usize, usize),
        Reverse(usize, usize),
        After(usize, usize),
        Before(usize, usize),
        ToHead(usize, usize),
        ToTail(usize, usize),
        SegmentAfter(usize, usize, usize),
        SegmentBefore(usize, usize, usize),
        SegmentToHead(usize, usize, usize),
        SegmentToTail(usize, usize, usize),
    }

    fn change() -> impl Strategy<Value = Change> {
        let id = || 0_usize..16;
        let extra = || 0_usize..4;
        prop_oneof![
            (id(), id()).prop_map(|(a, b)| Change::SwapVessels(a, b)),
            (id(), extra(), id(), extra())
                .prop_map(|(a, x, b, y)| Change::SwapSegments(a, x, b, y)),
            (id(), extra()).prop_map(|(a, x)| Change::Reverse(a, x)),
            (id(), id()).prop_map(|(a, b)| Change::After(a, b)),
            (id(), id()).prop_map(|(a, b)| Change::Before(a, b)),
            (id(), id()).prop_map(|(a, b)| Change::ToHead(a, b)),
            (id(), id()).prop_map(|(a, b)| Change::ToTail(a, b)),
            (id(), extra(), id()).prop_map(|(a, x, b)| Change::SegmentAfter(a, x, b)),
            (id(), extra(), id()).prop_map(|(a, x, b)| Change::SegmentBefore(a, x, b)),
            (id(), extra(), id()).prop_map(|(a, x, b)| Change::SegmentToHead(a, x, b)),
            (id(), extra(), id()).prop_map(|(a, x, b)| Change::SegmentToTail(a, x, b)),
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

    /// The vessels from `first` along its berth, `extra` more at most.
    fn segment_of(berths: &[Vec<usize>], first: usize, extra: usize) -> Vec<usize> {
        let (berth, place) = locate(berths, first);
        let end = (place + 1 + extra).min(berths[berth].len());
        berths[berth][place..end].to_vec()
    }

    /// The reference with `segment` taken out, and where it stood.
    fn take(berths: &mut [Vec<usize>], segment: &[usize]) -> (usize, usize) {
        let (berth, place) = locate(berths, segment[0]);
        berths[berth].drain(place..place + segment.len());
        (berth, place)
    }

    /// A graph moved by a mutator through checked moves, a second by unchecked ones, and the reference: the
    /// vessels of each berth in order, as a plain list.
    struct Harness {
        checked: ScheduleGraph,
        unchecked: ScheduleGraph,
        reference: Vec<Vec<usize>>,
        undo: ScheduleGraphUndoLog,
        diff: ScheduleGraphDiff,
        touched: TouchedBerths,
        unchecked_undo: ScheduleGraphUndoLog,
        unchecked_diff: ScheduleGraphDiff,
        unchecked_touched: TouchedBerths,
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
                checked: graph.clone(),
                unchecked: graph,
                reference,
                undo: ScheduleGraphUndoLog::new(0),
                diff: ScheduleGraphDiff::new(),
                touched: TouchedBerths::new(num_berths),
                unchecked_undo: ScheduleGraphUndoLog::new(0),
                unchecked_diff: ScheduleGraphDiff::new(),
                unchecked_touched: TouchedBerths::new(num_berths),
            }
        }

        /// Applies `change` to the reference and to both graphs, unless it breaks a precondition of the move.
        fn apply(&mut self, change: Change) {
            let n = self.checked.num_vessels();
            let berths = self.checked.num_berths();
            let mut checked = Mutator::new(
                &mut self.checked,
                &mut self.undo,
                &mut self.diff,
                &mut self.touched,
            );
            let mut unchecked = Mutator::new(
                &mut self.unchecked,
                &mut self.unchecked_undo,
                &mut self.unchecked_diff,
                &mut self.unchecked_touched,
            );
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
                }
                Change::SwapSegments(a, x, c, y) => {
                    let (one, two) = (
                        segment_of(reference, a % n, x),
                        segment_of(reference, c % n, y),
                    );
                    if one[0] == two[0] {
                        return;
                    }
                    if one.iter().any(|vessel| two.contains(vessel)) {
                        return;
                    }
                    // Each segment is written where the first vessel of the other stood.
                    let mut rebuilt = Vec::new();
                    for vessels in reference.iter() {
                        let mut out = Vec::new();
                        for &vessel in vessels {
                            if vessel == one[0] {
                                out.extend(&two);
                            } else if vessel == two[0] {
                                out.extend(&one);
                            } else if !one.contains(&vessel) && !two.contains(&vessel) {
                                out.push(vessel);
                            }
                        }
                        rebuilt.push(out);
                    }
                    *reference = rebuilt;
                    let (a0, a1, b0, b1) =
                        (one[0], *one.last().unwrap(), two[0], *two.last().unwrap());
                    checked.swap_segments(v(a0), v(a1), v(b0), v(b1));
                    // SAFETY: the four vessels are below the count.
                    unsafe { unchecked.swap_segments_unchecked(v(a0), v(a1), v(b0), v(b1)) };
                }
                Change::Reverse(a, x) => {
                    let segment = segment_of(reference, a % n, x);
                    let (berth, place) = locate(reference, segment[0]);
                    reference[berth][place..place + segment.len()].reverse();
                    let (first, last) = (segment[0], *segment.last().unwrap());
                    checked.reverse_segment(v(first), v(last));
                    // SAFETY: both vessels are below the count.
                    unsafe { unchecked.reverse_segment_unchecked(v(first), v(last)) };
                }
                Change::After(a, c) | Change::Before(a, c) => {
                    let after = matches!(change, Change::After(..));
                    let (a, c) = (a % n, c % n);
                    if a != c {
                        take(reference, &[a]);
                        let (berth, place) = locate(reference, c);
                        reference[berth].insert(if after { place + 1 } else { place }, a);
                    }
                    if after {
                        checked.relocate_after(v(a), v(c));
                        // SAFETY: both vessels are below the count.
                        unsafe { unchecked.relocate_after_unchecked(v(a), v(c)) };
                    } else {
                        checked.relocate_before(v(a), v(c));
                        // SAFETY: both vessels are below the count.
                        unsafe { unchecked.relocate_before_unchecked(v(a), v(c)) };
                    }
                }
                Change::ToHead(a, c) | Change::ToTail(a, c) => {
                    let head = matches!(change, Change::ToHead(..));
                    let (a, berth) = (a % n, c % berths);
                    take(reference, &[a]);
                    if head {
                        reference[berth].insert(0, a);
                        checked.relocate_to_head(v(a), b(berth));
                        // SAFETY: the vessel and the berth are below the counts.
                        unsafe { unchecked.relocate_to_head_unchecked(v(a), b(berth)) };
                    } else {
                        reference[berth].push(a);
                        checked.relocate_to_tail(v(a), b(berth));
                        // SAFETY: the vessel and the berth are below the counts.
                        unsafe { unchecked.relocate_to_tail_unchecked(v(a), b(berth)) };
                    }
                }
                Change::SegmentAfter(a, x, c) | Change::SegmentBefore(a, x, c) => {
                    let after = matches!(change, Change::SegmentAfter(..));
                    let segment = segment_of(reference, a % n, x);
                    let anchor = c % n;
                    if segment.contains(&anchor) {
                        return;
                    }
                    take(reference, &segment);
                    let (berth, place) = locate(reference, anchor);
                    let at = if after { place + 1 } else { place };
                    reference[berth].splice(at..at, segment.iter().copied());
                    let (first, last) = (v(segment[0]), v(*segment.last().unwrap()));
                    if after {
                        checked.relocate_segment_after(first, last, v(anchor));
                        // SAFETY: the three vessels are below the count.
                        unsafe {
                            unchecked.relocate_segment_after_unchecked(first, last, v(anchor))
                        };
                    } else {
                        checked.relocate_segment_before(first, last, v(anchor));
                        // SAFETY: the three vessels are below the count.
                        unsafe {
                            unchecked.relocate_segment_before_unchecked(first, last, v(anchor))
                        };
                    }
                }
                Change::SegmentToHead(a, x, c) | Change::SegmentToTail(a, x, c) => {
                    let head = matches!(change, Change::SegmentToHead(..));
                    let segment = segment_of(reference, a % n, x);
                    let berth = c % berths;
                    take(reference, &segment);
                    let at = if head { 0 } else { reference[berth].len() };
                    reference[berth].splice(at..at, segment.iter().copied());
                    let (first, last) = (v(segment[0]), v(*segment.last().unwrap()));
                    if head {
                        checked.relocate_segment_to_head(first, last, b(berth));
                        // SAFETY: the vessels and the berth are below the counts.
                        unsafe {
                            unchecked.relocate_segment_to_head_unchecked(first, last, b(berth))
                        };
                    } else {
                        checked.relocate_segment_to_tail(first, last, b(berth));
                        // SAFETY: the vessels and the berth are below the counts.
                        unsafe {
                            unchecked.relocate_segment_to_tail_unchecked(first, last, b(berth))
                        };
                    }
                }
            }
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
                let reversed: Vec<usize> = graph
                    .vessel_sequence_rev_iter(berth)
                    .map(VesselIndex::index)
                    .collect();
                assert!(
                    vessels.iter().rev().eq(reversed.iter()),
                    "the walks of {berth} should mirror each other"
                );
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
            let start = harness.checked.clone();
            let start_order = order_of(&start);
            for &change in &changes {
                let before = order_of(&harness.checked);
                harness.diff.clear();
                harness.touched.reset();
                harness.apply(change);
                let after = order_of(&harness.checked);
                prop_assert_eq!(&after, &harness.reference, "after {:?}", change);
                prop_assert_eq!(&order_of(&harness.unchecked), &after, "unchecked, after {:?}", change);
                for berth in 0..num_berths {
                    if before[berth] != after[berth] {
                        prop_assert!(harness.touched.is_touched(b(berth)), "{:?} changed berth {}", change, berth);
                    }
                }
                let moved: Vec<_> = harness.diff.reallocations().collect();
                for vessel in 0..berths.len() {
                    let (from, to) = (locate(&before, vessel).0, locate(&after, vessel).0);
                    let recorded = moved.iter().filter(|&&(recorded, _, _)| recorded == v(vessel)).count();
                    prop_assert_eq!(recorded, usize::from(from != to), "{:?} and vessel {}", change, vessel);
                }
            }
            harness.undo.apply_rollback(&mut harness.checked);
            // SAFETY: the log holds only moves a mutator over this graph recorded.
            unsafe { harness.unchecked_undo.apply_rollback_unchecked(&mut harness.unchecked) };
            prop_assert_eq!(&harness.checked, &start);
            prop_assert_eq!(&order_of(&harness.checked), &start_order);
            prop_assert_eq!(&order_of(&harness.unchecked), &start_order);
        }
    }
}
