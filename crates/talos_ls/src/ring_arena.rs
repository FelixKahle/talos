// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A fixed set of nodes arranged in closed rings, spliced in constant time.

use std::fmt;
use std::iter::FusedIterator;
use talos_core::{TypedIndex, TypedIndexTag};

/// The tag of [`Node`].
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeIndexTag;

impl TypedIndexTag for NodeIndexTag {
    const NAME: &'static str = "Node";
}

/// A node of a [`RingArena`]: a position in its links.
pub type Node = TypedIndex<NodeIndexTag>;

/// A fixed set of nodes arranged in closed rings, as a doubly linked list per ring.
///
/// Every node has a successor and a predecessor, following which leads around its ring and back; a node alone is
/// a ring of one. The rings are rearranged in place by steps that each touch a constant number of links (moving a
/// node or a segment, swapping two nodes or two segments) and by reversing a segment, which costs its length.
/// Nodes never enter or leave once the arena is built: [`len`](RingArena::len) stays fixed until the arena is
/// rebuilt by [`reset_to_self_loops`](RingArena::reset_to_self_loops) or one of the `overwrite_from_*` methods.
///
/// Every step comes as a pair: a safe method that checks in every build that the nodes it is handed lie in the
/// arena, and an `_unchecked` twin for the caller whose own invariant proves that they do.
///
/// The steps on segments also take a condition no constant-time check can confirm: `last` follows `first` along
/// the successors, and the segments or the anchor they are given do not overlap. It is stated in each step's
/// `# Preconditions` and checked in debug builds. Breaking it breaks the rings (a wrong answer), but never makes an
/// access out of bounds: every link the arena holds names one of its nodes, whatever the steps are handed.
// Invariant: `prev` and `next` have the same length, every entry of both is a node below that length, and
// `prev[next[v]] == v` for every node `v`, so that `next` is a permutation whose cycles are the rings and `prev` is
// its inverse. The first two parts are what memory rests on: they make every link a valid index into both columns,
// and every step keeps them, since a step writes only nodes it was handed (which lie in the arena) or read from a
// column. The last part makes the nodes rings; the steps keep it under their preconditions.
#[derive(Clone, PartialEq, Eq, Debug, Hash, Default)]
pub struct RingArena {
    /// The predecessor of each node, per node.
    prev: Vec<Node>,
    /// The successor of each node, per node.
    next: Vec<Node>,
}

/// A node of one arena that is known to lie inside it.
///
/// Only the arena makes one: from a node it has checked or been promised to lie inside it, or by reading one of its
/// own links, every one of which is a node of the arena by its invariant. A slot never leaves the method of the
/// arena that made it, and the arena's length does not change while that method runs, so the bound holds for as
/// long as the slot exists.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Slot(Node);

/// A node handed to an arena that does not lie in it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct NodeOutside {
    node: Node,
    len: usize,
}

impl fmt::Display for NodeOutside {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}, outside an arena of {} nodes", self.node, self.len)
    }
}

/// Why a pair of link columns does not describe closed rings.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum RingViolation {
    /// The two columns have different lengths.
    LengthsDiffer { prev: usize, next: usize },
    /// The columns have more nodes than a [`Node`] can name.
    TooManyNodes { len: usize },
    /// A link names a node outside the columns.
    LinkOutside { node: Node, link: Node, len: usize },
    /// The predecessor of a node's successor is not the node.
    NotInverse {
        node: Node,
        next: Node,
        prev_of_next: Node,
    },
}

impl fmt::Display for RingViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::LengthsDiffer { prev, next } => {
                write!(f, "{prev} predecessors and {next} successors")
            }
            Self::TooManyNodes { len } => write!(f, "{len} nodes, more than a node id can name"),
            Self::LinkOutside { node, link, len } => {
                write!(
                    f,
                    "a link from {node} to {link}, outside an arena of {len} nodes"
                )
            }
            Self::NotInverse {
                node,
                next,
                prev_of_next,
            } => write!(
                f,
                "{node} followed by {next}, whose predecessor is {prev_of_next}"
            ),
        }
    }
}

/// Whether every node of an arena of `len` nodes has a [`Node`] that names it.
fn nameable(len: usize) -> bool {
    len.checked_sub(1)
        .is_none_or(|last| Node::try_new(last).is_some())
}

/// The first way in which `prev` and `next` fail to describe closed rings, or `None` if they describe them.
fn ring_violation(prev: &[Node], next: &[Node]) -> Option<RingViolation> {
    let len = prev.len();
    if next.len() != len {
        return Some(RingViolation::LengthsDiffer {
            prev: len,
            next: next.len(),
        });
    }
    if !nameable(len) {
        return Some(RingViolation::TooManyNodes { len });
    }
    for (node, (&prev_link, &next_link)) in (0..len).map(Node::new).zip(prev.iter().zip(next)) {
        if let Some(link) = [prev_link, next_link]
            .into_iter()
            .find(|link| link.index() >= len)
        {
            return Some(RingViolation::LinkOutside { node, link, len });
        }
    }
    (0..len).map(Node::new).find_map(|node| {
        let following = next[node.index()];
        let prev_of_next = prev[following.index()];
        (prev_of_next != node).then_some(RingViolation::NotInverse {
            node,
            next: following,
            prev_of_next,
        })
    })
}

impl RingArena {
    /// The arena whose node `v` has predecessor `prev[v]` and successor `next[v]`.
    ///
    /// # Panics
    ///
    /// Panics unless the two columns describe closed rings: they must have the same length, every link must name
    /// a node below that length, and the predecessor of each node's successor must be the node.
    #[track_caller]
    pub fn new(prev: Vec<Node>, next: Vec<Node>) -> Self {
        if let Some(violation) = ring_violation(&prev, &next) {
            panic!("called `RingArena::new` with {violation}");
        }
        Self { prev, next }
    }

    /// Replaces the arena by the one [`new`](RingArena::new) builds from copies of the two columns, reusing the
    /// arena's memory.
    ///
    /// # Panics
    ///
    /// Panics unless the two columns describe closed rings, as for [`new`](RingArena::new); the arena is then
    /// left unchanged.
    #[track_caller]
    pub fn overwrite_from_slices(&mut self, prev: &[Node], next: &[Node]) {
        if let Some(violation) = ring_violation(prev, next) {
            panic!("called `RingArena::overwrite_from_slices` with {violation}");
        }
        self.prev.clear();
        self.prev.extend_from_slice(prev);
        self.next.clear();
        self.next.extend_from_slice(next);
    }

    /// Replaces the arena by a copy of `other`, reusing the arena's memory.
    pub fn overwrite_from_arena(&mut self, other: &RingArena) {
        self.prev.clone_from(&other.prev);
        self.next.clone_from(&other.next);
    }

    /// Replaces the arena by `len` nodes, each a ring of its own.
    ///
    /// # Panics
    ///
    /// Panics if a node of the arena would lie beyond what a [`Node`] can name; the arena is then left
    /// unchanged.
    #[track_caller]
    pub fn reset_to_self_loops(&mut self, len: usize) {
        // Checked before anything is cleared: a panic midway would leave columns of different lengths, which a
        // caller that catches the panic could then read out of bounds through a safe step.
        assert!(
            nameable(len),
            "called `RingArena::reset_to_self_loops` with {len} nodes, more than a node id can name"
        );
        self.prev.clear();
        self.next.clear();
        self.prev.extend((0..len).map(Node::new));
        self.next.extend_from_slice(&self.prev);
    }

    /// The number of nodes.
    #[inline]
    pub fn len(&self) -> usize {
        self.prev.len()
    }

    /// Whether the arena has no node.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.prev.is_empty()
    }

    /// The successor of `node` in its ring.
    ///
    /// # Panics
    ///
    /// Panics if `node` lies outside the arena.
    #[inline]
    #[track_caller]
    pub fn next(&self, node: Node) -> Node {
        let [node] = match self.slots([node]) {
            Ok(slots) => slots,
            Err(outside) => panic!("called `RingArena::next` with {outside}"),
        };
        self.next_slot(node).0
    }

    /// The successor of `node` in its ring, for a `node` the caller knows to lie in the arena.
    ///
    /// # Safety
    ///
    /// `node` must lie in the arena, the condition [`next`](RingArena::next) checks.
    #[inline]
    pub unsafe fn next_unchecked(&self, node: Node) -> Node {
        debug_assert!(
            node.index() < self.len(),
            "called `RingArena::next_unchecked` with {node} outside the arena"
        );
        // SAFETY: the caller promises that `node` lies in the arena.
        let node = unsafe { self.slot_unchecked(node) };
        self.next_slot(node).0
    }

    /// The predecessor of `node` in its ring.
    ///
    /// # Panics
    ///
    /// Panics if `node` lies outside the arena.
    #[inline]
    #[track_caller]
    pub fn prev(&self, node: Node) -> Node {
        let [node] = match self.slots([node]) {
            Ok(slots) => slots,
            Err(outside) => panic!("called `RingArena::prev` with {outside}"),
        };
        self.prev_slot(node).0
    }

    /// The predecessor of `node` in its ring, for a `node` the caller knows to lie in the arena.
    ///
    /// # Safety
    ///
    /// `node` must lie in the arena, the condition [`prev`](RingArena::prev) checks.
    #[inline]
    pub unsafe fn prev_unchecked(&self, node: Node) -> Node {
        debug_assert!(
            node.index() < self.len(),
            "called `RingArena::prev_unchecked` with {node} outside the arena"
        );
        // SAFETY: the caller promises that `node` lies in the arena.
        let node = unsafe { self.slot_unchecked(node) };
        self.prev_slot(node).0
    }

    /// The nodes from `start_node` along the successors, up to but not including `stop_node`.
    ///
    /// The walk ends after at most [`len`](RingArena::len) nodes: if `stop_node` is not on the ring of
    /// `start_node`, the walk goes round that ring until it has yielded that many.
    ///
    /// # Panics
    ///
    /// Panics if `start_node` or `stop_node` lies outside the arena.
    #[track_caller]
    pub fn sequence_iter(&self, start_node: Node, stop_node: Node) -> RingSequenceIter<'_> {
        let [start_node, stop_node] = match self.slots([start_node, stop_node]) {
            Ok(slots) => slots,
            Err(outside) => panic!("called `RingArena::sequence_iter` with {outside}"),
        };
        RingSequenceIter::new(&self.next, start_node, stop_node)
    }

    /// The nodes from `start_node` along the successors, up to but not including `stop_node`, for nodes the
    /// caller knows to lie in the arena.
    ///
    /// # Safety
    ///
    /// `start_node` and `stop_node` must lie in the arena, the condition
    /// [`sequence_iter`](RingArena::sequence_iter) checks.
    pub unsafe fn sequence_iter_unchecked(
        &self,
        start_node: Node,
        stop_node: Node,
    ) -> RingSequenceIter<'_> {
        debug_assert!(
            start_node.index() < self.len() && stop_node.index() < self.len(),
            "called `RingArena::sequence_iter_unchecked` with {start_node} or {stop_node} outside the arena"
        );
        // SAFETY: the caller promises that both nodes lie in the arena.
        let (start_node, stop_node) = unsafe {
            (
                self.slot_unchecked(start_node),
                self.slot_unchecked(stop_node),
            )
        };
        RingSequenceIter::new(&self.next, start_node, stop_node)
    }

    /// The nodes from `start_node` along the predecessors, up to but not including `stop_node`.
    ///
    /// The walk ends after at most [`len`](RingArena::len) nodes, as for
    /// [`sequence_iter`](RingArena::sequence_iter).
    ///
    /// # Panics
    ///
    /// Panics if `start_node` or `stop_node` lies outside the arena.
    #[track_caller]
    pub fn sequence_rev_iter(&self, start_node: Node, stop_node: Node) -> RingSequenceRevIter<'_> {
        let [start_node, stop_node] = match self.slots([start_node, stop_node]) {
            Ok(slots) => slots,
            Err(outside) => panic!("called `RingArena::sequence_rev_iter` with {outside}"),
        };
        RingSequenceRevIter::new(&self.prev, start_node, stop_node)
    }

    /// The nodes from `start_node` along the predecessors, up to but not including `stop_node`, for nodes the
    /// caller knows to lie in the arena.
    ///
    /// # Safety
    ///
    /// `start_node` and `stop_node` must lie in the arena, the condition
    /// [`sequence_rev_iter`](RingArena::sequence_rev_iter) checks.
    pub unsafe fn sequence_rev_iter_unchecked(
        &self,
        start_node: Node,
        stop_node: Node,
    ) -> RingSequenceRevIter<'_> {
        debug_assert!(
            start_node.index() < self.len() && stop_node.index() < self.len(),
            "called `RingArena::sequence_rev_iter_unchecked` with {start_node} or {stop_node} outside the arena"
        );
        // SAFETY: the caller promises that both nodes lie in the arena.
        let (start_node, stop_node) = unsafe {
            (
                self.slot_unchecked(start_node),
                self.slot_unchecked(stop_node),
            )
        };
        RingSequenceRevIter::new(&self.prev, start_node, stop_node)
    }

    /// The links `(node, successor)` from `start_node` along the successors whose successor is not `stop_node`.
    ///
    /// These are the links between consecutive nodes of [`sequence_iter`](RingArena::sequence_iter) with the
    /// same nodes.
    ///
    /// # Panics
    ///
    /// Panics if `start_node` or `stop_node` lies outside the arena.
    #[track_caller]
    pub fn edge_iter(&self, start_node: Node, stop_node: Node) -> RingEdgeIter<'_> {
        let [start_node, stop_node] = match self.slots([start_node, stop_node]) {
            Ok(slots) => slots,
            Err(outside) => panic!("called `RingArena::edge_iter` with {outside}"),
        };
        RingEdgeIter::new(&self.next, start_node, stop_node)
    }

    /// The links `(node, successor)` from `start_node` along the successors whose successor is not `stop_node`,
    /// for nodes the caller knows to lie in the arena.
    ///
    /// # Safety
    ///
    /// `start_node` and `stop_node` must lie in the arena, the condition [`edge_iter`](RingArena::edge_iter)
    /// checks.
    pub unsafe fn edge_iter_unchecked(
        &self,
        start_node: Node,
        stop_node: Node,
    ) -> RingEdgeIter<'_> {
        debug_assert!(
            start_node.index() < self.len() && stop_node.index() < self.len(),
            "called `RingArena::edge_iter_unchecked` with {start_node} or {stop_node} outside the arena"
        );
        // SAFETY: the caller promises that both nodes lie in the arena.
        let (start_node, stop_node) = unsafe {
            (
                self.slot_unchecked(start_node),
                self.slot_unchecked(stop_node),
            )
        };
        RingEdgeIter::new(&self.next, start_node, stop_node)
    }

    /// Exchanges the places of `first` and `second`, in one ring or across two.
    ///
    /// ```text
    /// before:  ... a_prev -> first -> a_next ...      ... b_prev -> second -> b_next ...
    /// after:   ... a_prev -> second -> a_next ...     ... b_prev -> first -> b_next ...
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `first` or `second` lies outside the arena.
    #[track_caller]
    pub fn swap_nodes(&mut self, first: Node, second: Node) {
        let [first, second] = match self.slots([first, second]) {
            Ok(slots) => slots,
            Err(outside) => panic!("called `RingArena::swap_nodes` with {outside}"),
        };
        self.swap(first, second);
    }

    /// Exchanges the places of `first` and `second`, for nodes the caller knows to lie in the arena.
    ///
    /// # Safety
    ///
    /// `first` and `second` must lie in the arena, the condition [`swap_nodes`](RingArena::swap_nodes) checks.
    pub unsafe fn swap_nodes_unchecked(&mut self, first: Node, second: Node) {
        debug_assert!(
            first.index() < self.len() && second.index() < self.len(),
            "called `RingArena::swap_nodes_unchecked` with {first} or {second} outside the arena"
        );
        // SAFETY: the caller promises that both nodes lie in the arena.
        let (first, second) = unsafe { (self.slot_unchecked(first), self.slot_unchecked(second)) };
        self.swap(first, second);
    }

    /// Exchanges the places of the segments `a_first ..= a_last` and `b_first ..= b_last`, in one ring or across
    /// two.
    ///
    /// ```text
    /// before:  ... a_prev -> [a_first ... a_last] -> a_next ...   ... b_prev -> [b_first ... b_last] -> b_next ...
    /// after:   ... a_prev -> [b_first ... b_last] -> a_next ...   ... b_prev -> [a_first ... a_last] -> b_next ...
    /// ```
    ///
    /// # Preconditions
    ///
    /// `a_last` follows `a_first` along the successors, `b_last` follows `b_first`, and the two segments share no
    /// node unless they are the same segment. Otherwise the rings break, though no access leaves the arena; this
    /// is checked in debug builds only, as the check walks both segments of a constant-time step.
    ///
    /// # Panics
    ///
    /// Panics if any of the four nodes lies outside the arena.
    #[track_caller]
    pub fn swap_segments(&mut self, a_first: Node, a_last: Node, b_first: Node, b_last: Node) {
        let [a_first, a_last, b_first, b_last] =
            match self.slots([a_first, a_last, b_first, b_last]) {
                Ok(slots) => slots,
                Err(outside) => panic!("called `RingArena::swap_segments` with {outside}"),
            };
        self.swap_ranges(a_first, a_last, b_first, b_last);
    }

    /// Exchanges the places of two segments, for nodes the caller knows to lie in the arena.
    ///
    /// # Preconditions
    ///
    /// Those of [`swap_segments`](RingArena::swap_segments).
    ///
    /// # Safety
    ///
    /// The four nodes must lie in the arena, the condition [`swap_segments`](RingArena::swap_segments) checks.
    pub unsafe fn swap_segments_unchecked(
        &mut self,
        a_first: Node,
        a_last: Node,
        b_first: Node,
        b_last: Node,
    ) {
        debug_assert!(
            [a_first, a_last, b_first, b_last]
                .iter()
                .all(|node| node.index() < self.len()),
            "called `RingArena::swap_segments_unchecked` with a node outside the arena"
        );
        // SAFETY: the caller promises that all four nodes lie in the arena.
        let [a_first, a_last, b_first, b_last] =
            [a_first, a_last, b_first, b_last].map(|node| unsafe { self.slot_unchecked(node) });
        self.swap_ranges(a_first, a_last, b_first, b_last);
    }

    /// Moves `node` out of its ring to directly after `anchor`; nothing changes if `node` is `anchor` or already
    /// follows it.
    ///
    /// ```text
    /// before:  ... prev -> node -> next ...      ... anchor -> anchor_next ...
    /// after:   ... prev -> next ...              ... anchor -> node -> anchor_next ...
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `node` or `anchor` lies outside the arena.
    #[track_caller]
    pub fn relocate_after(&mut self, node: Node, anchor: Node) {
        let [node, anchor] = match self.slots([node, anchor]) {
            Ok(slots) => slots,
            Err(outside) => panic!("called `RingArena::relocate_after` with {outside}"),
        };
        self.move_after(node, anchor);
    }

    /// Moves `node` to directly after `anchor`, for nodes the caller knows to lie in the arena.
    ///
    /// # Safety
    ///
    /// `node` and `anchor` must lie in the arena, the condition [`relocate_after`](RingArena::relocate_after)
    /// checks.
    pub unsafe fn relocate_after_unchecked(&mut self, node: Node, anchor: Node) {
        debug_assert!(
            node.index() < self.len() && anchor.index() < self.len(),
            "called `RingArena::relocate_after_unchecked` with {node} or {anchor} outside the arena"
        );
        // SAFETY: the caller promises that both nodes lie in the arena.
        let (node, anchor) = unsafe { (self.slot_unchecked(node), self.slot_unchecked(anchor)) };
        self.move_after(node, anchor);
    }

    /// Moves `node` out of its ring to directly before `anchor`; nothing changes if `node` already precedes it.
    ///
    /// # Panics
    ///
    /// Panics if `node` or `anchor` lies outside the arena.
    #[track_caller]
    pub fn relocate_before(&mut self, node: Node, anchor: Node) {
        let [node, anchor] = match self.slots([node, anchor]) {
            Ok(slots) => slots,
            Err(outside) => panic!("called `RingArena::relocate_before` with {outside}"),
        };
        let anchor_prev = self.prev_slot(anchor);
        self.move_after(node, anchor_prev);
    }

    /// Moves `node` to directly before `anchor`, for nodes the caller knows to lie in the arena.
    ///
    /// # Safety
    ///
    /// `node` and `anchor` must lie in the arena, the condition [`relocate_before`](RingArena::relocate_before)
    /// checks.
    pub unsafe fn relocate_before_unchecked(&mut self, node: Node, anchor: Node) {
        debug_assert!(
            node.index() < self.len() && anchor.index() < self.len(),
            "called `RingArena::relocate_before_unchecked` with {node} or {anchor} outside the arena"
        );
        // SAFETY: the caller promises that both nodes lie in the arena.
        let (node, anchor) = unsafe { (self.slot_unchecked(node), self.slot_unchecked(anchor)) };
        let anchor_prev = self.prev_slot(anchor);
        self.move_after(node, anchor_prev);
    }

    /// Moves the segment `first ..= last` out of its ring to directly after `anchor`; nothing changes if the
    /// segment already follows it.
    ///
    /// ```text
    /// before:  ... prev -> [first ... last] -> next ...      ... anchor -> anchor_next ...
    /// after:   ... prev -> next ...                          ... anchor -> [first ... last] -> anchor_next ...
    /// ```
    ///
    /// # Preconditions
    ///
    /// `last` follows `first` along the successors, and `anchor` is not a node of the segment. Otherwise the
    /// rings break, though no access leaves the arena; this is checked in debug builds only, as the check walks
    /// the segment of a constant-time step.
    ///
    /// # Panics
    ///
    /// Panics if `first`, `last` or `anchor` lies outside the arena.
    #[track_caller]
    pub fn relocate_segment_after(&mut self, first: Node, last: Node, anchor: Node) {
        let [first, last, anchor] = match self.slots([first, last, anchor]) {
            Ok(slots) => slots,
            Err(outside) => panic!("called `RingArena::relocate_segment_after` with {outside}"),
        };
        self.move_range_after(first, last, anchor);
    }

    /// Moves the segment `first ..= last` to directly after `anchor`, for nodes the caller knows to lie in the
    /// arena.
    ///
    /// # Preconditions
    ///
    /// Those of [`relocate_segment_after`](RingArena::relocate_segment_after).
    ///
    /// # Safety
    ///
    /// The three nodes must lie in the arena, the condition
    /// [`relocate_segment_after`](RingArena::relocate_segment_after) checks.
    pub unsafe fn relocate_segment_after_unchecked(
        &mut self,
        first: Node,
        last: Node,
        anchor: Node,
    ) {
        debug_assert!(
            [first, last, anchor]
                .iter()
                .all(|node| node.index() < self.len()),
            "called `RingArena::relocate_segment_after_unchecked` with a node outside the arena"
        );
        // SAFETY: the caller promises that all three nodes lie in the arena.
        let [first, last, anchor] =
            [first, last, anchor].map(|node| unsafe { self.slot_unchecked(node) });
        self.move_range_after(first, last, anchor);
    }

    /// Moves the segment `first ..= last` out of its ring to directly before `anchor`; nothing changes if the
    /// segment already precedes it.
    ///
    /// # Preconditions
    ///
    /// `last` follows `first` along the successors, and `anchor` is not a node of the segment, as for
    /// [`relocate_segment_after`](RingArena::relocate_segment_after).
    ///
    /// # Panics
    ///
    /// Panics if `first`, `last` or `anchor` lies outside the arena.
    #[track_caller]
    pub fn relocate_segment_before(&mut self, first: Node, last: Node, anchor: Node) {
        let [first, last, anchor] = match self.slots([first, last, anchor]) {
            Ok(slots) => slots,
            Err(outside) => panic!("called `RingArena::relocate_segment_before` with {outside}"),
        };
        self.move_range_before(first, last, anchor);
    }

    /// Moves the segment `first ..= last` to directly before `anchor`, for nodes the caller knows to lie in the
    /// arena.
    ///
    /// # Preconditions
    ///
    /// Those of [`relocate_segment_before`](RingArena::relocate_segment_before).
    ///
    /// # Safety
    ///
    /// The three nodes must lie in the arena, the condition
    /// [`relocate_segment_before`](RingArena::relocate_segment_before) checks.
    pub unsafe fn relocate_segment_before_unchecked(
        &mut self,
        first: Node,
        last: Node,
        anchor: Node,
    ) {
        debug_assert!(
            [first, last, anchor]
                .iter()
                .all(|node| node.index() < self.len()),
            "called `RingArena::relocate_segment_before_unchecked` with a node outside the arena"
        );
        // SAFETY: the caller promises that all three nodes lie in the arena.
        let [first, last, anchor] =
            [first, last, anchor].map(|node| unsafe { self.slot_unchecked(node) });
        self.move_range_before(first, last, anchor);
    }

    /// Reverses the order of the segment `first ..= last` within its ring.
    ///
    /// ```text
    /// before:  ... prev -> [a -> b -> c -> d] -> next ...
    /// after:   ... prev -> [d -> c -> b -> a] -> next ...
    /// ```
    ///
    /// The cost is the length of the segment.
    ///
    /// # Preconditions
    ///
    /// `last` follows `first` along the successors. Otherwise the rings break, though no access leaves the arena
    /// and the walk ends after [`len`](RingArena::len) nodes; this is checked in debug builds only, as the check
    /// walks the segment a second time.
    ///
    /// # Panics
    ///
    /// Panics if `first` or `last` lies outside the arena.
    #[track_caller]
    pub fn reverse_segment(&mut self, first: Node, last: Node) {
        let [first, last] = match self.slots([first, last]) {
            Ok(slots) => slots,
            Err(outside) => panic!("called `RingArena::reverse_segment` with {outside}"),
        };
        self.reverse(first, last);
    }

    /// Reverses the order of the segment `first ..= last`, for nodes the caller knows to lie in the arena.
    ///
    /// # Preconditions
    ///
    /// Those of [`reverse_segment`](RingArena::reverse_segment).
    ///
    /// # Safety
    ///
    /// `first` and `last` must lie in the arena, the condition [`reverse_segment`](RingArena::reverse_segment)
    /// checks.
    pub unsafe fn reverse_segment_unchecked(&mut self, first: Node, last: Node) {
        debug_assert!(
            first.index() < self.len() && last.index() < self.len(),
            "called `RingArena::reverse_segment_unchecked` with {first} or {last} outside the arena"
        );
        // SAFETY: the caller promises that both nodes lie in the arena.
        let (first, last) = unsafe { (self.slot_unchecked(first), self.slot_unchecked(last)) };
        self.reverse(first, last);
    }

    /// The slots of `nodes`, or the first of them that lies outside the arena.
    #[inline]
    fn slots<const N: usize>(&self, nodes: [Node; N]) -> Result<[Slot; N], NodeOutside> {
        let len = self.len();
        match nodes.iter().find(|node| node.index() >= len) {
            Some(&node) => Err(NodeOutside { node, len }),
            None => Ok(nodes.map(Slot)),
        }
    }

    /// The slot of `node`.
    ///
    /// # Safety
    ///
    /// `node` must lie in the arena: every later access through the slot trusts it.
    #[inline]
    unsafe fn slot_unchecked(&self, node: Node) -> Slot {
        debug_assert!(
            node.index() < self.len(),
            "called `RingArena::slot_unchecked` with {node} outside the arena"
        );
        Slot(node)
    }

    /// The slot of the successor of `node`.
    #[inline]
    fn next_slot(&self, node: Slot) -> Slot {
        debug_assert!(node.0.index() < self.next.len());
        // SAFETY: a `Slot` lies in its arena, and every link of the arena names one of its nodes (`RingArena`).
        Slot(unsafe { *self.next.get_unchecked(node.0.index()) })
    }

    /// The slot of the predecessor of `node`.
    #[inline]
    fn prev_slot(&self, node: Slot) -> Slot {
        debug_assert!(node.0.index() < self.prev.len());
        // SAFETY: a `Slot` lies in its arena, and every link of the arena names one of its nodes (`RingArena`).
        Slot(unsafe { *self.prev.get_unchecked(node.0.index()) })
    }

    /// Makes `to` the successor of `from` and `from` the predecessor of `to`.
    #[inline]
    fn link(&mut self, from: Slot, to: Slot) {
        debug_assert!(from.0.index() < self.next.len() && to.0.index() < self.prev.len());
        // SAFETY: both slots lie in the arena, whose two columns have its length (`Slot`, `RingArena`).
        unsafe {
            *self.next.get_unchecked_mut(from.0.index()) = to.0;
            *self.prev.get_unchecked_mut(to.0.index()) = from.0;
        }
    }

    /// Exchanges the successor and the predecessor of `node`.
    #[inline]
    fn flip(&mut self, node: Slot) {
        debug_assert!(node.0.index() < self.next.len() && node.0.index() < self.prev.len());
        // SAFETY: the slot lies in the arena, whose two columns have its length (`Slot`, `RingArena`).
        unsafe {
            std::mem::swap(
                self.next.get_unchecked_mut(node.0.index()),
                self.prev.get_unchecked_mut(node.0.index()),
            );
        }
    }

    /// Whether `last` is reached from `first` along the successors within one lap of `first`'s ring.
    fn follows(&self, first: Slot, last: Slot) -> bool {
        let mut current = first;
        // A ring holds at most every node of the arena, which bounds the walk even if the links do not close.
        for _ in 0..self.len() {
            if current == last {
                return true;
            }
            current = self.next_slot(current);
        }
        false
    }

    /// Whether `node` is one of `first ..= last`, a segment along the successors.
    fn segment_holds(&self, first: Slot, last: Slot, node: Slot) -> bool {
        let mut current = first;
        // A segment holds at most every node of the arena, which bounds the walk even if the links do not close.
        for _ in 0..self.len() {
            if current == node {
                return true;
            }
            if current == last {
                return false;
            }
            current = self.next_slot(current);
        }
        false
    }

    /// Exchanges the places of two nodes.
    fn swap(&mut self, first: Slot, second: Slot) {
        if first == second {
            return;
        }
        let (first_prev, first_next) = (self.prev_slot(first), self.next_slot(first));
        let (second_prev, second_next) = (self.prev_slot(second), self.next_slot(second));
        let first_alone = first_next == first;
        let second_alone = second_next == second;

        if (first_alone && second_alone) || (first_next == second && second_next == first) {
            // Two rings of one, or one ring of the two: the cyclic order does not change.
        } else if first_alone {
            self.link(second_prev, first);
            self.link(first, second_next);
            self.link(second, second);
        } else if second_alone {
            self.link(first_prev, second);
            self.link(second, first_next);
            self.link(first, first);
        } else if first_next == second {
            self.link(first_prev, second);
            self.link(second, first);
            self.link(first, second_next);
        } else if second_next == first {
            self.link(second_prev, first);
            self.link(first, second);
            self.link(second, first_next);
        } else {
            self.link(first_prev, second);
            self.link(second, first_next);
            self.link(second_prev, first);
            self.link(first, second_next);
        }
    }

    /// Exchanges the places of the segments `a_first ..= a_last` and `b_first ..= b_last`.
    fn swap_ranges(&mut self, a_first: Slot, a_last: Slot, b_first: Slot, b_last: Slot) {
        debug_assert!(
            self.follows(a_first, a_last) && self.follows(b_first, b_last),
            "called `RingArena::swap_ranges` with a last node that does not follow its first"
        );
        debug_assert!(
            a_first == b_first
                || !(self.segment_holds(a_first, a_last, b_first)
                    || self.segment_holds(b_first, b_last, a_first)),
            "called `RingArena::swap_ranges` with overlapping segments"
        );
        if a_first == b_first {
            return;
        }
        let (a_prev, a_next) = (self.prev_slot(a_first), self.next_slot(a_last));
        let (b_prev, b_next) = (self.prev_slot(b_first), self.next_slot(b_last));

        let a_is_ring = a_next == a_first;
        let b_is_ring = b_next == b_first;

        if (a_is_ring && b_is_ring) || (a_next == b_first && b_next == a_first) {
            // Two whole rings, or two segments that make up one ring: the exchange leaves the same rings.
        } else if a_is_ring {
            // A takes B's place in B's ring, and B closes into a ring of its own.
            self.link(b_prev, a_first);
            self.link(a_last, b_next);
            self.link(b_last, b_first);
        } else if b_is_ring {
            self.link(a_prev, b_first);
            self.link(b_last, a_next);
            self.link(a_last, a_first);
        } else if a_next == b_first {
            self.link(a_prev, b_first);
            self.link(b_last, a_first);
            self.link(a_last, b_next);
        } else if b_next == a_first {
            self.link(b_prev, a_first);
            self.link(a_last, b_first);
            self.link(b_last, a_next);
        } else {
            self.link(a_prev, b_first);
            self.link(b_last, a_next);
            self.link(b_prev, a_first);
            self.link(a_last, b_next);
        }
    }

    /// Moves `node` to directly after `anchor`.
    fn move_after(&mut self, node: Slot, anchor: Slot) {
        if node == anchor || self.prev_slot(node) == anchor {
            return;
        }
        let (prev, next) = (self.prev_slot(node), self.next_slot(node));
        self.link(prev, next);

        let anchor_next = self.next_slot(anchor);
        self.link(anchor, node);
        self.link(node, anchor_next);
    }

    /// Moves the segment `first ..= last` to directly after `anchor`.
    fn move_range_after(&mut self, first: Slot, last: Slot, anchor: Slot) {
        debug_assert!(
            self.follows(first, last),
            "called `RingArena::move_range_after` with a last node that does not follow the first"
        );
        debug_assert!(
            !self.segment_holds(first, last, anchor),
            "called `RingArena::move_range_after` with an anchor inside the segment"
        );
        let prev = self.prev_slot(first);
        if prev == anchor {
            return;
        }
        let next = self.next_slot(last);
        self.link(prev, next);

        let anchor_next = self.next_slot(anchor);
        self.link(anchor, first);
        self.link(last, anchor_next);
    }

    /// Moves the segment `first ..= last` to directly before `anchor`.
    fn move_range_before(&mut self, first: Slot, last: Slot, anchor: Slot) {
        // A segment that already precedes the anchor stays; otherwise the anchor's predecessor lies outside the
        // segment, since only `last` has a successor outside it, and the move after that predecessor is the move.
        if self.next_slot(last) == anchor {
            return;
        }
        let anchor_prev = self.prev_slot(anchor);
        self.move_range_after(first, last, anchor_prev);
    }

    /// Reverses the order of the segment `first ..= last`.
    fn reverse(&mut self, first: Slot, last: Slot) {
        debug_assert!(
            self.follows(first, last),
            "called `RingArena::reverse` with a last node that does not follow the first"
        );
        if first == last {
            return;
        }
        let (prev, next) = (self.prev_slot(first), self.next_slot(last));

        // A segment holds at most every node of the arena, which bounds the walk even if `last` does not follow.
        let mut current = first;
        for _ in 0..self.len() {
            let following = self.next_slot(current);
            self.flip(current);
            if current == last {
                break;
            }
            current = following;
        }

        // A segment that is a whole ring is reversed by the flips alone; it has no outside to relink.
        if prev != last {
            self.link(first, next);
            self.link(prev, last);
        }
    }
}

/// The nodes of a ring from a start along the successors, up to but not including a stop node.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct RingSequenceIter<'a> {
    /// The successor of each node of the arena.
    links: &'a [Node],
    /// The node to yield next; a node of the arena.
    current: Node,
    /// The node at which the walk ends, not yielded.
    stop: Node,
    /// How many more nodes may be yielded at most; bounds the walk to one lap of the arena.
    remaining: usize,
}

impl<'a> RingSequenceIter<'a> {
    fn new(links: &'a [Node], start: Slot, stop: Slot) -> Self {
        Self {
            links,
            current: start.0,
            stop: stop.0,
            remaining: links.len(),
        }
    }
}

impl Iterator for RingSequenceIter<'_> {
    type Item = Node;

    #[inline]
    fn next(&mut self) -> Option<Node> {
        if self.current == self.stop || self.remaining == 0 {
            return None;
        }
        let node = self.current;
        debug_assert!(node.index() < self.links.len());
        // SAFETY: `current` is the start, which the arena checked or was promised to lie in it, or a link read from
        // the arena's column, which names one of its nodes (`RingArena`).
        self.current = unsafe { *self.links.get_unchecked(node.index()) };
        self.remaining -= 1;
        Some(node)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let upper = if self.current == self.stop {
            0
        } else {
            self.remaining
        };
        (0, Some(upper))
    }
}

impl FusedIterator for RingSequenceIter<'_> {}

/// The nodes of a ring from a start along the predecessors, up to but not including a stop node.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct RingSequenceRevIter<'a> {
    /// The predecessor of each node of the arena.
    links: &'a [Node],
    /// The node to yield next; a node of the arena.
    current: Node,
    /// The node at which the walk ends, not yielded.
    stop: Node,
    /// How many more nodes may be yielded at most; bounds the walk to one lap of the arena.
    remaining: usize,
}

impl<'a> RingSequenceRevIter<'a> {
    fn new(links: &'a [Node], start: Slot, stop: Slot) -> Self {
        Self {
            links,
            current: start.0,
            stop: stop.0,
            remaining: links.len(),
        }
    }
}

impl Iterator for RingSequenceRevIter<'_> {
    type Item = Node;

    #[inline]
    fn next(&mut self) -> Option<Node> {
        if self.current == self.stop || self.remaining == 0 {
            return None;
        }
        let node = self.current;
        debug_assert!(node.index() < self.links.len());
        // SAFETY: `current` is the start, which the arena checked or was promised to lie in it, or a link read from
        // the arena's column, which names one of its nodes (`RingArena`).
        self.current = unsafe { *self.links.get_unchecked(node.index()) };
        self.remaining -= 1;
        Some(node)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let upper = if self.current == self.stop {
            0
        } else {
            self.remaining
        };
        (0, Some(upper))
    }
}

impl FusedIterator for RingSequenceRevIter<'_> {}

/// The links `(node, successor)` of a ring from a start along the successors, up to the link into a stop node.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct RingEdgeIter<'a> {
    /// The successor of each node of the arena.
    links: &'a [Node],
    /// The node whose link to its successor is yielded next; a node of the arena.
    current: Node,
    /// The node at which the walk ends; no link into it is yielded.
    stop: Node,
    /// How many more links may be yielded at most; bounds the walk to one lap of the arena.
    remaining: usize,
}

impl<'a> RingEdgeIter<'a> {
    fn new(links: &'a [Node], start: Slot, stop: Slot) -> Self {
        Self {
            links,
            current: start.0,
            stop: stop.0,
            remaining: links.len(),
        }
    }
}

impl Iterator for RingEdgeIter<'_> {
    type Item = (Node, Node);

    #[inline]
    fn next(&mut self) -> Option<(Node, Node)> {
        if self.current == self.stop || self.remaining == 0 {
            return None;
        }
        let from = self.current;
        debug_assert!(from.index() < self.links.len());
        // SAFETY: `current` is the start, which the arena checked or was promised to lie in it, or a link read from
        // the arena's column, which names one of its nodes (`RingArena`).
        let to = unsafe { *self.links.get_unchecked(from.index()) };
        self.current = to;
        if to == self.stop {
            return None;
        }
        self.remaining -= 1;
        Some((from, to))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let upper = if self.current == self.stop {
            0
        } else {
            self.remaining
        };
        (0, Some(upper))
    }
}

impl FusedIterator for RingEdgeIter<'_> {}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// A robust test fixture containing three distinct, closed rings:
    /// - Ring 0: [0 <-> 1 <-> 2 <-> 3] (4 nodes)
    /// - Ring 1: [4 <-> 5 <-> 6]       (3 nodes)
    /// - Ring 2: [7]                   (1 node, points to itself)
    fn complex_fixture() -> RingArena {
        RingArena::new(
            // prev pointers
            [
                3, 0, 1, 2, // Ring 0
                6, 4, 5, // Ring 1
                7, // Ring 2
            ]
            .map(Node::new)
            .to_vec(),
            // next pointers
            [
                1, 2, 3, 0, // Ring 0
                5, 6, 4, // Ring 1
                7, // Ring 2
            ]
            .map(Node::new)
            .to_vec(),
        )
    }

    /// Helper to verify the foundational mathematical invariant of the arena:
    /// For every node `i`, `prev[next[i]] == i` and `next[prev[i]] == i`.
    fn verify_integrity(arena: &RingArena) {
        for i in 0..arena.len() {
            let node = Node::new(i);
            let next_node = arena.next(node);
            let prev_node = arena.prev(node);
            assert_eq!(
                arena.prev(next_node),
                node,
                "Integrity failure at node {}: prev of next is not self",
                i
            );
            assert_eq!(
                arena.next(prev_node),
                node,
                "Integrity failure at node {}: next of prev is not self",
                i
            );
        }
    }

    /// Helper to extract a full ring into a Vec for easy assertions.
    /// Walks the `next` pointers until it cycles back to `start`.
    fn extract_ring(arena: &RingArena, start: usize) -> Vec<Node> {
        let start_node = Node::new(start);
        let mut result = vec![start_node];
        let mut current = arena.next(start_node);
        while current != start_node {
            result.push(current);
            current = arena.next(current);
        }
        result
    }

    #[test]
    fn test_initialization_and_integrity() {
        let arena = complex_fixture();
        assert_eq!(arena.len(), 8);
        assert!(!arena.is_empty());
        verify_integrity(&arena);

        assert_eq!(extract_ring(&arena, 0), [0, 1, 2, 3].map(Node::new));
        assert_eq!(extract_ring(&arena, 4), [4, 5, 6].map(Node::new));
        assert_eq!(extract_ring(&arena, 7), [7].map(Node::new));
    }

    // ----------------------------------------------------------------
    // Swap Nodes
    // ----------------------------------------------------------------

    #[test]
    fn test_swap_nodes_adjacent_same_ring() {
        let mut arena = complex_fixture();
        arena.swap_nodes(Node::new(1), Node::new(2));
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), [0, 2, 1, 3].map(Node::new));
    }

    #[test]
    fn test_swap_nodes_non_adjacent_same_ring() {
        let mut arena = complex_fixture();
        arena.swap_nodes(Node::new(0), Node::new(2));
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 2), [2, 1, 0, 3].map(Node::new));
    }

    #[test]
    fn test_swap_nodes_different_rings() {
        let mut arena = complex_fixture();
        arena.swap_nodes(Node::new(1), Node::new(5));
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), [0, 5, 2, 3].map(Node::new));
        assert_eq!(extract_ring(&arena, 4), [4, 1, 6].map(Node::new));
    }

    #[test]
    fn test_swap_nodes_with_single_node_ring() {
        let mut arena = complex_fixture();
        arena.swap_nodes(Node::new(2), Node::new(7));
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), [0, 1, 7, 3].map(Node::new));
        assert_eq!(extract_ring(&arena, 2), [2].map(Node::new));
    }

    #[test]
    fn test_swap_nodes_self_noop() {
        let mut arena = complex_fixture();
        arena.swap_nodes(Node::new(1), Node::new(1));
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), [0, 1, 2, 3].map(Node::new));
    }

    // ----------------------------------------------------------------
    // Swap Segments
    // ----------------------------------------------------------------

    #[test]
    fn test_swap_segments_different_rings() {
        let mut arena = complex_fixture();
        // Swap [1, 2] from Ring 0 with [5, 6] from Ring 1
        arena.swap_segments(Node::new(1), Node::new(2), Node::new(5), Node::new(6));
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), [0, 5, 6, 3].map(Node::new));
        assert_eq!(extract_ring(&arena, 4), [4, 1, 2].map(Node::new));
    }

    #[test]
    fn test_swap_segments_different_lengths() {
        let mut arena = complex_fixture();
        // Swap [1, 2, 3] from Ring 0 with [5] from Ring 1
        arena.swap_segments(Node::new(1), Node::new(3), Node::new(5), Node::new(5));
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), [0, 5].map(Node::new));
        assert_eq!(extract_ring(&arena, 4), [4, 1, 2, 3, 6].map(Node::new));
    }

    #[test]
    fn test_swap_segments_same_ring_adjacent() {
        let mut arena = complex_fixture();
        // Swap [0, 1] with [2, 3]
        arena.swap_segments(Node::new(0), Node::new(1), Node::new(2), Node::new(3));
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 2), [2, 3, 0, 1].map(Node::new));
    }

    // ----------------------------------------------------------------
    // Relocate Single Nodes
    // ----------------------------------------------------------------

    #[test]
    fn test_relocate_after_same_ring() {
        let mut arena = complex_fixture();
        // Move 1 to be after 3
        arena.relocate_after(Node::new(1), Node::new(3));
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), [0, 2, 3, 1].map(Node::new));
    }

    #[test]
    fn test_relocate_after_different_ring() {
        let mut arena = complex_fixture();
        // Move 1 from Ring 0 to after 5 in Ring 1
        arena.relocate_after(Node::new(1), Node::new(5));
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), [0, 2, 3].map(Node::new));
        assert_eq!(extract_ring(&arena, 4), [4, 5, 1, 6].map(Node::new));
    }

    #[test]
    fn test_relocate_after_noop() {
        let mut arena = complex_fixture();
        arena.relocate_after(Node::new(1), Node::new(0)); // 1 is already after 0
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), [0, 1, 2, 3].map(Node::new));
    }

    #[test]
    fn test_relocate_before_different_ring() {
        let mut arena = complex_fixture();
        // Move 1 from Ring 0 to before 5 in Ring 1 (which means after 4)
        arena.relocate_before(Node::new(1), Node::new(5));
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), [0, 2, 3].map(Node::new));
        assert_eq!(extract_ring(&arena, 4), [4, 1, 5, 6].map(Node::new));
    }

    // ----------------------------------------------------------------
    // Relocate Segments
    // ----------------------------------------------------------------

    #[test]
    fn test_relocate_segment_after_different_ring() {
        let mut arena = complex_fixture();
        // Move [1, 2] from Ring 0 to after 5 in Ring 1
        arena.relocate_segment_after(Node::new(1), Node::new(2), Node::new(5));
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), [0, 3].map(Node::new));
        assert_eq!(extract_ring(&arena, 4), [4, 5, 1, 2, 6].map(Node::new));
    }

    #[test]
    fn test_relocate_segment_before_different_ring() {
        let mut arena = complex_fixture();
        // Move [1, 2] from Ring 0 to before 5 in Ring 1 (after 4)
        arena.relocate_segment_before(Node::new(1), Node::new(2), Node::new(5));
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), [0, 3].map(Node::new));
        assert_eq!(extract_ring(&arena, 4), [4, 1, 2, 5, 6].map(Node::new));
    }

    #[test]
    fn test_relocate_segment_into_single_node_ring() {
        let mut arena = complex_fixture();
        // Move [1, 2] into the single-node ring (after 7)
        arena.relocate_segment_after(Node::new(1), Node::new(2), Node::new(7));
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), [0, 3].map(Node::new));
        assert_eq!(extract_ring(&arena, 7), [7, 1, 2].map(Node::new));
    }

    // ----------------------------------------------------------------
    // Reverse Segments
    // ----------------------------------------------------------------

    #[test]
    fn test_reverse_segment_partial_ring() {
        let mut arena = complex_fixture();
        // Reverse [1, 2] in Ring 0
        arena.reverse_segment(Node::new(1), Node::new(2));
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), [0, 2, 1, 3].map(Node::new));
    }

    #[test]
    fn test_reverse_segment_full_data_with_sentinel() {
        // Actually let's just build it correctly:
        // 4 -> 0 -> 1 -> 2 -> 3 -> 4
        // next: [1, 2, 3, 4, 0]
        // prev: [4, 0, 1, 2, 3]
        let mut arena = RingArena::new(
            [4, 0, 1, 2, 3].map(Node::new).to_vec(),
            [1, 2, 3, 4, 0].map(Node::new).to_vec(),
        );

        // Reverse all data nodes [0, 1, 2, 3], leaving sentinel 4 in place.
        arena.reverse_segment(Node::new(0), Node::new(3));
        verify_integrity(&arena);
        // Ring should now be: 4 -> 3 -> 2 -> 1 -> 0 -> 4
        assert_eq!(extract_ring(&arena, 4), [4, 3, 2, 1, 0].map(Node::new));
    }

    #[test]
    fn test_reverse_single_node_noop() {
        let mut arena = complex_fixture();
        arena.reverse_segment(Node::new(1), Node::new(1));
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), [0, 1, 2, 3].map(Node::new));
    }

    // ----------------------------------------------------------------
    // Iterators
    // ----------------------------------------------------------------

    #[test]
    fn test_sequence_iter() {
        let arena = complex_fixture();
        // Start at 1, stop at 0 (so it should yield 1, 2, 3)
        // SAFETY: every node handed over lies in the arena under test.
        let seq: Vec<Node> = unsafe {
            arena
                .sequence_iter_unchecked(Node::new(1), Node::new(0))
                .collect()
        };
        assert_eq!(seq, [1, 2, 3].map(Node::new).to_vec());

        // Start at 4, stop at 4 (empty)
        // SAFETY: every node handed over lies in the arena under test.
        let seq_empty: Vec<Node> = unsafe {
            arena
                .sequence_iter_unchecked(Node::new(4), Node::new(4))
                .collect()
        };
        assert!(seq_empty.is_empty());
    }

    #[test]
    fn test_sequence_rev_iter() {
        let arena = complex_fixture();
        // Start at 3, reverse stop at 0 (should yield 3, 2, 1)
        // SAFETY: every node handed over lies in the arena under test.
        let seq: Vec<Node> = unsafe {
            arena
                .sequence_rev_iter_unchecked(Node::new(3), Node::new(0))
                .collect()
        };
        assert_eq!(seq, [3, 2, 1].map(Node::new).to_vec());
    }

    #[test]
    fn test_edge_iter() {
        let arena = complex_fixture();
        // Edges starting at 0, stopping at 3 (should yield (0,1), (1,2))
        // SAFETY: every node handed over lies in the arena under test.
        let edges: Vec<(Node, Node)> = unsafe {
            arena
                .edge_iter_unchecked(Node::new(0), Node::new(3))
                .collect()
        };
        assert_eq!(
            edges,
            vec![(Node::new(0), Node::new(1)), (Node::new(1), Node::new(2))]
        );
    }

    #[test]
    fn test_overwrite() {
        let mut arena1 = complex_fixture();
        let arena2 = RingArena::new(
            [1, 0].map(Node::new).to_vec(),
            [1, 0].map(Node::new).to_vec(),
        ); // Two node ring

        arena1.overwrite_from_arena(&arena2);
        assert_eq!(arena1.len(), 2);
        assert_eq!(extract_ring(&arena1, 0), [0, 1].map(Node::new));

        let binding_prev = [Node::new(0)];
        let binding_next = [Node::new(0)];
        arena1.overwrite_from_slices(&binding_prev, &binding_next); // Single node ring
        assert_eq!(arena1.len(), 1);
        assert_eq!(extract_ring(&arena1, 0), [0].map(Node::new));
    }

    // ----------------------------------------------------------------
    // Construction, Accessors, and Resizing
    // ----------------------------------------------------------------

    #[test]
    fn test_empty_arena() {
        let arena = RingArena::new(vec![], vec![]);
        assert_eq!(arena.len(), 0);
        assert!(arena.is_empty());
    }

    #[test]
    #[should_panic(expected = "called `RingArena::new` with 1 predecessors and 0 successors")]
    fn test_new_mismatched_lengths_panics() {
        RingArena::new(vec![Node::new(0)], vec![]);
    }

    #[test]
    fn test_reset_to_self_loops() {
        let mut arena = complex_fixture();
        arena.reset_to_self_loops(5);
        assert_eq!(arena.len(), 5);
        verify_integrity(&arena);
        for i in 0..5 {
            assert_eq!(extract_ring(&arena, i), [Node::new(i)]);
        }
    }

    #[test]
    fn test_reset_to_self_loops_zero() {
        let mut arena = complex_fixture();
        arena.reset_to_self_loops(0);
        assert_eq!(arena.len(), 0);
        assert!(arena.is_empty());
    }

    #[test]
    fn test_prev_next_accessors() {
        let arena = complex_fixture();
        assert_eq!(arena.next(Node::new(0)), Node::new(1));
        assert_eq!(arena.prev(Node::new(1)), Node::new(0));
        assert_eq!(arena.next(Node::new(3)), Node::new(0));
        assert_eq!(arena.prev(Node::new(0)), Node::new(3));
        // Self-loop
        assert_eq!(arena.next(Node::new(7)), Node::new(7));
        assert_eq!(arena.prev(Node::new(7)), Node::new(7));
    }

    #[test]
    fn test_prev_next_unchecked() {
        let arena = complex_fixture();
        // SAFETY: every node handed over lies in the arena under test.
        unsafe {
            assert_eq!(arena.next_unchecked(Node::new(0)), Node::new(1));
            assert_eq!(arena.prev_unchecked(Node::new(1)), Node::new(0));
            assert_eq!(arena.next_unchecked(Node::new(7)), Node::new(7));
            assert_eq!(arena.prev_unchecked(Node::new(7)), Node::new(7));
        }
    }

    // ----------------------------------------------------------------
    // Swap Nodes (unchecked + extra edge cases)
    // ----------------------------------------------------------------

    #[test]
    fn test_swap_nodes_both_self_loops() {
        // Two distinct self-loop rings — swapping is a no-op.
        let mut arena = RingArena::new(
            [0, 1].map(Node::new).to_vec(),
            [0, 1].map(Node::new).to_vec(),
        );
        arena.swap_nodes(Node::new(0), Node::new(1));
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), [0].map(Node::new));
        assert_eq!(extract_ring(&arena, 1), [1].map(Node::new));
    }

    #[test]
    fn test_swap_nodes_end_nodes_same_ring() {
        // Swap the first and last node in Ring 0: [0,1,2,3] -> [3,1,2,0]
        let mut arena = complex_fixture();
        arena.swap_nodes(Node::new(0), Node::new(3));
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 3), [3, 1, 2, 0].map(Node::new));
    }

    #[test]
    fn test_swap_nodes_unchecked_adjacent() {
        let mut arena = complex_fixture();
        // SAFETY: every node handed over lies in the arena under test.
        unsafe { arena.swap_nodes_unchecked(Node::new(1), Node::new(2)) };
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), [0, 2, 1, 3].map(Node::new));
    }

    #[test]
    fn test_swap_nodes_unchecked_non_adjacent() {
        let mut arena = complex_fixture();
        // SAFETY: every node handed over lies in the arena under test.
        unsafe { arena.swap_nodes_unchecked(Node::new(0), Node::new(2)) };
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 2), [2, 1, 0, 3].map(Node::new));
    }

    #[test]
    fn test_swap_nodes_unchecked_different_rings() {
        let mut arena = complex_fixture();
        // SAFETY: every node handed over lies in the arena under test.
        unsafe { arena.swap_nodes_unchecked(Node::new(1), Node::new(5)) };
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), [0, 5, 2, 3].map(Node::new));
        assert_eq!(extract_ring(&arena, 4), [4, 1, 6].map(Node::new));
    }

    #[test]
    fn test_swap_nodes_unchecked_with_self_loop() {
        let mut arena = complex_fixture();
        // SAFETY: every node handed over lies in the arena under test.
        unsafe { arena.swap_nodes_unchecked(Node::new(2), Node::new(7)) };
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), [0, 1, 7, 3].map(Node::new));
        assert_eq!(extract_ring(&arena, 2), [2].map(Node::new));
    }

    #[test]
    fn test_swap_nodes_unchecked_self_noop() {
        let mut arena = complex_fixture();
        // SAFETY: every node handed over lies in the arena under test.
        unsafe { arena.swap_nodes_unchecked(Node::new(3), Node::new(3)) };
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), [0, 1, 2, 3].map(Node::new));
    }

    #[test]
    fn test_swap_nodes_unchecked_reverse_adjacent() {
        // Test the second_next == first branch.
        let mut arena = complex_fixture();
        // SAFETY: every node handed over lies in the arena under test.
        unsafe { arena.swap_nodes_unchecked(Node::new(2), Node::new(1)) };
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), [0, 2, 1, 3].map(Node::new));
    }

    // ----------------------------------------------------------------
    // Swap Segments (unchecked + extra edge cases)
    // ----------------------------------------------------------------

    #[test]
    fn test_swap_segments_self_noop() {
        let mut arena = complex_fixture();
        arena.swap_segments(Node::new(1), Node::new(2), Node::new(1), Node::new(2));
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), [0, 1, 2, 3].map(Node::new));
    }

    #[test]
    fn test_swap_segments_single_nodes() {
        let mut arena = complex_fixture();
        arena.swap_segments(Node::new(1), Node::new(1), Node::new(5), Node::new(5));
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), [0, 5, 2, 3].map(Node::new));
        assert_eq!(extract_ring(&arena, 4), [4, 1, 6].map(Node::new));
    }

    #[test]
    fn test_swap_segments_full_ring() {
        // Swap the two halves that together form the entire ring.
        // Ring: 0 <-> 1 <-> 2 <-> 3; swap [0,1] with [2,3].
        let mut arena = complex_fixture();
        arena.swap_segments(Node::new(0), Node::new(1), Node::new(2), Node::new(3));
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 2), [2, 3, 0, 1].map(Node::new));
    }

    #[test]
    fn test_swap_segments_reverse_adjacent() {
        // Adjacent: B immediately before A.
        let mut arena = complex_fixture();
        arena.swap_segments(Node::new(2), Node::new(3), Node::new(0), Node::new(1));
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), [0, 1, 2, 3].map(Node::new));
    }

    #[test]
    fn test_swap_segments_unchecked_different_rings() {
        let mut arena = complex_fixture();
        // SAFETY: every node handed over lies in the arena under test.
        unsafe {
            arena.swap_segments_unchecked(Node::new(1), Node::new(2), Node::new(5), Node::new(6));
        }
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), [0, 5, 6, 3].map(Node::new));
        assert_eq!(extract_ring(&arena, 4), [4, 1, 2].map(Node::new));
    }

    #[test]
    fn test_swap_segments_unchecked_adjacent() {
        let mut arena = complex_fixture();
        // SAFETY: every node handed over lies in the arena under test.
        unsafe {
            arena.swap_segments_unchecked(Node::new(0), Node::new(1), Node::new(2), Node::new(3));
        }
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 2), [2, 3, 0, 1].map(Node::new));
    }

    #[test]
    fn test_swap_segments_unchecked_reverse_adjacent() {
        let mut arena = complex_fixture();
        // SAFETY: every node handed over lies in the arena under test.
        unsafe {
            arena.swap_segments_unchecked(Node::new(2), Node::new(3), Node::new(0), Node::new(1));
        }
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), [0, 1, 2, 3].map(Node::new));
    }

    #[test]
    fn test_swap_segments_unchecked_self_noop() {
        let mut arena = complex_fixture();
        // SAFETY: every node handed over lies in the arena under test.
        unsafe {
            arena.swap_segments_unchecked(Node::new(1), Node::new(2), Node::new(1), Node::new(2));
        }
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), [0, 1, 2, 3].map(Node::new));
    }

    // ----------------------------------------------------------------
    // Relocate (unchecked + extra edge cases)
    // ----------------------------------------------------------------

    #[test]
    fn test_relocate_after_self_noop() {
        let mut arena = complex_fixture();
        arena.relocate_after(Node::new(1), Node::new(1));
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), [0, 1, 2, 3].map(Node::new));
    }

    #[test]
    fn test_relocate_before_same_ring() {
        let mut arena = complex_fixture();
        // Move 3 to before 1 (after 0)
        arena.relocate_before(Node::new(3), Node::new(1));
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), [0, 3, 1, 2].map(Node::new));
    }

    #[test]
    fn test_relocate_before_noop() {
        let mut arena = complex_fixture();
        // Node 1 is already before 2, moving 1 before 2 is a no-op.
        arena.relocate_before(Node::new(1), Node::new(2));
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), [0, 1, 2, 3].map(Node::new));
    }

    #[test]
    fn test_relocate_after_into_self_loop() {
        let mut arena = complex_fixture();
        // Move node 2 after the self-loop node 7.
        arena.relocate_after(Node::new(2), Node::new(7));
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), [0, 1, 3].map(Node::new));
        assert_eq!(extract_ring(&arena, 7), [7, 2].map(Node::new));
    }

    #[test]
    fn test_relocate_before_into_self_loop() {
        let mut arena = complex_fixture();
        // Move node 2 before the self-loop node 7 (which is after 7 since it's a self-loop).
        arena.relocate_before(Node::new(2), Node::new(7));
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), [0, 1, 3].map(Node::new));
        assert_eq!(extract_ring(&arena, 7), [7, 2].map(Node::new));
    }

    #[test]
    fn test_relocate_after_unchecked_same_ring() {
        let mut arena = complex_fixture();
        // SAFETY: every node handed over lies in the arena under test.
        unsafe { arena.relocate_after_unchecked(Node::new(1), Node::new(3)) };
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), [0, 2, 3, 1].map(Node::new));
    }

    #[test]
    fn test_relocate_after_unchecked_different_ring() {
        let mut arena = complex_fixture();
        // SAFETY: every node handed over lies in the arena under test.
        unsafe { arena.relocate_after_unchecked(Node::new(1), Node::new(5)) };
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), [0, 2, 3].map(Node::new));
        assert_eq!(extract_ring(&arena, 4), [4, 5, 1, 6].map(Node::new));
    }

    #[test]
    fn test_relocate_after_unchecked_noop() {
        let mut arena = complex_fixture();
        // SAFETY: every node handed over lies in the arena under test.
        unsafe { arena.relocate_after_unchecked(Node::new(1), Node::new(0)) };
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), [0, 1, 2, 3].map(Node::new));
    }

    #[test]
    fn test_relocate_after_unchecked_self_noop() {
        let mut arena = complex_fixture();
        // SAFETY: every node handed over lies in the arena under test.
        unsafe { arena.relocate_after_unchecked(Node::new(2), Node::new(2)) };
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), [0, 1, 2, 3].map(Node::new));
    }

    #[test]
    fn test_relocate_before_unchecked_different_ring() {
        let mut arena = complex_fixture();
        // SAFETY: every node handed over lies in the arena under test.
        unsafe { arena.relocate_before_unchecked(Node::new(1), Node::new(5)) };
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), [0, 2, 3].map(Node::new));
        assert_eq!(extract_ring(&arena, 4), [4, 1, 5, 6].map(Node::new));
    }

    #[test]
    fn test_relocate_segment_after_same_ring() {
        let mut arena = complex_fixture();
        // Move [1,2] to after 3 in Ring 0.
        arena.relocate_segment_after(Node::new(1), Node::new(2), Node::new(3));
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), [0, 3, 1, 2].map(Node::new));
    }

    #[test]
    fn test_relocate_segment_after_noop() {
        let mut arena = complex_fixture();
        // Segment [1,2] is already after 0, so using anchor 0 is a no-op.
        arena.relocate_segment_after(Node::new(1), Node::new(2), Node::new(0));
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), [0, 1, 2, 3].map(Node::new));
    }

    #[test]
    fn test_relocate_segment_before_same_ring() {
        let mut arena = complex_fixture();
        // Move [2,3] to before 1 (after 0).
        arena.relocate_segment_before(Node::new(2), Node::new(3), Node::new(1));
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), [0, 2, 3, 1].map(Node::new));
    }

    #[test]
    fn test_relocate_segment_before_the_node_it_already_precedes_changes_nothing() {
        // One ring 0 -> 1 -> 2 -> 3 -> 4; the segment [1, 2] already stands directly before 3.
        let mut arena = RingArena::new(
            [4, 0, 1, 2, 3].map(Node::new).to_vec(),
            [1, 2, 3, 4, 0].map(Node::new).to_vec(),
        );
        arena.relocate_segment_before(Node::new(1), Node::new(2), Node::new(3));
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), [0, 1, 2, 3, 4].map(Node::new));
    }

    #[test]
    fn test_relocate_segment_after_unchecked_different_ring() {
        let mut arena = complex_fixture();
        // SAFETY: every node handed over lies in the arena under test.
        unsafe {
            arena.relocate_segment_after_unchecked(Node::new(1), Node::new(2), Node::new(5));
        }
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), [0, 3].map(Node::new));
        assert_eq!(extract_ring(&arena, 4), [4, 5, 1, 2, 6].map(Node::new));
    }

    #[test]
    fn test_relocate_segment_after_unchecked_noop() {
        let mut arena = complex_fixture();
        // SAFETY: every node handed over lies in the arena under test.
        unsafe {
            arena.relocate_segment_after_unchecked(Node::new(1), Node::new(2), Node::new(0));
        }
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), [0, 1, 2, 3].map(Node::new));
    }

    #[test]
    fn test_relocate_segment_before_unchecked_different_ring() {
        let mut arena = complex_fixture();
        // SAFETY: every node handed over lies in the arena under test.
        unsafe {
            arena.relocate_segment_before_unchecked(Node::new(1), Node::new(2), Node::new(5));
        }
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), [0, 3].map(Node::new));
        assert_eq!(extract_ring(&arena, 4), [4, 1, 2, 5, 6].map(Node::new));
    }

    #[test]
    fn test_relocate_single_node_segment() {
        let mut arena = complex_fixture();
        arena.relocate_segment_after(Node::new(2), Node::new(2), Node::new(5));
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), [0, 1, 3].map(Node::new));
        assert_eq!(extract_ring(&arena, 4), [4, 5, 2, 6].map(Node::new));
    }

    // ----------------------------------------------------------------
    // Reverse Segments (unchecked + extra edge cases)
    // ----------------------------------------------------------------

    #[test]
    fn test_reverse_segment_three_nodes() {
        let mut arena = complex_fixture();
        // Reverse [1, 2, 3] in Ring 0.
        arena.reverse_segment(Node::new(1), Node::new(3));
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), [0, 3, 2, 1].map(Node::new));
    }

    #[test]
    fn test_reverse_segment_entire_ring_via_all_nodes() {
        // Ring: 4 -> 0 -> 1 -> 2 -> 3. Reverse [0,1,2,3].
        let mut arena = RingArena::new(
            [4, 0, 1, 2, 3].map(Node::new).to_vec(),
            [1, 2, 3, 4, 0].map(Node::new).to_vec(),
        );
        arena.reverse_segment(Node::new(0), Node::new(3));
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 4), [4, 3, 2, 1, 0].map(Node::new));
    }

    #[test]
    fn test_reverse_segment_two_nodes() {
        let mut arena = complex_fixture();
        arena.reverse_segment(Node::new(2), Node::new(3));
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), [0, 1, 3, 2].map(Node::new));
    }

    #[test]
    fn test_reverse_segment_partial_three_node_ring() {
        let mut arena = complex_fixture();
        // Reverse [5,6] within Ring 1: [4,5,6] -> [4,6,5]
        arena.reverse_segment(Node::new(5), Node::new(6));
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 4), [4, 6, 5].map(Node::new));
    }

    #[test]
    fn test_reverse_segment_unchecked_partial() {
        let mut arena = complex_fixture();
        // SAFETY: every node handed over lies in the arena under test.
        unsafe { arena.reverse_segment_unchecked(Node::new(1), Node::new(2)) };
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), [0, 2, 1, 3].map(Node::new));
    }

    #[test]
    fn test_reverse_segment_unchecked_three_nodes() {
        let mut arena = complex_fixture();
        // SAFETY: every node handed over lies in the arena under test.
        unsafe { arena.reverse_segment_unchecked(Node::new(1), Node::new(3)) };
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), [0, 3, 2, 1].map(Node::new));
    }

    #[test]
    fn test_reverse_segment_unchecked_single_noop() {
        let mut arena = complex_fixture();
        // SAFETY: every node handed over lies in the arena under test.
        unsafe { arena.reverse_segment_unchecked(Node::new(5), Node::new(5)) };
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 4), [4, 5, 6].map(Node::new));
    }

    // ----------------------------------------------------------------
    // Checked Iterator Variants
    // ----------------------------------------------------------------

    #[test]
    fn test_sequence_iter_checked() {
        let arena = complex_fixture();
        let seq: Vec<Node> = arena.sequence_iter(Node::new(1), Node::new(0)).collect();
        assert_eq!(seq, [1, 2, 3].map(Node::new).to_vec());
    }

    #[test]
    fn test_sequence_iter_checked_empty() {
        let arena = complex_fixture();
        let seq: Vec<Node> = arena.sequence_iter(Node::new(4), Node::new(4)).collect();
        assert!(seq.is_empty());
    }

    #[test]
    fn test_sequence_iter_checked_full_ring() {
        let arena = complex_fixture();
        // Start at 0, stop at 0 — yields nothing (stop is exclusive and immediate).
        let seq: Vec<Node> = arena.sequence_iter(Node::new(0), Node::new(0)).collect();
        assert!(seq.is_empty());
    }

    #[test]
    fn test_sequence_iter_single_node_ring() {
        let arena = complex_fixture();
        let seq: Vec<Node> = arena.sequence_iter(Node::new(7), Node::new(7)).collect();
        assert!(seq.is_empty());
    }

    #[test]
    fn test_sequence_iter_size_hint() {
        let arena = complex_fixture();
        let iter = arena.sequence_iter(Node::new(1), Node::new(0));
        let (lo, hi) = iter.size_hint();
        assert_eq!(lo, 0);
        assert!(hi.unwrap() >= 3);
    }

    #[test]
    fn test_sequence_rev_iter_full_ring() {
        let arena = complex_fixture();
        // Reverse iterate Ring 1 starting from 6, stopping at 4 to get [6, 5].
        // SAFETY: every node handed over lies in the arena under test.
        let seq: Vec<Node> = unsafe {
            arena
                .sequence_rev_iter_unchecked(Node::new(6), Node::new(4))
                .collect()
        };
        assert_eq!(seq, [6, 5].map(Node::new).to_vec());
    }

    #[test]
    fn test_sequence_rev_iter_empty() {
        let arena = complex_fixture();
        // SAFETY: every node handed over lies in the arena under test.
        let seq: Vec<Node> = unsafe {
            arena
                .sequence_rev_iter_unchecked(Node::new(5), Node::new(5))
                .collect()
        };
        assert!(seq.is_empty());
    }

    #[test]
    fn test_edge_iter_checked() {
        let arena = complex_fixture();
        let edges: Vec<(Node, Node)> = arena.edge_iter(Node::new(0), Node::new(3)).collect();
        assert_eq!(
            edges,
            vec![(Node::new(0), Node::new(1)), (Node::new(1), Node::new(2))]
        );
    }

    #[test]
    fn test_edge_iter_checked_empty() {
        let arena = complex_fixture();
        let edges: Vec<(Node, Node)> = arena.edge_iter(Node::new(4), Node::new(4)).collect();
        assert!(edges.is_empty());
    }

    #[test]
    fn test_edge_iter_single_edge() {
        let arena = complex_fixture();
        // From node 5, stop at 4. next(5)=6, next(6)=4 (stop). Should yield (5,6).
        let edges: Vec<(Node, Node)> = arena.edge_iter(Node::new(5), Node::new(4)).collect();
        assert_eq!(edges, vec![(Node::new(5), Node::new(6))]);
    }

    #[test]
    fn test_edge_iter_immediate_stop() {
        let arena = complex_fixture();
        // From 3, stop at 0. next(3) = 0 = stop, so no edges.
        let edges: Vec<(Node, Node)> = arena.edge_iter(Node::new(3), Node::new(0)).collect();
        assert!(edges.is_empty());
    }

    #[test]
    fn test_edge_iter_size_hint() {
        let arena = complex_fixture();
        let iter = arena.edge_iter(Node::new(0), Node::new(3));
        let (lo, hi) = iter.size_hint();
        assert_eq!(lo, 0);
        assert!(hi.is_some());
    }

    // ----------------------------------------------------------------
    // Combined / Compound Operations
    // ----------------------------------------------------------------

    #[test]
    fn test_swap_then_reverse() {
        let mut arena = complex_fixture();
        arena.swap_nodes(Node::new(1), Node::new(2));
        verify_integrity(&arena);
        // After swap: Ring 0 is [0, 2, 1, 3]. Reverse middle segment [2, 1].
        arena.reverse_segment(Node::new(2), Node::new(1));
        verify_integrity(&arena);
        // After reverse [2,1] -> [1,2]: Ring 0 is [0, 1, 2, 3] — back to original.
        assert_eq!(extract_ring(&arena, 0), [0, 1, 2, 3].map(Node::new));
    }

    #[test]
    fn test_relocate_then_swap() {
        let mut arena = complex_fixture();
        arena.relocate_after(Node::new(1), Node::new(3));
        verify_integrity(&arena);
        // Ring 0 is now [0, 2, 3, 1].
        arena.swap_nodes(Node::new(0), Node::new(3));
        verify_integrity(&arena);
        // After swap: [3, 2, 0, 1].
        assert_eq!(extract_ring(&arena, 3), [3, 2, 0, 1].map(Node::new));
    }

    #[test]
    fn test_reverse_then_relocate() {
        let mut arena = complex_fixture();
        arena.reverse_segment(Node::new(1), Node::new(3));
        verify_integrity(&arena);
        // Ring 0: [0, 3, 2, 1].
        arena.relocate_after(Node::new(0), Node::new(5));
        verify_integrity(&arena);
        // Ring 0 becomes [3, 2, 1], Ring 1 becomes [4, 5, 0, 6].
        assert_eq!(extract_ring(&arena, 3), [3, 2, 1].map(Node::new));
        assert_eq!(extract_ring(&arena, 4), [4, 5, 0, 6].map(Node::new));
    }

    // ----------------------------------------------------------------
    // Clone / Equality
    // ----------------------------------------------------------------

    #[test]
    fn test_clone_equality() {
        let arena = complex_fixture();
        let cloned = arena.clone();
        assert_eq!(arena, cloned);
    }

    #[test]
    fn test_inequality_after_mutation() {
        let arena = complex_fixture();
        let mut mutated = arena.clone();
        mutated.swap_nodes(Node::new(0), Node::new(1));
        assert_ne!(arena, mutated);
    }

    #[test]
    fn test_overwrite_from_arena_equality() {
        let source = complex_fixture();
        let mut target = RingArena::new(vec![], vec![]);
        target.overwrite_from_arena(&source);
        assert_eq!(source, target);
    }

    // ----------------------------------------------------------------
    // Panics
    // ----------------------------------------------------------------

    #[test]
    #[should_panic(
        expected = "called `RingArena::overwrite_from_slices` with 1 predecessors and 0 successors"
    )]
    fn test_overwrite_from_slices_mismatched_panics() {
        let mut arena = complex_fixture();
        arena.overwrite_from_slices(&[Node::new(0)], &[]);
    }

    #[test]
    #[should_panic(
        expected = "called `RingArena::sequence_iter` with Node 99, outside an arena of 8 nodes"
    )]
    fn test_sequence_iter_out_of_bounds_panics() {
        let arena = complex_fixture();
        let _ = arena.sequence_iter(Node::new(99), Node::new(0));
    }

    #[test]
    #[should_panic(
        expected = "called `RingArena::edge_iter` with Node 99, outside an arena of 8 nodes"
    )]
    fn test_edge_iter_out_of_bounds_panics() {
        let arena = complex_fixture();
        let _ = arena.edge_iter(Node::new(99), Node::new(0));
    }

    #[test]
    #[should_panic(
        expected = "called `RingArena::new` with a link from Node 2 to Node 3, outside an arena of 3 nodes"
    )]
    fn test_new_prev_out_of_bounds_panics() {
        // prev contains index 3, but len is only 3 (valid range 0..2).
        RingArena::new(
            vec![Node::new(1), Node::new(2), Node::new(3)],
            vec![Node::new(1), Node::new(2), Node::new(0)],
        );
    }

    #[test]
    #[should_panic(
        expected = "called `RingArena::new` with a link from Node 1 to Node 5, outside an arena of 3 nodes"
    )]
    fn test_new_next_out_of_bounds_panics() {
        // next contains index 5, but len is only 3.
        RingArena::new(
            vec![Node::new(2), Node::new(0), Node::new(1)],
            vec![Node::new(1), Node::new(5), Node::new(0)],
        );
    }

    #[test]
    #[should_panic(
        expected = "called `RingArena::overwrite_from_slices` with a link from Node 1 to Node 99, outside"
    )]
    fn test_overwrite_from_slices_prev_out_of_bounds_panics() {
        let mut arena = complex_fixture();
        arena.overwrite_from_slices(
            &[Node::new(1), Node::new(99)],
            &[Node::new(1), Node::new(0)],
        );
    }

    #[test]
    #[should_panic(
        expected = "called `RingArena::overwrite_from_slices` with a link from Node 1 to Node 42, outside"
    )]
    fn test_overwrite_from_slices_next_out_of_bounds_panics() {
        let mut arena = complex_fixture();
        arena.overwrite_from_slices(
            &[Node::new(1), Node::new(0)],
            &[Node::new(1), Node::new(42)],
        );
    }

    #[test]
    fn test_swap_nodes_in_two_node_ring_keeps_the_ring() {
        let mut arena = RingArena::new(
            [1, 0].map(Node::new).to_vec(),
            [1, 0].map(Node::new).to_vec(),
        );
        arena.swap_nodes(Node::new(0), Node::new(1));
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), [0, 1].map(Node::new));
    }

    #[test]
    fn test_swap_nodes_unchecked_in_two_node_ring_keeps_the_ring() {
        let mut arena = RingArena::new(
            [1, 0].map(Node::new).to_vec(),
            [1, 0].map(Node::new).to_vec(),
        );
        // SAFETY: every node handed over lies in the arena under test.
        unsafe { arena.swap_nodes_unchecked(Node::new(1), Node::new(0)) };
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), [0, 1].map(Node::new));
    }

    #[test]
    fn test_reverse_segment_over_a_whole_ring_reverses_it() {
        let mut arena = complex_fixture();
        arena.reverse_segment(Node::new(0), Node::new(3));
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), [0, 3, 2, 1].map(Node::new));
    }

    #[test]
    fn test_reverse_segment_unchecked_over_a_whole_ring_reverses_it() {
        let mut arena = complex_fixture();
        // SAFETY: every node handed over lies in the arena under test.
        unsafe { arena.reverse_segment_unchecked(Node::new(5), Node::new(4)) };
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 4), [4, 6, 5].map(Node::new));
    }

    #[test]
    fn test_swap_segments_with_a_whole_ring_moves_it_into_the_other_place() {
        // Node 7 is a ring of its own; swapping it with segment [5] of ring [4, 5, 6] exchanges their places.
        let mut arena = complex_fixture();
        arena.swap_segments(Node::new(7), Node::new(7), Node::new(5), Node::new(5));
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 4), [4, 7, 6].map(Node::new));
        assert_eq!(extract_ring(&arena, 5), [5].map(Node::new));
    }

    #[test]
    fn test_swap_segments_unchecked_with_a_whole_ring_moves_it_into_the_other_place() {
        // Ring [4, 5, 6] as one segment swapped with segment [1, 2] of ring [0, 1, 2, 3].
        let mut arena = complex_fixture();
        // SAFETY: every node handed over lies in the arena under test.
        unsafe {
            arena.swap_segments_unchecked(Node::new(1), Node::new(2), Node::new(4), Node::new(6))
        };
        verify_integrity(&arena);
        assert_eq!(extract_ring(&arena, 0), [0, 4, 5, 6, 3].map(Node::new));
        assert_eq!(extract_ring(&arena, 1), [1, 2].map(Node::new));
    }

    // ----------------------------------------------------------------
    // Private steps
    // ----------------------------------------------------------------

    fn nodes(indices: &[usize]) -> Vec<Node> {
        indices.iter().copied().map(Node::new).collect()
    }

    #[test]
    fn test_ring_violation_accepts_rings() {
        assert_eq!(
            ring_violation(&nodes(&[3, 0, 1, 2]), &nodes(&[1, 2, 3, 0])),
            None
        );
        assert_eq!(ring_violation(&[], &[]), None);
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn test_nameable_holds_exactly_up_to_one_node_per_u32() {
        let one_per_u32 = usize::try_from(u32::MAX).unwrap() + 1;
        assert!(nameable(0));
        assert!(nameable(1));
        assert!(nameable(one_per_u32));
        assert!(!nameable(one_per_u32 + 1));
        assert!(!nameable(usize::MAX));
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn test_reset_to_self_loops_refuses_too_many_nodes_before_changing_anything() {
        let mut arena = complex_fixture();
        let too_many = usize::try_from(u32::MAX).unwrap() + 2;
        let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            arena.reset_to_self_loops(too_many);
        }));
        assert!(refused.is_err());
        assert_eq!(arena, complex_fixture());
    }

    #[test]
    fn test_ring_violation_finds_columns_of_different_lengths() {
        assert_eq!(
            ring_violation(&nodes(&[0]), &[]),
            Some(RingViolation::LengthsDiffer { prev: 1, next: 0 })
        );
    }

    #[test]
    fn test_ring_violation_finds_a_successor_outside() {
        assert_eq!(
            ring_violation(&nodes(&[1, 0]), &nodes(&[1, 2])),
            Some(RingViolation::LinkOutside {
                node: Node::new(1),
                link: Node::new(2),
                len: 2
            })
        );
    }

    #[test]
    fn test_ring_violation_finds_links_that_are_not_inverse() {
        // 0 -> 1 -> 1: node 1 is its own successor, but node 0 claims it too.
        assert_eq!(
            ring_violation(&nodes(&[1, 0]), &nodes(&[1, 1])),
            Some(RingViolation::NotInverse {
                node: Node::new(1),
                next: Node::new(1),
                prev_of_next: Node::new(0)
            })
        );
    }

    #[test]
    #[should_panic(
        expected = "called `RingArena::new` with Node 1 followed by Node 1, whose predecessor is Node 0"
    )]
    fn test_new_refuses_links_that_are_no_rings() {
        let _ = RingArena::new(nodes(&[1, 0]), nodes(&[1, 1]));
    }

    #[test]
    fn test_slots_refuses_the_first_node_outside() {
        let arena = complex_fixture();
        assert_eq!(
            arena.slots([Node::new(7), Node::new(8), Node::new(9)]),
            Err(NodeOutside {
                node: Node::new(8),
                len: 8
            })
        );
        assert_eq!(
            arena.slots([Node::new(7), Node::new(0)]),
            Ok([Slot(Node::new(7)), Slot(Node::new(0))])
        );
    }

    #[test]
    fn test_follows_walks_one_ring() {
        let arena = complex_fixture();
        let slot = |index| Slot(Node::new(index));
        assert!(arena.follows(slot(1), slot(3)));
        assert!(arena.follows(slot(3), slot(1)));
        assert!(arena.follows(slot(2), slot(2)));
        assert!(!arena.follows(slot(1), slot(5)));
    }

    #[test]
    fn test_segment_holds_only_the_nodes_between_its_ends() {
        let arena = complex_fixture();
        let slot = |index| Slot(Node::new(index));
        assert!(arena.segment_holds(slot(1), slot(3), slot(2)));
        assert!(arena.segment_holds(slot(1), slot(3), slot(1)));
        assert!(arena.segment_holds(slot(1), slot(3), slot(3)));
        assert!(!arena.segment_holds(slot(1), slot(3), slot(0)));
        assert!(arena.segment_holds(slot(3), slot(1), slot(0)));
        assert!(!arena.segment_holds(slot(1), slot(3), slot(5)));
    }

    #[test]
    fn test_default_is_empty() {
        assert!(RingArena::default().is_empty());
    }

    #[test]
    fn test_sequence_rev_iter_checked() {
        let arena = complex_fixture();
        let seq: Vec<Node> = arena
            .sequence_rev_iter(Node::new(3), Node::new(0))
            .collect();
        assert_eq!(seq, nodes(&[3, 2, 1]));
    }

    #[test]
    #[should_panic(
        expected = "called `RingArena::swap_nodes` with Node 8, outside an arena of 8 nodes"
    )]
    fn test_swap_nodes_panics_outside_the_arena() {
        complex_fixture().swap_nodes(Node::new(0), Node::new(8));
    }

    #[test]
    #[should_panic(expected = "called `RingArena::next` with Node 8, outside an arena of 8 nodes")]
    fn test_next_panics_outside_the_arena() {
        let _ = complex_fixture().next(Node::new(8));
    }

    #[test]
    #[should_panic(
        expected = "called `RingArena::reverse` with a last node that does not follow the first"
    )]
    fn test_reverse_segment_across_rings_fails_its_precondition() {
        complex_fixture().reverse_segment(Node::new(1), Node::new(5));
    }

    #[test]
    #[should_panic(
        expected = "called `RingArena::move_range_after` with an anchor inside the segment"
    )]
    fn test_relocate_segment_onto_itself_fails_its_precondition() {
        complex_fixture().relocate_segment_after(Node::new(1), Node::new(3), Node::new(2));
    }

    #[test]
    #[should_panic(expected = "called `RingArena::swap_ranges` with overlapping segments")]
    fn test_swap_overlapping_segments_fails_its_precondition() {
        complex_fixture().swap_segments(Node::new(0), Node::new(2), Node::new(1), Node::new(3));
    }

    // ----------------------------------------------------------------
    // Against a model
    // ----------------------------------------------------------------

    // The rings as lists of node indices, each in successor order: the model the arena is checked against.
    type Rings = Vec<Vec<usize>>;

    fn arena_of(rings: &Rings, len: usize) -> RingArena {
        let mut prev = vec![Node::new(0); len];
        let mut next = vec![Node::new(0); len];
        for ring in rings {
            for (position, &node) in ring.iter().enumerate() {
                let following = ring[(position + 1) % ring.len()];
                next[node] = Node::new(following);
                prev[following] = Node::new(node);
            }
        }
        RingArena::new(prev, next)
    }

    // Each ring rotated to start at its least node, the rings in order: equal for equal cyclic orders.
    fn canonical(rings: &Rings) -> Rings {
        let mut rings: Rings = rings
            .iter()
            .filter(|ring| !ring.is_empty())
            .map(|ring| {
                let least = ring
                    .iter()
                    .enumerate()
                    .min_by_key(|&(_, &node)| node)
                    .unwrap()
                    .0;
                ring[least..]
                    .iter()
                    .chain(&ring[..least])
                    .copied()
                    .collect()
            })
            .collect();
        rings.sort();
        rings
    }

    fn rings_of(arena: &RingArena) -> Rings {
        let mut seen = vec![false; arena.len()];
        let mut rings = Rings::new();
        for start in 0..arena.len() {
            if !seen[start] {
                let ring = extract_ring(arena, start);
                for node in &ring {
                    seen[node.index()] = true;
                }
                rings.push(ring.iter().map(|node| node.index()).collect());
            }
        }
        canonical(&rings)
    }

    // The ring holding `node` and the node's place in it.
    fn locate(rings: &Rings, node: usize) -> (usize, usize) {
        rings
            .iter()
            .enumerate()
            .find_map(|(ring, nodes)| {
                nodes
                    .iter()
                    .position(|&n| n == node)
                    .map(|place| (ring, place))
            })
            .unwrap()
    }

    // The ring of `first` rotated to start at it, and the segment of `extra + 1` nodes it starts.
    fn segment(rings: &Rings, first: usize, extra: usize) -> (usize, Vec<usize>, Vec<usize>) {
        let (ring, place) = locate(rings, first);
        let nodes = &rings[ring];
        let rotated: Vec<usize> = nodes[place..]
            .iter()
            .chain(&nodes[..place])
            .copied()
            .collect();
        let length = extra % rotated.len() + 1;
        (ring, rotated[..length].to_vec(), rotated[length..].to_vec())
    }

    #[derive(Clone, Debug)]
    enum Step {
        SwapNodes(usize, usize),
        RelocateAfter(usize, usize),
        RelocateBefore(usize, usize),
        Reverse(usize, usize),
        RelocateSegmentAfter(usize, usize, usize),
        RelocateSegmentBefore(usize, usize, usize),
        SwapSegments(usize, usize, usize, usize),
    }

    fn step() -> impl Strategy<Value = Step> {
        let any = || 0usize..64;
        prop_oneof![
            (any(), any()).prop_map(|(a, b)| Step::SwapNodes(a, b)),
            (any(), any()).prop_map(|(a, b)| Step::RelocateAfter(a, b)),
            (any(), any()).prop_map(|(a, b)| Step::RelocateBefore(a, b)),
            (any(), any()).prop_map(|(a, b)| Step::Reverse(a, b)),
            (any(), any(), any()).prop_map(|(a, b, c)| Step::RelocateSegmentAfter(a, b, c)),
            (any(), any(), any()).prop_map(|(a, b, c)| Step::RelocateSegmentBefore(a, b, c)),
            (any(), any(), any(), any()).prop_map(|(a, b, c, d)| Step::SwapSegments(a, b, c, d)),
        ]
    }

    fn start() -> impl Strategy<Value = (usize, Rings)> {
        (1usize..10).prop_flat_map(|len| {
            (
                Just((0..len).collect::<Vec<_>>()).prop_shuffle(),
                proptest::collection::vec(any::<bool>(), len),
            )
                .prop_map(move |(order, cuts)| {
                    let mut rings = vec![Vec::new()];
                    for (node, cut) in order.into_iter().zip(cuts) {
                        if cut && !rings.last().unwrap().is_empty() {
                            rings.push(Vec::new());
                        }
                        rings.last_mut().unwrap().push(node);
                    }
                    (len, rings)
                })
        })
    }

    // Applies `step` to the model and to both arenas, the checked steps on `checked` and the unchecked twins on
    // `unchecked`, unless the step's nodes break its preconditions, which leaves all three as they were.
    fn apply(
        step: &Step,
        len: usize,
        rings: &mut Rings,
        checked: &mut RingArena,
        unchecked: &mut RingArena,
    ) {
        let node = |index: usize| Node::new(index % len);
        match *step {
            Step::SwapNodes(a, b) => {
                let ((ring_a, place_a), (ring_b, place_b)) =
                    (locate(rings, a % len), locate(rings, b % len));
                let held = rings[ring_a][place_a];
                rings[ring_a][place_a] = rings[ring_b][place_b];
                rings[ring_b][place_b] = held;
                checked.swap_nodes(node(a), node(b));
                // SAFETY: every node handed over lies in the arena under test.
                unsafe { unchecked.swap_nodes_unchecked(node(a), node(b)) };
            }
            Step::RelocateAfter(a, b) | Step::RelocateBefore(a, b) => {
                let (moved, anchor) = (a % len, b % len);
                if moved == anchor {
                    return;
                }
                let (ring, place) = locate(rings, moved);
                rings[ring].remove(place);
                let (ring, place) = locate(rings, anchor);
                let after = matches!(step, Step::RelocateAfter(..));
                rings[ring].insert(if after { place + 1 } else { place }, moved);
                if after {
                    checked.relocate_after(node(a), node(b));
                    // SAFETY: every node handed over lies in the arena under test.
                    unsafe { unchecked.relocate_after_unchecked(node(a), node(b)) };
                } else {
                    checked.relocate_before(node(a), node(b));
                    // SAFETY: every node handed over lies in the arena under test.
                    unsafe { unchecked.relocate_before_unchecked(node(a), node(b)) };
                }
            }
            Step::Reverse(a, extra) => {
                let (ring, mut nodes, rest) = segment(rings, a % len, extra);
                let last = *nodes.last().unwrap();
                nodes.reverse();
                rings[ring] = nodes.into_iter().chain(rest).collect();
                checked.reverse_segment(node(a), Node::new(last));
                // SAFETY: every node handed over lies in the arena under test.
                unsafe { unchecked.reverse_segment_unchecked(node(a), Node::new(last)) };
            }
            Step::RelocateSegmentAfter(a, extra, c) | Step::RelocateSegmentBefore(a, extra, c) => {
                let (ring, nodes, rest) = segment(rings, a % len, extra);
                let anchor = c % len;
                if nodes.contains(&anchor) {
                    return;
                }
                let (first, last) = (Node::new(nodes[0]), Node::new(*nodes.last().unwrap()));
                rings[ring] = rest;
                let (anchor_ring, place) = locate(rings, anchor);
                let after = matches!(step, Step::RelocateSegmentAfter(..));
                let at = if after { place + 1 } else { place };
                rings[anchor_ring].splice(at..at, nodes);
                rings.retain(|ring| !ring.is_empty());
                if after {
                    checked.relocate_segment_after(first, last, node(c));
                    // SAFETY: every node handed over lies in the arena under test.
                    unsafe { unchecked.relocate_segment_after_unchecked(first, last, node(c)) };
                } else {
                    checked.relocate_segment_before(first, last, node(c));
                    // SAFETY: every node handed over lies in the arena under test.
                    unsafe { unchecked.relocate_segment_before_unchecked(first, last, node(c)) };
                }
            }
            Step::SwapSegments(a, a_extra, b, b_extra) => {
                let (ring_a, nodes_a, rest_a) = segment(rings, a % len, a_extra);
                let (ring_b, nodes_b, rest_b) = segment(rings, b % len, b_extra);
                let same = nodes_a[0] == nodes_b[0];
                if !same && (nodes_a.contains(&nodes_b[0]) || nodes_b.contains(&nodes_a[0])) {
                    return;
                }
                let (a_first, a_last) = (nodes_a[0], *nodes_a.last().unwrap());
                let (b_first, b_last) = (nodes_b[0], *nodes_b.last().unwrap());
                if same {
                } else if ring_a == ring_b {
                    // The ring is A, X, B, Y with A first; it becomes B, X, A, Y.
                    let split = rest_a.iter().position(|&n| n == b_first).unwrap();
                    let (between, after_b) = (&rest_a[..split], &rest_a[split + nodes_b.len()..]);
                    rings[ring_a] = nodes_b
                        .iter()
                        .chain(between)
                        .chain(&nodes_a)
                        .chain(after_b)
                        .copied()
                        .collect();
                } else {
                    rings[ring_a] = nodes_b.iter().chain(&rest_a).copied().collect();
                    rings[ring_b] = nodes_a.iter().chain(&rest_b).copied().collect();
                }
                let ends = [a_first, a_last, b_first, b_last].map(Node::new);
                checked.swap_segments(ends[0], ends[1], ends[2], ends[3]);
                // SAFETY: every node handed over lies in the arena under test.
                unsafe { unchecked.swap_segments_unchecked(ends[0], ends[1], ends[2], ends[3]) };
            }
        }
    }

    proptest! {
        #[test]
        fn test_steps_match_the_model_of_rings_as_lists(
            (len, rings) in start(),
            steps in proptest::collection::vec(step(), 0..24),
        ) {
            let mut rings = rings;
            let mut checked = arena_of(&rings, len);
            let mut unchecked = checked.clone();
            prop_assert_eq!(rings_of(&checked), canonical(&rings));
            for step in &steps {
                apply(step, len, &mut rings, &mut checked, &mut unchecked);
                verify_integrity(&checked);
                prop_assert_eq!(rings_of(&checked), canonical(&rings), "after {:?}", step);
                prop_assert_eq!(&unchecked, &checked, "after {:?}", step);
            }
        }

        #[test]
        fn test_sequence_iter_walks_the_model_ring(
            (len, rings) in start(),
            start_at in 0usize..64,
            stop_at in 0usize..64,
        ) {
            let arena = arena_of(&rings, len);
            let start = start_at % len;
            let (ring, place) = locate(&rings, start);
            let nodes = &rings[ring];
            let stop = nodes[stop_at % nodes.len()];
            let expected: Vec<usize> = nodes[place..]
                .iter()
                .chain(&nodes[..place])
                .copied()
                .take_while(|&n| n != stop)
                .collect();
            let walked: Vec<usize> = arena.sequence_iter(Node::new(start), Node::new(stop)).map(Node::index).collect();
            // SAFETY: every node handed over lies in the arena under test.
            let unchecked: Vec<usize> = unsafe { arena.sequence_iter_unchecked(Node::new(start), Node::new(stop)) }
                .map(Node::index)
                .collect();
            prop_assert_eq!(&walked, &expected);
            prop_assert_eq!(&unchecked, &expected);
            let edges: Vec<(usize, usize)> =
                arena.edge_iter(Node::new(start), Node::new(stop)).map(|(a, b)| (a.index(), b.index())).collect();
            prop_assert_eq!(edges, expected.windows(2).map(|pair| (pair[0], pair[1])).collect::<Vec<_>>());
        }
    }
}
