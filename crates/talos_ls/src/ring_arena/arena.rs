// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The arena of rings: its columns and its steps.

use super::iter::RingSequenceIter;
use super::node::Node;
use super::slot::Slot;
use typed_index_collections::TiVec;

/// A fixed set of nodes arranged in closed rings, as a doubly linked list per ring.
///
/// Every node has a successor and a predecessor, following which leads around its ring and back; a node alone is
/// a ring of one. The rings are rearranged in place by steps that each touch a constant number of links (moving a
/// node or a segment, swapping two nodes or two segments) and by reversing a segment, which costs its length.
/// Nodes never enter or leave: [`len`](RingArena::len) stays fixed until the arena is rebuilt by
/// [`reset_to_self_loops`](RingArena::reset_to_self_loops) or overwritten by [`Clone::clone_from`], which reuses
/// the arena's memory.
///
/// Every step is `_unchecked`: the arena is private to the crate, and the schedule graph, which owns it, proves by
/// its own invariant that every node it hands over lies in the arena. The tests check the steps against a model of
/// the rings as lists.
///
/// The steps on segments also take a condition no constant-time check can confirm: `last` follows `first` along
/// the successors, and the segments or the anchor they are given do not overlap. It is stated in each step's
/// `# Preconditions` and checked in debug builds. Breaking it breaks the rings (a wrong answer), but never makes an
/// access out of bounds: every link the arena holds names one of its nodes, whatever the steps are handed.
// Invariant: `prev` and `next` have the same length, at most one node for every `u32`, every entry of both is a
// node below that length, and `prev[next[v]] == v` for every node `v`, so that `next` is a permutation whose cycles
// are the rings and `prev` is its inverse. The first parts are what memory rests on: they make every link a valid
// index into both columns, and every step keeps them, since a step writes only nodes it was handed (which lie in the
// arena) or read from a column. The last part makes the nodes rings; the steps keep it under their preconditions.
//
// The segment conditions are held by no type: whether `last` follows `first`, or a node lies in a segment, is a fact
// of the rings as they stand, which every step changes. Nor can the steps be made total: telling a segment from two
// nodes that are not one walks the segment, and the steps are constant-time. Their callers own the rings' shape and
// hold the conditions; a debug build checks them at each step's entry.
#[derive(PartialEq, Eq, Debug, Hash, Default)]
pub struct RingArena {
    /// The predecessor of each node, per node.
    prev: TiVec<Node, Node>,
    /// The successor of each node, per node.
    next: TiVec<Node, Node>,
}

/// Whether every node of an arena of `len` nodes has a [`Node`] that names it.
fn nameable(len: usize) -> bool {
    len.checked_sub(1)
        .is_none_or(|last| Node::try_new(last).is_some())
}

impl RingArena {
    /// Replaces the arena by `len` nodes, each a ring of its own, reusing the arena's memory.
    ///
    /// # Panics
    ///
    /// Panics if a node of the arena would lie beyond what a [`Node`] can name; the arena is then left
    /// unchanged.
    #[track_caller]
    pub fn reset_to_self_loops(&mut self, len: usize) {
        // Checked before anything is cleared: a panic midway would leave columns of different lengths, which a
        // caller that catches the panic would then read out of bounds through steps that trust their length.
        assert!(
            nameable(len),
            "called `RingArena::reset_to_self_loops` with {len} nodes, more than a node id can name"
        );
        self.prev.clear();
        self.prev.extend((0..len).map(Node::new));
        self.next.clear();
        self.next.extend_from_slice(&self.prev);
    }

    /// The number of nodes.
    #[inline]
    pub fn len(&self) -> usize {
        self.prev.len()
    }

    /// The successor of `node` in its ring, for a `node` the caller knows to lie in the arena.
    ///
    /// # Safety
    ///
    /// `node` must lie in the arena.
    #[inline]
    pub unsafe fn next_unchecked(&self, node: Node) -> Node {
        debug_assert!(
            node.index() < self.len(),
            "called `RingArena::next_unchecked` with {node}, outside an arena of {} nodes",
            self.len()
        );
        // SAFETY: `node` lies in the arena, the precondition of this function.
        let [node] = unsafe { self.slots_unchecked([node]) };
        self.next_slot(node).node()
    }

    /// The predecessor of `node` in its ring, for a `node` the caller knows to lie in the arena.
    ///
    /// # Safety
    ///
    /// `node` must lie in the arena.
    #[inline]
    pub unsafe fn prev_unchecked(&self, node: Node) -> Node {
        debug_assert!(
            node.index() < self.len(),
            "called `RingArena::prev_unchecked` with {node}, outside an arena of {} nodes",
            self.len()
        );
        // SAFETY: `node` lies in the arena, the precondition of this function.
        let [node] = unsafe { self.slots_unchecked([node]) };
        self.prev_slot(node).node()
    }

    /// The nodes from `start` along the successors, up to but not including `stop`.
    ///
    /// The walk ends after at most [`len`](RingArena::len) nodes: if `stop` is not on the ring of `start`, the
    /// walk goes round that ring until it has yielded that many.
    ///
    /// # Safety
    ///
    /// `start` and `stop` must lie in the arena.
    pub unsafe fn sequence_unchecked(&self, start: Node, stop: Node) -> RingSequenceIter<'_> {
        debug_assert!(
            start.index() < self.len() && stop.index() < self.len(),
            "called `RingArena::sequence_unchecked` with {start} or {stop}, outside an arena of {} nodes",
            self.len()
        );
        // SAFETY: both nodes lie in the arena, the precondition of this function.
        let [start, stop] = unsafe { self.slots_unchecked([start, stop]) };
        RingSequenceIter::new(&self.next, start, stop)
    }

    /// The nodes from `start` along the predecessors, up to but not including `stop`.
    ///
    /// The walk ends after at most [`len`](RingArena::len) nodes, as for
    /// [`sequence_unchecked`](RingArena::sequence_unchecked).
    ///
    /// # Safety
    ///
    /// `start` and `stop` must lie in the arena.
    pub unsafe fn sequence_rev_unchecked(&self, start: Node, stop: Node) -> RingSequenceIter<'_> {
        debug_assert!(
            start.index() < self.len() && stop.index() < self.len(),
            "called `RingArena::sequence_rev_unchecked` with {start} or {stop}, outside an arena of {} nodes",
            self.len()
        );
        // SAFETY: both nodes lie in the arena, the precondition of this function.
        let [start, stop] = unsafe { self.slots_unchecked([start, stop]) };
        RingSequenceIter::new(&self.prev, start, stop)
    }

    /// Exchanges the places of `first` and `second`, in one ring or across two.
    ///
    /// ```text
    /// before:  ... a_prev -> first -> a_next ...      ... b_prev -> second -> b_next ...
    /// after:   ... a_prev -> second -> a_next ...     ... b_prev -> first -> b_next ...
    /// ```
    ///
    /// # Safety
    ///
    /// `first` and `second` must lie in the arena.
    pub unsafe fn swap_nodes_unchecked(&mut self, first: Node, second: Node) {
        debug_assert!(
            first.index() < self.len() && second.index() < self.len(),
            "called `RingArena::swap_nodes_unchecked` with {first} or {second}, outside an arena of {} nodes",
            self.len()
        );
        // SAFETY: both nodes lie in the arena, the precondition of this function.
        let [first, second] = unsafe { self.slots_unchecked([first, second]) };
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
    /// node unless they are the same segment, which leaves the rings as they are. Otherwise the rings break,
    /// though no access leaves the arena; the caller that owns the rings' shape holds this, and a debug build
    /// checks it.
    ///
    /// # Safety
    ///
    /// The four nodes must lie in the arena.
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
            "called `RingArena::swap_segments_unchecked` with {a_first}, {a_last}, {b_first} or {b_last}, \
             outside an arena of {} nodes",
            self.len()
        );
        // SAFETY: the four nodes lie in the arena, the precondition of this function.
        let [a_first, a_last, b_first, b_last] =
            unsafe { self.slots_unchecked([a_first, a_last, b_first, b_last]) };
        // Held by no type and not made total, as the segment conditions on `RingArena` say.
        debug_assert!(
            self.follows(a_first, a_last) && self.follows(b_first, b_last),
            "called `RingArena::swap_segments_unchecked` with a last node that does not follow its first: \
             {a_first} to {a_last}, {b_first} to {b_last}"
        );
        debug_assert!(
            self.same_or_disjoint(a_first, a_last, b_first, b_last),
            "called `RingArena::swap_segments_unchecked` with overlapping segments {a_first} to {a_last} and \
             {b_first} to {b_last}"
        );
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
    /// # Safety
    ///
    /// `node` and `anchor` must lie in the arena.
    pub unsafe fn relocate_after_unchecked(&mut self, node: Node, anchor: Node) {
        debug_assert!(
            node.index() < self.len() && anchor.index() < self.len(),
            "called `RingArena::relocate_after_unchecked` with {node} or {anchor}, outside an arena of {} nodes",
            self.len()
        );
        // SAFETY: both nodes lie in the arena, the precondition of this function.
        let [node, anchor] = unsafe { self.slots_unchecked([node, anchor]) };
        self.move_after(node, anchor);
    }

    /// Moves `node` out of its ring to directly before `anchor`; nothing changes if `node` is `anchor` or already
    /// precedes it.
    ///
    /// # Safety
    ///
    /// `node` and `anchor` must lie in the arena.
    pub unsafe fn relocate_before_unchecked(&mut self, node: Node, anchor: Node) {
        debug_assert!(
            node.index() < self.len() && anchor.index() < self.len(),
            "called `RingArena::relocate_before_unchecked` with {node} or {anchor}, outside an arena of {} nodes",
            self.len()
        );
        // SAFETY: both nodes lie in the arena, the precondition of this function.
        let [node, anchor] = unsafe { self.slots_unchecked([node, anchor]) };
        self.move_before(node, anchor);
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
    /// rings break, though no access leaves the arena; the caller that owns the rings' shape holds this, and a
    /// debug build checks it.
    ///
    /// # Safety
    ///
    /// The three nodes must lie in the arena.
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
            "called `RingArena::relocate_segment_after_unchecked` with {first}, {last} or {anchor}, outside an \
             arena of {} nodes",
            self.len()
        );
        // SAFETY: the three nodes lie in the arena, the precondition of this function.
        let [first, last, anchor] = unsafe { self.slots_unchecked([first, last, anchor]) };
        // Held by no type and not made total, as the segment conditions on `RingArena` say.
        debug_assert!(
            self.follows(first, last),
            "called `RingArena::relocate_segment_after_unchecked` with {last}, which does not follow {first}"
        );
        debug_assert!(
            !self.segment_holds(first, last, anchor),
            "called `RingArena::relocate_segment_after_unchecked` with anchor {anchor} inside the segment \
             {first} to {last}"
        );
        self.move_range_after(first, last, anchor);
    }

    /// Moves the segment `first ..= last` out of its ring to directly before `anchor`; nothing changes if the
    /// segment already precedes it.
    ///
    /// ```text
    /// before:  ... prev -> [first ... last] -> next ...      ... anchor_prev -> anchor ...
    /// after:   ... prev -> next ...                          ... anchor_prev -> [first ... last] -> anchor ...
    /// ```
    ///
    /// # Preconditions
    ///
    /// `last` follows `first` along the successors, and `anchor` is not a node of the segment. Otherwise the
    /// rings break, though no access leaves the arena; the caller that owns the rings' shape holds this, and a
    /// debug build checks it.
    ///
    /// # Safety
    ///
    /// The three nodes must lie in the arena.
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
            "called `RingArena::relocate_segment_before_unchecked` with {first}, {last} or {anchor}, outside an \
             arena of {} nodes",
            self.len()
        );
        // SAFETY: the three nodes lie in the arena, the precondition of this function.
        let [first, last, anchor] = unsafe { self.slots_unchecked([first, last, anchor]) };
        // Held by no type and not made total, as the segment conditions on `RingArena` say.
        debug_assert!(
            self.follows(first, last),
            "called `RingArena::relocate_segment_before_unchecked` with {last}, which does not follow {first}"
        );
        debug_assert!(
            !self.segment_holds(first, last, anchor),
            "called `RingArena::relocate_segment_before_unchecked` with anchor {anchor} inside the segment \
             {first} to {last}"
        );
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
    /// and the walk ends after [`len`](RingArena::len) nodes; the caller that owns the rings' shape holds this,
    /// and a debug build checks it.
    ///
    /// # Safety
    ///
    /// `first` and `last` must lie in the arena.
    pub unsafe fn reverse_segment_unchecked(&mut self, first: Node, last: Node) {
        debug_assert!(
            first.index() < self.len() && last.index() < self.len(),
            "called `RingArena::reverse_segment_unchecked` with {first} or {last}, outside an arena of {} nodes",
            self.len()
        );
        // SAFETY: both nodes lie in the arena, the precondition of this function.
        let [first, last] = unsafe { self.slots_unchecked([first, last]) };
        // Held by no type and not made total, as the segment conditions on `RingArena` say.
        debug_assert!(
            self.follows(first, last),
            "called `RingArena::reverse_segment_unchecked` with {last}, which does not follow {first}"
        );
        self.reverse(first, last);
    }

    /// The slots of `nodes`.
    ///
    /// # Safety
    ///
    /// Every node of `nodes` must lie in the arena: every later access through its slot trusts it.
    #[inline]
    unsafe fn slots_unchecked<const N: usize>(&self, nodes: [Node; N]) -> [Slot; N] {
        nodes.map(|node| {
            debug_assert!(
                node.index() < self.len(),
                "called `RingArena::slots_unchecked` with {node}, outside an arena of {} nodes",
                self.len()
            );
            // SAFETY: the node lies in the arena, the precondition of this function.
            unsafe { Slot::new_unchecked(node) }
        })
    }

    /// The slot of the successor of `node`.
    #[inline]
    fn next_slot(&self, node: Slot) -> Slot {
        Self::follow(&self.next, node)
    }

    /// The slot of the predecessor of `node`.
    #[inline]
    fn prev_slot(&self, node: Slot) -> Slot {
        Self::follow(&self.prev, node)
    }

    /// The slot of the link of `node` in `column`, one of the arena's two columns.
    #[inline]
    fn follow(column: &TiVec<Node, Node>, node: Slot) -> Slot {
        let node = node.node();
        debug_assert!(
            node.index() < column.len(),
            "the slot {node} should lie in the arena of {} nodes",
            column.len()
        );
        // SAFETY: a slot lies in its arena, whose columns have its length (`Slot`, `RingArena`).
        let link = unsafe { *column.get_unchecked(node) };
        debug_assert!(
            link.index() < column.len(),
            "the link of {node} should name a node of the arena of {} nodes",
            column.len()
        );
        // SAFETY: every link of the arena names one of its nodes (`RingArena`).
        unsafe { Slot::new_unchecked(link) }
    }

    /// Makes `to` the successor of `from` and `from` the predecessor of `to`.
    #[inline]
    fn link(&mut self, from: Slot, to: Slot) {
        let (from, to) = (from.node(), to.node());
        debug_assert!(
            from.index() < self.next.len() && to.index() < self.prev.len(),
            "the slots {from} and {to} should lie in the arena of {} nodes",
            self.len()
        );
        // SAFETY: both slots lie in the arena, whose two columns have its length (`Slot`, `RingArena`).
        unsafe {
            *self.next.get_unchecked_mut(from) = to;
            *self.prev.get_unchecked_mut(to) = from;
        }
    }

    /// Exchanges the successor and the predecessor of `node`.
    #[inline]
    fn flip(&mut self, node: Slot) {
        let node = node.node();
        debug_assert!(
            node.index() < self.next.len() && node.index() < self.prev.len(),
            "the slot {node} should lie in the arena of {} nodes",
            self.len()
        );
        // SAFETY: the slot lies in the arena, whose two columns have its length (`Slot`, `RingArena`).
        unsafe {
            std::mem::swap(
                self.next.get_unchecked_mut(node),
                self.prev.get_unchecked_mut(node),
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

    /// Whether the segments `a_first ..= a_last` and `b_first ..= b_last` are one segment or share no node.
    fn same_or_disjoint(&self, a_first: Slot, a_last: Slot, b_first: Slot, b_last: Slot) -> bool {
        // Two segments of one ring that share a node hold each other's first node.
        (a_first == b_first && a_last == b_last)
            || !(self.segment_holds(a_first, a_last, b_first)
                || self.segment_holds(b_first, b_last, a_first))
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

    /// Exchanges the places of the segments `a_first ..= a_last` and `b_first ..= b_last`, which are one segment
    /// or share no node.
    fn swap_ranges(&mut self, a_first: Slot, a_last: Slot, b_first: Slot, b_last: Slot) {
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

    /// Moves `node` to directly before `anchor`.
    fn move_before(&mut self, node: Slot, anchor: Slot) {
        // Nothing moves when `node` already precedes the anchor or is the anchor: the move after the anchor's
        // predecessor then puts a node after itself or after its own predecessor.
        let anchor_prev = self.prev_slot(anchor);
        self.move_after(node, anchor_prev);
    }

    /// Moves the segment `first ..= last` to directly after `anchor`, a node outside it.
    fn move_range_after(&mut self, first: Slot, last: Slot, anchor: Slot) {
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

    /// Moves the segment `first ..= last` to directly before `anchor`, a node outside it.
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

impl Clone for RingArena {
    fn clone(&self) -> Self {
        Self {
            prev: self.prev.clone(),
            next: self.next.clone(),
        }
    }

    // Written out so that overwriting an arena reuses both of its buffers; the columns' own `clone_from` would not.
    fn clone_from(&mut self, source: &Self) {
        self.prev.raw.clone_from(&source.prev.raw);
        self.next.raw.clone_from(&source.next.raw);
    }
}

#[cfg(test)]
mod tests {
    use super::super::ring_lists::{
        InArena, Rings, arena_of, assert_links_are_inverse, canonical, locate, next, prev, ring_of,
        rings_of, segment, some_rings, three_rings,
    };
    use super::*;
    use proptest::prelude::*;

    fn nodes<const N: usize>(indices: [usize; N]) -> [Node; N] {
        indices.map(Node::new)
    }

    /// The rings of `arena` that hold `starts`, each from its start.
    fn rings_from(arena: &RingArena, starts: &[usize]) -> Rings {
        starts.iter().map(|&start| ring_of(arena, start)).collect()
    }

    /// The slot of node `index`, which the test asserts to lie in `arena`.
    fn slot_of(arena: &RingArena, index: usize) -> Slot {
        assert!(index < arena.len(), "node {index} should lie in the arena");
        // SAFETY: the node lies in the arena, as the assertion above found.
        let [slot] = unsafe { arena.slots_unchecked([Node::new(index)]) };
        slot
    }

    /// One ring `0 -> 1 -> ... -> len - 1`.
    fn one_ring(len: usize) -> RingArena {
        arena_of(&[(0..len).collect()], len)
    }

    #[test]
    fn test_reset_to_self_loops_makes_each_node_a_ring_of_its_own() {
        let mut arena = three_rings();
        arena.reset_to_self_loops(5);
        assert_eq!(arena.len(), 5);
        assert_eq!(rings_of(&arena), [[0], [1], [2], [3], [4]]);
        arena.reset_to_self_loops(0);
        assert_eq!(arena.len(), 0);
    }

    #[test]
    fn test_default_has_no_node() {
        assert_eq!(RingArena::default().len(), 0);
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
        let mut arena = three_rings();
        let too_many = usize::try_from(u32::MAX).unwrap() + 2;
        let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            arena.reset_to_self_loops(too_many);
        }));
        assert!(refused.is_err());
        assert_eq!(arena, three_rings());
    }

    #[test]
    fn test_next_and_prev_follow_the_rings() {
        let arena = three_rings();
        let [n0, n1, n3, n7] = nodes([0, 1, 3, 7]);
        assert_eq!((next(&arena, n0), prev(&arena, n1)), (n1, n0));
        assert_eq!((next(&arena, n3), prev(&arena, n0)), (n0, n3));
        assert_eq!((next(&arena, n7), prev(&arena, n7)), (n7, n7));
    }

    #[test]
    fn test_swap_nodes_in_one_ring() {
        for ([first, second], expected) in [
            ([1, 2], [0, 2, 1, 3]),
            ([2, 1], [0, 2, 1, 3]),
            ([0, 2], [0, 3, 2, 1]),
            ([0, 3], [0, 3, 1, 2]),
            ([1, 1], [0, 1, 2, 3]),
        ] {
            let mut arena = three_rings();
            arena.swap_nodes(Node::new(first), Node::new(second));
            assert_links_are_inverse(&arena);
            assert_eq!(
                ring_of(&arena, 0),
                expected,
                "after swapping {first} and {second}"
            );
        }
    }

    #[test]
    fn test_swap_nodes_across_rings() {
        let mut arena = three_rings();
        arena.swap_nodes(Node::new(1), Node::new(5));
        assert_eq!(
            rings_from(&arena, &[0, 4]),
            [vec![0, 5, 2, 3], vec![4, 1, 6]]
        );

        let mut arena = three_rings();
        arena.swap_nodes(Node::new(2), Node::new(7));
        assert_eq!(rings_from(&arena, &[0, 2]), [vec![0, 1, 7, 3], vec![2]]);
    }

    #[test]
    fn test_swap_nodes_of_rings_of_one_or_two_keeps_the_rings() {
        let mut arena = arena_of(&[vec![0], vec![1]], 2);
        arena.swap_nodes(Node::new(0), Node::new(1));
        assert_eq!(rings_of(&arena), [[0], [1]]);

        let mut arena = one_ring(2);
        arena.swap_nodes(Node::new(0), Node::new(1));
        assert_links_are_inverse(&arena);
        assert_eq!(rings_of(&arena), [[0, 1]]);
    }

    #[test]
    fn test_swap_segments_across_rings() {
        let mut arena = three_rings();
        arena.swap_segments(Node::new(1), Node::new(2), Node::new(5), Node::new(6));
        assert_eq!(
            rings_from(&arena, &[0, 4]),
            [vec![0, 5, 6, 3], vec![4, 1, 2]]
        );

        let mut arena = three_rings();
        arena.swap_segments(Node::new(1), Node::new(3), Node::new(5), Node::new(5));
        assert_eq!(
            rings_from(&arena, &[0, 4]),
            [vec![0, 5], vec![4, 1, 2, 3, 6]]
        );
    }

    #[test]
    fn test_swap_segments_in_one_ring() {
        let mut arena = three_rings();
        arena.swap_segments(Node::new(0), Node::new(1), Node::new(2), Node::new(3));
        assert_eq!(ring_of(&arena, 2), [2, 3, 0, 1]);

        let mut arena = three_rings();
        arena.swap_segments(Node::new(1), Node::new(2), Node::new(1), Node::new(2));
        assert_eq!(ring_of(&arena, 0), [0, 1, 2, 3]);

        let mut arena = one_ring(6);
        arena.swap_segments(Node::new(1), Node::new(1), Node::new(3), Node::new(4));
        assert_eq!(ring_of(&arena, 0), [0, 3, 4, 2, 1, 5]);
    }

    #[test]
    fn test_swap_segments_with_a_whole_ring_moves_it_into_the_other_place() {
        let mut arena = three_rings();
        arena.swap_segments(Node::new(7), Node::new(7), Node::new(5), Node::new(5));
        assert_eq!(rings_from(&arena, &[4, 5]), [vec![4, 7, 6], vec![5]]);

        let mut arena = three_rings();
        arena.swap_segments(Node::new(1), Node::new(2), Node::new(4), Node::new(6));
        assert_eq!(
            rings_from(&arena, &[0, 1]),
            [vec![0, 4, 5, 6, 3], vec![1, 2]]
        );
    }

    #[test]
    fn test_relocate_after() {
        for ([node, anchor], expected) in [
            ([1, 3], vec![vec![0, 2, 3, 1], vec![4, 5, 6]]),
            ([1, 5], vec![vec![0, 2, 3], vec![4, 5, 1, 6]]),
            ([1, 0], vec![vec![0, 1, 2, 3], vec![4, 5, 6]]),
            ([1, 1], vec![vec![0, 1, 2, 3], vec![4, 5, 6]]),
        ] {
            let mut arena = three_rings();
            arena.relocate_after(Node::new(node), Node::new(anchor));
            assert_links_are_inverse(&arena);
            assert_eq!(
                rings_from(&arena, &[0, 4]),
                expected,
                "after moving {node} after {anchor}"
            );
        }

        let mut arena = three_rings();
        arena.relocate_after(Node::new(2), Node::new(7));
        assert_eq!(rings_from(&arena, &[0, 7]), [vec![0, 1, 3], vec![7, 2]]);
    }

    #[test]
    fn test_relocate_before() {
        for ([node, anchor], expected) in [
            ([3, 1], vec![vec![0, 3, 1, 2], vec![4, 5, 6]]),
            ([1, 5], vec![vec![0, 2, 3], vec![4, 1, 5, 6]]),
            ([1, 2], vec![vec![0, 1, 2, 3], vec![4, 5, 6]]),
            ([2, 2], vec![vec![0, 1, 2, 3], vec![4, 5, 6]]),
        ] {
            let mut arena = three_rings();
            arena.relocate_before(Node::new(node), Node::new(anchor));
            assert_links_are_inverse(&arena);
            assert_eq!(
                rings_from(&arena, &[0, 4]),
                expected,
                "after moving {node} before {anchor}"
            );
        }

        let mut arena = three_rings();
        arena.relocate_before(Node::new(2), Node::new(7));
        assert_eq!(rings_from(&arena, &[0, 7]), [vec![0, 1, 3], vec![7, 2]]);
    }

    #[test]
    fn test_relocate_segment_after() {
        for ([first, last, anchor], expected) in [
            ([1, 2, 5], vec![vec![0, 3], vec![4, 5, 1, 2, 6], vec![7]]),
            ([1, 2, 3], vec![vec![0, 3, 1, 2], vec![4, 5, 6], vec![7]]),
            ([1, 2, 0], vec![vec![0, 1, 2, 3], vec![4, 5, 6], vec![7]]),
            ([2, 2, 5], vec![vec![0, 1, 3], vec![4, 5, 2, 6], vec![7]]),
            ([1, 2, 7], vec![vec![0, 3], vec![4, 5, 6], vec![7, 1, 2]]),
        ] {
            let mut arena = three_rings();
            arena.relocate_segment_after(Node::new(first), Node::new(last), Node::new(anchor));
            assert_links_are_inverse(&arena);
            assert_eq!(
                rings_from(&arena, &[0, 4, 7]),
                expected,
                "after moving {first} to {last} after {anchor}"
            );
        }
    }

    #[test]
    fn test_relocate_segment_before() {
        for ([first, last, anchor], expected) in [
            ([1, 2, 5], vec![vec![0, 3], vec![4, 1, 2, 5, 6]]),
            ([2, 3, 1], vec![vec![0, 2, 3, 1], vec![4, 5, 6]]),
            ([1, 2, 3], vec![vec![0, 1, 2, 3], vec![4, 5, 6]]),
        ] {
            let mut arena = three_rings();
            // SAFETY: every node handed over lies in the arena under test.
            unsafe {
                arena.relocate_segment_before_unchecked(
                    Node::new(first),
                    Node::new(last),
                    Node::new(anchor),
                );
            }
            assert_links_are_inverse(&arena);
            assert_eq!(
                rings_from(&arena, &[0, 4]),
                expected,
                "after moving {first} to {last} before {anchor}"
            );
        }
    }

    #[test]
    fn test_relocate_segment_before_the_node_it_already_precedes_changes_nothing() {
        // One ring 0 -> 1 -> 2 -> 3 -> 4; the segment [1, 2] already stands directly before 3.
        let mut arena = one_ring(5);
        // SAFETY: every node handed over lies in the arena under test.
        unsafe {
            arena.relocate_segment_before_unchecked(Node::new(1), Node::new(2), Node::new(3))
        };
        assert_links_are_inverse(&arena);
        assert_eq!(ring_of(&arena, 0), [0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_reverse_segment() {
        for ([first, last], expected) in [
            ([1, 2], vec![0, 2, 1, 3]),
            ([1, 3], vec![0, 3, 2, 1]),
            ([2, 3], vec![0, 1, 3, 2]),
            ([1, 1], vec![0, 1, 2, 3]),
            ([0, 3], vec![0, 3, 2, 1]),
            ([2, 1], vec![0, 3, 2, 1]),
        ] {
            let mut arena = three_rings();
            arena.reverse_segment(Node::new(first), Node::new(last));
            assert_links_are_inverse(&arena);
            assert_eq!(
                ring_of(&arena, 0),
                expected,
                "after reversing {first} to {last}"
            );
        }
    }

    #[test]
    fn test_reverse_segment_of_all_but_one_node_of_a_ring() {
        // Ring 4 -> 0 -> 1 -> 2 -> 3; reversing 0 to 3 leaves 4 in place.
        let mut arena = arena_of(&[vec![4, 0, 1, 2, 3]], 5);
        arena.reverse_segment(Node::new(0), Node::new(3));
        assert_links_are_inverse(&arena);
        assert_eq!(ring_of(&arena, 4), [4, 3, 2, 1, 0]);
    }

    #[test]
    fn test_steps_compose() {
        let mut arena = three_rings();
        arena.swap_nodes(Node::new(1), Node::new(2));
        arena.reverse_segment(Node::new(2), Node::new(1));
        assert_eq!(ring_of(&arena, 0), [0, 1, 2, 3]);

        arena.relocate_after(Node::new(1), Node::new(3));
        arena.swap_nodes(Node::new(0), Node::new(3));
        assert_eq!(ring_of(&arena, 3), [3, 2, 0, 1]);

        arena.relocate_after(Node::new(0), Node::new(5));
        assert_eq!(
            rings_from(&arena, &[3, 4]),
            [vec![3, 2, 1], vec![4, 5, 0, 6]]
        );
    }

    #[test]
    fn test_clone_from_reuses_the_buffers_and_copies_the_rings() {
        let source = three_rings();
        let mut target = RingArena::default();
        target.reset_to_self_loops(100);
        let capacities = (target.prev.capacity(), target.next.capacity());
        target.clone_from(&source);
        assert_eq!(target, source);
        assert_eq!((target.prev.capacity(), target.next.capacity()), capacities);
    }

    #[test]
    fn test_a_step_makes_the_arena_differ_from_its_clone() {
        let arena = three_rings();
        let mut changed = arena.clone();
        assert_eq!(changed, arena);
        changed.swap_nodes(Node::new(0), Node::new(1));
        assert_ne!(changed, arena);
    }

    #[test]
    fn test_follows_walks_one_ring() {
        let arena = three_rings();
        let slot = |index| slot_of(&arena, index);
        assert!(arena.follows(slot(1), slot(3)));
        assert!(arena.follows(slot(3), slot(1)));
        assert!(arena.follows(slot(2), slot(2)));
        assert!(!arena.follows(slot(1), slot(5)));
    }

    #[test]
    fn test_segment_holds_only_the_nodes_between_its_ends() {
        let arena = three_rings();
        let slot = |index| slot_of(&arena, index);
        assert!(arena.segment_holds(slot(1), slot(3), slot(2)));
        assert!(arena.segment_holds(slot(1), slot(3), slot(1)));
        assert!(arena.segment_holds(slot(1), slot(3), slot(3)));
        assert!(!arena.segment_holds(slot(1), slot(3), slot(0)));
        assert!(arena.segment_holds(slot(3), slot(1), slot(0)));
        assert!(!arena.segment_holds(slot(1), slot(3), slot(5)));
    }

    #[test]
    fn test_same_or_disjoint() {
        let arena = three_rings();
        let slot = |index| slot_of(&arena, index);
        assert!(arena.same_or_disjoint(slot(1), slot(2), slot(1), slot(2)));
        assert!(arena.same_or_disjoint(slot(0), slot(1), slot(2), slot(3)));
        assert!(arena.same_or_disjoint(slot(0), slot(1), slot(4), slot(6)));
        assert!(!arena.same_or_disjoint(slot(1), slot(2), slot(1), slot(3)));
        assert!(!arena.same_or_disjoint(slot(0), slot(2), slot(1), slot(3)));
        assert!(!arena.same_or_disjoint(slot(3), slot(1), slot(0), slot(0)));
    }

    // The precondition is checked by a `debug_assert!`, which a release build leaves out.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(
        expected = "called `RingArena::reverse_segment_unchecked` with Node 5, which does not follow Node 1"
    )]
    fn test_reverse_segment_across_rings_fails_its_precondition() {
        three_rings().reverse_segment(Node::new(1), Node::new(5));
    }

    // The precondition is checked by a `debug_assert!`, which a release build leaves out.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(
        expected = "called `RingArena::relocate_segment_after_unchecked` with anchor Node 2 inside the segment Node 1 \
                    to Node 3"
    )]
    fn test_relocate_segment_onto_itself_fails_its_precondition() {
        three_rings().relocate_segment_after(Node::new(1), Node::new(3), Node::new(2));
    }

    // The precondition is checked by a `debug_assert!`, which a release build leaves out.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(
        expected = "called `RingArena::relocate_segment_before_unchecked` with anchor Node 2 inside the segment"
    )]
    fn test_relocate_segment_before_onto_itself_fails_its_precondition() {
        // SAFETY: every node handed over lies in the arena under test.
        unsafe {
            three_rings().relocate_segment_before_unchecked(
                Node::new(1),
                Node::new(3),
                Node::new(2),
            )
        };
    }

    // The precondition is checked by a `debug_assert!`, which a release build leaves out.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(
        expected = "called `RingArena::swap_segments_unchecked` with overlapping segments"
    )]
    fn test_swap_overlapping_segments_fails_its_precondition() {
        three_rings().swap_segments(Node::new(0), Node::new(2), Node::new(1), Node::new(3));
    }

    // The precondition is checked by a `debug_assert!`, which a release build leaves out.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(
        expected = "called `RingArena::swap_segments_unchecked` with overlapping segments"
    )]
    fn test_swap_segments_of_one_first_node_but_different_last_nodes_fails_its_precondition() {
        three_rings().swap_segments(Node::new(1), Node::new(2), Node::new(1), Node::new(3));
    }

    #[derive(Clone, Debug)]
    enum Change {
        SwapNodes(usize, usize),
        RelocateAfter(usize, usize),
        RelocateBefore(usize, usize),
        Reverse(usize, usize),
        RelocateSegmentAfter(usize, usize, usize),
        RelocateSegmentBefore(usize, usize, usize),
        SwapSegments(usize, usize, usize, usize),
    }

    fn change() -> impl Strategy<Value = Change> {
        let any = || 0usize..64;
        prop_oneof![
            (any(), any()).prop_map(|(a, b)| Change::SwapNodes(a, b)),
            (any(), any()).prop_map(|(a, b)| Change::RelocateAfter(a, b)),
            (any(), any()).prop_map(|(a, b)| Change::RelocateBefore(a, b)),
            (any(), any()).prop_map(|(a, b)| Change::Reverse(a, b)),
            (any(), any(), any()).prop_map(|(a, b, c)| Change::RelocateSegmentAfter(a, b, c)),
            (any(), any(), any()).prop_map(|(a, b, c)| Change::RelocateSegmentBefore(a, b, c)),
            (any(), any(), any(), any()).prop_map(|(a, b, c, d)| Change::SwapSegments(a, b, c, d)),
        ]
    }

    /// The model and the arena kept in step: every step changes both, and the arena must keep the rings of the
    /// model.
    struct Harness {
        len: usize,
        rings: Rings,
        arena: RingArena,
    }

    impl Harness {
        fn new(len: usize, rings: Rings) -> Self {
            let arena = arena_of(&rings, len);
            Self { len, rings, arena }
        }

        fn node(&self, index: usize) -> Node {
            Node::new(index % self.len)
        }

        /// Applies `change` to the model and to the arena, unless its nodes break the step's preconditions,
        /// which leaves both as they were.
        fn apply(&mut self, change: &Change) {
            match *change {
                Change::SwapNodes(a, b) => self.swap_nodes(a % self.len, b % self.len),
                Change::RelocateAfter(a, b) => self.relocate(a % self.len, b % self.len, true),
                Change::RelocateBefore(a, b) => self.relocate(a % self.len, b % self.len, false),
                Change::Reverse(a, extra) => self.reverse(a % self.len, extra),
                Change::RelocateSegmentAfter(a, extra, c) => {
                    self.relocate_segment(a % self.len, extra, c, true)
                }
                Change::RelocateSegmentBefore(a, extra, c) => {
                    self.relocate_segment(a % self.len, extra, c, false)
                }
                Change::SwapSegments(a, a_extra, b, b_extra) => {
                    self.swap_segments(a % self.len, a_extra, b % self.len, b_extra);
                }
            }
        }

        fn swap_nodes(&mut self, a: usize, b: usize) {
            let ((ring_a, place_a), (ring_b, place_b)) =
                (locate(&self.rings, a), locate(&self.rings, b));
            self.rings[ring_a][place_a] = b;
            self.rings[ring_b][place_b] = a;
            let [a, b] = nodes([a, b]);
            // SAFETY: every node handed over lies in the arena under test.
            unsafe { self.arena.swap_nodes_unchecked(a, b) };
        }

        fn relocate(&mut self, moved: usize, anchor: usize, after: bool) {
            if moved != anchor {
                let (ring, place) = locate(&self.rings, moved);
                self.rings[ring].remove(place);
                let (ring, place) = locate(&self.rings, anchor);
                self.rings[ring].insert(if after { place + 1 } else { place }, moved);
                self.rings.retain(|ring| !ring.is_empty());
            }
            let [moved, anchor] = nodes([moved, anchor]);
            if after {
                // SAFETY: every node handed over lies in the arena under test.
                unsafe { self.arena.relocate_after_unchecked(moved, anchor) };
            } else {
                // SAFETY: every node handed over lies in the arena under test.
                unsafe { self.arena.relocate_before_unchecked(moved, anchor) };
            }
        }

        fn reverse(&mut self, first: usize, extra: usize) {
            let (ring, mut nodes, rest) = segment(&self.rings, first, extra);
            let last = Node::new(*nodes.last().unwrap());
            nodes.reverse();
            self.rings[ring] = nodes.into_iter().chain(rest).collect();
            let first = Node::new(first);
            // SAFETY: every node handed over lies in the arena under test.
            unsafe { self.arena.reverse_segment_unchecked(first, last) };
        }

        fn relocate_segment(&mut self, first: usize, extra: usize, anchor: usize, after: bool) {
            let (ring, nodes, rest) = segment(&self.rings, first, extra);
            let anchor = self.node(anchor);
            if nodes.contains(&anchor.index()) {
                return;
            }
            let (first, last) = (Node::new(nodes[0]), Node::new(*nodes.last().unwrap()));
            self.rings[ring] = rest;
            let (anchor_ring, place) = locate(&self.rings, anchor.index());
            let at = if after { place + 1 } else { place };
            self.rings[anchor_ring].splice(at..at, nodes);
            self.rings.retain(|ring| !ring.is_empty());
            if after {
                // SAFETY: every node handed over lies in the arena under test.
                unsafe {
                    self.arena
                        .relocate_segment_after_unchecked(first, last, anchor)
                };
            } else {
                // SAFETY: every node handed over lies in the arena under test.
                unsafe {
                    self.arena
                        .relocate_segment_before_unchecked(first, last, anchor)
                };
            }
        }

        fn swap_segments(&mut self, a: usize, a_extra: usize, b: usize, b_extra: usize) {
            let (ring_a, nodes_a, rest_a) = segment(&self.rings, a, a_extra);
            let (ring_b, nodes_b, rest_b) = segment(&self.rings, b, b_extra);
            let same = nodes_a == nodes_b;
            if !same && (nodes_a.contains(&nodes_b[0]) || nodes_b.contains(&nodes_a[0])) {
                return;
            }
            if same {
                // A segment swapped with itself stays.
            } else if ring_a == ring_b {
                // The ring is A, X, B, Y with A first; it becomes B, X, A, Y.
                let split = rest_a.iter().position(|&n| n == nodes_b[0]).unwrap();
                let (between, after_b) = (&rest_a[..split], &rest_a[split + nodes_b.len()..]);
                self.rings[ring_a] = nodes_b
                    .iter()
                    .chain(between)
                    .chain(&nodes_a)
                    .chain(after_b)
                    .copied()
                    .collect();
            } else {
                self.rings[ring_a] = nodes_b.iter().chain(&rest_a).copied().collect();
                self.rings[ring_b] = nodes_a.iter().chain(&rest_b).copied().collect();
            }
            let [a_first, a_last, b_first, b_last] = nodes([
                nodes_a[0],
                *nodes_a.last().unwrap(),
                nodes_b[0],
                *nodes_b.last().unwrap(),
            ]);
            // SAFETY: every node handed over lies in the arena under test.
            unsafe {
                self.arena
                    .swap_segments_unchecked(a_first, a_last, b_first, b_last)
            };
        }
    }

    proptest! {
        #[test]
        fn test_steps_match_the_model_of_rings_as_lists(
            (len, rings) in some_rings(),
            changes in proptest::collection::vec(change(), 0..24),
        ) {
            let mut harness = Harness::new(len, rings);
            prop_assert_eq!(rings_of(&harness.arena), canonical(&harness.rings));
            for change in &changes {
                harness.apply(change);
                assert_links_are_inverse(&harness.arena);
                prop_assert_eq!(rings_of(&harness.arena), canonical(&harness.rings), "after {:?}", change);
            }
        }

        #[test]
        fn test_next_and_prev_match_the_model_of_rings_as_lists((len, rings) in some_rings()) {
            let arena = arena_of(&rings, len);
            for node in (0..len).map(Node::new) {
                // SAFETY: every node below the length lies in the arena.
                let (next, prev) = unsafe { (arena.next_unchecked(node), arena.prev_unchecked(node)) };
                let ring = &rings[locate(&rings, node.index()).0];
                let place = ring.iter().position(|&n| n == node.index()).unwrap();
                prop_assert_eq!(next.index(), ring[(place + 1) % ring.len()]);
                prop_assert_eq!(prev.index(), ring[(place + ring.len() - 1) % ring.len()]);
            }
        }
    }
}
