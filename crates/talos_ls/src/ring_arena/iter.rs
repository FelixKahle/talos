// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The walks along a ring.

use super::node::Node;
use super::slot::Slot;
use std::iter::FusedIterator;
use typed_index_collections::TiSlice;

/// The nodes of a ring from a start along one direction, up to but not including a stop node.
///
/// The direction is the column the walk follows: the successors walk the ring forwards, the predecessors walk it
/// backwards. The walk yields at most as many nodes as the arena holds, so it ends even if the stop node is not on
/// the ring of the start: it then goes round that ring until it has yielded that many.
// Invariant: `column` is one column of the arena whose slots `current` and `stop` are, borrowed for as long as the
// walk lives, so its length is the arena's and every entry names one of its nodes (`RingArena`); `current` is so
// always a slot of that arena, and reading `column[current]` is in bounds.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct RingSequenceIter<'a> {
    /// The link of each node of the arena in the direction of the walk.
    column: &'a TiSlice<Node, Node>,
    /// The node to yield next.
    current: Slot,
    /// The node at which the walk ends, not yielded.
    stop: Slot,
    /// How many more nodes may be yielded at most; bounds the walk to the number of nodes of the arena.
    remaining: usize,
}

impl<'a> RingSequenceIter<'a> {
    /// The walk from `start` along `column` up to `stop`, where `column` is a column of the arena of both slots.
    pub(super) fn new(column: &'a TiSlice<Node, Node>, start: Slot, stop: Slot) -> Self {
        Self {
            column,
            current: start,
            stop,
            remaining: column.len(),
        }
    }

    /// The node the walk yields next, without yielding it.
    fn peek(&self) -> Option<Node> {
        (self.current != self.stop && self.remaining > 0).then_some(self.current.node())
    }
}

impl Iterator for RingSequenceIter<'_> {
    type Item = Node;

    #[inline]
    fn next(&mut self) -> Option<Node> {
        let node = self.peek()?;
        debug_assert!(
            node.index() < self.column.len(),
            "the walk's current node {node} should lie in the arena of {} nodes",
            self.column.len()
        );
        // SAFETY: `current` is a slot of the arena whose column this is (`RingSequenceIter`).
        let following = unsafe { *self.column.get_unchecked(node) };
        debug_assert!(
            following.index() < self.column.len(),
            "the link of {node} should name a node of the arena of {} nodes",
            self.column.len()
        );
        // SAFETY: every entry of the column names a node of the arena (`RingSequenceIter`, `RingArena`).
        self.current = unsafe { Slot::new_unchecked(following) };
        self.remaining -= 1;
        Some(node)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let upper = if self.peek().is_some() {
            self.remaining
        } else {
            0
        };
        (0, Some(upper))
    }
}

impl FusedIterator for RingSequenceIter<'_> {}

/// The links `(node, successor)` between consecutive nodes of a [`RingSequenceIter`] along the successors.
// Invariant: that of the sequence it walks; it reads no column of its own.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct RingLinkIter<'a> {
    /// The nodes whose consecutive pairs are the links.
    sequence: RingSequenceIter<'a>,
}

impl<'a> RingLinkIter<'a> {
    /// The links between consecutive nodes of `sequence`.
    pub(super) fn new(sequence: RingSequenceIter<'a>) -> Self {
        Self { sequence }
    }
}

impl Iterator for RingLinkIter<'_> {
    type Item = (Node, Node);

    #[inline]
    fn next(&mut self) -> Option<(Node, Node)> {
        let from = self.sequence.next()?;
        let to = self.sequence.peek()?;
        Some((from, to))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (_, upper) = self.sequence.size_hint();
        (0, upper.map(|nodes| nodes.saturating_sub(1)))
    }
}

impl FusedIterator for RingLinkIter<'_> {}

#[cfg(test)]
mod tests {
    use super::super::arena::RingArena;
    use super::super::ring_lists::{arena_of, locate, rotated_to, some_rings, three_rings};
    use super::*;
    use proptest::prelude::*;

    fn indices(nodes: impl Iterator<Item = Node>) -> Vec<usize> {
        nodes.map(Node::index).collect()
    }

    fn index_pairs(links: impl Iterator<Item = (Node, Node)>) -> Vec<(usize, usize)> {
        links.map(|(from, to)| (from.index(), to.index())).collect()
    }

    /// The nodes a walk from `start` along `ring`, which holds it, yields up to `stop` in an arena of `len` nodes:
    /// up to `stop` if it is on the ring, otherwise round the ring until `len` nodes are yielded.
    fn expected_walk(ring: &[usize], start: usize, stop: usize, len: usize) -> Vec<usize> {
        let place = ring.iter().position(|&n| n == start).unwrap();
        ring.iter()
            .cycle()
            .skip(place)
            .take(len)
            .copied()
            .take_while(|&n| n != stop)
            .collect()
    }

    /// Asserts that at every point of the walk, which yields at most `bound` items, the hint bounds the items left.
    fn assert_hints_bound_what_is_left<I: Iterator + Clone>(mut walk: I, bound: usize) {
        for _ in 0..=bound {
            let (lower, upper) = walk.size_hint();
            let left = walk.clone().count();
            assert!(
                lower <= left,
                "the lower hint {lower} should not exceed the {left} items left"
            );
            assert!(
                upper.is_none_or(|upper| left <= upper),
                "the upper hint {upper:?} should cover {left} items"
            );
            if walk.next().is_none() {
                return;
            }
        }
        panic!("the walk should end within {bound} items");
    }

    #[test]
    fn test_sequence_walks_up_to_the_stop() {
        let arena = three_rings();
        assert_eq!(
            indices(arena.sequence(Node::new(1), Node::new(0))),
            [1, 2, 3]
        );
        assert_eq!(
            indices(arena.sequence(Node::new(4), Node::new(4))),
            Vec::<usize>::new()
        );
        assert_eq!(
            indices(arena.sequence(Node::new(7), Node::new(7))),
            Vec::<usize>::new()
        );
    }

    #[test]
    fn test_sequence_rev_walks_back_to_the_stop() {
        let arena = three_rings();
        assert_eq!(
            indices(arena.sequence_rev(Node::new(3), Node::new(0))),
            [3, 2, 1]
        );
        assert_eq!(
            indices(arena.sequence_rev(Node::new(6), Node::new(4))),
            [6, 5]
        );
        assert_eq!(
            indices(arena.sequence_rev(Node::new(5), Node::new(5))),
            Vec::<usize>::new()
        );
    }

    #[test]
    fn test_sequence_past_a_stop_on_another_ring_goes_round_until_every_node_is_counted() {
        let arena = three_rings();
        assert_eq!(
            indices(arena.sequence(Node::new(4), Node::new(0))),
            [4, 5, 6, 4, 5, 6, 4, 5]
        );
    }

    #[test]
    fn test_links_join_consecutive_nodes_of_the_sequence() {
        let arena = three_rings();
        assert_eq!(
            index_pairs(arena.links(Node::new(0), Node::new(3))),
            [(0, 1), (1, 2)]
        );
        assert_eq!(
            index_pairs(arena.links(Node::new(5), Node::new(4))),
            [(5, 6)]
        );
        assert_eq!(index_pairs(arena.links(Node::new(3), Node::new(0))), []);
        assert_eq!(index_pairs(arena.links(Node::new(4), Node::new(4))), []);
    }

    #[test]
    fn test_size_hint_of_a_finished_walk_is_zero() {
        let arena = three_rings();
        let mut walk = arena.sequence(Node::new(1), Node::new(0));
        assert_eq!(walk.size_hint(), (0, Some(8)));
        walk.by_ref().for_each(drop);
        assert_eq!(walk.size_hint(), (0, Some(0)));
        let mut links = arena.links(Node::new(1), Node::new(0));
        links.by_ref().for_each(drop);
        assert_eq!(links.size_hint(), (0, Some(0)));
    }

    #[test]
    #[should_panic(
        expected = "called `RingArena::sequence` with Node 99, outside an arena of 8 nodes"
    )]
    fn test_sequence_refuses_a_node_outside() {
        let _ = three_rings().sequence(Node::new(99), Node::new(0));
    }

    #[test]
    #[should_panic(
        expected = "called `RingArena::sequence_rev` with Node 8, outside an arena of 8 nodes"
    )]
    fn test_sequence_rev_refuses_a_node_outside() {
        let _ = three_rings().sequence_rev(Node::new(0), Node::new(8));
    }

    #[test]
    #[should_panic(
        expected = "called `RingArena::links` with Node 99, outside an arena of 8 nodes"
    )]
    fn test_links_refuses_a_node_outside() {
        let _ = three_rings().links(Node::new(99), Node::new(0));
    }

    /// The walks of `arena` from `start` to `stop`, forwards and backwards, each checked and unchecked.
    fn walks(arena: &RingArena, start: Node, stop: Node) -> [(Vec<usize>, Vec<usize>); 2] {
        // SAFETY: the tests hand over only nodes of the arena.
        let (forwards, backwards) = unsafe {
            (
                arena.sequence_unchecked(start, stop),
                arena.sequence_rev_unchecked(start, stop),
            )
        };
        [
            (indices(arena.sequence(start, stop)), indices(forwards)),
            (indices(arena.sequence_rev(start, stop)), indices(backwards)),
        ]
    }

    proptest! {
        #[test]
        fn test_walks_follow_the_model_ring(
            (len, rings) in some_rings(),
            start_at in 0usize..64,
            stop_at in 0usize..64,
            stop_elsewhere in any::<bool>(),
        ) {
            let arena = arena_of(&rings, len);
            let start = start_at % len;
            let ring = &rings[locate(&rings, start).0];
            let stop = if stop_elsewhere { stop_at % len } else { ring[stop_at % ring.len()] };
            let backwards: Vec<usize> = rotated_to(&rings, start)[1..].iter().rev().copied().collect();
            let backwards: Vec<usize> = std::iter::once(start).chain(backwards).collect();

            let [(forwards_checked, forwards_unchecked), (backwards_checked, backwards_unchecked)] =
                walks(&arena, Node::new(start), Node::new(stop));
            let expected = expected_walk(ring, start, stop, len);
            prop_assert_eq!(&forwards_checked, &expected);
            prop_assert_eq!(&forwards_unchecked, &expected);
            let expected_backwards = expected_walk(&backwards, start, stop, len);
            prop_assert_eq!(&backwards_checked, &expected_backwards);
            prop_assert_eq!(&backwards_unchecked, &expected_backwards);

            let pairs: Vec<(usize, usize)> = expected.windows(2).map(|pair| (pair[0], pair[1])).collect();
            let (start, stop) = (Node::new(start), Node::new(stop));
            prop_assert_eq!(index_pairs(arena.links(start, stop)), pairs);

            assert_hints_bound_what_is_left(arena.sequence(start, stop), len);
            assert_hints_bound_what_is_left(arena.sequence_rev(start, stop), len);
            assert_hints_bound_what_is_left(arena.links(start, stop), len);
        }
    }
}
