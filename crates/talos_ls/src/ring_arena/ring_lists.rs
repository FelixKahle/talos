// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The rings of an arena as lists of node indices: the model the arena's tests check it against.

use super::arena::RingArena;
use super::node::Node;
use proptest::prelude::*;

/// The rings as lists of node indices, each in successor order.
pub(super) type Rings = Vec<Vec<usize>>;

/// The arena of `len` nodes whose rings are `rings`, built from rings of one by moving each node after the one
/// before it.
pub(super) fn arena_of(rings: &[Vec<usize>], len: usize) -> RingArena {
    let mut arena = RingArena::default();
    arena.reset_to_self_loops(len);
    for ring in rings {
        for pair in ring.windows(2) {
            arena.relocate_after(Node::new(pair[1]), Node::new(pair[0]));
        }
    }
    arena
}

/// The arena of the test fixtures: rings `[0, 1, 2, 3]`, `[4, 5, 6]` and `[7]`.
pub(super) fn three_rings() -> RingArena {
    arena_of(&[vec![0, 1, 2, 3], vec![4, 5, 6], vec![7]], 8)
}

/// The nodes of the ring of `start` along the successors, from `start`.
///
/// The walk is bounded by the arena's length: links that do not close into a ring fail an assertion.
pub(super) fn ring_of(arena: &RingArena, start: usize) -> Vec<usize> {
    let start = Node::new(start);
    let mut ring = vec![start.index()];
    let mut current = arena.next(start);
    while current != start {
        assert!(
            ring.len() < arena.len(),
            "the successors of {start} should close into a ring within {} nodes",
            arena.len()
        );
        ring.push(current.index());
        current = arena.next(current);
    }
    ring
}

/// Asserts that the predecessor of every node's successor, and the successor of its predecessor, is the node.
pub(super) fn assert_links_are_inverse(arena: &RingArena) {
    for node in (0..arena.len()).map(Node::new) {
        assert_eq!(
            arena.prev(arena.next(node)),
            node,
            "the predecessor of the successor of {node} should be {node}"
        );
        assert_eq!(
            arena.next(arena.prev(node)),
            node,
            "the successor of the predecessor of {node} should be {node}"
        );
    }
}

/// Each ring rotated to start at its least node, the rings in order: equal for equal cyclic orders.
pub(super) fn canonical(rings: &[Vec<usize>]) -> Rings {
    let mut rings: Rings = rings
        .iter()
        .filter(|ring| !ring.is_empty())
        .map(|ring| {
            let least = ring
                .iter()
                .enumerate()
                .min_by_key(|&(_, &node)| node)
                .map_or(0, |(place, _)| place);
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

/// The rings of `arena`, in canonical form.
pub(super) fn rings_of(arena: &RingArena) -> Rings {
    let mut seen = vec![false; arena.len()];
    let mut rings = Rings::new();
    for start in 0..arena.len() {
        if !seen[start] {
            let ring = ring_of(arena, start);
            for &node in &ring {
                seen[node] = true;
            }
            rings.push(ring);
        }
    }
    canonical(&rings)
}

/// The ring holding `node` and the node's place in it.
pub(super) fn locate(rings: &[Vec<usize>], node: usize) -> (usize, usize) {
    rings
        .iter()
        .enumerate()
        .find_map(|(ring, nodes)| {
            nodes
                .iter()
                .position(|&n| n == node)
                .map(|place| (ring, place))
        })
        .expect("every node should stand on one ring")
}

/// The ring of `node` rotated to start at it.
pub(super) fn rotated_to(rings: &[Vec<usize>], node: usize) -> Vec<usize> {
    let (ring, place) = locate(rings, node);
    let nodes = &rings[ring];
    nodes[place..]
        .iter()
        .chain(&nodes[..place])
        .copied()
        .collect()
}

/// A segment of `rings`: the ring of `first`, the `extra % ring length + 1` nodes from `first`, and the rest of the
/// ring after them.
pub(super) fn segment(
    rings: &[Vec<usize>],
    first: usize,
    extra: usize,
) -> (usize, Vec<usize>, Vec<usize>) {
    let (ring, _) = locate(rings, first);
    let rotated = rotated_to(rings, first);
    let length = extra % rotated.len() + 1;
    (ring, rotated[..length].to_vec(), rotated[length..].to_vec())
}

/// An arena of one to nine nodes, split into rings at random: its length and its rings.
pub(super) fn some_rings() -> impl Strategy<Value = (usize, Rings)> {
    (1usize..10).prop_flat_map(|len| {
        (
            Just((0..len).collect::<Vec<_>>()).prop_shuffle(),
            proptest::collection::vec(any::<bool>(), len),
        )
            .prop_map(move |(order, cuts)| {
                let mut rings: Rings = Vec::new();
                for (node, cut) in order.into_iter().zip(cuts) {
                    match rings.last_mut() {
                        Some(ring) if !cut => ring.push(node),
                        _ => rings.push(vec![node]),
                    }
                }
                (len, rings)
            })
    })
}
