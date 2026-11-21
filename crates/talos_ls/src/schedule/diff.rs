// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The record of what a sequence of moves changed in a schedule graph.

use crate::ring_arena::Node;
use std::iter::FusedIterator;
use talos_model::{BerthIndex, VesselIndex};

// ----------------------------------------------------------------
// ScheduleGraphDiffEdge
// ----------------------------------------------------------------

/// A raw link change between two nodes in the schedule graph.
///
/// Both `from` and `to` are raw arena nodes. They may refer to real vessels
/// or berth sentinels; interpretation is left to the consumer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScheduleGraphDiffEdge {
    pub from: Node,
    pub to: Node,
}

impl std::fmt::Display for ScheduleGraphDiffEdge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Node({}) -> Node({})",
            self.from.index(),
            self.to.index()
        )
    }
}

// ----------------------------------------------------------------
// DiffEdgeIter
// ----------------------------------------------------------------

/// Iterator over broken or created links in a `ScheduleGraphDiff`.
pub struct DiffEdgeIter<'a> {
    inner: std::iter::Zip<std::slice::Iter<'a, Node>, std::slice::Iter<'a, Node>>,
}

impl<'a> Iterator for DiffEdgeIter<'a> {
    type Item = ScheduleGraphDiffEdge;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner
            .next()
            .map(|(&f, &t)| ScheduleGraphDiffEdge { from: f, to: t })
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl FusedIterator for DiffEdgeIter<'_> {}

// ----------------------------------------------------------------
// DiffReallocationIter
// ----------------------------------------------------------------

/// Iterator over vessel reallocations between berths.
pub struct DiffReallocationIter<'a> {
    vessels: std::slice::Iter<'a, VesselIndex>,
    originals: std::slice::Iter<'a, BerthIndex>,
    targets: std::slice::Iter<'a, BerthIndex>,
}

impl<'a> Iterator for DiffReallocationIter<'a> {
    type Item = (VesselIndex, BerthIndex, BerthIndex);

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        let v = self.vessels.next()?;
        let o = self.originals.next()?;
        let t = self.targets.next()?;
        Some((*v, *o, *t))
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.vessels.size_hint()
    }
}

// ----------------------------------------------------------------
// ScheduleGraphDiff
// ----------------------------------------------------------------

/// A comprehensive diff structure that captures all changes between two schedule graphs
/// states, including broken and created links as well as vessel reallocations between berths.
#[derive(Debug, Clone)]
pub struct ScheduleGraphDiff {
    broken_from: Vec<Node>,
    broken_to: Vec<Node>,
    created_from: Vec<Node>,
    created_to: Vec<Node>,

    reallocated_vessels: Vec<VesselIndex>,
    original_berths: Vec<BerthIndex>,
    target_berths: Vec<BerthIndex>,
}

impl Default for ScheduleGraphDiff {
    fn default() -> Self {
        Self::new()
    }
}

impl ScheduleGraphDiff {
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            broken_from: Vec::new(),
            broken_to: Vec::new(),
            created_from: Vec::new(),
            created_to: Vec::new(),
            reallocated_vessels: Vec::new(),
            original_berths: Vec::new(),
            target_berths: Vec::new(),
        }
    }

    #[inline(always)]
    pub fn clear(&mut self) {
        self.broken_from.clear();
        self.broken_to.clear();
        self.created_from.clear();
        self.created_to.clear();
        self.reallocated_vessels.clear();
        self.original_berths.clear();
        self.target_berths.clear();
    }

    /// Overwrites this diff with the contents of `other`, reusing
    /// existing heap allocations.
    #[inline(always)]
    pub fn overwrite_from_graph_diff(&mut self, other: &Self) {
        self.broken_from.clear();
        self.broken_from.extend_from_slice(&other.broken_from);
        self.broken_to.clear();
        self.broken_to.extend_from_slice(&other.broken_to);

        self.created_from.clear();
        self.created_from.extend_from_slice(&other.created_from);
        self.created_to.clear();
        self.created_to.extend_from_slice(&other.created_to);

        self.reallocated_vessels.clear();
        self.reallocated_vessels
            .extend_from_slice(&other.reallocated_vessels);
        self.original_berths.clear();
        self.original_berths
            .extend_from_slice(&other.original_berths);
        self.target_berths.clear();
        self.target_berths.extend_from_slice(&other.target_berths);
    }

    #[inline(always)]
    pub fn push_link_broken(&mut self, from: Node, to: Node) {
        self.broken_from.push(from);
        self.broken_to.push(to);
    }

    #[inline(always)]
    pub fn push_link_created(&mut self, from: Node, to: Node) {
        self.created_from.push(from);
        self.created_to.push(to);
    }

    #[inline(always)]
    pub fn push_reallocation(&mut self, v: VesselIndex, from: BerthIndex, to: BerthIndex) {
        self.reallocated_vessels.push(v);
        self.original_berths.push(from);
        self.target_berths.push(to);
    }

    #[inline(always)]
    pub fn link_broken_count(&self) -> usize {
        self.broken_from.len()
    }

    #[inline(always)]
    pub fn link_created_count(&self) -> usize {
        self.created_from.len()
    }

    #[inline(always)]
    pub fn broken_links(&self) -> DiffEdgeIter<'_> {
        DiffEdgeIter {
            inner: self.broken_from.iter().zip(self.broken_to.iter()),
        }
    }

    #[inline(always)]
    pub fn created_links(&self) -> DiffEdgeIter<'_> {
        DiffEdgeIter {
            inner: self.created_from.iter().zip(self.created_to.iter()),
        }
    }

    #[inline(always)]
    pub fn reallocations(&self) -> DiffReallocationIter<'_> {
        DiffReallocationIter {
            vessels: self.reallocated_vessels.iter(),
            originals: self.original_berths.iter(),
            targets: self.target_berths.iter(),
        }
    }
}
