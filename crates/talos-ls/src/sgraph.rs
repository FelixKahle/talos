// Copyright (c) 2026 Felix Kahle.
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

//! High-performance, allocation-free topological representation of vessel schedules.
//!
//! This module provides the `ScheduleGraph`, which acts as the core state representation
//! (genotype) for Local Search algorithms (e.g., Simulated Annealing, Tabu Search, ALNS)
//! applied to the Dynamic Berth Allocation Problem (DBAP) or Vehicle Routing Problem (VRP) variants.
//!
//! # Architecture
//!
//! `ScheduleGraph` layers domain semantics on top of a `RingArena`. The arena handles
//! all raw topology (O(1) swaps, relocations, reversals), while the graph manages:
//!
//! - **Sentinel convention**: The arena's index space is partitioned into real vessels
//!   `[0, num_vessels)` and berth sentinels `[num_vessels, num_vessels + num_berths)`.
//! - **Berth tracking**: `vessel_berth[v]` and `berth_vessel_count[b]` are kept in sync
//!   with the topology by all mutation methods.
//!
//! # Sentinels and Memory Layout
//!
//! Every berth is represented as an independent, circular ring containing exactly one sentinel
//! node and zero or more real vessel nodes. An empty berth is simply a sentinel node whose
//! `prev` and `next` pointers point to itself.
//!
//! ```text
//! Index Space: [ 0, 1, ..., N-1 | N, N+1, ..., N+B-1 ]
//!              |__Real Vessels__|___Berth Sentinels__|
//! ```

use std::iter::FusedIterator;
use talos_core::container::rarena::{
    Node, RingArena, RingEdgeIter, RingSequenceIter, RingSequenceRevIter,
};
use talos_model::index::{BerthIndex, VesselIndex};

// ----------------------------------------------------------------
// VesselSequenceIter
// ----------------------------------------------------------------

/// Iterator over a vessel sequence assigned to a berth, from first to last.
#[derive(Clone, PartialEq, Eq)]
pub struct VesselSequenceIter<'a> {
    inner: RingSequenceIter<'a>,
    num_vessels: usize,
}

impl<'a> Iterator for VesselSequenceIter<'a> {
    type Item = VesselIndex;

    #[inline(always)]
    fn next(&mut self) -> Option<VesselIndex> {
        let raw_node = self.inner.next()?;
        let raw = raw_node.index();
        debug_assert!(
            raw < self.num_vessels,
            "VesselSequenceIter yielded a sentinel index: {} >= {}",
            raw,
            self.num_vessels
        );
        Some(VesselIndex::new(raw))
    }
}

impl FusedIterator for VesselSequenceIter<'_> {}

// ----------------------------------------------------------------
// VesselSequenceRevIter
// ----------------------------------------------------------------

/// Reverse iterator over the vessels assigned to a berth, from last to first.
#[derive(Clone, PartialEq, Eq)]
pub struct VesselSequenceRevIter<'a> {
    inner: RingSequenceRevIter<'a>,
    num_vessels: usize,
}

impl<'a> Iterator for VesselSequenceRevIter<'a> {
    type Item = VesselIndex;

    #[inline(always)]
    fn next(&mut self) -> Option<VesselIndex> {
        let raw_node = self.inner.next()?;
        let raw = raw_node.index();
        debug_assert!(raw < self.num_vessels);

        Some(VesselIndex::new(raw))
    }
}

impl FusedIterator for VesselSequenceRevIter<'_> {}

// ----------------------------------------------------------------
// ScheduleGraphEdge
// ----------------------------------------------------------------

/// A directed edge between two vessels on the same berth.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ScheduleGraphEdge {
    pub from: VesselIndex,
    pub to: VesselIndex,
}

impl std::fmt::Display for ScheduleGraphEdge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Edge(V{} -> V{})", self.from.get(), self.to.get())
    }
}

// ----------------------------------------------------------------
// ScheduleGraphFullEdge
// ----------------------------------------------------------------

/// A directed edge between two vessels, annotated with the berth.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ScheduleGraphFullEdge {
    pub from: VesselIndex,
    pub to: VesselIndex,
    pub on_berth: BerthIndex,
}

impl std::fmt::Display for ScheduleGraphFullEdge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Edge(V{} -> V{} on Berth {})",
            self.from.get(),
            self.to.get(),
            self.on_berth.get()
        )
    }
}

// ----------------------------------------------------------------
// BerthEdgeIter
// ----------------------------------------------------------------

/// Iterator over the edges (adjacent vessel pairs) within a single berth.
#[derive(Clone, Debug)]
pub struct BerthEdgeIter<'a> {
    inner: RingEdgeIter<'a>,
    num_vessels: usize,
}

impl<'a> Iterator for BerthEdgeIter<'a> {
    type Item = ScheduleGraphEdge;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        let (from_node, to_node) = self.inner.next()?;
        let from_raw = from_node.index();
        let to_raw = to_node.index();
        debug_assert!(from_raw < self.num_vessels && to_raw < self.num_vessels);
        Some(ScheduleGraphEdge {
            from: VesselIndex::new(from_raw),
            to: VesselIndex::new(to_raw),
        })
    }
}

impl FusedIterator for BerthEdgeIter<'_> {}

// ----------------------------------------------------------------
// AllEdgeIter
// ----------------------------------------------------------------

/// Iterator over all vessel-to-vessel edges across all berths.
#[derive(Clone, Debug)]
pub struct AllEdgeIter<'a> {
    arena: &'a RingArena,
    vessel_berth: &'a [BerthIndex],
    num_vessels: usize,
    current_vessel: usize,
}

impl<'a> Iterator for AllEdgeIter<'a> {
    type Item = ScheduleGraphFullEdge;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        while self.current_vessel < self.num_vessels {
            let from = self.current_vessel;
            self.current_vessel += 1;

            let to = unsafe { self.arena.next_unchecked(Node::new(from)) };

            if to.index() < self.num_vessels {
                let on_berth = *unsafe { self.vessel_berth.get_unchecked(from) };
                return Some(ScheduleGraphFullEdge {
                    from: VesselIndex::new(from),
                    to: VesselIndex::new(to.index()),
                    on_berth,
                });
            }
        }
        None
    }
}

impl FusedIterator for AllEdgeIter<'_> {}

// ----------------------------------------------------------------
// ScheduleGraph
// ----------------------------------------------------------------

/// Per-berth vessel ordering — the genotype of the local search.
///
/// `ScheduleGraph` maintains the explicit sequencing of vessels assigned to multiple berths.
/// It is built on top of a [`RingArena`] which handles all raw topology operations.
///
/// # Public API
///
/// All public methods use [`VesselIndex`] and [`BerthIndex`] exclusively. The sentinel
/// convention and arena internals are hidden from downstream consumers.
#[derive(Clone)]
pub struct ScheduleGraph {
    /// The underlying topology.
    arena: RingArena,

    /// O(1) lookup: the berth to which each vessel is assigned.
    /// Length: `num_vessels + num_berths`. Padded so sentinels map to their own berth.
    vessel_berth: Vec<BerthIndex>,

    /// O(1) lookup: the number of vessels assigned to each berth.
    /// Length: `num_berths`.
    berth_vessel_count: Vec<usize>,

    /// Total logical vessels. Used as the offset to calculate sentinel indices.
    num_vessels: usize,

    /// Total logical berths.
    num_berths: usize,
}

impl PartialEq for ScheduleGraph {
    fn eq(&self, other: &Self) -> bool {
        if self.num_berths != other.num_berths || self.num_vessels != other.num_vessels {
            return false;
        }
        for berth_idx in 0..self.num_berths {
            let berth = BerthIndex::new(berth_idx);
            if !self
                .vessel_sequence_iter(berth)
                .eq(other.vessel_sequence_iter(berth))
            {
                return false;
            }
        }
        true
    }
}

impl Eq for ScheduleGraph {}

impl std::fmt::Debug for ScheduleGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        struct Sequences<'a>(&'a ScheduleGraph);
        impl<'a> std::fmt::Debug for Sequences<'a> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                let mut map = f.debug_map();
                for berth_idx in 0..self.0.num_berths {
                    let berth = BerthIndex::new(berth_idx);
                    let seq: Vec<_> = self.0.vessel_sequence_iter(berth).collect();
                    map.entry(&berth, &seq);
                }
                map.finish()
            }
        }
        f.debug_struct("ScheduleGraph")
            .field("num_vessels", &self.num_vessels)
            .field("num_berths", &self.num_berths)
            .field("sequences", &Sequences(self))
            .finish()
    }
}

impl std::fmt::Display for ScheduleGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "ScheduleGraph (Berths: {}, Total Vessels: {})",
            self.num_berths, self.num_vessels
        )?;
        for berth_idx in 0..self.num_berths {
            let berth = BerthIndex::new(berth_idx);
            write!(f, "  Berth {}: ", berth_idx)?;
            let mut iter = self.vessel_sequence_iter(berth);
            if let Some(first) = iter.next() {
                write!(f, "V{}", first.get())?;
                for v in iter {
                    write!(f, " -> V{}", v.get())?;
                }
                writeln!(f)?;
            } else {
                writeln!(f, "(empty)")?;
            }
        }
        Ok(())
    }
}

impl std::hash::Hash for ScheduleGraph {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.num_berths.hash(state);
        for berth_idx in 0..self.num_berths {
            let berth = BerthIndex::new(berth_idx);
            for vessel in self.vessel_sequence_iter(berth) {
                vessel.hash(state);
            }
        }
    }
}

impl ScheduleGraph {
    // ----------------------------------------------------------------
    // Constructors
    // ----------------------------------------------------------------

    /// Creates a new `ScheduleGraph` from parallel slices of berths and start times.
    #[inline]
    pub fn from_slices<T>(berths: &[BerthIndex], start_times: &[T], num_berths: usize) -> Self
    where
        T: Ord,
    {
        let mut graph = Self {
            arena: RingArena::new(Vec::new(), Vec::new()),
            vessel_berth: Vec::new(),
            berth_vessel_count: Vec::new(),
            num_vessels: 0,
            num_berths: 0,
        };
        graph.overwrite_from_slices(berths, start_times, num_berths);
        graph
    }

    // ----------------------------------------------------------------
    // Overwrite methods
    // ----------------------------------------------------------------

    /// Overwrites the current `ScheduleGraph` with data from parallel slices.
    ///
    /// This is a convenience method that allocates a temporary scratchpad.
    /// For zero-allocation scenarios, use `overwrite_from_slices_in` and allocate
    /// a scratchpad with capacity `berths.len()` outside the method.
    ///
    /// # Panics
    ///
    /// Panics if `berths.len() != start_times.len()`, or if any assigned berth
    /// in the slice is out of bounds.
    #[inline]
    pub fn overwrite_from_slices<T>(
        &mut self,
        berths: &[BerthIndex],
        start_times: &[T],
        num_berths: usize,
    ) where
        T: Ord,
    {
        let mut scratchpad = Vec::with_capacity(berths.len());
        self.overwrite_from_slices_in(berths, start_times, num_berths, &mut scratchpad);
    }

    /// Overwrites the current `ScheduleGraph` with data from parallel slices,
    /// using a provided scratchpad to guarantee zero allocations.
    /// To totally avoid allocations, the caller must ensure `scratchpad.capacity() >= berths.len()`,
    /// though the method will not panic if this is not the case (it will just reallocate the scratchpad).
    ///
    /// # Panics
    ///
    /// Panics if `berths.len() != start_times.len()`, or if any assigned berth
    /// in the slice is out of bounds.
    pub fn overwrite_from_slices_in<T>(
        &mut self,
        berths: &[BerthIndex],
        start_times: &[T],
        num_berths: usize,
        scratchpad: &mut Vec<usize>,
    ) where
        T: Ord,
    {
        assert_eq!(
            berths.len(),
            start_times.len(),
            "called `ScheduleGraph::overwrite_from_slices_in` with mismatched slice lengths"
        );

        self.num_vessels = berths.len();
        self.num_berths = num_berths;
        let total_nodes = self.num_vessels + self.num_berths;

        self.arena.reset_to_self_loops(total_nodes);

        self.vessel_berth.clear();
        self.vessel_berth.resize(total_nodes, BerthIndex::new(0));
        self.berth_vessel_count.clear();
        self.berth_vessel_count.resize(self.num_berths, 0);

        for (vessel, &berth) in berths.iter().enumerate() {
            assert!(berth < self.num_berths);

            self.vessel_berth[vessel] = berth;
            self.berth_vessel_count[berth.get()] += 1;
        }

        for berth_idx in 0..self.num_berths {
            let sentinel = self.num_vessels + berth_idx;
            self.vessel_berth[sentinel] = BerthIndex::new(berth_idx);
        }

        if self.num_vessels == 0 {
            return;
        }

        scratchpad.clear();
        scratchpad.extend(0..self.num_vessels);
        scratchpad.sort_unstable_by(|&left, &right| {
            berths[left]
                .cmp(&berths[right])
                .then_with(|| start_times[left].cmp(&start_times[right]))
        });

        for &vessel_raw in scratchpad.iter() {
            let vessel = VesselIndex::new(vessel_raw);
            let berth = berths[vessel_raw];

            unsafe {
                // Since old_berth == new_berth (set in step 2), this just wires the topology
                // without double-counting the vessel tracking.
                self.relocate_to_tail_unchecked(vessel, berth);
            }
        }
    }

    /// Overwrites the current `ScheduleGraph` with data from another graph.
    #[inline]
    pub fn overwrite_from_graph(&mut self, other: &ScheduleGraph) {
        self.arena.overwrite_from_arena(&other.arena);
        self.vessel_berth.clone_from(&other.vessel_berth);
        self.berth_vessel_count
            .clone_from(&other.berth_vessel_count);
        self.num_vessels = other.num_vessels;
        self.num_berths = other.num_berths;
    }

    // ----------------------------------------------------------------
    // Accessors
    // ----------------------------------------------------------------

    /// Returns the raw node representing the boundary before the first vessel of a berth.
    #[inline(always)]
    pub fn berth_head_boundary_node(&self, berth: BerthIndex) -> Node {
        self.sentinel(berth)
    }

    /// Returns the raw node index representing the boundary after the last vessel of a berth.
    #[inline(always)]
    pub fn berth_tail_boundary_node(&self, berth: BerthIndex) -> Node {
        self.sentinel(berth)
    }

    /// Returns the raw node of the given vessel.
    #[inline(always)]
    pub fn vessel_node(&self, vessel: VesselIndex) -> Node {
        vessel.into()
    }

    /// Returns the node following `node`.
    #[inline(always)]
    pub fn next_node(&self, node: Node) -> Node {
        debug_assert!(node.index() < self.arena.len());

        self.arena.next(node)
    }

    /// Returns the node following `node`.
    ///
    /// # Safety
    ///
    /// `node.index()` must be `< self.arena.len()`.
    #[inline(always)]
    pub unsafe fn next_node_unchecked(&self, node: Node) -> Node {
        debug_assert!(node.index() < self.arena.len());

        unsafe { self.arena.next_unchecked(node) }
    }

    /// Returns the node preceding `node`.
    ///
    /// # Safety
    ///
    /// `node.index()` must be `< self.arena.len()`.
    #[inline(always)]
    pub unsafe fn prev_node_unchecked(&self, node: Node) -> Node {
        debug_assert!(node.index() < self.arena.len());

        unsafe { self.arena.prev_unchecked(node) }
    }

    /// Returns the node preceding `node`.
    #[inline(always)]
    pub fn prev_node(&self, node: Node) -> Node {
        debug_assert!(node.index() < self.arena.len());

        self.arena.prev(node)
    }

    /// Returns the raw arena index for a berth's sentinel node.
    #[inline(always)]
    pub fn sentinel(&self, berth: BerthIndex) -> Node {
        debug_assert!(berth < self.num_berths);

        Node::new(self.num_vessels + berth.get())
    }

    /// Returns the total number of logical berths.
    #[inline(always)]
    pub fn num_berths(&self) -> usize {
        self.num_berths
    }

    /// Returns the total number of logical vessels.
    #[inline(always)]
    pub fn num_vessels(&self) -> usize {
        self.num_vessels
    }

    /// Returns true if the given berth has no vessels assigned.
    ///
    /// # Panics
    ///
    /// Panics if `berth` is out of bounds.
    #[inline]
    pub fn is_empty(&self, berth: BerthIndex) -> bool {
        assert!(berth < self.num_berths);

        self.berth_vessel_count[berth.get()] == 0
    }

    /// Returns the berth to which the given vessel is currently assigned.
    ///
    /// # Panics
    ///
    /// Panics if `vessel` is out of bounds.
    #[inline]
    pub fn vessel_berth(&self, vessel: VesselIndex) -> BerthIndex {
        assert!(vessel < self.num_vessels);

        self.vessel_berth[vessel.get()]
    }

    /// Returns the berth to which the given vessel is currently assigned.
    ///
    /// # Safety
    ///
    /// `vessel.get()` must be `< self.num_vessels`.
    #[inline(always)]
    pub unsafe fn vessel_berth_unchecked(&self, vessel: VesselIndex) -> BerthIndex {
        debug_assert!(vessel < self.num_vessels);

        *unsafe { self.vessel_berth.get_unchecked(vessel.get()) }
    }

    /// Returns the number of vessels currently assigned to the given berth.
    ///
    /// # Panics
    ///
    /// Panics if `berth` is out of bounds.
    #[inline]
    pub fn vessel_count(&self, berth: BerthIndex) -> usize {
        assert!(berth < self.num_berths);
        self.berth_vessel_count[berth.get()]
    }

    /// Returns the number of vessels currently assigned to the given berth.
    ///
    /// # Safety
    ///
    /// `berth.get()` must be `< self.num_berths`.
    #[inline(always)]
    pub unsafe fn vessel_count_unchecked(&self, berth: BerthIndex) -> usize {
        debug_assert!(berth < self.num_berths);

        *unsafe { self.berth_vessel_count.get_unchecked(berth.get()) }
    }

    /// Returns the first vessel in the given berth, or `None` if empty.
    ///
    /// # Panics
    ///
    /// Panics if `berth` is out of bounds.
    #[inline]
    pub fn first_vessel(&self, berth: BerthIndex) -> Option<VesselIndex> {
        assert!(berth < self.num_berths);

        let next_of_sentinel = self.arena.next(self.sentinel(berth));
        if next_of_sentinel.index() < self.num_vessels {
            Some(VesselIndex::new(next_of_sentinel.index()))
        } else {
            None
        }
    }

    /// Returns the last vessel in the given berth, or `None` if empty.
    ///
    /// # Panics
    ///
    /// Panics if `berth` is out of bounds.
    #[inline]
    pub fn last_vessel(&self, berth: BerthIndex) -> Option<VesselIndex> {
        assert!(berth < self.num_berths);

        let prev_of_sentinel = self.arena.prev(self.sentinel(berth));
        if prev_of_sentinel.index() < self.num_vessels {
            Some(VesselIndex::new(prev_of_sentinel.index()))
        } else {
            None
        }
    }

    /// Returns the vessel immediately preceding `vessel`, or `None` if at head.
    ///
    /// # Panics
    ///
    /// Panics if `vessel` is out of bounds.
    #[inline]
    pub fn vessel_predecessor(&self, vessel: VesselIndex) -> Option<VesselIndex> {
        assert!(vessel < self.num_vessels);

        let pred = self.arena.prev(vessel.into());
        if pred.index() < self.num_vessels {
            Some(VesselIndex::new(pred.index()))
        } else {
            None
        }
    }

    /// Returns the vessel immediately preceding `vessel`, or `None` if at head.
    ///
    /// # Safety
    ///
    /// `vessel.get()` must be `< self.num_vessels`.
    #[inline(always)]
    pub unsafe fn vessel_predecessor_unchecked(&self, vessel: VesselIndex) -> Option<VesselIndex> {
        debug_assert!(vessel < self.num_vessels);

        let pred = unsafe { self.arena.prev_unchecked(vessel.into()) };
        if pred.index() < self.num_vessels {
            Some(VesselIndex::new(pred.index()))
        } else {
            None
        }
    }

    /// Returns the vessel immediately following `vessel`, or `None` if at tail.
    ///
    /// # Panics
    ///
    /// Panics if `vessel` is out of bounds.
    #[inline]
    pub fn vessel_successor(&self, vessel: VesselIndex) -> Option<VesselIndex> {
        assert!(vessel < self.num_vessels);

        let succ = self.arena.next(vessel.into());
        if succ.index() < self.num_vessels {
            Some(VesselIndex::new(succ.index()))
        } else {
            None
        }
    }

    /// Returns the vessel immediately following `vessel`, or `None` if at tail.
    ///
    /// # Safety
    ///
    /// `vessel.get()` must be `< self.num_vessels`.
    #[inline(always)]
    pub unsafe fn vessel_successor_unchecked(&self, vessel: VesselIndex) -> Option<VesselIndex> {
        debug_assert!(vessel < self.num_vessels);

        let succ = unsafe { self.arena.next_unchecked(vessel.into()) };
        if succ.index() < self.num_vessels {
            Some(VesselIndex::new(succ.index()))
        } else {
            None
        }
    }

    /// Returns an iterator over the vessels assigned to the given berth, in order.
    ///
    /// # Panics
    ///
    /// Panics if `berth` is out of bounds.
    #[inline]
    pub fn vessel_sequence_iter(&self, berth: BerthIndex) -> VesselSequenceIter<'_> {
        assert!(berth < self.num_berths);

        let sentinel = self.sentinel(berth);
        let start = self.arena.next(sentinel);

        VesselSequenceIter {
            inner: unsafe { self.arena.sequence_iter_unchecked(start, sentinel) },
            num_vessels: self.num_vessels,
        }
    }

    /// Returns a reverse iterator over the vessels assigned to the given berth.
    ///
    /// # Panics
    ///
    /// Panics if `berth` is out of bounds.
    #[inline]
    pub fn vessel_sequence_rev_iter(&self, berth: BerthIndex) -> VesselSequenceRevIter<'_> {
        assert!(berth < self.num_berths);

        let sentinel = self.sentinel(berth);
        let start = self.arena.prev(sentinel);

        VesselSequenceRevIter {
            inner: unsafe { self.arena.sequence_rev_iter_unchecked(start, sentinel) },
            num_vessels: self.num_vessels,
        }
    }

    /// # Safety
    ///
    /// `berth.get()` must be `< self.num_berths`.
    #[inline(always)]
    pub unsafe fn vessel_sequence_iter_unchecked(
        &self,
        berth: BerthIndex,
    ) -> VesselSequenceIter<'_> {
        debug_assert!(berth < self.num_berths);

        let sentinel = self.sentinel(berth);
        let start = unsafe { self.arena.next_unchecked(sentinel) };

        VesselSequenceIter {
            inner: unsafe { self.arena.sequence_iter_unchecked(start, sentinel) },
            num_vessels: self.num_vessels,
        }
    }

    /// # Safety
    ///
    /// `berth.get()` must be `< self.num_berths`.
    #[inline(always)]
    pub unsafe fn vessel_sequence_rev_iter_unchecked(
        &self,
        berth: BerthIndex,
    ) -> VesselSequenceRevIter<'_> {
        debug_assert!(berth < self.num_berths);

        let sentinel = self.sentinel(berth);
        let start = unsafe { self.arena.prev_unchecked(sentinel) };

        VesselSequenceRevIter {
            inner: unsafe { self.arena.sequence_rev_iter_unchecked(start, sentinel) },
            num_vessels: self.num_vessels,
        }
    }

    /// Returns an iterator over all edges (adjacent vessel pairs) within a berth.
    ///
    /// # Panics
    ///
    /// Panics if `berth` is out of bounds.
    #[inline]
    pub fn berth_edges(&self, berth: BerthIndex) -> BerthEdgeIter<'_> {
        assert!(berth < self.num_berths);

        let sentinel = self.sentinel(berth);
        let start = self.arena.next(sentinel);

        BerthEdgeIter {
            inner: unsafe { self.arena.edge_iter_unchecked(start, sentinel) },
            num_vessels: self.num_vessels,
        }
    }

    /// Returns an iterator over all vessel-to-vessel edges across all berths.
    #[inline(always)]
    pub fn all_edges(&self) -> AllEdgeIter<'_> {
        AllEdgeIter {
            arena: &self.arena,
            vessel_berth: &self.vessel_berth,
            num_vessels: self.num_vessels,
            current_vessel: 0,
        }
    }

    // ----------------------------------------------------------------
    // Mutators
    // ----------------------------------------------------------------

    /// Swaps the positions of two vessels.
    ///
    /// # Panics
    ///
    /// Panics if either vessel is out of bounds.
    #[inline]
    pub fn swap_vessels(&mut self, a: VesselIndex, b: VesselIndex) {
        assert!(a < self.num_vessels);
        assert!(b < self.num_vessels);

        if a == b {
            return;
        }

        self.arena.swap_nodes(a.into(), b.into());
        self.vessel_berth.swap(a.get(), b.get());
    }

    /// # Safety
    ///
    /// Both vessels must be in bounds.
    #[inline]
    pub unsafe fn swap_vessels_unchecked(&mut self, a: VesselIndex, b: VesselIndex) {
        debug_assert!(a < self.num_vessels);
        debug_assert!(b < self.num_vessels);

        if a == b {
            return;
        }
        unsafe { self.arena.swap_nodes_unchecked(a.into(), b.into()) };

        unsafe {
            std::ptr::swap(
                self.vessel_berth.as_mut_ptr().add(a.get()),
                self.vessel_berth.as_mut_ptr().add(b.get()),
            );
        }
    }

    /// Swaps two contiguous segments of vessels.
    ///
    /// # Panics
    ///
    /// Panics if any vessel index is out of bounds.
    #[inline]
    pub fn swap_segments(
        &mut self,
        a_first: VesselIndex,
        a_last: VesselIndex,
        b_first: VesselIndex,
        b_last: VesselIndex,
    ) {
        assert!(a_first < self.num_vessels);
        assert!(a_last < self.num_vessels);
        assert!(b_first < self.num_vessels);
        assert!(b_last < self.num_vessels);

        if a_first == b_first {
            return;
        }

        let berth_a = self.vessel_berth[a_first.get()];
        let berth_b = self.vessel_berth[b_first.get()];

        self.arena
            .swap_segments(a_first.into(), a_last.into(), b_first.into(), b_last.into());

        if berth_a != berth_b {
            self.update_segment_berth(a_first.into(), a_last.into(), berth_b);
            self.update_segment_berth(b_first.into(), b_last.into(), berth_a);
        }
    }

    /// # Safety
    ///
    /// All vessel indices must be in bounds. Segments must be valid and non-overlapping.
    #[inline]
    pub unsafe fn swap_segments_unchecked(
        &mut self,
        a_first: VesselIndex,
        a_last: VesselIndex,
        b_first: VesselIndex,
        b_last: VesselIndex,
    ) {
        debug_assert!(a_first < self.num_vessels);
        debug_assert!(a_last < self.num_vessels);
        debug_assert!(b_first < self.num_vessels);
        debug_assert!(b_last < self.num_vessels);

        if a_first == b_first {
            return;
        }

        let berth_a = *unsafe { self.vessel_berth.get_unchecked(a_first.get()) };
        let berth_b = *unsafe { self.vessel_berth.get_unchecked(b_first.get()) };

        unsafe {
            self.arena.swap_segments_unchecked(
                a_first.into(),
                a_last.into(),
                b_first.into(),
                b_last.into(),
            )
        };

        if berth_a != berth_b {
            unsafe { self.update_segment_berth_unchecked(a_first.into(), a_last.into(), berth_b) };
            unsafe { self.update_segment_berth_unchecked(b_first.into(), b_last.into(), berth_a) };
        }
    }

    /// Reverses a contiguous segment of vessels.
    ///
    /// # Panics
    ///
    /// Panics if either vessel is out of bounds.
    #[inline]
    pub fn reverse_segment(&mut self, first: VesselIndex, last: VesselIndex) {
        assert!(first < self.num_vessels);
        assert!(last < self.num_vessels);

        self.arena.reverse_segment(first.into(), last.into());
    }

    /// # Safety
    ///
    /// Both vessels must be in bounds and form a valid contiguous segment.
    #[inline]
    pub unsafe fn reverse_segment_unchecked(&mut self, first: VesselIndex, last: VesselIndex) {
        debug_assert!(first < self.num_vessels);
        debug_assert!(last < self.num_vessels);

        unsafe {
            self.arena
                .reverse_segment_unchecked(first.into(), last.into())
        };
    }

    /// Relocates a vessel to immediately follow another vessel.
    ///
    /// # Panics
    ///
    /// Panics if either vessel is out of bounds.
    #[inline]
    pub fn relocate_after(&mut self, vessel: VesselIndex, anchor: VesselIndex) {
        assert!(vessel < self.num_vessels);
        assert!(anchor < self.num_vessels);

        let old_berth = self.vessel_berth[vessel.get()];
        let new_berth = self.vessel_berth[anchor.get()];

        self.arena.relocate_after(vessel.into(), anchor.into());

        if old_berth != new_berth {
            self.transfer_vessel_berth(vessel.into(), old_berth, new_berth);
        }
    }

    /// # Safety
    ///
    /// Both vessels must be in bounds.
    #[inline]
    pub unsafe fn relocate_after_unchecked(&mut self, vessel: VesselIndex, anchor: VesselIndex) {
        debug_assert!(vessel < self.num_vessels);
        debug_assert!(anchor < self.num_vessels);

        let old_berth = *unsafe { self.vessel_berth.get_unchecked(vessel.get()) };
        let new_berth = *unsafe { self.vessel_berth.get_unchecked(anchor.get()) };

        unsafe {
            self.arena
                .relocate_after_unchecked(vessel.into(), anchor.into())
        };

        if old_berth != new_berth {
            unsafe { self.transfer_vessel_berth_unchecked(vessel.into(), old_berth, new_berth) };
        }
    }

    /// Relocates a vessel to immediately precede another vessel.
    ///
    /// # Panics
    ///
    /// Panics if either vessel is out of bounds.
    #[inline]
    pub fn relocate_before(&mut self, vessel: VesselIndex, reference: VesselIndex) {
        assert!(vessel < self.num_vessels);
        assert!(reference < self.num_vessels);

        let old_berth = self.vessel_berth[vessel.get()];
        let new_berth = self.vessel_berth[reference.get()];

        let anchor_predecessor = self.arena.prev(reference.into());
        self.arena.relocate_after(vessel.into(), anchor_predecessor);

        if old_berth != new_berth {
            self.transfer_vessel_berth(vessel.into(), old_berth, new_berth);
        }
    }

    /// # Safety
    ///
    /// Both vessels must be in bounds.
    #[inline]
    pub unsafe fn relocate_before_unchecked(
        &mut self,
        vessel: VesselIndex,
        reference: VesselIndex,
    ) {
        debug_assert!(vessel < self.num_vessels);
        debug_assert!(reference < self.num_vessels);

        let old_berth = *unsafe { self.vessel_berth.get_unchecked(vessel.get()) };
        let new_berth = *unsafe { self.vessel_berth.get_unchecked(reference.get()) };

        unsafe {
            self.arena
                .relocate_before_unchecked(vessel.into(), reference.into())
        };

        if old_berth != new_berth {
            unsafe { self.transfer_vessel_berth_unchecked(vessel.into(), old_berth, new_berth) };
        }
    }

    /// Relocates a vessel to the head of a berth.
    ///
    /// # Panics
    ///
    /// Panics if `vessel` or `berth` is out of bounds.
    #[inline]
    pub fn relocate_to_head(&mut self, vessel: VesselIndex, berth: BerthIndex) {
        assert!(vessel < self.num_vessels);
        assert!(berth < self.num_berths);

        let old_berth = self.vessel_berth[vessel.get()];
        let sentinel = self.sentinel(berth);

        self.arena.relocate_after(vessel.into(), sentinel);

        if old_berth != berth {
            self.transfer_vessel_berth(vessel.into(), old_berth, berth);
        }
    }

    /// # Safety
    ///
    /// `vessel` and `berth` must be in bounds.
    #[inline]
    pub unsafe fn relocate_to_head_unchecked(&mut self, vessel: VesselIndex, berth: BerthIndex) {
        debug_assert!(vessel < self.num_vessels);
        debug_assert!(berth < self.num_berths);

        let old_berth = *unsafe { self.vessel_berth.get_unchecked(vessel.get()) };
        let sentinel = self.sentinel(berth);

        unsafe { self.arena.relocate_after_unchecked(vessel.into(), sentinel) };

        if old_berth != berth {
            unsafe { self.transfer_vessel_berth_unchecked(vessel.into(), old_berth, berth) };
        }
    }

    /// Relocates a vessel to the tail of a berth.
    ///
    /// # Panics
    ///
    /// Panics if `vessel` or `berth` is out of bounds.
    #[inline]
    pub fn relocate_to_tail(&mut self, vessel: VesselIndex, berth: BerthIndex) {
        assert!(vessel < self.num_vessels);
        assert!(berth < self.num_berths);

        let old_berth = self.vessel_berth[vessel.get()];
        let sentinel = self.sentinel(berth);
        let tail = self.arena.prev(sentinel);

        self.arena.relocate_after(vessel.into(), tail);

        if old_berth != berth {
            self.transfer_vessel_berth(vessel.into(), old_berth, berth);
        }
    }

    /// # Safety
    ///
    /// `vessel` and `berth` must be in bounds.
    #[inline]
    pub unsafe fn relocate_to_tail_unchecked(&mut self, vessel: VesselIndex, berth: BerthIndex) {
        debug_assert!(vessel < self.num_vessels);
        debug_assert!(berth < self.num_berths);

        let old_berth = *unsafe { self.vessel_berth.get_unchecked(vessel.get()) };
        let sentinel = self.sentinel(berth);
        let tail = unsafe { self.arena.prev_unchecked(sentinel) };

        unsafe { self.arena.relocate_after_unchecked(vessel.into(), tail) };

        if old_berth != berth {
            unsafe { self.transfer_vessel_berth_unchecked(vessel.into(), old_berth, berth) };
        }
    }

    /// Relocates a segment to immediately follow another vessel.
    ///
    /// # Panics
    ///
    /// Panics if any vessel index is out of bounds.
    #[inline]
    pub fn relocate_segment_after(
        &mut self,
        first: VesselIndex,
        last: VesselIndex,
        anchor: VesselIndex,
    ) {
        assert!(first < self.num_vessels);
        assert!(last < self.num_vessels);
        assert!(anchor < self.num_vessels);

        self.arena
            .relocate_segment_after(first.into(), last.into(), anchor.into());

        let new_berth = self.vessel_berth[anchor.get()];
        self.update_segment_berth(first.into(), last.into(), new_berth);
    }

    /// # Safety
    ///
    /// All vessel indices must be in bounds.
    #[inline]
    pub unsafe fn relocate_segment_after_unchecked(
        &mut self,
        first: VesselIndex,
        last: VesselIndex,
        anchor: VesselIndex,
    ) {
        debug_assert!(first < self.num_vessels);
        debug_assert!(last < self.num_vessels);
        debug_assert!(anchor < self.num_vessels);

        unsafe {
            self.arena
                .relocate_segment_after_unchecked(first.into(), last.into(), anchor.into())
        };

        let new_berth = *unsafe { self.vessel_berth.get_unchecked(anchor.get()) };
        unsafe { self.update_segment_berth_unchecked(first.into(), last.into(), new_berth) };
    }

    /// Relocates a segment to immediately precede another vessel.
    ///
    /// # Panics
    ///
    /// Panics if any vessel index is out of bounds.
    #[inline]
    pub fn relocate_segment_before(
        &mut self,
        first: VesselIndex,
        last: VesselIndex,
        reference: VesselIndex,
    ) {
        assert!(first < self.num_vessels);
        assert!(last < self.num_vessels);
        assert!(reference < self.num_vessels);

        let anchor_predecessor = self.arena.prev(reference.into());
        self.relocate_segment_after(first, last, VesselIndex::new(anchor_predecessor.index()));
    }

    /// # Safety
    ///
    /// All vessel indices must be in bounds.
    #[inline]
    pub unsafe fn relocate_segment_before_unchecked(
        &mut self,
        first: VesselIndex,
        last: VesselIndex,
        reference: VesselIndex,
    ) {
        debug_assert!(first < self.num_vessels);
        debug_assert!(last < self.num_vessels);
        debug_assert!(reference < self.num_vessels);

        unsafe {
            self.arena.relocate_segment_before_unchecked(
                first.into(),
                last.into(),
                reference.into(),
            )
        };

        let new_berth = *unsafe { self.vessel_berth.get_unchecked(reference.get()) };
        unsafe { self.update_segment_berth_unchecked(first.into(), last.into(), new_berth) };
    }

    /// Relocates a segment to the head of a berth.
    ///
    /// # Panics
    ///
    /// Panics if any vessel index or berth is out of bounds.
    #[inline]
    pub fn relocate_segment_to_head(
        &mut self,
        first: VesselIndex,
        last: VesselIndex,
        berth: BerthIndex,
    ) {
        assert!(first < self.num_vessels);
        assert!(last < self.num_vessels);
        assert!(berth < self.num_berths);

        let sentinel = self.sentinel(berth);
        self.arena
            .relocate_segment_after(first.into(), last.into(), sentinel);
        self.update_segment_berth(first.into(), last.into(), berth);
    }

    /// # Safety
    ///
    /// All indices must be in bounds.
    #[inline]
    pub unsafe fn relocate_segment_to_head_unchecked(
        &mut self,
        first: VesselIndex,
        last: VesselIndex,
        berth: BerthIndex,
    ) {
        debug_assert!(first < self.num_vessels);
        debug_assert!(last < self.num_vessels);
        debug_assert!(berth < self.num_berths);

        let sentinel = self.sentinel(berth);
        unsafe {
            self.arena
                .relocate_segment_after_unchecked(first.into(), last.into(), sentinel)
        };
        unsafe { self.update_segment_berth_unchecked(first.into(), last.into(), berth) };
    }

    /// Relocates a segment to the tail of a berth.
    ///
    /// # Panics
    ///
    /// Panics if any vessel index or berth is out of bounds.
    #[inline]
    pub fn relocate_segment_to_tail(
        &mut self,
        first: VesselIndex,
        last: VesselIndex,
        berth: BerthIndex,
    ) {
        assert!(first < self.num_vessels);
        assert!(last < self.num_vessels);
        assert!(berth < self.num_berths);

        let sentinel = self.sentinel(berth);
        let tail = self.arena.prev(sentinel);
        self.arena
            .relocate_segment_after(first.into(), last.into(), tail);
        self.update_segment_berth(first.into(), last.into(), berth);
    }

    /// # Safety
    ///
    /// All indices must be in bounds.
    #[inline]
    pub unsafe fn relocate_segment_to_tail_unchecked(
        &mut self,
        first: VesselIndex,
        last: VesselIndex,
        berth: BerthIndex,
    ) {
        debug_assert!(first < self.num_vessels);
        debug_assert!(last < self.num_vessels);
        debug_assert!(berth < self.num_berths);

        let sentinel = self.sentinel(berth);
        let tail = unsafe { self.arena.prev_unchecked(sentinel) };
        unsafe {
            self.arena
                .relocate_segment_after_unchecked(first.into(), last.into(), tail)
        };
        unsafe { self.update_segment_berth_unchecked(first.into(), last.into(), berth) };
    }

    // ----------------------------------------------------------------------------
    // Private helpers
    // ----------------------------------------------------------------------------

    /// Reassigns an entire contiguous segment of vessels to `new_berth` in the
    /// berth-tracking side tables.
    ///
    /// This helper updates only the auxiliary metadata:
    /// - `vessel_berth[v]` for every vessel in the segment
    /// - `berth_vessel_count[old_berth]`
    /// - `berth_vessel_count[new_berth]`
    ///
    /// It does **not** modify the linked structure in `self.arena`. Call this only
    /// after a topology operation has already moved the segment to a new berth, or
    /// when repairing the metadata to match such a move.
    ///
    /// The segment is interpreted as the inclusive path
    /// `segment_first -> ... -> segment_last` following `next` pointers.
    ///
    /// If the segment already belongs to `new_berth`, this is a no-op.
    #[inline(always)]
    fn update_segment_berth(
        &mut self,
        segment_first: Node,
        segment_last: Node,
        new_berth: BerthIndex,
    ) {
        let old_berth = self.vessel_berth[segment_first.index()];
        if old_berth == new_berth {
            return;
        }

        let mut count = 0usize;
        let mut current = segment_first;
        loop {
            self.vessel_berth[current.index()] = new_berth;
            count += 1;
            if current == segment_last {
                break;
            }
            current = self.arena.next(current);
        }
        self.berth_vessel_count[old_berth.get()] -= count;
        self.berth_vessel_count[new_berth.get()] += count;
    }

    /// Reassigns an entire contiguous segment of vessels to `new_berth` in the
    /// berth-tracking side tables, without performing bounds checks.
    ///
    /// This is the unchecked counterpart to [`ScheduleGraph::update_segment_berth`].
    /// It updates only the metadata and does **not** change the arena topology.
    ///
    /// The segment is interpreted as the inclusive path
    /// `segment_first -> ... -> segment_last` following `next` pointers.
    ///
    /// If the segment already belongs to `new_berth`, this is a no-op.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - `segment_first` and `segment_last` are real vessel nodes
    ///   (`index() < self.num_vessels`)
    /// - `new_berth` is a valid berth index
    ///   (`new_berth < self.num_berths`)
    /// - `segment_first ..= segment_last` forms a valid contiguous path in the
    ///   current arena when following `next` pointers
    /// - every vessel on that path currently belongs to the same source berth
    ///
    /// Violating these assumptions can corrupt `vessel_berth` and
    /// `berth_vessel_count`, leaving them inconsistent with the arena.
    #[inline(always)]
    unsafe fn update_segment_berth_unchecked(
        &mut self,
        segment_first: Node,
        segment_last: Node,
        new_berth: BerthIndex,
    ) {
        debug_assert!(segment_first.index() < self.num_vessels);
        debug_assert!(segment_last.index() < self.num_vessels);
        debug_assert!(new_berth < self.num_berths);

        let old_berth = *unsafe { self.vessel_berth.get_unchecked(segment_first.index()) };
        if old_berth == new_berth {
            return;
        }

        let mut count = 0usize;
        let mut current = segment_first;
        loop {
            let next_node = unsafe { self.arena.next_unchecked(current) };
            *unsafe { self.vessel_berth.get_unchecked_mut(current.index()) } = new_berth;
            count += 1;
            if current == segment_last {
                break;
            }
            current = next_node;
        }
        *unsafe { self.berth_vessel_count.get_unchecked_mut(old_berth.get()) } -= count;
        *unsafe { self.berth_vessel_count.get_unchecked_mut(new_berth.get()) } += count;
    }

    /// Reassigns a single vessel to `new_berth` in the berth-tracking side tables.
    ///
    /// This helper updates only:
    /// - `vessel_berth[vessel]`
    /// - `berth_vessel_count[old_berth]`
    /// - `berth_vessel_count[new_berth]`
    ///
    /// It does **not** modify the arena topology. Call this only after a move that
    /// has already placed `vessel` on a different berth, or when repairing metadata
    /// to match such a move.
    #[inline(always)]
    fn transfer_vessel_berth(
        &mut self,
        vessel: Node,
        old_berth: BerthIndex,
        new_berth: BerthIndex,
    ) {
        self.vessel_berth[vessel.index()] = new_berth;
        self.berth_vessel_count[old_berth.get()] -= 1;
        self.berth_vessel_count[new_berth.get()] += 1;
    }

    /// Reassigns a single vessel to `new_berth` in the berth-tracking side tables,
    /// without performing bounds checks.
    ///
    /// This is the unchecked counterpart to [`ScheduleGraph::transfer_vessel_berth`].
    /// It updates only metadata and assumes the caller has already made the
    /// corresponding structural change in the arena, if any.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - `vessel` is a real vessel node (`index() < self.num_vessels`)
    /// - `old_berth` and `new_berth` are valid berth indices
    /// - the current metadata is consistent, i.e. `vessel` is currently counted on
    ///   `old_berth`
    ///
    /// Violating these assumptions can underflow/overflow berth counts or leave the
    /// berth metadata inconsistent with the arena topology.
    #[inline(always)]
    unsafe fn transfer_vessel_berth_unchecked(
        &mut self,
        vessel: Node,
        old_berth: BerthIndex,
        new_berth: BerthIndex,
    ) {
        debug_assert!(vessel.index() < self.num_vessels);
        debug_assert!(old_berth < self.num_berths);
        debug_assert!(new_berth < self.num_berths);

        unsafe {
            *self.vessel_berth.get_unchecked_mut(vessel.index()) = new_berth;
            *self.berth_vessel_count.get_unchecked_mut(old_berth.get()) -= 1;
            *self.berth_vessel_count.get_unchecked_mut(new_berth.get()) += 1;
        }
    }
}

// ----------------------------------------------------------------
// ScheduleGraphDiffEdge
// ----------------------------------------------------------------

/// A high-level representation of a link change.
///
/// `None` represents a Sentinel (the boundary of the berth).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScheduleGraphDiffEdge {
    pub from: Option<VesselIndex>,
    pub to: Option<VesselIndex>,
}

impl std::fmt::Display for ScheduleGraphDiffEdge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.from {
            Some(v) => write!(f, "Vessel({})", v.get())?,
            None => write!(f, "Sentinel")?,
        }
        write!(f, " -> ")?;
        match self.to {
            Some(v) => write!(f, "Vessel({})", v.get()),
            None => write!(f, "Sentinel"),
        }
    }
}

// ----------------------------------------------------------------
// DiffEdgeIter
// ----------------------------------------------------------------

/// Iterator over broken or created links in a [`ScheduleGraphDiff`].
pub struct DiffEdgeIter<'a> {
    inner: std::iter::Zip<std::slice::Iter<'a, Node>, std::slice::Iter<'a, Node>>,
    num_vessels: usize,
}

impl<'a> DiffEdgeIter<'a> {
    #[inline(always)]
    fn is_sentinel(&self, index: Node) -> bool {
        index.index() >= self.num_vessels
    }

    #[inline(always)]
    fn to_option(&self, index: Node) -> Option<VesselIndex> {
        if self.is_sentinel(index) {
            None
        } else {
            Some(VesselIndex::new(index.index()))
        }
    }
}

impl<'a> Iterator for DiffEdgeIter<'a> {
    type Item = ScheduleGraphDiffEdge;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(&f, &t)| ScheduleGraphDiffEdge {
            from: self.to_option(f),
            to: self.to_option(t),
        })
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

// ----------------------------------------------------------------
// DiffEdgeContextIter
// ----------------------------------------------------------------

/// Iterator over created links with their associated berth context.
pub struct DiffEdgeContextIter<'a> {
    inner: std::iter::Zip<std::slice::Iter<'a, Node>, std::slice::Iter<'a, Node>>,
    num_vessels: usize,
}

impl<'a> DiffEdgeContextIter<'a> {
    #[inline(always)]
    fn is_sentinel(&self, index: Node) -> bool {
        index.index() >= self.num_vessels
    }

    #[inline(always)]
    fn to_option(&self, index: Node) -> Option<VesselIndex> {
        if self.is_sentinel(index) {
            None
        } else {
            Some(VesselIndex::new(index.index()))
        }
    }

    #[inline(always)]
    fn to_berth(&self, index: Node) -> Option<BerthIndex> {
        if self.is_sentinel(index) {
            Some(BerthIndex::new(index.index() - self.num_vessels))
        } else {
            None
        }
    }
}

impl<'a> Iterator for DiffEdgeContextIter<'a> {
    type Item = (Option<VesselIndex>, Option<VesselIndex>, Option<BerthIndex>);

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(&f, &t)| {
            let b = self.to_berth(f).or_else(|| self.to_berth(t));
            (self.to_option(f), self.to_option(t), b)
        })
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

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
    // Links now store strongly typed topology indices (Node) internally
    broken_from: Vec<Node>,
    broken_to: Vec<Node>,
    created_from: Vec<Node>,
    created_to: Vec<Node>,

    // Reallocations only apply to real vessels, so these stay strongly typed!
    reallocated_vessels: Vec<VesselIndex>,
    original_berths: Vec<BerthIndex>,
    target_berths: Vec<BerthIndex>,
    num_vessels: usize,
}

impl ScheduleGraphDiff {
    #[inline(always)]
    pub fn new(num_vessels: usize) -> Self {
        Self {
            broken_from: Vec::new(),
            broken_to: Vec::new(),
            created_from: Vec::new(),
            created_to: Vec::new(),
            reallocated_vessels: Vec::new(),
            original_berths: Vec::new(),
            target_berths: Vec::new(),
            num_vessels,
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
            num_vessels: self.num_vessels,
        }
    }

    #[inline(always)]
    pub fn created_links(&self) -> DiffEdgeIter<'_> {
        DiffEdgeIter {
            inner: self.created_from.iter().zip(self.created_to.iter()),
            num_vessels: self.num_vessels,
        }
    }

    #[inline(always)]
    pub fn created_links_with_context(&self) -> DiffEdgeContextIter<'_> {
        DiffEdgeContextIter {
            inner: self.created_from.iter().zip(self.created_to.iter()),
            num_vessels: self.num_vessels,
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

// ----------------------------------------------------------------
// Tests
// ----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn vessel(i: usize) -> VesselIndex {
        VesselIndex::new(i)
    }

    fn berth(i: usize) -> BerthIndex {
        BerthIndex::new(i)
    }

    fn check_berth(graph: &ScheduleGraph, berth_index: usize, expected: &[usize]) {
        let b = berth(berth_index);

        let forward: Vec<usize> = graph.vessel_sequence_iter(b).map(|v| v.get()).collect();
        assert_eq!(
            forward, expected,
            "Forward mismatch for Berth {}",
            berth_index
        );

        let reverse: Vec<usize> = graph.vessel_sequence_rev_iter(b).map(|v| v.get()).collect();
        let mut expected_rev = expected.to_vec();
        expected_rev.reverse();
        assert_eq!(
            reverse, expected_rev,
            "Reverse mismatch for Berth {}",
            berth_index
        );

        if expected.is_empty() {
            assert_eq!(graph.first_vessel(b), None);
            assert_eq!(graph.last_vessel(b), None);
            assert!(graph.is_empty(b));
        } else {
            assert_eq!(graph.first_vessel(b).unwrap().get(), expected[0]);
            assert_eq!(
                graph.last_vessel(b).unwrap().get(),
                *expected.last().unwrap()
            );
            assert!(!graph.is_empty(b));
        }

        for &v in expected {
            assert_eq!(graph.vessel_berth(vessel(v)).get(), berth_index);
        }

        assert_eq!(graph.vessel_count(b), expected.len());
    }

    /// Berth 0: V0 -> V2 -> V1  (sorted by start times [10, 30, 20])
    /// Berth 1: V3 -> V4
    /// Berth 2: V5
    /// Berth 3: (empty)
    fn standard_fixture() -> ScheduleGraph {
        let berths = [berth(0), berth(0), berth(0), berth(1), berth(1), berth(2)];
        let starts = [10, 30, 20, 10, 20, 15];
        ScheduleGraph::from_slices(&berths, &starts, 4)
    }

    #[test]
    fn test_initialization() {
        let graph = standard_fixture();
        assert_eq!(graph.num_vessels(), 6);
        assert_eq!(graph.num_berths(), 4);
        check_berth(&graph, 0, &[0, 2, 1]);
        check_berth(&graph, 1, &[3, 4]);
        check_berth(&graph, 2, &[5]);
        check_berth(&graph, 3, &[]);
    }

    #[test]
    fn test_empty_graph() {
        let graph = ScheduleGraph::from_slices::<i32>(&[], &[], 3);
        assert_eq!(graph.num_vessels(), 0);
        check_berth(&graph, 0, &[]);
        check_berth(&graph, 1, &[]);
        check_berth(&graph, 2, &[]);
    }

    #[test]
    fn test_swap_vessels_adjacent() {
        let mut graph = standard_fixture();
        graph.swap_vessels(vessel(0), vessel(2));
        check_berth(&graph, 0, &[2, 0, 1]);
    }

    #[test]
    fn test_swap_vessels_non_adjacent() {
        let mut graph = standard_fixture();
        graph.swap_vessels(vessel(0), vessel(1));
        check_berth(&graph, 0, &[1, 2, 0]);
    }

    #[test]
    fn test_swap_vessels_cross_berth() {
        let mut graph = standard_fixture();
        graph.swap_vessels(vessel(2), vessel(4));
        check_berth(&graph, 0, &[0, 4, 1]);
        check_berth(&graph, 1, &[3, 2]);
    }

    #[test]
    fn test_swap_segments() {
        let mut graph = standard_fixture();
        graph.swap_segments(vessel(0), vessel(2), vessel(3), vessel(4));
        check_berth(&graph, 0, &[3, 4, 1]);
        check_berth(&graph, 1, &[0, 2]);
    }

    #[test]
    fn test_swap_segments_adjacent() {
        let mut graph = standard_fixture();
        graph.swap_segments(vessel(0), vessel(0), vessel(2), vessel(1));
        check_berth(&graph, 0, &[2, 1, 0]);
    }

    #[test]
    fn test_reverse_segment() {
        let mut graph = standard_fixture();
        graph.reverse_segment(vessel(0), vessel(1));
        check_berth(&graph, 0, &[1, 2, 0]);
    }

    #[test]
    fn test_reverse_single_noop() {
        let mut graph = standard_fixture();
        graph.reverse_segment(vessel(2), vessel(2));
        check_berth(&graph, 0, &[0, 2, 1]);
    }

    #[test]
    fn test_relocate_after() {
        let mut graph = standard_fixture();
        graph.relocate_after(vessel(0), vessel(2));
        check_berth(&graph, 0, &[2, 0, 1]);
    }

    #[test]
    fn test_relocate_after_cross_berth() {
        let mut graph = standard_fixture();
        graph.relocate_after(vessel(0), vessel(3));
        check_berth(&graph, 0, &[2, 1]);
        check_berth(&graph, 1, &[3, 0, 4]);
    }

    #[test]
    fn test_relocate_before() {
        let mut graph = standard_fixture();
        graph.relocate_before(vessel(1), vessel(0));
        check_berth(&graph, 0, &[1, 0, 2]);
    }

    #[test]
    fn test_relocate_to_head() {
        let mut graph = standard_fixture();
        graph.relocate_to_head(vessel(4), berth(0));
        check_berth(&graph, 0, &[4, 0, 2, 1]);
        check_berth(&graph, 1, &[3]);
    }

    #[test]
    fn test_relocate_to_tail() {
        let mut graph = standard_fixture();
        graph.relocate_to_tail(vessel(0), berth(3));
        check_berth(&graph, 0, &[2, 1]);
        check_berth(&graph, 3, &[0]);
    }

    #[test]
    fn test_relocate_segment_after() {
        let mut graph = standard_fixture();
        graph.relocate_segment_after(vessel(0), vessel(2), vessel(4));
        check_berth(&graph, 0, &[1]);
        check_berth(&graph, 1, &[3, 4, 0, 2]);
    }

    #[test]
    fn test_relocate_segment_before() {
        let mut graph = standard_fixture();
        graph.relocate_segment_after(vessel(0), vessel(2), vessel(1));
        check_berth(&graph, 0, &[1, 0, 2]);
    }

    #[test]
    fn test_relocate_segment_to_head() {
        let mut graph = standard_fixture();
        graph.relocate_segment_to_head(vessel(3), vessel(4), berth(3));
        check_berth(&graph, 1, &[]);
        check_berth(&graph, 3, &[3, 4]);
    }

    #[test]
    fn test_relocate_segment_to_tail() {
        let mut graph = standard_fixture();
        graph.relocate_segment_to_tail(vessel(0), vessel(2), berth(1));
        check_berth(&graph, 0, &[1]);
        check_berth(&graph, 1, &[3, 4, 0, 2]);
    }

    #[test]
    fn test_overwrite_from_graph() {
        let original = standard_fixture();
        let mut clone = ScheduleGraph::from_slices::<i32>(&[], &[], 4);
        clone.overwrite_from_graph(&original);
        assert_eq!(original, clone);

        clone.swap_vessels(vessel(0), vessel(2));
        assert_ne!(original, clone);
    }

    #[test]
    fn test_complex_multi_step() {
        let mut graph = standard_fixture();

        graph.swap_vessels(vessel(0), vessel(5));
        graph.relocate_segment_to_head(vessel(2), vessel(1), berth(3));
        graph.relocate_after(vessel(3), vessel(5));
        graph.reverse_segment(vessel(5), vessel(3));
        graph.swap_segments(vessel(3), vessel(5), vessel(2), vessel(1));

        check_berth(&graph, 0, &[2, 1]);
        check_berth(&graph, 1, &[4]);
        check_berth(&graph, 2, &[0]);
        check_berth(&graph, 3, &[3, 5]);
    }

    #[test]
    fn test_all_edges() {
        let graph = standard_fixture();
        let mut edges: Vec<ScheduleGraphFullEdge> = graph.all_edges().collect();
        edges.sort_by_key(|e| e.from.get());

        assert_eq!(edges.len(), 3);
        assert_eq!(
            edges[0],
            ScheduleGraphFullEdge {
                from: vessel(0),
                to: vessel(2),
                on_berth: berth(0)
            }
        );
        assert_eq!(
            edges[1],
            ScheduleGraphFullEdge {
                from: vessel(2),
                to: vessel(1),
                on_berth: berth(0)
            }
        );
        assert_eq!(
            edges[2],
            ScheduleGraphFullEdge {
                from: vessel(3),
                to: vessel(4),
                on_berth: berth(1)
            }
        );
    }

    #[test]
    fn test_berth_edges() {
        let graph = standard_fixture();
        let edges: Vec<_> = graph.berth_edges(berth(0)).collect();
        assert_eq!(edges.len(), 2);
        assert_eq!(
            edges[0],
            ScheduleGraphEdge {
                from: vessel(0),
                to: vessel(2)
            }
        );
        assert_eq!(
            edges[1],
            ScheduleGraphEdge {
                from: vessel(2),
                to: vessel(1)
            }
        );

        assert_eq!(graph.berth_edges(berth(2)).count(), 0);
        assert_eq!(graph.berth_edges(berth(3)).count(), 0);
    }

    #[test]
    fn test_predecessor_successor() {
        let graph = standard_fixture();

        assert_eq!(graph.vessel_predecessor(vessel(0)), None);
        assert_eq!(graph.vessel_predecessor(vessel(2)), Some(vessel(0)));
        assert_eq!(graph.vessel_predecessor(vessel(1)), Some(vessel(2)));

        assert_eq!(graph.vessel_successor(vessel(0)), Some(vessel(2)));
        assert_eq!(graph.vessel_successor(vessel(1)), None);
        assert_eq!(graph.vessel_successor(vessel(5)), None);
    }
}
