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
//! This module provides the [`ScheduleGraph`], which acts as the core state representation
//! (genotype) for Local Search algorithms (e.g., Simulated Annealing, Tabu Search, ALNS)
//! applied to the Dynamic Berth Allocation Problem (DBAP) or Vehicle Routing Problem (VRP) variants.
//!
//! # Architecture
//!
//! `ScheduleGraph` layers domain semantics on top of a [`RingArena`]. The arena handles
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
//!
//! # Public API
//!
//! The public API exclusively uses [`VesselIndex`] and [`BerthIndex`]. The sentinel
//! convention and the underlying [`RingArena`] are implementation details that never
//! leak to downstream consumers.

use std::iter::FusedIterator;
use talos_core::container::rarena::{
    RingArena, RingEdgeIter, RingSequenceIter, RingSequenceRevIter,
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
        let raw = self.inner.next()?;
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
        let raw = self.inner.next()?;
        debug_assert!(
            raw < self.num_vessels,
            "VesselSequenceRevIter yielded a sentinel index: {} >= {}",
            raw,
            self.num_vessels
        );
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
        let (from_raw, to_raw) = self.inner.next()?;
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

            let to = unsafe { self.arena.next_unchecked(from) };

            if to < self.num_vessels {
                let on_berth = *unsafe { self.vessel_berth.get_unchecked(from) };
                return Some(ScheduleGraphFullEdge {
                    from: VesselIndex::new(from),
                    to: VesselIndex::new(to),
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

    /// Overwrites the current `ScheduleGraph` with data from parallel slices.
    ///
    /// # Panics
    ///
    /// Panics if `berths.len() != start_times.len()`, or if any assigned berth
    /// in the slice is out of bounds.
    pub fn overwrite_from_slices<T>(
        &mut self,
        berths: &[BerthIndex],
        start_times: &[T],
        num_berths: usize,
    ) where
        T: Ord,
    {
        assert_eq!(
            berths.len(),
            start_times.len(),
            "called `ScheduleGraph::overwrite_from_slices` with mismatched slice lengths"
        );

        self.num_vessels = berths.len();
        self.num_berths = num_berths;
        let total_nodes = self.num_vessels + self.num_berths;

        // Reuse existing arena allocation.
        self.arena.resize(total_nodes);
        let (prev, next) = unsafe { self.arena.raw_mut() };

        // Initialize vessel_berth and berth_vessel_count.
        self.vessel_berth.clear();
        self.vessel_berth.resize(total_nodes, BerthIndex::new(0));
        self.berth_vessel_count.clear();
        self.berth_vessel_count.resize(self.num_berths, 0);

        // Initialize sentinels as self-loops.
        for berth_idx in 0..self.num_berths {
            let sentinel = self.num_vessels + berth_idx;
            next[sentinel] = sentinel;
            prev[sentinel] = sentinel;
            self.vessel_berth[sentinel] = BerthIndex::new(berth_idx);
        }

        if self.num_vessels == 0 {
            return;
        }

        for &berth in berths {
            assert!(
                berth.get() < self.num_berths,
                "called `ScheduleGraph::overwrite_from_slices` with out-of-bounds berth: berth = {}, num_berths = {}",
                berth.get(),
                self.num_berths
            );
        }

        // Populate vessel_berth and berth_vessel_count.
        for (vessel_idx, &berth) in berths.iter().enumerate() {
            self.vessel_berth[vessel_idx] = berth;
            self.berth_vessel_count[berth.get()] += 1;
        }

        // Sort vessels by (berth, start_time) using prev as scratch for indices.
        for (i, slot) in prev.iter_mut().enumerate().take(self.num_vessels) {
            *slot = i;
        }

        prev[0..self.num_vessels].sort_unstable_by(|&left, &right| {
            berths[left]
                .cmp(&berths[right])
                .then_with(|| start_times[left].cmp(&start_times[right]))
        });

        // Build the ring topology from sorted order.
        let mut current_berth: Option<BerthIndex> = None;
        let mut current_tail = 0usize;

        for &vessel in prev.iter().take(self.num_vessels) {
            let berth = berths[vessel];

            if Some(berth) != current_berth {
                if let Some(previous_berth) = current_berth {
                    next[current_tail] = self.num_vessels + previous_berth.get();
                }
                current_berth = Some(berth);
                current_tail = self.num_vessels + berth.get();
            }
            next[current_tail] = vessel;
            current_tail = vessel;
        }

        if let Some(final_berth) = current_berth {
            next[current_tail] = self.num_vessels + final_berth.get();
        }

        // Derive prev from next.
        for (node_idx, &successor) in next.iter().enumerate().take(total_nodes) {
            prev[successor] = node_idx;
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
    // Internal helpers
    // ----------------------------------------------------------------

    /// Returns the raw arena index for a berth's sentinel node.
    #[inline(always)]
    pub fn sentinel(&self, berth: BerthIndex) -> usize {
        debug_assert!(berth.get() < self.num_berths);
        self.num_vessels + berth.get()
    }

    /// Updates `vessel_berth` and `berth_vessel_count` for every node in a
    /// contiguous segment that has been moved to `new_berth`.
    #[inline(always)]
    unsafe fn update_segment_berth_unchecked(
        &mut self,
        segment_first: usize,
        segment_last: usize,
        new_berth: BerthIndex,
    ) {
        let old_berth = *unsafe { self.vessel_berth.get_unchecked(segment_first) };
        if old_berth == new_berth {
            return;
        }

        let mut count = 0usize;
        let mut current = segment_first;
        loop {
            let next_node = unsafe { self.arena.next_unchecked(current) };
            *unsafe { self.vessel_berth.get_unchecked_mut(current) } = new_berth;
            count += 1;
            if current == segment_last {
                break;
            }
            current = next_node;
        }
        *unsafe { self.berth_vessel_count.get_unchecked_mut(old_berth.get()) } -= count;
        *unsafe { self.berth_vessel_count.get_unchecked_mut(new_berth.get()) } += count;
    }

    /// Updates berth tracking for a single vessel that moved berths.
    #[inline(always)]
    unsafe fn transfer_vessel_berth_unchecked(
        &mut self,
        vessel: usize,
        old_berth: BerthIndex,
        new_berth: BerthIndex,
    ) {
        unsafe {
            *self.vessel_berth.get_unchecked_mut(vessel) = new_berth;
            *self.berth_vessel_count.get_unchecked_mut(old_berth.get()) -= 1;
            *self.berth_vessel_count.get_unchecked_mut(new_berth.get()) += 1;
        }
    }

    // ----------------------------------------------------------------
    // Crate-internal accessors (for Mutator / tracker)
    // ----------------------------------------------------------------

    /// Returns the underlying arena.
    ///
    /// Exposed for crate-internal performance-critical code (e.g., the `Mutator`'s
    /// `EdgeDeltaTracker`) that needs raw topology access.
    #[inline(always)]
    pub(crate) fn arena(&self) -> &RingArena {
        &self.arena
    }

    // ----------------------------------------------------------------
    // Public query API
    // ----------------------------------------------------------------

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
        assert!(berth.get() < self.num_berths);
        self.berth_vessel_count[berth.get()] == 0
    }

    /// Returns the berth to which the given vessel is currently assigned.
    ///
    /// # Panics
    ///
    /// Panics if `vessel` is out of bounds.
    #[inline]
    pub fn vessel_berth(&self, vessel: VesselIndex) -> BerthIndex {
        assert!(vessel.get() < self.num_vessels);
        self.vessel_berth[vessel.get()]
    }

    /// Returns the berth to which the given vessel is currently assigned.
    ///
    /// # Safety
    ///
    /// `vessel.get()` must be `< self.num_vessels`.
    #[inline(always)]
    pub unsafe fn vessel_berth_unchecked(&self, vessel: VesselIndex) -> BerthIndex {
        debug_assert!(vessel.get() < self.num_vessels);
        *unsafe { self.vessel_berth.get_unchecked(vessel.get()) }
    }

    /// Returns the number of vessels currently assigned to the given berth.
    ///
    /// # Panics
    ///
    /// Panics if `berth` is out of bounds.
    #[inline]
    pub fn vessel_count(&self, berth: BerthIndex) -> usize {
        assert!(berth.get() < self.num_berths);
        self.berth_vessel_count[berth.get()]
    }

    /// Returns the number of vessels currently assigned to the given berth.
    ///
    /// # Safety
    ///
    /// `berth.get()` must be `< self.num_berths`.
    #[inline(always)]
    pub unsafe fn vessel_count_unchecked(&self, berth: BerthIndex) -> usize {
        debug_assert!(berth.get() < self.num_berths);
        *unsafe { self.berth_vessel_count.get_unchecked(berth.get()) }
    }

    /// Returns the first vessel in the given berth, or `None` if empty.
    ///
    /// # Panics
    ///
    /// Panics if `berth` is out of bounds.
    #[inline]
    pub fn first_vessel(&self, berth: BerthIndex) -> Option<VesselIndex> {
        assert!(berth.get() < self.num_berths);
        let next_of_sentinel = self.arena.next(self.sentinel(berth));
        if next_of_sentinel < self.num_vessels {
            Some(VesselIndex::new(next_of_sentinel))
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
        assert!(berth.get() < self.num_berths);
        let prev_of_sentinel = self.arena.prev(self.sentinel(berth));
        if prev_of_sentinel < self.num_vessels {
            Some(VesselIndex::new(prev_of_sentinel))
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
        assert!(vessel.get() < self.num_vessels);
        unsafe { self.vessel_predecessor_unchecked(vessel) }
    }

    /// Returns the vessel immediately preceding `vessel`, or `None` if at head.
    ///
    /// # Safety
    ///
    /// `vessel.get()` must be `< self.num_vessels`.
    #[inline(always)]
    pub unsafe fn vessel_predecessor_unchecked(&self, vessel: VesselIndex) -> Option<VesselIndex> {
        debug_assert!(vessel.get() < self.num_vessels);
        let pred = unsafe { self.arena.prev_unchecked(vessel.get()) };
        if pred < self.num_vessels {
            Some(VesselIndex::new(pred))
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
        assert!(vessel.get() < self.num_vessels);
        unsafe { self.vessel_successor_unchecked(vessel) }
    }

    /// Returns the vessel immediately following `vessel`, or `None` if at tail.
    ///
    /// # Safety
    ///
    /// `vessel.get()` must be `< self.num_vessels`.
    #[inline(always)]
    pub unsafe fn vessel_successor_unchecked(&self, vessel: VesselIndex) -> Option<VesselIndex> {
        debug_assert!(vessel.get() < self.num_vessels);
        let succ = unsafe { self.arena.next_unchecked(vessel.get()) };
        if succ < self.num_vessels {
            Some(VesselIndex::new(succ))
        } else {
            None
        }
    }

    // ----------------------------------------------------------------
    // Iterators
    // ----------------------------------------------------------------

    /// Returns an iterator over the vessels assigned to the given berth, in order.
    ///
    /// # Panics
    ///
    /// Panics if `berth` is out of bounds.
    #[inline]
    pub fn vessel_sequence_iter(&self, berth: BerthIndex) -> VesselSequenceIter<'_> {
        assert!(berth.get() < self.num_berths);
        let sentinel = self.sentinel(berth);
        let start = self.arena.next(sentinel);
        VesselSequenceIter {
            inner: self.arena.sequence_iter(start, sentinel),
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
        assert!(berth.get() < self.num_berths);
        let sentinel = self.sentinel(berth);
        let start = self.arena.prev(sentinel);
        VesselSequenceRevIter {
            inner: self.arena.sequence_rev_iter(start, sentinel),
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
        debug_assert!(berth.get() < self.num_berths);
        let sentinel = self.sentinel(berth);
        let start = unsafe { self.arena.next_unchecked(sentinel) };
        VesselSequenceIter {
            inner: self.arena.sequence_iter(start, sentinel),
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
        debug_assert!(berth.get() < self.num_berths);
        let sentinel = self.sentinel(berth);
        let start = unsafe { self.arena.prev_unchecked(sentinel) };
        VesselSequenceRevIter {
            inner: self.arena.sequence_rev_iter(start, sentinel),
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
        assert!(berth.get() < self.num_berths);
        let sentinel = self.sentinel(berth);
        let start = self.arena.next(sentinel);
        BerthEdgeIter {
            inner: self.arena.edge_iter(start, sentinel),
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
    // Mutations — all public, all VesselIndex / BerthIndex only
    // ----------------------------------------------------------------

    /// Swaps the positions of two vessels.
    ///
    /// # Panics
    ///
    /// Panics if either vessel is out of bounds.
    #[inline]
    pub fn swap_vessels(&mut self, a: VesselIndex, b: VesselIndex) {
        assert!(a.get() < self.num_vessels && b.get() < self.num_vessels);
        unsafe { self.swap_vessels_unchecked(a, b) }
    }

    /// # Safety
    ///
    /// Both vessels must be in bounds.
    #[inline]
    pub unsafe fn swap_vessels_unchecked(&mut self, a: VesselIndex, b: VesselIndex) {
        debug_assert!(a.get() < self.num_vessels && b.get() < self.num_vessels);

        if a == b {
            return;
        }
        unsafe { self.arena.swap_nodes_unchecked(a.get(), b.get()) };

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
        assert!(
            a_first.get() < self.num_vessels
                && a_last.get() < self.num_vessels
                && b_first.get() < self.num_vessels
                && b_last.get() < self.num_vessels
        );
        unsafe { self.swap_segments_unchecked(a_first, a_last, b_first, b_last) }
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
        if a_first == b_first {
            return;
        }

        let berth_a = *unsafe { self.vessel_berth.get_unchecked(a_first.get()) };
        let berth_b = *unsafe { self.vessel_berth.get_unchecked(b_first.get()) };

        unsafe {
            self.arena.swap_segments_unchecked(
                a_first.get(),
                a_last.get(),
                b_first.get(),
                b_last.get(),
            )
        };

        if berth_a != berth_b {
            unsafe { self.update_segment_berth_unchecked(a_first.get(), a_last.get(), berth_b) };
            unsafe { self.update_segment_berth_unchecked(b_first.get(), b_last.get(), berth_a) };
        }
    }

    /// Reverses a contiguous segment of vessels.
    ///
    /// # Panics
    ///
    /// Panics if either vessel is out of bounds.
    #[inline]
    pub fn reverse_segment(&mut self, first: VesselIndex, last: VesselIndex) {
        assert!(first.get() < self.num_vessels && last.get() < self.num_vessels);
        unsafe { self.reverse_segment_unchecked(first, last) }
    }

    /// # Safety
    ///
    /// Both vessels must be in bounds and form a valid contiguous segment.
    #[inline]
    pub unsafe fn reverse_segment_unchecked(&mut self, first: VesselIndex, last: VesselIndex) {
        unsafe {
            self.arena
                .reverse_segment_unchecked(first.get(), last.get())
        };
        // No berth updates needed — reversal is purely intra-berth.
    }

    /// Relocates a vessel to immediately follow another vessel.
    ///
    /// # Panics
    ///
    /// Panics if either vessel is out of bounds.
    #[inline]
    pub fn relocate_after(&mut self, vessel: VesselIndex, anchor: VesselIndex) {
        assert!(vessel.get() < self.num_vessels && anchor.get() < self.num_vessels);
        unsafe { self.relocate_after_unchecked(vessel, anchor) }
    }

    /// # Safety
    ///
    /// Both vessels must be in bounds.
    #[inline]
    pub unsafe fn relocate_after_unchecked(&mut self, vessel: VesselIndex, anchor: VesselIndex) {
        let old_berth = *unsafe { self.vessel_berth.get_unchecked(vessel.get()) };
        let new_berth = *unsafe { self.vessel_berth.get_unchecked(anchor.get()) };

        unsafe {
            self.arena
                .relocate_after_unchecked(vessel.get(), anchor.get())
        };

        if old_berth != new_berth {
            unsafe { self.transfer_vessel_berth_unchecked(vessel.get(), old_berth, new_berth) };
        }
    }

    /// Relocates a vessel to immediately precede another vessel.
    ///
    /// # Panics
    ///
    /// Panics if either vessel is out of bounds.
    #[inline]
    pub fn relocate_before(&mut self, vessel: VesselIndex, reference: VesselIndex) {
        assert!(vessel.get() < self.num_vessels && reference.get() < self.num_vessels);
        unsafe { self.relocate_before_unchecked(vessel, reference) }
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
        let old_berth = *unsafe { self.vessel_berth.get_unchecked(vessel.get()) };
        let new_berth = *unsafe { self.vessel_berth.get_unchecked(reference.get()) };

        unsafe {
            self.arena
                .relocate_before_unchecked(vessel.get(), reference.get())
        };

        if old_berth != new_berth {
            unsafe { self.transfer_vessel_berth_unchecked(vessel.get(), old_berth, new_berth) };
        }
    }

    /// Relocates a vessel to the head of a berth.
    ///
    /// # Panics
    ///
    /// Panics if `vessel` or `berth` is out of bounds.
    #[inline]
    pub fn relocate_to_head(&mut self, vessel: VesselIndex, berth: BerthIndex) {
        assert!(vessel.get() < self.num_vessels && berth.get() < self.num_berths);
        unsafe { self.relocate_to_head_unchecked(vessel, berth) }
    }

    /// # Safety
    ///
    /// `vessel` and `berth` must be in bounds.
    #[inline]
    pub unsafe fn relocate_to_head_unchecked(&mut self, vessel: VesselIndex, berth: BerthIndex) {
        let old_berth = *unsafe { self.vessel_berth.get_unchecked(vessel.get()) };
        let sentinel = self.sentinel(berth);

        unsafe { self.arena.relocate_after_unchecked(vessel.get(), sentinel) };

        if old_berth != berth {
            unsafe { self.transfer_vessel_berth_unchecked(vessel.get(), old_berth, berth) };
        }
    }

    /// Relocates a vessel to the tail of a berth.
    ///
    /// # Panics
    ///
    /// Panics if `vessel` or `berth` is out of bounds.
    #[inline]
    pub fn relocate_to_tail(&mut self, vessel: VesselIndex, berth: BerthIndex) {
        assert!(vessel.get() < self.num_vessels && berth.get() < self.num_berths);
        unsafe { self.relocate_to_tail_unchecked(vessel, berth) }
    }

    /// # Safety
    ///
    /// `vessel` and `berth` must be in bounds.
    #[inline]
    pub unsafe fn relocate_to_tail_unchecked(&mut self, vessel: VesselIndex, berth: BerthIndex) {
        let old_berth = *unsafe { self.vessel_berth.get_unchecked(vessel.get()) };
        let sentinel = self.sentinel(berth);
        let tail = unsafe { self.arena.prev_unchecked(sentinel) };

        unsafe { self.arena.relocate_after_unchecked(vessel.get(), tail) };

        if old_berth != berth {
            unsafe { self.transfer_vessel_berth_unchecked(vessel.get(), old_berth, berth) };
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
        assert!(
            first.get() < self.num_vessels
                && last.get() < self.num_vessels
                && anchor.get() < self.num_vessels
        );
        unsafe { self.relocate_segment_after_unchecked(first, last, anchor) }
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
        unsafe {
            self.arena
                .relocate_segment_after_unchecked(first.get(), last.get(), anchor.get())
        };

        let new_berth = *unsafe { self.vessel_berth.get_unchecked(anchor.get()) };
        unsafe { self.update_segment_berth_unchecked(first.get(), last.get(), new_berth) };
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
        assert!(
            first.get() < self.num_vessels
                && last.get() < self.num_vessels
                && reference.get() < self.num_vessels
        );
        unsafe { self.relocate_segment_before_unchecked(first, last, reference) }
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
        unsafe {
            self.arena
                .relocate_segment_before_unchecked(first.get(), last.get(), reference.get())
        };

        let new_berth = *unsafe { self.vessel_berth.get_unchecked(reference.get()) };
        unsafe { self.update_segment_berth_unchecked(first.get(), last.get(), new_berth) };
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
        assert!(
            first.get() < self.num_vessels
                && last.get() < self.num_vessels
                && berth.get() < self.num_berths
        );
        unsafe { self.relocate_segment_to_head_unchecked(first, last, berth) }
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
        let sentinel = self.sentinel(berth);
        unsafe {
            self.arena
                .relocate_segment_after_unchecked(first.get(), last.get(), sentinel)
        };
        unsafe { self.update_segment_berth_unchecked(first.get(), last.get(), berth) };
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
        assert!(
            first.get() < self.num_vessels
                && last.get() < self.num_vessels
                && berth.get() < self.num_berths
        );
        unsafe { self.relocate_segment_to_tail_unchecked(first, last, berth) }
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
        let sentinel = self.sentinel(berth);
        let tail = unsafe { self.arena.prev_unchecked(sentinel) };
        unsafe {
            self.arena
                .relocate_segment_after_unchecked(first.get(), last.get(), tail)
        };
        unsafe { self.update_segment_berth_unchecked(first.get(), last.get(), berth) };
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
