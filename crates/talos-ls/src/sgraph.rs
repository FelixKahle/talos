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
//! To evaluate thousands of neighborhood moves per second, operations like swapping vessels,
//! relocating segments, or reversing sequences must be `O(1)` and entirely free of heap allocations.
//!
//! `ScheduleGraph` achieves this by using a **flat-array, arena-backed doubly-linked list** design.
//! Instead of using standard Rust `Vec<Vec<VesselIndex>>` or pointer-based nodes, the graph maintains
//! two parallel `Vec<VesselIndex>` arrays: `prev` and `next`.
//!
//! Because the graph strictly consists of valid, closed rings, these two arrays are exact
//! mathematical inverses: `prev[next[v]] == v` for all `v`.
//!
//! # Sentinels and Memory Layout
//!
//! Every berth is represented as an independent, circular ring containing exactly one "sentinel" node
//! and zero or more real vessel nodes. An empty berth is simply a sentinel node whose `prev` and `next`
//! pointers point to itself.
//!
//! The index space of the internal arrays is rigidly partitioned to implicitly encode whether a node
//! is a real vessel or a sentinel, avoiding the need for `Option` or `enum` wrappers:
//!
//! ```text
//! Index Space: [ 0, 1, ..., N-1 | N, N+1, ..., N+B-1 ]
//!              |__Real Vessels__|___Berth Sentinels__|
//! ```
//!
//! - **Real Vessels:** Indices `< num_vessels`.
//! - **Sentinels:** Indices `>= num_vessels`. The explicit sentinel index for a given `BerthIndex` `b`
//!   is calculated as `num_vessels + b.get()`.
//!
//! # State Synchronization
//!
//! The highly-optimized `_unchecked` relocation and swapping methods handle rewiring the raw $O(1)$
//! topology pointers first, and then perform a minimal $O(N)$ pass to synchronize these state vectors
//! *only* if the vessels crossed berth boundaries.

use std::iter::FusedIterator;
use talos_model::index::{BerthIndex, VesselIndex};

// ----------------------------------------------------------------
// ScheduleGraphEdge
// ----------------------------------------------------------------

/// A directed edge between two vessels in the schedule graph, representing a direct precedence
/// on the same berth. This struct is used for iterating over the edges of a berth's vessel sequence.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ScheduleGraphEdge {
    pub from: VesselIndex,
    pub to: VesselIndex,
}

impl std::fmt::Display for ScheduleGraphEdge {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "Edge(V{} -> V{})",
            self.from.get(),
            self.to.get()
        )
    }
}

// ----------------------------------------------------------------
// ScheduleGraphFullEdge
// ----------------------------------------------------------------

/// A directed edge between two vessels in the schedule graph, annotated with the berth on which
/// the precedence holds. This struct is used for iterating over all edges in the graph.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ScheduleGraphFullEdge {
    pub from: VesselIndex,
    pub to: VesselIndex,
    pub on_berth: BerthIndex,
}

impl std::fmt::Display for ScheduleGraphFullEdge {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "Edge(V{} -> V{} on Berth {})",
            self.from.get(),
            self.to.get(),
            self.on_berth.get()
        )
    }
}

// ----------------------------------------------------------------
// VesselSequenceIter
// ----------------------------------------------------------------

/// Iterator over a vessel sequence assigned to a berth, from first to last.
#[derive(Clone, PartialEq, Eq)]
pub struct VesselSequenceIter<'a> {
    /// The `next` array borrowed from the owning `ScheduleGraph`.
    next_pointers: &'a [VesselIndex],
    /// Current cursor into the ring.
    current_node: VesselIndex,
    /// The remaining number of vessels to yield.
    remaining_vessels: usize,
}

impl<'a> Iterator for VesselSequenceIter<'a> {
    type Item = VesselIndex;

    #[inline(always)]
    fn next(&mut self) -> Option<VesselIndex> {
        if self.remaining_vessels == 0 {
            return None;
        }
        let current_vessel = self.current_node;
        // SAFETY: The graph invariant guarantees `current_vessel.get()` is a valid index
        // into `next_pointers` (it is either a vessel index < num_vessels or a sentinel
        // index < num_vessels + num_berths, both within the allocation).
        self.current_node = *unsafe { self.next_pointers.get_unchecked(current_vessel.get()) };
        self.remaining_vessels -= 1;

        // Verify the cursor we just loaded is within the allocation, catching
        // berth_vessel_count desynchronization before it causes an out-of-bounds
        // read on the next call.
        debug_assert!(
            self.remaining_vessels == 0 || self.current_node.get() < self.next_pointers.len(),
            "invariant violation: vessel sequence iterator cursor out of bounds: current_node = {}, next_pointers_len = {}",
            self.current_node.get(),
            self.next_pointers.len()
        );

        Some(current_vessel)
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining_vessels, Some(self.remaining_vessels))
    }
}

impl<'a> FusedIterator for VesselSequenceIter<'a> {}
impl<'a> ExactSizeIterator for VesselSequenceIter<'a> {}

// ----------------------------------------------------------------
// VesselSequenceRevIter
// ----------------------------------------------------------------

/// Reverse iterator over the vessels assigned to a berth, from last to first.
#[derive(Clone, PartialEq, Eq)]
pub struct VesselSequenceRevIter<'a> {
    /// The `prev` array borrowed from the owning `ScheduleGraph`.
    prev_pointers: &'a [VesselIndex],
    /// Current cursor into the ring. Equals `sentinel_node` when exhausted.
    current_node: VesselIndex,
    /// The remaining numbers of vessels to yield.
    remaining_vessels: usize,
}

impl<'a> Iterator for VesselSequenceRevIter<'a> {
    type Item = VesselIndex;

    #[inline(always)]
    fn next(&mut self) -> Option<VesselIndex> {
        if self.remaining_vessels == 0 {
            return None;
        }
        let current_vessel = self.current_node;
        // SAFETY: The graph invariant guarantees `current_vessel.get()` is a valid index
        // into `prev_pointers` (it is either a vessel index < num_vessels or a sentinel
        // index < num_vessels + num_berths, both within the allocation).
        self.current_node = *unsafe { self.prev_pointers.get_unchecked(current_vessel.get()) };
        self.remaining_vessels -= 1;

        debug_assert!(
            self.remaining_vessels == 0 || self.current_node.get() < self.prev_pointers.len(),
            "invariant violation: vessel sequence reverse iterator cursor out of bounds: current_node = {}, prev_pointers_len = {}",
            self.current_node.get(),
            self.prev_pointers.len()
        );

        Some(current_vessel)
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining_vessels, Some(self.remaining_vessels))
    }
}

impl<'a> FusedIterator for VesselSequenceRevIter<'a> {}
impl<'a> ExactSizeIterator for VesselSequenceRevIter<'a> {}

// ----------------------------------------------------------------
// BerthEdgeIter
// ----------------------------------------------------------------

/// Iterator over the edges (adjacent vessel pairs) within a single berth.
#[derive(Clone, Debug)]
pub struct BerthEdgeIter<'a> {
    /// The `next` array borrowed from the owning `ScheduleGraph`.
    next_pointers: &'a [VesselIndex],
    /// The left-hand side of the edge we are about to yield.
    current_node: VesselIndex,
    /// The remaining number of edges to yield.
    remaining_edges: usize,
}

impl<'a> Iterator for BerthEdgeIter<'a> {
    type Item = ScheduleGraphEdge;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining_edges == 0 {
            return None;
        }

        let from = self.current_node;
        // SAFETY: The graph invariant guarantees `from.get()` is a valid index.
        let to = *unsafe { self.next_pointers.get_unchecked(from.get()) };

        self.current_node = to;
        self.remaining_edges -= 1;

        Some(ScheduleGraphEdge { from, to })
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining_edges, Some(self.remaining_edges))
    }
}

impl<'a> FusedIterator for BerthEdgeIter<'a> {}
impl<'a> ExactSizeIterator for BerthEdgeIter<'a> {}

// ----------------------------------------------------------------
// AllEdgeIter
// ----------------------------------------------------------------

/// Iterator over all edges in the graph, across all berths.
/// Yields edges in berth order, and within each berth from first to last.
#[derive(Clone, Debug)]
pub struct AllEdgeIter<'a> {
    next_pointers: &'a [VesselIndex],
    vessel_berth: &'a [BerthIndex],
    current_vessel: usize,
}

impl<'a> Iterator for AllEdgeIter<'a> {
    type Item = ScheduleGraphFullEdge;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        while self.current_vessel < self.vessel_berth.len() {
            let from = VesselIndex::new(self.current_vessel);
            self.current_vessel += 1;

            // SAFETY: current_vessel is strictly < vessel_berth.len() (which is num_vessels)
            let to = *unsafe { self.next_pointers.get_unchecked(from.get()) };

            // If `to` is a real vessel (not a sentinel), we found a valid edge!
            if to.get() < self.vessel_berth.len() {
                let on_berth = *unsafe { self.vessel_berth.get_unchecked(from.get()) };
                return Some(ScheduleGraphFullEdge { from, to, on_berth });
            }
            // Otherwise, `from` was the last vessel in its berth. Skip and continue.
        }

        None
    }
}

impl<'a> FusedIterator for AllEdgeIter<'a> {}

// ----------------------------------------------------------------
// ScheduleGraph
// ----------------------------------------------------------------

/// Per-berth vessel ordering — the genotype of the local search.
///
/// `ScheduleGraph` maintains the explicit sequencing of vessels assigned to multiple berths.
/// It acts as the primary state representation (genotype) for local search algorithms
/// (such as Simulated Annealing or Tabu Search) applied to vessel routing and scheduling.
///
/// # Architecture
///
/// To achieve `O(1)` neighborhood moves (swaps, relocations, reversals) without the
/// overhead of heap allocations or pointer chasing, this graph is implemented as a
/// flat-array, arena-backed collection of circular, doubly-linked lists.
///
/// Each berth is represented as an independent ring containing exactly one "sentinel"
/// node and zero or more vessel nodes. An empty berth is simply a sentinel node whose
/// `prev` and `next` pointers point to itself.
///
/// # Memory Layout
///
/// The underlying arrays (`prev` and `next`) have a length of `num_vessels + num_berths`.
/// The index space is strictly partitioned:
///
/// ```text
/// Index Space: [ 0, 1, ..., N-1 | N, N+1, ..., N+B-1 ]
///              |__Real Vessels__|___Berth Sentinels__|
/// ```
///
/// - **Real Vessels:** Indices `< num_vessels`.
/// - **Sentinels:** Indices `>= num_vessels`. The sentinel for a given `BerthIndex` `b`
///   is located exactly at index `num_vessels + b`.
///
/// # State Synchronization
///
/// In addition to the topology arrays (`prev` and `next`), the graph maintains two
/// auxiliary state vectors for `O(1)` berth lookups:
///
/// - `vessel_berth[v]`: The berth to which vessel `v` is currently assigned.
/// - `berth_vessel_count[b]`: The number of vessels currently assigned to berth `b`.
///
/// These vectors are kept in sync by all mutation methods. Single-vessel operations
/// update them in `O(1)`. Segment operations update them in `O(|segment|)`, which
/// is the minimum possible since every relocated vessel must have its berth updated.
///
/// # Safety and `_unchecked` Methods
///
/// For maximum performance during high-throughput local search iterations, this struct
/// exposes many `_unchecked` methods that omit bounds checking.
///
/// **Calling `_unchecked` methods is only safe if:**
/// 1. The provided `VesselIndex` and `BerthIndex` are strictly within their respective
///    bounds (`< num_vessels` and `< num_berths`).
/// 2. When dealing with segments (`first` to `last`), the nodes must be sequentially
///    linked in the same berth, and the `target` node must **not** be part of that segment.
#[derive(Clone)]
pub struct ScheduleGraph {
    /// O(1) lookup: the node immediately preceding this one.
    ///
    /// Layout: `[0 .. num_vessels]` are real vessels,
    ///         `[num_vessels .. num_vessels + num_berths]` are berth sentinels.
    prev: Vec<VesselIndex>,

    /// O(1) lookup: the node immediately following this one.
    ///
    /// Layout matches `prev`.
    next: Vec<VesselIndex>,

    /// O(1) lookup: the berth to which each vessel is assigned.
    ///
    /// Length: `num_vessels`. Indexed by `VesselIndex::get()`.
    vessel_berth: Vec<BerthIndex>,

    /// O(1) lookup: the number of vessels assigned to each berth.
    ///
    /// Length: `num_berths`. Indexed by `BerthIndex::get()`.
    berth_vessel_count: Vec<usize>,

    /// Total logical vessels. Used as the offset to calculate sentinel indices.
    num_vessels: usize,

    /// Total logical berths.
    num_berths: usize,
}

impl PartialEq for ScheduleGraph {
    fn eq(&self, other: &Self) -> bool {
        if self.num_berths() != other.num_berths() || self.num_vessels() != other.num_vessels() {
            return false;
        }
        for berth_idx in 0..self.num_berths() {
            let berth = BerthIndex::new(berth_idx);
            let self_sequence = self.vessel_sequence_iter(berth);
            let other_sequence = other.vessel_sequence_iter(berth);
            if !self_sequence.eq(other_sequence) {
                return false;
            }
        }
        true
    }
}

impl Eq for ScheduleGraph {}

impl std::fmt::Debug for ScheduleGraph {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        struct Sequences<'a>(&'a ScheduleGraph);

        impl<'a> std::fmt::Debug for Sequences<'a> {
            fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                let mut map = formatter.debug_map();
                for berth_idx in 0..self.0.num_berths() {
                    let berth = BerthIndex::new(berth_idx);
                    let sequence: Vec<_> = self.0.vessel_sequence_iter(berth).collect();
                    map.entry(&berth, &sequence);
                }
                map.finish()
            }
        }

        formatter
            .debug_struct("ScheduleGraph")
            .field("num_vessels", &self.num_vessels)
            .field("num_berths", &self.num_berths)
            .field("sequences", &Sequences(self))
            .finish()
    }
}

impl std::fmt::Display for ScheduleGraph {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            formatter,
            "ScheduleGraph (Berths: {}, Total Vessels: {})",
            self.num_berths(),
            self.num_vessels()
        )?;

        for berth_idx in 0..self.num_berths() {
            let berth = BerthIndex::new(berth_idx);
            write!(formatter, "  Berth {}: ", berth_idx)?;

            let mut iter = self.vessel_sequence_iter(berth);
            if let Some(first_vessel) = iter.next() {
                write!(formatter, "V{}", first_vessel.get())?;
                for vessel in iter {
                    write!(formatter, " -> V{}", vessel.get())?;
                }
                writeln!(formatter)?;
            } else {
                writeln!(formatter, "(empty)")?;
            }
        }
        Ok(())
    }
}

impl std::hash::Hash for ScheduleGraph {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.num_berths().hash(state);
        for berth_idx in 0..self.num_berths() {
            let berth = BerthIndex::new(berth_idx);
            for vessel in self.vessel_sequence_iter(berth) {
                vessel.hash(state);
            }
        }
    }
}

impl ScheduleGraph {
    /// Overwrites the current `ScheduleGraph` with data from parallel slices.
    ///
    /// This method is designed for high-performance initialization. It reuses the memory
    /// allocations of the underlying arrays (`prev` and `next`)
    /// by clearing and extending them.
    ///
    /// # Panics
    ///
    /// Panics if `berths.len() != start_times.len()`, or if any assigned berth
    /// in the slice is out of bounds (i.e., `>= num_berths`).
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
            "called `ScheduleGraph::overwrite_from_slices` with mismatched slice lengths: berths.len() = {}, start_times.len() = {}",
            berths.len(),
            start_times.len()
        );

        self.num_vessels = berths.len();
        self.num_berths = num_berths;

        let total_nodes = self.num_vessels + self.num_berths;
        self.prev.clear();
        self.prev.resize(total_nodes, VesselIndex::new(0));
        self.next.clear();
        self.next.resize(total_nodes, VesselIndex::new(0));

        // Initialize vessel_berth and berth_vessel_count.
        self.vessel_berth.clear();
        self.vessel_berth
            .resize(self.num_vessels, BerthIndex::new(0));
        self.berth_vessel_count.clear();
        self.berth_vessel_count.resize(self.num_berths, 0);

        for berth_idx in 0..self.num_berths {
            let sentinel_node = self.sentinel(BerthIndex::new(berth_idx));
            // SAFETY: berth_idx < num_berths, so sentinel_node < total_nodes.
            unsafe {
                *self.next.get_unchecked_mut(sentinel_node.get()) = sentinel_node;
                *self.prev.get_unchecked_mut(sentinel_node.get()) = sentinel_node;
            }
        }

        if self.num_vessels == 0 {
            return;
        }

        for &berth in berths {
            assert!(
                berth.get() < self.num_berths,
                "called `ScheduleGraph::overwrite_from_slices` with out-of-bounds berth in input slice: berth = {}, num_berths = {}",
                berth.get(),
                self.num_berths
            );
        }

        // Populate vessel_berth and berth_vessel_count from the input slice.
        for vessel_idx in 0..self.num_vessels {
            // SAFETY: vessel_idx < num_vessels == berths.len(). berth.get() < num_berths (asserted above).
            let berth = *unsafe { berths.get_unchecked(vessel_idx) };
            *unsafe { self.vessel_berth.get_unchecked_mut(vessel_idx) } = berth;
            *unsafe { self.berth_vessel_count.get_unchecked_mut(berth.get()) } += 1;
        }

        for vessel_idx in 0..self.num_vessels {
            // SAFETY: vessel_idx < num_vessels <= total_nodes
            unsafe { *self.prev.get_unchecked_mut(vessel_idx) = VesselIndex::new(vessel_idx) };
        }

        self.prev[0..self.num_vessels].sort_unstable_by(|&left_vessel, &right_vessel| {
            // SAFETY: We populated left_vessel and right_vessel with 0..num_vessels. sort_unstable_by only permutes them.
            // Therefore, left_vessel.get() and right_vessel.get() are strictly < num_vessels.
            // Because berths.len() == start_times.len() == num_vessels, these accesses are strictly in-bounds.
            unsafe {
                berths
                    .get_unchecked(left_vessel.get())
                    .cmp(berths.get_unchecked(right_vessel.get()))
                    .then_with(|| {
                        start_times
                            .get_unchecked(left_vessel.get())
                            .cmp(start_times.get_unchecked(right_vessel.get()))
                    })
            }
        });

        let mut current_berth = None;
        let mut current_tail_node = VesselIndex::new(0);

        for sorted_idx in 0..self.num_vessels {
            // SAFETY: sorted_idx < num_vessels, vessel.get() < num_vessels
            let vessel = unsafe { *self.prev.get_unchecked(sorted_idx) };
            let berth = unsafe { *berths.get_unchecked(vessel.get()) };

            if Some(berth) != current_berth {
                if let Some(previous_berth) = current_berth {
                    // SAFETY: current_tail_node is either a vessel (< num_vessels) or a sentinel (< total_nodes)
                    unsafe {
                        *self.next.get_unchecked_mut(current_tail_node.get()) =
                            self.sentinel(previous_berth)
                    };
                }
                current_berth = Some(berth);
                current_tail_node = self.sentinel(berth);
            }
            // SAFETY: current_tail_node is valid
            unsafe { *self.next.get_unchecked_mut(current_tail_node.get()) = vessel };
            current_tail_node = vessel;
        }

        if let Some(final_berth) = current_berth {
            // SAFETY: current_tail_node is valid
            unsafe {
                *self.next.get_unchecked_mut(current_tail_node.get()) = self.sentinel(final_berth)
            };
        }

        for node_idx in 0..total_nodes {
            // SAFETY: node_idx < total_nodes. successor was mapped to values strictly < total_nodes in the loops above.
            let successor = unsafe { *self.next.get_unchecked(node_idx) };
            unsafe { *self.prev.get_unchecked_mut(successor.get()) = VesselIndex::new(node_idx) };
        }
    }

    /// Overwrites the current `ScheduleGraph` with data from another graph.
    #[inline]
    pub fn overwrite_from_graph(&mut self, other: &ScheduleGraph) {
        self.prev.clone_from(&other.prev);
        self.next.clone_from(&other.next);
        self.vessel_berth.clone_from(&other.vessel_berth);
        self.berth_vessel_count
            .clone_from(&other.berth_vessel_count);

        self.num_vessels = other.num_vessels;
        self.num_berths = other.num_berths;
    }

    /// Creates a new `ScheduleGraph` from parallel slices of berths and start times.
    #[inline]
    pub fn from_slices<T>(berths: &[BerthIndex], start_times: &[T], num_berths: usize) -> Self
    where
        T: Ord,
    {
        let mut graph = Self {
            prev: Vec::new(),
            next: Vec::new(),
            vessel_berth: Vec::new(),
            berth_vessel_count: Vec::new(),
            num_vessels: 0,
            num_berths: 0,
        };
        graph.overwrite_from_slices(berths, start_times, num_berths);
        graph
    }

    /// Calculates the explicit sentinel node index for a given berth.
    ///
    /// The sentinel node acts as the fixed anchor for a berth's circular linked list.
    ///
    /// ```text
    /// [ Vessel 0 | Vessel 1 | ... | Sentinel 0 | Sentinel 1 ]
    ///                               ^^^^^^^^^^
    ///                               Calculated Index
    /// ```
    #[inline(always)]
    fn sentinel(&self, berth: BerthIndex) -> VesselIndex {
        debug_assert!(
            berth.get() < self.num_berths,
            "called `ScheduleGraph::sentinel` with out-of-bounds berth: berth = {}, num_berths = {}",
            berth.get(),
            self.num_berths
        );

        VesselIndex::new(self.num_vessels + berth.get())
    }

    /// Returns true if the given node index corresponds to a sentinel (i.e., a berth anchor
    /// rather than a real vessel.
    #[inline]
    pub fn is_sentinel(&self, node: VesselIndex) -> bool {
        node.get() >= self.num_vessels
    }

    /// Detaches a node from its current position in the ring.
    ///
    /// After this call, the node's own `prev` and `next` slots are **stale**. The
    /// caller must reinsert it or otherwise fix them up before the graph is observed again.
    ///
    /// ```text
    /// Before:
    /// ... <-> PrevNode <-> Node <-> NextNode <-> ...
    ///
    /// After:
    /// ... <-> PrevNode <----------> NextNode <-> ...
    ///                      Node (stale pointers)
    /// ```
    ///
    /// # Safety
    ///
    /// `node_to_extract` must be a valid index within the `prev` and `next` arrays.
    #[inline(always)]
    unsafe fn extract_node_unchecked(&mut self, node_to_extract: VesselIndex) {
        debug_assert!(
            node_to_extract.get() < self.prev.len(),
            "called `extract_node_unchecked` with out-of-bounds node index: node = {}, prev_len = {}",
            node_to_extract.get(),
            self.prev.len()
        );

        let predecessor = *unsafe { self.prev.get_unchecked(node_to_extract.get()) };
        let successor = *unsafe { self.next.get_unchecked(node_to_extract.get()) };
        *unsafe { self.next.get_unchecked_mut(predecessor.get()) } = successor;
        *unsafe { self.prev.get_unchecked_mut(successor.get()) } = predecessor;
    }

    /// Inserts a node immediately following the `insertion_point` node.
    ///
    /// ```text
    /// Before:
    /// ... <-> After <----------> NextNode <-> ...
    ///                  Node
    ///
    /// After:
    /// ... <-> After <-> Node <-> NextNode <-> ...
    /// ```
    ///
    /// # Safety
    ///
    /// Both `node_to_insert` and `insertion_point` must be valid indices within `prev` and `next`.
    /// `node_to_insert` must not currently be linked into any active ring (i.e., it must have
    /// been freshly extracted).
    #[inline(always)]
    unsafe fn insert_node_after_unchecked(
        &mut self,
        node_to_insert: VesselIndex,
        insertion_point: VesselIndex,
    ) {
        debug_assert!(
            node_to_insert.get() < self.prev.len(),
            "called `insert_node_after_unchecked` with out-of-bounds node index: node = {}, prev_len = {}",
            node_to_insert.get(),
            self.prev.len()
        );
        debug_assert!(
            insertion_point.get() < self.prev.len(),
            "called `insert_node_after_unchecked` with out-of-bounds after index: after = {}, prev_len = {}",
            insertion_point.get(),
            self.prev.len()
        );

        let successor_of_insertion_point =
            *unsafe { self.next.get_unchecked(insertion_point.get()) };
        *unsafe { self.next.get_unchecked_mut(insertion_point.get()) } = node_to_insert;
        *unsafe { self.prev.get_unchecked_mut(node_to_insert.get()) } = insertion_point;
        *unsafe { self.next.get_unchecked_mut(node_to_insert.get()) } =
            successor_of_insertion_point;
        *unsafe {
            self.prev
                .get_unchecked_mut(successor_of_insertion_point.get())
        } = node_to_insert;
    }

    /// Updates `vessel_berth` and `berth_vessel_count` for every vessel in a contiguous
    /// segment `[segment_first .. segment_last]` that has been moved to `new_berth`.
    ///
    /// This method walks the segment via the `next` pointers (whose internal links are
    /// guaranteed to be intact after any relocation or swap operation).
    ///
    /// This is a no-op if the segment is already assigned to `new_berth`.
    ///
    /// # Safety
    ///
    /// - `segment_first` and `segment_last` must be valid vessel indices (`< self.num_vessels`).
    /// - They must form a valid, contiguous segment linked via `next` pointers.
    /// - `new_berth.get() < self.num_berths`.
    #[inline(always)]
    unsafe fn update_segment_berth_unchecked(
        &mut self,
        segment_first: VesselIndex,
        segment_last: VesselIndex,
        new_berth: BerthIndex,
    ) {
        debug_assert!(
            segment_first.get() < self.num_vessels && segment_last.get() < self.num_vessels,
            "called `update_segment_berth_unchecked` with out-of-bounds vessel: first = {}, last = {}, num_vessels = {}",
            segment_first.get(),
            segment_last.get(),
            self.num_vessels
        );
        debug_assert!(
            new_berth.get() < self.num_berths,
            "called `update_segment_berth_unchecked` with out-of-bounds berth: berth = {}, num_berths = {}",
            new_berth.get(),
            self.num_berths
        );

        let old_berth = *unsafe { self.vessel_berth.get_unchecked(segment_first.get()) };
        if old_berth == new_berth {
            return;
        }

        let mut count = 0usize;
        let mut current = segment_first;
        loop {
            let next_node = *unsafe { self.next.get_unchecked(current.get()) };
            *unsafe { self.vessel_berth.get_unchecked_mut(current.get()) } = new_berth;
            count += 1;
            if current == segment_last {
                break;
            }
            current = next_node;
        }
        *unsafe { self.berth_vessel_count.get_unchecked_mut(old_berth.get()) } -= count;
        *unsafe { self.berth_vessel_count.get_unchecked_mut(new_berth.get()) } += count;
    }

    /// Updates `vessel_berth` and `berth_vessel_count` for a single vessel
    /// that has been moved from `old_berth` to `new_berth`.
    ///
    /// Correct even when `old_berth == new_berth` (branchless no-op).
    ///
    /// # Safety
    ///
    /// The caller must ensure `vessel.get() < self.num_vessels`,
    /// `old_berth.get() < self.num_berths`, and `new_berth.get() < self.num_berths`.
    #[inline(always)]
    unsafe fn transfer_vessel_berth_unchecked(
        &mut self,
        vessel: VesselIndex,
        old_berth: BerthIndex,
        new_berth: BerthIndex,
    ) {
        debug_assert!(
            vessel.get() < self.num_vessels,
            "called `transfer_vessel_berth_unchecked` with out-of-bounds vessel: vessel = {}, num_vessels = {}",
            vessel.get(),
            self.num_vessels
        );
        debug_assert!(
            old_berth.get() < self.num_berths && new_berth.get() < self.num_berths,
            "called `transfer_vessel_berth_unchecked` with out-of-bounds berth: old = {}, new = {}, num_berths = {}",
            old_berth.get(),
            new_berth.get(),
            self.num_berths
        );

        *unsafe { self.vessel_berth.get_unchecked_mut(vessel.get()) } = new_berth;
        *unsafe { self.berth_vessel_count.get_unchecked_mut(old_berth.get()) } -= 1;
        *unsafe { self.berth_vessel_count.get_unchecked_mut(new_berth.get()) } += 1;
    }

    /// Returns true if the given berth has no vessels assigned to it.
    ///
    /// # Panics
    ///
    /// Panics if `berth` is out of bounds, meaning `berth.get() >= self.num_berths`.
    #[inline]
    pub fn is_empty(&self, berth: BerthIndex) -> bool {
        assert!(
            berth.get() < self.num_berths,
            "called `ScheduleGraph::is_empty` with out-of-bounds berth: berth = {}, num_berths = {}",
            berth.get(),
            self.num_berths
        );
        unsafe { self.is_empty_unchecked(berth) }
    }

    /// Returns true if the given berth has no vessels assigned to it.
    ///
    /// # Safety
    ///
    /// The caller must ensure `berth.get() < self.num_berths`.
    #[inline(always)]
    pub unsafe fn is_empty_unchecked(&self, berth: BerthIndex) -> bool {
        debug_assert!(
            berth.get() < self.num_berths,
            "called `ScheduleGraph::is_empty_unchecked` with out-of-bounds berth: berth = {}, num_berths = {}",
            berth.get(),
            self.num_berths
        );

        *unsafe { self.berth_vessel_count.get_unchecked(berth.get()) } == 0
    }

    /// Returns the total number of logical berths in this schedule graph.
    #[inline(always)]
    pub fn num_berths(&self) -> usize {
        self.num_berths
    }

    /// Returns the total number of logical vessels in this schedule graph.
    #[inline(always)]
    pub fn num_vessels(&self) -> usize {
        self.num_vessels
    }

    /// Returns the berth to which the given vessel is currently assigned.
    ///
    /// # Panics
    ///
    /// Panics if `vessel` is out of bounds, meaning `vessel.get() >= self.num_vessels`.
    #[inline]
    pub fn vessel_berth(&self, vessel: VesselIndex) -> BerthIndex {
        assert!(
            vessel.get() < self.num_vessels,
            "called `ScheduleGraph::vessel_berth` with out-of-bounds vessel: vessel = {}, num_vessels = {}",
            vessel.get(),
            self.num_vessels
        );
        unsafe { self.vessel_berth_unchecked(vessel) }
    }

    /// Returns the berth to which the given vessel is currently assigned.
    ///
    /// # Safety
    ///
    /// The caller must ensure `vessel.get() < self.num_vessels`.
    /// No bounds checking is performed.
    #[inline(always)]
    pub unsafe fn vessel_berth_unchecked(&self, vessel: VesselIndex) -> BerthIndex {
        debug_assert!(
            vessel.get() < self.num_vessels,
            "called `ScheduleGraph::vessel_berth_unchecked` with out-of-bounds vessel: vessel = {}, num_vessels = {}",
            vessel.get(),
            self.num_vessels
        );

        *unsafe { self.vessel_berth.get_unchecked(vessel.get()) }
    }

    /// Returns the number of vessels currently assigned to the given berth.
    ///
    /// # Panics
    ///
    /// Panics if `berth` is out of bounds, meaning `berth.get() >= self.num_berths`.
    #[inline]
    pub fn vessel_count(&self, berth: BerthIndex) -> usize {
        assert!(
            berth.get() < self.num_berths,
            "called `ScheduleGraph::vessel_count` with out-of-bounds berth: berth = {}, num_berths = {}",
            berth.get(),
            self.num_berths
        );
        unsafe { self.vessel_count_unchecked(berth) }
    }

    /// Returns the number of vessels currently assigned to the given berth.
    ///
    /// # Safety
    ///
    /// The caller must ensure `berth.get() < self.num_berths`.
    /// No bounds checking is performed.
    #[inline(always)]
    pub unsafe fn vessel_count_unchecked(&self, berth: BerthIndex) -> usize {
        debug_assert!(
            berth.get() < self.num_berths,
            "called `ScheduleGraph::vessel_count_unchecked` with out-of-bounds berth: berth = {}, num_berths = {}",
            berth.get(),
            self.num_berths
        );

        *unsafe { self.berth_vessel_count.get_unchecked(berth.get()) }
    }

    /// Returns the raw internal `next` pointer (can be a vessel or a sentinel).
    ///
    /// # Note
    ///
    /// The returned `VesselIndex` may represent either a real vessel (if its index is `< num_vessels`)
    /// or a sentinel (if its index is `>= num_vessels`). The caller must interpret it accordingly.
    ///
    /// # Panics
    ///
    /// Panics if `node` is out of bounds, meaning `node.get() >= self.num_vessels + self.num_berths`.
    #[inline]
    pub fn raw_next(&self, node: VesselIndex) -> VesselIndex {
        debug_assert!(
            node.get() < self.next.len(),
            "called `ScheduleGraph::raw_next` with out-of-bounds node index: node = {}, next_len = {}",
            node.get(),
            self.next.len()
        );

        self.next[node.get()]
    }

    /// Returns the raw internal `next` pointer (can be a vessel or a sentinel).
    ///
    /// # Note
    ///
    /// The returned `VesselIndex` may represent either a real vessel (if its index is `< num_vessels`)
    /// or a sentinel (if its index is `>= num_vessels`). The caller must interpret it accordingly.
    ///
    /// # Safety
    ///
    /// The caller must ensure `node.get() < self.num_vessels + self.num_berths`.
    /// No bounds checking is performed.
    #[inline(always)]
    pub unsafe fn raw_next_unchecked(&self, node: VesselIndex) -> VesselIndex {
        debug_assert!(
            node.get() < self.next.len(),
            "called `ScheduleGraph::raw_next_unchecked` with out-of-bounds node index: node = {}, next_len = {}",
            node.get(),
            self.next.len()
        );

        *unsafe { self.next.get_unchecked(node.get()) }
    }

    /// Returns the raw internal `prev` pointer (can be a vessel or a sentinel).
    ///
    /// # Note
    ///
    /// The returned `VesselIndex` may represent either a real vessel (if its index is `< num_vessels`)
    /// or a sentinel (if its index is `>= num_vessels`). The caller must interpret it accordingly.
    ///
    /// # Panics
    ///
    /// Panics if `node` is out of bounds, meaning `node.get() >= self.num_vessels + self.num_berths`.
    #[inline]
    pub fn raw_prev(&self, node: VesselIndex) -> VesselIndex {
        debug_assert!(
            node.get() < self.prev.len(),
            "called `ScheduleGraph::raw_prev` with out-of-bounds node index: node = {}, prev_len = {}",
            node.get(),
            self.prev.len()
        );

        self.prev[node.get()]
    }

    /// Returns the raw internal `prev` pointer (can be a vessel or a sentinel).
    ///
    /// # Note
    ///
    /// The returned `VesselIndex` may represent either a real vessel (if its index is `< num_vessels`)
    /// or a sentinel (if its index is `>= num_vessels`). The caller
    /// must interpret it accordingly.
    ///
    /// # Safety
    ///
    /// The caller must ensure `node.get() < self.num_vessels + self.num_berths`.
    /// No bounds checking is performed.
    #[inline(always)]
    pub unsafe fn raw_prev_unchecked(&self, node: VesselIndex) -> VesselIndex {
        debug_assert!(
            node.get() < self.prev.len(),
            "called `ScheduleGraph::raw_prev_unchecked` with out-of-bounds node index: node = {}, prev_len = {}",
            node.get(),
            self.prev.len()
        );

        *unsafe { self.prev.get_unchecked(node.get()) }
    }

    /// Returns the first vessel in the sequence for the given berth,
    /// or `None` if the berth is empty.
    ///
    /// ```text
    /// [ Sentinel ] <-> [ First Vessel ] <-> ...
    ///                  ^^^^^^^^^^^^^^^^
    ///                  Returned Value
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `berth` is out of bounds.
    #[inline]
    pub fn first_vessel(&self, berth: BerthIndex) -> Option<VesselIndex> {
        assert!(
            berth.get() < self.num_berths,
            "called `ScheduleGraph::first_vessel` with out-of-bounds berth: berth = {}, num_berths = {}",
            berth.get(),
            self.num_berths
        );
        unsafe { self.first_vessel_unchecked(berth) }
    }

    /// Returns the first vessel in the sequence for the given berth,
    /// or `None` if the berth is empty.
    ///
    /// ```text
    /// [ Sentinel ] <-> [ First Vessel ] <-> ...
    ///                  ^^^^^^^^^^^^^^^^
    ///                  Returned Value
    /// ```
    ///
    /// # Safety
    ///
    /// The caller must ensure `berth.get() < self.num_berths`.
    #[inline(always)]
    pub unsafe fn first_vessel_unchecked(&self, berth: BerthIndex) -> Option<VesselIndex> {
        debug_assert!(
            berth.get() < self.num_berths,
            "called `ScheduleGraph::first_vessel_unchecked` with out-of-bounds berth: berth = {}, num_berths = {}",
            berth.get(),
            self.num_berths
        );
        let vessel = *unsafe { self.next.get_unchecked(self.sentinel(berth).get()) };
        (vessel.get() < self.num_vessels).then_some(vessel)
    }

    /// Returns the last vessel in the sequence for the given berth,
    /// or `None` if the berth is empty.
    ///
    /// ```text
    /// ... <-> [ Last Vessel ] <-> [ Sentinel ]
    ///         ^^^^^^^^^^^^^^^
    ///         Returned Value
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `berth` is out of bounds.
    #[inline]
    pub fn last_vessel(&self, berth: BerthIndex) -> Option<VesselIndex> {
        assert!(
            berth.get() < self.num_berths,
            "called `ScheduleGraph::last_vessel` with out-of-bounds berth: berth = {}, num_berths = {}",
            berth.get(),
            self.num_berths
        );
        unsafe { self.last_vessel_unchecked(berth) }
    }

    /// Returns the last vessel in the sequence for the given berth,
    /// or `None` if the berth is empty.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `berth` is a valid berth index, meaning
    /// `berth.get() < self.num_berths`. No bounds checking is performed.
    #[inline(always)]
    pub unsafe fn last_vessel_unchecked(&self, berth: BerthIndex) -> Option<VesselIndex> {
        debug_assert!(
            berth.get() < self.num_berths,
            "called `ScheduleGraph::last_vessel_unchecked` with out-of-bounds berth: berth = {}, num_berths = {}",
            berth.get(),
            self.num_berths
        );
        let predecessor_of_sentinel =
            *unsafe { self.prev.get_unchecked(self.sentinel(berth).get()) };
        (predecessor_of_sentinel.get() < self.num_vessels).then_some(predecessor_of_sentinel)
    }

    /// Returns the vessel immediately preceding the given vessel in its berth sequence,
    /// or `None` if the given vessel is the first in its berth.
    ///
    /// # Panics
    ///
    /// Panics if `vessel_index` is out of bounds,
    /// meaning `vessel_index.get() >= self.num_vessels`.
    #[inline]
    pub fn vessel_predecessor(&self, vessel_index: VesselIndex) -> Option<VesselIndex> {
        assert!(
            vessel_index.get() < self.num_vessels,
            "called `ScheduleGraph::vessel_predecessor` with out-of-bounds vessel: vessel = {}, num_vessels = {}",
            vessel_index.get(),
            self.num_vessels
        );
        unsafe { self.vessel_predecessor_unchecked(vessel_index) }
    }

    /// Returns the vessel immediately preceding the given vessel in its berth sequence,
    /// or `None` if the given vessel is the first in its berth.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `vessel_index` is a valid vessel index, meaning
    /// `vessel_index.get() < self.num_vessels`. No bounds checking is performed.
    #[inline]
    pub unsafe fn vessel_predecessor_unchecked(
        &self,
        vessel_index: VesselIndex,
    ) -> Option<VesselIndex> {
        debug_assert!(
            vessel_index.get() < self.num_vessels,
            "called `ScheduleGraph::vessel_predecessor_unchecked` with out-of-bounds vessel: vessel = {}, num_vessels = {}",
            vessel_index.get(),
            self.num_vessels
        );
        let predecessor = *unsafe { self.prev.get_unchecked(vessel_index.get()) };
        (predecessor.get() < self.num_vessels).then_some(predecessor)
    }

    /// Returns the vessel immediately following the given vessel in its berth sequence,
    /// or `None` if the given vessel is the last in its berth.
    ///
    /// # Panics
    ///
    /// Panics if `vessel_index` is out of bounds,
    /// meaning `vessel_index.get() >= self.num_vessels`.
    #[inline]
    pub fn vessel_successor(&self, vessel_index: VesselIndex) -> Option<VesselIndex> {
        assert!(
            vessel_index.get() < self.num_vessels,
            "called `ScheduleGraph::vessel_successor` with out-of-bounds vessel: vessel = {}, num_vessels = {}",
            vessel_index.get(),
            self.num_vessels
        );
        unsafe { self.vessel_successor_unchecked(vessel_index) }
    }

    /// Returns the vessel immediately following the given vessel in its berth sequence,
    /// or `None` if the given vessel is the last in its berth.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `vessel_index` is a valid vessel index, meaning
    /// `vessel_index.get() < self.num_vessels`. No bounds checking is performed.
    #[inline]
    pub unsafe fn vessel_successor_unchecked(
        &self,
        vessel_index: VesselIndex,
    ) -> Option<VesselIndex> {
        debug_assert!(
            vessel_index.get() < self.num_vessels,
            "called `ScheduleGraph::vessel_successor_unchecked` with out-of-bounds vessel: vessel = {}, num_vessels = {}",
            vessel_index.get(),
            self.num_vessels
        );
        let successor = *unsafe { self.next.get_unchecked(vessel_index.get()) };
        (successor.get() < self.num_vessels).then_some(successor)
    }

    /// Returns an iterator over the vessels assigned to the given berth, in order.
    /// The iterator is guaranteed to yield exactly `self.vessel_count(berth_index)` elements.
    ///
    /// # Panics
    ///
    /// Panics if `berth_index` is out of bounds,
    /// meaning `berth_index.get() >= self.num_berths`.
    #[inline]
    pub fn vessel_sequence_iter(&self, berth_index: BerthIndex) -> VesselSequenceIter<'_> {
        assert!(
            berth_index.get() < self.num_berths,
            "called `ScheduleGraph::vessel_sequence_iter` with out-of-bounds berth: berth = {}, num_berths = {}",
            berth_index.get(),
            self.num_berths
        );
        unsafe { self.vessel_sequence_iter_unchecked(berth_index) }
    }

    /// Returns an iterator over the vessels assigned to the given berth, in order.
    /// The iterator is guaranteed to yield exactly `self.vessel_count(berth_index)` elements.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `berth_index` is a valid berth index, meaning
    /// `berth_index.get() < self.num_berths`. No bounds checking is performed.
    #[inline]
    pub unsafe fn vessel_sequence_iter_unchecked(
        &self,
        berth_index: BerthIndex,
    ) -> VesselSequenceIter<'_> {
        debug_assert!(
            berth_index.get() < self.num_berths,
            "called `ScheduleGraph::vessel_sequence_iter_unchecked` with out-of-bounds berth: berth = {}, num_berths = {}",
            berth_index.get(),
            self.num_berths
        );

        let sentinel_node = self.sentinel(berth_index);
        let first_node = *unsafe { self.next.get_unchecked(sentinel_node.get()) };
        let remaining_vessels = unsafe { self.vessel_count_unchecked(berth_index) };

        VesselSequenceIter {
            next_pointers: &self.next,
            current_node: first_node,
            remaining_vessels,
        }
    }

    /// Returns an iterator over the vessels assigned to the given berth, in reverse order.
    /// The iterator is guaranteed to yield exactly `self.vessel_count(berth_index)` elements.
    ///
    /// # Panics
    ///
    /// Panics if `berth_index` is out of bounds,
    /// meaning `berth_index.get() >= self.num_berths`.
    #[inline]
    pub fn vessel_sequence_rev_iter(&self, berth_index: BerthIndex) -> VesselSequenceRevIter<'_> {
        assert!(
            berth_index.get() < self.num_berths,
            "called `ScheduleGraph::vessel_sequence_rev_iter` with out-of-bounds berth: berth = {}, num_berths = {}",
            berth_index.get(),
            self.num_berths
        );
        unsafe { self.vessel_sequence_rev_iter_unchecked(berth_index) }
    }

    /// Returns an iterator over the vessels assigned to the given berth, in reverse order.
    /// The iterator is guaranteed to yield exactly `self.vessel_count(berth_index)` elements.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `berth_index` is a valid berth index, meaning
    /// `berth_index.get() < self.num_berths`. No bounds checking is performed.
    #[inline]
    pub unsafe fn vessel_sequence_rev_iter_unchecked(
        &self,
        berth_index: BerthIndex,
    ) -> VesselSequenceRevIter<'_> {
        debug_assert!(
            berth_index.get() < self.num_berths,
            "called `ScheduleGraph::vessel_sequence_rev_iter_unchecked` with out-of-bounds berth: berth = {}, num_berths = {}",
            berth_index.get(),
            self.num_berths
        );

        let sentinel_node = self.sentinel(berth_index);
        let last_node = *unsafe { self.prev.get_unchecked(sentinel_node.get()) };
        let remaining_vessels = unsafe { self.vessel_count_unchecked(berth_index) };

        VesselSequenceRevIter {
            prev_pointers: &self.prev,
            current_node: last_node,
            remaining_vessels,
        }
    }

    /// Returns an iterator over all edges (adjacent vessel pairs) within a specific berth.
    ///
    /// # Panics
    ///
    /// Panics if `berth` is out of bounds.
    #[inline]
    pub fn berth_edges(&self, berth: BerthIndex) -> BerthEdgeIter<'_> {
        assert!(
            berth.get() < self.num_berths,
            "called `ScheduleGraph::berth_edges` with out-of-bounds berth: berth = {}, num_berths = {}",
            berth.get(),
            self.num_berths
        );
        unsafe { self.berth_edges_unchecked(berth) }
    }

    /// Returns an iterator over all edges (adjacent vessel pairs) within a specific berth.
    ///
    /// # Safety
    ///
    /// The caller must ensure `berth.get() < self.num_berths`.
    #[inline]
    pub unsafe fn berth_edges_unchecked(&self, berth: BerthIndex) -> BerthEdgeIter<'_> {
        let sentinel_node = self.sentinel(berth);
        let first_node = *unsafe { self.next.get_unchecked(sentinel_node.get()) };
        let count = unsafe { self.vessel_count_unchecked(berth) };

        // A berth with N vessels has exactly max(0, N-1) consecutive edges.
        let remaining_edges = count.saturating_sub(1);

        BerthEdgeIter {
            next_pointers: &self.next,
            current_node: first_node,
            remaining_edges,
        }
    }

    /// Returns a fast, lazy iterator over all explicit vessel-to-vessel edges.
    #[inline(always)]
    pub fn all_edges(&self) -> AllEdgeIter<'_> {
        AllEdgeIter {
            next_pointers: &self.next,
            vessel_berth: &self.vessel_berth,
            current_vessel: 0, // Start scanning from Vessel 0
        }
    }

    /// Swaps the positions of two vessels in the schedule graph.
    ///
    /// ```text
    /// Before:
    /// Berth A: ... <-> A_Prev <-> V1 <-> A_Next <-> ...
    /// Berth B: ... <-> B_Prev <-> V2 <-> B_Next <-> ...
    ///
    /// After:
    /// Berth A: ... <-> A_Prev <-> V2 <-> A_Next <-> ...
    /// Berth B: ... <-> B_Prev <-> V1 <-> B_Next <-> ...
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if either `first_vessel` or `second_vessel` is out of bounds,
    /// meaning `first_vessel.get() >= self.num_vessels` or `second_vessel.get() >= self.num_vessels`.
    #[inline]
    pub fn swap_vessels(&mut self, first_vessel: VesselIndex, second_vessel: VesselIndex) {
        assert!(
            first_vessel.get() < self.num_vessels && second_vessel.get() < self.num_vessels,
            "called `ScheduleGraph::swap_vessels` with out-of-bounds vessel: v1 = {}, v2 = {}, num_vessels = {}",
            first_vessel.get(),
            second_vessel.get(),
            self.num_vessels
        );

        unsafe { self.swap_vessels_unchecked(first_vessel, second_vessel) }
    }

    /// Swaps the positions of two vessels in the schedule graph.
    ///
    /// ```text
    /// Before:
    /// Berth A: ... <-> A_Prev <-> V1 <-> A_Next <-> ...
    /// Berth B: ... <-> B_Prev <-> V2 <-> B_Next <-> ...
    ///
    /// After:
    /// Berth A: ... <-> A_Prev <-> V2 <-> A_Next <-> ...
    /// Berth B: ... <-> B_Prev <-> V1 <-> B_Next <-> ...
    /// ```
    ///
    /// # Safety
    ///
    /// Both `first_vessel` and `second_vessel` must be valid vessel indices, meaning their `get()` values are both less than `self.num_vessels`.
    /// No bounds checking is performed.
    #[inline]
    pub unsafe fn swap_vessels_unchecked(
        &mut self,
        first_vessel: VesselIndex,
        second_vessel: VesselIndex,
    ) {
        debug_assert!(
            first_vessel.get() < self.num_vessels && second_vessel.get() < self.num_vessels,
            "called `ScheduleGraph::swap_vessels_unchecked` with out-of-bounds vessel: v1 = {}, v2 = {}, num_vessels = {}",
            first_vessel.get(),
            second_vessel.get(),
            self.num_vessels
        );

        if first_vessel == second_vessel {
            return;
        }

        let first_vessel_predecessor = *unsafe { self.prev.get_unchecked(first_vessel.get()) };
        let first_vessel_successor = *unsafe { self.next.get_unchecked(first_vessel.get()) };

        let second_vessel_predecessor = *unsafe { self.prev.get_unchecked(second_vessel.get()) };
        let second_vessel_successor = *unsafe { self.next.get_unchecked(second_vessel.get()) };

        let first_is_before_second = first_vessel_successor == second_vessel;
        let second_is_before_first = second_vessel_successor == first_vessel;

        unsafe { self.extract_node_unchecked(first_vessel) };
        unsafe { self.extract_node_unchecked(second_vessel) };

        if first_is_before_second {
            // first_vessel -> second_vessel
            unsafe { self.insert_node_after_unchecked(second_vessel, first_vessel_predecessor) };
            unsafe { self.insert_node_after_unchecked(first_vessel, second_vessel) };
        } else if second_is_before_first {
            // second_vessel -> first_vessel
            unsafe { self.insert_node_after_unchecked(first_vessel, second_vessel_predecessor) };
            unsafe { self.insert_node_after_unchecked(second_vessel, first_vessel) };
        } else {
            // non-adjacent
            unsafe { self.insert_node_after_unchecked(first_vessel, second_vessel_predecessor) };
            unsafe { self.insert_node_after_unchecked(second_vessel, first_vessel_predecessor) };
        }

        unsafe {
            std::ptr::swap(
                self.vessel_berth.as_mut_ptr().add(first_vessel.get()),
                self.vessel_berth.as_mut_ptr().add(second_vessel.get()),
            );
        }
    }

    /// Swaps the positions of two contiguous segments of vessels.
    ///
    /// ```text
    /// Before:
    /// Berth A: ... <-> A_Prev <-> [ A_First ... A_Last ] <-> A_Next <-> ...
    /// Berth B: ... <-> B_Prev <-> [ B_First ... B_Last ] <-> B_Next <-> ...
    ///
    /// After:
    /// Berth A: ... <-> A_Prev <-> [ B_First ... B_Last ] <-> A_Next <-> ...
    /// Berth B: ... <-> B_Prev <-> [ A_First ... A_Last ] <-> B_Next <-> ...
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if any of the vessel indices are out of bounds,
    /// meaning any of `segment_a_first.get()`, `segment_a_last.get()`, `segment_b_first.get()`, or `segment_b_last.get()` is
    /// greater than or equal to `self.num_vessels`.
    #[inline]
    pub fn swap_segments(
        &mut self,
        segment_a_first: VesselIndex,
        segment_a_last: VesselIndex,
        segment_b_first: VesselIndex,
        segment_b_last: VesselIndex,
    ) {
        assert!(
            segment_a_first.get() < self.num_vessels
                && segment_a_last.get() < self.num_vessels
                && segment_b_first.get() < self.num_vessels
                && segment_b_last.get() < self.num_vessels,
            "called `ScheduleGraph::swap_segments` with out-of-bounds vessel indices: \
            a_first = {}, a_last = {}, b_first = {}, b_last = {}, num_vessels = {}",
            segment_a_first.get(),
            segment_a_last.get(),
            segment_b_first.get(),
            segment_b_last.get(),
            self.num_vessels
        );

        unsafe {
            self.swap_segments_unchecked(
                segment_a_first,
                segment_a_last,
                segment_b_first,
                segment_b_last,
            )
        }
    }

    /// Swaps the positions of two contiguous segments of vessels.
    ///
    /// ```text
    /// Before:
    /// Berth A: ... <-> A_Prev <-> [ A_First ... A_Last ] <-> A_Next <-> ...
    /// Berth B: ... <-> B_Prev <-> [ B_First ... B_Last ] <-> B_Next <-> ...
    ///
    /// After:
    /// Berth A: ... <-> A_Prev <-> [ B_First ... B_Last ] <-> A_Next <-> ...
    /// Berth B: ... <-> B_Prev <-> [ A_First ... A_Last ] <-> B_Next <-> ...
    /// ```
    ///
    /// # Safety
    ///
    /// All of `segment_a_first`, `segment_a_last`, `segment_b_first`, and `segment_b_last` must be valid vessel indices, meaning
    /// their `get()` values are all less than `self.num_vessels`. No bounds checking is performed.
    #[inline]
    pub unsafe fn swap_segments_unchecked(
        &mut self,
        segment_a_first: VesselIndex,
        segment_a_last: VesselIndex,
        segment_b_first: VesselIndex,
        segment_b_last: VesselIndex,
    ) {
        debug_assert!(
            segment_a_first.get() < self.num_vessels
                && segment_a_last.get() < self.num_vessels
                && segment_b_first.get() < self.num_vessels
                && segment_b_last.get() < self.num_vessels,
            "called `ScheduleGraph::swap_segments_unchecked` with out-of-bounds vessel indices: \
            a_first = {}, a_last = {}, b_first = {}, b_last = {}, num_vessels = {}",
            segment_a_first.get(),
            segment_a_last.get(),
            segment_b_first.get(),
            segment_b_last.get(),
            self.num_vessels
        );

        // Identity Case: If the segments are the same, there's nothing to swap.
        // It will also not corrupt the graph structure, so we can just return early without doing any work.
        if segment_a_first == segment_b_first {
            return;
        }

        // Very complex validation logic to catch all possible ways the segments could be malformed or overlapping.
        // This cannot be run in release builds due to the performance cost,
        // but it's invaluable for catching bugs during development and testing.
        #[cfg(debug_assertions)]
        {
            // Validate Segment A and check if it overlaps with B's boundaries
            let mut current = segment_a_first;
            let mut a_is_valid = false;
            for _ in 0..self.num_vessels {
                assert!(
                    current != segment_b_first && current != segment_b_last,
                    "called `ScheduleGraph::swap_segments` with overlapping segments: \
                    a_first = {}, a_last = {}, b_first = {}, b_last = {}",
                    segment_a_first.get(),
                    segment_a_last.get(),
                    segment_b_first.get(),
                    segment_b_last.get()
                );

                if current == segment_a_last {
                    a_is_valid = true;
                    break;
                }

                current = self.next[current.get()];

                assert!(
                    current.get() < self.num_vessels,
                    "called `ScheduleGraph::swap_segments` with invalid Segment A (hits a sentinel): \
                    a_first = {}, a_last = {}, b_first = {}, b_last = {}",
                    segment_a_first.get(),
                    segment_a_last.get(),
                    segment_b_first.get(),
                    segment_b_last.get(),
                );
            }
            assert!(
                a_is_valid,
                "called `ScheduleGraph::swap_segments` with invalid Segment A (a_last is not reachable from a_first): \
                a_first = {}, a_last = {}, b_first = {}, b_last = {}",
                segment_a_first.get(),
                segment_a_last.get(),
                segment_b_first.get(),
                segment_b_last.get(),
            );

            // Validate Segment B and check if it overlaps with A's boundaries
            // This catches the case where A is entirely contained inside B.
            let mut current = segment_b_first;
            let mut b_is_valid = false;
            for _ in 0..self.num_vessels {
                assert!(
                    current != segment_a_first && current != segment_a_last,
                    "called `ScheduleGraph::swap_segments` with overlapping segments: \
                    a_first = {}, a_last = {}, b_first = {}, b_last = {}",
                    segment_a_first.get(),
                    segment_a_last.get(),
                    segment_b_first.get(),
                    segment_b_last.get()
                );

                if current == segment_b_last {
                    b_is_valid = true;
                    break;
                }

                current = self.next[current.get()];

                assert!(
                    current.get() < self.num_vessels,
                    "called `ScheduleGraph::swap_segments` with invalid Segment B (hits a sentinel): \
                    a_first = {}, a_last = {}, b_first = {}, b_last = {}",
                    segment_a_first.get(),
                    segment_a_last.get(),
                    segment_b_first.get(),
                    segment_b_last.get(),
                );
            }
            assert!(
                b_is_valid,
                "called `ScheduleGraph::swap_segments` with invalid Segment B (b_last is not reachable from b_first): \
                a_first = {}, a_last = {}, b_first = {}, b_last = {}",
                segment_a_first.get(),
                segment_a_last.get(),
                segment_b_first.get(),
                segment_b_last.get(),
            );
        }

        let segment_a_predecessor = *unsafe { self.prev.get_unchecked(segment_a_first.get()) };
        let segment_a_successor = *unsafe { self.next.get_unchecked(segment_a_last.get()) };

        let segment_b_predecessor = *unsafe { self.prev.get_unchecked(segment_b_first.get()) };
        let segment_b_successor = *unsafe { self.next.get_unchecked(segment_b_last.get()) };

        if segment_a_successor == segment_b_first {
            // Adjacency Case 1: Segment A is immediately before Segment B
            unsafe {
                *self.next.get_unchecked_mut(segment_a_predecessor.get()) = segment_b_first;
                *self.prev.get_unchecked_mut(segment_b_first.get()) = segment_a_predecessor;

                *self.next.get_unchecked_mut(segment_b_last.get()) = segment_a_first;
                *self.prev.get_unchecked_mut(segment_a_first.get()) = segment_b_last;

                *self.next.get_unchecked_mut(segment_a_last.get()) = segment_b_successor;
                *self.prev.get_unchecked_mut(segment_b_successor.get()) = segment_a_last;
            }
        } else if segment_b_successor == segment_a_first {
            // Adjacency Case 2: Segment B is immediately before Segment A
            unsafe {
                *self.next.get_unchecked_mut(segment_b_predecessor.get()) = segment_a_first;
                *self.prev.get_unchecked_mut(segment_a_first.get()) = segment_b_predecessor;

                *self.next.get_unchecked_mut(segment_a_last.get()) = segment_b_first;
                *self.prev.get_unchecked_mut(segment_b_first.get()) = segment_a_last;

                *self.next.get_unchecked_mut(segment_b_last.get()) = segment_a_successor;
                *self.prev.get_unchecked_mut(segment_a_successor.get()) = segment_b_last;
            }
        } else {
            // Standard Case: Segments are not adjacent
            unsafe {
                *self.next.get_unchecked_mut(segment_a_predecessor.get()) = segment_b_first;
                *self.prev.get_unchecked_mut(segment_b_first.get()) = segment_a_predecessor;

                *self.next.get_unchecked_mut(segment_b_last.get()) = segment_a_successor;
                *self.prev.get_unchecked_mut(segment_a_successor.get()) = segment_b_last;

                *self.next.get_unchecked_mut(segment_b_predecessor.get()) = segment_a_first;
                *self.prev.get_unchecked_mut(segment_a_first.get()) = segment_b_predecessor;

                *self.next.get_unchecked_mut(segment_a_last.get()) = segment_b_successor;
                *self.prev.get_unchecked_mut(segment_b_successor.get()) = segment_a_last;
            }
        }

        // Synchronize vessel_berth and berth_vessel_count.
        // Internal next pointers within each segment are preserved by all three cases above,
        // so walking from segment_X_first to segment_X_last via next pointers is valid.
        let berth_a = *unsafe { self.vessel_berth.get_unchecked(segment_a_first.get()) };
        let berth_b = *unsafe { self.vessel_berth.get_unchecked(segment_b_first.get()) };
        if berth_a != berth_b {
            // Walk segment A: assign to berth_b, count size.
            let mut size_a = 0usize;
            let mut current = segment_a_first;
            loop {
                *unsafe { self.vessel_berth.get_unchecked_mut(current.get()) } = berth_b;
                size_a += 1;
                if current == segment_a_last {
                    break;
                }
                current = *unsafe { self.next.get_unchecked(current.get()) };
            }

            // Walk segment B: assign to berth_a, count size.
            let mut size_b = 0usize;
            current = segment_b_first;
            loop {
                *unsafe { self.vessel_berth.get_unchecked_mut(current.get()) } = berth_a;
                size_b += 1;
                if current == segment_b_last {
                    break;
                }
                current = *unsafe { self.next.get_unchecked(current.get()) };
            }

            // Adjust counts: berth_a lost size_a, gained size_b (and vice versa).
            let count_a = unsafe { self.berth_vessel_count.get_unchecked_mut(berth_a.get()) };
            *count_a = count_a.wrapping_sub(size_a).wrapping_add(size_b);
            let count_b = unsafe { self.berth_vessel_count.get_unchecked_mut(berth_b.get()) };
            *count_b = count_b.wrapping_sub(size_b).wrapping_add(size_a);
        }
    }

    /// Relocates a single vessel to immediately follow the target vessel.
    ///
    /// ```text
    /// Before:
    /// Source: ... <-> Prev <-> Subject <-> Next <-> ...
    /// Target: ... <-> Target <-> Target_Next <-> ...
    ///
    /// After:
    /// Source: ... <-> Prev <-----------> Next <-> ...
    /// Target: ... <-> Target <-> Subject <-> Target_Next <-> ...
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if either `vessel_to_move` or `insertion_anchor` is out of bounds,
    /// meaning `vessel_to_move.get() >= self.num_vessels` or `insertion_anchor.get() >= self.num_vessels`.
    #[inline]
    pub fn relocate_after(&mut self, vessel_to_move: VesselIndex, insertion_anchor: VesselIndex) {
        assert!(
            vessel_to_move.get() < self.num_vessels && insertion_anchor.get() < self.num_vessels,
            "called `ScheduleGraph::relocate_after` with out-of-bounds vessel: subject = {}, target = {}, num_vessels = {}",
            vessel_to_move.get(),
            insertion_anchor.get(),
            self.num_vessels
        );

        unsafe { self.relocate_after_unchecked(vessel_to_move, insertion_anchor) }
    }

    /// Relocates a single vessel to immediately follow the target vessel.
    ///
    /// ```text
    /// Before:
    /// Source: ... <-> Prev <-> Subject <-> Next <-> ...
    /// Target: ... <-> Target <-> Target_Next <-> ...
    ///
    /// After:
    /// Source: ... <-> Prev <-----------> Next <-> ...
    /// Target: ... <-> Target <-> Subject <-> Target_Next <-> ...
    /// ```
    ///
    /// # Safety
    ///
    /// The caller must ensure that both `vessel_to_move` and `insertion_anchor` are valid vessel indices, meaning
    /// `vessel_to_move.get() < self.num_vessels` and `insertion_anchor.get() < self.num_vessels`.
    /// No bounds checking is performed.
    #[inline]
    pub unsafe fn relocate_after_unchecked(
        &mut self,
        vessel_to_move: VesselIndex,
        insertion_anchor: VesselIndex,
    ) {
        debug_assert!(
            vessel_to_move.get() < self.num_vessels && insertion_anchor.get() < self.num_vessels,
            "called `ScheduleGraph::relocate_after_unchecked` with out-of-bounds vessel: subject = {}, target = {}, num_vessels = {}",
            vessel_to_move.get(),
            insertion_anchor.get(),
            self.num_vessels
        );

        if vessel_to_move == insertion_anchor {
            return;
        }

        if *unsafe { self.prev.get_unchecked(vessel_to_move.get()) } == insertion_anchor {
            return;
        }

        let old_berth = *unsafe { self.vessel_berth.get_unchecked(vessel_to_move.get()) };
        let new_berth = *unsafe { self.vessel_berth.get_unchecked(insertion_anchor.get()) };

        unsafe { self.extract_node_unchecked(vessel_to_move) };
        unsafe { self.insert_node_after_unchecked(vessel_to_move, insertion_anchor) };
        unsafe { self.transfer_vessel_berth_unchecked(vessel_to_move, old_berth, new_berth) };
    }

    /// Relocates a contiguous segment of vessels to immediately follow the target vessel.
    ///
    /// ```text
    /// Before:
    /// Source: ... <-> Prev <-> [ First ... Last ] <-> Next <-> ...
    /// Target: ... <-> Target <-> Target_Next <-> ...
    ///
    /// After:
    /// Source: ... <-> Prev <--------------------> Next <-> ...
    /// Target: ... <-> Target <-> [ First ... Last ] <-> Target_Next <-> ...
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if any of the vessel indices are out of bounds,
    /// meaning `segment_first.get() >= self.num_vessels`, `segment_last.get() >= self.num_vessels`,
    /// or `insertion_anchor.get() >= self.num_vessels`.
    #[inline]
    pub fn relocate_segment_after(
        &mut self,
        segment_first: VesselIndex,
        segment_last: VesselIndex,
        insertion_anchor: VesselIndex,
    ) {
        assert!(
            segment_first.get() < self.num_vessels
                && segment_last.get() < self.num_vessels
                && insertion_anchor.get() < self.num_vessels,
            "called `ScheduleGraph::relocate_segment_after` with out-of-bounds vessel: \
            first = {}, last = {}, target = {}, num_vessels = {}",
            segment_first.get(),
            segment_last.get(),
            insertion_anchor.get(),
            self.num_vessels
        );

        unsafe {
            self.relocate_segment_after_unchecked(segment_first, segment_last, insertion_anchor)
        }
    }

    /// Relocates a contiguous segment of vessels to immediately follow the target vessel.
    ///
    /// ```text
    /// Before:
    /// Source: ... <-> Prev <-> [ First ... Last ] <-> Next <-> ...
    /// Target: ... <-> Target <-> Target_Next <-> ...
    ///
    /// After:
    /// Source: ... <-> Prev <--------------------> Next <-> ...
    /// Target: ... <-> Target <-> [ First ... Last ] <-> Target_Next <-> ...
    /// ```
    ///
    /// # Safety
    ///
    /// The caller must ensure that `segment_first`, `segment_last`, and `insertion_anchor` are valid vessel indices, meaning
    /// `segment_first.get() < self.num_vessels`, `segment_last.get() < self.num_vessels`, and `insertion_anchor.get() < self.num_vessels`.
    /// No bounds checking is performed.
    #[inline]
    pub unsafe fn relocate_segment_after_unchecked(
        &mut self,
        segment_first: VesselIndex,
        segment_last: VesselIndex,
        insertion_anchor: VesselIndex,
    ) {
        debug_assert!(
            segment_first.get() < self.num_vessels
                && segment_last.get() < self.num_vessels
                && insertion_anchor.get() < self.num_vessels,
            "called `ScheduleGraph::relocate_segment_after_unchecked` with out-of-bounds vessel: \
            first = {}, last = {}, target = {}, num_vessels = {}",
            segment_first.get(),
            segment_last.get(),
            insertion_anchor.get(),
            self.num_vessels
        );

        if *unsafe { self.prev.get_unchecked(segment_first.get()) } == insertion_anchor {
            return;
        }

        let before_segment = *unsafe { self.prev.get_unchecked(segment_first.get()) };
        let after_segment = *unsafe { self.next.get_unchecked(segment_last.get()) };

        unsafe {
            *self.next.get_unchecked_mut(before_segment.get()) = after_segment;
            *self.prev.get_unchecked_mut(after_segment.get()) = before_segment;
        }

        let successor_of_anchor = *unsafe { self.next.get_unchecked(insertion_anchor.get()) };

        unsafe {
            *self.next.get_unchecked_mut(insertion_anchor.get()) = segment_first;
            *self.prev.get_unchecked_mut(segment_first.get()) = insertion_anchor;

            *self.next.get_unchecked_mut(segment_last.get()) = successor_of_anchor;
            *self.prev.get_unchecked_mut(successor_of_anchor.get()) = segment_last;
        }

        // Synchronize vessel_berth and berth_vessel_count.
        // Internal next pointers within the segment are preserved, so the walk is valid.
        let target_berth = *unsafe { self.vessel_berth.get_unchecked(insertion_anchor.get()) };
        unsafe { self.update_segment_berth_unchecked(segment_first, segment_last, target_berth) };
    }

    /// Relocates a single vessel to immediately precede the target vessel.
    ///
    /// ```text
    /// Before:
    /// Source: ... <-> Prev <-> Subject <-> Next <-> ...
    /// Target: ... <-> Target_Prev <-> Target <-> ...
    ///
    /// After:
    /// Source: ... <-> Prev <-----------> Next <-> ...
    /// Target: ... <-> Target_Prev <-> Subject <-> Target <-> ...
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if either `vessel_to_move` or `reference_vessel` is out of bounds,
    /// meaning `vessel_to_move.get() >= self.num_vessels` or `reference_vessel.get() >= self.num_vessels`.
    #[inline]
    pub fn relocate_before(&mut self, vessel_to_move: VesselIndex, reference_vessel: VesselIndex) {
        assert!(
            vessel_to_move.get() < self.num_vessels && reference_vessel.get() < self.num_vessels,
            "called `ScheduleGraph::relocate_before` with out-of-bounds indices: subject = {}, target = {}, num_vessels = {}",
            vessel_to_move.get(),
            reference_vessel.get(),
            self.num_vessels
        );

        unsafe { self.relocate_before_unchecked(vessel_to_move, reference_vessel) }
    }

    /// Relocates a single vessel to immediately precede the target vessel.
    ///
    /// ```text
    /// Before:
    /// Source: ... <-> Prev <-> Subject <-> Next <-> ...
    /// Target: ... <-> Target_Prev <-> Target <-> ...
    ///
    /// After:
    /// Source: ... <-> Prev <-----------> Next <-> ...
    /// Target: ... <-> Target_Prev <-> Subject <-> Target <-> ...
    /// ```
    ///
    /// # Safety
    ///
    /// The caller must ensure that `vessel_to_move` and `reference_vessel` are valid vessel indices, meaning
    /// `vessel_to_move.get() < self.num_vessels` and `reference_vessel.get() < self.num_vessels`.
    /// No bounds checking is performed.
    #[inline]
    pub unsafe fn relocate_before_unchecked(
        &mut self,
        vessel_to_move: VesselIndex,
        reference_vessel: VesselIndex,
    ) {
        debug_assert!(
            vessel_to_move.get() < self.num_vessels && reference_vessel.get() < self.num_vessels,
            "called `ScheduleGraph::relocate_before_unchecked` with out-of-bounds indices: subject = {}, target = {}, num_vessels = {}",
            vessel_to_move.get(),
            reference_vessel.get(),
            self.num_vessels
        );

        let reference_predecessor = *unsafe { self.prev.get_unchecked(reference_vessel.get()) };
        if reference_predecessor.get() >= self.num_vessels {
            let berth = BerthIndex::new(reference_predecessor.get() - self.num_vessels);
            unsafe { self.relocate_to_head_unchecked(vessel_to_move, berth) };
        } else {
            unsafe { self.relocate_after_unchecked(vessel_to_move, reference_predecessor) };
        }
    }

    /// Relocates a contiguous segment of vessels to immediately precede the target vessel.
    ///
    /// ```text
    /// Before:
    /// Source: ... <-> Prev <-> [ First ... Last ] <-> Next <-> ...
    /// Target: ... <-> Target_Prev <-> Target <-> ...
    ///
    /// After:
    /// Source: ... <-> Prev <--------------------> Next <-> ...
    /// Target: ... <-> Target_Prev <-> [ First ... Last ] <-> Target <-> ...
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if any of the vessel indices are out of bounds,
    /// meaning `segment_first.get() >= self.num_vessels`, `segment_last.get() >= self.num_vessels`, or `reference_vessel.get() >= self.num_vessels`.
    #[inline]
    pub fn relocate_segment_before(
        &mut self,
        segment_first: VesselIndex,
        segment_last: VesselIndex,
        reference_vessel: VesselIndex,
    ) {
        assert!(
            segment_first.get() < self.num_vessels
                && segment_last.get() < self.num_vessels
                && reference_vessel.get() < self.num_vessels,
            "called `ScheduleGraph::relocate_segment_before` with out-of-bounds indices: \
            first = {}, last = {}, target = {}, num_vessels = {}",
            segment_first.get(),
            segment_last.get(),
            reference_vessel.get(),
            self.num_vessels
        );

        unsafe {
            self.relocate_segment_before_unchecked(segment_first, segment_last, reference_vessel)
        }
    }

    /// Relocates a contiguous segment of vessels to immediately precede the target vessel.
    ///
    /// ```text
    /// Before:
    /// Source: ... <-> Prev <-> [ First ... Last ] <-> Next <-> ...
    /// Target: ... <-> Target_Prev <-> Target <-> ...
    ///
    /// After:
    /// Source: ... <-> Prev <--------------------> Next <-> ...
    /// Target: ... <-> Target_Prev <-> [ First ... Last ] <-> Target <-> ...
    /// ```
    ///
    /// # Safety
    ///
    /// The caller must ensure that `segment_first`, `segment_last`, and `reference_vessel` are valid vessel indices, meaning
    /// `segment_first.get() < self.num_vessels`, `segment_last.get() < self.num_vessels`, and `reference_vessel.get() < self.num_vessels`.
    /// No bounds checking
    #[inline]
    pub unsafe fn relocate_segment_before_unchecked(
        &mut self,
        segment_first: VesselIndex,
        segment_last: VesselIndex,
        reference_vessel: VesselIndex,
    ) {
        debug_assert!(
            segment_first.get() < self.num_vessels
                && segment_last.get() < self.num_vessels
                && reference_vessel.get() < self.num_vessels,
            "called `ScheduleGraph::relocate_segment_before_unchecked` with out-of-bounds indices: \
            first = {}, last = {}, target = {}, num_vessels = {}",
            segment_first.get(),
            segment_last.get(),
            reference_vessel.get(),
            self.num_vessels
        );

        let reference_predecessor = *unsafe { self.prev.get_unchecked(reference_vessel.get()) };

        if reference_predecessor.get() >= self.num_vessels {
            let berth = BerthIndex::new(reference_predecessor.get() - self.num_vessels);
            unsafe { self.relocate_segment_to_head_unchecked(segment_first, segment_last, berth) };
        } else {
            unsafe {
                self.relocate_segment_after_unchecked(
                    segment_first,
                    segment_last,
                    reference_predecessor,
                )
            };
        }
    }

    /// Relocates a single vessel to the head (beginning) of the target berth's sequence.
    ///
    /// ```text
    /// Before:
    /// Source: ... <-> Prev <-> Subject <-> Next <-> ...
    /// Target: [ Sentinel ] <-> Current_Head <-> ...
    ///
    /// After:
    /// Source: ... <-> Prev <-----------> Next <-> ...
    /// Target: [ Sentinel ] <-> Subject <-> Current_Head <-> ...
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `vessel_to_move` or `target_berth` is out of bounds,
    /// meaning `vessel_to_move.get() >= self.num_vessels` or `target_berth.get() >= self.num_berths`.
    #[inline]
    pub fn relocate_to_head(&mut self, vessel_to_move: VesselIndex, target_berth: BerthIndex) {
        assert!(
            vessel_to_move.get() < self.num_vessels,
            "called `ScheduleGraph::relocate_to_head` with out-of-bounds vessel: subject = {}, num_vessels = {}",
            vessel_to_move.get(),
            self.num_vessels
        );
        assert!(
            target_berth.get() < self.num_berths,
            "called `ScheduleGraph::relocate_to_head` with out-of-bounds berth: target_berth = {}, num_berths = {}",
            target_berth.get(),
            self.num_berths
        );

        unsafe { self.relocate_to_head_unchecked(vessel_to_move, target_berth) }
    }

    /// Relocates a single vessel to the head (beginning) of the target berth's sequence.
    ///
    /// ```text
    /// Before:
    /// Source: ... <-> Prev <-> Subject <-> Next <-> ...
    /// Target: [ Sentinel ] <-> Current_Head <-> ...
    ///
    /// After:
    /// Source: ... <-> Prev <-----------> Next <-> ...
    /// Target: [ Sentinel ] <-> Subject <-> Current_Head <-> ...
    /// ```
    ///
    /// # Safety
    ///
    /// The caller must ensure that `vessel_to_move` is a valid vessel index, meaning `vessel_to_move.get() < self.num_vessels`,
    /// and that `target_berth` is a valid berth index, meaning `target_berth.get() < self.num_berths`.
    /// No bounds checking is performed.
    #[inline]
    pub unsafe fn relocate_to_head_unchecked(
        &mut self,
        vessel_to_move: VesselIndex,
        target_berth: BerthIndex,
    ) {
        debug_assert!(
            vessel_to_move.get() < self.num_vessels,
            "called `ScheduleGraph::relocate_to_head_unchecked` with out-of-bounds vessel: subject = {}, num_vessels = {}",
            vessel_to_move.get(),
            self.num_vessels
        );
        debug_assert!(
            target_berth.get() < self.num_berths,
            "called `ScheduleGraph::relocate_to_head_unchecked` with out-of-bounds berth: target_berth = {}, num_berths = {}",
            target_berth.get(),
            self.num_berths
        );

        let sentinel_node = self.sentinel(target_berth);

        if *unsafe { self.prev.get_unchecked(vessel_to_move.get()) } == sentinel_node {
            return;
        }

        let old_berth = *unsafe { self.vessel_berth.get_unchecked(vessel_to_move.get()) };

        unsafe { self.extract_node_unchecked(vessel_to_move) };
        unsafe { self.insert_node_after_unchecked(vessel_to_move, sentinel_node) };
        unsafe { self.transfer_vessel_berth_unchecked(vessel_to_move, old_berth, target_berth) };
    }

    /// Relocates a single vessel to the tail (end) of the target berth's sequence.
    ///
    /// ```text
    /// Before:
    /// Source: ... <-> Prev <-> Subject <-> Next <-> ...
    /// Target: ... <-> Current_Tail <-> [ Sentinel ]
    ///
    /// After:
    /// Source: ... <-> Prev <-----------> Next <-> ...
    /// Target: ... <-> Current_Tail <-> Subject <-> [ Sentinel ]
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `vessel_to_move` or `target_berth` is out of bounds,
    /// meaning `vessel_to_move.get() >= self.num_vessels` or `target_berth.get() >= self.num_berths`.
    #[inline]
    pub fn relocate_to_tail(&mut self, vessel_to_move: VesselIndex, target_berth: BerthIndex) {
        assert!(
            vessel_to_move.get() < self.num_vessels,
            "called `ScheduleGraph::relocate_to_tail` with out-of-bounds vessel: subject = {}, num_vessels = {}",
            vessel_to_move.get(),
            self.num_vessels
        );
        assert!(
            target_berth.get() < self.num_berths,
            "called `ScheduleGraph::relocate_to_tail` with out-of-bounds berth: target_berth = {}, num_berths = {}",
            target_berth.get(),
            self.num_berths
        );

        unsafe { self.relocate_to_tail_unchecked(vessel_to_move, target_berth) }
    }

    /// Relocates a single vessel to the tail (end) of the target berth's sequence.
    ///
    /// ```text
    /// Before:
    /// Source: ... <-> Prev <-> Subject <-> Next <-> ...
    /// Target: ... <-> Current_Tail <-> [ Sentinel ]
    ///
    /// After:
    /// Source: ... <-> Prev <-----------> Next <-> ...
    /// Target: ... <-> Current_Tail <-> Subject <-> [ Sentinel ]
    /// ```
    ///
    /// # Safety
    ///
    /// The caller must ensure that `vessel_to_move` is a valid vessel index, meaning `vessel_to_move.get() < self.num_vessels`,
    /// and that `target_berth` is a valid berth index, meaning `target_berth.get() < self.num_berths`.
    /// No bounds checking is performed.
    #[inline]
    pub unsafe fn relocate_to_tail_unchecked(
        &mut self,
        vessel_to_move: VesselIndex,
        target_berth: BerthIndex,
    ) {
        debug_assert!(
            vessel_to_move.get() < self.num_vessels,
            "called `ScheduleGraph::relocate_to_tail_unchecked` with out-of-bounds vessel: subject = {}, num_vessels = {}",
            vessel_to_move.get(),
            self.num_vessels
        );
        debug_assert!(
            target_berth.get() < self.num_berths,
            "called `ScheduleGraph::relocate_to_tail_unchecked` with out-of-bounds berth: target_berth = {}, num_berths = {}",
            target_berth.get(),
            self.num_berths
        );

        let sentinel_node = self.sentinel(target_berth);
        let current_tail = *unsafe { self.prev.get_unchecked(sentinel_node.get()) };
        if *unsafe { self.next.get_unchecked(vessel_to_move.get()) } == sentinel_node {
            return;
        }

        let old_berth = *unsafe { self.vessel_berth.get_unchecked(vessel_to_move.get()) };
        unsafe { self.extract_node_unchecked(vessel_to_move) };
        unsafe { self.insert_node_after_unchecked(vessel_to_move, current_tail) };
        unsafe { self.transfer_vessel_berth_unchecked(vessel_to_move, old_berth, target_berth) };
    }

    /// Relocates a contiguous segment of vessels to the head (beginning) of the target berth's sequence.
    ///
    /// ```text
    /// Before:
    /// Source: ... <-> Prev <-> [ First ... Last ] <-> Next <-> ...
    /// Target: [ Sentinel ] <-> Current_Head <-> ...
    ///
    /// After:
    /// Source: ... <-> Prev <--------------------> Next <-> ...
    /// Target: [ Sentinel ] <-> [ First ... Last ] <-> Current_Head <-> ...
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `segment_first` or `segment_last` is out of bounds, or if `target_berth` is out of bounds,
    /// meaning `segment_first.get() >= self.num_vessels`, `segment_last.get() >= self.num_vessels`, or
    /// `target_berth.get() >= self.num_berths`.
    #[inline]
    pub fn relocate_segment_to_head(
        &mut self,
        segment_first: VesselIndex,
        segment_last: VesselIndex,
        target_berth: BerthIndex,
    ) {
        assert!(
            segment_first.get() < self.num_vessels && segment_last.get() < self.num_vessels,
            "called `ScheduleGraph::relocate_segment_to_head` with out-of-bounds vessel: first = {}, last = {}, num_vessels = {}",
            segment_first.get(),
            segment_last.get(),
            self.num_vessels
        );
        assert!(
            target_berth.get() < self.num_berths,
            "called `ScheduleGraph::relocate_segment_to_head` with out-of-bounds berth: target_berth = {}, num_berths = {}",
            target_berth.get(),
            self.num_berths
        );

        unsafe {
            self.relocate_segment_to_head_unchecked(segment_first, segment_last, target_berth)
        }
    }

    /// Relocates a contiguous segment of vessels to the head (beginning) of the target berth's sequence.
    ///
    /// ```text
    /// Before:
    /// Source: ... <-> Prev <-> [ First ... Last ] <-> Next <-> ...
    /// Target: [ Sentinel ] <-> Current_Head <-> ...
    ///
    /// After:
    /// Source: ... <-> Prev <--------------------> Next <-> ...
    /// Target: [ Sentinel ] <-> [ First ... Last ] <-> Current_Head <-> ...
    /// ```
    ///
    /// # Safety
    ///
    /// The caller must ensure that `segment_first` and `segment_last` are valid vessel indices, meaning
    /// `segment_first.get() < self.num_vessels` and `segment_last.get() < self.num_vessels`,
    /// and that `target_berth` is a valid berth index, meaning `target_berth.get() < self.num_berths`.
    /// No bounds checking is performed.
    #[inline]
    pub unsafe fn relocate_segment_to_head_unchecked(
        &mut self,
        segment_first: VesselIndex,
        segment_last: VesselIndex,
        target_berth: BerthIndex,
    ) {
        debug_assert!(
            segment_first.get() < self.num_vessels && segment_last.get() < self.num_vessels,
            "called `ScheduleGraph::relocate_segment_to_head_unchecked` with out-of-bounds vessel: first = {}, last = {}, num_vessels = {}",
            segment_first.get(),
            segment_last.get(),
            self.num_vessels
        );
        debug_assert!(
            target_berth.get() < self.num_berths,
            "called `ScheduleGraph::relocate_segment_to_head_unchecked` with out-of-bounds berth: target_berth = {}, num_berths = {}",
            target_berth.get(),
            self.num_berths
        );

        let sentinel_node = self.sentinel(target_berth);
        if *unsafe { self.prev.get_unchecked(segment_first.get()) } == sentinel_node {
            return;
        }

        let before_segment = *unsafe { self.prev.get_unchecked(segment_first.get()) };
        let after_segment = *unsafe { self.next.get_unchecked(segment_last.get()) };

        unsafe {
            *self.next.get_unchecked_mut(before_segment.get()) = after_segment;
            *self.prev.get_unchecked_mut(after_segment.get()) = before_segment;
        }

        let successor_of_sentinel = *unsafe { self.next.get_unchecked(sentinel_node.get()) };

        unsafe {
            *self.next.get_unchecked_mut(sentinel_node.get()) = segment_first;
            *self.prev.get_unchecked_mut(segment_first.get()) = sentinel_node;

            *self.next.get_unchecked_mut(segment_last.get()) = successor_of_sentinel;
            *self.prev.get_unchecked_mut(successor_of_sentinel.get()) = segment_last;
        }

        // Synchronize vessel_berth and berth_vessel_count.
        // Internal next pointers within the segment are preserved, so the walk is valid.
        unsafe { self.update_segment_berth_unchecked(segment_first, segment_last, target_berth) };
    }

    /// Relocates a contiguous segment of vessels to the tail (end) of the target berth's sequence.
    ///
    /// ```text
    /// Before:
    /// Source: ... <-> Prev <-> [ First ... Last ] <-> Next <-> ...
    /// Target: ... <-> Current_Tail <-> [ Sentinel ]
    ///
    /// After:
    /// Source: ... <-> Prev <--------------------> Next <-> ...
    /// Target: ... <-> Current_Tail <-> [ First ... Last ] <-> [ Sentinel ]
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `segment_first` or `segment_last` is out of bounds, or if `target_berth` is out of bounds,
    /// meaning `segment_first.get() >= self.num_vessels`, `segment_last.get() >= self.num_vessels`, or
    /// `target_berth.get() >= self.num_berths`.
    #[inline]
    pub fn relocate_segment_to_tail(
        &mut self,
        segment_first: VesselIndex,
        segment_last: VesselIndex,
        target_berth: BerthIndex,
    ) {
        assert!(
            segment_first.get() < self.num_vessels && segment_last.get() < self.num_vessels,
            "called `ScheduleGraph::relocate_segment_to_tail` with out-of-bounds vessel: first = {}, last = {}, num_vessels = {}",
            segment_first.get(),
            segment_last.get(),
            self.num_vessels
        );
        assert!(
            target_berth.get() < self.num_berths,
            "called `ScheduleGraph::relocate_segment_to_tail` with out-of-bounds berth: target_berth = {}, num_berths = {}",
            target_berth.get(),
            self.num_berths
        );

        unsafe {
            self.relocate_segment_to_tail_unchecked(segment_first, segment_last, target_berth)
        }
    }

    /// Relocates a contiguous segment of vessels to the tail (end) of the target berth's sequence.
    ///
    /// ```text
    /// Before:
    /// Source: ... <-> Prev <-> [ First ... Last ] <-> Next <-> ...
    /// Target: ... <-> Current_Tail <-> [ Sentinel ]
    ///
    /// After:
    /// Source: ... <-> Prev <--------------------> Next <-> ...
    /// Target: ... <-> Current_Tail <-> [ First ... Last ] <-> [ Sentinel ]
    /// ```
    ///
    /// # Safety
    ///
    /// The caller must ensure that `segment_first` and `segment_last` are valid vessel indices, meaning
    /// `segment_first.get() < self.num_vessels` and `segment_last.get() < self.num_vessels`,
    /// and that `target_berth` is a valid berth index, meaning `target_berth.get() < self.num_berths`.
    /// No bounds checking is performed.
    #[inline]
    pub unsafe fn relocate_segment_to_tail_unchecked(
        &mut self,
        segment_first: VesselIndex,
        segment_last: VesselIndex,
        target_berth: BerthIndex,
    ) {
        debug_assert!(
            segment_first.get() < self.num_vessels && segment_last.get() < self.num_vessels,
            "called `ScheduleGraph::relocate_segment_to_tail_unchecked` with out-of-bounds vessel: first = {}, last = {}, num_vessels = {}",
            segment_first.get(),
            segment_last.get(),
            self.num_vessels
        );
        debug_assert!(
            target_berth.get() < self.num_berths,
            "called `ScheduleGraph::relocate_segment_to_tail_unchecked` with out-of-bounds berth: target_berth = {}, num_berths = {}",
            target_berth.get(),
            self.num_berths
        );

        let sentinel_node = self.sentinel(target_berth);
        let current_tail = *unsafe { self.prev.get_unchecked(sentinel_node.get()) };

        if *unsafe { self.next.get_unchecked(segment_last.get()) } == sentinel_node {
            return;
        }

        let before_segment = *unsafe { self.prev.get_unchecked(segment_first.get()) };
        let after_segment = *unsafe { self.next.get_unchecked(segment_last.get()) };

        unsafe {
            *self.next.get_unchecked_mut(before_segment.get()) = after_segment;
            *self.prev.get_unchecked_mut(after_segment.get()) = before_segment;
        }

        let successor_of_tail = *unsafe { self.next.get_unchecked(current_tail.get()) };

        unsafe {
            *self.next.get_unchecked_mut(current_tail.get()) = segment_first;
            *self.prev.get_unchecked_mut(segment_first.get()) = current_tail;

            *self.next.get_unchecked_mut(segment_last.get()) = successor_of_tail;
            *self.prev.get_unchecked_mut(successor_of_tail.get()) = segment_last;
        }

        // Synchronize vessel_berth and berth_vessel_count.
        // Internal next pointers within the segment are preserved, so the walk is valid.
        unsafe { self.update_segment_berth_unchecked(segment_first, segment_last, target_berth) };
    }

    /// Reverses the order of a contiguous segment of vessels in the schedule graph,
    /// without changing their berth assignments.
    ///
    /// The segment is defined by the inclusive `segment_first` and `segment_last` vessel indices.
    ///
    /// ```text
    /// Before:
    /// ... <-> Prev <-> [ A <-> B <-> C <-> D ] <-> Next <-> ...
    ///                    ^                 ^
    ///                  first              last
    ///
    /// After:
    /// ... <-> Prev <-> [ D <-> C <-> B <-> A ] <-> Next <-> ...
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if either `segment_first` or `segment_last` is out of bounds, meaning
    /// `segment_first.get() >= self.num_vessels` or `segment_last.get() >= self.num_vessels`.
    ///
    /// **In debug builds only:** Panics if `segment_first` and `segment_last` do not form a valid,
    /// continuous segment within the same berth.
    #[inline]
    pub fn reverse_segment(&mut self, segment_first: VesselIndex, segment_last: VesselIndex) {
        assert!(
            segment_first.get() < self.num_vessels && segment_last.get() < self.num_vessels,
            "called `ScheduleGraph::reverse_segment` with out-of-bounds vessel: first = {}, last = {}, num_vessels = {}",
            segment_first.get(),
            segment_last.get(),
            self.num_vessels
        );

        unsafe { self.reverse_segment_unchecked(segment_first, segment_last) }
    }

    /// Reverses the order of a contiguous segment of vessels in the schedule graph,
    /// without changing their berth assignments.
    ///
    /// The segment is defined by the inclusive `segment_first` and `segment_last` vessel indices.
    ///
    /// ```text
    /// Before:
    /// ... <-> Prev <-> [ A <-> B <-> C <-> D ] <-> Next <-> ...
    ///                    ^                 ^
    ///                  first              last
    ///
    /// After:
    /// ... <-> Prev <-> [ D <-> C <-> B <-> A ] <-> Next <-> ...
    /// ```
    /// # Safety
    ///
    /// - `segment_first` and `segment_last` must be `< self.num_vessels`.
    /// - `segment_first` and `segment_last` must form a valid, continuous segment within the same berth.
    #[inline]
    pub unsafe fn reverse_segment_unchecked(
        &mut self,
        segment_first: VesselIndex,
        segment_last: VesselIndex,
    ) {
        debug_assert!(
            segment_first.get() < self.num_vessels && segment_last.get() < self.num_vessels,
            "called `ScheduleGraph::reverse_segment_unchecked` with out-of-bounds vessel: first = {}, last = {}, num_vessels = {}",
            segment_first.get(),
            segment_last.get(),
            self.num_vessels
        );

        // $O(N)$ debug-only structural validation
        #[cfg(debug_assertions)]
        {
            let mut current_node = segment_first;
            let mut is_valid = false;

            for _ in 0..self.num_vessels {
                if current_node == segment_last {
                    is_valid = true;
                    break;
                }

                current_node = self.next[current_node.get()];
                if current_node.get() >= self.num_vessels {
                    break;
                }
            }
            debug_assert!(
                is_valid,
                "called `ScheduleGraph::reverse_segment_unchecked` with non-contiguous or cross-berth segment"
            );
        }

        if segment_first == segment_last {
            return;
        }

        let predecessor_of_segment = *unsafe { self.prev.get_unchecked(segment_first.get()) };
        let successor_of_segment = *unsafe { self.next.get_unchecked(segment_last.get()) };

        // Core reversal: swap prev and next for every node in the segment.
        // This is branchless per iteration — just two loads, two stores via ptr::swap,
        // plus one termination check.
        let prev_ptr = self.prev.as_mut_ptr();
        let next_ptr = self.next.as_mut_ptr();
        let mut current_node = segment_first;
        loop {
            // Read original next BEFORE the swap overwrites it.
            let original_next = *unsafe { self.next.get_unchecked(current_node.get()) };
            unsafe {
                std::ptr::swap(
                    prev_ptr.add(current_node.get()),
                    next_ptr.add(current_node.get()),
                );
            }
            if current_node == segment_last {
                break;
            }
            current_node = original_next;
        }

        // After swapping every node's prev/next:
        //   - segment_first.next == predecessor_of_segment (was prev), needs to be successor_of_segment
        //   - segment_last.prev  == successor_of_segment  (was next), needs to be predecessor_of_segment
        *unsafe { self.next.get_unchecked_mut(segment_first.get()) } = successor_of_segment;
        *unsafe { self.prev.get_unchecked_mut(segment_last.get()) } = predecessor_of_segment;

        // Stitch the external neighbors to the new head (segment_last) and tail (segment_first).
        *unsafe { self.next.get_unchecked_mut(predecessor_of_segment.get()) } = segment_last;
        *unsafe { self.prev.get_unchecked_mut(successor_of_segment.get()) } = segment_first;

        // No vessel_berth or berth_vessel_count updates needed:
        // reversal only reorders vessels within the same berth.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[inline]
    fn vessel(index: usize) -> VesselIndex {
        VesselIndex::new(index)
    }

    #[inline]
    fn berth(index: usize) -> BerthIndex {
        BerthIndex::new(index)
    }

    /// Comprehensively verifies all graph invariants for a single berth.
    fn check_berth(graph: &ScheduleGraph, berth_index: usize, expected: &[usize]) {
        let berth_idx = berth(berth_index);

        let forward_sequence: Vec<usize> = graph
            .vessel_sequence_iter(berth_idx)
            .map(|vessel| vessel.get())
            .collect();
        assert_eq!(
            forward_sequence, expected,
            "Forward sequence mismatch for Berth {}",
            berth_index
        );

        let reverse_sequence: Vec<usize> = graph
            .vessel_sequence_rev_iter(berth_idx)
            .map(|vessel| vessel.get())
            .collect();
        let mut expected_reversed = expected.to_vec();
        expected_reversed.reverse();
        assert_eq!(
            reverse_sequence, expected_reversed,
            "Reverse sequence mismatch for Berth {}",
            berth_index
        );

        if expected.is_empty() {
            assert_eq!(graph.first_vessel(berth_idx), None);
            assert_eq!(graph.last_vessel(berth_idx), None);
            assert!(graph.is_empty(berth_idx));
        } else {
            assert_eq!(graph.first_vessel(berth_idx).unwrap().get(), expected[0]);
            assert_eq!(
                graph.last_vessel(berth_idx).unwrap().get(),
                *expected.last().unwrap()
            );
            assert!(!graph.is_empty(berth_idx));
        }

        // Verify vessel_berth mapping for every vessel in this berth.
        for &v in expected {
            assert_eq!(
                graph.vessel_berth(vessel(v)).get(),
                berth_index,
                "vessel_berth mismatch for vessel V{} — expected Berth {}, got Berth {}",
                v,
                berth_index,
                graph.vessel_berth(vessel(v)).get()
            );
        }

        // Verify berth_vessel_count is consistent with the sequence length.
        assert_eq!(
            graph.vessel_count(berth_idx),
            expected.len(),
            "vessel_count mismatch for Berth {} — expected {}, got {}",
            berth_index,
            expected.len(),
            graph.vessel_count(berth_idx)
        );
    }

    /// Creates a standard fixture:
    /// Berth 0: V0 -> V1 -> V2
    /// Berth 1: V3 -> V4
    /// Berth 2: V5
    /// Berth 3: (empty)
    fn standard_fixture() -> ScheduleGraph {
        let berths = [berth(0), berth(0), berth(0), berth(1), berth(1), berth(2)];
        let starts = [10, 30, 20, 10, 20, 15];

        ScheduleGraph::from_slices(&berths, &starts, 4)
    }

    #[test]
    fn test_initialization_and_sorting() {
        let graph = standard_fixture();

        assert_eq!(graph.num_vessels(), 6);
        assert_eq!(graph.num_berths(), 4);

        // Due to the start times [10, 30, 20] in Berth 0, V1 and V2 should be swapped
        check_berth(&graph, 0, &[0, 2, 1]);
        check_berth(&graph, 1, &[3, 4]);
        check_berth(&graph, 2, &[5]);
        check_berth(&graph, 3, &[]);
    }

    #[test]
    fn test_empty_graph() {
        let graph = ScheduleGraph::from_slices::<i32>(&[], &[], 3);
        assert_eq!(graph.num_vessels(), 0);
        assert_eq!(graph.num_berths(), 3);

        check_berth(&graph, 0, &[]);
        check_berth(&graph, 1, &[]);
        check_berth(&graph, 2, &[]);
    }

    #[test]
    fn test_relocate_after_intra_berth() {
        let mut graph = standard_fixture();
        // Move V0 after V2 (Berth 0: 0 -> 2 -> 1 becomes 2 -> 0 -> 1)
        graph.relocate_after(vessel(0), vessel(2));
        check_berth(&graph, 0, &[2, 0, 1]);
    }

    #[test]
    fn test_relocate_after_inter_berth() {
        let mut graph = standard_fixture();
        // Move V0 after V3 (from Berth 0 to Berth 1)
        graph.relocate_after(vessel(0), vessel(3));
        check_berth(&graph, 0, &[2, 1]);
        check_berth(&graph, 1, &[3, 0, 4]);
    }

    #[test]
    fn test_relocate_before_intra_berth() {
        let mut graph = standard_fixture();
        // Move V1 before V0 (Berth 0: 0 -> 2 -> 1 becomes 1 -> 0 -> 2)
        graph.relocate_before(vessel(1), vessel(0));
        check_berth(&graph, 0, &[1, 0, 2]);
    }

    #[test]
    fn test_relocate_to_head_and_tail() {
        let mut graph = standard_fixture();

        // Move V4 to head of Berth 0
        graph.relocate_to_head(vessel(4), berth(0));
        check_berth(&graph, 0, &[4, 0, 2, 1]);
        check_berth(&graph, 1, &[3]);

        // Move V0 to tail of empty Berth 3
        graph.relocate_to_tail(vessel(0), berth(3));
        check_berth(&graph, 0, &[4, 2, 1]);
        check_berth(&graph, 3, &[0]);
    }

    #[test]
    fn test_swap_vessels_adjacent() {
        let mut graph = standard_fixture();
        // Swap V0 and V2 (adjacent in Berth 0)
        graph.swap_vessels(vessel(0), vessel(2));
        check_berth(&graph, 0, &[2, 0, 1]);
    }

    #[test]
    fn test_swap_vessels_non_adjacent_same_berth() {
        let mut graph = standard_fixture();
        // Swap V0 and V1 (separated by V2 in Berth 0)
        graph.swap_vessels(vessel(0), vessel(1));
        check_berth(&graph, 0, &[1, 2, 0]);
    }

    #[test]
    fn test_swap_vessels_different_berths() {
        let mut graph = standard_fixture();
        // Swap V2 (Berth 0) with V4 (Berth 1)
        graph.swap_vessels(vessel(2), vessel(4));
        check_berth(&graph, 0, &[0, 4, 1]);
        check_berth(&graph, 1, &[3, 2]);
    }

    #[test]
    fn test_relocate_segment_after_intra_berth() {
        let mut graph = standard_fixture();
        // Move [0, 2] after 1 (Berth 0: 0 -> 2 -> 1 becomes 1 -> 0 -> 2)
        graph.relocate_segment_after(vessel(0), vessel(2), vessel(1));
        check_berth(&graph, 0, &[1, 0, 2]);
    }

    #[test]
    fn test_relocate_segment_after_inter_berth() {
        let mut graph = standard_fixture();
        // Move [0, 2] after 4 (from Berth 0 to Berth 1)
        graph.relocate_segment_after(vessel(0), vessel(2), vessel(4));
        check_berth(&graph, 0, &[1]);
        check_berth(&graph, 1, &[3, 4, 0, 2]);
    }

    #[test]
    fn test_relocate_segment_to_head_empty_berth() {
        let mut graph = standard_fixture();
        // Move [3, 4] to empty Berth 3
        graph.relocate_segment_to_head(vessel(3), vessel(4), berth(3));
        check_berth(&graph, 1, &[]);
        check_berth(&graph, 3, &[3, 4]);
    }

    #[test]
    fn test_relocate_segment_to_tail() {
        let mut graph = standard_fixture();
        // Move [0, 2] to tail of Berth 1
        graph.relocate_segment_to_tail(vessel(0), vessel(2), berth(1));
        check_berth(&graph, 0, &[1]);
        check_berth(&graph, 1, &[3, 4, 0, 2]);
    }

    #[test]
    fn test_swap_segments_different_berths_different_lengths() {
        let mut graph = standard_fixture();
        // Swap [0, 2] (len 2) with [3, 4] (len 2)
        graph.swap_segments(vessel(0), vessel(2), vessel(3), vessel(4));
        check_berth(&graph, 0, &[3, 4, 1]);
        check_berth(&graph, 1, &[0, 2]);
    }

    #[test]
    fn test_swap_segments_adjacent_same_berth() {
        let mut graph = standard_fixture();
        // Berth 0 is [0, 2, 1]. Swap [0] with [2, 1]
        graph.swap_segments(vessel(0), vessel(0), vessel(2), vessel(1));
        check_berth(&graph, 0, &[2, 1, 0]);

        // Swap back using reverse order (B before A)
        graph.swap_segments(vessel(2), vessel(1), vessel(0), vessel(0));
        check_berth(&graph, 0, &[0, 2, 1]);
    }

    #[test]
    fn test_reverse_segment_full_berth() {
        let mut graph = standard_fixture();
        // Reverse Berth 0: [0, 2, 1] -> [1, 2, 0]
        graph.reverse_segment(vessel(0), vessel(1));
        check_berth(&graph, 0, &[1, 2, 0]);
    }

    #[test]
    fn test_reverse_segment_partial_berth() {
        let mut graph = standard_fixture();
        // Insert V5 into Berth 0 to make it [0, 2, 1, 5]
        graph.relocate_to_tail(vessel(5), berth(0));
        check_berth(&graph, 0, &[0, 2, 1, 5]);

        // Reverse [2, 1] -> [1, 2]
        graph.reverse_segment(vessel(2), vessel(1));
        check_berth(&graph, 0, &[0, 1, 2, 5]);
    }

    #[test]
    fn test_reverse_single_element_no_op() {
        let mut graph = standard_fixture();
        graph.reverse_segment(vessel(2), vessel(2));
        check_berth(&graph, 0, &[0, 2, 1]);
    }

    #[test]
    fn test_overwrite_from_graph_exact_copy() {
        let original = standard_fixture();
        let mut clone = ScheduleGraph::from_slices::<i32>(&[], &[], 4);
        clone.overwrite_from_graph(&original);

        assert_eq!(original, clone);

        // Mutating the clone shouldn't affect the original
        clone.swap_vessels(vessel(0), vessel(2));
        assert_ne!(original, clone);
        check_berth(&original, 0, &[0, 2, 1]);
        check_berth(&clone, 0, &[2, 0, 1]);
    }

    #[test]
    fn test_relocate_after_noop() {
        let mut graph = standard_fixture();

        // Moving a vessel after itself is a no-op (identity check).
        graph.relocate_after(vessel(0), vessel(0));
        check_berth(&graph, 0, &[0, 2, 1]);

        // Moving V2 after V0 (its current predecessor) is a no-op (already in position).
        graph.relocate_after(vessel(2), vessel(0));
        check_berth(&graph, 0, &[0, 2, 1]);
    }

    #[test]
    fn test_relocate_before_noop() {
        let mut graph = standard_fixture();

        // Moving V0 *before* V2 (its current successor) should be a no-op.
        graph.relocate_before(vessel(0), vessel(2));
        check_berth(&graph, 0, &[0, 2, 1]);
    }

    #[test]
    fn test_relocate_segment_after_noop() {
        let mut graph = standard_fixture();

        // Segment [0, 2] is already immediately after the sentinel, but we can't
        // pass the sentinel to relocate_segment_after (it requires vessel indices).
        // Instead, test the intra-berth no-op: segment [2, 1] is already after V0.
        graph.relocate_segment_after(vessel(2), vessel(1), vessel(0));
        check_berth(&graph, 0, &[0, 2, 1]);
    }

    #[test]
    fn test_swap_segments_same_length_inter_berth() {
        let mut graph = standard_fixture();
        // Berth 0: [0, 2, 1], Berth 1: [3, 4]
        // Swap [0, 2] with [3, 4]
        graph.swap_segments(vessel(0), vessel(2), vessel(3), vessel(4));

        check_berth(&graph, 0, &[3, 4, 1]);
        check_berth(&graph, 1, &[0, 2]);
    }

    #[test]
    fn test_swap_segments_with_empty_berth_is_handled_by_relocate() {
        // Note: swap_segments requires real vessel segments.
        // Swapping with an empty berth is conceptually a 'relocate_segment_to_head/tail'.
        // This test just ensures we haven't broken the relocation logic that handles this.
        let mut graph = standard_fixture();
        graph.relocate_segment_to_head(vessel(3), vessel(4), berth(3));
        check_berth(&graph, 1, &[]);
        check_berth(&graph, 3, &[3, 4]);
    }

    #[test]
    fn test_reverse_segment_entire_graph() {
        let mut graph = standard_fixture();
        // Move everything to Berth 0 to make a long segment
        graph.relocate_to_tail(vessel(3), berth(0));
        graph.relocate_to_tail(vessel(4), berth(0));
        graph.relocate_to_tail(vessel(5), berth(0));

        check_berth(&graph, 0, &[0, 2, 1, 3, 4, 5]);

        // Reverse the entire sequence
        graph.reverse_segment(vessel(0), vessel(5));
        check_berth(&graph, 0, &[5, 4, 3, 1, 2, 0]);

        // Reverse a middle chunk
        graph.reverse_segment(vessel(3), vessel(2));
        check_berth(&graph, 0, &[5, 4, 2, 1, 3, 0]);
    }

    #[test]
    fn test_iterator_fused_behavior() {
        let graph = standard_fixture();
        let mut iter = graph.vessel_sequence_iter(berth(0));

        assert_eq!(iter.next(), Some(vessel(0)));
        assert_eq!(iter.next(), Some(vessel(2)));
        assert_eq!(iter.next(), Some(vessel(1)));

        // It should return None forever once exhausted
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_iterator_exact_size() {
        let graph = standard_fixture();
        let mut iter = graph.vessel_sequence_iter(berth(0));

        assert_eq!(iter.size_hint(), (3, Some(3)));
        assert_eq!(iter.len(), 3);

        iter.next();
        assert_eq!(iter.size_hint(), (2, Some(2)));
        assert_eq!(iter.len(), 2);
    }

    #[test]
    fn test_complex_multi_step_mutation() {
        // This test simulates a sequence of random moves and checks the final state
        // to ensure invariants (counts, assignments, linkages) survive a thrashing.
        let mut graph = standard_fixture();

        // Initial:
        // B0: 0 -> 2 -> 1
        // B1: 3 -> 4
        // B2: 5
        // B3: empty

        graph.swap_vessels(vessel(0), vessel(5));
        // B0: 5 -> 2 -> 1
        // B1: 3 -> 4
        // B2: 0

        graph.relocate_segment_to_head(vessel(2), vessel(1), berth(3));
        // B0: 5
        // B1: 3 -> 4
        // B2: 0
        // B3: 2 -> 1

        graph.relocate_after(vessel(3), vessel(5));
        // B0: 5 -> 3
        // B1: 4
        // B2: 0
        // B3: 2 -> 1

        graph.reverse_segment(vessel(5), vessel(3));
        // B0: 3 -> 5

        graph.swap_segments(vessel(3), vessel(5), vessel(2), vessel(1));
        // B0: 2 -> 1
        // B1: 4
        // B2: 0
        // B3: 3 -> 5

        check_berth(&graph, 0, &[2, 1]);
        check_berth(&graph, 1, &[4]);
        check_berth(&graph, 2, &[0]);
        check_berth(&graph, 3, &[3, 5]);
    }

    #[test]
    fn test_berth_edges_typical() {
        let graph = standard_fixture();
        // B0 is: V0 -> V2 -> V1
        let mut iter = graph.berth_edges(berth(0));

        assert_eq!(iter.len(), 2);

        let edge1 = iter.next().unwrap();
        assert_eq!(edge1.from, vessel(0));
        assert_eq!(edge1.to, vessel(2));
        assert_eq!(iter.len(), 1);

        let edge2 = iter.next().unwrap();
        assert_eq!(edge2.from, vessel(2));
        assert_eq!(edge2.to, vessel(1));
        assert_eq!(iter.len(), 0);

        assert_eq!(iter.next(), None);
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_berth_edges_single_and_empty() {
        let graph = standard_fixture();

        // B2 is: V5
        // A single vessel has no internal edges.
        let mut iter_single = graph.berth_edges(berth(2));
        assert_eq!(iter_single.len(), 0);
        assert_eq!(iter_single.next(), None);

        // B3 is empty.
        let mut iter_empty = graph.berth_edges(berth(3));
        assert_eq!(iter_empty.len(), 0);
        assert_eq!(iter_empty.next(), None);
    }

    #[test]
    fn test_all_edges_linear_scan() {
        let graph = standard_fixture();
        // Expected edges across all berths:
        // B0: Edge(V0 -> V2), Edge(V2 -> V1)
        // B1: Edge(V3 -> V4)
        // B2: (None, only V5)
        // B3: (None, empty)

        let mut edges: Vec<ScheduleGraphFullEdge> = graph.all_edges().collect();

        // Because all_edges scans memory sequentially (V0, V1, V2...),
        // the edges will be yielded in ascending order of the `from` index,
        // NOT in topological schedule order.

        // Let's trace memory order:
        // Vessel 0: next is V2 (B0). -> Yields Edge(V0 -> V2 on B0)
        // Vessel 1: next is Sentinel (B0). -> Skips
        // Vessel 2: next is V1 (B0). -> Yields Edge(V2 -> V1 on B0)
        // Vessel 3: next is V4 (B1). -> Yields Edge(V3 -> V4 on B1)
        // Vessel 4: next is Sentinel (B1). -> Skips
        // Vessel 5: next is Sentinel (B2). -> Skips

        assert_eq!(edges.len(), 3);

        // Sort by `from` vessel just in case, though the iterator naturally yields them this way
        edges.sort_by_key(|e| e.from.get());

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
    fn test_all_edges_empty_graph() {
        let empty_graph = ScheduleGraph::from_slices::<i32>(&[], &[], 3);
        let mut iter = empty_graph.all_edges();
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_all_edges_after_mutations() {
        let mut graph = standard_fixture();

        // B0: V0 -> V2 -> V1
        // Move V4 (from B1) to between V0 and V2
        // B0 becomes: V0 -> V4 -> V2 -> V1
        graph.relocate_after(vessel(4), vessel(0));

        // Let's trace expected memory order edges for the new graph:
        // V0: next is V4 (B0) -> Yields Edge(V0 -> V4 on B0)
        // V1: next is Sentinel (B0) -> Skips
        // V2: next is V1 (B0) -> Yields Edge(V2 -> V1 on B0)
        // V3: next is Sentinel (B1) -> Skips (V4 was removed)
        // V4: next is V2 (B0) -> Yields Edge(V4 -> V2 on B0)
        // V5: next is Sentinel (B2) -> Skips

        let mut edges: Vec<ScheduleGraphFullEdge> = graph.all_edges().collect();
        edges.sort_by_key(|e| e.from.get());

        assert_eq!(edges.len(), 3);
        assert_eq!(
            edges[0],
            ScheduleGraphFullEdge {
                from: vessel(0),
                to: vessel(4),
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
                from: vessel(4),
                to: vessel(2),
                on_berth: berth(0)
            }
        );
    }
}
