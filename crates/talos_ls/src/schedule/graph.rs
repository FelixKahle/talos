// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The order of the vessels on each berth, as one ring per berth.

use crate::ring_arena::{Node, RingArena, RingSequenceIter};
use std::fmt;
use std::iter::FusedIterator;
use talos_model::{BerthIndex, BerthIndexIter, VesselIndex, VesselIndexIter};
use typed_index_collections::TiVec;

/// The node that stands for `vessel`: the vessels take the first nodes of a graph's arena, in their order.
#[inline(always)]
fn node_of(vessel: VesselIndex) -> Node {
    Node::new(vessel.index())
}

/// The vessel that `node` stands for, a node below the graph's count of vessels.
#[inline(always)]
fn vessel_of(node: Node) -> VesselIndex {
    VesselIndex::new(node.index())
}

// ----------------------------------------------------------------
// VesselSequenceIter
// ----------------------------------------------------------------

/// The vessels of a berth, or of a segment of one, in their order along it or against it.
///
/// The walk yields only vessels of its graph, in every build: it ends at the first node that is not a vessel's,
/// as well as at the end of the berth or the segment. A graph whose order a move against its preconditions has
/// broken so gives a wrong or a short order, never an id outside the graph, and a caller may index its columns of
/// vessels by every vessel the walk yields without a check.
// Invariant: every node `inner` yields lies in the arena of a graph of `num_vessels` vessels, and the walk yields
// a node only if it is below `num_vessels`, so only the node of one of the graph's vessels.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct VesselSequenceIter<'a> {
    /// The walk over the nodes, which stops before the berth's sentinel or after the segment.
    inner: RingSequenceIter<'a>,
    /// The graph's count of vessels; the walk ends at the first node at or beyond it, a sentinel.
    num_vessels: usize,
}

impl<'a> VesselSequenceIter<'a> {
    /// The vessels of the segment `first ..= last` of the rings of `arena`, a graph's arena of `num_vessels`
    /// vessels, from `first` along its ring.
    ///
    /// # Safety
    ///
    /// `first` and `last` must be below `num_vessels`, and so nodes of `arena`.
    #[inline(always)]
    unsafe fn segment_unchecked(
        arena: &'a RingArena,
        num_vessels: usize,
        first: VesselIndex,
        last: VesselIndex,
    ) -> Self {
        debug_assert!(
            first.index() < num_vessels && last.index() < num_vessels && num_vessels <= arena.len(),
            "called `VesselSequenceIter::segment_unchecked` with {first} or {last}, outside a graph of \
             {num_vessels} vessels"
        );
        // SAFETY: `first` and `last` are below the count of vessels, the precondition of this function, so their
        // nodes lie in the arena by the invariant of `ScheduleGraph`, and so does the node read after `last`.
        let inner = unsafe {
            let after_last = arena.next_unchecked(node_of(last));
            arena.sequence_unchecked(node_of(first), after_last)
        };
        Self { inner, num_vessels }
    }
}

impl Iterator for VesselSequenceIter<'_> {
    type Item = VesselIndex;

    #[inline(always)]
    fn next(&mut self) -> Option<VesselIndex> {
        let node = self.inner.next()?;
        if node.index() < self.num_vessels {
            Some(vessel_of(node))
        } else {
            self.inner.end();
            None
        }
    }
}

impl FusedIterator for VesselSequenceIter<'_> {}

// ----------------------------------------------------------------
// BerthTopologyIter
// ----------------------------------------------------------------

/// Where a [`BerthTopologyIter`] stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BerthTopologyState {
    /// The sentinel that opens the berth is next.
    StartSentinel,
    /// The berth's vessels are next, and after them the sentinel that closes it.
    Vessels,
    /// Every node has been yielded.
    Done,
}

/// The nodes of a berth's ring, from its sentinel through its vessels back to its sentinel, so that consecutive
/// nodes are the links of the berth, its ends included.
#[derive(Clone, Debug)]
pub struct BerthTopologyIter<'a> {
    /// The walk over the berth's vessels.
    inner: VesselSequenceIter<'a>,
    /// The berth's sentinel.
    sentinel: Node,
    /// Where the walk stands.
    state: BerthTopologyState,
}

impl Iterator for BerthTopologyIter<'_> {
    type Item = Node;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        match self.state {
            BerthTopologyState::StartSentinel => {
                self.state = BerthTopologyState::Vessels;
                Some(self.sentinel)
            }
            BerthTopologyState::Vessels => self.inner.next().map(node_of).or_else(|| {
                self.state = BerthTopologyState::Done;
                Some(self.sentinel)
            }),
            BerthTopologyState::Done => None,
        }
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lower, upper) = self.inner.inner.size_hint();
        let sentinels = match self.state {
            BerthTopologyState::StartSentinel => 2,
            BerthTopologyState::Vessels => 1,
            BerthTopologyState::Done => 0,
        };
        let (lower, upper) = match self.state {
            BerthTopologyState::Done => (0, Some(0)),
            _ => (lower, upper),
        };
        (lower + sentinels, upper.map(|upper| upper + sentinels))
    }
}

impl FusedIterator for BerthTopologyIter<'_> {}

// ----------------------------------------------------------------
// Refusal
// ----------------------------------------------------------------

/// An id that is not one of a graph's.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Outside {
    /// A vessel at or beyond the graph's count of vessels.
    Vessel {
        vessel: VesselIndex,
        num_vessels: usize,
    },
    /// A berth at or beyond the graph's count of berths.
    Berth {
        berth: BerthIndex,
        num_berths: usize,
    },
}

impl fmt::Display for Outside {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Vessel {
                vessel,
                num_vessels,
            } => write!(f, "{vessel}, outside a graph of {num_vessels} vessels"),
            Self::Berth { berth, num_berths } => {
                write!(f, "{berth}, outside a graph of {num_berths} berths")
            }
        }
    }
}

/// A safe step of a [`ScheduleGraph`], as its refusal names it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Step {
    VesselBerth,
    VesselCount,
    VesselPredecessor,
    VesselSuccessor,
    VesselSequenceIter,
    VesselSequenceRevIter,
    BerthTopologyIter,
    Sentinel,
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

impl fmt::Display for Step {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::VesselBerth => "vessel_berth",
            Self::VesselCount => "vessel_count",
            Self::VesselPredecessor => "vessel_predecessor",
            Self::VesselSuccessor => "vessel_successor",
            Self::VesselSequenceIter => "vessel_sequence_iter",
            Self::VesselSequenceRevIter => "vessel_sequence_rev_iter",
            Self::BerthTopologyIter => "berth_topology_iter",
            Self::Sentinel => "sentinel",
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
        write!(f, "`ScheduleGraph::{name}`")
    }
}

/// An id handed to a safe step of the schedule (`S` names the step) that is not one of its graph's.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct Refusal<S> {
    /// The step that was handed the id.
    step: S,
    /// The first id handed over that is not the graph's, vessels before berths.
    outside: Outside,
}

impl<S> Refusal<S>
where
    S: fmt::Display,
{
    /// Panics with the refusal of the step.
    // Kept out of line so that the check of every safe step stays a compare and a branch on the caller's path.
    #[cold]
    #[inline(never)]
    #[track_caller]
    pub(super) fn panic(self) -> ! {
        panic!("{self}")
    }
}

impl<S> fmt::Display for Refusal<S>
where
    S: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "called {} with {}", self.step, self.outside)
    }
}

// ----------------------------------------------------------------
// ScheduleGraph
// ----------------------------------------------------------------

/// The order of the vessels on each berth: the genotype the local search changes, one move at a time.
///
/// Each berth is a ring of a ring arena that holds the berth's sentinel and its vessels in order, so that a move
/// (a swap, a relocation, a reversal) relinks a constant number of nodes, or as many as a segment it reverses. The
/// graph also keeps each vessel's berth and each berth's count of vessels, which every move keeps up to date.
///
/// Every step comes as a safe form, which checks in every build that the vessels and berths it is handed are the
/// graph's, and an `_unchecked` twin for the caller whose own invariant proves that they are. The steps on
/// segments also take conditions on the order no constant-time check confirms, stated in their `# Preconditions`.
// Invariant: the arena has `num_vessels + num_berths` nodes, where `num_vessels` is the length of `vessel_berth`
// and `num_berths` that of `berth_vessel_count`; node `v` below `num_vessels` stands for vessel `v`, and node
// `num_vessels + b` is the sentinel of berth `b`. Every entry of `vessel_berth` is below `num_berths`. These are
// what memory rests on: the node of every vessel and the sentinel of every berth lie in the arena, and every berth
// read from `vessel_berth` indexes `berth_vessel_count`. The counts never change but through `overwrite_*`, which
// takes `&mut self`, so no id proven against the graph outlives a change of them.
// Further, each ring holds exactly one sentinel, `vessel_berth[v]` is the berth whose sentinel shares the ring of
// `v`, and `berth_vessel_count[b]` counts the vessels on the ring of `b`'s sentinel. The moves keep these under
// their preconditions; breaking those gives a wrong order and wrong counts, never an access out of bounds: every
// column stays in range whatever the rings hold, and every walk over vessels ends at the first node that is not a
// vessel's (`VesselSequenceIter`), so it yields only the graph's vessels even from rings that hold two sentinels.
pub struct ScheduleGraph {
    /// The rings: one per berth, through the berth's sentinel and its vessels in order.
    arena: RingArena,
    /// The berth each vessel is on, per vessel.
    vessel_berth: TiVec<VesselIndex, BerthIndex>,
    /// The number of vessels on each berth, per berth.
    berth_vessel_count: TiVec<BerthIndex, usize>,
}

impl Clone for ScheduleGraph {
    fn clone(&self) -> Self {
        Self {
            arena: self.arena.clone(),
            vessel_berth: self.vessel_berth.clone(),
            berth_vessel_count: self.berth_vessel_count.clone(),
        }
    }

    // Written out so that overwriting a graph reuses the buffers of its arena and its columns; the derived form
    // would build a new graph, and the columns' own `clone_from` a new buffer each.
    fn clone_from(&mut self, source: &Self) {
        self.arena.clone_from(&source.arena);
        self.vessel_berth.raw.clone_from(&source.vessel_berth.raw);
        self.berth_vessel_count
            .raw
            .clone_from(&source.berth_vessel_count.raw);
    }
}

impl PartialEq for ScheduleGraph {
    fn eq(&self, other: &Self) -> bool {
        self.num_berths() == other.num_berths()
            && self.num_vessels() == other.num_vessels()
            && self.berths().all(|berth| {
                self.vessel_sequence_iter(berth)
                    .eq(other.vessel_sequence_iter(berth))
            })
    }
}

impl Eq for ScheduleGraph {}

impl std::hash::Hash for ScheduleGraph {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.num_vessels().hash(state);
        for berth in self.berths() {
            self.vessel_count(berth).hash(state);
            for vessel in self.vessel_sequence_iter(berth) {
                vessel.hash(state);
            }
        }
    }
}

impl fmt::Debug for ScheduleGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map()
            .entries(self.berths().map(|berth| {
                let vessels: Vec<_> = self.vessel_sequence_iter(berth).collect();
                (berth, vessels)
            }))
            .finish()
    }
}

impl fmt::Display for ScheduleGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for berth in self.berths() {
            write!(f, "{berth}:")?;
            for vessel in self.vessel_sequence_iter(berth) {
                write!(f, " {vessel}")?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

impl ScheduleGraph {
    /// The graph of `num_berths` berths that holds vessel `v` on `berths[v]`, the vessels of each berth ordered by
    /// their `start_times`, ties by vessel.
    ///
    /// # Panics
    ///
    /// Panics if `berths` and `start_times` differ in length, if a berth is not below `num_berths`, or if the
    /// vessels and berths together are more nodes than a node id can name.
    #[track_caller]
    pub fn from_slices<T>(berths: &[BerthIndex], start_times: &[T], num_berths: usize) -> Self
    where
        T: Ord,
    {
        let mut graph = Self {
            arena: RingArena::default(),
            vessel_berth: TiVec::new(),
            berth_vessel_count: TiVec::new(),
        };
        graph.overwrite_from_slices(berths, start_times, num_berths);
        graph
    }

    /// Makes the graph the one [`from_slices`](Self::from_slices) builds from the same arguments, reusing its
    /// memory but for one table of a position per vessel.
    ///
    /// # Panics
    ///
    /// Panics if `berths` and `start_times` differ in length, if a berth is not below `num_berths`, or if the
    /// vessels and berths together are more nodes than a node id can name; the graph is then left unchanged.
    #[track_caller]
    pub fn overwrite_from_slices<T>(
        &mut self,
        berths: &[BerthIndex],
        start_times: &[T],
        num_berths: usize,
    ) where
        T: Ord,
    {
        assert!(
            berths.len() == start_times.len(),
            "called `ScheduleGraph::overwrite_from_slices` with {} berths and {} start times",
            berths.len(),
            start_times.len()
        );
        if let Some(berth) = berths.iter().find(|berth| berth.index() >= num_berths) {
            panic!(
                "called `ScheduleGraph::overwrite_from_slices` with {berth}, outside a graph of {num_berths} \
                 berths"
            );
        }
        let num_nodes = berths.len().checked_add(num_berths);
        assert!(
            num_nodes.is_some_and(|num_nodes| Node::try_new(num_nodes.saturating_sub(1)).is_some()),
            "called `ScheduleGraph::overwrite_from_slices` with {} vessels and {num_berths} berths, more nodes \
             than a node id can name",
            berths.len()
        );
        let mut order: Vec<VesselIndex> = VesselIndexIter::new(berths.len()).collect();
        order.sort_by(|&left, &right| {
            berths[left.index()]
                .cmp(&berths[right.index()])
                .then_with(|| start_times[left.index()].cmp(&start_times[right.index()]))
        });

        self.arena.reset_to_self_loops(berths.len() + num_berths);
        self.vessel_berth.clear();
        self.vessel_berth.raw.extend_from_slice(berths);
        self.berth_vessel_count.clear();
        self.berth_vessel_count.resize(num_berths, 0);
        for vessel in order {
            let berth = berths[vessel.index()];
            let sentinel = self.sentinel(berth);
            debug_assert!(
                node_of(vessel).index() < self.arena.len() && sentinel.index() < self.arena.len()
            );
            // SAFETY: `vessel` is below the count of vessels and `sentinel` is a berth's sentinel, so both nodes lie
            // in the arena just reset to that many nodes.
            unsafe {
                self.arena
                    .relocate_before_unchecked(node_of(vessel), sentinel)
            };
            self.berth_vessel_count[berth] += 1;
        }
    }

    /// Makes the graph a copy of `other`, reusing its memory.
    #[inline]
    pub fn overwrite_from_graph(&mut self, other: &ScheduleGraph) {
        self.clone_from(other);
    }

    /// `vessel` as a refusal names it if it is not one of the graph's vessels.
    #[inline]
    pub(super) fn vessel_outside(&self, vessel: VesselIndex) -> Option<Outside> {
        (vessel.index() >= self.num_vessels()).then_some(Outside::Vessel {
            vessel,
            num_vessels: self.num_vessels(),
        })
    }

    /// `berth` as a refusal names it if it is not one of the graph's berths.
    #[inline]
    pub(super) fn berth_outside(&self, berth: BerthIndex) -> Option<Outside> {
        (berth.index() >= self.num_berths()).then_some(Outside::Berth {
            berth,
            num_berths: self.num_berths(),
        })
    }

    /// The refusal of `step` if one of `vessels` or `berths` is not one of the graph's.
    #[inline]
    pub(super) fn refusal<S, const V: usize, const B: usize>(
        &self,
        step: S,
        vessels: [VesselIndex; V],
        berths: [BerthIndex; B],
    ) -> Option<Refusal<S>> {
        let vessel = vessels
            .into_iter()
            .find_map(|vessel| self.vessel_outside(vessel));
        let berth = || {
            berths
                .into_iter()
                .find_map(|berth| self.berth_outside(berth))
        };
        vessel
            .or_else(berth)
            .map(|outside| Refusal { step, outside })
    }

    /// Panics with the refusal of `step` if one of `vessels` or `berths` is not one of the graph's.
    #[inline]
    #[track_caller]
    fn check<const V: usize, const B: usize>(
        &self,
        step: Step,
        vessels: [VesselIndex; V],
        berths: [BerthIndex; B],
    ) {
        if let Some(refusal) = self.refusal(step, vessels, berths) {
            refusal.panic();
        }
    }

    /// The node that stands for `vessel` in the graph's rings.
    #[inline(always)]
    pub(super) fn vessel_node(&self, vessel: VesselIndex) -> Node {
        debug_assert!(
            vessel.index() < self.num_vessels(),
            "called `ScheduleGraph::vessel_node` with {vessel}, outside a graph of {} vessels",
            self.num_vessels()
        );
        node_of(vessel)
    }

    /// The vessel that `node` stands for, or `None` if it is a berth's sentinel.
    #[inline(always)]
    pub(super) fn node_vessel(&self, node: Node) -> Option<VesselIndex> {
        (node.index() < self.num_vessels()).then(|| vessel_of(node))
    }

    /// The number of vessels.
    #[inline(always)]
    pub fn num_vessels(&self) -> usize {
        self.vessel_berth.len()
    }

    /// The number of berths.
    #[inline(always)]
    pub fn num_berths(&self) -> usize {
        self.berth_vessel_count.len()
    }

    /// Every berth, in increasing order.
    #[inline(always)]
    pub fn berths(&self) -> BerthIndexIter {
        BerthIndexIter::new(self.num_berths())
    }

    /// Every vessel, in increasing order.
    #[inline(always)]
    pub fn vessels(&self) -> VesselIndexIter {
        VesselIndexIter::new(self.num_vessels())
    }

    /// The node that opens and closes the ring of `berth`.
    ///
    /// # Panics
    ///
    /// Panics if `berth` is not one of the graph's.
    #[inline]
    #[track_caller]
    pub fn sentinel(&self, berth: BerthIndex) -> Node {
        self.check(Step::Sentinel, [], [berth]);
        Node::new(self.num_vessels() + berth.index())
    }

    /// The node that opens and closes the ring of `berth`, for a berth the caller knows to be the graph's.
    ///
    /// No memory rests on `berth`: the node is only computed. The node lies in the arena when `berth` is the
    /// graph's, which is what a caller that reads through it rests on.
    #[inline(always)]
    pub(super) fn sentinel_of(&self, berth: BerthIndex) -> Node {
        debug_assert!(
            berth.index() < self.num_berths(),
            "called `ScheduleGraph::sentinel_of` with {berth}, outside a graph of {} berths",
            self.num_berths()
        );
        Node::new(self.num_vessels() + berth.index())
    }

    /// The node after `node` on its ring.
    ///
    /// # Panics
    ///
    /// Panics if `node` is neither a vessel's node nor a berth's sentinel of the graph.
    #[inline]
    #[track_caller]
    pub fn next_node(&self, node: Node) -> Node {
        self.arena.next(node)
    }

    /// The node after `node` on its ring, for a node the caller knows to be the graph's.
    ///
    /// # Safety
    ///
    /// `node` must be a vessel's node or a berth's sentinel of the graph, the condition
    /// [`next_node`](Self::next_node) checks.
    #[inline(always)]
    pub unsafe fn next_node_unchecked(&self, node: Node) -> Node {
        debug_assert!(
            node.index() < self.arena.len(),
            "called `ScheduleGraph::next_node_unchecked` with {node}, outside a graph of {} nodes",
            self.arena.len()
        );
        // SAFETY: `node` lies in the arena, the precondition of this function.
        unsafe { self.arena.next_unchecked(node) }
    }

    /// The node before `node` on its ring.
    ///
    /// # Panics
    ///
    /// Panics if `node` is neither a vessel's node nor a berth's sentinel of the graph.
    #[inline]
    #[track_caller]
    pub fn prev_node(&self, node: Node) -> Node {
        self.arena.prev(node)
    }

    /// The node before `node` on its ring, for a node the caller knows to be the graph's.
    ///
    /// # Safety
    ///
    /// `node` must be a vessel's node or a berth's sentinel of the graph, the condition
    /// [`prev_node`](Self::prev_node) checks.
    #[inline(always)]
    pub unsafe fn prev_node_unchecked(&self, node: Node) -> Node {
        debug_assert!(
            node.index() < self.arena.len(),
            "called `ScheduleGraph::prev_node_unchecked` with {node}, outside a graph of {} nodes",
            self.arena.len()
        );
        // SAFETY: `node` lies in the arena, the precondition of this function.
        unsafe { self.arena.prev_unchecked(node) }
    }

    /// The berth `vessel` is on.
    ///
    /// # Panics
    ///
    /// Panics if `vessel` is not one of the graph's.
    #[inline]
    #[track_caller]
    pub fn vessel_berth(&self, vessel: VesselIndex) -> BerthIndex {
        self.check(Step::VesselBerth, [vessel], []);
        // SAFETY: `vessel` is the graph's, as the check above found.
        unsafe { self.vessel_berth_unchecked(vessel) }
    }

    /// The berth `vessel` is on, for a vessel the caller knows to be the graph's.
    ///
    /// # Safety
    ///
    /// `vessel` must be one of the graph's, the condition [`vessel_berth`](Self::vessel_berth) checks.
    #[inline(always)]
    pub unsafe fn vessel_berth_unchecked(&self, vessel: VesselIndex) -> BerthIndex {
        debug_assert!(
            vessel.index() < self.num_vessels(),
            "called `ScheduleGraph::vessel_berth_unchecked` with {vessel}, outside a graph of {} vessels",
            self.num_vessels()
        );
        // SAFETY: `vessel` is the graph's, the precondition of this function.
        *unsafe { self.vessel_berth.get_unchecked(vessel) }
    }

    /// The number of vessels on `berth`.
    ///
    /// # Panics
    ///
    /// Panics if `berth` is not one of the graph's.
    #[inline]
    #[track_caller]
    pub fn vessel_count(&self, berth: BerthIndex) -> usize {
        self.check(Step::VesselCount, [], [berth]);
        // SAFETY: `berth` is the graph's, as the check above found.
        unsafe { self.vessel_count_unchecked(berth) }
    }

    /// The number of vessels on `berth`, for a berth the caller knows to be the graph's.
    ///
    /// # Safety
    ///
    /// `berth` must be one of the graph's, the condition [`vessel_count`](Self::vessel_count) checks.
    #[inline(always)]
    pub unsafe fn vessel_count_unchecked(&self, berth: BerthIndex) -> usize {
        debug_assert!(
            berth.index() < self.num_berths(),
            "called `ScheduleGraph::vessel_count_unchecked` with {berth}, outside a graph of {} berths",
            self.num_berths()
        );
        // SAFETY: `berth` is the graph's, the precondition of this function.
        *unsafe { self.berth_vessel_count.get_unchecked(berth) }
    }

    /// The vessel directly before `vessel` on its berth, or `None` if `vessel` heads it.
    ///
    /// # Panics
    ///
    /// Panics if `vessel` is not one of the graph's.
    #[inline]
    #[track_caller]
    pub fn vessel_predecessor(&self, vessel: VesselIndex) -> Option<VesselIndex> {
        self.check(Step::VesselPredecessor, [vessel], []);
        // SAFETY: `vessel` is the graph's, as the check above found.
        unsafe { self.vessel_predecessor_unchecked(vessel) }
    }

    /// The vessel directly before `vessel` on its berth, for a vessel the caller knows to be the graph's.
    ///
    /// # Safety
    ///
    /// `vessel` must be one of the graph's, the condition [`vessel_predecessor`](Self::vessel_predecessor) checks.
    #[inline(always)]
    pub unsafe fn vessel_predecessor_unchecked(&self, vessel: VesselIndex) -> Option<VesselIndex> {
        debug_assert!(
            vessel.index() < self.num_vessels(),
            "called `ScheduleGraph::vessel_predecessor_unchecked` with {vessel}, outside a graph of {} vessels",
            self.num_vessels()
        );
        // SAFETY: `vessel` is the graph's, the precondition of this function, so its node lies in the arena by the
        // invariant of `ScheduleGraph`.
        let node = unsafe { self.arena.prev_unchecked(node_of(vessel)) };
        self.node_vessel(node)
    }

    /// The vessel directly after `vessel` on its berth, or `None` if `vessel` ends it.
    ///
    /// # Panics
    ///
    /// Panics if `vessel` is not one of the graph's.
    #[inline]
    #[track_caller]
    pub fn vessel_successor(&self, vessel: VesselIndex) -> Option<VesselIndex> {
        self.check(Step::VesselSuccessor, [vessel], []);
        // SAFETY: `vessel` is the graph's, as the check above found.
        unsafe { self.vessel_successor_unchecked(vessel) }
    }

    /// The vessel directly after `vessel` on its berth, for a vessel the caller knows to be the graph's.
    ///
    /// # Safety
    ///
    /// `vessel` must be one of the graph's, the condition [`vessel_successor`](Self::vessel_successor) checks.
    #[inline(always)]
    pub unsafe fn vessel_successor_unchecked(&self, vessel: VesselIndex) -> Option<VesselIndex> {
        debug_assert!(
            vessel.index() < self.num_vessels(),
            "called `ScheduleGraph::vessel_successor_unchecked` with {vessel}, outside a graph of {} vessels",
            self.num_vessels()
        );
        // SAFETY: `vessel` is the graph's, the precondition of this function, so its node lies in the arena by the
        // invariant of `ScheduleGraph`.
        let node = unsafe { self.arena.next_unchecked(node_of(vessel)) };
        self.node_vessel(node)
    }

    /// The vessels of `berth`, from its head to its tail.
    ///
    /// # Panics
    ///
    /// Panics if `berth` is not one of the graph's.
    #[inline]
    #[track_caller]
    pub fn vessel_sequence_iter(&self, berth: BerthIndex) -> VesselSequenceIter<'_> {
        self.check(Step::VesselSequenceIter, [], [berth]);
        // SAFETY: `berth` is the graph's, as the check above found.
        unsafe { self.vessel_sequence_iter_unchecked(berth) }
    }

    /// The vessels of `berth`, from its head to its tail, for a berth the caller knows to be the graph's.
    ///
    /// # Safety
    ///
    /// `berth` must be one of the graph's, the condition [`vessel_sequence_iter`](Self::vessel_sequence_iter)
    /// checks.
    #[inline(always)]
    pub unsafe fn vessel_sequence_iter_unchecked(
        &self,
        berth: BerthIndex,
    ) -> VesselSequenceIter<'_> {
        debug_assert!(
            berth.index() < self.num_berths(),
            "called `ScheduleGraph::vessel_sequence_iter_unchecked` with {berth}, outside a graph of {} berths",
            self.num_berths()
        );
        let sentinel = self.sentinel_of(berth);
        // SAFETY: `berth` is the graph's, the precondition of this function, so its sentinel lies in the arena by
        // the invariant of `ScheduleGraph`, and so does the node read after it.
        let inner = unsafe {
            let head = self.arena.next_unchecked(sentinel);
            self.arena.sequence_unchecked(head, sentinel)
        };
        VesselSequenceIter {
            inner,
            num_vessels: self.num_vessels(),
        }
    }

    /// The vessels of `berth`, from its tail to its head.
    ///
    /// # Panics
    ///
    /// Panics if `berth` is not one of the graph's.
    #[inline]
    #[track_caller]
    pub fn vessel_sequence_rev_iter(&self, berth: BerthIndex) -> VesselSequenceIter<'_> {
        self.check(Step::VesselSequenceRevIter, [], [berth]);
        // SAFETY: `berth` is the graph's, as the check above found.
        unsafe { self.vessel_sequence_rev_iter_unchecked(berth) }
    }

    /// The vessels of `berth`, from its tail to its head, for a berth the caller knows to be the graph's.
    ///
    /// # Safety
    ///
    /// `berth` must be one of the graph's, the condition
    /// [`vessel_sequence_rev_iter`](Self::vessel_sequence_rev_iter) checks.
    #[inline(always)]
    pub unsafe fn vessel_sequence_rev_iter_unchecked(
        &self,
        berth: BerthIndex,
    ) -> VesselSequenceIter<'_> {
        debug_assert!(
            berth.index() < self.num_berths(),
            "called `ScheduleGraph::vessel_sequence_rev_iter_unchecked` with {berth}, outside a graph of {} \
             berths",
            self.num_berths()
        );
        let sentinel = self.sentinel_of(berth);
        // SAFETY: `berth` is the graph's, the precondition of this function, so its sentinel lies in the arena by
        // the invariant of `ScheduleGraph`, and so does the node read before it.
        let inner = unsafe {
            let tail = self.arena.prev_unchecked(sentinel);
            self.arena.sequence_rev_unchecked(tail, sentinel)
        };
        VesselSequenceIter {
            inner,
            num_vessels: self.num_vessels(),
        }
    }

    /// The vessels of the segment `first ..= last`, from `first` along its berth.
    ///
    /// # Preconditions
    ///
    /// `last` follows `first` on its berth. Otherwise the walk yields the wrong vessels, though only vessels of the
    /// graph: it ends at the first sentinel it meets, and after at most the graph's count of nodes.
    ///
    /// # Safety
    ///
    /// `first` and `last` must be vessels of the graph.
    #[inline(always)]
    pub(super) unsafe fn segment_unchecked(
        &self,
        first: VesselIndex,
        last: VesselIndex,
    ) -> VesselSequenceIter<'_> {
        debug_assert!(
            first.index() < self.num_vessels() && last.index() < self.num_vessels(),
            "called `ScheduleGraph::segment_unchecked` with {first} or {last}, outside a graph of {} vessels",
            self.num_vessels()
        );
        // SAFETY: `first` and `last` are the graph's, the precondition of this function.
        unsafe {
            VesselSequenceIter::segment_unchecked(&self.arena, self.num_vessels(), first, last)
        }
    }

    /// The nodes of the ring of `berth`: its sentinel, its vessels from head to tail, and its sentinel again.
    ///
    /// # Panics
    ///
    /// Panics if `berth` is not one of the graph's.
    #[inline]
    #[track_caller]
    pub fn berth_topology_iter(&self, berth: BerthIndex) -> BerthTopologyIter<'_> {
        self.check(Step::BerthTopologyIter, [], [berth]);
        // SAFETY: `berth` is the graph's, as the check above found.
        unsafe { self.berth_topology_iter_unchecked(berth) }
    }

    /// The nodes of the ring of `berth`, sentinel to sentinel, for a berth the caller knows to be the graph's.
    ///
    /// # Safety
    ///
    /// `berth` must be one of the graph's, the condition [`berth_topology_iter`](Self::berth_topology_iter)
    /// checks.
    #[inline(always)]
    pub unsafe fn berth_topology_iter_unchecked(&self, berth: BerthIndex) -> BerthTopologyIter<'_> {
        debug_assert!(
            berth.index() < self.num_berths(),
            "called `ScheduleGraph::berth_topology_iter_unchecked` with {berth}, outside a graph of {} berths",
            self.num_berths()
        );
        // SAFETY: `berth` is the graph's, the precondition of this function.
        let inner = unsafe { self.vessel_sequence_iter_unchecked(berth) };
        BerthTopologyIter {
            inner,
            sentinel: self.sentinel_of(berth),
            state: BerthTopologyState::StartSentinel,
        }
    }

    /// Exchanges the places of `a` and `b`, on one berth or across two; nothing changes if they are one vessel.
    ///
    /// # Panics
    ///
    /// Panics if `a` or `b` is not one of the graph's.
    #[inline]
    #[track_caller]
    pub fn swap_vessels(&mut self, a: VesselIndex, b: VesselIndex) {
        self.check(Step::SwapVessels, [a, b], []);
        // SAFETY: both vessels are the graph's, as the check above found.
        unsafe { self.swap_vessels_unchecked(a, b) }
    }

    /// Exchanges the places of `a` and `b`, for vessels the caller knows to be the graph's.
    ///
    /// # Safety
    ///
    /// `a` and `b` must be the graph's, the condition [`swap_vessels`](Self::swap_vessels) checks.
    #[inline]
    pub unsafe fn swap_vessels_unchecked(&mut self, a: VesselIndex, b: VesselIndex) {
        debug_assert!(
            a.index() < self.num_vessels() && b.index() < self.num_vessels(),
            "called `ScheduleGraph::swap_vessels_unchecked` with {a} or {b}, outside a graph of {} vessels",
            self.num_vessels()
        );
        if a == b {
            return;
        }
        // SAFETY: both vessels are the graph's, the precondition of this function, so their nodes lie in the
        // arena by the invariant of `ScheduleGraph`.
        unsafe { self.arena.swap_nodes_unchecked(node_of(a), node_of(b)) };
        // SAFETY: both vessels are the graph's, the precondition of this function.
        unsafe {
            let berth_a = self.vessel_berth_unchecked(a);
            let berth_b = self.vessel_berth_unchecked(b);
            *self.vessel_berth.get_unchecked_mut(a) = berth_b;
            *self.vessel_berth.get_unchecked_mut(b) = berth_a;
        }
    }

    /// Exchanges the places of the segments `a_first ..= a_last` and `b_first ..= b_last`, on one berth or across
    /// two; nothing changes if both start at one vessel.
    ///
    /// # Preconditions
    ///
    /// `a_last` follows `a_first` on its berth, `b_last` follows `b_first`, and the segments share no vessel unless
    /// they are one segment. Otherwise the order breaks, though no access leaves the graph; a debug build checks
    /// it.
    ///
    /// # Panics
    ///
    /// Panics if any of the four vessels is not one of the graph's.
    #[inline]
    #[track_caller]
    pub fn swap_segments(
        &mut self,
        a_first: VesselIndex,
        a_last: VesselIndex,
        b_first: VesselIndex,
        b_last: VesselIndex,
    ) {
        self.check(Step::SwapSegments, [a_first, a_last, b_first, b_last], []);
        // SAFETY: the four vessels are the graph's, as the check above found.
        unsafe { self.swap_segments_unchecked(a_first, a_last, b_first, b_last) }
    }

    /// Exchanges the places of two segments, for vessels the caller knows to be the graph's.
    ///
    /// # Preconditions
    ///
    /// Those of [`swap_segments`](Self::swap_segments).
    ///
    /// # Safety
    ///
    /// The four vessels must be the graph's, the condition [`swap_segments`](Self::swap_segments) checks.
    #[inline]
    pub unsafe fn swap_segments_unchecked(
        &mut self,
        a_first: VesselIndex,
        a_last: VesselIndex,
        b_first: VesselIndex,
        b_last: VesselIndex,
    ) {
        debug_assert!(
            [a_first, a_last, b_first, b_last]
                .iter()
                .all(|vessel| vessel.index() < self.num_vessels()),
            "called `ScheduleGraph::swap_segments_unchecked` with {a_first}, {a_last}, {b_first} or {b_last}, \
             outside a graph of {} vessels",
            self.num_vessels()
        );
        if a_first == b_first {
            return;
        }
        // SAFETY: both first vessels are the graph's, the precondition of this function.
        let (berth_a, berth_b) = unsafe {
            (
                self.vessel_berth_unchecked(a_first),
                self.vessel_berth_unchecked(b_first),
            )
        };
        // SAFETY: the four vessels are the graph's, the precondition of this function, so their nodes lie in the
        // arena by the invariant of `ScheduleGraph`.
        unsafe {
            self.arena.swap_segments_unchecked(
                node_of(a_first),
                node_of(a_last),
                node_of(b_first),
                node_of(b_last),
            );
        }
        if berth_a != berth_b {
            // SAFETY: the vessels are the graph's, the precondition of this function, and `berth_a` and `berth_b`
            // were read from `vessel_berth`, so they are the graph's by the invariant of `ScheduleGraph`.
            unsafe {
                self.move_segment_berth_unchecked(a_first, a_last, berth_a, berth_b);
                self.move_segment_berth_unchecked(b_first, b_last, berth_b, berth_a);
            }
        }
    }

    /// Reverses the order of the segment `first ..= last` on its berth, at a cost of its length.
    ///
    /// # Preconditions
    ///
    /// `last` follows `first` on its berth. Otherwise the order breaks, though no access leaves the graph; a debug
    /// build checks it.
    ///
    /// # Panics
    ///
    /// Panics if `first` or `last` is not one of the graph's.
    #[inline]
    #[track_caller]
    pub fn reverse_segment(&mut self, first: VesselIndex, last: VesselIndex) {
        self.check(Step::ReverseSegment, [first, last], []);
        // SAFETY: both vessels are the graph's, as the check above found.
        unsafe { self.reverse_segment_unchecked(first, last) }
    }

    /// Reverses the order of the segment `first ..= last`, for vessels the caller knows to be the graph's.
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
            first.index() < self.num_vessels() && last.index() < self.num_vessels(),
            "called `ScheduleGraph::reverse_segment_unchecked` with {first} or {last}, outside a graph of {} \
             vessels",
            self.num_vessels()
        );
        // SAFETY: both vessels are the graph's, the precondition of this function, so their nodes lie in the arena
        // by the invariant of `ScheduleGraph`.
        unsafe {
            self.arena
                .reverse_segment_unchecked(node_of(first), node_of(last))
        };
    }

    /// Moves `vessel` to directly after `anchor`, on its berth or another; nothing changes if `vessel` is `anchor`
    /// or already follows it.
    ///
    /// # Panics
    ///
    /// Panics if `vessel` or `anchor` is not one of the graph's.
    #[inline]
    #[track_caller]
    pub fn relocate_after(&mut self, vessel: VesselIndex, anchor: VesselIndex) {
        self.check(Step::RelocateAfter, [vessel, anchor], []);
        // SAFETY: both vessels are the graph's, as the check above found.
        unsafe { self.relocate_after_unchecked(vessel, anchor) }
    }

    /// Moves `vessel` to directly after `anchor`, for vessels the caller knows to be the graph's.
    ///
    /// # Safety
    ///
    /// `vessel` and `anchor` must be the graph's, the condition [`relocate_after`](Self::relocate_after) checks.
    #[inline]
    pub unsafe fn relocate_after_unchecked(&mut self, vessel: VesselIndex, anchor: VesselIndex) {
        debug_assert!(
            vessel.index() < self.num_vessels() && anchor.index() < self.num_vessels(),
            "called `ScheduleGraph::relocate_after_unchecked` with {vessel} or {anchor}, outside a graph of {} \
             vessels",
            self.num_vessels()
        );
        // SAFETY: both vessels are the graph's, the precondition of this function, so their nodes lie in the arena
        // by the invariant of `ScheduleGraph`.
        unsafe {
            let berth = self.vessel_berth_unchecked(anchor);
            self.arena
                .relocate_after_unchecked(node_of(vessel), node_of(anchor));
            self.move_vessel_berth_unchecked(vessel, berth);
        }
    }

    /// Moves `vessel` to directly before `reference`, on its berth or another; nothing changes if `vessel` is
    /// `reference` or already precedes it.
    ///
    /// # Panics
    ///
    /// Panics if `vessel` or `reference` is not one of the graph's.
    #[inline]
    #[track_caller]
    pub fn relocate_before(&mut self, vessel: VesselIndex, reference: VesselIndex) {
        self.check(Step::RelocateBefore, [vessel, reference], []);
        // SAFETY: both vessels are the graph's, as the check above found.
        unsafe { self.relocate_before_unchecked(vessel, reference) }
    }

    /// Moves `vessel` to directly before `reference`, for vessels the caller knows to be the graph's.
    ///
    /// # Safety
    ///
    /// `vessel` and `reference` must be the graph's, the condition [`relocate_before`](Self::relocate_before)
    /// checks.
    #[inline]
    pub unsafe fn relocate_before_unchecked(
        &mut self,
        vessel: VesselIndex,
        reference: VesselIndex,
    ) {
        debug_assert!(
            vessel.index() < self.num_vessels() && reference.index() < self.num_vessels(),
            "called `ScheduleGraph::relocate_before_unchecked` with {vessel} or {reference}, outside a graph of {} \
             vessels",
            self.num_vessels()
        );
        // SAFETY: both vessels are the graph's, the precondition of this function, so their nodes lie in the arena
        // by the invariant of `ScheduleGraph`.
        unsafe {
            let berth = self.vessel_berth_unchecked(reference);
            self.arena
                .relocate_before_unchecked(node_of(vessel), node_of(reference));
            self.move_vessel_berth_unchecked(vessel, berth);
        }
    }

    /// Moves `vessel` to the head of `berth`.
    ///
    /// # Panics
    ///
    /// Panics if `vessel` or `berth` is not one of the graph's.
    #[inline]
    #[track_caller]
    pub fn relocate_to_head(&mut self, vessel: VesselIndex, berth: BerthIndex) {
        self.check(Step::RelocateToHead, [vessel], [berth]);
        // SAFETY: the vessel and the berth are the graph's, as the check above found.
        unsafe { self.relocate_to_head_unchecked(vessel, berth) }
    }

    /// Moves `vessel` to the head of `berth`, for a vessel and a berth the caller knows to be the graph's.
    ///
    /// # Safety
    ///
    /// `vessel` and `berth` must be the graph's, the condition [`relocate_to_head`](Self::relocate_to_head)
    /// checks.
    #[inline]
    pub unsafe fn relocate_to_head_unchecked(&mut self, vessel: VesselIndex, berth: BerthIndex) {
        debug_assert!(
            vessel.index() < self.num_vessels() && berth.index() < self.num_berths(),
            "called `ScheduleGraph::relocate_to_head_unchecked` with {vessel} or {berth}, outside a graph of {} \
             vessels and {} berths",
            self.num_vessels(),
            self.num_berths()
        );
        let sentinel = self.sentinel_of(berth);
        // SAFETY: the vessel and the berth are the graph's, the precondition of this function, so the vessel's node
        // and the berth's sentinel lie in the arena by the invariant of `ScheduleGraph`.
        unsafe {
            self.arena
                .relocate_after_unchecked(node_of(vessel), sentinel);
            self.move_vessel_berth_unchecked(vessel, berth);
        }
    }

    /// Moves `vessel` to the tail of `berth`.
    ///
    /// # Panics
    ///
    /// Panics if `vessel` or `berth` is not one of the graph's.
    #[inline]
    #[track_caller]
    pub fn relocate_to_tail(&mut self, vessel: VesselIndex, berth: BerthIndex) {
        self.check(Step::RelocateToTail, [vessel], [berth]);
        // SAFETY: the vessel and the berth are the graph's, as the check above found.
        unsafe { self.relocate_to_tail_unchecked(vessel, berth) }
    }

    /// Moves `vessel` to the tail of `berth`, for a vessel and a berth the caller knows to be the graph's.
    ///
    /// # Safety
    ///
    /// `vessel` and `berth` must be the graph's, the condition [`relocate_to_tail`](Self::relocate_to_tail)
    /// checks.
    #[inline]
    pub unsafe fn relocate_to_tail_unchecked(&mut self, vessel: VesselIndex, berth: BerthIndex) {
        debug_assert!(
            vessel.index() < self.num_vessels() && berth.index() < self.num_berths(),
            "called `ScheduleGraph::relocate_to_tail_unchecked` with {vessel} or {berth}, outside a graph of {} \
             vessels and {} berths",
            self.num_vessels(),
            self.num_berths()
        );
        let sentinel = self.sentinel_of(berth);
        // SAFETY: the vessel and the berth are the graph's, the precondition of this function, so the vessel's node
        // and the berth's sentinel lie in the arena by the invariant of `ScheduleGraph`.
        unsafe {
            self.arena
                .relocate_before_unchecked(node_of(vessel), sentinel);
            self.move_vessel_berth_unchecked(vessel, berth);
        }
    }

    /// Moves the segment `first ..= last` to directly after `anchor`, on its berth or another; nothing changes if
    /// it already follows `anchor`.
    ///
    /// # Preconditions
    ///
    /// `last` follows `first` on its berth, and `anchor` is not a vessel of the segment. Otherwise the order
    /// breaks, though no access leaves the graph; a debug build checks it.
    ///
    /// # Panics
    ///
    /// Panics if `first`, `last` or `anchor` is not one of the graph's.
    #[inline]
    #[track_caller]
    pub fn relocate_segment_after(
        &mut self,
        first: VesselIndex,
        last: VesselIndex,
        anchor: VesselIndex,
    ) {
        self.check(Step::RelocateSegmentAfter, [first, last, anchor], []);
        // SAFETY: the three vessels are the graph's, as the check above found.
        unsafe { self.relocate_segment_after_unchecked(first, last, anchor) }
    }

    /// Moves the segment `first ..= last` to directly after `anchor`, for vessels the caller knows to be the
    /// graph's.
    ///
    /// # Preconditions
    ///
    /// Those of [`relocate_segment_after`](Self::relocate_segment_after).
    ///
    /// # Safety
    ///
    /// The three vessels must be the graph's, the condition
    /// [`relocate_segment_after`](Self::relocate_segment_after) checks.
    #[inline]
    pub unsafe fn relocate_segment_after_unchecked(
        &mut self,
        first: VesselIndex,
        last: VesselIndex,
        anchor: VesselIndex,
    ) {
        debug_assert!(
            [first, last, anchor]
                .iter()
                .all(|vessel| vessel.index() < self.num_vessels()),
            "called `ScheduleGraph::relocate_segment_after_unchecked` with {first}, {last} or {anchor}, outside a \
             graph of {} vessels",
            self.num_vessels()
        );
        // SAFETY: the three vessels are the graph's, the precondition of this function, so their nodes lie in the
        // arena by the invariant of `ScheduleGraph`.
        unsafe {
            let (from, to) = (
                self.vessel_berth_unchecked(first),
                self.vessel_berth_unchecked(anchor),
            );
            self.arena.relocate_segment_after_unchecked(
                node_of(first),
                node_of(last),
                node_of(anchor),
            );
            self.move_segment_berth_unchecked(first, last, from, to);
        }
    }

    /// Moves the segment `first ..= last` to directly before `reference`, on its berth or another; nothing
    /// changes if it already precedes `reference`.
    ///
    /// # Preconditions
    ///
    /// `last` follows `first` on its berth, and `reference` is not a vessel of the segment. Otherwise the order
    /// breaks, though no access leaves the graph; a debug build checks it.
    ///
    /// # Panics
    ///
    /// Panics if `first`, `last` or `reference` is not one of the graph's.
    #[inline]
    #[track_caller]
    pub fn relocate_segment_before(
        &mut self,
        first: VesselIndex,
        last: VesselIndex,
        reference: VesselIndex,
    ) {
        self.check(Step::RelocateSegmentBefore, [first, last, reference], []);
        // SAFETY: the three vessels are the graph's, as the check above found.
        unsafe { self.relocate_segment_before_unchecked(first, last, reference) }
    }

    /// Moves the segment `first ..= last` to directly before `reference`, for vessels the caller knows to be the
    /// graph's.
    ///
    /// # Preconditions
    ///
    /// Those of [`relocate_segment_before`](Self::relocate_segment_before).
    ///
    /// # Safety
    ///
    /// The three vessels must be the graph's, the condition
    /// [`relocate_segment_before`](Self::relocate_segment_before) checks.
    #[inline]
    pub unsafe fn relocate_segment_before_unchecked(
        &mut self,
        first: VesselIndex,
        last: VesselIndex,
        reference: VesselIndex,
    ) {
        debug_assert!(
            [first, last, reference]
                .iter()
                .all(|vessel| vessel.index() < self.num_vessels()),
            "called `ScheduleGraph::relocate_segment_before_unchecked` with {first}, {last} or {reference}, \
             outside a graph of {} vessels",
            self.num_vessels()
        );
        // SAFETY: the three vessels are the graph's, the precondition of this function, so their nodes lie in the
        // arena by the invariant of `ScheduleGraph`.
        unsafe {
            let (from, to) = (
                self.vessel_berth_unchecked(first),
                self.vessel_berth_unchecked(reference),
            );
            self.arena.relocate_segment_before_unchecked(
                node_of(first),
                node_of(last),
                node_of(reference),
            );
            self.move_segment_berth_unchecked(first, last, from, to);
        }
    }

    /// Moves the segment `first ..= last` to the head of `berth`.
    ///
    /// # Preconditions
    ///
    /// `last` follows `first` on its berth. Otherwise the order breaks, though no access leaves the graph; a debug
    /// build checks it.
    ///
    /// # Panics
    ///
    /// Panics if `first`, `last` or `berth` is not one of the graph's.
    #[inline]
    #[track_caller]
    pub fn relocate_segment_to_head(
        &mut self,
        first: VesselIndex,
        last: VesselIndex,
        berth: BerthIndex,
    ) {
        self.check(Step::RelocateSegmentToHead, [first, last], [berth]);
        // SAFETY: the vessels and the berth are the graph's, as the check above found.
        unsafe { self.relocate_segment_to_head_unchecked(first, last, berth) }
    }

    /// Moves the segment `first ..= last` to the head of `berth`, for vessels and a berth the caller knows to be
    /// the graph's.
    ///
    /// # Preconditions
    ///
    /// Those of [`relocate_segment_to_head`](Self::relocate_segment_to_head).
    ///
    /// # Safety
    ///
    /// The vessels and the berth must be the graph's, the condition
    /// [`relocate_segment_to_head`](Self::relocate_segment_to_head) checks.
    #[inline]
    pub unsafe fn relocate_segment_to_head_unchecked(
        &mut self,
        first: VesselIndex,
        last: VesselIndex,
        berth: BerthIndex,
    ) {
        debug_assert!(
            first.index() < self.num_vessels()
                && last.index() < self.num_vessels()
                && berth.index() < self.num_berths(),
            "called `ScheduleGraph::relocate_segment_to_head_unchecked` with {first}, {last} or {berth}, outside \
             a graph of {} vessels and {} berths",
            self.num_vessels(),
            self.num_berths()
        );
        let sentinel = self.sentinel_of(berth);
        // SAFETY: the vessels and the berth are the graph's, the precondition of this function, so the vessels'
        // nodes and the berth's sentinel lie in the arena by the invariant of `ScheduleGraph`.
        unsafe {
            let from = self.vessel_berth_unchecked(first);
            self.arena
                .relocate_segment_after_unchecked(node_of(first), node_of(last), sentinel);
            self.move_segment_berth_unchecked(first, last, from, berth);
        }
    }

    /// Moves the segment `first ..= last` to the tail of `berth`; nothing changes if it already ends `berth`.
    ///
    /// # Preconditions
    ///
    /// `last` follows `first` on its berth. Otherwise the order breaks, though no access leaves the graph; a debug
    /// build checks it.
    ///
    /// # Panics
    ///
    /// Panics if `first`, `last` or `berth` is not one of the graph's.
    #[inline]
    #[track_caller]
    pub fn relocate_segment_to_tail(
        &mut self,
        first: VesselIndex,
        last: VesselIndex,
        berth: BerthIndex,
    ) {
        self.check(Step::RelocateSegmentToTail, [first, last], [berth]);
        // SAFETY: the vessels and the berth are the graph's, as the check above found.
        unsafe { self.relocate_segment_to_tail_unchecked(first, last, berth) }
    }

    /// Moves the segment `first ..= last` to the tail of `berth`, for vessels and a berth the caller knows to be
    /// the graph's.
    ///
    /// # Preconditions
    ///
    /// Those of [`relocate_segment_to_tail`](Self::relocate_segment_to_tail).
    ///
    /// # Safety
    ///
    /// The vessels and the berth must be the graph's, the condition
    /// [`relocate_segment_to_tail`](Self::relocate_segment_to_tail) checks.
    #[inline]
    pub unsafe fn relocate_segment_to_tail_unchecked(
        &mut self,
        first: VesselIndex,
        last: VesselIndex,
        berth: BerthIndex,
    ) {
        debug_assert!(
            first.index() < self.num_vessels()
                && last.index() < self.num_vessels()
                && berth.index() < self.num_berths(),
            "called `ScheduleGraph::relocate_segment_to_tail_unchecked` with {first}, {last} or {berth}, outside \
             a graph of {} vessels and {} berths",
            self.num_vessels(),
            self.num_berths()
        );
        let sentinel = self.sentinel_of(berth);
        // Before the sentinel rather than after the tail, which lies inside the segment when it ends the berth.
        // SAFETY: the vessels and the berth are the graph's, the precondition of this function, so the vessels'
        // nodes and the berth's sentinel lie in the arena by the invariant of `ScheduleGraph`.
        unsafe {
            let from = self.vessel_berth_unchecked(first);
            self.arena
                .relocate_segment_before_unchecked(node_of(first), node_of(last), sentinel);
            self.move_segment_berth_unchecked(first, last, from, berth);
        }
    }

    /// Records that `vessel` now stands on `berth`, which its ring has just made it.
    ///
    /// # Safety
    ///
    /// `vessel` and `berth` must be the graph's.
    #[inline(always)]
    unsafe fn move_vessel_berth_unchecked(&mut self, vessel: VesselIndex, berth: BerthIndex) {
        debug_assert!(
            vessel.index() < self.num_vessels() && berth.index() < self.num_berths(),
            "called `ScheduleGraph::move_vessel_berth_unchecked` with {vessel} or {berth}, outside the graph"
        );
        // SAFETY: `vessel` is the graph's, the precondition of this function.
        let from = unsafe { self.vessel_berth_unchecked(vessel) };
        if from == berth {
            return;
        }
        // SAFETY: `vessel` and `berth` are the graph's, the precondition of this function, and `from` was read from
        // `vessel_berth`, so it is the graph's by the invariant of `ScheduleGraph`.
        unsafe {
            *self.vessel_berth.get_unchecked_mut(vessel) = berth;
            *self.berth_vessel_count.get_unchecked_mut(from) -= 1;
            *self.berth_vessel_count.get_unchecked_mut(berth) += 1;
        }
    }

    /// Records that the vessels of the segment `first ..= last` now stand on `to`, which their ring has just made
    /// them, moved from `from`; at a cost of the segment's length if `from` and `to` differ.
    ///
    /// # Preconditions
    ///
    /// `last` follows `first` on its ring, and every vessel of the segment stood on `from`. Otherwise the berths
    /// and counts go wrong, and a debug build panics if the walk does not reach `last`; the walk ends at the first
    /// sentinel it meets, and after at most the graph's count of nodes.
    ///
    /// # Safety
    ///
    /// `first`, `last`, `from` and `to` must be the graph's.
    #[inline(always)]
    unsafe fn move_segment_berth_unchecked(
        &mut self,
        first: VesselIndex,
        last: VesselIndex,
        from: BerthIndex,
        to: BerthIndex,
    ) {
        debug_assert!(
            first.index() < self.num_vessels()
                && last.index() < self.num_vessels()
                && from.index() < self.num_berths()
                && to.index() < self.num_berths(),
            "called `ScheduleGraph::move_segment_berth_unchecked` with {first}, {last}, {from} or {to}, outside \
             the graph"
        );
        if from == to {
            return;
        }
        // SAFETY: `first` and `last` are the graph's, the precondition of this function.
        let segment = unsafe {
            VesselSequenceIter::segment_unchecked(&self.arena, self.vessel_berth.len(), first, last)
        };
        let mut moved = 0;
        let mut walked_to = first;
        for vessel in segment {
            debug_assert!(
                vessel.index() < self.vessel_berth.len(),
                "the walk over a segment should yield only vessels of the graph, but yielded {vessel}"
            );
            // SAFETY: the walk yields only vessels of the graph (`VesselSequenceIter`), which index `vessel_berth`.
            *unsafe { self.vessel_berth.get_unchecked_mut(vessel) } = to;
            moved += 1;
            walked_to = vessel;
        }
        debug_assert!(
            walked_to == last,
            "called `ScheduleGraph::move_segment_berth_unchecked` with {last}, which does not follow {first} on its \
             berth"
        );
        // SAFETY: `from` and `to` are the graph's, the precondition of this function.
        unsafe {
            *self.berth_vessel_count.get_unchecked_mut(from) -= moved;
            *self.berth_vessel_count.get_unchecked_mut(to) += moved;
        }
    }
}

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

        let forward: Vec<usize> = graph.vessel_sequence_iter(b).map(|v| v.index()).collect();
        assert_eq!(
            forward, expected,
            "Forward mismatch for Berth {}",
            berth_index
        );

        let reverse: Vec<usize> = graph
            .vessel_sequence_rev_iter(b)
            .map(|v| v.index())
            .collect();
        let mut expected_rev = expected.to_vec();
        expected_rev.reverse();
        assert_eq!(
            reverse, expected_rev,
            "Reverse mismatch for Berth {}",
            berth_index
        );

        for pair in expected.windows(2) {
            assert_eq!(
                graph.vessel_successor(vessel(pair[0])),
                Some(vessel(pair[1]))
            );
            assert_eq!(
                graph.vessel_predecessor(vessel(pair[1])),
                Some(vessel(pair[0]))
            );
        }
        if let (Some(&head), Some(&tail)) = (expected.first(), expected.last()) {
            assert_eq!(graph.vessel_predecessor(vessel(head)), None);
            assert_eq!(graph.vessel_successor(vessel(tail)), None);
        }

        for &v in expected {
            assert_eq!(graph.vessel_berth(vessel(v)).index(), berth_index);
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

    // The reference heads berth 1, so the node before it is the berth's sentinel, not a vessel.
    #[test]
    fn test_relocate_segment_before_the_head_of_a_berth() {
        let mut graph = standard_fixture();
        graph.relocate_segment_before(vessel(0), vessel(2), vessel(3));
        check_berth(&graph, 0, &[1]);
        check_berth(&graph, 1, &[0, 2, 3, 4]);
    }

    // The segment already stands before the reference, so the node before the reference is the segment's last.
    #[test]
    fn test_relocate_segment_before_the_vessel_it_precedes_changes_nothing() {
        let mut graph = standard_fixture();
        graph.relocate_segment_before(vessel(0), vessel(2), vessel(1));
        check_berth(&graph, 0, &[0, 2, 1]);
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

    // The segment [V3, V4] already ends berth 1, so the berth's tail lies inside it.
    #[test]
    fn test_relocate_segment_to_the_tail_it_already_ends_changes_nothing() {
        let mut graph = standard_fixture();
        graph.relocate_segment_to_tail(vessel(3), vessel(4), berth(1));
        check_berth(&graph, 1, &[3, 4]);
        // SAFETY: the vessels and the berth are the graph's.
        unsafe { graph.relocate_segment_to_tail_unchecked(vessel(3), vessel(4), berth(1)) };
        check_berth(&graph, 1, &[3, 4]);
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

    // The target is larger than the source in both counts, so its buffers hold the copy without growing.
    #[test]
    fn test_overwrite_from_graph_reuses_the_buffers() {
        let source = standard_fixture();
        let mut target = ScheduleGraph::from_slices(&[berth(0); 10], &[0; 10], 6);
        let buffers = (
            target.vessel_berth.as_ptr(),
            target.berth_vessel_count.as_ptr(),
        );
        target.overwrite_from_graph(&source);
        assert_eq!(target, source);
        assert_eq!(
            (
                target.vessel_berth.as_ptr(),
                target.berth_vessel_count.as_ptr()
            ),
            buffers
        );
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

    /// Asserts that every walk over the vessels of `graph`, along each berth and against it, yields only vessels of
    /// the graph.
    #[cfg(not(debug_assertions))]
    fn assert_walks_yield_only_vessels(graph: &ScheduleGraph) {
        for b in graph.berths() {
            let along = graph.vessel_sequence_iter(b);
            let against = graph.vessel_sequence_rev_iter(b);
            for v in along.chain(against) {
                assert!(
                    v.index() < graph.num_vessels(),
                    "the walk over {b} should yield only vessels of a graph of {} vessels, but yielded {v}",
                    graph.num_vessels()
                );
            }
        }
    }

    // B0: V0 V1, B1: V2 V3. The segment V0 ..= V3 runs over the end of B0 into B1, against the precondition of
    // `swap_segments`, which only a debug build checks; the order then breaks, but no walk may yield a sentinel.
    #[cfg(not(debug_assertions))]
    #[test]
    fn test_walks_yield_only_vessels_after_a_swap_of_a_segment_across_two_berths() {
        let mut graph =
            ScheduleGraph::from_slices(&[berth(0), berth(0), berth(1), berth(1)], &[0, 1, 0, 1], 2);
        graph.swap_segments(vessel(0), vessel(3), vessel(1), vessel(1));
        assert_walks_yield_only_vessels(&graph);
    }

    // B0: V0 V1, B1: V2 V3. The segment V1 ..= V0 runs through the sentinel of B0, against the precondition of
    // `swap_segments`; swapped with V2 it takes that sentinel into the ring of B1, and the vessels it moves to B1
    // are recorded by a walk that must stop at the sentinel rather than read past the vessels.
    #[cfg(not(debug_assertions))]
    #[test]
    fn test_walks_yield_only_vessels_after_a_swap_of_a_segment_through_a_sentinel() {
        let mut graph =
            ScheduleGraph::from_slices(&[berth(0), berth(0), berth(1), berth(1)], &[0, 1, 0, 1], 2);
        graph.swap_segments(vessel(1), vessel(0), vessel(2), vessel(2));
        assert_walks_yield_only_vessels(&graph);
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
