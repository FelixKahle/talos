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
// LinkEnd
// ----------------------------------------------------------------

/// One end of a link between consecutive places of a berth: a vessel, or the boundary of a berth.
///
/// A link runs from the head boundary of its berth or a vessel to a vessel or the tail boundary of its berth: a
/// [`Berth`](Self::Berth) end is the berth's head when it is where a link starts, and its tail when it is where a
/// link ends. An empty berth has the one link from its head to its tail.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum LinkEnd {
    /// The vessel served at this place of the berth.
    Vessel(VesselIndex),
    /// The boundary of the berth: its head where a link starts, its tail where one ends.
    Berth(BerthIndex),
}

impl fmt::Display for LinkEnd {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Vessel(vessel) => write!(f, "{vessel}"),
            Self::Berth(berth) => write!(f, "the boundary of {berth}"),
        }
    }
}

// ----------------------------------------------------------------
// VesselSequenceIter
// ----------------------------------------------------------------

/// The vessels of a berth, from its head to its tail.
///
/// The walk yields only vessels of its graph, in every build: it ends at the first node that is not a vessel's,
/// as well as at the end of the berth. A ring that held two sentinels so gives a wrong or a short order, never an
/// id outside the graph, and a caller may index its columns of vessels by every vessel the walk yields without a
/// check.
// Invariant: every node `inner` yields lies in the arena of a graph of `num_vessels` vessels, and the walk yields
// a node only if it is below `num_vessels`, so only the node of one of the graph's vessels.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct VesselSequenceIter<'a> {
    /// The walk over the nodes, which stops before the berth's sentinel.
    inner: RingSequenceIter<'a>,
    /// The graph's count of vessels; the walk ends at the first node at or beyond it, a sentinel.
    num_vessels: usize,
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
    /// The berth's head is next.
    Head,
    /// The berth's vessels are next, and after them its tail.
    Vessels,
    /// Every end has been yielded.
    Done,
}

/// The places of a berth from its head through its vessels to its tail, so that consecutive ends are the links of
/// the berth, its boundaries included.
#[derive(Clone, Debug)]
pub struct BerthTopologyIter<'a> {
    /// The walk over the berth's vessels.
    inner: VesselSequenceIter<'a>,
    /// The berth walked.
    berth: BerthIndex,
    /// Where the walk stands.
    state: BerthTopologyState,
}

impl Iterator for BerthTopologyIter<'_> {
    type Item = LinkEnd;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        match self.state {
            BerthTopologyState::Head => {
                self.state = BerthTopologyState::Vessels;
                Some(LinkEnd::Berth(self.berth))
            }
            BerthTopologyState::Vessels => self.inner.next().map(LinkEnd::Vessel).or_else(|| {
                self.state = BerthTopologyState::Done;
                Some(LinkEnd::Berth(self.berth))
            }),
            BerthTopologyState::Done => None,
        }
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lower, upper) = self.inner.inner.size_hint();
        let sentinels = match self.state {
            BerthTopologyState::Head => 2,
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
    VesselsAfter,
    VesselSequenceIter,
    BerthTopologyIter,
    #[cfg(test)]
    SwapVessels,
    #[cfg(test)]
    RelocateAfter,
    #[cfg(test)]
    RelocateToHead,
}

impl fmt::Display for Step {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::VesselBerth => "vessel_berth",
            Self::VesselCount => "vessel_count",
            Self::VesselPredecessor => "vessel_predecessor",
            Self::VesselsAfter => "vessels_after",
            Self::VesselSequenceIter => "vessel_sequence_iter",
            Self::BerthTopologyIter => "berth_topology_iter",
            #[cfg(test)]
            Self::SwapVessels => "swap_vessels",
            #[cfg(test)]
            Self::RelocateAfter => "relocate_after",
            #[cfg(test)]
            Self::RelocateToHead => "relocate_to_head",
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
/// (a swap of two vessels, a relocation of one) relinks a constant number of nodes. The graph also keeps each
/// vessel's berth and each berth's count of vessels, which every move keeps up to date.
///
/// Every read comes as a safe form, which checks in every build that the vessels and berths it is handed are the
/// graph's, and an `_unchecked` twin for the caller whose own invariant proves that they are. The moves are made
/// through their `_unchecked` forms; their safe forms are the references the tests check them against.
// Invariant: the arena has `num_vessels + num_berths` nodes, where `num_vessels` is the length of `vessel_berth`
// and `num_berths` that of `berth_vessel_count`; node `v` below `num_vessels` stands for vessel `v`, and node
// `num_vessels + b` is the sentinel of berth `b`. Every entry of `vessel_berth` is below `num_berths`. These are
// what memory rests on: the node of every vessel and the sentinel of every berth lie in the arena, and every berth
// read from `vessel_berth` indexes `berth_vessel_count`. The counts never change but through `overwrite_*`, which
// takes `&mut self`, so no id proven against the graph outlives a change of them.
// Further, each ring holds exactly one sentinel, `vessel_berth[v]` is the berth whose sentinel shares the ring of
// `v`, and `berth_vessel_count[b]` counts the vessels on the ring of `b`'s sentinel. The moves keep these; were
// they broken, the order and the counts would go wrong but no access would leave the graph: every column stays in
// range whatever the rings hold, and every walk over vessels ends at the first node that is not a vessel's
// (`VesselSequenceIter`), so it yields only the graph's vessels even from a ring that held two sentinels.
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
            let sentinel = self.sentinel_of(berth);
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

    /// The link end that `node` stands for: its vessel, or the berth whose sentinel it is.
    ///
    /// A node beyond the arena gives a berth beyond the graph's; no memory rests on it, since the end is only
    /// computed.
    #[inline(always)]
    pub(super) fn link_end(&self, node: Node) -> LinkEnd {
        match self.node_vessel(node) {
            Some(vessel) => LinkEnd::Vessel(vessel),
            None => LinkEnd::Berth(BerthIndex::new(node.index() - self.num_vessels())),
        }
    }

    /// The node after `node` on its ring, for a node the caller knows to be the graph's.
    ///
    /// # Safety
    ///
    /// `node` must be a vessel's node or a berth's sentinel of the graph.
    #[inline(always)]
    pub(super) unsafe fn next_node_unchecked(&self, node: Node) -> Node {
        debug_assert!(
            node.index() < self.arena.len(),
            "called `ScheduleGraph::next_node_unchecked` with {node}, outside a graph of {} nodes",
            self.arena.len()
        );
        // SAFETY: `node` lies in the arena, the precondition of this function.
        unsafe { self.arena.next_unchecked(node) }
    }

    /// The node before `node` on its ring, for a node the caller knows to be the graph's.
    ///
    /// # Safety
    ///
    /// `node` must be a vessel's node or a berth's sentinel of the graph.
    #[inline(always)]
    pub(super) unsafe fn prev_node_unchecked(&self, node: Node) -> Node {
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

    /// The vessels after `vessel` on its berth, from the one directly after it to the tail.
    ///
    /// # Panics
    ///
    /// Panics if `vessel` is not one of the graph's.
    #[inline]
    #[track_caller]
    pub fn vessels_after(&self, vessel: VesselIndex) -> VesselSequenceIter<'_> {
        self.check(Step::VesselsAfter, [vessel], []);
        // SAFETY: `vessel` is the graph's, as the check above found.
        unsafe { self.vessels_after_unchecked(vessel) }
    }

    /// The vessels after `vessel` on its berth, from the one directly after it to the tail, for a vessel the
    /// caller knows to be the graph's.
    ///
    /// # Safety
    ///
    /// `vessel` must be one of the graph's, the condition [`vessels_after`](Self::vessels_after) checks.
    #[inline(always)]
    pub unsafe fn vessels_after_unchecked(&self, vessel: VesselIndex) -> VesselSequenceIter<'_> {
        debug_assert!(
            vessel.index() < self.num_vessels(),
            "called `ScheduleGraph::vessels_after_unchecked` with {vessel}, outside a graph of {} vessels",
            self.num_vessels()
        );
        // SAFETY: `vessel` is the graph's, the precondition of this function.
        let berth = unsafe { self.vessel_berth_unchecked(vessel) };
        let sentinel = self.sentinel_of(berth);
        // SAFETY: `vessel` is the graph's, the precondition of this function, and `berth` was read from its column
        // of berths, so the vessel's node, the berth's sentinel and the node read after the vessel's lie in the
        // arena by the invariant of `ScheduleGraph`.
        let inner = unsafe {
            let after = self.arena.next_unchecked(node_of(vessel));
            self.arena.sequence_unchecked(after, sentinel)
        };
        VesselSequenceIter {
            inner,
            num_vessels: self.num_vessels(),
        }
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

    /// The places of `berth` as link ends: its head, its vessels from head to tail, and its tail.
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

    /// The places of `berth` as link ends, head to tail, for a berth the caller knows to be the graph's.
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
            berth,
            state: BerthTopologyState::Head,
        }
    }

    /// Exchanges the places of `a` and `b`, on one berth or across two; nothing changes if they are one vessel.
    ///
    /// # Panics
    ///
    /// Panics if `a` or `b` is not one of the graph's.
    #[cfg(test)]
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

    /// Moves `vessel` to directly after `anchor`, on its berth or another; nothing changes if `vessel` is `anchor`
    /// or already follows it.
    ///
    /// # Panics
    ///
    /// Panics if `vessel` or `anchor` is not one of the graph's.
    #[cfg(test)]
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

    /// Moves `vessel` to the head of `berth`.
    ///
    /// # Panics
    ///
    /// Panics if `vessel` or `berth` is not one of the graph's.
    #[cfg(test)]
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

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

        for pair in expected.windows(2) {
            assert_eq!(
                graph.vessel_predecessor(vessel(pair[1])),
                Some(vessel(pair[0]))
            );
        }
        for (place, &v) in expected.iter().enumerate() {
            let after: Vec<usize> = graph.vessels_after(vessel(v)).map(|v| v.index()).collect();
            assert_eq!(after, expected[place + 1..], "the vessels after {v}");
            // SAFETY: `v` stands on the berth, so it is one of the graph's.
            let unchecked: Vec<usize> = unsafe { graph.vessels_after_unchecked(vessel(v)) }
                .map(|v| v.index())
                .collect();
            assert_eq!(unchecked, after, "the vessels after {v}, unchecked");
        }
        if let Some(&head) = expected.first() {
            assert_eq!(graph.vessel_predecessor(vessel(head)), None);
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
    fn test_relocate_to_head() {
        let mut graph = standard_fixture();
        graph.relocate_to_head(vessel(4), berth(0));
        check_berth(&graph, 0, &[4, 0, 2, 1]);
        check_berth(&graph, 1, &[3]);
    }

    #[test]
    fn test_berth_topology_walks_from_the_head_through_the_vessels_to_the_tail() {
        let graph = standard_fixture();
        let topology: Vec<_> = graph.berth_topology_iter(berth(0)).collect();
        assert_eq!(
            topology,
            [
                LinkEnd::Berth(berth(0)),
                LinkEnd::Vessel(vessel(0)),
                LinkEnd::Vessel(vessel(2)),
                LinkEnd::Vessel(vessel(1)),
                LinkEnd::Berth(berth(0)),
            ]
        );
        let empty: Vec<_> = graph.berth_topology_iter(berth(3)).collect();
        assert_eq!(empty, [LinkEnd::Berth(berth(3)), LinkEnd::Berth(berth(3))]);
    }

    #[test]
    fn test_link_end_names_the_vessel_of_a_vessel_node_and_the_berth_of_a_sentinel() {
        let graph = standard_fixture();
        assert_eq!(
            graph.link_end(graph.vessel_node(vessel(5))),
            LinkEnd::Vessel(vessel(5))
        );
        assert_eq!(
            graph.link_end(graph.sentinel_of(berth(0))),
            LinkEnd::Berth(berth(0))
        );
        assert_eq!(
            graph.link_end(graph.sentinel_of(berth(3))),
            LinkEnd::Berth(berth(3))
        );
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
        graph.relocate_to_head(vessel(1), berth(3));
        graph.relocate_after(vessel(3), vessel(5));
        graph.relocate_after(vessel(2), vessel(1));
        graph.swap_vessels(vessel(3), vessel(4));

        check_berth(&graph, 0, &[5, 4]);
        check_berth(&graph, 1, &[3]);
        check_berth(&graph, 2, &[0]);
        check_berth(&graph, 3, &[1, 2]);
    }

    #[test]
    fn test_predecessor() {
        let graph = standard_fixture();
        assert_eq!(graph.vessel_predecessor(vessel(0)), None);
        assert_eq!(graph.vessel_predecessor(vessel(2)), Some(vessel(0)));
        assert_eq!(graph.vessel_predecessor(vessel(1)), Some(vessel(2)));
        assert_eq!(graph.vessel_predecessor(vessel(5)), None);
    }

    /// A move of the walk test: a swap of two vessels or a relocation of one after another, named by numbers read
    /// modulo the count of vessels.
    #[derive(Clone, Copy, Debug)]
    enum Shuffle {
        Swap(usize, usize),
        After(usize, usize),
    }

    fn shuffle() -> impl Strategy<Value = Shuffle> {
        prop_oneof![
            (0_usize..16, 0_usize..16).prop_map(|(a, b)| Shuffle::Swap(a, b)),
            (0_usize..16, 0_usize..16).prop_map(|(a, b)| Shuffle::After(a, b)),
        ]
    }

    proptest! {
        // Whatever moves the graph made, every walk over its vessels yields only vessels of the graph, each vessel
        // once over all berths, on the berth the graph records for it and as many as the berth counts: the
        // columns of vessels may be indexed by what the walks yield without a check.
        #[test]
        fn test_walks_yield_each_vessel_once_after_any_moves(
            berths in proptest::collection::vec(0_usize..3, 1..9),
            shuffles in proptest::collection::vec(shuffle(), 0..16),
        ) {
            let ids: Vec<BerthIndex> = berths.iter().map(|&b| berth(b)).collect();
            let mut graph = ScheduleGraph::from_slices(&ids, &vec![0; ids.len()], 3);
            let n = graph.num_vessels();
            for shuffle in shuffles {
                match shuffle {
                    Shuffle::Swap(a, b) => graph.swap_vessels(vessel(a % n), vessel(b % n)),
                    Shuffle::After(a, b) => graph.relocate_after(vessel(a % n), vessel(b % n)),
                }
            }
            let mut seen = vec![0_usize; n];
            for b in graph.berths() {
                let walked: Vec<VesselIndex> = graph.vessel_sequence_iter(b).collect();
                prop_assert_eq!(walked.len(), graph.vessel_count(b));
                for v in walked {
                    prop_assert!(v.index() < n, "the walk over {} yielded {}", b, v);
                    prop_assert_eq!(graph.vessel_berth(v), b);
                    seen[v.index()] += 1;
                }
            }
            prop_assert!(seen.iter().all(|&times| times == 1), "{:?}", seen);
        }
    }

    /// A safe step of the graph, as the refusal test calls it.
    #[derive(Clone, Copy, Debug)]
    enum Call {
        VesselBerth,
        VesselCount,
        VesselPredecessor,
        VesselsAfter,
        VesselSequenceIter,
        BerthTopologyIter,
        SwapVessels,
        RelocateAfter,
        RelocateToHead,
    }

    impl Call {
        const ALL: [Self; 9] = [
            Self::VesselBerth,
            Self::VesselCount,
            Self::VesselPredecessor,
            Self::VesselsAfter,
            Self::VesselSequenceIter,
            Self::BerthTopologyIter,
            Self::SwapVessels,
            Self::RelocateAfter,
            Self::RelocateToHead,
        ];

        /// The number of vessels and of berths the step takes.
        fn arity(self) -> (usize, usize) {
            match self {
                Self::VesselBerth | Self::VesselPredecessor | Self::VesselsAfter => (1, 0),
                Self::VesselCount | Self::VesselSequenceIter | Self::BerthTopologyIter => (0, 1),
                Self::SwapVessels | Self::RelocateAfter => (2, 0),
                Self::RelocateToHead => (1, 1),
            }
        }

        /// Makes the step on `graph` with the first vessels of `vessels` and the berth of `berths`.
        fn make(
            self,
            graph: &mut ScheduleGraph,
            vessels: [VesselIndex; 2],
            berths: [BerthIndex; 1],
        ) {
            let [a, b] = vessels;
            let [berth] = berths;
            match self {
                Self::VesselBerth => drop(graph.vessel_berth(a)),
                Self::VesselCount => drop(graph.vessel_count(berth)),
                Self::VesselPredecessor => drop(graph.vessel_predecessor(a)),
                Self::VesselsAfter => drop(graph.vessels_after(a)),
                Self::VesselSequenceIter => drop(graph.vessel_sequence_iter(berth)),
                Self::BerthTopologyIter => drop(graph.berth_topology_iter(berth)),
                Self::SwapVessels => graph.swap_vessels(a, b),
                Self::RelocateAfter => graph.relocate_after(a, b),
                Self::RelocateToHead => graph.relocate_to_head(a, berth),
            }
        }
    }

    /// The message of the panic `payload` carries.
    fn message_of(payload: &(dyn std::any::Any + Send)) -> String {
        payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| {
                payload
                    .downcast_ref::<&str>()
                    .map(|text| (*text).to_owned())
            })
            .unwrap_or_default()
    }

    proptest! {
        // Every id but one is the graph's; the one outside stands at any place the step takes an id.
        #[test]
        fn test_safe_steps_refuse_every_id_outside_and_change_nothing(
            call in proptest::sample::select(Call::ALL.to_vec()),
            inside in proptest::array::uniform2(0_usize..6),
            berth_inside in 0_usize..4,
            place in 0_usize..3,
            beyond in 0_usize..32,
        ) {
            let (num_vessels, num_berths) = call.arity();
            let place = place % (num_vessels + num_berths);
            let mut vessels = inside.map(vessel);
            let mut berths = [berth(berth_inside)];
            let outside = if place < num_vessels {
                vessels[place] = vessel(6 + beyond);
                vessels[place].to_string()
            } else {
                berths[0] = berth(4 + beyond);
                berths[0].to_string()
            };
            let mut graph = standard_fixture();
            let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                call.make(&mut graph, vessels, berths);
            }));
            let message = message_of(refused.expect_err("the step should refuse an id outside").as_ref());
            prop_assert!(message.starts_with("called `ScheduleGraph::"), "{}", message);
            prop_assert!(message.contains(&format!("with {outside}, outside a graph")), "{}", message);
            prop_assert_eq!(graph, standard_fixture());
        }
    }
}
