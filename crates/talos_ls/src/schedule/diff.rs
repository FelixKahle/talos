// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The record of what a sequence of moves changed in a schedule graph.

use super::graph::LinkEnd;
use std::fmt;
use std::iter::FusedIterator;
use talos_model::{BerthIndex, VesselIndex};

// ----------------------------------------------------------------
// Link
// ----------------------------------------------------------------

/// A link between consecutive places of a berth.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Link {
    /// The place the link leaves: a vessel, or the head of its berth.
    pub from: LinkEnd,
    /// The place the link reaches: a vessel, or the tail of its berth.
    pub to: LinkEnd,
}

impl fmt::Display for Link {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} -> {}", self.from, self.to)
    }
}

// ----------------------------------------------------------------
// Reallocation
// ----------------------------------------------------------------

/// A vessel a move took from one berth to another.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Reallocation {
    /// The vessel moved.
    pub vessel: VesselIndex,
    /// The berth the vessel stood on before the move.
    pub from: BerthIndex,
    /// The berth the move took the vessel to.
    pub to: BerthIndex,
}

impl fmt::Display for Reallocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} from {} to {}", self.vessel, self.from, self.to)
    }
}

// ----------------------------------------------------------------
// DiffLinkIter
// ----------------------------------------------------------------

/// The links a [`ScheduleGraphDiff`] records as broken, or as created, in the order the moves recorded them.
// Invariant: `from` and `to` walk two columns of one length in step, so they end together.
#[derive(Clone, Debug)]
pub struct DiffLinkIter<'a> {
    /// The place each link leaves.
    from: std::slice::Iter<'a, LinkEnd>,
    /// The place each link reaches.
    to: std::slice::Iter<'a, LinkEnd>,
}

impl Iterator for DiffLinkIter<'_> {
    type Item = Link;

    #[inline(always)]
    fn next(&mut self) -> Option<Link> {
        let from = *self.from.next()?;
        let to = *self.to.next()?;
        Some(Link { from, to })
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.from.size_hint()
    }
}

impl ExactSizeIterator for DiffLinkIter<'_> {}

impl FusedIterator for DiffLinkIter<'_> {}

// ----------------------------------------------------------------
// DiffReallocationIter
// ----------------------------------------------------------------

/// The vessels a [`ScheduleGraphDiff`] records as moved between berths, in the order the moves recorded them.
// Invariant: the three walks run over columns of one length in step, so they end together.
#[derive(Clone, Debug)]
pub struct DiffReallocationIter<'a> {
    /// The vessel of each reallocation.
    vessels: std::slice::Iter<'a, VesselIndex>,
    /// The berth each vessel left.
    from: std::slice::Iter<'a, BerthIndex>,
    /// The berth each vessel was taken to.
    to: std::slice::Iter<'a, BerthIndex>,
}

impl Iterator for DiffReallocationIter<'_> {
    type Item = Reallocation;

    #[inline(always)]
    fn next(&mut self) -> Option<Reallocation> {
        let vessel = *self.vessels.next()?;
        let from = *self.from.next()?;
        let to = *self.to.next()?;
        Some(Reallocation { vessel, from, to })
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.vessels.size_hint()
    }
}

impl ExactSizeIterator for DiffReallocationIter<'_> {}

impl FusedIterator for DiffReallocationIter<'_> {}

// ----------------------------------------------------------------
// ScheduleGraphDiff
// ----------------------------------------------------------------

/// The links a sequence of moves broke and made, and the vessels it took from one berth to another.
///
/// A [`Mutator`](super::Mutator) records each move here as it makes it, and a metaheuristic reads the record to
/// price a candidate by what changed rather than by the whole schedule: the links of the graph before the moves,
/// less the broken ones and with the created ones added, are the links after them, counted with multiplicity.
/// Clearing keeps the memory, so a run records its moves without allocating once the record has grown.
// Invariant: `broken_from` and `broken_to` have one length, so do `created_from` and `created_to`, and so do the
// three columns of reallocations; entry `i` of each group of columns is one record.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct ScheduleGraphDiff {
    /// The place each broken link left, per broken link.
    broken_from: Vec<LinkEnd>,
    /// The place each broken link reached, per broken link.
    broken_to: Vec<LinkEnd>,
    /// The place each created link leaves, per created link.
    created_from: Vec<LinkEnd>,
    /// The place each created link reaches, per created link.
    created_to: Vec<LinkEnd>,
    /// The vessel of each reallocation, per reallocation.
    reallocated_vessels: Vec<VesselIndex>,
    /// The berth each reallocated vessel left, per reallocation.
    reallocated_from: Vec<BerthIndex>,
    /// The berth each reallocated vessel was taken to, per reallocation.
    reallocated_to: Vec<BerthIndex>,
}

impl Clone for ScheduleGraphDiff {
    fn clone(&self) -> Self {
        Self {
            broken_from: self.broken_from.clone(),
            broken_to: self.broken_to.clone(),
            created_from: self.created_from.clone(),
            created_to: self.created_to.clone(),
            reallocated_vessels: self.reallocated_vessels.clone(),
            reallocated_from: self.reallocated_from.clone(),
            reallocated_to: self.reallocated_to.clone(),
        }
    }

    // Written out so that overwriting a diff reuses the buffers of its columns; the derived form would allocate.
    fn clone_from(&mut self, source: &Self) {
        self.broken_from.clone_from(&source.broken_from);
        self.broken_to.clone_from(&source.broken_to);
        self.created_from.clone_from(&source.created_from);
        self.created_to.clone_from(&source.created_to);
        self.reallocated_vessels
            .clone_from(&source.reallocated_vessels);
        self.reallocated_from.clone_from(&source.reallocated_from);
        self.reallocated_to.clone_from(&source.reallocated_to);
    }
}

impl ScheduleGraphDiff {
    /// The empty record.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Empties the record, keeping its memory.
    #[inline(always)]
    pub fn clear(&mut self) {
        self.broken_from.clear();
        self.broken_to.clear();
        self.created_from.clear();
        self.created_to.clear();
        self.reallocated_vessels.clear();
        self.reallocated_from.clear();
        self.reallocated_to.clear();
    }

    /// Makes the record a copy of `other`, reusing its memory.
    #[inline(always)]
    pub fn overwrite_from_graph_diff(&mut self, other: &Self) {
        self.clone_from(other);
    }

    /// Records that a move broke the link `from -> to`.
    #[inline(always)]
    pub(super) fn push_link_broken(&mut self, from: LinkEnd, to: LinkEnd) {
        self.broken_from.push(from);
        self.broken_to.push(to);
    }

    /// Records that a move made the link `from -> to`.
    #[inline(always)]
    pub(super) fn push_link_created(&mut self, from: LinkEnd, to: LinkEnd) {
        self.created_from.push(from);
        self.created_to.push(to);
    }

    /// Records that a move took `vessel` from berth `from` to berth `to`.
    #[inline(always)]
    pub(super) fn push_reallocation(
        &mut self,
        vessel: VesselIndex,
        from: BerthIndex,
        to: BerthIndex,
    ) {
        self.reallocated_vessels.push(vessel);
        self.reallocated_from.push(from);
        self.reallocated_to.push(to);
    }

    /// The links recorded as broken, in the order the moves recorded them.
    #[inline(always)]
    pub fn broken_links(&self) -> DiffLinkIter<'_> {
        DiffLinkIter {
            from: self.broken_from.iter(),
            to: self.broken_to.iter(),
        }
    }

    /// The links recorded as created, in the order the moves recorded them.
    #[inline(always)]
    pub fn created_links(&self) -> DiffLinkIter<'_> {
        DiffLinkIter {
            from: self.created_from.iter(),
            to: self.created_to.iter(),
        }
    }

    /// The vessels recorded as moved between berths, in the order the moves recorded them.
    #[inline(always)]
    pub fn reallocations(&self) -> DiffReallocationIter<'_> {
        DiffReallocationIter {
            vessels: self.reallocated_vessels.iter(),
            from: self.reallocated_from.iter(),
            to: self.reallocated_to.iter(),
        }
    }

    /// Whether every vessel the records name is below `num_vessels` and every berth below `num_berths`: the
    /// condition of every unchecked read through them. It costs one pass over the records.
    pub fn names_only_ids_below(&self, num_vessels: usize, num_berths: usize) -> bool {
        let end_below = |end: LinkEnd| match end {
            LinkEnd::Vessel(vessel) => vessel.index() < num_vessels,
            LinkEnd::Berth(berth) => berth.index() < num_berths,
        };
        let links_below = self
            .broken_links()
            .chain(self.created_links())
            .all(|link| end_below(link.from) && end_below(link.to));
        let reallocations_below = self.reallocations().all(|moved| {
            moved.vessel.index() < num_vessels
                && moved.from.index() < num_berths
                && moved.to.index() < num_berths
        });
        links_below && reallocations_below
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_names_only_ids_below_checks_every_vessel_and_berth_of_the_records() {
        let diff = recorded();
        assert!(diff.names_only_ids_below(3, 2));
        assert!(!diff.names_only_ids_below(2, 2));
        assert!(!diff.names_only_ids_below(3, 1));
        assert!(ScheduleGraphDiff::new().names_only_ids_below(0, 0));
    }

    fn vessel(index: usize) -> LinkEnd {
        LinkEnd::Vessel(VesselIndex::new(index))
    }

    fn berth(index: usize) -> BerthIndex {
        BerthIndex::new(index)
    }

    fn recorded() -> ScheduleGraphDiff {
        let mut diff = ScheduleGraphDiff::new();
        diff.push_link_broken(LinkEnd::Berth(berth(0)), vessel(1));
        diff.push_link_broken(vessel(1), vessel(2));
        diff.push_link_created(LinkEnd::Berth(berth(0)), vessel(2));
        diff.push_reallocation(VesselIndex::new(1), berth(0), berth(1));
        diff
    }

    #[test]
    fn test_the_walks_yield_the_records_in_their_order() {
        let diff = recorded();
        assert_eq!(
            diff.broken_links().collect::<Vec<_>>(),
            [
                Link {
                    from: LinkEnd::Berth(berth(0)),
                    to: vessel(1)
                },
                Link {
                    from: vessel(1),
                    to: vessel(2)
                },
            ]
        );
        assert_eq!(
            diff.created_links().collect::<Vec<_>>(),
            [Link {
                from: LinkEnd::Berth(berth(0)),
                to: vessel(2)
            }]
        );
        assert_eq!(
            diff.reallocations().collect::<Vec<_>>(),
            [Reallocation {
                vessel: VesselIndex::new(1),
                from: berth(0),
                to: berth(1)
            }]
        );
    }

    #[test]
    fn test_the_walks_know_their_length() {
        let diff = recorded();
        let mut broken = diff.broken_links();
        assert_eq!(broken.len(), 2);
        broken.next();
        assert_eq!(broken.len(), 1);
        assert_eq!(diff.created_links().len(), 1);
        assert_eq!(diff.reallocations().len(), 1);
    }

    #[test]
    fn test_clear_empties_the_record_and_keeps_its_memory() {
        let mut diff = recorded();
        let capacity = diff.broken_from.capacity();
        diff.clear();
        assert_eq!(diff, ScheduleGraphDiff::new());
        assert_eq!(diff.broken_from.capacity(), capacity);
    }

    #[test]
    fn test_overwrite_copies_the_record_into_the_buffers_it_has() {
        let source = recorded();
        let mut target = recorded();
        target.push_link_broken(vessel(3), vessel(4));
        let buffer = target.broken_from.as_ptr();
        target.overwrite_from_graph_diff(&source);
        assert_eq!(target, source);
        assert_eq!(target.broken_from.as_ptr(), buffer);
    }

    #[test]
    fn test_links_and_reallocations_print_their_ends() {
        let link = Link {
            from: LinkEnd::Berth(berth(0)),
            to: vessel(1),
        };
        assert_eq!(link.to_string(), "the boundary of Berth 0 -> Vessel 1");
        let reallocation = Reallocation {
            vessel: VesselIndex::new(1),
            from: berth(0),
            to: berth(1),
        };
        assert_eq!(reallocation.to_string(), "Vessel 1 from Berth 0 to Berth 1");
    }
}
