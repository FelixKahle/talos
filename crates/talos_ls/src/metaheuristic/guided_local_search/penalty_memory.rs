// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The memory of the penalties of a guided local search: a count per feature of the schedules of one instance.
//!
//! The features of a schedule are its links (a vessel next to a vessel or to the boundary of its berth, with the
//! boundaries of all berths taken as one) and its assignments (a vessel on a berth).

use crate::schedule::{LinkEnd, ScheduleGraph, ScheduleGraphDiff};
use talos_model::{BerthIndex, VesselIndex};

/// A feature of a schedule, by its place in the tables of a penalty memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Feature {
    /// The link at this index of the table of links.
    Link(usize),
    /// The assignment at this index of the table of assignments.
    Assignment(usize),
}

/// Where a feature's utility stands against the best utility found so far.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UtilityRank {
    Better,
    Tie,
    Worse,
}

/// Where `utility` stands against `best`: a tie when they differ by at most a relative `1e-12` of the larger
/// magnitude, or of one where both are smaller.
///
/// The tolerance absorbs the rounding of `1 / (1 + penalty)` while keeping utilities of different penalties apart,
/// which differ by far more.
fn rank_utility(utility: f64, best: f64) -> UtilityRank {
    const RELATIVE_TOLERANCE: f64 = 1e-12;
    let difference = utility - best;
    let scale = utility.abs().max(best.abs()).max(1.0);
    if best == f64::NEG_INFINITY || difference > RELATIVE_TOLERANCE * scale {
        UtilityRank::Better
    } else if difference.abs() <= RELATIVE_TOLERANCE * scale {
        UtilityRank::Tie
    } else {
        UtilityRank::Worse
    }
}

/// The row or column of `end` in the table of links: a vessel's own index, and `num_vessels` for the boundary of
/// every berth, which all share one row and one column.
///
/// `end` is a boundary or a vessel below `num_vessels`.
#[inline]
fn end_index(end: LinkEnd, num_vessels: usize) -> usize {
    match end {
        LinkEnd::Vessel(vessel) => {
            debug_assert!(
                vessel.index() < num_vessels,
                "called `end_index` with {vessel}, outside a table of {num_vessels} vessels"
            );
            vessel.index()
        }
        LinkEnd::Berth(_) => num_vessels,
    }
}

/// The index of the link from `from` to `to` in a table of links of `num_vessels` vessels, below
/// `(num_vessels + 1)^2`.
///
/// `from` and `to` are boundaries or vessels below `num_vessels`.
#[inline]
fn link_index(from: LinkEnd, to: LinkEnd, num_vessels: usize) -> usize {
    end_index(from, num_vessels) * (num_vessels + 1) + end_index(to, num_vessels)
}

/// Whether the link from `from` to `to` is a feature: one of its ends is a vessel. The link from the head to the
/// tail of an empty berth is none.
#[inline]
fn is_feature(from: LinkEnd, to: LinkEnd) -> bool {
    matches!(from, LinkEnd::Vessel(_)) || matches!(to, LinkEnd::Vessel(_))
}

// A penalty count grows by at most one per penalization, and a run penalizes at most once per candidate and once
// per exhausted scan, so a count, and a sum of the counts of the at most `2 * num_vessels + num_berths` features of
// one schedule, stays far below 2^53 in any run shorter than years: it is exact as an `i64` and as an `f64`.

/// `count`, a penalty count, as an `i64`.
#[inline(always)] // Called per entry of the diff of every candidate the search decides on.
#[expect(
    clippy::as_conversions,
    reason = "a penalty count stays below 2^53, so it is an exact `i64`"
)]
fn signed(count: u64) -> i64 {
    count as i64
}

/// `count`, a penalty count, as an `f64`.
#[expect(
    clippy::as_conversions,
    reason = "a penalty count stays below 2^53, so it is an exact `f64`"
)]
fn real(count: u64) -> f64 {
    count as f64
}

/// `sum`, a sum of penalty counts, as an `f64`.
#[inline(always)] // Called twice for every candidate the search decides on.
#[expect(
    clippy::as_conversions,
    reason = "a sum of the penalty counts of one schedule stays below 2^53, so it is an exact `f64`"
)]
pub(super) fn real_sum(sum: i64) -> f64 {
    sum as f64
}

/// `count` multiplied by `factor`, in `[0, 1]`, and rounded toward zero.
#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "the product of a count and a factor in [0, 1] lies in [0, count], and rounding it toward zero is the \
              decay"
)]
fn scaled(count: u64, factor: f64) -> u64 {
    (real(count) * factor) as u64
}

/// The penalty count of every feature of the schedules of one instance, zero at the start of a run.
///
/// The table of links has `(num_vessels + 1)^2` entries, the row and column `num_vessels` standing for the
/// boundaries of all berths; the table of assignments has `num_vessels * num_berths`, by vessel and then berth.
/// The memory so takes `8 * ((num_vessels + 1)^2 + num_vessels * num_berths)` bytes: 8 MB for a thousand vessels,
/// 800 MB for ten thousand and 20 GB for fifty thousand. A count is a `u64` and not a `u32`: a run that penalizes at
/// every candidate, as one triggered after a single non-improving step does, could raise one feature past 2^32
/// within minutes.
#[derive(Debug, Clone, Default)]
pub(super) struct PenaltyMemory {
    links: Vec<u64>,
    assignments: Vec<u64>,
    num_vessels: usize,
    num_berths: usize,
}

impl PenaltyMemory {
    /// Sizes the memory for `num_vessels` vessels and `num_berths` berths, with every count zero.
    pub(super) fn reset(&mut self, num_vessels: usize, num_berths: usize) {
        self.links.clear();
        self.links.resize((num_vessels + 1) * (num_vessels + 1), 0);
        self.assignments.clear();
        self.assignments.resize(num_vessels * num_berths, 0);
        self.num_vessels = num_vessels;
        self.num_berths = num_berths;
    }

    /// The index of the assignment of `vessel` to `berth`.
    ///
    /// `vessel` and `berth` are the memory's.
    #[inline]
    pub(super) fn assignment_index(&self, vessel: VesselIndex, berth: BerthIndex) -> usize {
        debug_assert!(
            vessel.index() < self.num_vessels && berth.index() < self.num_berths,
            "called `PenaltyMemory::assignment_index` with {vessel} or {berth}, outside the memory"
        );
        vessel.index() * self.num_berths + berth.index()
    }

    /// Whether the index of `feature` is below the length of its table, the condition of
    /// [`count_unchecked`](Self::count_unchecked).
    fn holds(&self, feature: Feature) -> bool {
        match feature {
            Feature::Link(index) => index < self.links.len(),
            Feature::Assignment(index) => index < self.assignments.len(),
        }
    }

    /// Whether every vessel and berth `diff` names is the memory's, the condition of
    /// [`penalty_delta_unchecked`](Self::penalty_delta_unchecked); it costs one pass over the diff.
    pub(super) fn covers(&self, diff: &ScheduleGraphDiff) -> bool {
        diff.names_only_ids_below(self.num_vessels, self.num_berths)
    }

    /// The count of the feature at `feature`.
    ///
    /// # Safety
    ///
    /// The index of `feature` must be below the length of its table.
    #[inline]
    unsafe fn count_unchecked(&self, feature: Feature) -> u64 {
        match feature {
            Feature::Link(index) => {
                debug_assert!(
                    index < self.links.len(),
                    "called `PenaltyMemory::count_unchecked` with link {index}, outside a table of {}",
                    self.links.len()
                );
                // SAFETY: the index is below the length of the table of links, the precondition of this function.
                unsafe { *self.links.get_unchecked(index) }
            }
            Feature::Assignment(index) => {
                debug_assert!(
                    index < self.assignments.len(),
                    "called `PenaltyMemory::count_unchecked` with assignment {index}, outside a table of {}",
                    self.assignments.len()
                );
                // SAFETY: the index is below the length of the table of assignments, the precondition of this
                // function.
                unsafe { *self.assignments.get_unchecked(index) }
            }
        }
    }

    /// The count of the link from `from` to `to`.
    ///
    /// # Panics
    ///
    /// Panics if `from` or `to` is a vessel outside the memory.
    #[cfg(test)]
    pub(super) fn link_penalty(&self, from: LinkEnd, to: LinkEnd) -> u64 {
        let inside = |end: LinkEnd| match end {
            LinkEnd::Vessel(vessel) => vessel.index() < self.num_vessels,
            LinkEnd::Berth(_) => true,
        };
        assert!(
            inside(from) && inside(to),
            "called `PenaltyMemory::link_penalty` with a link outside a memory of {} vessels",
            self.num_vessels
        );
        self.links[link_index(from, to, self.num_vessels)]
    }

    /// The count of the assignment of `vessel` to `berth`.
    ///
    /// # Panics
    ///
    /// Panics if `vessel` or `berth` is outside the memory.
    #[cfg(test)]
    pub(super) fn assignment_penalty(&self, vessel: VesselIndex, berth: BerthIndex) -> u64 {
        assert!(
            vessel.index() < self.num_vessels && berth.index() < self.num_berths,
            "called `PenaltyMemory::assignment_penalty` with {vessel} and {berth}, outside a memory of {} vessels \
             and {} berths",
            self.num_vessels,
            self.num_berths
        );
        self.assignments[self.assignment_index(vessel, berth)]
    }

    /// The vessels and berths the memory counts.
    #[cfg(test)]
    pub(super) fn counts(&self) -> (usize, usize) {
        (self.num_vessels, self.num_berths)
    }

    /// Sets the count of the link from `from` to `to`.
    ///
    /// # Panics
    ///
    /// Panics if `from` or `to` is a vessel outside the memory.
    #[cfg(test)]
    pub(super) fn set_link_penalty(&mut self, from: LinkEnd, to: LinkEnd, count: u64) {
        let _ = self.link_penalty(from, to);
        let index = link_index(from, to, self.num_vessels);
        self.links[index] = count;
    }

    /// Sets the count of the assignment of `vessel` to `berth`.
    ///
    /// # Panics
    ///
    /// Panics if `vessel` or `berth` is outside the memory.
    #[cfg(test)]
    pub(super) fn set_assignment_penalty(
        &mut self,
        vessel: VesselIndex,
        berth: BerthIndex,
        count: u64,
    ) {
        let _ = self.assignment_penalty(vessel, berth);
        let index = self.assignment_index(vessel, berth);
        self.assignments[index] = count;
    }

    /// The change in the sum of penalties that the moves of `diff` make: the counts of the links they made and
    /// the assignments they reached, less those of the links they broke and the assignments they left.
    ///
    /// It costs one read per entry of the diff.
    ///
    /// # Safety
    ///
    /// Every vessel and berth `diff` names must be the memory's.
    pub(super) unsafe fn penalty_delta_unchecked(&self, diff: &ScheduleGraphDiff) -> i64 {
        debug_assert!(
            self.covers(diff),
            "called `PenaltyMemory::penalty_delta_unchecked` with a diff that names an id outside the memory"
        );
        let mut delta: i64 = 0;
        for link in diff.broken_links() {
            let feature = Feature::Link(link_index(link.from, link.to, self.num_vessels));
            debug_assert!(self.holds(feature));
            // SAFETY: both ends are boundaries or vessels of the memory, the precondition of this function, so
            // the index is below `(num_vessels + 1)^2`, the length of the table of links.
            delta -= signed(unsafe { self.count_unchecked(feature) });
        }
        for link in diff.created_links() {
            let feature = Feature::Link(link_index(link.from, link.to, self.num_vessels));
            debug_assert!(self.holds(feature));
            // SAFETY: both ends are boundaries or vessels of the memory, the precondition of this function, so
            // the index is below `(num_vessels + 1)^2`, the length of the table of links.
            delta += signed(unsafe { self.count_unchecked(feature) });
        }
        for reallocation in diff.reallocations() {
            let left =
                Feature::Assignment(self.assignment_index(reallocation.vessel, reallocation.from));
            let reached =
                Feature::Assignment(self.assignment_index(reallocation.vessel, reallocation.to));
            debug_assert!(self.holds(left) && self.holds(reached));
            // SAFETY: the vessel and both berths are the memory's, the precondition of this function, so both
            // indices are below `num_vessels * num_berths`, the length of the table of assignments.
            let (left, reached) =
                unsafe { (self.count_unchecked(left), self.count_unchecked(reached)) };
            delta += signed(reached) - signed(left);
        }
        delta
    }

    /// The sum of the penalties of every feature of `graph`.
    ///
    /// # Panics
    ///
    /// Panics if `graph` does not count the vessels and berths of the memory.
    pub(super) fn penalty_sum(&self, graph: &ScheduleGraph) -> i64 {
        self.assert_counts_of(graph);
        let mut sum: i64 = 0;
        for berth in graph.berths() {
            let ends = graph.berth_topology_iter(berth);
            for (from, to) in ends.clone().zip(ends.skip(1)) {
                if is_feature(from, to) {
                    let feature = Feature::Link(link_index(from, to, self.num_vessels));
                    debug_assert!(self.holds(feature));
                    // SAFETY: both ends are boundaries or vessels of the graph, which counts the memory's
                    // vessels, as checked above.
                    sum += signed(unsafe { self.count_unchecked(feature) });
                }
            }
        }
        for vessel in graph.vessels() {
            let feature =
                Feature::Assignment(self.assignment_index(vessel, graph.vessel_berth(vessel)));
            debug_assert!(self.holds(feature));
            // SAFETY: the vessel and its berth are the graph's, which counts the memory's vessels and berths, as
            // checked above.
            sum += signed(unsafe { self.count_unchecked(feature) });
        }
        sum
    }

    /// Adds one penalty to every feature of `graph` of the largest utility `1 / (1 + penalty)`, collecting them
    /// in `winners`.
    ///
    /// # Panics
    ///
    /// Panics if `graph` does not count the vessels and berths of the memory.
    pub(super) fn penalize(&mut self, graph: &ScheduleGraph, winners: &mut Vec<Feature>) {
        self.assert_counts_of(graph);
        winners.clear();
        let mut best_utility = f64::NEG_INFINITY;
        let mut consider = |feature: Feature, penalty: u64| {
            let utility = 1.0 / (1.0 + real(penalty));
            match rank_utility(utility, best_utility) {
                UtilityRank::Better => {
                    best_utility = utility;
                    winners.clear();
                    winners.push(feature);
                }
                UtilityRank::Tie => winners.push(feature),
                UtilityRank::Worse => {}
            }
        };
        for berth in graph.berths() {
            let ends = graph.berth_topology_iter(berth);
            for (from, to) in ends.clone().zip(ends.skip(1)) {
                if is_feature(from, to) {
                    let feature = Feature::Link(link_index(from, to, self.num_vessels));
                    debug_assert!(self.holds(feature));
                    // SAFETY: both ends are boundaries or vessels of the graph, which counts the memory's
                    // vessels, as checked above.
                    consider(feature, unsafe { self.count_unchecked(feature) });
                }
            }
            for vessel in graph.vessel_sequence_iter(berth) {
                let feature = Feature::Assignment(self.assignment_index(vessel, berth));
                debug_assert!(self.holds(feature));
                // SAFETY: the vessel and the berth are the graph's, which counts the memory's vessels and berths,
                // as checked above.
                consider(feature, unsafe { self.count_unchecked(feature) });
            }
        }
        for &feature in winners.iter() {
            let count = match feature {
                Feature::Link(index) => &mut self.links[index],
                Feature::Assignment(index) => &mut self.assignments[index],
            };
            *count = count.saturating_add(1);
        }
    }

    /// Multiplies every count by `factor`, rounding toward zero.
    ///
    /// `factor` is in `[0, 1]`.
    pub(super) fn decay(&mut self, factor: f64) {
        debug_assert!(
            (0.0..=1.0).contains(&factor),
            "called `PenaltyMemory::decay` with factor {factor}, outside [0, 1]"
        );
        for count in self.links.iter_mut().chain(self.assignments.iter_mut()) {
            *count = scaled(*count, factor);
        }
    }

    /// Panics unless `graph` counts the vessels and berths of the memory, which every unchecked read through the
    /// features of `graph` rests on.
    fn assert_counts_of(&self, graph: &ScheduleGraph) {
        assert!(
            graph.num_vessels() == self.num_vessels && graph.num_berths() == self.num_berths,
            "the graph of {} vessels and {} berths should have the counts of the penalty memory, {} and {}",
            graph.num_vessels(),
            graph.num_berths(),
            self.num_vessels,
            self.num_berths
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schedule::Mutator;

    fn vessel(index: usize) -> VesselIndex {
        VesselIndex::new(index)
    }

    /// The memory of `num_vessels` vessels and `num_berths` berths, every count zero.
    fn memory(num_vessels: usize, num_berths: usize) -> PenaltyMemory {
        let mut memory = PenaltyMemory::default();
        memory.reset(num_vessels, num_berths);
        memory
    }

    fn berth(index: usize) -> BerthIndex {
        BerthIndex::new(index)
    }

    /// V0 and V1 on B0 in that order, V2 on B1.
    fn three_vessels_on_two_berths() -> ScheduleGraph {
        ScheduleGraph::from_slices(&[berth(0), berth(0), berth(1)], &[0_i64, 1, 0], 2)
    }

    #[test]
    fn test_rank_utility_ties_within_the_relative_tolerance_and_ranks_the_rest() {
        assert_eq!(rank_utility(0.5, f64::NEG_INFINITY), UtilityRank::Better);
        assert_eq!(rank_utility(1e8 + 1e-6, 1e8), UtilityRank::Tie);
        assert_eq!(rank_utility(0.5, 0.5 + 1e-13), UtilityRank::Tie);
        assert_eq!(rank_utility(1.0, 0.5), UtilityRank::Better);
        assert_eq!(rank_utility(0.5, 1.0), UtilityRank::Worse);
    }

    #[test]
    fn test_link_index_gives_every_boundary_one_row_and_column() {
        let boundary = |index| LinkEnd::Berth(berth(index));
        let first = LinkEnd::Vessel(vessel(1));
        assert_eq!(link_index(first, LinkEnd::Vessel(vessel(2)), 4), 7);
        assert_eq!(
            link_index(boundary(0), first, 4),
            link_index(boundary(3), first, 4)
        );
        assert_eq!(link_index(first, boundary(0), 4), 9);
        assert_eq!(link_index(boundary(0), boundary(1), 4), 24);
    }

    #[test]
    fn test_link_index_stays_inside_the_table() {
        let num_vessels = 5;
        let ends: Vec<LinkEnd> = (0..num_vessels)
            .map(|v| LinkEnd::Vessel(vessel(v)))
            .chain([LinkEnd::Berth(berth(7))])
            .collect();
        for &from in &ends {
            for &to in &ends {
                assert!(link_index(from, to, num_vessels) < (num_vessels + 1) * (num_vessels + 1));
            }
        }
    }

    #[test]
    fn test_penalty_memory_starts_at_zero_and_resets_to_new_counts() {
        let mut memory = memory(3, 2);
        assert_eq!(memory.links.len(), 16);
        assert_eq!(memory.assignments.len(), 6);
        memory.links[5] = 3;
        memory.reset(4, 1);
        assert_eq!(memory.links, vec![0; 25]);
        assert_eq!(memory.assignments, vec![0; 4]);
    }

    #[test]
    fn test_penalty_memory_reads_the_counts_of_links_and_assignments() {
        let mut memory = memory(3, 2);
        let (from, to) = (LinkEnd::Vessel(vessel(0)), LinkEnd::Vessel(vessel(2)));
        memory.links[link_index(from, to, 3)] = 7;
        let index = memory.assignment_index(vessel(2), berth(1));
        memory.assignments[index] = 4;
        assert_eq!(memory.link_penalty(from, to), 7);
        assert_eq!(memory.link_penalty(to, from), 0);
        assert_eq!(memory.assignment_penalty(vessel(2), berth(1)), 4);
        assert_eq!(memory.assignment_penalty(vessel(1), berth(1)), 0);
    }

    // A berth beyond the memory must be refused, not read as another vessel's row.
    #[test]
    #[should_panic(
        expected = "called `PenaltyMemory::assignment_penalty` with Vessel 0 and Berth 3, outside a memory of 2 \
                    vessels and 2 berths"
    )]
    fn test_penalty_memory_assignment_penalty_refuses_a_berth_outside() {
        let _ = memory(2, 2).assignment_penalty(vessel(0), berth(3));
    }

    // A vessel beyond the memory must be refused, not read as the boundary of a berth.
    #[test]
    #[should_panic(
        expected = "called `PenaltyMemory::link_penalty` with a link outside a memory of 2 vessels"
    )]
    fn test_penalty_memory_link_penalty_refuses_a_vessel_outside() {
        let _ = memory(2, 2).link_penalty(LinkEnd::Vessel(vessel(2)), LinkEnd::Vessel(vessel(0)));
    }

    #[test]
    #[should_panic(
        expected = "the graph of 3 vessels and 2 berths should have the counts of the penalty memory"
    )]
    fn test_penalty_sum_refuses_a_graph_of_other_counts() {
        let _ = memory(2, 2).penalty_sum(&three_vessels_on_two_berths());
    }

    #[test]
    fn test_penalty_delta_counts_made_and_reached_less_broken_and_left() {
        let graph = three_vessels_on_two_berths();
        let mut memory = memory(3, 2);
        for count in memory.links.iter_mut() {
            *count = 1;
        }
        let index = memory.assignment_index(vessel(1), berth(0));
        memory.assignments[index] = 5;
        let index = memory.assignment_index(vessel(1), berth(1));
        memory.assignments[index] = 2;
        let mut mutator = Mutator::new(graph);
        // V1 moves from after V0 on B0 to after V2 on B1: it breaks V0-V1 and V1-tail and V2-tail, and makes
        // V0-tail, V2-V1 and V1-tail: as many links, each of count one.
        mutator.relocate_after(vessel(1), vessel(2));
        // SAFETY: the diff names vessels and berths of a graph of the memory's counts.
        let delta = unsafe { memory.penalty_delta_unchecked(mutator.diff()) };
        assert_eq!(delta, 2 - 5);
    }

    #[test]
    fn test_penalty_sum_counts_every_link_with_a_vessel_and_every_assignment() {
        let graph = three_vessels_on_two_berths();
        let mut memory = memory(3, 2);
        for count in memory.links.iter_mut().chain(memory.assignments.iter_mut()) {
            *count = 1;
        }
        // B0: head-V0, V0-V1, V1-tail; B1: head-V2, V2-tail; and three assignments.
        assert_eq!(memory.penalty_sum(&graph), 5 + 3);
    }

    #[test]
    fn test_penalize_raises_every_feature_of_the_largest_utility() {
        let graph = three_vessels_on_two_berths();
        let mut memory = memory(3, 2);
        let mut winners = Vec::new();
        memory.penalize(&graph, &mut winners);
        // Every feature is unpenalized: all eight tie and all are raised.
        assert_eq!(winners.len(), 8);
        assert_eq!(memory.penalty_sum(&graph), 8);
        // Now the assignment of V2 alone is lower than the rest.
        let index = memory.assignment_index(vessel(2), berth(1));
        memory.assignments[index] = 0;
        memory.penalize(&graph, &mut winners);
        assert_eq!(
            winners,
            [Feature::Assignment(
                memory.assignment_index(vessel(2), berth(1))
            )]
        );
        assert_eq!(memory.assignment_penalty(vessel(2), berth(1)), 1);
    }

    #[test]
    fn test_decay_rounds_every_count_toward_zero() {
        let mut memory = memory(1, 1);
        memory.links[0] = 10;
        memory.links[1] = 3;
        memory.assignments[0] = 1;
        memory.decay(0.5);
        assert_eq!(memory.links[..2], [5, 1]);
        assert_eq!(memory.assignments[0], 0);
    }

    #[test]
    fn test_decay_by_one_keeps_a_count_below_two_to_the_fifty_third() {
        let mut memory = memory(1, 1);
        memory.links[0] = (1_u64 << 53) - 1;
        memory.decay(1.0);
        assert_eq!(memory.links[0], (1_u64 << 53) - 1);
    }
}
