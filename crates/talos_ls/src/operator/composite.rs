// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The operator of a run: several neighbourhoods, with a rule that chooses which one proposes the next move.
//!
//! A scan of the neighbourhood of the accepted schedule asks one neighbourhood for moves until it is exhausted and
//! then turns to the next one the rule names; the scan is exhausted once every neighbourhood is. The rules differ
//! in where a scan starts and in the order of the neighbourhoods:
//!
//! - round robin starts every scan at the first neighbourhood and keeps the order they were given in;
//! - random selection starts every scan at a neighbourhood drawn at random and goes on in circular order;
//! - the multi-armed bandit ranks the neighbourhoods before every scan by the UCB1 score of the improvement their
//!   accepted moves brought, and goes through them in the order of the ranking.

use super::neighborhood::{AcceptedMove, LocalSearchOperator, Neighborhood, NeighborhoodOperator};
use crate::schedule::{Mutator, ScheduleGraph};
use rand::RngExt;
use rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use std::cmp::Ordering;
use talos_core::SolverNumeric;
use talos_model::Model;

/// The rule that chooses which neighbourhood proposes the next move of a scan.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum NeighborhoodSelection {
    /// Every scan starts at the first neighbourhood and takes them in the order they were given in.
    #[default]
    RoundRobin,
    /// Every scan starts at a neighbourhood drawn uniformly from the run's seeded generator and takes them in
    /// circular order from there.
    Random,
    /// Every scan takes the neighbourhoods in the order of their UCB1 scores.
    MultiArmedBandit(MultiArmedBandit),
}

/// The settings of the multi-armed bandit, which ranks the neighbourhoods by the UCB1 score of the improvement of
/// the objective each brought.
///
/// The score of a neighbourhood is its average improvement, divided by the largest improvement seen, plus the
/// bonus `exploration * sqrt(2 ln(1 + samples) / samples of the neighbourhood)`; a neighbourhood not yet sampled
/// scores infinity. The memory is the weight of the newest improvement in a neighbourhood's average, in `(0, 1]`:
/// the average moves by this fraction of the distance to the newest improvement. The exploration weight is finite
/// and not negative; the standard UCB1 constant is the square root of two.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MultiArmedBandit {
    memory: f64,
    exploration: f64,
}

impl MultiArmedBandit {
    /// The bandit of `memory` and `exploration`.
    ///
    /// # Panics
    ///
    /// Panics if `memory` is not in `(0, 1]` or `exploration` is not finite and at least zero.
    pub fn new(memory: f64, exploration: f64) -> Self {
        match Self::try_new(memory, exploration) {
            Some(bandit) => bandit,
            None => panic!(
                "called `MultiArmedBandit::new` with memory {memory} and exploration {exploration}, not a memory in \
                 (0, 1] and a finite exploration of at least zero"
            ),
        }
    }

    /// The bandit of [`new`](Self::new), or `None` if `memory` is not in `(0, 1]` or `exploration` is not finite
    /// and at least zero.
    pub fn try_new(memory: f64, exploration: f64) -> Option<Self> {
        let valid = memory > 0.0 && memory <= 1.0 && exploration.is_finite() && exploration >= 0.0;
        valid.then_some(Self {
            memory,
            exploration,
        })
    }
}

impl Default for MultiArmedBandit {
    /// A memory of 0.2, which follows a change in a neighbourhood's effect within a few scans, and the standard
    /// UCB1 exploration weight, the square root of two.
    fn default() -> Self {
        Self {
            memory: 0.2,
            exploration: std::f64::consts::SQRT_2,
        }
    }
}

/// The neighbourhoods of a run with the rule that chooses among them.
///
/// Every index a rule holds into the scans (the current scan, the ranking of the bandit) is below the number of
/// scans, which is at least one, and the tables of the bandit have one entry per scan.
#[derive(Debug, Clone)]
pub struct CompoundOperator {
    scans: Vec<NeighborhoodOperator>,
    rule: SelectionRule,
}

/// The state of the rule of a compound operator.
#[derive(Debug, Clone)]
enum SelectionRule {
    RoundRobin(RoundRobin),
    Random(RandomSelection),
    MultiArmedBandit(Bandit),
}

impl CompoundOperator {
    /// The operator over `neighborhoods`, in that order, chosen among by `selection`; a random choice draws from a
    /// generator seeded with `seed`.
    ///
    /// # Panics
    ///
    /// Panics if `neighborhoods` is empty.
    pub fn new(
        neighborhoods: &[Neighborhood],
        selection: NeighborhoodSelection,
        seed: u64,
    ) -> Self {
        assert!(
            !neighborhoods.is_empty(),
            "called `CompoundOperator::new` with no neighborhood"
        );
        let scans: Vec<NeighborhoodOperator> = neighborhoods
            .iter()
            .map(|&neighborhood| NeighborhoodOperator::new(neighborhood))
            .collect();
        let rule = match selection {
            NeighborhoodSelection::RoundRobin => {
                SelectionRule::RoundRobin(RoundRobin { current: 0 })
            }
            NeighborhoodSelection::Random => SelectionRule::Random(RandomSelection {
                generator: Xoshiro256PlusPlus::seed_from_u64(seed),
                current: 0,
            }),
            NeighborhoodSelection::MultiArmedBandit(settings) => {
                SelectionRule::MultiArmedBandit(Bandit::new(scans.len(), settings))
            }
        };
        Self { scans, rule }
    }
}

impl<T> LocalSearchOperator<T> for CompoundOperator
where
    T: SolverNumeric,
{
    fn prepare(&mut self, accepted_objective: T, graph: &ScheduleGraph) {
        for scan in &mut self.scans {
            scan.prepare(graph);
        }
        match &mut self.rule {
            SelectionRule::RoundRobin(rule) => rule.current = 0,
            SelectionRule::Random(rule) => {
                rule.current = rule.generator.random_range(0..self.scans.len())
            }
            SelectionRule::MultiArmedBandit(rule) => {
                rule.prepare(accepted_objective.to_f64().unwrap_or(0.0))
            }
        }
    }

    unsafe fn next_neighbor_unchecked(&mut self, model: &Model<T>, mutator: &mut Mutator) -> bool {
        let scans = self.scans.as_mut_slice();
        match &mut self.rule {
            SelectionRule::RoundRobin(rule) => {
                debug_assert!(rule.current <= scans.len());
                // SAFETY: the current scan is at most the number of scans, by the invariant of `CompoundOperator`,
                // and the graph is as every scan needs it, the precondition of this function.
                unsafe { rule.next_neighbor_unchecked(scans, model, mutator) }
            }
            SelectionRule::Random(rule) => {
                debug_assert!(rule.current < scans.len());
                // SAFETY: the current scan is below the number of scans, by the invariant of `CompoundOperator`,
                // and the graph is as every scan needs it, the precondition of this function.
                unsafe { rule.next_neighbor_unchecked(scans, model, mutator) }
            }
            SelectionRule::MultiArmedBandit(rule) => {
                debug_assert!(
                    rule.active_rank < scans.len() && rule.ranking.iter().all(|&n| n < scans.len())
                );
                // SAFETY: the ranking and the active rank are below the number of scans, by the invariant of
                // `CompoundOperator`, and the graph is as every scan needs it, the precondition of this function.
                unsafe { rule.next_neighbor_unchecked(scans, model, mutator) }
            }
        }
    }

    fn on_kept_aside(&mut self) {
        if let SelectionRule::MultiArmedBandit(rule) = &mut self.rule {
            rule.on_kept_aside();
        }
    }

    fn on_accepted(&mut self, accepted: AcceptedMove, accepted_objective: T) {
        if let SelectionRule::MultiArmedBandit(rule) = &mut self.rule {
            rule.on_accepted(accepted, accepted_objective.to_f64().unwrap_or(0.0));
        }
    }
}

/// Round robin: the scan asks the neighbourhoods in order, each until it is exhausted.
#[derive(Debug, Clone)]
struct RoundRobin {
    /// The neighbourhood that proposes the next move, or the number of neighbourhoods once all are exhausted.
    current: usize,
}

impl RoundRobin {
    /// The next move of the scan, from the current neighbourhood or the next one not yet exhausted.
    ///
    /// # Safety
    ///
    /// `current` must be at most the length of `scans`, and the graph of `mutator` as every scan needs it.
    unsafe fn next_neighbor_unchecked<T>(
        &mut self,
        scans: &mut [NeighborhoodOperator],
        model: &Model<T>,
        mutator: &mut Mutator,
    ) -> bool
    where
        T: SolverNumeric,
    {
        while self.current < scans.len() {
            debug_assert!(self.current < scans.len());
            // SAFETY: `current` is below the length of `scans`, as the loop condition found.
            let scan = unsafe { scans.get_unchecked_mut(self.current) };
            // SAFETY: the graph is as every scan needs it, the precondition of this function.
            if unsafe { scan.next_neighbor_unchecked(model, mutator) } {
                return true;
            }
            self.current += 1;
        }
        false
    }
}

/// Random selection: the scan starts at a neighbourhood drawn at random and asks the neighbourhoods in circular
/// order from there, each until it is exhausted.
#[derive(Debug, Clone)]
struct RandomSelection {
    /// The generator the neighbourhood that starts a scan is drawn from.
    generator: Xoshiro256PlusPlus,
    /// The neighbourhood that proposed the last move, where the next one is asked first.
    current: usize,
}

impl RandomSelection {
    /// The next move of the scan, from the current neighbourhood or the next one in circular order not yet
    /// exhausted.
    ///
    /// # Safety
    ///
    /// `current` must be below the length of `scans`, and the graph of `mutator` as every scan needs it.
    unsafe fn next_neighbor_unchecked<T>(
        &mut self,
        scans: &mut [NeighborhoodOperator],
        model: &Model<T>,
        mutator: &mut Mutator,
    ) -> bool
    where
        T: SolverNumeric,
    {
        let start = self.current;
        let mut index = start;
        for _ in 0..scans.len() {
            debug_assert!(index < scans.len());
            // SAFETY: `index` starts at `current`, below the length of `scans` by the precondition of this
            // function, and steps in circular order below it.
            let scan = unsafe { scans.get_unchecked_mut(index) };
            // SAFETY: the graph is as every scan needs it, the precondition of this function.
            if unsafe { scan.next_neighbor_unchecked(model, mutator) } {
                self.current = index;
                return true;
            }
            index = (index + 1) % scans.len();
        }
        false
    }
}

/// The multi-armed bandit: it credits the neighbourhood that proposed an accepted move with the improvement of the
/// objective over the scan, ranks the neighbourhoods by their UCB1 scores before every scan, and the scan asks them
/// in the order of the ranking.
#[derive(Debug, Clone)]
struct Bandit {
    /// The memory and the exploration weight of the scores.
    settings: MultiArmedBandit,
    /// The moves each neighbourhood proposed, per neighbourhood.
    samples: Vec<u64>,
    /// The moves all neighbourhoods proposed.
    total_samples: u64,
    /// The average improvement each neighbourhood brought, per neighbourhood.
    averages: Vec<f64>,
    /// The largest improvement seen, at least one, by which the averages are scaled into `[0, 1]`.
    largest_improvement: f64,
    /// The neighbourhoods, best score first.
    ranking: Vec<usize>,
    /// The place in the ranking of the neighbourhood asked first for the next move, and so of the one that made
    /// the last move.
    active_rank: usize,
    /// The neighbourhood that made the move kept aside in the current scan, or `None` while none is.
    kept_aside: Option<usize>,
    /// The objective of the accepted schedule when the current scan started, zero before the first scan.
    scan_objective: f64,
}

impl Bandit {
    /// The bandit over `count` neighbourhoods, none of them sampled.
    fn new(count: usize, settings: MultiArmedBandit) -> Self {
        Self {
            settings,
            samples: vec![0; count],
            total_samples: 0,
            averages: vec![0.0; count],
            largest_improvement: 1.0,
            ranking: (0..count).collect(),
            active_rank: 0,
            kept_aside: None,
            scan_objective: 0.0,
        }
    }

    /// Starts a scan from the accepted schedule of `accepted_objective`: ranks the neighbourhoods anew and starts
    /// at the first rank.
    fn prepare(&mut self, accepted_objective: f64) {
        self.scan_objective = accepted_objective;
        self.kept_aside = None;
        self.rank();
        self.active_rank = 0;
    }

    /// Takes note that the neighbourhood at the active rank made the move kept aside.
    fn on_kept_aside(&mut self) {
        self.kept_aside = Some(self.ranking[self.active_rank]);
    }

    /// Credits the neighbourhood that made `accepted` with the improvement from the objective at the start of the
    /// scan to `accepted_objective`, none where it is no lower.
    fn on_accepted(&mut self, accepted: AcceptedMove, accepted_objective: f64) {
        let credited = match accepted {
            AcceptedMove::Last => Some(self.ranking[self.active_rank]),
            AcceptedMove::KeptAside => self.kept_aside,
        };
        debug_assert!(
            credited.is_some(),
            "called `Bandit::on_accepted` for a kept-aside move with none kept aside"
        );
        if let Some(neighborhood) = credited {
            self.learn(
                neighborhood,
                (self.scan_objective - accepted_objective).max(0.0),
            );
        }
    }

    /// Moves the average of `neighborhood` by the memory's fraction toward `improvement`, and raises the largest
    /// improvement to it.
    fn learn(&mut self, neighborhood: usize, improvement: f64) {
        if improvement > self.largest_improvement {
            self.largest_improvement = improvement;
        }
        let average = &mut self.averages[neighborhood];
        *average += self.settings.memory * (improvement - *average);
    }

    /// The UCB1 score of `neighborhood`: infinite while it has no sample.
    #[expect(
        clippy::as_conversions,
        reason = "counts of moves stay far below 2^53, where every count is an exact `f64`"
    )]
    fn score(&self, neighborhood: usize) -> f64 {
        let samples = self.samples[neighborhood] as f64;
        if samples == 0.0 {
            return f64::INFINITY;
        }
        let total = self.total_samples as f64;
        let exploitation = self.averages[neighborhood] / self.largest_improvement;
        let bonus = self.settings.exploration * ((2.0 * (1.0 + total).ln()) / samples).sqrt();
        exploitation + bonus
    }

    /// Orders the ranking by score, the best first, and equal scores by neighbourhood.
    fn rank(&mut self) {
        let mut ranking = std::mem::take(&mut self.ranking);
        ranking.sort_by(|&a, &b| {
            self.score(b)
                .partial_cmp(&self.score(a))
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.cmp(&b))
        });
        self.ranking = ranking;
    }

    /// The next move of the scan, from the neighbourhood at the active rank or the next rank in circular order
    /// whose neighbourhood is not yet exhausted; counts the move as a sample of its neighbourhood.
    ///
    /// # Safety
    ///
    /// The ranking must hold indices below the length of `scans` and the active rank must be below it, and the
    /// graph of `mutator` must be as every scan needs it.
    unsafe fn next_neighbor_unchecked<T>(
        &mut self,
        scans: &mut [NeighborhoodOperator],
        model: &Model<T>,
        mutator: &mut Mutator,
    ) -> bool
    where
        T: SolverNumeric,
    {
        let start = self.active_rank;
        for _ in 0..scans.len() {
            debug_assert!(
                self.active_rank < self.ranking.len() && self.ranking.len() == scans.len()
            );
            // SAFETY: the active rank is below the length of the ranking, which has one entry per scan, the
            // precondition of this function.
            let neighborhood = unsafe { *self.ranking.get_unchecked(self.active_rank) };
            debug_assert!(neighborhood < scans.len() && neighborhood < self.samples.len());
            // SAFETY: every index of the ranking is below the length of `scans`, the precondition of this
            // function.
            let scan = unsafe { scans.get_unchecked_mut(neighborhood) };
            // SAFETY: the graph is as every scan needs it, the precondition of this function.
            if unsafe { scan.next_neighbor_unchecked(model, mutator) } {
                self.total_samples += 1;
                // SAFETY: the samples have one entry per scan, by the invariant of `CompoundOperator`.
                unsafe { *self.samples.get_unchecked_mut(neighborhood) += 1 };
                return true;
            }
            self.active_rank = (self.active_rank + 1) % scans.len();
            if self.active_rank == start {
                return false;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "called `CompoundOperator::new` with no neighborhood")]
    fn test_compound_operator_refuses_no_neighborhood() {
        let _ = CompoundOperator::new(&[], NeighborhoodSelection::RoundRobin, 0);
    }

    #[test]
    fn test_multi_armed_bandit_refuses_a_memory_outside_its_range_and_an_exploration_that_is_not_finite()
     {
        assert_eq!(
            MultiArmedBandit::try_new(1.0, 0.0),
            Some(MultiArmedBandit::new(1.0, 0.0))
        );
        for (memory, exploration) in [
            (0.0, 1.0),
            (1.5, 1.0),
            (f64::NAN, 1.0),
            (0.5, -1.0),
            (0.5, f64::INFINITY),
            (0.5, f64::NAN),
        ] {
            assert_eq!(MultiArmedBandit::try_new(memory, exploration), None);
        }
    }

    #[test]
    #[should_panic(expected = "called `MultiArmedBandit::new` with memory 0 and exploration 1,")]
    fn test_multi_armed_bandit_new_refuses_a_memory_of_zero() {
        let _ = MultiArmedBandit::new(0.0, 1.0);
    }

    #[test]
    fn test_bandit_scores_an_unsampled_neighborhood_infinite_and_ranks_it_first() {
        let mut bandit = Bandit::new(3, MultiArmedBandit::default());
        bandit.samples = vec![4, 0, 2];
        bandit.total_samples = 6;
        bandit.averages = vec![0.5, 0.0, 0.25];
        assert_eq!(bandit.score(1), f64::INFINITY);
        bandit.rank();
        assert_eq!(bandit.ranking[0], 1);
    }

    #[test]
    fn test_bandit_score_is_the_scaled_average_plus_the_bonus() {
        let mut bandit = Bandit::new(2, MultiArmedBandit::new(0.5, 1.0));
        bandit.samples = vec![2, 2];
        bandit.total_samples = 4;
        bandit.averages = vec![3.0, 1.0];
        bandit.largest_improvement = 4.0;
        let bonus = ((2.0 * 5.0_f64.ln()) / 2.0).sqrt();
        assert_eq!(bandit.score(0), 0.75 + bonus);
        assert_eq!(bandit.score(1), 0.25 + bonus);
        bandit.rank();
        assert_eq!(bandit.ranking, [0, 1]);
    }

    // A scan exhausted without an accepted move improved nothing, and no neighbourhood proposed a move that was
    // taken: the next scan must leave every average as it was.
    #[test]
    fn test_bandit_credits_nothing_for_a_scan_without_an_accepted_move() {
        let mut bandit = Bandit::new(2, MultiArmedBandit::new(0.5, 1.0));
        bandit.prepare(100.0);
        bandit.averages = vec![4.0, 4.0];
        bandit.prepare(100.0);
        assert_eq!(bandit.averages, [4.0, 4.0]);
    }

    #[test]
    fn test_bandit_credits_the_improvement_of_the_scan_to_the_neighborhood_of_the_accepted_move() {
        let mut bandit = Bandit::new(2, MultiArmedBandit::new(0.5, 1.0));
        bandit.prepare(100.0);
        bandit.active_rank = 1;
        let credited = bandit.ranking[1];
        bandit.on_accepted(AcceptedMove::Last, 90.0);
        assert_eq!(bandit.averages[credited], 5.0);
        assert_eq!(bandit.averages[bandit.ranking[0]], 0.0);
        assert_eq!(bandit.largest_improvement, 10.0);
        // A worse objective is no improvement, and only decays the average.
        bandit.prepare(90.0);
        bandit.active_rank = bandit
            .ranking
            .iter()
            .position(|&n| n == credited)
            .expect("ranked");
        bandit.on_accepted(AcceptedMove::Last, 95.0);
        assert_eq!(bandit.averages[credited], 2.5);
    }

    #[test]
    fn test_bandit_credits_a_kept_aside_move_to_the_neighborhood_that_made_it() {
        let mut bandit = Bandit::new(2, MultiArmedBandit::new(0.5, 1.0));
        bandit.prepare(100.0);
        let maker = bandit.ranking[0];
        bandit.on_kept_aside();
        bandit.active_rank = 1;
        bandit.on_accepted(AcceptedMove::KeptAside, 80.0);
        assert_eq!(bandit.averages[maker], 10.0);
        assert_eq!(bandit.averages[bandit.ranking[1]], 0.0);
    }
}
