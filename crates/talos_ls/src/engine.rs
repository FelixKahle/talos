// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The engine: the states and records of a run, and the loop that draws candidates, decodes them and decides
//! their fate.

use crate::{
    decoding::{ScheduleState, decode_full_unchecked, decode_unchecked},
    limits::{Clock, LimitWatch, SearchLimits, StepCheck},
    metaheuristic::{
        AcceptanceOutcome, AcceptanceRule, GreedyDescentRule, GuidedLocalSearchRule, Metaheuristic,
        NeighborhoodExhaustionOutcome,
    },
    monitor::{LocalSearchMonitor, SearchProgress},
    objective::{CompletionCost, Objective, TurnaroundCost, VesselCost},
    operator::{AcceptedMove, CompoundOperator, LocalSearchOperator},
    outcome::LocalSearchOutcome,
    parameters::{LocalSearchError, LocalSearchParameters, check_initial_solution, check_run},
    schedule::{Mutator, ScheduleGraph, ScheduleGraphDiff},
    statistics::LocalSearchStatistics,
    termination::{SearchCommand, TerminationReason},
};
use rand_core::{Rng, SeedableRng};
use rand_xoshiro::SplitMix64;
use std::marker::PhantomData;
use talos_core::SolverNumeric;
use talos_model::{Model, Solution, SolutionView};

/// The local-search engine: it improves a schedule of a model by moves from a set of neighbourhoods, under a
/// metaheuristic, within limits.
///
/// An engine owns the states and records a run works in, and keeps their memory from one run to the next. A run is
/// reproducible from its model, its initial schedule, its parameters and its monitor, unless a deadline ends it;
/// two engines share nothing, so runs on different engines may go on in parallel.
///
/// The states of an engine count the vessels and berths of the model of its last run, and so do its graphs, once a
/// run has loaded it.
#[derive(Debug, Clone)]
pub struct Engine<T> {
    /// The schedule the run moves from.
    accepted: ScheduleState<T>,
    /// The candidate a move made: decoded on the berths the move touched, and as `accepted` elsewhere.
    candidate: ScheduleState<T>,
    /// The order of the candidate, with the records of the moves that made it from the accepted schedule.
    mutator: Mutator,
    /// The best candidate of the current scan, kept aside by a best-improvement descent.
    buffered: ScheduleState<T>,
    /// The order of the buffered candidate.
    buffered_graph: ScheduleGraph,
    /// The moves that made the buffered candidate from the accepted schedule.
    buffered_diff: ScheduleGraphDiff,
    /// The best schedule found so far.
    best: ScheduleState<T>,
}

impl<T> Default for Engine<T>
where
    T: SolverNumeric,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Engine<T>
where
    T: SolverNumeric,
{
    /// An engine that has not yet run; its first run sizes it for its model.
    pub fn new() -> Self {
        let empty_state = || ScheduleState::new(Vec::new(), Vec::new(), Vec::new(), T::ZERO);
        let empty_graph = ScheduleGraph::from_slices::<T>(&[], &[], 0);
        Self {
            accepted: empty_state(),
            candidate: empty_state(),
            mutator: Mutator::new(empty_graph.clone()),
            buffered: empty_state(),
            buffered_graph: empty_graph,
            buffered_diff: ScheduleGraphDiff::new(),
            best: empty_state(),
        }
    }

    /// Improves `initial`, a schedule of `model`, under `parameters`, and tells `monitor` of the coarse events of
    /// the search.
    ///
    /// The run keeps the order of the vessels on each berth of `initial`, by start time, and decodes it: each
    /// vessel is served as early as it fits; the objective of `initial` is not read. It then scans the
    /// neighbourhood of the accepted schedule, decodes each candidate on the berths its move touched, and lets
    /// the metaheuristic accept, keep aside or reject it, until the metaheuristic, a limit or the monitor ends
    /// it. The outcome holds the best schedule found, which is never worse than the initial one.
    ///
    /// The checks at the door cost one pass over the vessels. A run allocates a table of a vessel per vessel to
    /// order the initial schedule, its operator, the memory of its metaheuristic, its outcome, and its states,
    /// graphs and records where the model is larger than any the engine has run on. The memory of a guided local
    /// search is quadratic in the vessels, 8 MB for a thousand and 20 GB for fifty thousand, as
    /// [`GuidedLocalSearch`](crate::GuidedLocalSearch) details. The steps of the search reuse the memory of the
    /// engine; the records of the moves of a candidate grow to the most any candidate needed and then keep it.
    ///
    /// # Errors
    ///
    /// [`LocalSearchError::NoNeighborhood`] if `parameters` name no neighbourhood;
    /// [`LocalSearchError::VesselCountMismatch`], [`LocalSearchError::BerthOutOfBounds`] or
    /// [`LocalSearchError::IncompatibleBerth`] if `initial` is not a schedule of the vessels and berths of `model`;
    /// [`LocalSearchError::InfeasibleInitialSolution`] if the order of `initial` decodes to no feasible schedule.
    pub fn run<M>(
        &mut self,
        model: &Model<T>,
        initial: SolutionView<'_, T>,
        parameters: &LocalSearchParameters,
        monitor: &mut M,
    ) -> Result<LocalSearchOutcome<T>, LocalSearchError>
    where
        M: LocalSearchMonitor<T>,
    {
        check_run(model, initial, parameters)?;
        match parameters.objective {
            Objective::WeightedTurnaroundTime => self.load::<TurnaroundCost>(model, initial)?,
            Objective::WeightedCompletionTime => self.load::<CompletionCost>(model, initial)?,
        }
        let seeds = RunSeeds::new(parameters.seed);
        let mut operator = CompoundOperator::new(
            &parameters.neighborhoods,
            parameters.selection,
            seeds.selection,
        );
        let outcome = match (parameters.objective, &parameters.metaheuristic) {
            (Objective::WeightedTurnaroundTime, Metaheuristic::GreedyDescent(settings)) => {
                let mut rule = GreedyDescentRule::new(settings, seeds.tie_breaking);
                self.search::<TurnaroundCost, _, _, _>(
                    model,
                    &mut rule,
                    &mut operator,
                    parameters.limits,
                    monitor,
                )
            }
            (Objective::WeightedTurnaroundTime, Metaheuristic::GuidedLocalSearch(settings)) => {
                let mut rule = GuidedLocalSearchRule::new(settings);
                self.search::<TurnaroundCost, _, _, _>(
                    model,
                    &mut rule,
                    &mut operator,
                    parameters.limits,
                    monitor,
                )
            }
            (Objective::WeightedCompletionTime, Metaheuristic::GreedyDescent(settings)) => {
                let mut rule = GreedyDescentRule::new(settings, seeds.tie_breaking);
                self.search::<CompletionCost, _, _, _>(
                    model,
                    &mut rule,
                    &mut operator,
                    parameters.limits,
                    monitor,
                )
            }
            (Objective::WeightedCompletionTime, Metaheuristic::GuidedLocalSearch(settings)) => {
                let mut rule = GuidedLocalSearchRule::new(settings);
                self.search::<CompletionCost, _, _, _>(
                    model,
                    &mut rule,
                    &mut operator,
                    parameters.limits,
                    monitor,
                )
            }
        };
        Ok(outcome)
    }

    /// Sizes the engine for `model`, builds its graphs from `initial` and decodes it with the cost `V` into every
    /// state but the buffered one.
    ///
    /// `initial` is a schedule of the vessels and berths of `model`, as [`check_initial_solution`] accepts it.
    ///
    /// # Errors
    ///
    /// [`LocalSearchError::InfeasibleInitialSolution`] if the order of `initial` decodes to no feasible schedule.
    fn load<V>(
        &mut self,
        model: &Model<T>,
        initial: SolutionView<'_, T>,
    ) -> Result<(), LocalSearchError>
    where
        V: VesselCost<T>,
    {
        debug_assert_eq!(check_initial_solution(model, initial), Ok(()));
        let (num_vessels, num_berths) = (model.num_vessels(), model.num_berths());
        for state in [
            &mut self.accepted,
            &mut self.candidate,
            &mut self.buffered,
            &mut self.best,
        ] {
            state.resize(num_vessels, num_berths);
        }
        self.mutator.overwrite_from_slices(
            &initial.assigned_berths().raw,
            &initial.start_times().raw,
            num_berths,
        );
        self.buffered_graph
            .overwrite_from_graph(self.mutator.graph());
        self.assert_loaded_for(model);
        // SAFETY: the graph, the candidate and the model count the same vessels and berths, as checked above.
        unsafe { decode_full_unchecked::<T, V>(self.mutator.graph(), &mut self.candidate, model) }
            .ok_or(LocalSearchError::InfeasibleInitialSolution)?;
        self.accepted.overwrite_from_state(&self.candidate);
        self.best.overwrite_from_state(&self.candidate);
        Ok(())
    }

    /// Panics unless every state and graph of the engine counts the vessels and berths of `model`, the condition
    /// every unchecked read of a run rests on; it costs constant time.
    fn assert_loaded_for(&self, model: &Model<T>) {
        assert!(
            self.is_loaded_for(model),
            "the engine should be loaded for a model of {} vessels and {} berths",
            model.num_vessels(),
            model.num_berths()
        );
    }

    /// Whether every state and graph of the engine, and the touched berths, count the vessels and berths of
    /// `model`; it costs constant time.
    fn is_loaded_for(&self, model: &Model<T>) -> bool {
        let (num_vessels, num_berths) = (model.num_vessels(), model.num_berths());
        let states_fit = [&self.accepted, &self.candidate, &self.buffered, &self.best]
            .iter()
            .all(|state| state.num_vessels() == num_vessels && state.num_berths() == num_berths);
        let graphs_fit = [self.mutator.graph(), &self.buffered_graph]
            .iter()
            .all(|graph| graph.num_vessels() == num_vessels && graph.num_berths() == num_berths);
        states_fit && graphs_fit && self.mutator.touched().num_berths() == num_berths
    }

    /// Whether `diff` names only vessels and berths of the graph of the run.
    fn names_only_ids_of_the_run(&self, diff: &ScheduleGraphDiff) -> bool {
        let graph = self.mutator.graph();
        diff.names_only_ids_below(graph.num_vessels(), graph.num_berths())
    }

    /// Runs the search from the loaded schedule with the cost `V`, the rule `rule` and the operator `operator`.
    ///
    /// # Panics
    ///
    /// Panics if the engine is not loaded for `model`.
    fn search<V, R, O, M>(
        &mut self,
        model: &Model<T>,
        rule: &mut R,
        operator: &mut O,
        limits: SearchLimits,
        monitor: &mut M,
    ) -> LocalSearchOutcome<T>
    where
        V: VesselCost<T>,
        R: AcceptanceRule<T>,
        O: LocalSearchOperator<T>,
        M: LocalSearchMonitor<T>,
    {
        self.assert_loaded_for(model);
        rule.on_start(self.mutator.graph(), self.accepted.objective());
        Run::<'_, T, V, R, O, M> {
            engine: self,
            model,
            rule,
            operator,
            monitor,
            statistics: LocalSearchStatistics::default(),
            watch: LimitWatch::start(limits),
            buffer: Buffer::Empty,
            cost: PhantomData,
        }
        .search()
    }

    /// Where the run stands, for a monitor.
    fn progress<'a>(
        &'a self,
        buffer: Buffer,
        statistics: &'a LocalSearchStatistics,
    ) -> SearchProgress<'a, T> {
        SearchProgress {
            best: self.best.as_solution_view(),
            accepted: self.accepted.as_solution_view(),
            buffered: match buffer {
                Buffer::Holding => Some(self.buffered.as_solution_view()),
                Buffer::Empty => None,
            },
            statistics,
        }
    }

    /// Makes the candidate the accepted schedule: copies its touched berths into the accepted state and keeps its
    /// moves.
    fn accept_candidate(&mut self) {
        debug_assert!(
            self.accepted.num_vessels() == self.mutator.graph().num_vessels()
                && self.candidate.num_berths() == self.mutator.touched().num_berths()
        );
        // SAFETY: the states, the touched berths and the graph count the same vessels and berths, as checked when
        // the run started (`Engine::assert_loaded_for`); no step of a run changes a count.
        unsafe {
            self.accepted.patch_from_delta_unchecked(
                &self.candidate,
                self.mutator.touched(),
                self.mutator.graph(),
            );
        }
        self.mutator.accept();
    }

    /// Keeps the candidate aside, with its order and its moves, before it is undone.
    fn buffer_candidate(&mut self) {
        self.buffered_graph
            .overwrite_from_graph(self.mutator.graph());
        self.buffered_diff
            .overwrite_from_graph_diff(self.mutator.diff());
        self.buffered.overwrite_from_state(&self.accepted);
        debug_assert!(
            self.buffered.num_vessels() == self.mutator.graph().num_vessels()
                && self.candidate.num_berths() == self.mutator.touched().num_berths()
        );
        // SAFETY: the states, the touched berths and the graph count the same vessels and berths, as checked when
        // the run started (`Engine::assert_loaded_for`); no step of a run changes a count.
        unsafe {
            self.buffered.patch_from_delta_unchecked(
                &self.candidate,
                self.mutator.touched(),
                self.mutator.graph(),
            );
        }
    }

    /// Makes the buffered candidate the accepted schedule.
    fn accept_buffered(&mut self) {
        self.accepted.overwrite_from_state(&self.buffered);
        self.mutator.overwrite_from_graph(&self.buffered_graph);
    }
}

/// The seeds of the random choices of a run, drawn in a fixed order from the run's seed by SplitMix64, so that
/// each choice draws from a stream of its own whatever the others draw.
#[derive(Debug, Clone, Copy)]
struct RunSeeds {
    /// The seed of the random selection of neighbourhoods.
    selection: u64,
    /// The seed of the random tie-breaking of a descent.
    tie_breaking: u64,
}

impl RunSeeds {
    /// The seeds of a run of `seed`.
    fn new(seed: u64) -> Self {
        let mut seeds = SplitMix64::seed_from_u64(seed);
        let selection = seeds.next_u64();
        let tie_breaking = seeds.next_u64();
        Self {
            selection,
            tie_breaking,
        }
    }
}

/// Whether a candidate is kept aside in the current scan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Buffer {
    Empty,
    Holding,
}

/// What follows a candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StepEnd {
    /// The scan goes on with the next candidate.
    NextCandidate,
    /// The candidate was accepted, and a new scan starts.
    NextScan,
    /// The run ends.
    Terminate(TerminationReason),
}

/// One run of an engine: the engine and what drives it, the work done, the limits, and whether a candidate is kept
/// aside; the vessel cost is `V`.
///
/// The engine is loaded for `model`.
struct Run<'a, T, V, R, O, M>
where
    T: SolverNumeric,
{
    engine: &'a mut Engine<T>,
    model: &'a Model<T>,
    rule: &'a mut R,
    operator: &'a mut O,
    monitor: &'a mut M,
    statistics: LocalSearchStatistics,
    watch: LimitWatch,
    buffer: Buffer,
    cost: PhantomData<V>,
}

impl<T, V, R, O, M> std::fmt::Debug for Run<'_, T, V, R, O, M>
where
    T: SolverNumeric,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Run")
            .field("statistics", &self.statistics)
            .field("watch", &self.watch)
            .field("buffer", &self.buffer)
            .finish_non_exhaustive()
    }
}

impl<T, V, R, O, M> Run<'_, T, V, R, O, M>
where
    T: SolverNumeric,
    V: VesselCost<T>,
    R: AcceptanceRule<T>,
    O: LocalSearchOperator<T>,
    M: LocalSearchMonitor<T>,
{
    /// Runs scans until the run ends, and answers its outcome.
    fn search(mut self) -> LocalSearchOutcome<T> {
        self.monitor.on_start(
            self.model,
            self.engine.progress(self.buffer, &self.statistics),
        );
        let reason = match self.watch.reached(&self.statistics, Clock::Read) {
            Some(reason) => reason,
            None => self.scan_until_end(),
        };
        self.monitor
            .on_end(self.engine.progress(self.buffer, &self.statistics));
        LocalSearchOutcome {
            solution: Solution::from(self.engine.best.as_solution_view()),
            termination_reason: reason,
            statistics: self.statistics,
        }
    }

    /// Asks the monitor before each scan and scans, until the monitor or a scan ends the run, with its reason.
    fn scan_until_end(&mut self) -> TerminationReason {
        loop {
            if let Some(reason) = self.monitor_command() {
                return reason;
            }
            if let Some(reason) = self.scan() {
                return reason;
            }
        }
    }

    /// The end the monitor asks for, or `None`.
    fn monitor_command(&mut self) -> Option<TerminationReason> {
        match self
            .monitor
            .search_command(self.engine.progress(self.buffer, &self.statistics))
        {
            SearchCommand::Continue => None,
            SearchCommand::Terminate(reason) => Some(reason),
        }
    }

    /// Scans the neighbourhood of the accepted schedule until a candidate is accepted, the scan is exhausted and
    /// the metaheuristic restarts it, or the run ends, with its reason.
    fn scan(&mut self) -> Option<TerminationReason> {
        self.operator.prepare(
            self.engine.accepted.objective(),
            self.engine.mutator.graph(),
        );
        let solutions_at_start = self.statistics.total_solutions;
        loop {
            self.engine.mutator.clear_diff();
            debug_assert!(self.engine.is_loaded_for(self.model));
            // SAFETY: the graph counts the vessels the operator was prepared for and has the vessels and berths of
            // the model, as checked when the run started (`Engine::assert_loaded_for`); no move changes a count.
            if !unsafe {
                self.operator
                    .next_neighbor_unchecked(self.model, &mut self.engine.mutator)
            } {
                return self.exhaust(solutions_at_start);
            }
            match self.step() {
                StepEnd::NextCandidate => {}
                StepEnd::NextScan => return None,
                StepEnd::Terminate(reason) => return Some(reason),
            }
        }
    }

    /// Ends an exhausted scan that started with `solutions_at_start` feasible candidates counted: commits the
    /// candidate kept aside where the metaheuristic asks, and restarts the scan, unless the metaheuristic or a
    /// limit ends the run; the limits are checked with the deadlines.
    ///
    /// A scan that yielded no feasible candidate, none at all or only infeasible ones, and committed nothing ends
    /// the run: the accepted schedule is unchanged, so every later scan proposes the same moves, and whether a
    /// move is feasible does not depend on the metaheuristic.
    fn exhaust(&mut self, solutions_at_start: u64) -> Option<TerminationReason> {
        self.statistics.cycles += 1;
        self.monitor
            .on_neighborhood_exhausted(self.engine.progress(self.buffer, &self.statistics));

        let committed = self.buffer == Buffer::Holding && self.rule.commits_buffered();
        if committed {
            self.commit_buffered();
        }
        if !committed && self.statistics.total_solutions == solutions_at_start {
            return Some(TerminationReason::NeighborhoodExhausted);
        }

        self.rule
            .after_step(self.engine.best.objective(), self.engine.mutator.graph());
        match self
            .rule
            .on_neighborhood_exhausted(self.engine.mutator.graph())
        {
            NeighborhoodExhaustionOutcome::Restart => {
                self.watch.reached(&self.statistics, Clock::Read)
            }
            NeighborhoodExhaustionOutcome::Terminate => {
                Some(TerminationReason::NeighborhoodExhausted)
            }
        }
    }

    /// Makes the candidate kept aside the accepted schedule.
    fn commit_buffered(&mut self) {
        let previous_best = self.engine.best.objective();
        self.engine.accept_buffered();
        self.buffer = Buffer::Empty;
        self.statistics.accepted_solutions += 1;
        self.operator
            .on_accepted(AcceptedMove::KeptAside, self.engine.accepted.objective());
        debug_assert!(
            self.engine
                .names_only_ids_of_the_run(&self.engine.buffered_diff)
        );
        // SAFETY: the buffered diff records moves on the graph of the run, which has the counts of the graph
        // `on_start` was given (`Engine::assert_loaded_for`; no move changes a count).
        unsafe {
            self.rule.on_accept_unchecked(
                self.engine.accepted.objective(),
                self.engine.mutator.graph(),
                &self.engine.buffered_diff,
            );
        }
        self.monitor
            .on_buffered_solution_accepted(self.engine.progress(self.buffer, &self.statistics));
        if self.engine.accepted.objective() < previous_best {
            self.record_new_best();
        }
    }

    /// Makes the accepted schedule the best one.
    fn record_new_best(&mut self) {
        self.engine.best.overwrite_from_state(&self.engine.accepted);
        self.watch.on_new_best(&self.statistics);
        self.monitor
            .on_best_solution_updated(self.engine.progress(self.buffer, &self.statistics));
        self.rule.on_new_best(self.engine.best.objective());
    }

    /// Decodes the candidate the last move made and carries out its fate.
    fn step(&mut self) -> StepEnd {
        self.statistics.iterations += 1;
        let engine = &mut *self.engine;
        debug_assert!(engine.is_loaded_for(self.model));
        // SAFETY: the touched berths, the graph, the states and the model count the same vessels and berths, as
        // checked when the run started (`Engine::assert_loaded_for`); no step of a run changes a count.
        let feasible = unsafe {
            decode_unchecked::<T, V>(
                engine.mutator.touched(),
                engine.mutator.graph(),
                &mut engine.candidate,
                &engine.accepted,
                self.model,
            )
        };
        if feasible.is_none() {
            return self.discard_infeasible();
        }

        self.statistics.total_solutions += 1;
        self.watch.note_solution(self.statistics.total_solutions);
        let candidate_objective = self.engine.candidate.objective();
        let buffered_objective = match self.buffer {
            Buffer::Holding => Some(self.engine.buffered.objective()),
            Buffer::Empty => None,
        };
        debug_assert!(
            self.engine
                .names_only_ids_of_the_run(self.engine.mutator.diff())
        );
        // SAFETY: the diff records moves on the graph of the run, which has the counts of the graph `on_start` was
        // given (`Engine::assert_loaded_for`; no move changes a count).
        let outcome = unsafe {
            self.rule.decide_unchecked(
                self.engine.accepted.objective(),
                buffered_objective,
                candidate_objective,
                self.engine.mutator.diff(),
            )
        };
        match outcome {
            AcceptanceOutcome::Accept => self.accept(),
            AcceptanceOutcome::Buffer => self.keep_aside(),
            AcceptanceOutcome::Reject => self.reject(),
        }
    }

    /// Undoes a candidate that decoded to no feasible schedule.
    fn discard_infeasible(&mut self) -> StepEnd {
        self.engine.mutator.rollback();
        self.statistics.infeasible_moves += 1;
        self.end_step(StepEnd::NextCandidate)
    }

    /// Makes the candidate the accepted schedule, and the best where it is better.
    fn accept(&mut self) -> StepEnd {
        self.engine.accept_candidate();
        self.statistics.accepted_solutions += 1;
        self.operator
            .on_accepted(AcceptedMove::Last, self.engine.accepted.objective());
        debug_assert!(
            self.engine
                .names_only_ids_of_the_run(self.engine.mutator.diff())
        );
        // SAFETY: the diff records moves on the graph of the run, which has the counts of the graph `on_start` was
        // given (`Engine::assert_loaded_for`; no move changes a count).
        unsafe {
            self.rule.on_accept_unchecked(
                self.engine.accepted.objective(),
                self.engine.mutator.graph(),
                self.engine.mutator.diff(),
            );
        }
        self.monitor
            .on_candidate_accepted(self.engine.progress(self.buffer, &self.statistics));
        if self.engine.accepted.objective() < self.engine.best.objective() {
            self.record_new_best();
        }
        self.buffer = Buffer::Empty;
        self.end_step(StepEnd::NextScan)
    }

    /// Keeps the candidate aside and undoes it.
    fn keep_aside(&mut self) -> StepEnd {
        self.engine.buffer_candidate();
        self.buffer = Buffer::Holding;
        self.operator.on_kept_aside();
        self.engine.mutator.rollback();
        self.end_step(StepEnd::NextCandidate)
    }

    /// Undoes the rejected candidate.
    fn reject(&mut self) -> StepEnd {
        self.engine.mutator.rollback();
        self.end_step(StepEnd::NextCandidate)
    }

    /// Ends a step: tells the metaheuristic, and answers `next` unless a limit or, at a checkpoint, the monitor
    /// ends the run; only a step the limits name as due is checked.
    fn end_step(&mut self, next: StepEnd) -> StepEnd {
        self.rule
            .after_step(self.engine.best.objective(), self.engine.mutator.graph());
        if !self.watch.is_due(self.statistics.iterations) {
            return next;
        }
        match self.watch.check_step(&self.statistics) {
            StepCheck::Reached(reason) => StepEnd::Terminate(reason),
            StepCheck::Checkpoint => self.monitor_command().map_or(next, StepEnd::Terminate),
            StepCheck::Continue => next,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        metaheuristic::{
            GreedyDescent, GuidedLocalSearch, MoveSelection, PenalizationTrigger, TieBreaking,
        },
        monitor::NoMonitor,
        operator::{MultiArmedBandit, Neighborhood, NeighborhoodSelection},
    };
    use std::time::Duration;
    use talos_core::ClosedOpenInterval;
    use talos_model::{BerthIndex, VesselIndex};

    /// Two vessels on two berths, both open on [0, 200): V0 takes 5 and V1 takes 10 at either berth, both arrive
    /// at 0, must leave by 100 and weigh 1.
    fn two_vessels() -> Model<i64> {
        Model::new(
            vec![0, 0],
            vec![100, 100],
            vec![1, 1],
            vec![Some(5), Some(5), Some(10), Some(10)],
            vec![
                vec![ClosedOpenInterval::new(0, 200)],
                vec![ClosedOpenInterval::new(0, 200)],
            ],
        )
    }

    /// `num_vessels` vessels on `num_berths` berths open on [0, 2000): each takes 10 everywhere, arrives at 0,
    /// must leave by 1000 and weighs 1.
    fn uniform_model(num_vessels: usize, num_berths: usize) -> Model<i64> {
        Model::new(
            vec![0; num_vessels],
            vec![1000; num_vessels],
            vec![1; num_vessels],
            vec![Some(10); num_vessels * num_berths],
            vec![vec![ClosedOpenInterval::new(0, 2000)]; num_berths],
        )
    }

    /// V0 then V1 on B0: V0 from 0, V1 from 5.
    const BOTH_ON_FIRST_BERTH: [BerthIndex; 2] = [BerthIndex::new(0), BerthIndex::new(0)];
    const BOTH_ON_FIRST_BERTH_STARTS: [i64; 2] = [0, 5];

    /// A vessel's weight times its start, as its cost.
    #[derive(Debug)]
    struct WeightedStartCost;

    impl VesselCost<i64> for WeightedStartCost {
        unsafe fn cost_unchecked(
            model: &Model<i64>,
            vessel: VesselIndex,
            _: BerthIndex,
            start: i64,
        ) -> Option<i64> {
            Some(model.vessel_weight(vessel) * start)
        }
    }

    /// Swaps V0 and V1 up to `moves` times per scan.
    #[derive(Debug)]
    struct SwapFirstTwo {
        moves: u64,
        made: u64,
        num_vessels: usize,
    }

    impl SwapFirstTwo {
        fn new(moves: u64) -> Self {
            Self {
                moves,
                made: 0,
                num_vessels: 0,
            }
        }
    }

    impl LocalSearchOperator<i64> for SwapFirstTwo {
        fn prepare(&mut self, _accepted_objective: i64, graph: &ScheduleGraph) {
            self.made = 0;
            self.num_vessels = graph.num_vessels();
        }

        unsafe fn next_neighbor_unchecked(
            &mut self,
            _model: &Model<i64>,
            mutator: &mut Mutator,
        ) -> bool {
            if self.made >= self.moves || self.num_vessels < 2 {
                return false;
            }
            self.made += 1;
            mutator.swap_vessels(VesselIndex::new(0), VesselIndex::new(1));
            true
        }
    }

    /// A rule that gives every candidate the same fate, commits what it kept aside, and ends the run at the first
    /// exhausted scan; it counts its steps.
    #[derive(Debug)]
    struct FixedFate {
        fate: AcceptanceOutcome,
        steps: u64,
    }

    impl FixedFate {
        fn new(fate: AcceptanceOutcome) -> Self {
            Self { fate, steps: 0 }
        }
    }

    impl AcceptanceRule<i64> for FixedFate {
        fn on_start(&mut self, _graph: &ScheduleGraph, _initial_objective: i64) {}

        unsafe fn decide_unchecked(
            &mut self,
            _: i64,
            _: Option<i64>,
            _: i64,
            _: &ScheduleGraphDiff,
        ) -> AcceptanceOutcome {
            self.fate
        }

        unsafe fn on_accept_unchecked(&mut self, _: i64, _: &ScheduleGraph, _: &ScheduleGraphDiff) {
        }

        fn on_new_best(&mut self, _best_objective: i64) {}

        fn after_step(&mut self, _best_objective: i64, _graph: &ScheduleGraph) {
            self.steps += 1;
        }

        fn commits_buffered(&self) -> bool {
            true
        }

        fn on_neighborhood_exhausted(
            &mut self,
            _graph: &ScheduleGraph,
        ) -> NeighborhoodExhaustionOutcome {
            NeighborhoodExhaustionOutcome::Terminate
        }
    }

    /// Counts the calls of every method.
    #[derive(Debug, Default)]
    struct CountingMonitor {
        starts: u64,
        ends: u64,
        commands: u64,
        accepted: u64,
        buffered_accepted: u64,
        exhausted: u64,
        bests: Vec<i64>,
    }

    impl LocalSearchMonitor<i64> for CountingMonitor {
        fn on_start(&mut self, _: &Model<i64>, _: SearchProgress<'_, i64>) {
            self.starts += 1;
        }
        fn on_end(&mut self, _: SearchProgress<'_, i64>) {
            self.ends += 1;
        }
        fn on_candidate_accepted(&mut self, _: SearchProgress<'_, i64>) {
            self.accepted += 1;
        }
        fn on_buffered_solution_accepted(&mut self, _: SearchProgress<'_, i64>) {
            self.buffered_accepted += 1;
        }
        fn on_neighborhood_exhausted(&mut self, _: SearchProgress<'_, i64>) {
            self.exhausted += 1;
        }
        fn on_best_solution_updated(&mut self, progress: SearchProgress<'_, i64>) {
            self.bests.push(progress.best.objective_value());
        }
        fn search_command(&mut self, _: SearchProgress<'_, i64>) -> SearchCommand {
            self.commands += 1;
            SearchCommand::Continue
        }
    }

    /// Loads `model` with `berths` and `starts`, and searches with the cost [`WeightedStartCost`].
    fn search_mocked<R, O, M>(
        model: &Model<i64>,
        initial: (&[BerthIndex], &[i64]),
        rule: &mut R,
        operator: &mut O,
        limits: SearchLimits,
        monitor: &mut M,
    ) -> LocalSearchOutcome<i64>
    where
        R: AcceptanceRule<i64>,
        O: LocalSearchOperator<i64>,
        M: LocalSearchMonitor<i64>,
    {
        let mut engine = Engine::new();
        engine
            .load::<WeightedStartCost>(model, SolutionView::new(initial.0, initial.1, 0))
            .expect("the initial schedule should decode");
        engine.search::<WeightedStartCost, R, O, M>(model, rule, operator, limits, monitor)
    }

    fn both_on_first_berth() -> (&'static [BerthIndex], &'static [i64]) {
        (&BOTH_ON_FIRST_BERTH, &BOTH_ON_FIRST_BERTH_STARTS)
    }

    fn iterations(limit: u64) -> SearchLimits {
        SearchLimits {
            iterations: Some(limit),
            ..SearchLimits::default()
        }
    }

    #[test]
    fn test_run_refuses_parameters_without_a_neighborhood() {
        let parameters = LocalSearchParameters {
            neighborhoods: Vec::new(),
            ..LocalSearchParameters::default()
        };
        let initial = SolutionView::new(&BOTH_ON_FIRST_BERTH, &BOTH_ON_FIRST_BERTH_STARTS, 0);
        let refusal = Engine::new().run(&two_vessels(), initial, &parameters, &mut NoMonitor);
        assert_eq!(refusal, Err(LocalSearchError::NoNeighborhood));
    }

    // A schedule of three vessels for a model of two would build a graph whose third vessel every unchecked read
    // of the model takes for one of its own; the run must refuse it at its door, in every build.
    #[test]
    fn test_run_refuses_a_schedule_of_more_vessels_than_the_model() {
        let berths = [BerthIndex::new(0); 3];
        let initial = SolutionView::new(&berths, &[0, 5, 15], 0);
        let refusal = Engine::new().run(
            &two_vessels(),
            initial,
            &LocalSearchParameters::default(),
            &mut NoMonitor,
        );
        assert_eq!(
            refusal,
            Err(LocalSearchError::VesselCountMismatch {
                model_vessels: 2,
                solution_vessels: 3
            })
        );
    }

    #[test]
    fn test_run_refuses_a_berth_beyond_the_model() {
        let berths = [BerthIndex::new(0), BerthIndex::new(2)];
        let initial = SolutionView::new(&berths, &[0, 5], 0);
        let refusal = Engine::new().run(
            &two_vessels(),
            initial,
            &LocalSearchParameters::default(),
            &mut NoMonitor,
        );
        assert!(matches!(
            refusal,
            Err(LocalSearchError::BerthOutOfBounds { .. })
        ));
    }

    #[test]
    fn test_run_refuses_an_order_without_a_feasible_schedule() {
        // Fifteen vessels of 10 on one berth end at 150, past the latest departure of 100.
        let model = Model::new(
            vec![0_i64; 15],
            vec![100; 15],
            vec![1; 15],
            vec![Some(10); 15],
            vec![vec![ClosedOpenInterval::new(0, 1000)]],
        );
        let berths = [BerthIndex::new(0); 15];
        let starts: Vec<i64> = (0..15).collect();
        let initial = SolutionView::new(&berths, &starts, 0);
        let refusal = Engine::new().run(
            &model,
            initial,
            &LocalSearchParameters::default(),
            &mut NoMonitor,
        );
        assert_eq!(refusal, Err(LocalSearchError::InfeasibleInitialSolution));
    }

    #[test]
    fn test_search_ends_at_once_under_a_budget_of_zero() {
        let outcome = search_mocked(
            &two_vessels(),
            both_on_first_berth(),
            &mut FixedFate::new(AcceptanceOutcome::Accept),
            &mut SwapFirstTwo::new(10),
            iterations(0),
            &mut NoMonitor,
        );
        assert_eq!(
            outcome.termination_reason,
            TerminationReason::IterationLimitReached
        );
        assert_eq!(outcome.statistics, LocalSearchStatistics::default());
    }

    #[test]
    fn test_accepting_every_candidate_counts_each_and_starts_a_scan_after_each() {
        let mut monitor = CountingMonitor::default();
        let mut rule = FixedFate::new(AcceptanceOutcome::Accept);
        let outcome = search_mocked(
            &two_vessels(),
            both_on_first_berth(),
            &mut rule,
            &mut SwapFirstTwo::new(100),
            iterations(5),
            &mut monitor,
        );
        assert_eq!(
            outcome.termination_reason,
            TerminationReason::IterationLimitReached
        );
        assert_eq!(outcome.statistics.iterations, 5);
        assert_eq!(outcome.statistics.accepted_solutions, 5);
        assert_eq!(outcome.statistics.total_solutions, 5);
        assert_eq!(rule.steps, 5);
        // The monitor is asked before each of the five scans, and the run ends within the fifth.
        assert_eq!((monitor.starts, monitor.ends, monitor.commands), (1, 1, 5));
        assert_eq!(monitor.accepted, 5);
        // V0 first costs 0 + 5, the best: the swaps alternate between 10 and 5, never below.
        assert!(monitor.bests.is_empty());
        assert_eq!(outcome.solution.objective_value(), 5);
    }

    #[test]
    fn test_rejecting_every_candidate_keeps_the_initial_schedule() {
        let mut monitor = CountingMonitor::default();
        let mut rule = FixedFate::new(AcceptanceOutcome::Reject);
        let outcome = search_mocked(
            &two_vessels(),
            both_on_first_berth(),
            &mut rule,
            &mut SwapFirstTwo::new(100),
            iterations(7),
            &mut monitor,
        );
        assert_eq!(outcome.statistics.iterations, 7);
        assert_eq!(outcome.statistics.accepted_solutions, 0);
        assert_eq!(outcome.statistics.rejected_solutions(), 7);
        assert_eq!(monitor.accepted, 0);
        assert_eq!(rule.steps, 7);
        assert_eq!(outcome.solution.objective_value(), 5);
        assert_eq!(outcome.solution.start_times().raw, [0, 5]);
    }

    #[test]
    fn test_keeping_every_candidate_aside_commits_one_at_the_exhausted_scan() {
        let mut monitor = CountingMonitor::default();
        let outcome = search_mocked(
            &two_vessels(),
            both_on_first_berth(),
            &mut FixedFate::new(AcceptanceOutcome::Buffer),
            &mut SwapFirstTwo::new(2),
            iterations(100),
            &mut monitor,
        );
        assert_eq!(
            outcome.termination_reason,
            TerminationReason::NeighborhoodExhausted
        );
        assert_eq!(
            (outcome.statistics.iterations, outcome.statistics.cycles),
            (2, 1)
        );
        assert_eq!(outcome.statistics.accepted_solutions, 1);
        assert_eq!(
            (
                monitor.accepted,
                monitor.buffered_accepted,
                monitor.exhausted
            ),
            (0, 1, 1)
        );
    }

    /// Two vessels on two berths that both close at 15: V0 takes 5 and arrives at 0, V1 takes 10 and arrives at 5.
    /// Served first, V1 ends at 15 and V0 fits nowhere after it, so the swap of the two on one berth is infeasible.
    fn v1_first_is_infeasible() -> Model<i64> {
        Model::new(
            vec![0_i64, 5],
            vec![100, 100],
            vec![1, 1],
            vec![Some(5), Some(5), Some(10), Some(10)],
            vec![
                vec![ClosedOpenInterval::new(0, 15)],
                vec![ClosedOpenInterval::new(0, 15)],
            ],
        )
    }

    #[test]
    fn test_an_infeasible_candidate_is_counted_and_undone() {
        let model = v1_first_is_infeasible();
        let mut monitor = CountingMonitor::default();
        let outcome = search_mocked(
            &model,
            both_on_first_berth(),
            &mut FixedFate::new(AcceptanceOutcome::Accept),
            &mut SwapFirstTwo::new(3),
            iterations(3),
            &mut monitor,
        );
        assert_eq!(outcome.statistics.infeasible_moves, 3);
        assert_eq!(outcome.statistics.total_solutions, 0);
        assert_eq!(monitor.accepted, 0);
        assert_eq!(outcome.solution.start_times().raw, [0, 5]);
    }

    // Best-improvement descent from a schedule whose only move worsens it stands at a local optimum: the first
    // scan finds nothing to commit, and the search must end there instead of scanning again until a budget runs
    // out.
    #[test]
    fn test_best_improvement_descent_ends_at_a_local_optimum() {
        let settings = GreedyDescent {
            selection: MoveSelection::BestImprovement(TieBreaking::KeepFirst),
        };
        let outcome = search_mocked(
            &two_vessels(),
            both_on_first_berth(),
            &mut GreedyDescentRule::new(&settings, 0),
            &mut SwapFirstTwo::new(1),
            iterations(1000),
            &mut NoMonitor,
        );
        assert_eq!(
            outcome.termination_reason,
            TerminationReason::NeighborhoodExhausted
        );
        assert_eq!(outcome.statistics.iterations, 1);
        assert_eq!(outcome.solution.objective_value(), 5);
    }

    // A schedule with no neighbour at all cannot change: every scan of its neighbourhood is empty, whatever the
    // metaheuristic makes of the exhaustion. The search must end at the first empty scan instead of restarting
    // until a budget runs out, and a budget counted in candidates would never run out.
    #[test]
    fn test_search_ends_at_a_scan_without_candidates() {
        let outcome = search_mocked(
            &two_vessels(),
            both_on_first_berth(),
            &mut GuidedLocalSearchRule::new(&GuidedLocalSearch::default()),
            &mut SwapFirstTwo::new(0),
            SearchLimits {
                cycles: Some(1000),
                ..SearchLimits::default()
            },
            &mut NoMonitor,
        );
        assert_eq!(
            outcome.termination_reason,
            TerminationReason::NeighborhoodExhausted
        );
        assert_eq!(outcome.statistics.cycles, 1);
    }

    // A scan whose every candidate is infeasible commits nothing, so every later scan proposes the same infeasible
    // candidates: the search must end at the first such scan, and a budget counted in feasible candidates would
    // never run out. The cycle limit only bounds the test if the search does not end.
    #[test]
    fn test_search_ends_at_a_scan_whose_every_candidate_is_infeasible() {
        let outcome = search_mocked(
            &v1_first_is_infeasible(),
            both_on_first_berth(),
            &mut GuidedLocalSearchRule::new(&GuidedLocalSearch::default()),
            &mut SwapFirstTwo::new(1),
            SearchLimits {
                solutions: Some(10),
                cycles: Some(1000),
                ..SearchLimits::default()
            },
            &mut NoMonitor,
        );
        assert_eq!(
            outcome.termination_reason,
            TerminationReason::NeighborhoodExhausted
        );
        assert_eq!(outcome.statistics.cycles, 1);
        assert_eq!(outcome.statistics.infeasible_moves, 1);
    }

    #[test]
    fn test_one_engine_serves_models_of_any_size_in_turn() {
        let mut engine = Engine::new();
        for (num_vessels, num_berths) in [(2, 2), (10, 5), (3, 2), (50, 10), (2, 1)] {
            let model = uniform_model(num_vessels, num_berths);
            let berths: Vec<BerthIndex> = (0..num_vessels)
                .map(|v| BerthIndex::new(v % num_berths))
                .collect();
            let starts: Vec<i64> = (0..num_vessels)
                .map(|v| i64::try_from(v / num_berths).unwrap_or(0) * 10)
                .collect();
            let parameters = LocalSearchParameters {
                limits: iterations(200),
                ..LocalSearchParameters::default()
            };
            let outcome = engine
                .run(
                    &model,
                    SolutionView::new(&berths, &starts, 0),
                    &parameters,
                    &mut NoMonitor,
                )
                .expect("the schedule should be one of the model");
            assert_eq!(outcome.solution.num_vessels(), num_vessels);
            assert!(outcome.statistics.iterations > 0);
            assert_eq!(model.check_feasibility(outcome.solution.as_view()), Ok(()));
        }
    }

    /// A model of twelve vessels with staggered arrivals and weights on three berths, each vessel barred from at
    /// most one berth, with an initial schedule of it: the vessels by index, each on a berth it may use.
    fn staggered_instance() -> (Model<i64>, Vec<BerthIndex>, Vec<i64>) {
        let arrivals: Vec<i64> = (0..12).map(|v| (v * 7) % 23).collect();
        let weights: Vec<i64> = (0..12).map(|v| 1 + (v * 5) % 4).collect();
        let processing: Vec<Option<i64>> = (0..36)
            .map(|i| {
                if i % 11 == 4 {
                    None
                } else {
                    Some(3 + (i * 13) % 9)
                }
            })
            .collect();
        let model = Model::new(
            arrivals,
            vec![1000; 12],
            weights,
            processing,
            vec![vec![ClosedOpenInterval::new(0, 5000)]; 3],
        );
        let berths = (0..12)
            .map(|v| {
                (0..3)
                    .map(|b| BerthIndex::new((v + b) % 3))
                    .find(|&b| {
                        model
                            .vessel_processing_time(VesselIndex::new(v), b)
                            .is_some()
                    })
                    .expect("every vessel should have a berth it may use")
            })
            .collect();
        (model, berths, (0..12).collect())
    }

    /// The objectives of every accepted schedule of a run of `parameters` on the staggered instance, and the
    /// statistics of the run.
    fn trajectory(parameters: &LocalSearchParameters) -> (Vec<i64>, LocalSearchStatistics) {
        #[derive(Debug, Default)]
        struct Trajectory(Vec<i64>);
        impl LocalSearchMonitor<i64> for Trajectory {
            fn on_candidate_accepted(&mut self, progress: SearchProgress<'_, i64>) {
                self.0.push(progress.accepted.objective_value());
            }
            fn on_buffered_solution_accepted(&mut self, progress: SearchProgress<'_, i64>) {
                self.0.push(progress.accepted.objective_value());
            }
        }
        let (model, berths, starts) = staggered_instance();
        let mut monitor = Trajectory::default();
        let outcome = Engine::new()
            .run(
                &model,
                SolutionView::new(&berths, &starts, 0),
                parameters,
                &mut monitor,
            )
            .expect("the schedule should be one of the model");
        (monitor.0, outcome.statistics)
    }

    #[test]
    fn test_a_run_repeats_its_trajectory_for_a_seed_and_varies_it_with_the_seed() {
        let parameters = |seed| LocalSearchParameters {
            selection: NeighborhoodSelection::Random,
            limits: iterations(3000),
            seed,
            ..LocalSearchParameters::default()
        };
        let first = trajectory(&parameters(1));
        assert!(first.0.len() > 10);
        assert_eq!(first, trajectory(&parameters(1)));
        assert_ne!(first, trajectory(&parameters(2)));
    }

    #[test]
    fn test_round_robin_runs_do_not_depend_on_the_seed() {
        let parameters = |seed| LocalSearchParameters {
            neighborhoods: vec![Neighborhood::InterBerthShift, Neighborhood::IntraBerthSwap],
            limits: iterations(2000),
            seed,
            ..LocalSearchParameters::default()
        };
        let first = trajectory(&parameters(1));
        assert!(first.0.len() > 10);
        assert_eq!(first, trajectory(&parameters(2)));
    }

    // The bandit learns only from accepted moves, those taken at once and those kept aside and committed at the end
    // of a scan; under either a run stays reproducible and ends with a schedule of the model.
    #[test]
    fn test_a_run_under_the_multi_armed_bandit_repeats_itself_and_stays_feasible() {
        let (model, berths, starts) = staggered_instance();
        for metaheuristic in [
            Metaheuristic::default(),
            Metaheuristic::GreedyDescent(GreedyDescent {
                selection: MoveSelection::BestImprovement(TieBreaking::Random),
            }),
        ] {
            let parameters = LocalSearchParameters {
                selection: NeighborhoodSelection::MultiArmedBandit(MultiArmedBandit::default()),
                metaheuristic,
                limits: iterations(3000),
                ..LocalSearchParameters::default()
            };
            let first = trajectory(&parameters);
            assert!(!first.0.is_empty());
            assert_eq!(first, trajectory(&parameters));
            let outcome = Engine::new()
                .run(
                    &model,
                    SolutionView::new(&berths, &starts, 0),
                    &parameters,
                    &mut NoMonitor,
                )
                .expect("the schedule should be one of the model");
            assert!(outcome.statistics.accepted_solutions > 0);
            assert_eq!(model.check_feasibility(outcome.solution.as_view()), Ok(()));
        }
    }

    /// The outcome of a guided local search of `trigger` on the staggered instance, ended after `cycles` exhausted
    /// scans.
    fn guided_search_on_staggered_instance(
        trigger: PenalizationTrigger,
        cycles: u64,
    ) -> LocalSearchOutcome<i64> {
        let (model, berths, starts) = staggered_instance();
        let parameters = LocalSearchParameters {
            metaheuristic: Metaheuristic::GuidedLocalSearch(GuidedLocalSearch {
                trigger,
                ..GuidedLocalSearch::default()
            }),
            limits: SearchLimits {
                cycles: Some(cycles),
                ..SearchLimits::default()
            },
            ..LocalSearchParameters::default()
        };
        Engine::new()
            .run(
                &model,
                SolutionView::new(&berths, &starts, 0),
                &parameters,
                &mut NoMonitor,
            )
            .expect("the schedule should be one of the model")
    }

    // An exhausted scan accepted no move, so a trigger that counts accepted moves cannot fire from it: unless the
    // exhaustion penalizes, every later scan repeats it and the search stands still until a limit ends it.
    #[test]
    fn test_a_guided_search_triggered_by_moves_leaves_its_first_local_optimum() {
        let trigger = PenalizationTrigger::AfterMoves(
            std::num::NonZeroU64::new(1_000_000).expect("a million is not zero"),
        );
        let first = guided_search_on_staggered_instance(trigger, 1);
        let later = guided_search_on_staggered_instance(trigger, 10);
        assert_eq!(
            later.termination_reason,
            TerminationReason::CycleLimitReached
        );
        assert!(first.statistics.accepted_solutions > 0);
        assert!(later.statistics.accepted_solutions > first.statistics.accepted_solutions);
    }

    // Under a trigger that counts steps without a new best, an exhausted scan does not penalize, but each of its
    // candidates is such a step: the count fires within a few scans, and the search leaves its local optimum.
    #[test]
    fn test_a_guided_search_triggered_by_non_improving_steps_leaves_its_first_local_optimum() {
        let trigger = PenalizationTrigger::AfterNonImprovements(
            std::num::NonZeroU64::new(200).expect("two hundred is not zero"),
        );
        let first = guided_search_on_staggered_instance(trigger, 1);
        let later = guided_search_on_staggered_instance(trigger, 20);
        assert_eq!(
            later.termination_reason,
            TerminationReason::CycleLimitReached
        );
        assert!(later.statistics.accepted_solutions > first.statistics.accepted_solutions);
    }

    /// The outcome of a run of `parameters` from the initial schedule of the staggered instance on `engine`.
    fn run_staggered(
        engine: &mut Engine<i64>,
        parameters: &LocalSearchParameters,
    ) -> LocalSearchOutcome<i64> {
        let (model, berths, starts) = staggered_instance();
        engine
            .run(
                &model,
                SolutionView::new(&berths, &starts, 0),
                parameters,
                &mut NoMonitor,
            )
            .expect("the schedule should be one of the model")
    }

    // An engine keeps the memory of its states and records from one run to the next, and nothing else: a run on a
    // reused engine, after runs on other models, answers what the same run on a fresh engine answers.
    #[test]
    fn test_a_reused_engine_answers_as_a_fresh_one() {
        let parameters = LocalSearchParameters {
            selection: NeighborhoodSelection::Random,
            limits: iterations(5000),
            seed: 3,
            ..LocalSearchParameters::default()
        };
        let fresh = run_staggered(&mut Engine::new(), &parameters);
        let mut reused = Engine::new();
        let _ = run_staggered(&mut reused, &parameters);
        let other = uniform_model(30, 4);
        let berths: Vec<BerthIndex> = (0..30).map(|v| BerthIndex::new(v % 4)).collect();
        let starts: Vec<i64> = (0..30).map(|v| (v / 4) * 10).collect();
        let _ = reused
            .run(
                &other,
                SolutionView::new(&berths, &starts, 0),
                &parameters,
                &mut NoMonitor,
            )
            .expect("the schedule should be one of the model");
        assert_eq!(run_staggered(&mut reused, &parameters), fresh);
    }

    #[test]
    fn test_a_deadline_already_past_ends_the_run_before_its_first_candidate() {
        for (limits, reason) in [
            (
                SearchLimits {
                    time: Some(Duration::ZERO),
                    ..SearchLimits::default()
                },
                TerminationReason::TimeLimitReached,
            ),
            (
                SearchLimits {
                    non_improving_time: Some(Duration::ZERO),
                    ..SearchLimits::default()
                },
                TerminationReason::MaxNonImprovingTime,
            ),
        ] {
            let parameters = LocalSearchParameters {
                limits,
                ..LocalSearchParameters::default()
            };
            let outcome = run_staggered(&mut Engine::new(), &parameters);
            assert_eq!(outcome.termination_reason, reason);
            assert_eq!(outcome.statistics.iterations, 0);
        }
    }

    // A deadline is measured at every checkpoint and every exhausted scan, so a run of no other end stops soon
    // after it; the budget of candidates only bounds the test should the deadline not be measured.
    #[test]
    fn test_a_deadline_ends_a_run_that_nothing_else_ends() {
        let parameters = LocalSearchParameters {
            limits: SearchLimits {
                time: Some(Duration::from_millis(20)),
                iterations: Some(200_000_000),
                ..SearchLimits::default()
            },
            ..LocalSearchParameters::default()
        };
        let outcome = run_staggered(&mut Engine::new(), &parameters);
        assert_eq!(
            outcome.termination_reason,
            TerminationReason::TimeLimitReached
        );
        assert!(outcome.statistics.iterations > 0);
    }

    #[test]
    fn test_engine_and_parameters_can_move_to_another_thread() {
        fn assert_send<S: Send>() {}
        assert_send::<Engine<i64>>();
        assert_send::<LocalSearchParameters>();
        assert_send::<LocalSearchOutcome<i64>>();
    }
}
