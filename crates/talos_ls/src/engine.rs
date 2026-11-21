// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    decoding::{ScheduleState, decode_full_unchecked, decode_unchecked},
    metaheuristic::acceptance::{AcceptanceOutcome, Metaheuristic, NeighborhoodExhaustionOutcome},
    monitor::lsmonitor::LocalSearchMonitor,
    objective::{CompletionCost, Objective, TurnaroundCost, VesselCost},
    operator::LocalSearchOperator,
    outcome::LocalSearchOutcome,
    parameters::{MutableLocalSearchParams, check_initial_solution},
    schedule::{Mutator, ScheduleGraph, ScheduleGraphDiff},
    statistics::LocalSearchStatistics,
    termination::{SearchCommand, TerminationReason},
};
use std::time::Instant;
use talos_core::SolverNumeric;
use talos_model::{BerthIndex, Solution, SolutionView};

/// The core orchestration unit for the local search metaheuristic.
///
/// The `Engine` manages a "Quad-Buffer" state system and the necessary
/// auxiliary data structures to perform millions of mutations per second
/// with zero heap allocations in the hot loop.
#[derive(Debug, Clone)]
pub struct Engine<T> {
    /// **The Anchor**: Represents the currently accepted solution in the search process.
    /// All neighborhood moves are evaluated relative to this state.
    accepted_state: ScheduleState<T>,

    /// **The Sandbox**: A scratchpad where mutations and partial decodes are applied.
    /// If a move is rejected, this state is considered "dirty" and may be
    /// partially rolled back or overwritten.
    candidate_state: ScheduleState<T>,

    /// The schedule graph of the candidate, with the records of the moves that made it from the accepted state:
    /// how to undo them, the links and berths they changed, and the berths whose order they changed.
    mutator: Mutator,

    /// **The Neighborhood Buffer**: Used during multi-move exploration (e.g., Tabu Search).
    /// Holds a potential candidate until the entire neighborhood is sampled,
    /// allowing the engine to pick the "best-of-N" moves.
    buffered_state: ScheduleState<T>,

    /// The corresponding graph for the `buffered_state`. This allows the engine to
    /// maintain a separate topological representation of the buffered candidate, which can be
    /// accepted later in the search.
    buffered_topology_graph: ScheduleGraph,

    /// **The Global Optimum**: Stores the mathematically best solution found since the
    /// start of the search. This is the "return value" of the solver.
    best_state: ScheduleState<T>,

    /// Saved copy of the structural diff corresponding to the `buffered_state`.
    /// Captured when a candidate is buffered and passed to `on_accept` when
    /// the buffer is committed.
    buffered_schedule_graph_diff: ScheduleGraphDiff,
}

impl<T> Engine<T> {
    /// Creates a new engine pre-allocated for the given problem dimensions.
    pub fn new(num_vessels: usize, num_berths: usize) -> Self
    where
        T: SolverNumeric,
    {
        let berths = vec![BerthIndex::new(0); num_vessels];
        let starts = vec![T::ZERO; num_vessels];
        let costs = vec![T::ZERO; num_berths];

        let make_state =
            || ScheduleState::new(berths.clone(), starts.clone(), costs.clone(), T::ZERO);

        let dummy_starts: Vec<i32> = (0..num_vessels as i32).collect();
        let topology_graph = ScheduleGraph::from_slices(&berths, &dummy_starts, num_berths);
        let buffered_topology_graph = topology_graph.clone();

        Self {
            accepted_state: make_state(),
            candidate_state: make_state(),
            buffered_state: make_state(),
            best_state: make_state(),
            mutator: Mutator::new(topology_graph),
            buffered_topology_graph,
            buffered_schedule_graph_diff: ScheduleGraphDiff::new(),
        }
    }

    /// Runs the local search engine loop, minimising `objective`.
    ///
    /// Initializes the internal states from `params`, then repeatedly
    /// generates neighbors via the operator, decodes them, and asks the
    /// metaheuristic whether to accept, buffer, or reject each candidate.
    /// The monitor is notified of all lifecycle events and may request early
    /// termination. The `callback` is invoked whenever a new global best
    /// solution is found.
    ///
    /// Returns the best solution found, the termination reason, and statistics.
    ///
    /// Every vessel and berth of the graph the run builds from `params` is one of `params.model`, as the check at
    /// its door establishes; the run's unchecked reads, and the operator's, rest on that.
    ///
    /// # Panics
    ///
    /// Panics if `params.berths` and `params.start_times` are not a schedule of the vessels and berths of
    /// `params.model`, as [`check_initial_solution`] refuses it, or if the initial solution cannot be decoded.
    pub fn run<H, O, M, C>(
        &mut self,
        params: MutableLocalSearchParams<'_, T, H, O, M>,
        objective: Objective,
        callback: C,
    ) -> LocalSearchOutcome<T>
    where
        T: SolverNumeric,
        H: Metaheuristic<T>,
        O: LocalSearchOperator<T>,
        M: LocalSearchMonitor<T>,
        C: FnMut(SolutionView<'_, T>),
    {
        match objective {
            Objective::WeightedTurnaroundTime => {
                self.run_with_cost::<TurnaroundCost, H, O, M, C>(params, callback)
            }
            Objective::WeightedCompletionTime => {
                self.run_with_cost::<CompletionCost, H, O, M, C>(params, callback)
            }
        }
    }

    /// Runs the local search loop as [`Engine::run`] does, with `V` as the cost of each vessel's service.
    fn run_with_cost<V, H, O, M, C>(
        &mut self,
        params: MutableLocalSearchParams<'_, T, H, O, M>,
        mut callback: C,
    ) -> LocalSearchOutcome<T>
    where
        T: SolverNumeric,
        V: VesselCost<T>,
        H: Metaheuristic<T>,
        O: LocalSearchOperator<T>,
        M: LocalSearchMonitor<T>,
        C: FnMut(SolutionView<'_, T>),
    {
        if let Err(error) = check_initial_solution(params.model, params.berths, params.start_times)
        {
            panic!("called `Engine::run` with {error}");
        }

        let model = params.model;
        let operator = params.operator;
        let metaheuristic = params.metaheuristic;
        let mut monitor = params.monitor;

        // Grow all buffers if this problem is larger than anything seen before.
        self.ensure_capacity(model.num_vessels(), model.num_berths());

        // Initialise topology from the validated initial solution, which empties the records of the moves, which
        // may hold moves of an earlier run that ended in a panic.
        self.mutator.overwrite_from_slices(
            &params.berths.raw,
            &params.start_times.raw,
            model.num_berths(),
        );

        // Full-decode to populate the candidate buffer, then propagate to all states.
        // This ensures all four states start from the same decoded solution.
        debug_assert!(
            self.mutator.graph().num_vessels() == model.num_vessels()
                && self.mutator.graph().num_berths() == model.num_berths()
        );
        // SAFETY: the graph was built from `params`, which the check at the door found to be a schedule of the
        // vessels and berths of `model`, and every state was sized for those counts by `ensure_capacity`.
        unsafe {
            decode_full_unchecked::<T, V>(self.mutator.graph(), &mut self.candidate_state, model)
                .expect("initial solution must be decodable");
        }
        self.accepted_state
            .overwrite_from_state(&self.candidate_state);
        self.best_state.overwrite_from_state(&self.candidate_state);
        self.buffered_topology_graph
            .overwrite_from_graph(self.mutator.graph());

        let mut stats = LocalSearchStatistics::default();
        let start_time = Instant::now();
        let mut has_buffered = false;

        monitor.on_start(model, self.accepted_state.as_solution_view());
        metaheuristic.on_start(
            model,
            self.accepted_state.as_solution_view(),
            self.mutator.graph(),
        );

        let termination_reason;

        'outer: loop {
            // Check termination via metaheuristic.
            let cmd = metaheuristic.search_command(
                stats.iterations,
                model,
                self.best_state.as_solution_view(),
                self.accepted_state.as_solution_view(),
                if has_buffered {
                    Some(self.buffered_state.as_solution_view())
                } else {
                    None
                },
            );
            if let SearchCommand::Terminate(reason) = cmd {
                termination_reason = reason;
                break 'outer;
            }

            // Check termination via monitor.
            let cmd = monitor.search_command(
                self.best_state.as_solution_view(),
                self.accepted_state.as_solution_view(),
                if has_buffered {
                    Some(self.buffered_state.as_solution_view())
                } else {
                    None
                },
                &stats,
            );
            if let SearchCommand::Terminate(reason) = cmd {
                termination_reason = reason;
                break 'outer;
            }

            monitor.on_iteration(
                self.best_state.as_solution_view(),
                self.accepted_state.as_solution_view(),
                if has_buffered {
                    Some(self.buffered_state.as_solution_view())
                } else {
                    None
                },
                &stats,
            );

            operator.prepare(self.accepted_state.objective(), self.mutator.graph());
            let iterations_at_scan_start = stats.iterations;

            'inner: loop {
                self.mutator.clear_diff();

                let mutated = {
                    let mutator = &mut self.mutator;
                    // SAFETY: the graph in `mutator` is the one `prepare` saw, and it has the vessels and berths
                    // of `model`, as the check at the door of the run established.
                    unsafe { operator.next_neighbor_unchecked(model, mutator) }
                };

                if !mutated {
                    stats.on_cycle();

                    monitor.on_neighborhood_exhausted(
                        self.best_state.as_solution_view(),
                        self.accepted_state.as_solution_view(),
                        if has_buffered {
                            Some(self.buffered_state.as_solution_view())
                        } else {
                            None
                        },
                        &stats,
                    );

                    // Check whether the buffered solution should be committed.
                    let committed = has_buffered
                        && metaheuristic.should_commit_buffered(
                            model,
                            self.best_state.as_solution_view(),
                            self.accepted_state.as_solution_view(),
                            Some(self.buffered_state.as_solution_view()),
                            self.mutator.graph(),
                            &self.buffered_topology_graph,
                        );
                    if committed {
                        let was_best_obj = self.best_state.objective();
                        self.accept_buffered();
                        has_buffered = false;
                        stats.on_accepted_solution();

                        // Notify the metaheuristic with the saved diff so it
                        // can update its internal state (e.g. tabu memory).
                        metaheuristic.on_accept(
                            model,
                            self.best_state.as_solution_view(),
                            self.accepted_state.as_solution_view(),
                            None,
                            self.mutator.graph(),
                            &self.buffered_schedule_graph_diff,
                        );

                        monitor.on_buffered_solution_accepted(
                            self.best_state.as_solution_view(),
                            self.accepted_state.as_solution_view(),
                            &stats,
                        );

                        if self.accepted_state.objective() < was_best_obj {
                            // Notify BEFORE overwriting best so the monitor can
                            // observe the previous best.
                            monitor.on_best_solution_updated(
                                self.best_state.as_solution_view(),
                                self.accepted_state.as_solution_view(),
                                None,
                                self.accepted_state.as_solution_view(),
                                &stats,
                            );
                            self.update_best();
                            callback(self.best_state.as_solution_view());
                            metaheuristic.on_new_best(
                                model,
                                self.best_state.as_solution_view(),
                                self.mutator.graph(),
                                &self.buffered_schedule_graph_diff,
                            );
                        }
                    }

                    // A scan that yielded no candidate and committed nothing leaves the schedule as it was, and
                    // every later scan would see the same empty neighbourhood.
                    if !committed && stats.iterations == iterations_at_scan_start {
                        termination_reason = TerminationReason::NeighborhoodExhausted;
                        break 'outer;
                    }

                    metaheuristic.on_iteration(
                        stats.iterations,
                        model,
                        self.best_state.as_solution_view(),
                        self.accepted_state.as_solution_view(),
                        if has_buffered {
                            Some(self.buffered_state.as_solution_view())
                        } else {
                            None
                        },
                        self.mutator.graph(),
                    );

                    // Ask the metaheuristic how to proceed after exhaustion.
                    match metaheuristic.on_neighbourhood_exhausted(
                        model,
                        self.best_state.as_solution_view(),
                        self.accepted_state.as_solution_view(),
                        if has_buffered {
                            Some(self.buffered_state.as_solution_view())
                        } else {
                            None
                        },
                        self.mutator.graph(),
                    ) {
                        NeighborhoodExhaustionOutcome::Restart => {
                            operator.reset();
                            continue 'outer;
                        }
                        NeighborhoodExhaustionOutcome::Terminate => {
                            termination_reason = TerminationReason::NeighborhoodExhausted;
                            break 'outer;
                        }
                    }
                }

                stats.on_iteration();

                // SAFETY: the graph has the vessels and berths of `model`, as the check at the door of the run
                // established and as every mutation keeps, and the touched berths count its berths by the
                // invariant of `Mutator`.
                let decode_ok = unsafe {
                    decode_unchecked::<T, V>(
                        self.mutator.touched(),
                        self.mutator.graph(),
                        &mut self.candidate_state,
                        &self.accepted_state,
                        model,
                    )
                };

                if decode_ok.is_none() {
                    // Infeasible candidate — roll back immediately.
                    self.reject_candidate();
                    stats.on_infeasible_move();
                    monitor.on_candidate_infeasible(
                        self.best_state.as_solution_view(),
                        self.accepted_state.as_solution_view(),
                        if has_buffered {
                            Some(self.buffered_state.as_solution_view())
                        } else {
                            None
                        },
                        &stats,
                    );

                    metaheuristic.on_iteration(
                        stats.iterations,
                        model,
                        self.best_state.as_solution_view(),
                        self.accepted_state.as_solution_view(),
                        if has_buffered {
                            Some(self.buffered_state.as_solution_view())
                        } else {
                            None
                        },
                        self.mutator.graph(),
                    );

                    let cmd = monitor.search_command(
                        self.best_state.as_solution_view(),
                        self.accepted_state.as_solution_view(),
                        if has_buffered {
                            Some(self.buffered_state.as_solution_view())
                        } else {
                            None
                        },
                        &stats,
                    );
                    if let SearchCommand::Terminate(reason) = cmd {
                        termination_reason = reason;
                        break 'outer;
                    }

                    continue 'inner;
                }

                stats.on_found_solution();

                let candidate_objective = self.candidate_state.objective();

                monitor.on_candidate_generated(
                    self.best_state.as_solution_view(),
                    self.accepted_state.as_solution_view(),
                    if has_buffered {
                        Some(self.buffered_state.as_solution_view())
                    } else {
                        None
                    },
                    candidate_objective,
                    &stats,
                );

                let decision = metaheuristic.decide_fate(
                    model,
                    self.best_state.as_solution_view(),
                    self.accepted_state.as_solution_view(),
                    if has_buffered {
                        Some(self.buffered_state.as_solution_view())
                    } else {
                        None
                    },
                    candidate_objective,
                    self.mutator.graph(),
                    self.mutator.diff(),
                );

                match decision {
                    AcceptanceOutcome::Accept => {
                        self.accept_candidate();
                        stats.on_accepted_solution();

                        // Notify accept (best_state still holds the old best).
                        metaheuristic.on_accept(
                            model,
                            self.best_state.as_solution_view(),
                            self.accepted_state.as_solution_view(),
                            if has_buffered {
                                Some(self.buffered_state.as_solution_view())
                            } else {
                                None
                            },
                            self.mutator.graph(),
                            self.mutator.diff(),
                        );
                        monitor.on_candidate_accepted(
                            self.best_state.as_solution_view(),
                            self.accepted_state.as_solution_view(),
                            if has_buffered {
                                Some(self.buffered_state.as_solution_view())
                            } else {
                                None
                            },
                            &stats,
                        );

                        // New global best?
                        if self.accepted_state.objective() < self.best_state.objective() {
                            // Notify BEFORE overwriting so the monitor sees the
                            // previous best in `best_state`.
                            monitor.on_best_solution_updated(
                                self.best_state.as_solution_view(),
                                self.accepted_state.as_solution_view(),
                                if has_buffered {
                                    Some(self.buffered_state.as_solution_view())
                                } else {
                                    None
                                },
                                self.accepted_state.as_solution_view(),
                                &stats,
                            );
                            self.update_best();
                            callback(self.best_state.as_solution_view());
                            metaheuristic.on_new_best(
                                model,
                                self.best_state.as_solution_view(),
                                self.mutator.graph(),
                                self.mutator.diff(),
                            );
                        }

                        // Accepting a new solution invalidates the buffer.
                        has_buffered = false;

                        metaheuristic.on_iteration(
                            stats.iterations,
                            model,
                            self.best_state.as_solution_view(),
                            self.accepted_state.as_solution_view(),
                            None,
                            self.mutator.graph(),
                        );

                        let cmd = monitor.search_command(
                            self.best_state.as_solution_view(),
                            self.accepted_state.as_solution_view(),
                            None,
                            &stats,
                        );
                        if let SearchCommand::Terminate(reason) = cmd {
                            termination_reason = reason;
                            break 'outer;
                        }

                        break 'inner;
                    }

                    AcceptanceOutcome::Buffer => {
                        self.save_candidate_to_buffer();
                        has_buffered = true;

                        monitor.on_solution_buffered(
                            self.best_state.as_solution_view(),
                            self.accepted_state.as_solution_view(),
                            self.buffered_state.as_solution_view(),
                            &stats,
                        );

                        // Roll back the topology graph so the next mutation
                        // starts from the accepted state.
                        self.mutator.rollback();

                        metaheuristic.on_iteration(
                            stats.iterations,
                            model,
                            self.best_state.as_solution_view(),
                            self.accepted_state.as_solution_view(),
                            Some(self.buffered_state.as_solution_view()),
                            self.mutator.graph(),
                        );

                        let cmd = monitor.search_command(
                            self.best_state.as_solution_view(),
                            self.accepted_state.as_solution_view(),
                            Some(self.buffered_state.as_solution_view()),
                            &stats,
                        );
                        if let SearchCommand::Terminate(reason) = cmd {
                            termination_reason = reason;
                            break 'outer;
                        }

                        continue 'inner;
                    }

                    AcceptanceOutcome::Reject => {
                        // Notify while the candidate topology is still live so
                        // the metaheuristic can inspect the diff / graph.
                        metaheuristic.on_reject(
                            model,
                            self.best_state.as_solution_view(),
                            self.accepted_state.as_solution_view(),
                            if has_buffered {
                                Some(self.buffered_state.as_solution_view())
                            } else {
                                None
                            },
                            candidate_objective,
                            self.mutator.graph(),
                            self.mutator.diff(),
                        );
                        monitor.on_candidate_rejected(
                            self.best_state.as_solution_view(),
                            self.accepted_state.as_solution_view(),
                            if has_buffered {
                                Some(self.buffered_state.as_solution_view())
                            } else {
                                None
                            },
                            candidate_objective,
                            &stats,
                        );

                        self.reject_candidate();

                        metaheuristic.on_iteration(
                            stats.iterations,
                            model,
                            self.best_state.as_solution_view(),
                            self.accepted_state.as_solution_view(),
                            if has_buffered {
                                Some(self.buffered_state.as_solution_view())
                            } else {
                                None
                            },
                            self.mutator.graph(),
                        );

                        let cmd = monitor.search_command(
                            self.best_state.as_solution_view(),
                            self.accepted_state.as_solution_view(),
                            if has_buffered {
                                Some(self.buffered_state.as_solution_view())
                            } else {
                                None
                            },
                            &stats,
                        );
                        if let SearchCommand::Terminate(reason) = cmd {
                            termination_reason = reason;
                            break 'outer;
                        }

                        continue 'inner;
                    }
                }
            }
        }

        stats.set_total_time(start_time.elapsed());

        monitor.on_end(self.best_state.as_solution_view(), &stats);
        metaheuristic.on_end(
            model,
            self.best_state.as_solution_view(),
            self.mutator.graph(),
        );

        let solution = Solution::from(self.best_state.as_solution_view());

        LocalSearchOutcome::new(solution, termination_reason, stats)
    }

    #[inline(always)]
    fn update_best(&mut self)
    where
        T: SolverNumeric,
    {
        self.best_state.overwrite_from_state(&self.accepted_state);
    }

    /// Accepts the current candidate immediately.
    /// This is the fastest commit path, skipping full array copies.
    #[inline(always)]
    fn accept_candidate(&mut self)
    where
        T: SolverNumeric,
    {
        unsafe {
            self.accepted_state.patch_from_delta_unchecked(
                &self.candidate_state,
                self.mutator.touched(),
                self.mutator.graph(),
            );
        }
        self.mutator.accept();
    }

    /// Rejects the current candidate and restores the graph to its previous state.
    #[inline(always)]
    fn reject_candidate(&mut self) {
        self.mutator.rollback();
    }

    /// Saves the current candidate to the buffer for later comparison.
    /// Used by Tabu Search to evaluate an entire neighborhood before committing.
    /// Also saves the current structural diff so it is available when the
    /// buffer is committed.
    #[inline(always)]
    fn save_candidate_to_buffer(&mut self)
    where
        T: SolverNumeric,
    {
        self.buffered_topology_graph
            .overwrite_from_graph(self.mutator.graph());
        self.buffered_schedule_graph_diff
            .overwrite_from_graph_diff(self.mutator.diff());
        self.buffered_state
            .overwrite_from_state(&self.accepted_state);

        unsafe {
            self.buffered_state.patch_from_delta_unchecked(
                &self.candidate_state,
                self.mutator.touched(),
                self.mutator.graph(),
            );
        }
    }

    /// Commits the buffered state, making it the new accepted state.
    #[inline(always)]
    pub fn accept_buffered(&mut self)
    where
        T: SolverNumeric,
    {
        self.accepted_state
            .overwrite_from_state(&self.buffered_state);
        self.mutator
            .overwrite_from_graph(&self.buffered_topology_graph);
    }

    /// Ensures all internal buffers and data structures are large enough to handle
    /// the given problem dimensions.
    ///
    /// ## Note
    ///
    /// This will leave most states in a partially uninitialized state,
    /// so it should only be called at the start of `run`.
    #[inline]
    fn ensure_capacity(&mut self, num_vessels: usize, num_berths: usize)
    where
        T: SolverNumeric,
    {
        self.accepted_state.resize(num_vessels, num_berths);
        self.candidate_state.resize(num_vessels, num_berths);
        self.buffered_state.resize(num_vessels, num_berths);
        self.best_state.resize(num_vessels, num_berths);

        // The mutator's graph and records, and the buffered graph, are sized by `overwrite_from_slices` and
        // `overwrite_from_graph`, which grow them as needed.
    }
}

// ----------------------------------------------------------------
// Tests
// ----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metaheuristic::acceptance::{
        AcceptanceOutcome, Metaheuristic, NeighborhoodExhaustionOutcome,
    };
    use crate::monitor::lsmonitor::LocalSearchMonitor;
    use crate::operator::LocalSearchOperator;
    use crate::parameters::MutableLocalSearchParams;
    use crate::statistics::LocalSearchStatistics;
    use crate::termination::{SearchCommand, TerminationReason};
    use std::cell::Cell;
    use std::sync::atomic::{AtomicU64, Ordering};
    use talos_core::ClosedOpenInterval;
    use talos_model::{BerthIndex, Model, SolutionView, VesselIndex};
    use typed_index_collections::TiSlice;

    /// Builds a minimal 2-vessel / 2-berth model.
    /// V0: arrival=0, deadline=100, weight=1, p(B0)=5, p(B1)=5
    /// V1: arrival=0, deadline=100, weight=1, p(B0)=10, p(B1)=10
    /// Both berths open [0, 200).
    fn build_test_model() -> Model<i64> {
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

    /// Builds a model with the specified number of vessels and berths.
    /// All vessels have arrival=0, deadline=1000, weight=1.
    /// All processing times are 10.
    /// All berths are open [0, 2000).
    fn build_test_model_with_size(num_vessels: usize, num_berths: usize) -> Model<i64> {
        let arrivals = vec![0; num_vessels];
        let deadlines = vec![1000; num_vessels];
        let weights = vec![1; num_vessels];
        let processing_times = vec![Some(10); num_vessels * num_berths];
        let time_windows = vec![vec![ClosedOpenInterval::new(0, 2000)]; num_berths];

        Model::new(arrivals, deadlines, weights, processing_times, time_windows)
    }

    /// A vessel's weight times its start, as its cost.
    #[derive(Debug)]
    struct WeightedStartCost;

    impl VesselCost<i64> for WeightedStartCost {
        unsafe fn cost_unchecked(
            model: &Model<i64>,
            vessel: VesselIndex,
            _berth: BerthIndex,
            start: i64,
        ) -> Option<i64> {
            Some(model.vessel_weight(vessel) * start)
        }
    }

    #[derive(Debug, Default)]
    struct MockMonitor {
        starts: u64,
        ends: u64,
        iterations: u64,
        candidates_generated: u64,
        candidates_accepted: u64,
        candidates_rejected: u64,
        candidates_infeasible: u64,
        solutions_buffered: u64,
        buffered_accepted: u64,
        neighborhoods_exhausted: u64,
        best_updates: u64,
    }

    impl LocalSearchMonitor<i64> for MockMonitor {
        fn name(&self) -> &str {
            "MockMonitor"
        }

        fn on_start(&mut self, _model: &Model<i64>, _initial: SolutionView<'_, i64>) {
            self.starts += 1;
        }

        fn on_end(&mut self, _best: SolutionView<'_, i64>, _stats: &LocalSearchStatistics) {
            self.ends += 1;
        }

        fn on_iteration(
            &mut self,
            _best: SolutionView<'_, i64>,
            _accepted: SolutionView<'_, i64>,
            _buffered: Option<SolutionView<'_, i64>>,
            _stats: &LocalSearchStatistics,
        ) {
            self.iterations += 1;
        }

        fn on_candidate_generated(
            &mut self,
            _best: SolutionView<'_, i64>,
            _accepted: SolutionView<'_, i64>,
            _buffered: Option<SolutionView<'_, i64>>,
            _candidate_objective: i64,
            _stats: &LocalSearchStatistics,
        ) {
            self.candidates_generated += 1;
        }

        fn on_candidate_accepted(
            &mut self,
            _best: SolutionView<'_, i64>,
            _accepted: SolutionView<'_, i64>,
            _buffered: Option<SolutionView<'_, i64>>,
            _stats: &LocalSearchStatistics,
        ) {
            self.candidates_accepted += 1;
        }

        fn on_candidate_rejected(
            &mut self,
            _best: SolutionView<'_, i64>,
            _accepted: SolutionView<'_, i64>,
            _buffered: Option<SolutionView<'_, i64>>,
            _rejected_objective: i64,
            _stats: &LocalSearchStatistics,
        ) {
            self.candidates_rejected += 1;
        }

        fn on_candidate_infeasible(
            &mut self,
            _best: SolutionView<'_, i64>,
            _accepted: SolutionView<'_, i64>,
            _buffered: Option<SolutionView<'_, i64>>,
            _stats: &LocalSearchStatistics,
        ) {
            self.candidates_infeasible += 1;
        }

        fn on_solution_buffered(
            &mut self,
            _best: SolutionView<'_, i64>,
            _accepted: SolutionView<'_, i64>,
            _buffered: SolutionView<'_, i64>,
            _stats: &LocalSearchStatistics,
        ) {
            self.solutions_buffered += 1;
        }

        fn on_buffered_solution_accepted(
            &mut self,
            _best: SolutionView<'_, i64>,
            _accepted: SolutionView<'_, i64>,
            _stats: &LocalSearchStatistics,
        ) {
            self.buffered_accepted += 1;
        }

        fn on_neighborhood_exhausted(
            &mut self,
            _best: SolutionView<'_, i64>,
            _accepted: SolutionView<'_, i64>,
            _buffered: Option<SolutionView<'_, i64>>,
            _stats: &LocalSearchStatistics,
        ) {
            self.neighborhoods_exhausted += 1;
        }

        fn on_best_solution_updated(
            &mut self,
            _prev_best: SolutionView<'_, i64>,
            _accepted: SolutionView<'_, i64>,
            _buffered: Option<SolutionView<'_, i64>>,
            _new_best: SolutionView<'_, i64>,
            _stats: &LocalSearchStatistics,
        ) {
            self.best_updates += 1;
        }
    }

    /// Produces `max_moves` swap(V0, V1) mutations, then reports exhaustion.
    /// All moves swap V0 and V1 within berth 0.
    struct MockOperator {
        max_moves: u64,
        cursor: u64,
    }

    impl MockOperator {
        fn new(max_moves: u64) -> Self {
            Self {
                max_moves,
                cursor: 0,
            }
        }
    }

    impl LocalSearchOperator<i64> for MockOperator {
        fn prepare(&mut self, _accepted_objective: i64, _graph: &ScheduleGraph) {
            self.cursor = 0;
        }

        unsafe fn next_neighbor_unchecked(
            &mut self,
            _model: &Model<i64>,
            mutator: &mut Mutator,
        ) -> bool {
            if self.cursor >= self.max_moves {
                return false;
            }
            self.cursor += 1;
            // Swap V0 and V1 (the only two vessels).
            mutator.swap_vessels(VesselIndex::new(0), VesselIndex::new(1));
            true
        }

        fn reset(&mut self) {
            self.cursor = 0;
        }
    }

    /// Always-accept metaheuristic that terminates after `max_iterations`.
    struct AcceptAllMetaheuristic {
        max_iterations: u64,
        on_iteration_calls: AtomicU64,
    }

    impl AcceptAllMetaheuristic {
        fn new(max_iterations: u64) -> Self {
            Self {
                max_iterations,
                on_iteration_calls: AtomicU64::new(0),
            }
        }
    }

    impl Metaheuristic<i64> for AcceptAllMetaheuristic {
        fn name(&self) -> &str {
            "AcceptAll"
        }

        fn on_start(
            &mut self,
            _model: &Model<i64>,
            _initial: SolutionView<'_, i64>,
            _graph: &ScheduleGraph,
        ) {
        }

        fn on_end(
            &mut self,
            _model: &Model<i64>,
            _final: SolutionView<'_, i64>,
            _graph: &ScheduleGraph,
        ) {
        }

        fn on_neighbourhood_exhausted(
            &mut self,
            _model: &Model<i64>,
            _best: SolutionView<'_, i64>,
            _accepted: SolutionView<'_, i64>,
            _buffered: Option<SolutionView<'_, i64>>,
            _graph: &ScheduleGraph,
        ) -> NeighborhoodExhaustionOutcome {
            NeighborhoodExhaustionOutcome::Terminate
        }

        fn should_commit_buffered(
            &mut self,
            _model: &Model<i64>,
            _best: SolutionView<'_, i64>,
            _accepted: SolutionView<'_, i64>,
            _buffered: Option<SolutionView<'_, i64>>,
            _layout: &ScheduleGraph,
            _buffer_layout: &ScheduleGraph,
        ) -> bool {
            false
        }

        fn search_command(
            &mut self,
            iteration: u64,
            _model: &Model<i64>,
            _best: SolutionView<'_, i64>,
            _accepted: SolutionView<'_, i64>,
            _buffered: Option<SolutionView<'_, i64>>,
        ) -> SearchCommand {
            if iteration >= self.max_iterations {
                SearchCommand::Terminate(TerminationReason::IterationLimitReached)
            } else {
                SearchCommand::Continue
            }
        }

        #[allow(clippy::too_many_arguments)]
        fn decide_fate(
            &mut self,
            _model: &Model<i64>,
            _best: SolutionView<'_, i64>,
            _accepted: SolutionView<'_, i64>,
            _buffered: Option<SolutionView<'_, i64>>,
            _candidate_objective: i64,
            _graph: &ScheduleGraph,
            _graph_diff: &ScheduleGraphDiff,
        ) -> AcceptanceOutcome {
            AcceptanceOutcome::Accept
        }

        fn on_accept(
            &mut self,
            _model: &Model<i64>,
            _best: SolutionView<'_, i64>,
            _new_accepted: SolutionView<'_, i64>,
            _buffered: Option<SolutionView<'_, i64>>,
            _graph: &ScheduleGraph,
            _graph_diff: &ScheduleGraphDiff,
        ) {
        }

        #[allow(clippy::too_many_arguments)]
        fn on_reject(
            &mut self,
            _model: &Model<i64>,
            _best: SolutionView<'_, i64>,
            _accepted: SolutionView<'_, i64>,
            _buffered: Option<SolutionView<'_, i64>>,
            _candidate_objective: i64,
            _graph: &ScheduleGraph,
            _graph_diff: &ScheduleGraphDiff,
        ) {
        }

        fn on_new_best(
            &mut self,
            _model: &Model<i64>,
            _new_best: SolutionView<'_, i64>,
            _graph: &ScheduleGraph,
            _graph_diff: &ScheduleGraphDiff,
        ) {
        }

        fn on_iteration(
            &mut self,
            _iteration: u64,
            _model: &Model<i64>,
            _best: SolutionView<'_, i64>,
            _accepted: SolutionView<'_, i64>,
            _buffered: Option<SolutionView<'_, i64>>,
            _graph: &ScheduleGraph,
        ) {
            self.on_iteration_calls.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Reject-all metaheuristic — rejects every candidate.
    struct RejectAllMetaheuristic {
        max_iterations: u64,
    }

    impl RejectAllMetaheuristic {
        fn new(max_iterations: u64) -> Self {
            Self { max_iterations }
        }
    }

    impl Metaheuristic<i64> for RejectAllMetaheuristic {
        fn name(&self) -> &str {
            "RejectAll"
        }

        fn on_start(
            &mut self,
            _model: &Model<i64>,
            _initial: SolutionView<'_, i64>,
            _graph: &ScheduleGraph,
        ) {
        }

        fn on_end(
            &mut self,
            _model: &Model<i64>,
            _final: SolutionView<'_, i64>,
            _graph: &ScheduleGraph,
        ) {
        }

        fn on_neighbourhood_exhausted(
            &mut self,
            _model: &Model<i64>,
            _best: SolutionView<'_, i64>,
            _accepted: SolutionView<'_, i64>,
            _buffered: Option<SolutionView<'_, i64>>,
            _graph: &ScheduleGraph,
        ) -> NeighborhoodExhaustionOutcome {
            NeighborhoodExhaustionOutcome::Terminate
        }

        fn should_commit_buffered(
            &mut self,
            _model: &Model<i64>,
            _best: SolutionView<'_, i64>,
            _accepted: SolutionView<'_, i64>,
            _buffered: Option<SolutionView<'_, i64>>,
            _layout: &ScheduleGraph,
            _buffer_layout: &ScheduleGraph,
        ) -> bool {
            false
        }

        fn search_command(
            &mut self,
            iteration: u64,
            _model: &Model<i64>,
            _best: SolutionView<'_, i64>,
            _accepted: SolutionView<'_, i64>,
            _buffered: Option<SolutionView<'_, i64>>,
        ) -> SearchCommand {
            if iteration >= self.max_iterations {
                SearchCommand::Terminate(TerminationReason::IterationLimitReached)
            } else {
                SearchCommand::Continue
            }
        }

        #[allow(clippy::too_many_arguments)]
        fn decide_fate(
            &mut self,
            _model: &Model<i64>,
            _best: SolutionView<'_, i64>,
            _accepted: SolutionView<'_, i64>,
            _buffered: Option<SolutionView<'_, i64>>,
            _candidate_objective: i64,
            _graph: &ScheduleGraph,
            _graph_diff: &ScheduleGraphDiff,
        ) -> AcceptanceOutcome {
            AcceptanceOutcome::Reject
        }

        fn on_accept(
            &mut self,
            _model: &Model<i64>,
            _best: SolutionView<'_, i64>,
            _new_accepted: SolutionView<'_, i64>,
            _buffered: Option<SolutionView<'_, i64>>,
            _graph: &ScheduleGraph,
            _graph_diff: &ScheduleGraphDiff,
        ) {
        }

        #[allow(clippy::too_many_arguments)]
        fn on_reject(
            &mut self,
            _model: &Model<i64>,
            _best: SolutionView<'_, i64>,
            _accepted: SolutionView<'_, i64>,
            _buffered: Option<SolutionView<'_, i64>>,
            _candidate_objective: i64,
            _graph: &ScheduleGraph,
            _graph_diff: &ScheduleGraphDiff,
        ) {
        }

        fn on_new_best(
            &mut self,
            _model: &Model<i64>,
            _new_best: SolutionView<'_, i64>,
            _graph: &ScheduleGraph,
            _graph_diff: &ScheduleGraphDiff,
        ) {
        }

        fn on_iteration(
            &mut self,
            _iteration: u64,
            _model: &Model<i64>,
            _best: SolutionView<'_, i64>,
            _accepted: SolutionView<'_, i64>,
            _buffered: Option<SolutionView<'_, i64>>,
            _graph: &ScheduleGraph,
        ) {
        }
    }

    /// Buffer-all metaheuristic — buffers every candidate, commits on exhaustion.
    struct BufferAllMetaheuristic {
        max_iterations: u64,
    }

    impl BufferAllMetaheuristic {
        fn new(max_iterations: u64) -> Self {
            Self { max_iterations }
        }
    }

    impl Metaheuristic<i64> for BufferAllMetaheuristic {
        fn name(&self) -> &str {
            "BufferAll"
        }

        fn on_start(
            &mut self,
            _model: &Model<i64>,
            _initial: SolutionView<'_, i64>,
            _graph: &ScheduleGraph,
        ) {
        }

        fn on_end(
            &mut self,
            _model: &Model<i64>,
            _final: SolutionView<'_, i64>,
            _graph: &ScheduleGraph,
        ) {
        }

        fn on_neighbourhood_exhausted(
            &mut self,
            _model: &Model<i64>,
            _best: SolutionView<'_, i64>,
            _accepted: SolutionView<'_, i64>,
            _buffered: Option<SolutionView<'_, i64>>,
            _graph: &ScheduleGraph,
        ) -> NeighborhoodExhaustionOutcome {
            NeighborhoodExhaustionOutcome::Terminate
        }

        fn should_commit_buffered(
            &mut self,
            _model: &Model<i64>,
            _best: SolutionView<'_, i64>,
            _accepted: SolutionView<'_, i64>,
            _buffered: Option<SolutionView<'_, i64>>,
            _layout: &ScheduleGraph,
            _buffer_layout: &ScheduleGraph,
        ) -> bool {
            true // always commit
        }

        fn search_command(
            &mut self,
            iteration: u64,
            _model: &Model<i64>,
            _best: SolutionView<'_, i64>,
            _accepted: SolutionView<'_, i64>,
            _buffered: Option<SolutionView<'_, i64>>,
        ) -> SearchCommand {
            if iteration >= self.max_iterations {
                SearchCommand::Terminate(TerminationReason::IterationLimitReached)
            } else {
                SearchCommand::Continue
            }
        }

        #[allow(clippy::too_many_arguments)]
        fn decide_fate(
            &mut self,
            _model: &Model<i64>,
            _best: SolutionView<'_, i64>,
            _accepted: SolutionView<'_, i64>,
            _buffered: Option<SolutionView<'_, i64>>,
            _candidate_objective: i64,
            _graph: &ScheduleGraph,
            _graph_diff: &ScheduleGraphDiff,
        ) -> AcceptanceOutcome {
            AcceptanceOutcome::Buffer
        }

        fn on_accept(
            &mut self,
            _model: &Model<i64>,
            _best: SolutionView<'_, i64>,
            _new_accepted: SolutionView<'_, i64>,
            _buffered: Option<SolutionView<'_, i64>>,
            _graph: &ScheduleGraph,
            _graph_diff: &ScheduleGraphDiff,
        ) {
        }

        #[allow(clippy::too_many_arguments)]
        fn on_reject(
            &mut self,
            _model: &Model<i64>,
            _best: SolutionView<'_, i64>,
            _accepted: SolutionView<'_, i64>,
            _buffered: Option<SolutionView<'_, i64>>,
            _candidate_objective: i64,
            _graph: &ScheduleGraph,
            _graph_diff: &ScheduleGraphDiff,
        ) {
        }

        fn on_new_best(
            &mut self,
            _model: &Model<i64>,
            _new_best: SolutionView<'_, i64>,
            _graph: &ScheduleGraph,
            _graph_diff: &ScheduleGraphDiff,
        ) {
        }

        fn on_iteration(
            &mut self,
            _iteration: u64,
            _model: &Model<i64>,
            _best: SolutionView<'_, i64>,
            _accepted: SolutionView<'_, i64>,
            _buffered: Option<SolutionView<'_, i64>>,
            _graph: &ScheduleGraph,
        ) {
        }
    }

    fn run_engine<H: Metaheuristic<i64>, O: LocalSearchOperator<i64>>(
        metaheuristic: &mut H,
        operator: &mut O,
        monitor: MockMonitor,
    ) -> (LocalSearchOutcome<i64>, MockMonitor) {
        let model = build_test_model();
        // Initial solution: V0→B0, V1→B0, positions give ordering V0 then V1.
        let berths = [BerthIndex::new(0), BerthIndex::new(0)];
        let start_times = [0_i64, 5_i64];

        // We need to get the monitor back out after the run. Wrap it so we can
        // extract it later. Since `run` takes `M: LocalSearchMonitor` by value
        // inside `MutableLocalSearchParams`, we pass it directly and rely on
        // the return.
        let params = MutableLocalSearchParams {
            model: &model,
            operator,
            metaheuristic,
            monitor,
            berths: TiSlice::from_ref(&berths),
            start_times: TiSlice::from_ref(&start_times),
            objective_value: 5, // weight=1 * start=0 + weight=1 * start=5
        };

        let mut engine = Engine::<i64>::new(2, 2);
        let mut best_callbacks = 0u32;
        let outcome = engine.run_with_cost::<WeightedStartCost, _, _, _, _>(params, |_| {
            best_callbacks += 1;
        });
        // We cannot get the monitor back from the engine, so we reconstruct
        // assertions from the outcome stats. For monitor-level assertions we
        // use a Cell-based approach in specific tests.
        // NOTE: This helper is for stats/outcome assertions only.
        // For monitor assertions, use `run_engine_with_monitor_cell`.
        (outcome, MockMonitor::default())
    }

    /// Like `run_engine` but uses a `&Cell`-based monitor so we can inspect
    /// the monitor's state after the run.
    struct CellMonitor<'a>(&'a Cell<MockMonitor>);

    impl<'a> LocalSearchMonitor<i64> for CellMonitor<'a> {
        fn name(&self) -> &str {
            "CellMonitor"
        }

        fn on_start(&mut self, model: &Model<i64>, initial: SolutionView<'_, i64>) {
            let mut m = self.0.take();
            m.on_start(model, initial);
            self.0.set(m);
        }

        fn on_end(&mut self, best: SolutionView<'_, i64>, stats: &LocalSearchStatistics) {
            let mut m = self.0.take();
            m.on_end(best, stats);
            self.0.set(m);
        }

        fn on_iteration(
            &mut self,
            best: SolutionView<'_, i64>,
            accepted: SolutionView<'_, i64>,
            buffered: Option<SolutionView<'_, i64>>,
            stats: &LocalSearchStatistics,
        ) {
            let mut m = self.0.take();
            m.on_iteration(best, accepted, buffered, stats);
            self.0.set(m);
        }

        fn on_candidate_generated(
            &mut self,
            best: SolutionView<'_, i64>,
            accepted: SolutionView<'_, i64>,
            buffered: Option<SolutionView<'_, i64>>,
            candidate_objective: i64,
            stats: &LocalSearchStatistics,
        ) {
            let mut m = self.0.take();
            m.on_candidate_generated(best, accepted, buffered, candidate_objective, stats);
            self.0.set(m);
        }

        fn on_candidate_accepted(
            &mut self,
            best: SolutionView<'_, i64>,
            accepted: SolutionView<'_, i64>,
            buffered: Option<SolutionView<'_, i64>>,
            stats: &LocalSearchStatistics,
        ) {
            let mut m = self.0.take();
            m.on_candidate_accepted(best, accepted, buffered, stats);
            self.0.set(m);
        }

        fn on_candidate_rejected(
            &mut self,
            best: SolutionView<'_, i64>,
            accepted: SolutionView<'_, i64>,
            buffered: Option<SolutionView<'_, i64>>,
            rejected_objective: i64,
            stats: &LocalSearchStatistics,
        ) {
            let mut m = self.0.take();
            m.on_candidate_rejected(best, accepted, buffered, rejected_objective, stats);
            self.0.set(m);
        }

        fn on_candidate_infeasible(
            &mut self,
            best: SolutionView<'_, i64>,
            accepted: SolutionView<'_, i64>,
            buffered: Option<SolutionView<'_, i64>>,
            stats: &LocalSearchStatistics,
        ) {
            let mut m = self.0.take();
            m.on_candidate_infeasible(best, accepted, buffered, stats);
            self.0.set(m);
        }

        fn on_solution_buffered(
            &mut self,
            best: SolutionView<'_, i64>,
            accepted: SolutionView<'_, i64>,
            buffered: SolutionView<'_, i64>,
            stats: &LocalSearchStatistics,
        ) {
            let mut m = self.0.take();
            m.on_solution_buffered(best, accepted, buffered, stats);
            self.0.set(m);
        }

        fn on_buffered_solution_accepted(
            &mut self,
            best: SolutionView<'_, i64>,
            accepted: SolutionView<'_, i64>,
            stats: &LocalSearchStatistics,
        ) {
            let mut m = self.0.take();
            m.on_buffered_solution_accepted(best, accepted, stats);
            self.0.set(m);
        }

        fn on_neighborhood_exhausted(
            &mut self,
            best: SolutionView<'_, i64>,
            accepted: SolutionView<'_, i64>,
            buffered: Option<SolutionView<'_, i64>>,
            stats: &LocalSearchStatistics,
        ) {
            let mut m = self.0.take();
            m.on_neighborhood_exhausted(best, accepted, buffered, stats);
            self.0.set(m);
        }

        fn on_best_solution_updated(
            &mut self,
            prev_best: SolutionView<'_, i64>,
            accepted: SolutionView<'_, i64>,
            buffered: Option<SolutionView<'_, i64>>,
            new_best: SolutionView<'_, i64>,
            stats: &LocalSearchStatistics,
        ) {
            let mut m = self.0.take();
            m.on_best_solution_updated(prev_best, accepted, buffered, new_best, stats);
            self.0.set(m);
        }
    }

    fn run_with_cell_monitor<H: Metaheuristic<i64>, O: LocalSearchOperator<i64>>(
        metaheuristic: &mut H,
        operator: &mut O,
        cell: &Cell<MockMonitor>,
    ) -> LocalSearchOutcome<i64> {
        let model = build_test_model();
        let berths = [BerthIndex::new(0), BerthIndex::new(0)];
        let start_times = [0_i64, 5_i64];

        let params = MutableLocalSearchParams {
            model: &model,
            operator,
            metaheuristic,
            monitor: CellMonitor(cell),
            berths: TiSlice::from_ref(&berths),
            start_times: TiSlice::from_ref(&start_times),
            objective_value: 5,
        };

        let mut engine = Engine::<i64>::new(2, 2);
        engine.run_with_cost::<WeightedStartCost, _, _, _, _>(params, |_| {})
    }

    // A schedule of three vessels for a model of two would build a graph whose third vessel every unchecked read
    // of the model takes for one of its own; the run must refuse it at its door, in every build.
    #[test]
    #[should_panic(
        expected = "called `Engine::run` with vessel count mismatch: model has 2, but initial solution has 3"
    )]
    fn test_engine_run_refuses_a_schedule_of_more_vessels_than_the_model() {
        let model = build_test_model();
        let berths = [BerthIndex::new(0); 3];
        let start_times = [0_i64, 5, 15];
        let params = MutableLocalSearchParams {
            model: &model,
            operator: &mut MockOperator::new(1),
            metaheuristic: &mut AcceptAllMetaheuristic::new(1),
            monitor: MockMonitor::default(),
            berths: TiSlice::from_ref(&berths),
            start_times: TiSlice::from_ref(&start_times),
            objective_value: 0,
        };
        let _ =
            Engine::<i64>::new(3, 2).run_with_cost::<WeightedStartCost, _, _, _, _>(params, |_| {});
    }

    #[test]
    fn test_engine_run_refuses_a_berth_beyond_the_model() {
        let model = build_test_model();
        let berths = [BerthIndex::new(0), BerthIndex::new(2)];
        let start_times = [0_i64, 5];
        let params = MutableLocalSearchParams {
            model: &model,
            operator: &mut MockOperator::new(1),
            metaheuristic: &mut AcceptAllMetaheuristic::new(1),
            monitor: MockMonitor::default(),
            berths: TiSlice::from_ref(&berths),
            start_times: TiSlice::from_ref(&start_times),
            objective_value: 0,
        };
        let refusal = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            Engine::<i64>::new(2, 2).run_with_cost::<WeightedStartCost, _, _, _, _>(params, |_| {})
        }))
        .expect_err("a berth beyond the model should be refused");
        let message = refusal
            .downcast_ref::<String>()
            .expect("the refusal should carry a formatted message");
        assert!(
            message.starts_with(
                "called `Engine::run` with initial solution contains an out-of-bounds berth"
            ),
            "unexpected refusal: {message}"
        );
    }

    #[test]
    fn test_engine_terminates_immediately_at_zero_iterations() {
        let mut meta = AcceptAllMetaheuristic::new(0);
        let mut op = MockOperator::new(10);
        let (outcome, _) = run_engine(&mut meta, &mut op, MockMonitor::default());

        assert_eq!(
            outcome.termination_reason(),
            TerminationReason::IterationLimitReached
        );
        assert_eq!(outcome.stats().iterations, 0);
        assert_eq!(outcome.stats().cycles, 0);
    }

    #[test]
    fn test_engine_accept_all_counts_iterations() {
        let cell = Cell::new(MockMonitor::default());
        let mut meta = AcceptAllMetaheuristic::new(5);
        let mut op = MockOperator::new(100); // more than enough

        let outcome = run_with_cell_monitor(&mut meta, &mut op, &cell);

        assert_eq!(outcome.stats().iterations, 5);
        assert_eq!(outcome.stats().accepted_solutions, 5);
        assert_eq!(outcome.stats().total_solutions, 5);
        assert_eq!(outcome.stats().infeasible_moves, 0);

        let mon = cell.take();
        assert_eq!(mon.starts, 1);
        assert_eq!(mon.ends, 1);
        assert_eq!(mon.candidates_generated, 5);
        assert_eq!(mon.candidates_accepted, 5);
        assert_eq!(mon.candidates_rejected, 0);
    }

    #[test]
    fn test_engine_reject_all_counts_correctly() {
        let cell = Cell::new(MockMonitor::default());
        // Operator serves 3 moves then exhausts → cycle.
        let mut meta = RejectAllMetaheuristic::new(100);
        let mut op = MockOperator::new(3);

        let outcome = run_with_cell_monitor(&mut meta, &mut op, &cell);

        // All 3 moves generated + rejected, then neighbourhood exhausted → terminate.
        assert_eq!(outcome.stats().iterations, 3);
        assert_eq!(outcome.stats().total_solutions, 3);
        assert_eq!(outcome.stats().accepted_solutions, 0);
        assert_eq!(outcome.stats().cycles, 1);
        assert_eq!(
            outcome.termination_reason(),
            TerminationReason::NeighborhoodExhausted
        );

        let mon = cell.take();
        assert_eq!(mon.candidates_generated, 3);
        assert_eq!(mon.candidates_rejected, 3);
        assert_eq!(mon.candidates_accepted, 0);
        assert_eq!(mon.neighborhoods_exhausted, 1);
    }

    #[test]
    fn test_engine_buffer_all_commits_on_exhaustion() {
        let cell = Cell::new(MockMonitor::default());
        let mut meta = BufferAllMetaheuristic::new(100);
        let mut op = MockOperator::new(2);

        let outcome = run_with_cell_monitor(&mut meta, &mut op, &cell);

        // 2 moves buffered → exhaustion → commit buffered → terminate.
        assert_eq!(outcome.stats().iterations, 2);
        assert_eq!(outcome.stats().total_solutions, 2);
        assert_eq!(outcome.stats().cycles, 1);
        // The buffered solution is committed during exhaustion handling.
        assert_eq!(outcome.stats().accepted_solutions, 1);
        assert_eq!(
            outcome.termination_reason(),
            TerminationReason::NeighborhoodExhausted
        );

        let mon = cell.take();
        assert_eq!(mon.solutions_buffered, 2);
        assert_eq!(mon.buffered_accepted, 1);
        assert_eq!(mon.neighborhoods_exhausted, 1);
    }

    #[test]
    fn test_engine_on_iteration_called_for_every_try() {
        // AcceptAll tracks on_iteration call count via AtomicU64.
        let mut meta = AcceptAllMetaheuristic::new(4);
        let mut op = MockOperator::new(100);

        let (outcome, _) = run_engine(&mut meta, &mut op, MockMonitor::default());

        assert_eq!(outcome.stats().iterations, 4);
        // on_iteration should have been called once per iteration.
        assert_eq!(meta.on_iteration_calls.load(Ordering::Relaxed), 4);
    }

    #[test]
    fn test_engine_on_iteration_called_on_reject() {
        // RejectAll + 3 moves → 3 rejected tries → 3 on_iteration calls.
        // We wrap RejectAll with an AtomicU64 counter.
        struct CountingRejectAll {
            inner: RejectAllMetaheuristic,
            on_iteration_calls: AtomicU64,
        }

        impl Metaheuristic<i64> for CountingRejectAll {
            fn name(&self) -> &str {
                self.inner.name()
            }
            fn on_start(&mut self, m: &Model<i64>, s: SolutionView<'_, i64>, g: &ScheduleGraph) {
                self.inner.on_start(m, s, g);
            }
            fn on_end(&mut self, m: &Model<i64>, s: SolutionView<'_, i64>, g: &ScheduleGraph) {
                self.inner.on_end(m, s, g);
            }
            fn on_neighbourhood_exhausted(
                &mut self,
                m: &Model<i64>,
                b: SolutionView<'_, i64>,
                a: SolutionView<'_, i64>,
                bf: Option<SolutionView<'_, i64>>,
                g: &ScheduleGraph,
            ) -> NeighborhoodExhaustionOutcome {
                self.inner.on_neighbourhood_exhausted(m, b, a, bf, g)
            }
            fn should_commit_buffered(
                &mut self,
                m: &Model<i64>,
                b: SolutionView<'_, i64>,
                a: SolutionView<'_, i64>,
                bf: Option<SolutionView<'_, i64>>,
                l: &ScheduleGraph,
                bl: &ScheduleGraph,
            ) -> bool {
                self.inner.should_commit_buffered(m, b, a, bf, l, bl)
            }
            fn search_command(
                &mut self,
                i: u64,
                m: &Model<i64>,
                b: SolutionView<'_, i64>,
                a: SolutionView<'_, i64>,
                bf: Option<SolutionView<'_, i64>>,
            ) -> SearchCommand {
                self.inner.search_command(i, m, b, a, bf)
            }
            #[allow(clippy::too_many_arguments)]
            fn decide_fate(
                &mut self,
                m: &Model<i64>,
                b: SolutionView<'_, i64>,
                a: SolutionView<'_, i64>,
                bf: Option<SolutionView<'_, i64>>,
                co: i64,
                g: &ScheduleGraph,
                gd: &ScheduleGraphDiff,
            ) -> AcceptanceOutcome {
                self.inner.decide_fate(m, b, a, bf, co, g, gd)
            }
            fn on_accept(
                &mut self,
                m: &Model<i64>,
                b: SolutionView<'_, i64>,
                na: SolutionView<'_, i64>,
                bf: Option<SolutionView<'_, i64>>,
                g: &ScheduleGraph,
                gd: &ScheduleGraphDiff,
            ) {
                self.inner.on_accept(m, b, na, bf, g, gd);
            }
            #[allow(clippy::too_many_arguments)]
            fn on_reject(
                &mut self,
                m: &Model<i64>,
                b: SolutionView<'_, i64>,
                a: SolutionView<'_, i64>,
                bf: Option<SolutionView<'_, i64>>,
                co: i64,
                g: &ScheduleGraph,
                gd: &ScheduleGraphDiff,
            ) {
                self.inner.on_reject(m, b, a, bf, co, g, gd);
            }
            fn on_new_best(
                &mut self,
                m: &Model<i64>,
                nb: SolutionView<'_, i64>,
                g: &ScheduleGraph,
                gd: &ScheduleGraphDiff,
            ) {
                self.inner.on_new_best(m, nb, g, gd);
            }
            fn on_iteration(
                &mut self,
                i: u64,
                m: &Model<i64>,
                b: SolutionView<'_, i64>,
                a: SolutionView<'_, i64>,
                bf: Option<SolutionView<'_, i64>>,
                g: &ScheduleGraph,
            ) {
                self.on_iteration_calls.fetch_add(1, Ordering::Relaxed);
                self.inner.on_iteration(i, m, b, a, bf, g);
            }
        }

        let mut meta = CountingRejectAll {
            inner: RejectAllMetaheuristic::new(100),
            on_iteration_calls: AtomicU64::new(0),
        };
        let mut op = MockOperator::new(3);
        let (outcome, _) = run_engine(&mut meta, &mut op, MockMonitor::default());

        assert_eq!(outcome.stats().iterations, 3);
        assert_eq!(meta.on_iteration_calls.load(Ordering::Relaxed), 4);
    }

    #[test]
    fn test_engine_returns_valid_solution() {
        let mut meta = AcceptAllMetaheuristic::new(1);
        let mut op = MockOperator::new(10);
        let (outcome, _) = run_engine(&mut meta, &mut op, MockMonitor::default());

        let sol = outcome.solution();
        assert_eq!(sol.num_vessels(), 2);
        // The objective must be non-negative for this model.
        assert!(sol.objective_value() >= 0);
    }

    // Best-improvement descent from a schedule whose only move worsens it stands at a local optimum: the first scan
    // finds nothing to commit, and the search must end there instead of scanning again until a budget runs out.
    #[test]
    fn test_best_improvement_descent_ends_at_a_local_optimum() {
        use crate::metaheuristic::{
            greedy_descent::GreedyDescent, move_selection::SelectionStrategy,
        };
        use crate::monitor::iteration::IterationLimitMonitor;
        let model = build_test_model();
        let berths = [BerthIndex::new(0), BerthIndex::new(0)];
        let start_times = [0_i64, 5];
        let mut descent = GreedyDescent::new().with_selection(SelectionStrategy::BestImprovement);
        let params = MutableLocalSearchParams {
            model: &model,
            operator: &mut MockOperator::new(1),
            metaheuristic: &mut descent,
            monitor: IterationLimitMonitor::new(1000),
            berths: TiSlice::from_ref(&berths),
            start_times: TiSlice::from_ref(&start_times),
            objective_value: 5,
        };
        let outcome =
            Engine::<i64>::new(2, 2).run_with_cost::<WeightedStartCost, _, _, _, _>(params, |_| {});
        assert_eq!(
            outcome.termination_reason(),
            TerminationReason::NeighborhoodExhausted
        );
        assert_eq!(outcome.stats().iterations, 1);
        assert_eq!(outcome.solution().objective_value(), 5);
    }

    // A schedule with no neighbour at all cannot change: every scan of its neighbourhood is empty, whatever the
    // metaheuristic makes of the exhaustion. The search must end at the first empty scan instead of restarting
    // until a budget runs out, and a budget counted in candidates would never run out.
    #[test]
    fn test_engine_ends_at_a_scan_without_candidates() {
        use crate::metaheuristic::guided_local_search::GuidedLocalSearch;
        use crate::monitor::cycle::CycleLimitMonitor;
        let model = build_test_model();
        let berths = [BerthIndex::new(0), BerthIndex::new(0)];
        let start_times = [0_i64, 5];
        let mut gls = GuidedLocalSearch::new(2, 2);
        let params = MutableLocalSearchParams {
            model: &model,
            operator: &mut MockOperator::new(0),
            metaheuristic: &mut gls,
            monitor: CycleLimitMonitor::new(1000),
            berths: TiSlice::from_ref(&berths),
            start_times: TiSlice::from_ref(&start_times),
            objective_value: 5,
        };
        let outcome =
            Engine::<i64>::new(2, 2).run_with_cost::<WeightedStartCost, _, _, _, _>(params, |_| {});
        assert_eq!(
            outcome.termination_reason(),
            TerminationReason::NeighborhoodExhausted
        );
        assert_eq!(outcome.stats().cycles, 1);
    }

    #[test]
    fn test_engine_neighborhood_exhaustion_terminates() {
        // Operator produces only 1 move, accept-all meta, so cycle happens after 1 move.
        let mut meta = AcceptAllMetaheuristic::new(100);
        let mut op = MockOperator::new(1);
        let (outcome, _) = run_engine(&mut meta, &mut op, MockMonitor::default());

        // After accept, outer loop re-enters, operator.prepare resets cursor.
        // But operator only serves 1 move per prepare. After the first accept,
        // the second iteration accepts again, and so on until max_iterations.
        // Actually: prepare resets cursor to 0, so it serves 1 move each outer
        // loop pass. The meta terminates at 100 iterations.
        assert_eq!(outcome.stats().iterations, 100);
        assert_eq!(
            outcome.termination_reason(),
            TerminationReason::IterationLimitReached
        );
    }

    #[test]
    fn test_engine_stats_time_is_nonzero() {
        let mut meta = AcceptAllMetaheuristic::new(10);
        let mut op = MockOperator::new(100);
        let (outcome, _) = run_engine(&mut meta, &mut op, MockMonitor::default());

        assert!(outcome.stats().time_total.as_nanos() > 0);
    }

    /// A size-aware mock operator that swaps the first two vessels if they exist.
    struct SizeAwareMockOperator {
        max_moves: u64,
        cursor: u64,
        num_vessels: usize,
    }

    impl SizeAwareMockOperator {
        fn new(max_moves: u64) -> Self {
            Self {
                max_moves,
                cursor: 0,
                num_vessels: 0,
            }
        }
    }

    impl LocalSearchOperator<i64> for SizeAwareMockOperator {
        fn prepare(&mut self, _accepted_objective: i64, graph: &ScheduleGraph) {
            self.cursor = 0;
            self.num_vessels = graph.num_vessels();
        }

        unsafe fn next_neighbor_unchecked(
            &mut self,
            _model: &Model<i64>,
            mutator: &mut Mutator,
        ) -> bool {
            if self.cursor >= self.max_moves || self.num_vessels < 2 {
                return false;
            }
            self.cursor += 1;
            // Swap the first two vessels (safe for any model with 2+ vessels).
            mutator.swap_vessels(VesselIndex::new(0), VesselIndex::new(1));
            true
        }

        fn reset(&mut self) {
            self.cursor = 0;
        }
    }

    #[test]
    fn test_engine_reuse_with_different_model_sizes() {
        // Test that the same engine can be reused with models of different sizes
        // going from small → large → small to ensure internal buffers resize correctly.
        let mut engine = Engine::<i64>::new(2, 2);

        // Test 1: Small model (2 vessels, 2 berths)
        {
            let model = build_test_model_with_size(2, 2);
            let berths = [BerthIndex::new(0), BerthIndex::new(1)];
            let start_times = [0_i64, 0_i64];

            let mut meta = AcceptAllMetaheuristic::new(5);
            let mut op = SizeAwareMockOperator::new(10);
            let monitor = MockMonitor::default();

            let params = MutableLocalSearchParams {
                model: &model,
                operator: &mut op,
                metaheuristic: &mut meta,
                monitor,
                berths: TiSlice::from_ref(&berths),
                start_times: TiSlice::from_ref(&start_times),
                objective_value: 0,
            };

            let outcome = engine.run_with_cost::<WeightedStartCost, _, _, _, _>(params, |_| {});
            assert_eq!(outcome.solution().num_vessels(), 2);
            assert!(outcome.stats().iterations > 0);
        }

        // Test 2: Large model (10 vessels, 5 berths) - reuse same engine
        {
            let model = build_test_model_with_size(10, 5);
            let berths = [
                BerthIndex::new(0),
                BerthIndex::new(1),
                BerthIndex::new(2),
                BerthIndex::new(3),
                BerthIndex::new(4),
                BerthIndex::new(0),
                BerthIndex::new(1),
                BerthIndex::new(2),
                BerthIndex::new(3),
                BerthIndex::new(4),
            ];
            let start_times = [0_i64, 0, 0, 0, 0, 10, 10, 10, 10, 10];

            let mut meta = AcceptAllMetaheuristic::new(5);
            let mut op = SizeAwareMockOperator::new(20);
            let monitor = MockMonitor::default();

            let params = MutableLocalSearchParams {
                model: &model,
                operator: &mut op,
                metaheuristic: &mut meta,
                monitor,
                berths: TiSlice::from_ref(&berths),
                start_times: TiSlice::from_ref(&start_times),
                objective_value: 0,
            };

            let outcome = engine.run_with_cost::<WeightedStartCost, _, _, _, _>(params, |_| {});
            assert_eq!(outcome.solution().num_vessels(), 10);
            assert!(outcome.stats().iterations > 0);
        }

        // Test 3: Back to small model (3 vessels, 2 berths) - ensure no issues
        {
            let model = build_test_model_with_size(3, 2);
            let berths = [BerthIndex::new(0), BerthIndex::new(1), BerthIndex::new(0)];
            let start_times = [0_i64, 0, 10];

            let mut meta = AcceptAllMetaheuristic::new(5);
            let mut op = SizeAwareMockOperator::new(15);
            let monitor = MockMonitor::default();

            let params = MutableLocalSearchParams {
                model: &model,
                operator: &mut op,
                metaheuristic: &mut meta,
                monitor,
                berths: TiSlice::from_ref(&berths),
                start_times: TiSlice::from_ref(&start_times),
                objective_value: 0,
            };

            let outcome = engine.run_with_cost::<WeightedStartCost, _, _, _, _>(params, |_| {});
            assert_eq!(outcome.solution().num_vessels(), 3);
            assert!(outcome.stats().iterations > 0);
        }

        // Test 4: Very large model (50 vessels, 10 berths)
        {
            let model = build_test_model_with_size(50, 10);
            let berths: Vec<BerthIndex> = (0..50).map(|i| BerthIndex::new(i % 10)).collect();
            let start_times: Vec<i64> = (0..50).map(|i| (i / 10) * 10).collect();

            let mut meta = AcceptAllMetaheuristic::new(3);
            let mut op = SizeAwareMockOperator::new(10);
            let monitor = MockMonitor::default();

            let params = MutableLocalSearchParams {
                model: &model,
                operator: &mut op,
                metaheuristic: &mut meta,
                monitor,
                berths: TiSlice::from_ref(&berths),
                start_times: TiSlice::from_ref(&start_times),
                objective_value: 0,
            };

            let outcome = engine.run_with_cost::<WeightedStartCost, _, _, _, _>(params, |_| {});
            assert_eq!(outcome.solution().num_vessels(), 50);
            assert!(outcome.stats().iterations > 0);
        }

        // Test 5: Back to smallest (2 vessels, 1 berth)
        // Note: We use 2 vessels since the SizeAwareMockOperator requires at least 2 vessels to swap
        {
            let model = build_test_model_with_size(2, 1);
            let berths = [BerthIndex::new(0), BerthIndex::new(0)];
            let start_times = [0_i64, 10];

            let mut meta = AcceptAllMetaheuristic::new(5);
            let mut op = SizeAwareMockOperator::new(5);
            let monitor = MockMonitor::default();

            let params = MutableLocalSearchParams {
                model: &model,
                operator: &mut op,
                metaheuristic: &mut meta,
                monitor,
                berths: TiSlice::from_ref(&berths),
                start_times: TiSlice::from_ref(&start_times),
                objective_value: 0,
            };

            let outcome = engine.run_with_cost::<WeightedStartCost, _, _, _, _>(params, |_| {});
            assert_eq!(outcome.solution().num_vessels(), 2);
            assert!(outcome.stats().iterations > 0);
        }
    }
}
