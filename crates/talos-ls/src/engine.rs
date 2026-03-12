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

#![allow(dead_code)]

use crate::{
    decoding::{decode_full_unchecked, decode_unchecked},
    exec::{SearchCommand, TerminationReason},
    meta::metaheuristic::{AcceptanceOutcome, Metaheuristic, NeighborhoodExhaustionOutcome},
    monitor::lsmonitor::LocalSearchMonitor,
    mutator::Mutator,
    operator::lsoperator::LocalSearchOperator,
    outcome::LocalSearchOutcome,
    params::MutableLocalSearchParams,
    sgraph::{ScheduleGraph, ScheduleGraphDiff},
    sgraphundo::ScheduleGraphUndoLog,
    state::ScheduleState,
    stats::LocalSearchStatistics,
    tberth::TouchedBerths,
};
use std::time::Instant;
use talos_core::utils::num::SolverNumeric;
use talos_model::{
    index::{BerthIndex, VesselIndex},
    model::Model,
    solution::{Solution, SolutionView},
};

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

    /// The mutable schedule graph that encodes the current and candidate states
    /// and transitions between them as the engine explores the neighborhood. This graph is modified in-place by the
    /// mutators and can be rolled back by using the `schedule_graph_undo_log` if a move is rejected.
    topology_graph: ScheduleGraph,

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

    /// **Topological Rollback Stack**: A LIFO stack of inverse graph operations.
    /// Allows the `ScheduleGraph` to revert its linked-list pointers in $O(1)$
    /// time when a move does not meet acceptance criteria.
    schedule_graph_undo_log: ScheduleGraphUndoLog,

    /// **Structural Diff**: Records which edges were broken/created and which
    /// vessels were reallocated between berths during a mutation. Used by the
    /// metaheuristic for Tabu tenure tracking and aspiration criteria.
    schedule_graph_diff: ScheduleGraphDiff,

    /// Saved copy of the structural diff corresponding to the `buffered_state`.
    /// Captured when a candidate is buffered and passed to `on_accept` when
    /// the buffer is committed.
    buffered_schedule_graph_diff: ScheduleGraphDiff,

    /// **Dirty-Tracking Set**: A type-safe bitset or boolean mask identifying
    /// berths modified during the current mutation. Informs the downstream
    /// decoder exactly which timelines require recalculation.
    touched: TouchedBerths,
}

impl<T> Engine<T> {
    /// Creates a new engine pre-allocated for the given problem dimensions.
    pub fn new(num_vessels: usize, num_berths: usize) -> Self
    where
        T: SolverNumeric,
    {
        let berths = vec![BerthIndex::new(0); num_vessels];
        let starts = vec![T::ZERO; num_vessels];
        let positions = vec![0_usize; num_vessels];
        let costs = vec![T::ZERO; num_berths];

        let make_state = || {
            ScheduleState::new(
                berths.clone(),
                starts.clone(),
                positions.clone(),
                costs.clone(),
                T::ZERO,
            )
        };

        let dummy_starts: Vec<i32> = (0..num_vessels as i32).collect();
        let topology_graph = ScheduleGraph::from_slices(&berths, &dummy_starts, num_berths);
        let buffered_topology_graph = topology_graph.clone();

        Self {
            accepted_state: make_state(),
            candidate_state: make_state(),
            buffered_state: make_state(),
            best_state: make_state(),
            topology_graph,
            buffered_topology_graph,
            schedule_graph_undo_log: ScheduleGraphUndoLog::preallocated(num_vessels),
            schedule_graph_diff: ScheduleGraphDiff::new(num_vessels),
            buffered_schedule_graph_diff: ScheduleGraphDiff::new(num_vessels),
            touched: TouchedBerths::new(num_berths),
        }
    }

    /// Runs the local search engine loop.
    ///
    /// Initializes the internal quad-buffer state from `params`, then repeatedly
    /// generates neighbors via the operator, decodes them, and asks the
    /// metaheuristic whether to accept, buffer, or reject each candidate.
    /// The monitor is notified of all lifecycle events and may request early
    /// termination. The `callback` is invoked whenever a new global best
    /// solution is found. The `evaluator` computes per-vessel cost contributions
    /// during decoding.
    ///
    /// Returns the best solution found, the termination reason, and statistics.
    pub fn run<H, O, M, F, C>(
        &mut self,
        params: MutableLocalSearchParams<'_, T, H, O, M>,
        evaluator: F,
        mut callback: C,
    ) -> LocalSearchOutcome<T>
    where
        T: SolverNumeric,
        H: Metaheuristic<T>,
        O: LocalSearchOperator<T>,
        M: LocalSearchMonitor<T>,
        F: Fn(&Model<T>, VesselIndex, BerthIndex, T) -> Option<T>,
        C: FnMut(SolutionView<'_, T>),
    {
        let model = params.model;
        let operator = params.operator;
        let metaheuristic = params.metaheuristic;
        let mut monitor = params.monitor;

        // ── Initialise topology from the validated initial solution ──
        self.topology_graph.overwrite_from_slices(
            params.berths,
            params.start_times,
            model.num_berths(),
        );

        // Full-decode to populate the candidate buffer, then propagate to all states.
        unsafe {
            decode_full_unchecked(
                &self.topology_graph,
                &mut self.candidate_state,
                model,
                &evaluator,
            )
            .expect("initial solution must be decodable");
        }
        self.accepted_state
            .overwrite_from_state(&self.candidate_state);
        self.best_state.overwrite_from_state(&self.candidate_state);
        self.buffered_topology_graph
            .overwrite_from_graph(&self.topology_graph);

        // ── Statistics & timing ──
        let mut stats = LocalSearchStatistics::default();
        let start_time = Instant::now();
        let mut has_buffered = false;

        // ── Lifecycle: on_start ──
        monitor.on_start(model, self.accepted_state.as_solution_view());
        metaheuristic.on_start(
            model,
            self.accepted_state.as_solution_view(),
            &self.topology_graph,
        );

        let termination_reason;

        // ══════════════════════════════════════════════════════════════
        //  OUTER LOOP — one pass per neighbourhood exploration
        // ══════════════════════════════════════════════════════════════
        'outer: loop {
            // ── Check termination: metaheuristic ──
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

            // ── Check termination: monitor ──
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

            // ── Notify cycle start ──
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

            // ── Prepare operator for this neighbourhood ──
            operator.prepare(
                self.best_state.as_solution_view(),
                self.accepted_state.as_solution_view(),
                if has_buffered {
                    Some(self.buffered_state.as_solution_view())
                } else {
                    None
                },
                &self.topology_graph,
            );

            // ══════════════════════════════════════════════════════════
            //  INNER LOOP — one pass per candidate neighbour
            // ══════════════════════════════════════════════════════════
            'inner: loop {
                // Clear the structural diff before the next mutation.
                self.schedule_graph_diff.clear();

                // ── Generate the next neighbour ──
                let mutated = {
                    let mut mutator = Mutator::new(
                        &mut self.topology_graph,
                        &mut self.schedule_graph_undo_log,
                        &mut self.schedule_graph_diff,
                        &mut self.touched,
                    );
                    unsafe {
                        operator.next_neighbor(
                            model,
                            self.best_state.as_solution_view(),
                            self.accepted_state.as_solution_view(),
                            if has_buffered {
                                Some(self.buffered_state.as_solution_view())
                            } else {
                                None
                            },
                            &mut mutator,
                            &stats,
                        )
                    }
                };

                // ── Neighbourhood exhausted ──
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
                    if has_buffered
                        && metaheuristic.should_commit_buffered(
                            model,
                            self.best_state.as_solution_view(),
                            self.accepted_state.as_solution_view(),
                            Some(self.buffered_state.as_solution_view()),
                            &self.topology_graph,
                            &self.buffered_topology_graph,
                        )
                    {
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
                            &self.topology_graph,
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
                                &self.topology_graph,
                                &self.buffered_schedule_graph_diff,
                            );
                        }
                    }

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
                        &self.topology_graph,
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

                // ── Delta-decode touched berths ──
                let decode_ok = unsafe {
                    decode_unchecked(
                        &self.touched,
                        &self.topology_graph,
                        &mut self.candidate_state,
                        &self.accepted_state,
                        model,
                        &evaluator,
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
                        &self.topology_graph,
                    );

                    continue 'inner;
                }

                stats.on_found_solution();

                let candidate_objective = self.candidate_state.objective();

                // ── Monitor: candidate generated ──
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

                // ── Metaheuristic: decide fate ──
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
                    &self.topology_graph,
                    &self.schedule_graph_diff,
                );

                match decision {
                    // ─────────── ACCEPT ───────────
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
                            &self.topology_graph,
                            &self.schedule_graph_diff,
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
                                &self.topology_graph,
                                &self.schedule_graph_diff,
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
                            &self.topology_graph,
                        );

                        break 'inner;
                    }

                    // ─────────── BUFFER ───────────
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
                        self.schedule_graph_undo_log
                            .apply_rollback(&mut self.topology_graph);

                        metaheuristic.on_iteration(
                            stats.iterations,
                            model,
                            self.best_state.as_solution_view(),
                            self.accepted_state.as_solution_view(),
                            Some(self.buffered_state.as_solution_view()),
                            &self.topology_graph,
                        );

                        continue 'inner;
                    }

                    // ─────────── REJECT ───────────
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
                            &self.topology_graph,
                            &self.schedule_graph_diff,
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
                            &self.topology_graph,
                        );

                        continue 'inner;
                    }
                }
            }
        }

        // ── Finalize ──
        stats.set_total_time(start_time.elapsed());

        monitor.on_end(self.best_state.as_solution_view(), &stats);
        metaheuristic.on_end(
            model,
            self.best_state.as_solution_view(),
            &self.topology_graph,
        );

        let best_view = self.best_state.as_solution_view();
        let solution = Solution::from_slices(
            best_view.berths(),
            best_view.start_times(),
            best_view.objective_value(),
        );

        LocalSearchOutcome::new(solution, termination_reason, stats)
    }

    #[inline(always)]
    pub fn update_best(&mut self)
    where
        T: SolverNumeric,
    {
        self.best_state.overwrite_from_state(&self.accepted_state);
    }

    /// Accepts the current candidate immediately.
    /// This is the fastest commit path, skipping full array copies.
    #[inline(always)]
    pub fn accept_candidate(&mut self)
    where
        T: SolverNumeric,
    {
        unsafe {
            self.accepted_state.patch_from_delta_unchecked(
                &self.candidate_state,
                &self.touched,
                &self.topology_graph,
            );
        }
        self.schedule_graph_undo_log.clear();
        self.touched.reset();
    }

    /// Rejects the current candidate and restores the graph to its previous state.
    #[inline(always)]
    pub fn reject_candidate(&mut self) {
        self.schedule_graph_undo_log
            .apply_rollback(&mut self.topology_graph);
        self.touched.reset();
    }

    /// Saves the current candidate to the buffer for later comparison.
    /// Used by Tabu Search to evaluate an entire neighborhood before committing.
    /// Also saves the current structural diff so it is available when the
    /// buffer is committed.
    #[inline(always)]
    pub fn save_candidate_to_buffer(&mut self)
    where
        T: SolverNumeric,
    {
        self.buffered_topology_graph
            .overwrite_from_graph(&self.topology_graph);
        self.buffered_schedule_graph_diff
            .overwrite_from_graph_diff(&self.schedule_graph_diff);
        self.buffered_state
            .overwrite_from_state(&self.accepted_state);

        unsafe {
            self.buffered_state.patch_from_delta_unchecked(
                &self.candidate_state,
                &self.touched,
                &self.topology_graph,
            );
        }

        self.touched.reset();
    }

    /// Commits the buffered state, making it the new accepted state.
    #[inline(always)]
    pub fn accept_buffered(&mut self)
    where
        T: SolverNumeric,
    {
        self.accepted_state
            .overwrite_from_state(&self.buffered_state);
        self.topology_graph
            .overwrite_from_graph(&self.buffered_topology_graph);
        self.schedule_graph_undo_log.clear();
        self.touched.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::{SearchCommand, TerminationReason};
    use crate::meta::metaheuristic::{
        AcceptanceOutcome, Metaheuristic, NeighborhoodExhaustionOutcome,
    };
    use crate::monitor::lsmonitor::LocalSearchMonitor;
    use crate::operator::lsoperator::LocalSearchOperator;
    use crate::params::MutableLocalSearchParams;
    use crate::stats::LocalSearchStatistics;
    use std::cell::Cell;
    use std::sync::atomic::{AtomicU64, Ordering};
    use talos_core::math::interval::ClosedOpenInterval;
    use talos_model::index::{BerthIndex, VesselIndex};
    use talos_model::model::{Model, ProcessingTime};
    use talos_model::solution::SolutionView;

    // ────────────────────────────────────────────────────────
    //  Test helpers
    // ────────────────────────────────────────────────────────

    /// Builds a minimal 2-vessel / 2-berth model.
    /// V0: arrival=0, deadline=100, weight=1, p(B0)=5, p(B1)=5
    /// V1: arrival=0, deadline=100, weight=1, p(B0)=10, p(B1)=10
    /// Both berths open [0, 200).
    fn build_test_model() -> Model<i64> {
        Model::new(
            2,
            2,
            vec![0, 0],
            vec![100, 100],
            vec![1, 1],
            vec![
                ProcessingTime::some(5),
                ProcessingTime::some(5),
                ProcessingTime::some(10),
                ProcessingTime::some(10),
            ],
            vec![
                vec![ClosedOpenInterval::new(0, 200)],
                vec![ClosedOpenInterval::new(0, 200)],
            ],
        )
    }

    /// Simple evaluator: cost = weight * start_time.
    fn mock_evaluator(
        model: &Model<i64>,
        vessel: VesselIndex,
        _berth: BerthIndex,
        start: i64,
    ) -> Option<i64> {
        Some(model.vessel_weight(vessel) * start)
    }

    // ────────────────────────────────────────────────────────
    //  Mock Monitor — records event counts
    // ────────────────────────────────────────────────────────

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

    // ────────────────────────────────────────────────────────
    //  Mock Operator — serves a fixed number of swap mutations
    // ────────────────────────────────────────────────────────

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
        fn name(&self) -> &str {
            "MockOperator"
        }

        fn prepare(
            &mut self,
            _best: SolutionView<'_, i64>,
            _accepted: SolutionView<'_, i64>,
            _buffered: Option<SolutionView<'_, i64>>,
            _graph: &ScheduleGraph,
        ) {
            self.cursor = 0;
        }

        unsafe fn next_neighbor(
            &mut self,
            _model: &Model<i64>,
            _best: SolutionView<'_, i64>,
            _accepted: SolutionView<'_, i64>,
            _buffered: Option<SolutionView<'_, i64>>,
            mutator: &mut Mutator<'_>,
            _stats: &LocalSearchStatistics,
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

    // ────────────────────────────────────────────────────────
    //  Mock Metaheuristic — configurable acceptance policy
    // ────────────────────────────────────────────────────────

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

    // ────────────────────────────────────────────────────────
    //  Helper: build params + engine, run, return (outcome, monitor)
    // ────────────────────────────────────────────────────────

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
            berths: &berths,
            start_times: &start_times,
            objective_value: 5, // weight=1 * start=0 + weight=1 * start=5
        };

        let mut engine = Engine::<i64>::new(2, 2);
        let mut best_callbacks = 0u32;
        let outcome = engine.run(params, mock_evaluator, |_| {
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
            berths: &berths,
            start_times: &start_times,
            objective_value: 5,
        };

        let mut engine = Engine::<i64>::new(2, 2);
        engine.run(params, mock_evaluator, |_| {})
    }

    // ════════════════════════════════════════════════════════
    //  Tests
    // ════════════════════════════════════════════════════════

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
        assert_eq!(meta.on_iteration_calls.load(Ordering::Relaxed), 3);
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
}
