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
    sgraph::ScheduleGraph, sgraphundo::ScheduleGraphUndoLog, state::ScheduleState,
    tberth::TouchedBerths,
};
use talos_core::utils::num::SolverNumeric;

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

    /// The mutable schedule hraph that encodes the current and canditate states
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

    /// **Dirty-Tracking Set**: A type-safe bitset or boolean mask identifying
    /// berths modified during the current mutation. Informs the downstream
    /// decoder exactly which timelines require recalculation.
    touched: TouchedBerths,
}

impl<T> Engine<T> {
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
    #[inline(always)]
    pub fn save_candidate_to_buffer(&mut self)
    where
        T: SolverNumeric,
    {
        self.buffered_topology_graph
            .overwrite_from_graph(&self.topology_graph);
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
