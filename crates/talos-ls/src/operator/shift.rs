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

use crate::{
    mutator::Mutator, operator::lsoperator::LocalSearchOperator, sgraph::ScheduleGraph,
    stats::LocalSearchStatistics,
};
use std::marker::PhantomData;
use talos_core::utils::num::SolverNumeric;
use talos_model::{index::VesselIndex, model::Model, solution::SolutionView};

// ----------------------------------------------------------------
// IntraBerthShiftOperator
// ----------------------------------------------------------------

/// An operator that shifts a vessel to immediately follow another vessel
/// within the SAME berth.
pub struct IntraBerthShiftOperator<T, F>
where
    T: SolverNumeric,
    F: Fn(VesselIndex, VesselIndex, SolutionView<'_, T>, &ScheduleGraph) -> bool + Send + Sync,
{
    filter: F,
    num_vessels: usize,
    cursor_v: VesselIndex,      // The vessel to move
    cursor_anchor: VesselIndex, // The vessel to move *after*
    _phantom: PhantomData<T>,
}

impl<T, F> IntraBerthShiftOperator<T, F>
where
    T: SolverNumeric,
    F: Fn(VesselIndex, VesselIndex, SolutionView<'_, T>, &ScheduleGraph) -> bool + Send + Sync,
{
    pub fn new(filter: F) -> Self {
        Self {
            filter,
            num_vessels: 0,
            cursor_v: VesselIndex::new(0),
            cursor_anchor: VesselIndex::new(0),
            _phantom: PhantomData,
        }
    }
}

impl<T, F> LocalSearchOperator<T> for IntraBerthShiftOperator<T, F>
where
    T: SolverNumeric,
    F: Fn(VesselIndex, VesselIndex, SolutionView<'_, T>, &ScheduleGraph) -> bool + Send + Sync,
{
    #[inline(always)]
    fn name(&self) -> &str {
        "IntraBerthShift"
    }

    fn prepare(
        &mut self,
        _best_solution: SolutionView<'_, T>,
        _accepted_solution: SolutionView<'_, T>,
        _buffered_solution: Option<SolutionView<'_, T>>,
        graph: &ScheduleGraph,
    ) {
        self.num_vessels = graph.num_vessels();
        self.cursor_v = VesselIndex::new(0);
        self.cursor_anchor = VesselIndex::new(0);
    }

    unsafe fn next_neighbor(
        &mut self,
        _model: &Model<T>,
        _best_solution: SolutionView<'_, T>,
        accepted_solution: SolutionView<'_, T>,
        _buffered_solution: Option<SolutionView<'_, T>>,
        mutator: &mut Mutator,
        _stats: &LocalSearchStatistics,
    ) -> bool {
        loop {
            if self.cursor_v >= self.num_vessels {
                return false;
            }
            if self.cursor_anchor >= self.num_vessels {
                self.cursor_v += 1;
                self.cursor_anchor = VesselIndex::new(0);
                continue;
            }

            let v = self.cursor_v;
            let anchor = self.cursor_anchor;
            self.cursor_anchor += 1;

            // 1. Cannot shift a vessel after itself
            if v == anchor {
                continue;
            }

            // 2. Strict Intra-Berth Check ($O(1)$ Unchecked)
            let berth_v = unsafe { mutator.graph().vessel_berth_unchecked(v) };
            let berth_anchor = unsafe { mutator.graph().vessel_berth_unchecked(anchor) };

            if berth_v == berth_anchor {
                // 3. Topological Optimization: Skip if already in this position
                let current_pred = unsafe { mutator.graph().vessel_predecessor_unchecked(v) };
                if current_pred == Some(anchor) {
                    continue;
                }

                if (self.filter)(v, anchor, accepted_solution, mutator.graph()) {
                    unsafe {
                        mutator.relocate_after_unchecked(v, anchor);
                    }
                    return true;
                }
            }
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.cursor_v = VesselIndex::new(0);
        self.cursor_anchor = VesselIndex::new(0);
    }
}

// ----------------------------------------------------------------
// InterBerthShiftOperator
// ----------------------------------------------------------------

/// An operator that shifts a vessel to immediately follow another vessel
/// in a DIFFERENT berth.
pub struct InterBerthShiftOperator<T, F>
where
    T: SolverNumeric,
    F: Fn(VesselIndex, VesselIndex, SolutionView<'_, T>, &ScheduleGraph) -> bool + Send + Sync,
{
    filter: F,
    num_vessels: usize,
    cursor_v: VesselIndex,      // The vessel to move
    cursor_anchor: VesselIndex, // The vessel to move *after*
    _phantom: PhantomData<T>,
}

impl<T, F> InterBerthShiftOperator<T, F>
where
    T: SolverNumeric,
    F: Fn(VesselIndex, VesselIndex, SolutionView<'_, T>, &ScheduleGraph) -> bool + Send + Sync,
{
    pub fn new(filter: F) -> Self {
        Self {
            filter,
            num_vessels: 0,
            cursor_v: VesselIndex::new(0),
            cursor_anchor: VesselIndex::new(0),
            _phantom: PhantomData,
        }
    }
}

impl<T, F> LocalSearchOperator<T> for InterBerthShiftOperator<T, F>
where
    T: SolverNumeric,
    F: Fn(VesselIndex, VesselIndex, SolutionView<'_, T>, &ScheduleGraph) -> bool + Send + Sync,
{
    #[inline(always)]
    fn name(&self) -> &str {
        "InterBerthShift"
    }

    fn prepare(
        &mut self,
        _best_solution: SolutionView<'_, T>,
        _accepted_solution: SolutionView<'_, T>,
        _buffered_solution: Option<SolutionView<'_, T>>,
        graph: &ScheduleGraph,
    ) {
        self.num_vessels = graph.num_vessels();
        self.cursor_v = VesselIndex::new(0);
        self.cursor_anchor = VesselIndex::new(0);
    }

    unsafe fn next_neighbor(
        &mut self,
        _model: &Model<T>,
        _best_solution: SolutionView<'_, T>,
        accepted_solution: SolutionView<'_, T>,
        _buffered_solution: Option<SolutionView<'_, T>>,
        mutator: &mut Mutator,
        _stats: &LocalSearchStatistics,
    ) -> bool {
        loop {
            if self.cursor_v >= self.num_vessels {
                return false;
            }
            if self.cursor_anchor >= self.num_vessels {
                self.cursor_v += 1;
                self.cursor_anchor = VesselIndex::new(0);
                continue;
            }

            let v = self.cursor_v;
            let anchor = self.cursor_anchor;
            self.cursor_anchor += 1;

            if v == anchor {
                continue;
            }

            // 2. Strict Inter-Berth Check ($O(1)$ Unchecked)
            let berth_v = unsafe { mutator.graph().vessel_berth_unchecked(v) };
            let berth_anchor = unsafe { mutator.graph().vessel_berth_unchecked(anchor) };

            if berth_v != berth_anchor
                && (self.filter)(v, anchor, accepted_solution, mutator.graph())
            {
                unsafe {
                    mutator.relocate_after_unchecked(v, anchor);
                }
                return true;
            }
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.cursor_v = VesselIndex::new(0);
        self.cursor_anchor = VesselIndex::new(0);
    }
}
