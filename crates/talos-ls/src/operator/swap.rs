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
// IntraBerthSwapOperator
// ----------------------------------------------------------------

pub struct IntraBerthSwapOperator<T, F>
where
    T: SolverNumeric,
    F: Fn(VesselIndex, VesselIndex, SolutionView<'_, T>, &ScheduleGraph, &Model<T>) -> bool
        + Send
        + Sync,
{
    filter: F,
    num_vessels: usize,
    cursor_a: VesselIndex,
    cursor_b: VesselIndex,
    _phantom: PhantomData<T>,
}

impl<T, F> IntraBerthSwapOperator<T, F>
where
    T: SolverNumeric,
    F: Fn(VesselIndex, VesselIndex, SolutionView<'_, T>, &ScheduleGraph, &Model<T>) -> bool
        + Send
        + Sync,
{
    pub fn new(filter: F) -> Self {
        Self {
            filter,
            num_vessels: 0,
            cursor_a: VesselIndex::new(0),
            cursor_b: VesselIndex::new(1),
            _phantom: PhantomData,
        }
    }
}

impl<T, F> LocalSearchOperator<T> for IntraBerthSwapOperator<T, F>
where
    T: SolverNumeric,
    F: Fn(VesselIndex, VesselIndex, SolutionView<'_, T>, &ScheduleGraph, &Model<T>) -> bool
        + Send
        + Sync,
{
    #[inline(always)]
    fn name(&self) -> &str {
        "IntraBerthSwap"
    }

    fn prepare(
        &mut self,
        _best_solution: SolutionView<'_, T>,
        _accepted_solution: SolutionView<'_, T>,
        _buffered_solution: Option<SolutionView<'_, T>>,
        graph: &ScheduleGraph,
    ) {
        self.num_vessels = graph.num_vessels();
        self.cursor_a = VesselIndex::new(0);
        self.cursor_b = VesselIndex::new(1);
    }

    unsafe fn next_neighbor(
        &mut self,
        model: &Model<T>,
        _best_solution: SolutionView<'_, T>,
        accepted_solution: SolutionView<'_, T>,
        _buffered_solution: Option<SolutionView<'_, T>>,
        mutator: &mut Mutator,
        _stats: &LocalSearchStatistics,
    ) -> bool {
        loop {
            if self.cursor_a >= self.num_vessels.saturating_sub(1) {
                return false;
            }
            if self.cursor_b >= self.num_vessels {
                self.cursor_a += 1;
                self.cursor_b = self.cursor_a + 1;
                continue;
            }

            let v_a = self.cursor_a;
            let v_b = self.cursor_b;
            self.cursor_b += 1;

            // Strict Intra-Berth Check ($O(1)$ Unchecked)
            let berth_a = unsafe { mutator.graph().vessel_berth_unchecked(v_a) };
            let berth_b = unsafe { mutator.graph().vessel_berth_unchecked(v_b) };

            if berth_a == berth_b
                && (self.filter)(v_a, v_b, accepted_solution, mutator.graph(), model)
            {
                unsafe {
                    mutator.swap_vessels_unchecked(v_a, v_b);
                }
                return true;
            }
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.cursor_a = VesselIndex::new(0);
        self.cursor_b = VesselIndex::new(1);
    }
}

// ----------------------------------------------------------------
// InterBerthSwapOperator
// ----------------------------------------------------------------

pub struct InterBerthSwapOperator<T, F>
where
    T: SolverNumeric,
    F: Fn(VesselIndex, VesselIndex, SolutionView<'_, T>, &ScheduleGraph, &Model<T>) -> bool
        + Send
        + Sync,
{
    filter: F,
    num_vessels: usize,
    cursor_a: VesselIndex,
    cursor_b: VesselIndex,
    _phantom: PhantomData<T>,
}

impl<T, F> InterBerthSwapOperator<T, F>
where
    T: SolverNumeric,
    F: Fn(VesselIndex, VesselIndex, SolutionView<'_, T>, &ScheduleGraph, &Model<T>) -> bool
        + Send
        + Sync,
{
    pub fn new(filter: F) -> Self {
        Self {
            filter,
            num_vessels: 0,
            cursor_a: VesselIndex::new(0),
            cursor_b: VesselIndex::new(1),
            _phantom: PhantomData,
        }
    }
}

impl<T, F> LocalSearchOperator<T> for InterBerthSwapOperator<T, F>
where
    T: SolverNumeric,
    F: Fn(VesselIndex, VesselIndex, SolutionView<'_, T>, &ScheduleGraph, &Model<T>) -> bool
        + Send
        + Sync,
{
    #[inline(always)]
    fn name(&self) -> &str {
        "InterBerthSwap"
    }

    fn prepare(
        &mut self,
        _best_solution: SolutionView<'_, T>,
        _accepted_solution: SolutionView<'_, T>,
        _buffered_solution: Option<SolutionView<'_, T>>,
        graph: &ScheduleGraph,
    ) {
        self.num_vessels = graph.num_vessels();
        self.cursor_a = VesselIndex::new(0);
        self.cursor_b = VesselIndex::new(1);
    }

    unsafe fn next_neighbor(
        &mut self,
        model: &Model<T>,
        _best_solution: SolutionView<'_, T>,
        accepted_solution: SolutionView<'_, T>,
        _buffered_solution: Option<SolutionView<'_, T>>,
        mutator: &mut Mutator,
        _stats: &LocalSearchStatistics,
    ) -> bool {
        loop {
            if self.cursor_a >= self.num_vessels.saturating_sub(1) {
                return false;
            }
            if self.cursor_b >= self.num_vessels {
                self.cursor_a += 1;
                self.cursor_b = self.cursor_a + 1;
                continue;
            }

            let v_a = self.cursor_a;
            let v_b = self.cursor_b;
            self.cursor_b += 1;

            // Strict Inter-Berth Check ($O(1)$ Unchecked)
            let berth_a = unsafe { mutator.graph().vessel_berth_unchecked(v_a) };
            let berth_b = unsafe { mutator.graph().vessel_berth_unchecked(v_b) };

            if berth_a != berth_b
                && (self.filter)(v_a, v_b, accepted_solution, mutator.graph(), model)
            {
                unsafe {
                    mutator.swap_vessels_unchecked(v_a, v_b);
                }
                return true;
            }
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.cursor_a = VesselIndex::new(0);
        self.cursor_b = VesselIndex::new(1);
    }
}
