// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    mutator::Mutator, operator::lsoperator::LocalSearchOperator, sgraph::ScheduleGraph,
    stats::LocalSearchStatistics,
};
use std::marker::PhantomData;
use talos_core::math::num::SolverNumeric;
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
