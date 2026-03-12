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

//! Integration tests exercising the full engine pipeline with real operators,
//! metaheuristics, and monitors on artificial DBAP instances.

use std::time::Duration;

use rand::SeedableRng;
use rand::rngs::StdRng;

use talos_core::math::interval::ClosedOpenInterval;
use talos_ls::engine::Engine;
use talos_ls::eval::calculate_weighted_completion_time;
use talos_ls::exec::TerminationReason;
use talos_ls::meta::gls::GuidedLocalSearch;
use talos_ls::meta::sa::{GeometricCooling, SimulatedAnnealing};
use talos_ls::meta::tabu::{FixedTenure, TabuSearch};
use talos_ls::monitor::composite::CompositeLocalSearchMonitor;
use talos_ls::monitor::cycle::CycleLimitMonitor;
use talos_ls::monitor::iteration::IterationLimitMonitor;
use talos_ls::monitor::nimpr::NoImprovementMonitor;
use talos_ls::monitor::time::TimeLimitMonitor;
use talos_ls::operator::composite::RoundRobinCompoundOperator;
use talos_ls::operator::filter::{
    inter_berth_shift_filter_unchecked, inter_berth_swap_filter_unchecked,
    intra_berth_shift_filter_unchecked, intra_berth_swap_filter_unchecked,
};
use talos_ls::operator::lsoperator::LocalSearchOperator;
use talos_ls::operator::shift::{InterBerthShiftOperator, IntraBerthShiftOperator};
use talos_ls::operator::swap::{InterBerthSwapOperator, IntraBerthSwapOperator};
use talos_ls::params::MutableLocalSearchParams;
use talos_model::index::{BerthIndex, VesselIndex};
use talos_model::model::{Model, ProcessingTime};

// ════════════════════════════════════════════════════════════════════════════
//  Artificial Instance Builders
// ════════════════════════════════════════════════════════════════════════════

/// Tiny 2-vessel / 2-berth instance.
///
/// ```text
/// V0: arrival=0  deadline=100  weight=1  p(B0)=5   p(B1)=8
/// V1: arrival=0  deadline=100  weight=2  p(B0)=10  p(B1)=6
/// Both berths open [0, 200).
/// ```
///
/// Initial solution: both on B0, V0@t=0, V1@t=5.
/// Initial objective = (0+5)*1 + (5+10)*2 = 5 + 30 = 35.
fn tiny_instance() -> (Model<i64>, Vec<BerthIndex>, Vec<i64>, i64) {
    let model = Model::new(
        2,
        2,
        vec![0, 0],
        vec![100, 100],
        vec![1, 2],
        vec![
            ProcessingTime::some(5),
            ProcessingTime::some(8),
            ProcessingTime::some(10),
            ProcessingTime::some(6),
        ],
        vec![
            vec![ClosedOpenInterval::new(0, 200)],
            vec![ClosedOpenInterval::new(0, 200)],
        ],
    );
    let berths = vec![BerthIndex::new(0), BerthIndex::new(0)];
    let starts = vec![0, 5];
    let obj = compute_objective(&model, &berths, &starts);
    (model, berths, starts, obj)
}

/// Small 5-vessel / 3-berth instance with varied weights and processing times.
///
/// All vessels may dock at all berths. Berths open [0, 1000).
///
/// Initial solution packs all vessels onto B0 sequentially, producing a high
/// objective that the solver should easily improve by spreading work.
fn small_instance() -> (Model<i64>, Vec<BerthIndex>, Vec<i64>, i64) {
    let num_v = 5;
    let num_b = 3;
    let arrivals = vec![0; num_v];
    let deadlines = vec![1000; num_v];
    let weights = vec![3, 1, 4, 1, 5];

    // Row-major: [V0B0 V0B1 V0B2 | V1B0 V1B1 V1B2 | ...]
    let ptimes = vec![
        ProcessingTime::some(10),
        ProcessingTime::some(12),
        ProcessingTime::some(8),
        ProcessingTime::some(15),
        ProcessingTime::some(9),
        ProcessingTime::some(11),
        ProcessingTime::some(8),
        ProcessingTime::some(10),
        ProcessingTime::some(7),
        ProcessingTime::some(20),
        ProcessingTime::some(14),
        ProcessingTime::some(18),
        ProcessingTime::some(6),
        ProcessingTime::some(8),
        ProcessingTime::some(5),
    ];

    let opening = vec![vec![ClosedOpenInterval::new(0, 1000)]; num_b];

    let model = Model::new(num_v, num_b, arrivals, deadlines, weights, ptimes, opening);

    // All on B0, stacked sequentially: 0, 10, 25, 33, 53
    let berths = vec![BerthIndex::new(0); num_v];
    let mut starts = Vec::with_capacity(num_v);
    let mut t = 0i64;
    for v in 0..num_v {
        starts.push(t);
        t += model
            .vessel_processing_time(VesselIndex::new(v), BerthIndex::new(0))
            .unwrap_unchecked();
    }

    let obj = compute_objective(&model, &berths, &starts);
    (model, berths, starts, obj)
}

/// Medium 10-vessel / 4-berth instance with some berth restrictions.
///
/// Vessels 0..4 can only dock at B0 or B1. Vessels 5..9 can dock at any berth.
/// Varied processing times and weights create a non-trivial optimization
/// landscape.
fn medium_instance() -> (Model<i64>, Vec<BerthIndex>, Vec<i64>, i64) {
    let num_v = 10;
    let num_b = 4;
    let arrivals = vec![0; num_v];
    let deadlines = vec![2000; num_v];
    let weights = vec![2, 5, 1, 3, 4, 2, 6, 1, 3, 7];

    // Build processing times: V0..V4 → only B0, B1 allowed.
    let mut ptimes = Vec::with_capacity(num_v * num_b);
    let base_p = [8, 12, 6, 10, 14, 9, 7, 15, 11, 5];
    for (v, &bp) in base_p.iter().enumerate() {
        for b in 0..num_b {
            if v < 5 && b >= 2 {
                ptimes.push(ProcessingTime::none());
            } else {
                let p = bp + (b as i64) * 2;
                ptimes.push(ProcessingTime::some(p));
            }
        }
    }

    let opening = vec![vec![ClosedOpenInterval::new(0, 2000)]; num_b];
    let model = Model::new(num_v, num_b, arrivals, deadlines, weights, ptimes, opening);

    // Initial: spread across B0 and B1 only (safe for all vessels).
    let berths: Vec<BerthIndex> = (0..num_v).map(|v| BerthIndex::new(v % 2)).collect();

    // Stack vessels sequentially per-berth.
    let mut starts = vec![0i64; num_v];
    let mut berth_clock = vec![0i64; num_b];
    for v in 0..num_v {
        let b = berths[v];
        starts[v] = berth_clock[b.get()];
        berth_clock[b.get()] = starts[v]
            + model
                .vessel_processing_time(VesselIndex::new(v), b)
                .unwrap_unchecked();
    }

    let obj = compute_objective(&model, &berths, &starts);
    (model, berths, starts, obj)
}

// ════════════════════════════════════════════════════════════════════════════
//  Helpers
// ════════════════════════════════════════════════════════════════════════════

/// Evaluator for `Engine::run`: weighted completion time.
fn evaluator(
    model: &Model<i64>,
    vessel: VesselIndex,
    berth: BerthIndex,
    start: i64,
) -> Option<i64> {
    calculate_weighted_completion_time(model, vessel, berth, start)
}

/// Compute objective value for a full assignment.
fn compute_objective(model: &Model<i64>, berths: &[BerthIndex], starts: &[i64]) -> i64 {
    (0..model.num_vessels())
        .map(|v| {
            calculate_weighted_completion_time(model, VesselIndex::new(v), berths[v], starts[v])
                .expect("initial solution must be feasible")
        })
        .sum()
}

/// Build a round-robin compound operator with all four move types.
fn build_full_operator<'a>() -> RoundRobinCompoundOperator<'a, i64> {
    let ops: Vec<Box<dyn LocalSearchOperator<i64> + 'a>> = vec![
        Box::new(IntraBerthSwapOperator::new(
            |v_a, v_b, sol, graph, model| unsafe {
                intra_berth_swap_filter_unchecked(v_a, v_b, sol, graph, model)
            },
        )),
        Box::new(InterBerthSwapOperator::new(
            |v_a, v_b, sol, graph, model| unsafe {
                inter_berth_swap_filter_unchecked(v_a, v_b, sol, graph, model)
            },
        )),
        Box::new(IntraBerthShiftOperator::new(
            |v, anchor, sol, graph, model| unsafe {
                intra_berth_shift_filter_unchecked(v, anchor, sol, graph, model)
            },
        )),
        Box::new(InterBerthShiftOperator::new(
            |v, anchor, sol, graph, model| unsafe {
                inter_berth_shift_filter_unchecked(v, anchor, sol, graph, model)
            },
        )),
    ];
    RoundRobinCompoundOperator::new(ops)
}

/// Build a swap-only compound operator (no shift moves).
fn build_swap_operator<'a>() -> RoundRobinCompoundOperator<'a, i64> {
    let ops: Vec<Box<dyn LocalSearchOperator<i64> + 'a>> = vec![
        Box::new(IntraBerthSwapOperator::new(
            |v_a, v_b, sol, graph, model| unsafe {
                intra_berth_swap_filter_unchecked(v_a, v_b, sol, graph, model)
            },
        )),
        Box::new(InterBerthSwapOperator::new(
            |v_a, v_b, sol, graph, model| unsafe {
                inter_berth_swap_filter_unchecked(v_a, v_b, sol, graph, model)
            },
        )),
    ];
    RoundRobinCompoundOperator::new(ops)
}

/// Run the engine with the given components and return the outcome.
fn run_search<H, O, M>(
    model: &Model<i64>,
    operator: &mut O,
    metaheuristic: &mut H,
    monitor: M,
    berths: &[BerthIndex],
    start_times: &[i64],
    objective_value: i64,
) -> talos_ls::outcome::LocalSearchOutcome<i64>
where
    H: talos_ls::meta::metaheuristic::Metaheuristic<i64>,
    O: LocalSearchOperator<i64>,
    M: talos_ls::monitor::lsmonitor::LocalSearchMonitor<i64>,
{
    let mut engine = Engine::<i64>::new(model.num_vessels(), model.num_berths());
    let params = MutableLocalSearchParams {
        model,
        operator,
        metaheuristic,
        monitor,
        berths,
        start_times,
        objective_value,
    };
    engine.run(params, evaluator, |_| {})
}

// ════════════════════════════════════════════════════════════════════════════
//  Simulated Annealing Tests
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn sa_tiny_instance_improves() {
    let (model, berths, starts, obj) = tiny_instance();
    let cooling = GeometricCooling::new(100.0, 0.99, 0.01);
    let mut sa = SimulatedAnnealing::new(cooling, StdRng::seed_from_u64(42));
    let mut op = build_full_operator();
    let monitor = IterationLimitMonitor::new(5_000);

    let outcome = run_search(&model, &mut op, &mut sa, monitor, &berths, &starts, obj);

    assert!(
        outcome.solution().objective_value() <= obj,
        "SA should not worsen: got {} vs initial {}",
        outcome.solution().objective_value(),
        obj
    );
    assert!(outcome.stats().iterations > 0);
}

#[test]
fn sa_small_instance_finds_improvement() {
    let (model, berths, starts, obj) = small_instance();
    let cooling = GeometricCooling::new(200.0, 0.995, 0.01);
    let mut sa = SimulatedAnnealing::new(cooling, StdRng::seed_from_u64(123));
    let mut op = build_full_operator();
    let monitor = IterationLimitMonitor::new(10_000);

    let outcome = run_search(&model, &mut op, &mut sa, monitor, &berths, &starts, obj);

    assert!(
        outcome.solution().objective_value() < obj,
        "SA on a badly packed initial should improve: got {} vs initial {}",
        outcome.solution().objective_value(),
        obj
    );
}

#[test]
fn sa_medium_instance_respects_berth_constraints() {
    let (model, berths, starts, obj) = medium_instance();
    let cooling = GeometricCooling::new(500.0, 0.997, 0.01);
    let mut sa = SimulatedAnnealing::new(cooling, StdRng::seed_from_u64(7));
    let mut op = build_full_operator();
    let monitor = IterationLimitMonitor::new(20_000);

    let outcome = run_search(&model, &mut op, &mut sa, monitor, &berths, &starts, obj);

    let sol = outcome.solution();
    assert!(sol.objective_value() <= obj);

    // Verify feasibility: every vessel must be on an allowed berth.
    for v in 0..model.num_vessels() {
        let b = sol.berths()[v];
        assert!(
            model.vessel_allowed_on_berth(VesselIndex::new(v), b),
            "V{v} assigned to forbidden berth B{}",
            b.get()
        );
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  Tabu Search Tests
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn tabu_tiny_instance_improves() {
    let (model, berths, starts, obj) = tiny_instance();
    let mut ts = TabuSearch::new(FixedTenure::new(5), model.num_vessels(), model.num_berths());
    let mut op = build_full_operator();
    let monitor = CycleLimitMonitor::new(50);

    let outcome = run_search(&model, &mut op, &mut ts, monitor, &berths, &starts, obj);

    assert!(
        outcome.solution().objective_value() <= obj,
        "Tabu should not worsen: got {} vs initial {}",
        outcome.solution().objective_value(),
        obj
    );
}

#[test]
fn tabu_small_instance_finds_improvement() {
    let (model, berths, starts, obj) = small_instance();
    let mut ts = TabuSearch::new(FixedTenure::new(7), model.num_vessels(), model.num_berths());
    let mut op = build_full_operator();
    let monitor = CycleLimitMonitor::new(100);

    let outcome = run_search(&model, &mut op, &mut ts, monitor, &berths, &starts, obj);

    assert!(
        outcome.solution().objective_value() < obj,
        "Tabu on a badly packed initial should improve: got {} vs initial {}",
        outcome.solution().objective_value(),
        obj
    );
}

#[test]
fn tabu_medium_instance_respects_berth_constraints() {
    let (model, berths, starts, obj) = medium_instance();
    let mut ts = TabuSearch::new(
        FixedTenure::new(10),
        model.num_vessels(),
        model.num_berths(),
    );
    let mut op = build_full_operator();
    let monitor = CycleLimitMonitor::new(200);

    let outcome = run_search(&model, &mut op, &mut ts, monitor, &berths, &starts, obj);

    let sol = outcome.solution();
    assert!(sol.objective_value() <= obj);

    for v in 0..model.num_vessels() {
        let b = sol.berths()[v];
        assert!(
            model.vessel_allowed_on_berth(VesselIndex::new(v), b),
            "V{v} assigned to forbidden berth B{}",
            b.get()
        );
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  Guided Local Search Tests
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn gls_tiny_instance_improves() {
    let (model, berths, starts, obj) = tiny_instance();
    let mut gls = GuidedLocalSearch::new(0.3, model.num_vessels(), model.num_berths());
    let mut op = build_full_operator();
    let monitor = IterationLimitMonitor::new(5_000);

    let outcome = run_search(&model, &mut op, &mut gls, monitor, &berths, &starts, obj);

    assert!(
        outcome.solution().objective_value() <= obj,
        "GLS should not worsen: got {} vs initial {}",
        outcome.solution().objective_value(),
        obj
    );
}

#[test]
fn gls_small_instance_finds_improvement() {
    let (model, berths, starts, obj) = small_instance();
    let lambda = talos_ls::meta::gls::heuristic_lambda(
        obj as f64,
        model.num_vessels() * model.num_berths(),
        0.3,
    );
    let mut gls = GuidedLocalSearch::new(lambda, model.num_vessels(), model.num_berths());
    let mut op = build_full_operator();
    let monitor = IterationLimitMonitor::new(10_000);

    let outcome = run_search(&model, &mut op, &mut gls, monitor, &berths, &starts, obj);

    assert!(
        outcome.solution().objective_value() < obj,
        "GLS on a badly packed initial should improve: got {} vs initial {}",
        outcome.solution().objective_value(),
        obj
    );
}

#[test]
fn gls_medium_instance_respects_berth_constraints() {
    let (model, berths, starts, obj) = medium_instance();
    let mut gls = GuidedLocalSearch::new(0.5, model.num_vessels(), model.num_berths());
    let mut op = build_full_operator();
    let monitor = IterationLimitMonitor::new(20_000);

    let outcome = run_search(&model, &mut op, &mut gls, monitor, &berths, &starts, obj);

    let sol = outcome.solution();
    assert!(sol.objective_value() <= obj);

    for v in 0..model.num_vessels() {
        let b = sol.berths()[v];
        assert!(
            model.vessel_allowed_on_berth(VesselIndex::new(v), b),
            "V{v} assigned to forbidden berth B{}",
            b.get()
        );
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  Monitor Termination Tests
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn iteration_limit_terminates_correctly() {
    let (model, berths, starts, obj) = small_instance();
    let cooling = GeometricCooling::new(100.0, 0.999, 0.01);
    let mut sa = SimulatedAnnealing::new(cooling, StdRng::seed_from_u64(1));
    let mut op = build_full_operator();
    let limit = 500;
    let monitor = IterationLimitMonitor::new(limit);

    let outcome = run_search(&model, &mut op, &mut sa, monitor, &berths, &starts, obj);

    assert_eq!(
        outcome.termination_reason(),
        TerminationReason::IterationLimitReached
    );
    assert!(outcome.stats().iterations <= limit);
}

#[test]
fn cycle_limit_terminates_correctly() {
    let (model, berths, starts, obj) = small_instance();
    let mut ts = TabuSearch::new(FixedTenure::new(5), model.num_vessels(), model.num_berths());
    let mut op = build_full_operator();
    let limit = 10;
    let monitor = CycleLimitMonitor::new(limit);

    let outcome = run_search(&model, &mut op, &mut ts, monitor, &berths, &starts, obj);

    assert_eq!(
        outcome.termination_reason(),
        TerminationReason::CycleLimitReached
    );
    assert!(outcome.stats().cycles <= limit);
}

#[test]
fn time_limit_terminates_correctly() {
    let (model, berths, starts, obj) = medium_instance();
    let cooling = GeometricCooling::new(200.0, 0.999, 0.001);
    let mut sa = SimulatedAnnealing::new(cooling, StdRng::seed_from_u64(99)).with_reheat(1.5);
    let mut op = build_full_operator();
    // Use a very short time limit; reheat prevents neighborhood exhaustion.
    let monitor = TimeLimitMonitor::new(Duration::from_millis(50));

    let outcome = run_search(&model, &mut op, &mut sa, monitor, &berths, &starts, obj);

    assert_eq!(
        outcome.termination_reason(),
        TerminationReason::TimeLimitReached
    );
}

#[test]
fn no_improvement_iteration_patience_terminates() {
    let (model, berths, starts, obj) = tiny_instance();
    let cooling = GeometricCooling::new(0.001, 0.999, 0.0001);
    let mut sa = SimulatedAnnealing::new(cooling, StdRng::seed_from_u64(55));
    let mut op = build_swap_operator();
    let monitor = NoImprovementMonitor::with_iteration_patience(200);

    let outcome = run_search(&model, &mut op, &mut sa, monitor, &berths, &starts, obj);

    assert_eq!(
        outcome.termination_reason(),
        TerminationReason::MaxNonImprovingIterations
    );
}

#[test]
fn composite_monitor_first_limit_wins() {
    let (model, berths, starts, obj) = small_instance();
    let cooling = GeometricCooling::new(100.0, 0.999, 0.01);
    let mut sa = SimulatedAnnealing::new(cooling, StdRng::seed_from_u64(77));
    let mut op = build_full_operator();

    let mut monitor = CompositeLocalSearchMonitor::new();
    monitor.add_monitor(IterationLimitMonitor::new(200));
    monitor.add_monitor(TimeLimitMonitor::new(Duration::from_secs(60)));

    let outcome = run_search(&model, &mut op, &mut sa, monitor, &berths, &starts, obj);

    // The iteration limit (200) should fire well before the 60s time limit.
    assert_eq!(
        outcome.termination_reason(),
        TerminationReason::IterationLimitReached
    );
}

// ════════════════════════════════════════════════════════════════════════════
//  Operator Variation Tests
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn swap_only_operator_improves_small_instance() {
    let (model, berths, starts, obj) = small_instance();
    let cooling = GeometricCooling::new(200.0, 0.995, 0.01);
    let mut sa = SimulatedAnnealing::new(cooling, StdRng::seed_from_u64(42));
    let mut op = build_swap_operator();
    let monitor = IterationLimitMonitor::new(5_000);

    let outcome = run_search(&model, &mut op, &mut sa, monitor, &berths, &starts, obj);

    assert!(
        outcome.solution().objective_value() < obj,
        "Swap-only operator should still improve a packed initial: got {} vs {}",
        outcome.solution().objective_value(),
        obj
    );
}

#[test]
fn single_intra_berth_swap_operator() {
    let (model, berths, starts, obj) = small_instance();
    let mut ts = TabuSearch::new(FixedTenure::new(5), model.num_vessels(), model.num_berths());
    let mut op = IntraBerthSwapOperator::new(|v_a, v_b, sol, graph, model| unsafe {
        intra_berth_swap_filter_unchecked(v_a, v_b, sol, graph, model)
    });
    let monitor = CycleLimitMonitor::new(50);

    let outcome = run_search(&model, &mut op, &mut ts, monitor, &berths, &starts, obj);

    // With all vessels on B0, intra-berth swaps should be able to reorder.
    assert!(
        outcome.solution().objective_value() <= obj,
        "Single intra-berth swap should not worsen: got {} vs {}",
        outcome.solution().objective_value(),
        obj
    );
}

#[test]
fn single_inter_berth_swap_operator_spreads_load() {
    let (model, berths, starts, obj) = small_instance();
    let mut ts = TabuSearch::new(FixedTenure::new(5), model.num_vessels(), model.num_berths());
    let mut op = InterBerthSwapOperator::new(|v_a, v_b, sol, graph, model| unsafe {
        inter_berth_swap_filter_unchecked(v_a, v_b, sol, graph, model)
    });

    let mut monitor = CompositeLocalSearchMonitor::new();
    monitor.add_monitor(CycleLimitMonitor::new(100));
    monitor.add_monitor(IterationLimitMonitor::new(20_000));

    let outcome = run_search(&model, &mut op, &mut ts, monitor, &berths, &starts, obj);

    assert!(
        outcome.solution().objective_value() <= obj,
        "Inter-berth swap should not worsen: got {} vs {}",
        outcome.solution().objective_value(),
        obj
    );
}

// ════════════════════════════════════════════════════════════════════════════
//  Determinism & Reproducibility Tests
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn sa_deterministic_with_same_seed() {
    let (model, berths, starts, obj) = small_instance();

    let run = |seed: u64| {
        let cooling = GeometricCooling::new(100.0, 0.99, 0.01);
        let mut sa = SimulatedAnnealing::new(cooling, StdRng::seed_from_u64(seed));
        let mut op = build_full_operator();
        let monitor = IterationLimitMonitor::new(2_000);
        run_search(&model, &mut op, &mut sa, monitor, &berths, &starts, obj)
    };

    let outcome_a = run(42);
    let outcome_b = run(42);

    assert_eq!(
        outcome_a.solution().objective_value(),
        outcome_b.solution().objective_value(),
        "Same seed must produce identical results"
    );
    assert_eq!(outcome_a.solution().berths(), outcome_b.solution().berths(),);
    assert_eq!(
        outcome_a.solution().start_times(),
        outcome_b.solution().start_times(),
    );
}

#[test]
fn sa_different_seeds_can_differ() {
    let (model, berths, starts, obj) = small_instance();

    let run = |seed: u64| {
        let cooling = GeometricCooling::new(100.0, 0.99, 0.01);
        let mut sa = SimulatedAnnealing::new(cooling, StdRng::seed_from_u64(seed));
        let mut op = build_full_operator();
        let monitor = IterationLimitMonitor::new(2_000);
        run_search(&model, &mut op, &mut sa, monitor, &berths, &starts, obj)
    };

    let outcome_a = run(1);
    let outcome_b = run(999);

    // They *can* differ — we just check both are valid (≤ initial).
    assert!(outcome_a.solution().objective_value() <= obj);
    assert!(outcome_b.solution().objective_value() <= obj);
}

// ════════════════════════════════════════════════════════════════════════════
//  Solution Feasibility Invariant Tests
// ════════════════════════════════════════════════════════════════════════════

/// Verifies that the returned solution is globally feasible:
/// - Every vessel is on an allowed berth.
/// - Start times respect arrival times.
/// - Completion does not exceed deadline.
/// - The objective value matches a recomputation.
fn assert_solution_feasible(
    model: &Model<i64>,
    outcome: &talos_ls::outcome::LocalSearchOutcome<i64>,
) {
    let sol = outcome.solution();
    let n = model.num_vessels();
    assert_eq!(sol.berths().len(), n);
    assert_eq!(sol.start_times().len(), n);

    let mut recomputed_obj = 0i64;
    for v in 0..n {
        let vi = VesselIndex::new(v);
        let bi = sol.berths()[v];
        let st = sol.start_times()[v];

        assert!(
            model.vessel_allowed_on_berth(vi, bi),
            "V{v} on forbidden berth B{}",
            bi.get()
        );
        assert!(
            st >= model.vessel_arrival_time(vi),
            "V{v} starts at {st} before arrival {}",
            model.vessel_arrival_time(vi)
        );

        let pt = model.vessel_processing_time(vi, bi);
        assert!(
            !pt.is_none(),
            "V{v} has no processing time on B{}",
            bi.get()
        );
        let completion = st + pt.unwrap_unchecked();
        assert!(
            completion <= model.vessel_latest_departure_time(vi),
            "V{v} completes at {completion} past deadline {}",
            model.vessel_latest_departure_time(vi)
        );

        recomputed_obj += calculate_weighted_completion_time(model, vi, bi, st)
            .expect("feasible assignment must have a cost");
    }

    assert_eq!(
        sol.objective_value(),
        recomputed_obj,
        "Stored objective {} != recomputed {}",
        sol.objective_value(),
        recomputed_obj
    );
}

#[test]
fn sa_solution_feasibility_small() {
    let (model, berths, starts, obj) = small_instance();
    let cooling = GeometricCooling::new(200.0, 0.995, 0.01);
    let mut sa = SimulatedAnnealing::new(cooling, StdRng::seed_from_u64(42));
    let mut op = build_full_operator();
    let monitor = IterationLimitMonitor::new(5_000);

    let outcome = run_search(&model, &mut op, &mut sa, monitor, &berths, &starts, obj);
    assert_solution_feasible(&model, &outcome);
}

#[test]
fn tabu_solution_feasibility_medium() {
    let (model, berths, starts, obj) = medium_instance();
    let mut ts = TabuSearch::new(FixedTenure::new(8), model.num_vessels(), model.num_berths());
    let mut op = build_full_operator();
    let monitor = CycleLimitMonitor::new(100);

    let outcome = run_search(&model, &mut op, &mut ts, monitor, &berths, &starts, obj);
    assert_solution_feasible(&model, &outcome);
}

#[test]
fn gls_solution_feasibility_medium() {
    let (model, berths, starts, obj) = medium_instance();
    let mut gls = GuidedLocalSearch::new(0.5, model.num_vessels(), model.num_berths());
    let mut op = build_full_operator();
    let monitor = IterationLimitMonitor::new(10_000);

    let outcome = run_search(&model, &mut op, &mut gls, monitor, &berths, &starts, obj);
    assert_solution_feasible(&model, &outcome);
}

// ════════════════════════════════════════════════════════════════════════════
//  Callback & Statistics Tests
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn new_best_callback_receives_improving_solutions() {
    let (model, berths, starts, obj) = small_instance();
    let cooling = GeometricCooling::new(200.0, 0.995, 0.01);
    let mut sa = SimulatedAnnealing::new(cooling, StdRng::seed_from_u64(42));
    let mut op = build_full_operator();
    let monitor = IterationLimitMonitor::new(5_000);

    let mut best_values = Vec::new();
    let mut engine = Engine::<i64>::new(model.num_vessels(), model.num_berths());
    let params = MutableLocalSearchParams {
        model: &model,
        operator: &mut op,
        metaheuristic: &mut sa,
        monitor,
        berths: &berths,
        start_times: &starts,
        objective_value: obj,
    };
    engine.run(params, evaluator, |sol| {
        best_values.push(sol.objective_value());
    });

    // Every callback should report a strictly improving value.
    for window in best_values.windows(2) {
        assert!(
            window[1] < window[0],
            "Best callback values must be strictly decreasing: {:?}",
            best_values
        );
    }
}

#[test]
fn statistics_counters_are_nonzero_after_search() {
    let (model, berths, starts, obj) = small_instance();
    let cooling = GeometricCooling::new(100.0, 0.99, 0.01);
    let mut sa = SimulatedAnnealing::new(cooling, StdRng::seed_from_u64(42));
    let mut op = build_full_operator();
    let monitor = IterationLimitMonitor::new(1_000);

    let outcome = run_search(&model, &mut op, &mut sa, monitor, &berths, &starts, obj);
    let stats = outcome.stats();

    assert!(stats.iterations > 0, "should have run iterations");
    assert!(stats.total_solutions > 0, "should have generated solutions");
    assert!(
        stats.time_total > Duration::ZERO,
        "should have elapsed time"
    );
}

// ════════════════════════════════════════════════════════════════════════════
//  Engine Reuse Tests
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn engine_can_be_reused_across_runs() {
    let (model, berths, starts, obj) = small_instance();
    let mut engine = Engine::<i64>::new(model.num_vessels(), model.num_berths());

    // First run: SA
    let cooling = GeometricCooling::new(100.0, 0.99, 0.01);
    let mut sa = SimulatedAnnealing::new(cooling, StdRng::seed_from_u64(42));
    let mut op = build_full_operator();
    let params = MutableLocalSearchParams {
        model: &model,
        operator: &mut op,
        metaheuristic: &mut sa,
        monitor: IterationLimitMonitor::new(1_000),
        berths: &berths,
        start_times: &starts,
        objective_value: obj,
    };
    let outcome1 = engine.run(params, evaluator, |_| {});
    assert!(outcome1.solution().objective_value() <= obj);

    // Second run: Tabu Search (same engine, same instance)
    let mut ts = TabuSearch::new(FixedTenure::new(5), model.num_vessels(), model.num_berths());
    let mut op2 = build_full_operator();
    let params2 = MutableLocalSearchParams {
        model: &model,
        operator: &mut op2,
        metaheuristic: &mut ts,
        monitor: CycleLimitMonitor::new(50),
        berths: &berths,
        start_times: &starts,
        objective_value: obj,
    };
    let outcome2 = engine.run(params2, evaluator, |_| {});
    assert!(outcome2.solution().objective_value() <= obj);
}

// ════════════════════════════════════════════════════════════════════════════
//  Validated Params Tests
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn validated_params_run_succeeds() {
    let (model, berths, starts, obj) = small_instance();
    let cooling = GeometricCooling::new(100.0, 0.99, 0.01);
    let mut sa = SimulatedAnnealing::new(cooling, StdRng::seed_from_u64(42));
    let mut op = build_full_operator();
    let monitor = IterationLimitMonitor::new(500);

    let params = talos_ls::params::LocalSearchParams::new(
        &model, &mut op, &mut sa, monitor, &berths, &starts, obj,
    )
    .expect("valid params should not error");

    let mut engine = Engine::<i64>::new(model.num_vessels(), model.num_berths());
    let outcome = engine.run(params.into(), evaluator, |_| {});

    assert!(outcome.solution().objective_value() <= obj);
}

#[test]
fn validated_params_rejects_mismatched_dimensions() {
    let (model, _, _, _) = small_instance();
    let cooling = GeometricCooling::new(100.0, 0.99, 0.01);
    let mut sa = SimulatedAnnealing::new(cooling, StdRng::seed_from_u64(42));
    let mut op = build_full_operator();
    let monitor = IterationLimitMonitor::new(500);

    // Wrong number of berth assignments (3 instead of 5).
    let bad_berths = vec![BerthIndex::new(0); 3];
    let bad_starts = vec![0i64; 3];

    let result = talos_ls::params::LocalSearchParams::new(
        &model,
        &mut op,
        &mut sa,
        monitor,
        &bad_berths,
        &bad_starts,
        0,
    );
    assert!(result.is_err(), "mismatched dimensions should be rejected");
}
