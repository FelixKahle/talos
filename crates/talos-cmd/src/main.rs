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

use crate::loading::{InstanceLoader, ProblemLoaderError};
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::time::Duration;
use talos_ls::engine::Engine;
use talos_ls::eval::calculate_weighted_turnaround_time_unchecked;
use talos_ls::meta::gd::GreedyDescent;
use talos_ls::meta::gls::{
    DynamicLambda, GuidedLocalSearch, PenalizationTrigger, heuristic_lambda,
};
use talos_ls::meta::sa::{GeometricCooling, SimulatedAnnealing};
use talos_ls::meta::tabu::{FixedTenure, TabuSearch};
use talos_ls::meta::teleport::ExhaustionTeleport;
use talos_ls::operator::composite::RoundRobinCompoundOperator;
use talos_ls::operator::filter::{
    inter_berth_shift_filter_unchecked, inter_berth_swap_filter_unchecked,
    intra_berth_shift_filter_unchecked, intra_berth_swap_filter_unchecked,
};
use talos_ls::operator::lsoperator::LocalSearchOperator;
use talos_ls::operator::shift::{InterBerthShiftOperator, IntraBerthShiftOperator};
use talos_ls::operator::swap::{InterBerthSwapOperator, IntraBerthSwapOperator};
use talos_ls::portfolio::LocalSearchSolver;
use talos_model::index::{BerthIndex, VesselIndex};
use talos_model::model::Model;
use talos_search::inc::IncumbentStore;
use talos_search::monitor::time::TimeLimitMonitor as PortfolioTimeLimitMonitor;
use talos_solver::pssolver::ParallelPortfolioSolver;

mod edf;
mod loading;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct InstanceBenchmarkConfig {
    filename: String,
    time_limit: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct InstanceBenchmarkResult {
    filename: String,
    greedy_objective: i64,
    portfolio_objective: i64,
    portfolio_time: Duration,
    improvement: f64,
}

fn setup_benchmarks() -> Vec<InstanceBenchmarkConfig> {
    const TIME_LIMIT: Duration = Duration::from_secs(120);

    let filenames = [
        "f200x15-01.txt",
        "f200x15-02.txt",
        "f200x15-03.txt",
        "f200x15-04.txt",
        "f200x15-05.txt",
        "f200x15-06.txt",
        "f200x15-07.txt",
        "f200x15-08.txt",
        "f200x15-09.txt",
        "f200x15-10.txt",
        "f250x20-01.txt",
        "f250x20-02.txt",
        "f250x20-03.txt",
        "f250x20-04.txt",
        "f250x20-05.txt",
        "f250x20-06.txt",
        "f250x20-07.txt",
        "f250x20-08.txt",
        "f250x20-09.txt",
        "f250x20-10.txt",
    ];

    filenames
        .iter()
        .map(|f| InstanceBenchmarkConfig {
            filename: f.to_string(),
            time_limit: TIME_LIMIT,
        })
        .collect()
}

fn find_instances_dir() -> Option<PathBuf> {
    let mut cur: Option<&std::path::Path> = Some(std::path::Path::new(env!("CARGO_MANIFEST_DIR")));
    while let Some(p) = cur {
        let cand = p.join("data");
        if cand.is_dir() {
            return Some(cand);
        }
        cur = p.parent();
    }
    None
}

fn load_instance(path: &std::path::Path) -> Result<Model<i64>, ProblemLoaderError> {
    let loader = InstanceLoader::<i64>::new(99999);
    loader.load_dbap_file(path)
}

fn evaluator(
    model: &Model<i64>,
    vessel: VesselIndex,
    berth: BerthIndex,
    start: i64,
) -> Option<i64> {
    unsafe { calculate_weighted_turnaround_time_unchecked(model, vessel, berth, start) }
}

fn build_full_operator() -> RoundRobinCompoundOperator<'static, i64> {
    let ops: Vec<Box<dyn LocalSearchOperator<i64>>> = vec![
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

fn run_benchmark(
    config: &InstanceBenchmarkConfig,
    data_dir: &std::path::Path,
) -> InstanceBenchmarkResult {
    let instance_path = data_dir.join(&config.filename);

    let model = load_instance(&instance_path)
        .unwrap_or_else(|e| panic!("Failed to load {}: {e}", config.filename));

    let greedy_sol = edf::generate_greedy_edf_schedule(&model)
        .expect("EDF heuristic failed to find a feasible schedule");
    let greedy_obj = greedy_sol.objective_value();

    let nv = model.num_vessels();
    let nb = model.num_berths();

    // Lambda parameters for GLS
    let lambda_01 = heuristic_lambda(greedy_obj as f64, nv * nb, 0.12);
    let lambda_02 = heuristic_lambda(greedy_obj as f64, nv * nb, 0.2);
    let lambda_03 = heuristic_lambda(greedy_obj as f64, nv * nb, 0.32);

    // SA cooling parameters from greedy objective
    let cooling = SimulatedAnnealing::<StdRng, GeometricCooling>::heuristic_geometric_params(
        greedy_obj as f64,
        0.5,
        0.05,
        0.9995,
    );

    // GLS solver
    let gls_solver = LocalSearchSolver::new(
        "GLS".to_string(),
        Engine::new(nv, nb),
        GuidedLocalSearch::new(nv, nb)
            .with_lambda_strategy(DynamicLambda::new(lambda_02, 0.12, lambda_01, lambda_03))
            .with_trigger(PenalizationTrigger::OnExhaustion)
            .with_teleport(ExhaustionTeleport::new(100)),
        build_full_operator(),
        evaluator,
        |m: &Model<i64>| edf::generate_greedy_edf_schedule(m).expect("EDF heuristic failed"),
    );

    // SA solver
    let sa_solver = LocalSearchSolver::new(
        "SA".to_string(),
        Engine::new(nv, nb),
        SimulatedAnnealing::new(cooling, StdRng::seed_from_u64(42))
            .with_reheat(1.5)
            .with_teleport(ExhaustionTeleport::new(1)),
        build_full_operator(),
        evaluator,
        |m: &Model<i64>| edf::generate_greedy_edf_schedule(m).expect("EDF heuristic failed"),
    );

    // Greedy Descent solver
    let gd_solver = LocalSearchSolver::new(
        "GD".to_string(),
        Engine::new(nv, nb),
        GreedyDescent::new().with_teleport(ExhaustionTeleport::new(1)),
        build_full_operator(),
        evaluator,
        |m: &Model<i64>| edf::generate_greedy_edf_schedule(m).expect("EDF heuristic failed"),
    );

    // Tabu Search solver
    let tabu_solver = LocalSearchSolver::new(
        "Tabu".to_string(),
        Engine::new(nv, nb),
        TabuSearch::new(FixedTenure::new(7), nv, nb).with_teleport(ExhaustionTeleport::new(100)),
        build_full_operator(),
        evaluator,
        |m: &Model<i64>| edf::generate_greedy_edf_schedule(m).expect("EDF heuristic failed"),
    );

    let oracle = IncumbentStore::new(4);
    let monitor = PortfolioTimeLimitMonitor::new(config.time_limit);

    let mut portfolio = ParallelPortfolioSolver::with_capacity(oracle, 4)
        .with_solver(gls_solver)
        .with_solver(sa_solver)
        .with_solver(gd_solver)
        .with_solver(tabu_solver);

    let start = std::time::Instant::now();
    let best = portfolio.solve(&model, monitor);
    let elapsed = start.elapsed();

    let best_obj = best.objective_value();
    let improvement = (1.0 - best_obj as f64 / greedy_obj as f64) * 100.0;

    println!(
        "  {:<18} {:>16} {:>16} {:>10.2}% {:>14.2?}",
        config.filename, greedy_obj, best_obj, improvement, elapsed,
    );

    InstanceBenchmarkResult {
        filename: config.filename.clone(),
        greedy_objective: greedy_obj,
        portfolio_objective: best_obj,
        portfolio_time: elapsed,
        improvement,
    }
}

fn main() {
    let data_dir = find_instances_dir().expect("Could not find the 'data' directory.");
    let results_dir = data_dir.parent().unwrap().join("results");
    fs::create_dir_all(&results_dir).expect("Failed to create results directory");

    let benchmarks = setup_benchmarks();
    let mut all_results = Vec::with_capacity(benchmarks.len());

    println!(
        "  {:<18} {:>16} {:>16} {:>11} {:>14}",
        "Instance", "EDF", "Portfolio", "Impr%", "Time"
    );
    println!("  {}", "-".repeat(78));

    for config in &benchmarks {
        let result = run_benchmark(config, &data_dir);

        // Write per-instance JSON
        let stem = config
            .filename
            .strip_suffix(".txt")
            .unwrap_or(&config.filename);
        let per_file = results_dir.join(format!("{stem}.json"));
        let json = serde_json::to_string_pretty(&result).expect("Failed to serialize result");
        fs::write(&per_file, &json).unwrap_or_else(|e| {
            panic!("Failed to write {}: {e}", per_file.display());
        });

        all_results.push(result);
    }

    // Write combined JSON
    let combined_path = results_dir.join("all_results.json");
    let combined_json =
        serde_json::to_string_pretty(&all_results).expect("Failed to serialize all results");
    fs::write(&combined_path, &combined_json).unwrap_or_else(|e| {
        panic!("Failed to write {}: {e}", combined_path.display());
    });

    println!("\nResults written to {}", results_dir.display());
}
