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
use talos_ls::meta::gls::{DynamicLambda, GuidedLocalSearch, heuristic_lambda};
use talos_ls::meta::sa::{GeometricCooling, SimulatedAnnealing};
use talos_ls::meta::teleport::{ExhaustionTeleport, NoTeleport, StagnationTeleport};
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
use talos_solver::pssolver::ParallelPortfolioSolver;

mod edf;
mod loading;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct InstanceBenchmarkConfig {
    filename: String,
    first_run_time_limit: std::time::Duration,
    second_run_time_limit: std::time::Duration,
    non_improving_time_limit: std::time::Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct InstanceBenchmarkResult {
    filename: String,
    greedy_objective: i64,
    first_run_objective: i64,
    second_run_objective: i64,
    first_run_time: std::time::Duration,
    second_run_time: std::time::Duration,
    first_run_improvement: f64,
    second_run_improvement: f64,
}

fn setup_benchmarks() -> Vec<InstanceBenchmarkConfig> {
    const MAX_TIME: std::time::Duration = std::time::Duration::from_secs(600);

    vec![
        InstanceBenchmarkConfig {
            filename: "f200x15-01.txt".to_string(),
            first_run_time_limit: std::time::Duration::from_millis(60_700),
            second_run_time_limit: MAX_TIME,
            non_improving_time_limit: std::time::Duration::from_secs(120),
        },
        InstanceBenchmarkConfig {
            filename: "f200x15-02.txt".to_string(),
            first_run_time_limit: std::time::Duration::from_millis(60_000),
            second_run_time_limit: MAX_TIME,
            non_improving_time_limit: std::time::Duration::from_secs(120),
        },
        InstanceBenchmarkConfig {
            filename: "f200x15-03.txt".to_string(),
            first_run_time_limit: std::time::Duration::from_millis(37_100),
            second_run_time_limit: MAX_TIME,
            non_improving_time_limit: std::time::Duration::from_secs(120),
        },
        InstanceBenchmarkConfig {
            filename: "f200x15-04.txt".to_string(),
            first_run_time_limit: std::time::Duration::from_millis(34_700),
            second_run_time_limit: MAX_TIME,
            non_improving_time_limit: std::time::Duration::from_secs(120),
        },
        InstanceBenchmarkConfig {
            filename: "f200x15-05.txt".to_string(),
            first_run_time_limit: std::time::Duration::from_millis(35_700),
            second_run_time_limit: MAX_TIME,
            non_improving_time_limit: std::time::Duration::from_secs(120),
        },
        InstanceBenchmarkConfig {
            filename: "f200x15-06.txt".to_string(),
            first_run_time_limit: std::time::Duration::from_millis(36_900),
            second_run_time_limit: MAX_TIME,
            non_improving_time_limit: std::time::Duration::from_secs(120),
        },
        InstanceBenchmarkConfig {
            filename: "f200x15-07.txt".to_string(),
            first_run_time_limit: std::time::Duration::from_millis(35_500),
            second_run_time_limit: MAX_TIME,
            non_improving_time_limit: std::time::Duration::from_secs(120),
        },
        InstanceBenchmarkConfig {
            filename: "f200x15-08.txt".to_string(),
            first_run_time_limit: std::time::Duration::from_millis(36_100),
            second_run_time_limit: MAX_TIME,
            non_improving_time_limit: std::time::Duration::from_secs(120),
        },
        InstanceBenchmarkConfig {
            filename: "f200x15-09.txt".to_string(),
            first_run_time_limit: std::time::Duration::from_millis(35_500),
            second_run_time_limit: MAX_TIME,
            non_improving_time_limit: std::time::Duration::from_secs(120),
        },
        InstanceBenchmarkConfig {
            filename: "f200x15-10.txt".to_string(),
            first_run_time_limit: std::time::Duration::from_millis(35_000),
            second_run_time_limit: MAX_TIME,
            non_improving_time_limit: std::time::Duration::from_secs(120),
        },
        InstanceBenchmarkConfig {
            filename: "f250x20-01.txt".to_string(),
            first_run_time_limit: std::time::Duration::from_millis(78_000),
            second_run_time_limit: MAX_TIME,
            non_improving_time_limit: std::time::Duration::from_secs(120),
        },
        InstanceBenchmarkConfig {
            filename: "f250x20-02.txt".to_string(),
            first_run_time_limit: std::time::Duration::from_millis(84_000),
            second_run_time_limit: MAX_TIME,
            non_improving_time_limit: std::time::Duration::from_secs(120),
        },
        InstanceBenchmarkConfig {
            filename: "f250x20-03.txt".to_string(),
            first_run_time_limit: std::time::Duration::from_millis(77_900),
            second_run_time_limit: MAX_TIME,
            non_improving_time_limit: std::time::Duration::from_secs(120),
        },
        InstanceBenchmarkConfig {
            filename: "f250x20-04.txt".to_string(),
            first_run_time_limit: std::time::Duration::from_millis(83_000),
            second_run_time_limit: MAX_TIME,
            non_improving_time_limit: std::time::Duration::from_secs(120),
        },
        InstanceBenchmarkConfig {
            filename: "f250x20-05.txt".to_string(),
            first_run_time_limit: std::time::Duration::from_millis(77_300),
            second_run_time_limit: MAX_TIME,
            non_improving_time_limit: std::time::Duration::from_secs(120),
        },
        InstanceBenchmarkConfig {
            filename: "f250x20-06.txt".to_string(),
            first_run_time_limit: std::time::Duration::from_millis(82_600),
            second_run_time_limit: MAX_TIME,
            non_improving_time_limit: std::time::Duration::from_secs(120),
        },
        InstanceBenchmarkConfig {
            filename: "f250x20-07.txt".to_string(),
            first_run_time_limit: std::time::Duration::from_millis(84_100),
            second_run_time_limit: MAX_TIME,
            non_improving_time_limit: std::time::Duration::from_secs(120),
        },
        InstanceBenchmarkConfig {
            filename: "f250x20-08.txt".to_string(),
            first_run_time_limit: std::time::Duration::from_millis(79_400),
            second_run_time_limit: MAX_TIME,
            non_improving_time_limit: std::time::Duration::from_secs(120),
        },
        InstanceBenchmarkConfig {
            filename: "f250x20-09.txt".to_string(),
            first_run_time_limit: std::time::Duration::from_millis(82_500),
            second_run_time_limit: MAX_TIME,
            non_improving_time_limit: std::time::Duration::from_secs(120),
        },
        InstanceBenchmarkConfig {
            filename: "f250x20-10.txt".to_string(),
            first_run_time_limit: std::time::Duration::from_millis(81_000),
            second_run_time_limit: MAX_TIME,
            non_improving_time_limit: std::time::Duration::from_secs(120),
        },
    ]
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

    // Generate initial solution using Earliest Deadline First
    let greedy_sol = edf::generate_greedy_edf_schedule(&model)
        .expect("EDF heuristic failed to find a feasible schedule");
    let greedy_obj = greedy_sol.objective_value();

    let nv = model.num_vessels();
    let nb = model.num_berths();

    // Shared closure for worker initialization
    let init_edf = |m: &Model<i64>| edf::generate_greedy_edf_schedule(m).expect("EDF failed");

    // --- Parameter Calculation ---
    let lambda_01 = heuristic_lambda(greedy_obj as f64, nv * nb, 0.1);
    let lambda_02 = heuristic_lambda(greedy_obj as f64, nv * nb, 0.2);
    let lambda_03 = heuristic_lambda(greedy_obj as f64, nv * nb, 0.3);

    let lambda_04 = heuristic_lambda(greedy_obj as f64, nv * nb, 0.1);
    let lambda_05 = heuristic_lambda(greedy_obj as f64, nv * nb, 0.3);
    let lambda_06 = heuristic_lambda(greedy_obj as f64, nv * nb, 0.5);

    let gls_titan = LocalSearchSolver::new(
        "GLS_Titan".to_string(),
        Engine::new(nv, nb),
        GuidedLocalSearch::new(nv, nb)
            .with_lambda_strategy(DynamicLambda::new(lambda_02, 0.1, lambda_01, lambda_03))
            .with_teleport(NoTeleport),
        build_full_operator(),
        evaluator,
        init_edf,
    );

    let gls_titan2 = LocalSearchSolver::new(
        "GLS_Titan".to_string(),
        Engine::new(nv, nb),
        GuidedLocalSearch::new(nv, nb)
            .with_lambda_strategy(DynamicLambda::new(lambda_05, 0.2, lambda_04, lambda_06))
            .with_teleport(NoTeleport),
        build_full_operator(),
        evaluator,
        init_edf,
    );

    let sa_scavenger = {
        // We want this agent to be extremely agile.
        // It accepts bad moves easily (high T) so it can "vibrate" out of
        // the Titan's current local minimum.
        let cooling_agile =
            SimulatedAnnealing::<StdRng, GeometricCooling>::heuristic_geometric_params(
                greedy_obj as f64,
                0.4,  // 40% initial acceptance - "Warm" but not boiling
                0.05, // Sensitive to small changes
                0.99, // Fast cooling to settle quickly after a jump
            );

        LocalSearchSolver::new(
            "SA_Scavenger".to_string(),
            Engine::new(nv, nb),
            SimulatedAnnealing::new(cooling_agile, StdRng::seed_from_u64(420))
                .with_reheat(1.8)
                // THE TRICK: Stagnation Teleport is short.
                // Every 8 seconds of no improvement, it "looks up" at the Titans' work
                // and jumps to whoever is winning.
                .with_teleport(StagnationTeleport::new(Duration::from_secs(8))),
            build_full_operator(),
            evaluator,
            init_edf,
        )
    };

    let gd_solver = LocalSearchSolver::new(
        "GD".to_string(),
        Engine::new(nv, nb),
        GreedyDescent::new().with_teleport(ExhaustionTeleport::instant()),
        build_full_operator(),
        evaluator,
        |m: &Model<i64>| edf::generate_greedy_edf_schedule(m).expect("EDF heuristic failed"),
    );

    // SHUT DOWN the others to give these 5 agents max CPU cache and thermal headroom.
    // Set IncumbentStore to 2 to keep the quality extreme.
    let oracle = IncumbentStore::new(1);

    let mut portfolio = ParallelPortfolioSolver::with_capacity(oracle, 5)
        .with_solver(gls_titan)
        .with_solver(gls_titan2)
        .with_solver(sa_scavenger)
        .with_solver(gd_solver);

    // --- Run 1 ---
    let start1 = std::time::Instant::now();
    let best1 = portfolio.solve(
        &model,
        config.first_run_time_limit,
        config.non_improving_time_limit,
    );
    let elapsed1 = start1.elapsed();
    let first_obj = best1.objective_value();
    let first_impr = (1.0 - first_obj as f64 / greedy_obj as f64) * 100.0;

    // --- Rebuild portfolio for Run 2 (fresh EDF start) ---
    let init_edf2 = |m: &Model<i64>| edf::generate_greedy_edf_schedule(m).expect("EDF failed");

    let lambda_01 = heuristic_lambda(greedy_obj as f64, nv * nb, 0.1);
    let lambda_02 = heuristic_lambda(greedy_obj as f64, nv * nb, 0.2);
    let lambda_03 = heuristic_lambda(greedy_obj as f64, nv * nb, 0.3);
    let lambda_04 = heuristic_lambda(greedy_obj as f64, nv * nb, 0.1);
    let lambda_05 = heuristic_lambda(greedy_obj as f64, nv * nb, 0.3);
    let lambda_06 = heuristic_lambda(greedy_obj as f64, nv * nb, 0.5);

    let gls_titan_r2 = LocalSearchSolver::new(
        "GLS_Titan".to_string(),
        Engine::new(nv, nb),
        GuidedLocalSearch::new(nv, nb)
            .with_lambda_strategy(DynamicLambda::new(lambda_02, 0.1, lambda_01, lambda_03))
            .with_teleport(NoTeleport),
        build_full_operator(),
        evaluator,
        init_edf2,
    );

    let gls_titan2_r2 = LocalSearchSolver::new(
        "GLS_Titan".to_string(),
        Engine::new(nv, nb),
        GuidedLocalSearch::new(nv, nb)
            .with_lambda_strategy(DynamicLambda::new(lambda_05, 0.2, lambda_04, lambda_06))
            .with_teleport(NoTeleport),
        build_full_operator(),
        evaluator,
        init_edf2,
    );

    let sa_scavenger_r2 = {
        let cooling_agile =
            SimulatedAnnealing::<StdRng, GeometricCooling>::heuristic_geometric_params(
                greedy_obj as f64,
                0.4,
                0.05,
                0.99,
            );

        LocalSearchSolver::new(
            "SA_Scavenger".to_string(),
            Engine::new(nv, nb),
            SimulatedAnnealing::new(cooling_agile, StdRng::seed_from_u64(420))
                .with_reheat(1.8)
                .with_teleport(StagnationTeleport::new(Duration::from_secs(8))),
            build_full_operator(),
            evaluator,
            init_edf2,
        )
    };

    let gd_solver_r2 = LocalSearchSolver::new(
        "GD".to_string(),
        Engine::new(nv, nb),
        GreedyDescent::new().with_teleport(ExhaustionTeleport::instant()),
        build_full_operator(),
        evaluator,
        |m: &Model<i64>| edf::generate_greedy_edf_schedule(m).expect("EDF heuristic failed"),
    );

    let oracle2 = IncumbentStore::new(1);
    let mut portfolio2 = ParallelPortfolioSolver::with_capacity(oracle2, 5)
        .with_solver(gls_titan_r2)
        .with_solver(gls_titan2_r2)
        .with_solver(sa_scavenger_r2)
        .with_solver(gd_solver_r2);

    let start2 = std::time::Instant::now();
    let best2 = portfolio2.solve(
        &model,
        config.second_run_time_limit,
        config.non_improving_time_limit,
    );
    let elapsed2 = start2.elapsed();
    let second_obj = best2.objective_value();
    let second_impr = (1.0 - second_obj as f64 / greedy_obj as f64) * 100.0;

    println!(
        "  {:<18} {:>16} {:>16} {:>10.2}% {:>14.2?} {:>16} {:>10.2}% {:>14.2?}",
        config.filename,
        greedy_obj,
        first_obj,
        first_impr,
        elapsed1,
        second_obj,
        second_impr,
        elapsed2,
    );

    InstanceBenchmarkResult {
        filename: config.filename.clone(),
        greedy_objective: greedy_obj,
        first_run_objective: first_obj,
        second_run_objective: second_obj,
        first_run_time: elapsed1,
        second_run_time: elapsed2,
        first_run_improvement: first_impr,
        second_run_improvement: second_impr,
    }
}

fn main() {
    let data_dir = find_instances_dir().expect("Could not find the 'data' directory.");
    let results_dir = data_dir.parent().unwrap().join("results");
    fs::create_dir_all(&results_dir).expect("Failed to create results directory");

    let benchmarks = setup_benchmarks();
    let mut all_results = Vec::with_capacity(benchmarks.len());

    println!(
        "  {:<18} {:>16} {:>16} {:>11} {:>14} {:>16} {:>11} {:>14}",
        "Instance", "EDF", "Run1 Obj", "Impr%", "Time", "Run2 Obj", "Impr%", "Time"
    );
    println!("  {}", "-".repeat(120));

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
