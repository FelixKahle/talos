// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::loading::{InstanceLoader, ProblemLoaderError};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;
use talos_ls::{
    DynamicLambda, Engine, GuidedLocalSearch, Lambda, LocalSearchParameters, Metaheuristic,
    NoMonitor, PenalizationTrigger, SearchLimits, heuristic_lambda,
};
use talos_model::Model;

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
    const MAX_TIME: std::time::Duration = std::time::Duration::from_secs(360);

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

fn run_benchmark(
    config: &InstanceBenchmarkConfig,
    data_dir: &std::path::Path,
) -> InstanceBenchmarkResult {
    let instance_path = data_dir.join(&config.filename);

    let model = load_instance(&instance_path)
        .unwrap_or_else(|e| panic!("Failed to load {}: {e}", config.filename));

    let initial = edf::generate_greedy_edf_schedule(&model)
        .expect("EDF heuristic failed to find a feasible schedule");
    let greedy_obj = initial.objective_value();

    let lambda_01 = heuristic_lambda(
        greedy_obj as f64,
        model.num_vessels() * model.num_berths(),
        0.1,
    );
    let lambda_02 = heuristic_lambda(
        greedy_obj as f64,
        model.num_vessels() * model.num_berths(),
        0.2,
    );
    let lambda_03 = heuristic_lambda(
        greedy_obj as f64,
        model.num_vessels() * model.num_berths(),
        0.3,
    );

    let guided_local_search = Metaheuristic::GuidedLocalSearch(GuidedLocalSearch {
        lambda: Lambda::Dynamic(DynamicLambda::new(
            lambda_02, // Initial (Alpha 0.2)
            0.1,       // Increase step
            0.1,       // Decay step
            lambda_01, // Min (Alpha 0.1)
            lambda_03, // Max (Alpha 0.3)
        )),
        trigger: PenalizationTrigger::OnExhaustion,
        ..GuidedLocalSearch::default()
    });
    let mut engine = Engine::<i64>::new();

    let parameters = LocalSearchParameters {
        metaheuristic: guided_local_search,
        limits: SearchLimits {
            time: Some(config.first_run_time_limit),
            ..SearchLimits::default()
        },
        ..LocalSearchParameters::default()
    };
    let started = Instant::now();
    let outcome = engine
        .run(&model, initial.as_view(), &parameters, &mut NoMonitor)
        .expect("the EDF schedule should be a feasible schedule of the model");
    let first_time = started.elapsed();
    let first_obj = outcome.solution.objective_value();
    let first_improvement = (1.0 - first_obj as f64 / greedy_obj as f64) * 100.0;

    let parameters = LocalSearchParameters {
        metaheuristic: guided_local_search,
        limits: SearchLimits {
            time: Some(config.second_run_time_limit),
            non_improving_time: Some(config.non_improving_time_limit),
            ..SearchLimits::default()
        },
        ..LocalSearchParameters::default()
    };
    let started = Instant::now();
    let outcome = engine
        .run(&model, initial.as_view(), &parameters, &mut NoMonitor)
        .expect("the EDF schedule should be a feasible schedule of the model");
    let second_time = started.elapsed();
    let second_obj = outcome.solution.objective_value();
    let second_improvement = (1.0 - second_obj as f64 / greedy_obj as f64) * 100.0;

    println!(
        "  {:<18} {:>16} {:>16} {:>10.2}% {:>14.2?} {:>16} {:>10.2}% {:>14.2?}",
        config.filename,
        greedy_obj,
        first_obj,
        first_improvement,
        first_time,
        second_obj,
        second_improvement,
        second_time,
    );

    InstanceBenchmarkResult {
        filename: config.filename.clone(),
        greedy_objective: greedy_obj,
        first_run_objective: first_obj,
        second_run_objective: second_obj,
        first_run_time: first_time,
        second_run_time: second_time,
        first_run_improvement: first_improvement,
        second_run_improvement: second_improvement,
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
    println!("  {}", "-".repeat(124));

    for config in &benchmarks {
        if config.filename != "f200x15-01.txt" {
            continue;
        }
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
