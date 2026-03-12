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
use talos_ls::engine::Engine;
use talos_ls::eval::calculate_weighted_turnaround_time_unchecked;
use talos_ls::meta::gls::{GuidedLocalSearch, ReactiveLambda, heuristic_lambda};
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
use talos_model::model::Model;

mod edf;
mod loading;

fn find_instances_dir() -> Option<std::path::PathBuf> {
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

fn main() {
    let data_dir = find_instances_dir().expect("Could not find the 'data' directory.");
    let target_filename = "f250x20-02.txt";
    let instance_path = data_dir.join(target_filename);

    println!("Loading instance: {}", instance_path.display());
    let model = load_instance(&instance_path).expect("Failed to load instance");
    println!(
        "Loaded {} vessels, {} berths",
        model.num_vessels(),
        model.num_berths()
    );

    // Build initial solution via EDF greedy heuristic.
    let initial = edf::generate_greedy_edf_schedule(&model)
        .expect("EDF heuristic failed to find a feasible schedule");
    let init_obj = initial.objective_value();
    println!("EDF initial objective: {}", init_obj);

    // Set up GLS.
    let alpha = 0.1;
    let lambda = heuristic_lambda(
        init_obj as f64,
        model.num_vessels() * model.num_berths(),
        alpha,
    );
    let mut gls =
        GuidedLocalSearch::new(lambda, model.num_vessels(), model.num_berths()).with_lambda(
            ReactiveLambda::new(lambda, 1.02, 0.2, lambda - lambda * 0.9, lambda * 1.4),
        );
    let mut operator = build_full_operator();
    let monitor = TimeLimitMonitor::new(std::time::Duration::from_secs(120));

    // Run the engine.
    let mut engine = Engine::<i64>::new(model.num_vessels(), model.num_berths());
    let params = MutableLocalSearchParams {
        model: &model,
        operator: &mut operator,
        metaheuristic: &mut gls,
        monitor,
        berths: initial.berths(),
        start_times: initial.start_times(),
        objective_value: init_obj,
    };

    let outcome = engine.run(params, evaluator, |sol| {
        println!("  New best: {}", sol.objective_value());
    });

    // Report results.
    let (solution, reason, stats) = outcome.into_inner();
    println!("\n=== Search Complete ===");
    println!("Termination reason: {:?}", reason);
    println!("Final objective:    {}", solution.objective_value());
    println!(
        "Improvement:        {:.2}%",
        (1.0 - solution.objective_value() as f64 / init_obj as f64) * 100.0
    );
    println!("Iterations:         {}", stats.iterations);
    println!("Cycles:             {}", stats.cycles);
    println!("Total solutions:    {}", stats.total_solutions);
    println!("Accepted solutions: {}", stats.accepted_solutions);
    println!("Infeasible moves:   {}", stats.infeasible_moves);
    println!("Time:               {:?}", stats.time_total);
}
