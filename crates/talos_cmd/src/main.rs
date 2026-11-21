// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::loading::{InstanceLoader, ProblemLoaderError};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::{Duration, Instant};
use talos_ls::{
    AtNewBest, DynamicLambda, Execution, GuidedLocalSearch, Lambda, LocalSearchMonitor,
    LocalSearchParameters, Metaheuristic, PenalizationTrigger, Portfolio, PortfolioMember,
    SearchLimits, SearchProgress, heuristic_lambda,
};
use talos_model::{Model, Solution};

/// The monitor that times a search, from its start to its end, without the loading of the schedule before it.
#[derive(Debug, Default)]
struct SearchClock {
    /// When the search started, once it has.
    started: Option<Instant>,
    /// The time from the start of the search to its end, zero until it has ended.
    elapsed: Duration,
}

impl LocalSearchMonitor<i64> for SearchClock {
    fn on_start(&mut self, _model: &Model<i64>, _progress: SearchProgress<'_, i64>) {
        self.started = Some(Instant::now());
    }

    fn on_end(&mut self, _progress: SearchProgress<'_, i64>) {
        self.elapsed = self
            .started
            .map_or(Duration::ZERO, |started| started.elapsed());
    }
}

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
            first_run_time_limit: std::time::Duration::from_millis(67_600),
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

/// The instance a run of the binary takes when none is named.
const DEFAULT_INSTANCE: &str = "f200x15-01.txt";

/// The share of the objective that sets the dynamic lambda of a run when no shares are named.
const DEFAULT_LAMBDA_ALPHA: &str = "0.2";

/// Which of the two runs of an instance the binary makes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Runs {
    /// The first run, then the second, with the table and the result files.
    Both,
    /// The run limited to the time of the reference metaheuristic alone.
    First,
    /// The run of up to six minutes that stops after two without a new best alone.
    Second,
}

/// The label of one run of an instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum RunLabel {
    First,
    Second,
}

/// A share of the objective, held as the decimal it was written as: `digits` times ten to the power of minus
/// `decimals`, never zero. Its half and its three halves are then decimals too, and each becomes the `f64` nearest
/// to it, as the share itself does, so that the bounds of the share 0.2 are the `f64`s of 0.1 and 0.3.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Alpha {
    digits: u64,
    decimals: u32,
}

impl Alpha {
    /// The share written as `text`, digits with at most one decimal point, or why it is refused.
    fn parse(text: &str) -> Result<Self, String> {
        let (whole, fraction) = text.split_once('.').unwrap_or((text, ""));
        let all_digits = |part: &str| part.bytes().all(|byte| byte.is_ascii_digit());
        if whole.is_empty() && fraction.is_empty() || !all_digits(whole) || !all_digits(fraction) {
            return Err(format!("lambda alpha `{text}` is not a decimal number"));
        }
        let digits: u64 = format!("{whole}{fraction}")
            .parse()
            .map_err(|_| format!("lambda alpha `{text}` has too many digits"))?;
        let decimals = u32::try_from(fraction.len())
            .map_err(|_| format!("lambda alpha `{text}` has too many decimals"))?;
        if digits == 0 {
            return Err(format!("lambda alpha `{text}` is zero"));
        }
        // The three halves multiply the digits by 15, which must not overflow.
        if digits.checked_mul(15).is_none() {
            return Err(format!("lambda alpha `{text}` has too many digits"));
        }
        Ok(Self { digits, decimals })
    }

    /// The `f64` nearest to the share times `tenths` tenths.
    fn tenths(self, tenths: u64) -> f64 {
        // `parse` refused every share whose digits times 15 overflow, and `tenths` is at most 15.
        let scaled = self.digits * tenths;
        format!("{scaled}e-{}", self.decimals + 1)
            .parse()
            .expect("a decimal in exponent form should parse as an `f64`")
    }

    /// The share.
    fn value(self) -> f64 {
        self.tenths(10)
    }
}

/// What a call of the binary asks for.
#[derive(Debug)]
struct Arguments {
    /// The file name of the instance, one of the benchmark configurations.
    instance: String,
    /// The runs to make.
    runs: Runs,
    /// The share of each member of a portfolio, one member per share; absent where none were named, and the run
    /// is then the single run of the share 0.2.
    lambda_alphas: Option<Vec<Alpha>>,
    /// The threads the members run on, at least one.
    threads: usize,
}

impl Arguments {
    /// The arguments of `args`, the command line without the name of the binary, or why they are refused.
    fn parse(mut args: impl Iterator<Item = String>) -> Result<Self, String> {
        let mut arguments = Self {
            instance: DEFAULT_INSTANCE.to_string(),
            runs: Runs::Both,
            lambda_alphas: None,
            threads: 1,
        };
        while let Some(flag) = args.next() {
            let mut value = || args.next().ok_or_else(|| format!("`{flag}` needs a value"));
            match flag.as_str() {
                "--instance" => arguments.instance = value()?,
                "--run" => {
                    arguments.runs = match value()?.as_str() {
                        "first" => Runs::First,
                        "second" => Runs::Second,
                        "both" => Runs::Both,
                        other => {
                            return Err(format!("`--run {other}` is not first, second or both"));
                        }
                    }
                }
                "--lambda-alphas" => {
                    let alphas = value()?
                        .split(',')
                        .map(Alpha::parse)
                        .collect::<Result<Vec<_>, _>>()?;
                    arguments.lambda_alphas = Some(alphas);
                }
                "--threads" => {
                    let text = value()?;
                    arguments.threads = match text.parse::<usize>() {
                        Ok(threads) if threads > 0 => threads,
                        _ => return Err(format!("`--threads {text}` is not a positive count")),
                    };
                }
                other => return Err(format!("unknown argument `{other}`")),
            }
        }
        Ok(arguments)
    }

    /// The share of each member, in order.
    fn alphas(&self) -> Vec<Alpha> {
        self.lambda_alphas.clone().unwrap_or_else(|| {
            vec![Alpha::parse(DEFAULT_LAMBDA_ALPHA).expect("the default share should parse")]
        })
    }
}

/// What one portfolio of several shares answered, beside the record of its best member.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PortfolioRecord {
    /// The share of each member, in order.
    lambda_alphas: Vec<f64>,
    /// The threads the members ran on.
    threads: usize,
    /// The member of the lowest objective, the first among those that tie.
    best_member: usize,
    /// The objective each member reached, in order.
    member_objectives: Vec<i64>,
    /// The time of each member's search loop in seconds, in order.
    member_search_seconds: Vec<f64>,
    /// The candidates each member evaluated, in order.
    member_iterations: Vec<u64>,
}

/// What one run of an instance answered: of its best member, where it is a portfolio.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct RunRecord {
    filename: String,
    run: RunLabel,
    greedy_objective: i64,
    objective: i64,
    /// The time of the search loop, without loading, construction and the initial decode.
    time: Duration,
    search_seconds: f64,
    improvement: f64,
    /// The candidates evaluated, feasible or not.
    iterations: u64,
    cycles: u64,
    accepted_solutions: u64,
    infeasible_moves: u64,
    /// Whether the model found every member's schedule feasible; a run that finds one infeasible fails instead.
    feasible: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    portfolio: Option<PortfolioRecord>,
}

/// The member of the share `alpha`: the dynamic lambda starts at λ(α) and stays within [λ(α/2), λ(3α/2)], where
/// λ(a) is the lambda at which one penalty of every feature adds the share `a` of `greedy_objective`.
fn member(
    alpha: Alpha,
    greedy_objective: i64,
    model: &Model<i64>,
    limits: SearchLimits,
) -> PortfolioMember<SearchClock> {
    let features = model.num_vessels() * model.num_berths();
    let lambda = |tenths| heuristic_lambda(greedy_objective as f64, features, alpha.tenths(tenths));
    let guided_local_search = Metaheuristic::GuidedLocalSearch(GuidedLocalSearch {
        lambda: Lambda::Dynamic(DynamicLambda::new(
            lambda(10),
            0.1,
            AtNewBest::Decrease(0.1),
            lambda(5),
            lambda(15),
        )),
        trigger: PenalizationTrigger::OnExhaustion,
        ..GuidedLocalSearch::default()
    });
    PortfolioMember {
        parameters: LocalSearchParameters {
            metaheuristic: guided_local_search,
            limits,
            ..LocalSearchParameters::default()
        },
        monitor: SearchClock::default(),
    }
}

/// One run of every share of `arguments` from `initial` under `limits`, as a record of its best member, or why it
/// was refused or failed: a portfolio that does not give every member a thread of its own, since every run here
/// is time-limited, or a member whose schedule the model finds infeasible.
#[expect(clippy::too_many_arguments, reason = "the binary is restyled later")]
fn run_once(
    portfolio: &mut Portfolio<i64>,
    model: &Model<i64>,
    initial: &Solution<i64>,
    arguments: &Arguments,
    execution: Execution<'_>,
    filename: &str,
    run: RunLabel,
    limits: SearchLimits,
) -> Result<RunRecord, String> {
    let alphas = arguments.alphas();
    if !execution.gives_every_member_a_thread(alphas.len()) {
        return Err(format!(
            "a time-limited portfolio of {} members needs at least as many threads, but runs on {}",
            alphas.len(),
            arguments.threads
        ));
    }
    let greedy_objective = initial.objective_value();
    let mut members: Vec<_> = alphas
        .iter()
        .map(|&alpha| member(alpha, greedy_objective, model, limits))
        .collect();
    let outcome = portfolio
        .run(model, initial.as_view(), &mut members, execution)
        .map_err(|error| format!("the portfolio refused to run: {error}"))?;

    for (index, member_outcome) in outcome.outcomes().iter().enumerate() {
        model
            .check_feasibility(member_outcome.solution.as_view())
            .map_err(|error| {
                format!("member {index} of {filename} answered an infeasible schedule: {error}")
            })?;
    }

    let best_member = outcome.best_member();
    let best = outcome.best();
    let objective = best.solution.objective_value();
    let time = members[best_member].monitor.elapsed;
    let statistics = best.statistics;
    let portfolio_record = arguments
        .lambda_alphas
        .as_ref()
        .map(|alphas| PortfolioRecord {
            lambda_alphas: alphas.iter().map(|alpha| alpha.value()).collect(),
            threads: arguments.threads,
            best_member,
            member_objectives: outcome
                .outcomes()
                .iter()
                .map(|member_outcome| member_outcome.solution.objective_value())
                .collect(),
            member_search_seconds: members
                .iter()
                .map(|member| member.monitor.elapsed.as_secs_f64())
                .collect(),
            member_iterations: outcome
                .outcomes()
                .iter()
                .map(|member_outcome| member_outcome.statistics.iterations)
                .collect(),
        });
    Ok(RunRecord {
        filename: filename.to_string(),
        run,
        greedy_objective,
        objective,
        time,
        search_seconds: time.as_secs_f64(),
        improvement: (1.0 - objective as f64 / greedy_objective as f64) * 100.0,
        iterations: statistics.iterations,
        cycles: statistics.cycles,
        accepted_solutions: statistics.accepted_solutions,
        infeasible_moves: statistics.infeasible_moves,
        feasible: true,
        portfolio: portfolio_record,
    })
}

fn run_benchmark(
    config: &InstanceBenchmarkConfig,
    data_dir: &std::path::Path,
    arguments: &Arguments,
    execution: Execution<'_>,
) -> Result<Vec<RunRecord>, String> {
    let instance_path = data_dir.join(&config.filename);

    let model = load_instance(&instance_path)
        .map_err(|e| format!("failed to load {}: {e}", config.filename))?;

    let initial = edf::generate_greedy_edf_schedule(&model)
        .ok_or_else(|| "the EDF heuristic found no feasible schedule".to_string())?;

    let first_limits = SearchLimits {
        time: Some(config.first_run_time_limit),
        ..SearchLimits::default()
    };
    let second_limits = SearchLimits {
        time: Some(config.second_run_time_limit),
        non_improving_time: Some(config.non_improving_time_limit),
        ..SearchLimits::default()
    };
    let runs: &[(RunLabel, SearchLimits)] = match arguments.runs {
        Runs::Both => &[
            (RunLabel::First, first_limits),
            (RunLabel::Second, second_limits),
        ],
        Runs::First => &[(RunLabel::First, first_limits)],
        Runs::Second => &[(RunLabel::Second, second_limits)],
    };

    let mut portfolio = Portfolio::<i64>::new();
    let mut records = Vec::with_capacity(runs.len());
    for &(run, limits) in runs {
        let record = run_once(
            &mut portfolio,
            &model,
            &initial,
            arguments,
            execution,
            &config.filename,
            run,
            limits,
        )?;
        println!(
            "{}",
            serde_json::to_string(&record).expect("a run record should serialize")
        );
        records.push(record);
    }
    Ok(records)
}

/// The result of both runs of an instance, in the form of the result files.
fn instance_result(first: &RunRecord, second: &RunRecord) -> InstanceBenchmarkResult {
    InstanceBenchmarkResult {
        filename: first.filename.clone(),
        greedy_objective: first.greedy_objective,
        first_run_objective: first.objective,
        second_run_objective: second.objective,
        first_run_time: first.time,
        second_run_time: second.time,
        first_run_improvement: first.improvement,
        second_run_improvement: second.improvement,
    }
}

fn main() -> ExitCode {
    match run(std::env::args().skip(1)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::FAILURE
        }
    }
}

/// The runs `args` ask for, or why they were refused or failed.
fn run(args: impl Iterator<Item = String>) -> Result<(), String> {
    let arguments = Arguments::parse(args)?;
    let benchmarks = setup_benchmarks();
    let config = benchmarks
        .iter()
        .find(|config| config.filename == arguments.instance)
        .ok_or_else(|| format!("`{}` is not a benchmark instance", arguments.instance))?;
    let pool = match arguments.threads {
        1 => None,
        threads => Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .map_err(|error| format!("could not build a pool of {threads} threads: {error}"))?,
        ),
    };
    let execution = pool.as_ref().map_or(Execution::Serial, Execution::Pool);

    let data_dir =
        find_instances_dir().ok_or_else(|| "could not find the `data` directory".to_string())?;
    if arguments.runs != Runs::Both {
        run_benchmark(config, &data_dir, &arguments, execution)?;
        return Ok(());
    }

    let results_dir = data_dir.parent().unwrap().join("results");
    fs::create_dir_all(&results_dir).expect("Failed to create results directory");
    println!(
        "  {:<18} {:>16} {:>16} {:>11} {:>14} {:>16} {:>11} {:>14}",
        "Instance", "EDF", "Run1 Obj", "Impr%", "Time", "Run2 Obj", "Impr%", "Time"
    );
    println!("  {}", "-".repeat(124));

    let records = run_benchmark(config, &data_dir, &arguments, execution)?;
    let result = instance_result(&records[0], &records[1]);
    println!(
        "  {:<18} {:>16} {:>16} {:>10.2}% {:>14.2?} {:>16} {:>10.2}% {:>14.2?}",
        result.filename,
        result.greedy_objective,
        result.first_run_objective,
        result.first_run_improvement,
        result.first_run_time,
        result.second_run_objective,
        result.second_run_improvement,
        result.second_run_time,
    );

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

    // Write combined JSON
    let combined_path = results_dir.join("all_results.json");
    let combined_json =
        serde_json::to_string_pretty(&[result]).expect("Failed to serialize all results");
    fs::write(&combined_path, &combined_json).unwrap_or_else(|e| {
        panic!("Failed to write {}: {e}", combined_path.display());
    });

    println!("\nResults written to {}", results_dir.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // The bounds of the share 0.2 are the `f64`s of 0.1 and 0.3, so its member is the run the binary made before
    // it took shares; 0.2 * 1.5 would round to the `f64` above 0.3.
    #[test]
    fn test_the_share_of_two_tenths_has_the_bounds_of_one_and_three_tenths() {
        let alpha = Alpha::parse("0.2").expect("0.2 should parse");
        assert_eq!(alpha.value().to_bits(), 0.2_f64.to_bits());
        assert_eq!(alpha.tenths(5).to_bits(), 0.1_f64.to_bits());
        assert_eq!(alpha.tenths(15).to_bits(), 0.3_f64.to_bits());
        assert_ne!((0.2_f64 * 1.5).to_bits(), 0.3_f64.to_bits());
    }

    #[test]
    fn test_a_share_is_a_nonzero_decimal() {
        assert_eq!(Alpha::parse("0.15").map(Alpha::value), Ok(0.15));
        assert_eq!(Alpha::parse(".45").map(|alpha| alpha.tenths(15)), Ok(0.675));
        assert_eq!(Alpha::parse("1").map(|alpha| alpha.tenths(5)), Ok(0.5));
        for refused in ["", ".", "0", "0.00", "-0.1", "0.1.2", "1e-1", "abc"] {
            assert!(
                Alpha::parse(refused).is_err(),
                "`{refused}` should be refused"
            );
        }
    }

    #[test]
    fn test_the_arguments_name_the_instance_the_runs_the_shares_and_the_threads() {
        let args = [
            "--instance",
            "f250x20-05.txt",
            "--run",
            "second",
            "--lambda-alphas",
            "0.1,0.2",
            "--threads",
            "8",
        ];
        let arguments =
            Arguments::parse(args.iter().map(|arg| (*arg).to_string())).expect("should parse");
        assert_eq!(arguments.instance, "f250x20-05.txt");
        assert_eq!(arguments.runs, Runs::Second);
        assert_eq!(
            arguments
                .alphas()
                .iter()
                .map(|alpha| alpha.value())
                .collect::<Vec<_>>(),
            [0.1, 0.2]
        );
        assert_eq!(arguments.threads, 8);

        let defaults = Arguments::parse(std::iter::empty()).expect("no arguments should parse");
        assert_eq!(defaults.instance, DEFAULT_INSTANCE);
        assert_eq!(defaults.runs, Runs::Both);
        assert!(defaults.lambda_alphas.is_none());
        assert_eq!(
            defaults
                .alphas()
                .iter()
                .map(|alpha| alpha.value())
                .collect::<Vec<_>>(),
            [0.2]
        );
        assert_eq!(defaults.threads, 1);

        for refused in [
            &["--threads", "0"][..],
            &["--run", "third"],
            &["--threads"],
            &["--seed", "1"],
        ] {
            assert!(
                Arguments::parse(refused.iter().map(|arg| (*arg).to_string())).is_err(),
                "{refused:?}"
            );
        }
    }
}
