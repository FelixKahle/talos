// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The engines of a portfolio, and the runs of its members, one after another or on a thread pool.

use super::{
    cancellation::{CancelOnPanic, CancellableMonitor, Cancellation},
    member::PortfolioMember,
    outcome::{PortfolioError, PortfolioOutcome},
};
use crate::{
    engine::Engine, monitor::LocalSearchMonitor, outcome::LocalSearchOutcome,
    parameters::check_neighborhoods,
};
use rayon::{
    ThreadPool,
    iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator},
};
use talos_core::SolverNumeric;
use talos_model::{Model, SolutionView};

/// Where the members of a portfolio run.
///
/// The members run alike wherever they run: each on an engine of its own, its outcome placed at its index, so a
/// portfolio answers the same on the calling thread and on a pool of any size, unless a deadline or a monitor that
/// reads the clock ends a run.
///
/// A deadline of a member's limits runs from the start of that member's own search, not from the call: where the
/// members outnumber the threads, a member starts only once one before it has ended, so a portfolio whose members
/// all have the same time limit takes about that limit times the members per thread, rounded up, on one thread
/// and on a pool alike. Members that run side by side share the cores and the memory bandwidth of the machine, so
/// each does less work in its time than a lone run would. A time-limited portfolio is run where
/// [`gives_every_member_a_thread`](Self::gives_every_member_a_thread) holds, on a pool of at least as many threads
/// as members.
#[derive(Debug, Clone, Copy)]
pub enum Execution<'a> {
    /// One member after another, in their order, on the calling thread.
    Serial,
    /// The members side by side on the threads of the pool the caller built; the calling thread waits for them.
    Pool(&'a ThreadPool),
}

impl Execution<'_> {
    /// Whether each of `members` members runs on a thread of its own, all of them from the start: on the calling
    /// thread for at most one member, and on a pool of at least as many threads as members.
    pub fn gives_every_member_a_thread(&self, members: usize) -> bool {
        match self {
            Self::Serial => members <= 1,
            Self::Pool(pool) => members <= pool.current_num_threads(),
        }
    }
}

/// The independent portfolio: several runs of the engine from one model and one initial schedule, each with
/// parameters and a monitor of its own, whose best outcome is the answer.
///
/// Each member runs on an engine of its own and shares nothing with the others while it runs: no schedule, no
/// penalty and no bound passes between them. A member's outcome is the one a lone [`Engine::run`] of its parameters
/// and its monitor answers, unless a deadline ends its run; the best member is the one with the lowest objective,
/// the lowest index among those that tie. The members differ by their parameters, such as the penalty weight of
/// their guided local search.
///
/// A portfolio keeps its engines from one run to the next, as an [`Engine`] keeps its memory, and adds an engine
/// only where a run has more members than any before: a caller that keeps the portfolio allocates no engine on a
/// later run of as many members. The time limit of a member runs from the start of its own search; how long a
/// time-limited portfolio takes is said at [`Execution`]. An outcome holds no time: a member's monitor, told of the
/// start and the end of its search, reads the clock where the time of a member is wanted.
///
/// ```
/// # use talos_core::ClosedOpenInterval;
/// # use talos_ls::{
/// #     Execution, GuidedLocalSearch, Lambda, LocalSearchParameters, Metaheuristic, NoMonitor, Portfolio,
/// #     PortfolioMember, SearchLimits, StaticLambda,
/// # };
/// # use talos_model::{BerthIndex, Model, SolutionView};
/// // Three vessels that take 10, 10 and 1 on either of two berths: the first two on the first berth, from 0 and
/// // from 10, and the third on the second berth.
/// let model = Model::new(
///     vec![0, 0, 0],
///     vec![100, 100, 100],
///     vec![1, 1, 1],
///     vec![Some(10), Some(10), Some(10), Some(10), Some(1), Some(1)],
///     vec![vec![ClosedOpenInterval::new(0, 200)]; 2],
/// );
/// let berths = [BerthIndex::new(0), BerthIndex::new(0), BerthIndex::new(1)];
/// let initial = SolutionView::new(&berths, &[0, 10, 0], 31);
///
/// // One member per penalty weight, each under a budget of a thousand candidates.
/// let mut members = [0.5, 1.0, 2.0].map(|weight| PortfolioMember {
///     parameters: LocalSearchParameters {
///         metaheuristic: Metaheuristic::GuidedLocalSearch(GuidedLocalSearch {
///             lambda: Lambda::Static(StaticLambda::new(weight)),
///             ..GuidedLocalSearch::default()
///         }),
///         limits: SearchLimits { iterations: Some(1000), ..SearchLimits::default() },
///         ..LocalSearchParameters::default()
///     },
///     monitor: NoMonitor,
/// });
///
/// let mut portfolio = Portfolio::new();
/// let outcome = portfolio.run(&model, initial, &mut members, Execution::Serial)?;
/// assert_eq!(outcome.outcomes().len(), 3);
/// // One of the long vessels follows the short one: 10 + 1 + 11.
/// assert_eq!(outcome.best().solution.objective_value(), 22);
/// # Ok::<(), talos_ls::PortfolioError>(())
/// ```
#[derive(Debug, Clone)]
pub struct Portfolio<T> {
    /// One engine per member of the largest portfolio run so far; a run of fewer members uses the first ones.
    engines: Vec<Engine<T>>,
}

impl<T> Default for Portfolio<T>
where
    T: SolverNumeric,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Portfolio<T>
where
    T: SolverNumeric,
{
    /// A portfolio that has not yet run; its first run makes an engine per member.
    pub fn new() -> Self {
        Self {
            engines: Vec::new(),
        }
    }

    /// Runs every member of `members` from `initial`, a schedule of `model`, where `execution` says, and answers
    /// the outcome of each and which is best.
    ///
    /// Each member's run is the run [`Engine::run`] makes of its parameters and its monitor, on an engine of the
    /// portfolio; its monitor is told of the events of that run alone, on the thread the run is on. The outcome
    /// holds the members' outcomes in their order, and each is the one a lone run of the member answers, unless
    /// a deadline ends it.
    ///
    /// What [`Engine::run`] refuses, the portfolio refuses at its door, before any member runs, so that no member
    /// is refused once another has started. The door checks the neighbourhoods of each member, and the initial
    /// schedule once, in one pass over the vessels and one decode of its order; each member's run then checks and
    /// decodes the schedule again, as a lone run does.
    ///
    /// The portfolio keeps an engine per member, each of memory linear in the vessels and berths, and a run adds
    /// the engines it lacks. Each member's run allocates what a lone run allocates, and the memory of a guided
    /// local search is quadratic in the vessels: at most one member's at a time on the calling thread, and at most
    /// one per member that runs at once on a pool, as many as the fewer of its members and its threads.
    ///
    /// The monitors are `Send` on every execution, the serial one included, so that one signature serves both. On
    /// a pool, the members' monitors are called on the threads of the pool while the calling thread waits: a
    /// monitor that needs a lock the caller holds, such as that of a host language, has the caller release it
    /// around the run.
    ///
    /// # Errors
    ///
    /// [`PortfolioError::NoMember`] if `members` is empty; else [`PortfolioError::NoNeighborhood`] with the member
    /// of lowest index whose parameters name no neighbourhood; else [`PortfolioError::InitialSolution`] if
    /// [`Engine::run`] refuses `initial`, with its reason.
    ///
    /// # Panics
    ///
    /// Panics with the panic of a member's run or monitor. On the calling thread, the members after it never run.
    /// On a pool, every other member that runs stops at its next scan or checkpoint, and one that has not yet
    /// started does not run; the panic reaches the caller once every member has stopped, and of several members
    /// that panic, one panic does.
    pub fn run<M>(
        &mut self,
        model: &Model<T>,
        initial: SolutionView<'_, T>,
        members: &mut [PortfolioMember<M>],
        execution: Execution<'_>,
    ) -> Result<PortfolioOutcome<T>, PortfolioError>
    where
        M: LocalSearchMonitor<T> + Send,
    {
        check_members(members)?;
        self.provide_engines(members.len());
        let engines = &mut self.engines[..members.len()];
        // The order decodes alike under every objective, so the first member's speaks for all.
        engines[0].load_initial(model, initial, members[0].parameters.objective)?;

        let cancellation = Cancellation::default();
        let runs: Option<Vec<_>> = match execution {
            Execution::Serial => engines
                .iter_mut()
                .zip(members.iter_mut())
                .map(|(engine, member)| run_member(engine, model, initial, member, &cancellation))
                .collect(),
            // One member per job, so that an idle thread takes any member that has not started.
            Execution::Pool(pool) => pool.install(|| {
                engines
                    .par_iter_mut()
                    .zip(members.par_iter_mut())
                    .with_max_len(1)
                    .map(|(engine, member)| {
                        run_member(engine, model, initial, member, &cancellation)
                    })
                    .collect()
            }),
        };
        let outcomes = runs.expect(
            "a member should be skipped only after a sibling panicked, whose panic ends the call",
        );
        Ok(PortfolioOutcome::new(outcomes))
    }

    /// Makes engines until there is one for each of `members` members.
    fn provide_engines(&mut self, members: usize) {
        if self.engines.len() < members {
            self.engines.resize_with(members, Engine::new);
        }
    }
}

/// Refuses a portfolio without members, or one with a member whose parameters name no neighbourhood, the member
/// of lowest index; it costs one look per member.
///
/// # Errors
///
/// [`PortfolioError::NoMember`] if `members` is empty; [`PortfolioError::NoNeighborhood`] with the first member
/// whose parameters name no neighbourhood.
fn check_members<M>(members: &[PortfolioMember<M>]) -> Result<(), PortfolioError> {
    if members.is_empty() {
        return Err(PortfolioError::NoMember);
    }
    members
        .iter()
        .position(|member| check_neighborhoods(&member.parameters).is_err())
        .map_or(Ok(()), |member| {
            Err(PortfolioError::NoNeighborhood { member })
        })
}

/// The outcome of the run of `member` on `engine`, which ends early once `cancellation` is raised and raises it if
/// the run panics; or `None`, with no run, if `cancellation` is raised before it starts.
///
/// The portfolio has accepted `member` and `initial` at its door.
///
/// # Panics
///
/// Panics with the panic of the run or of the member's monitor, and if the run is refused, which the door of the
/// portfolio rules out.
fn run_member<T, M>(
    engine: &mut Engine<T>,
    model: &Model<T>,
    initial: SolutionView<'_, T>,
    member: &mut PortfolioMember<M>,
    cancellation: &Cancellation,
) -> Option<LocalSearchOutcome<T>>
where
    T: SolverNumeric,
    M: LocalSearchMonitor<T>,
{
    if cancellation.is_raised() {
        return None;
    }
    let _guard = CancelOnPanic::new(cancellation);
    let mut monitor = CancellableMonitor::new(&mut member.monitor, cancellation);
    let outcome = engine
        .run(model, initial, &member.parameters, &mut monitor)
        .expect("the door of the portfolio should have refused what the run of a member refuses");
    Some(outcome)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        limits::SearchLimits,
        metaheuristic::{GreedyDescent, GuidedLocalSearch, Lambda, Metaheuristic, StaticLambda},
        monitor::{NoMonitor, SearchProgress},
        objective::Objective,
        parameters::{InitialSolutionError, LocalSearchParameters},
    };
    use rayon::ThreadPoolBuilder;
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use talos_core::ClosedOpenInterval;
    use talos_model::{BerthIndex, VesselIndex};

    /// A model of `num_vessels` vessels with staggered arrivals and weights on three berths, each vessel barred from
    /// at most one berth, with an initial schedule of it: the vessels by index, each on a berth it may use.
    fn staggered_instance(num_vessels: usize) -> (Model<i64>, Vec<BerthIndex>, Vec<i64>) {
        let count = i64::try_from(num_vessels).expect("a test instance is small");
        let arrivals: Vec<i64> = (0..count).map(|v| (v * 7) % 23).collect();
        let weights: Vec<i64> = (0..count).map(|v| 1 + (v * 5) % 4).collect();
        let processing: Vec<Option<i64>> = (0..3 * count)
            .map(|i| (i % 11 != 4).then_some(3 + (i * 13) % 9))
            .collect();
        let model = Model::new(
            arrivals,
            vec![10_000; num_vessels],
            weights,
            processing,
            vec![vec![ClosedOpenInterval::new(0, 50_000)]; 3],
        );
        let berths = (0..num_vessels)
            .map(|v| {
                (0..3)
                    .map(|b| BerthIndex::new((v + b) % 3))
                    .find(|&b| {
                        model
                            .vessel_processing_time(VesselIndex::new(v), b)
                            .is_some()
                    })
                    .expect("every vessel should have a berth it may use")
            })
            .collect();
        (model, berths, (0..count).collect())
    }

    /// Parameters of a guided local search under a budget of `iterations` candidates.
    fn guided(iterations: u64) -> LocalSearchParameters {
        LocalSearchParameters {
            limits: SearchLimits {
                iterations: Some(iterations),
                ..SearchLimits::default()
            },
            ..LocalSearchParameters::default()
        }
    }

    /// Static penalty weights spread over four orders of magnitude, one per member.
    fn lambdas(members: usize) -> impl Iterator<Item = Lambda> {
        [0.01, 0.1, 1.0, 10.0, 100.0]
            .into_iter()
            .cycle()
            .zip(1_u32..)
            .map(|(weight, round)| Lambda::Static(StaticLambda::new(weight * f64::from(round))))
            .take(members)
    }

    /// One member per weight of `lambdas`, in their order, each with the parameters of `base` but the penalty
    /// weight of its guided local search, and observed by `monitor`.
    fn varying_lambda<M: Clone>(
        base: &LocalSearchParameters,
        lambdas: impl Iterator<Item = Lambda>,
        monitor: &M,
    ) -> Vec<PortfolioMember<M>> {
        let Metaheuristic::GuidedLocalSearch(guided) = base.metaheuristic else {
            panic!("the base of a test portfolio should run a guided local search");
        };
        lambdas
            .map(|lambda| PortfolioMember {
                parameters: LocalSearchParameters {
                    metaheuristic: Metaheuristic::GuidedLocalSearch(GuidedLocalSearch {
                        lambda,
                        ..guided
                    }),
                    ..base.clone()
                },
                monitor: monitor.clone(),
            })
            .collect()
    }

    fn pool(threads: usize) -> ThreadPool {
        ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("a test pool should be built")
    }

    /// A monitor that records the objective of every accepted schedule of its run.
    #[derive(Debug, Clone, Default, PartialEq)]
    struct Trajectory(Vec<i64>);

    impl LocalSearchMonitor<i64> for Trajectory {
        fn on_candidate_accepted(&mut self, progress: SearchProgress<'_, i64>) {
            self.0.push(progress.accepted.objective_value());
        }
        fn on_buffered_solution_accepted(&mut self, progress: SearchProgress<'_, i64>) {
            self.0.push(progress.accepted.objective_value());
        }
    }

    #[test]
    fn test_each_member_answers_what_a_lone_run_of_it_answers() {
        let (model, berths, starts) = staggered_instance(16);
        let initial = SolutionView::new(&berths, &starts, 0);
        let thread_pool = pool(2);
        for execution in [Execution::Serial, Execution::Pool(&thread_pool)] {
            let mut members = varying_lambda(&guided(4000), lambdas(5), &Trajectory::default());
            let outcome = Portfolio::new()
                .run(&model, initial, &mut members, execution)
                .expect("every member should run");
            for (member, answered) in members.iter().zip(outcome.outcomes()) {
                let mut lone = Trajectory::default();
                let expected = Engine::new()
                    .run(&model, initial, &member.parameters, &mut lone)
                    .expect("the member should run alone");
                assert_eq!(answered, &expected);
                assert_eq!(member.monitor, lone);
                assert!(!lone.0.is_empty());
            }
            // The members differ by their lambda alone, and that sets them on trajectories of their own.
            let distinct = members
                .iter()
                .filter(|member| member.monitor != members[0].monitor)
                .count();
            assert!(distinct >= 2, "the lambdas should set the members apart");
        }
    }

    #[test]
    fn test_the_portfolio_answers_alike_serially_and_on_pools_of_one_two_and_eight_threads() {
        let (model, berths, starts) = staggered_instance(20);
        let initial = SolutionView::new(&berths, &starts, 0);
        let by_cycles = LocalSearchParameters {
            limits: SearchLimits {
                cycles: Some(30),
                iterations: Some(1_000_000),
                ..SearchLimits::default()
            },
            ..LocalSearchParameters::default()
        };
        let mut members = varying_lambda(&guided(5000), lambdas(5), &NoMonitor);
        members.extend(varying_lambda(&by_cycles, lambdas(2), &NoMonitor));
        members.push(PortfolioMember {
            parameters: LocalSearchParameters {
                metaheuristic: Metaheuristic::GreedyDescent(GreedyDescent::default()),
                ..guided(5000)
            },
            monitor: NoMonitor,
        });
        let serial = Portfolio::new()
            .run(&model, initial, &mut members, Execution::Serial)
            .expect("every member should run");
        assert_eq!(serial.outcomes().len(), 8);
        for threads in [1, 2, 8] {
            let parallel = Portfolio::new()
                .run(
                    &model,
                    initial,
                    &mut members,
                    Execution::Pool(&pool(threads)),
                )
                .expect("every member should run");
            assert_eq!(parallel, serial, "a pool of {threads} threads");
            assert_eq!(parallel.best_member(), serial.best_member());
        }
    }

    /// A monitor that panics at the start of its run if told to, and otherwise records whether its run started and
    /// the candidates it made when it ended.
    #[derive(Debug, Clone, Default)]
    struct PanicProbe {
        panics: bool,
        started: bool,
        ended_after: Option<u64>,
    }

    impl LocalSearchMonitor<i64> for PanicProbe {
        fn on_start(&mut self, _: &Model<i64>, _: SearchProgress<'_, i64>) {
            assert!(!self.panics, "the probe panics");
            self.started = true;
        }
        fn on_end(&mut self, progress: SearchProgress<'_, i64>) {
            self.ended_after = Some(progress.statistics.iterations);
        }
    }

    // The siblings of the member that panics have a budget no test should spend; they stop early only if the flag
    // reaches them, and the budget bounds the test should it not.
    #[test]
    fn test_a_panic_in_a_member_stops_its_siblings_and_reaches_the_caller() {
        const SIBLING_BUDGET: u64 = 50_000_000;
        let (model, berths, starts) = staggered_instance(16);
        let initial = SolutionView::new(&berths, &starts, 0);
        let pools = [pool(1), pool(2), pool(8)];
        let executions = [Execution::Serial]
            .into_iter()
            .chain(pools.iter().map(Execution::Pool));
        let mut portfolio = Portfolio::new();
        for execution in executions {
            let mut members: Vec<_> = (0..6)
                .map(|member| PortfolioMember {
                    parameters: guided(SIBLING_BUDGET),
                    monitor: PanicProbe {
                        panics: member == 0,
                        ..PanicProbe::default()
                    },
                })
                .collect();
            let unwound = catch_unwind(AssertUnwindSafe(|| {
                portfolio.run(&model, initial, &mut members, execution)
            }));
            let payload = unwound.expect_err("the panic of member 0 should reach the caller");
            assert_eq!(
                payload.downcast_ref::<&str>(),
                Some(&"the probe panics"),
                "{execution:?}"
            );
            let siblings = members[1..].iter().map(|member| &member.monitor);
            match execution {
                Execution::Serial => {
                    for sibling in siblings {
                        assert!(!sibling.started, "a sibling after the panic started");
                        assert_eq!(sibling.ended_after, None, "a sibling after the panic ended");
                    }
                }
                Execution::Pool(_) => {
                    // A sibling that started before the flag was raised stops at its next scan or checkpoint; one
                    // that had not started is skipped.
                    for sibling in siblings {
                        match sibling.started {
                            true => assert!(
                                sibling
                                    .ended_after
                                    .is_some_and(|iterations| iterations < SIBLING_BUDGET),
                                "{execution:?}: a sibling that started ended at {:?}",
                                sibling.ended_after
                            ),
                            false => assert_eq!(sibling.ended_after, None, "{execution:?}"),
                        }
                    }
                }
            }
        }

        // The engines of a run that panicked are loaded afresh by the next run.
        let mut members = varying_lambda(&guided(3000), lambdas(6), &NoMonitor);
        let fresh = Portfolio::new()
            .run(&model, initial, &mut members, Execution::Serial)
            .expect("every member should run");
        let reused = portfolio
            .run(&model, initial, &mut members, Execution::Pool(&pools[1]))
            .expect("every member should run");
        assert_eq!(reused, fresh);
    }

    #[test]
    fn test_a_portfolio_without_members_is_refused() {
        let (model, berths, starts) = staggered_instance(4);
        let mut members: Vec<PortfolioMember<NoMonitor>> = Vec::new();
        assert_eq!(
            Portfolio::new().run(
                &model,
                SolutionView::new(&berths, &starts, 0),
                &mut members,
                Execution::Serial
            ),
            Err(PortfolioError::NoMember)
        );
    }

    #[test]
    fn test_a_member_without_a_neighborhood_is_named_before_any_member_runs() {
        let (model, berths, starts) = staggered_instance(4);
        let initial = SolutionView::new(&berths, &starts, 0);
        let without_neighborhood = LocalSearchParameters {
            neighborhoods: Vec::new(),
            ..guided(100)
        };
        let mut members: Vec<_> = [
            guided(100),
            guided(100),
            without_neighborhood.clone(),
            without_neighborhood,
        ]
        .into_iter()
        .map(|parameters| PortfolioMember {
            parameters,
            monitor: PanicProbe::default(),
        })
        .collect();
        assert_eq!(
            Portfolio::new().run(&model, initial, &mut members, Execution::Pool(&pool(2))),
            Err(PortfolioError::NoNeighborhood { member: 2 })
        );
        assert!(members.iter().all(|member| !member.monitor.started));
    }

    #[test]
    fn test_an_initial_schedule_of_another_model_is_refused_before_any_member_runs() {
        let (model, berths, starts) = staggered_instance(4);
        let shorter = SolutionView::new(&berths[1..], &starts[1..], 0);
        let mut members = varying_lambda(&guided(100), lambdas(2), &PanicProbe::default());
        assert_eq!(
            Portfolio::new().run(&model, shorter, &mut members, Execution::Pool(&pool(2))),
            Err(PortfolioError::InitialSolution(
                InitialSolutionError::VesselCountMismatch {
                    model_vessels: 4,
                    solution_vessels: 3
                }
            ))
        );
        assert!(members.iter().all(|member| !member.monitor.started));
    }

    // Two vessels of 8 on one berth end at 16, past the latest departure of 10, under either objective, so the
    // portfolio refuses the order once, whatever its members minimise.
    #[test]
    fn test_an_initial_order_without_a_feasible_schedule_is_refused_before_any_member_runs() {
        let model = Model::new(
            vec![0, 0],
            vec![10, 10],
            vec![1, 1],
            vec![Some(8), Some(8), Some(8), Some(8)],
            vec![vec![ClosedOpenInterval::new(0, 100)]; 2],
        );
        let berths = [BerthIndex::new(0), BerthIndex::new(0)];
        let initial = SolutionView::new(&berths, &[0, 8], 0);
        for objective in [
            Objective::WeightedTurnaroundTime,
            Objective::WeightedCompletionTime,
        ] {
            let base = LocalSearchParameters {
                objective,
                ..guided(100)
            };
            let mut members = varying_lambda(&base, lambdas(3), &PanicProbe::default());
            assert_eq!(
                Portfolio::new().run(&model, initial, &mut members, Execution::Pool(&pool(2))),
                Err(PortfolioError::InitialSolution(
                    InitialSolutionError::InfeasibleOrder
                ))
            );
            assert!(members.iter().all(|member| !member.monitor.started));
        }
    }

    #[test]
    fn test_execution_gives_every_member_a_thread_only_with_a_thread_per_member() {
        assert!(Execution::Serial.gives_every_member_a_thread(0));
        assert!(Execution::Serial.gives_every_member_a_thread(1));
        assert!(!Execution::Serial.gives_every_member_a_thread(2));
        let thread_pool = pool(3);
        let on_pool = Execution::Pool(&thread_pool);
        assert!(on_pool.gives_every_member_a_thread(1));
        assert!(on_pool.gives_every_member_a_thread(3));
        assert!(!on_pool.gives_every_member_a_thread(4));
    }

    // A kept portfolio keeps its engines and nothing else: after runs of more members on another model, it answers
    // what a fresh portfolio answers, and holds no engine beyond the most members it ran.
    #[test]
    fn test_a_kept_portfolio_answers_as_a_fresh_one_and_keeps_its_engines() {
        let (model, berths, starts) = staggered_instance(16);
        let initial = SolutionView::new(&berths, &starts, 0);
        let (other, other_berths, other_starts) = staggered_instance(30);
        let thread_pool = pool(2);
        let mut members = varying_lambda(&guided(3000), lambdas(3), &NoMonitor);
        let fresh = Portfolio::new()
            .run(&model, initial, &mut members, Execution::Serial)
            .expect("every member should run");

        let mut portfolio = Portfolio::new();
        let _ = portfolio
            .run(&model, initial, &mut members, Execution::Pool(&thread_pool))
            .expect("every member should run");
        let mut more = varying_lambda(&guided(3000), lambdas(5), &NoMonitor);
        let _ = portfolio
            .run(
                &other,
                SolutionView::new(&other_berths, &other_starts, 0),
                &mut more,
                Execution::Pool(&thread_pool),
            )
            .expect("every member should run");
        assert_eq!(portfolio.engines.len(), 5);
        let kept = portfolio
            .run(&model, initial, &mut members, Execution::Pool(&thread_pool))
            .expect("every member should run");
        assert_eq!(kept, fresh);
        assert_eq!(portfolio.engines.len(), 5);
    }

    #[test]
    fn test_a_portfolio_and_its_answer_can_move_to_another_thread() {
        fn assert_send<S: Send>() {}
        assert_send::<Portfolio<i64>>();
        assert_send::<PortfolioOutcome<i64>>();
        assert_send::<PortfolioMember<NoMonitor>>();
    }
}
