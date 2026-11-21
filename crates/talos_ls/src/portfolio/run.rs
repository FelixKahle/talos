// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The engines of a portfolio, and the runs of its members, one after another or on a thread pool.

use super::{
    cancellation::{CancelOnPanic, CancellableMonitor, Cancellation},
    member::PortfolioMember,
    outcome::{PortfolioError, PortfolioOutcome},
};
use crate::{
    engine::Engine,
    monitor::LocalSearchMonitor,
    outcome::LocalSearchOutcome,
    parameters::{LocalSearchError, check_run},
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
#[derive(Debug, Clone, Copy)]
pub enum Execution<'a> {
    /// One member after another, in their order, on the calling thread.
    Serial,
    /// The members side by side on the threads of the pool the caller built; the calling thread waits for them.
    Pool(&'a ThreadPool),
}

/// The independent portfolio: several runs of the engine from one model and one initial schedule, each with
/// parameters and a monitor of its own, whose best outcome is the answer.
///
/// Each member runs on an engine of its own and shares nothing with the others while it runs: no schedule, no
/// penalty and no bound passes between them. A member's outcome is the one a lone [`Engine::run`] of its parameters
/// and its monitor answers, unless a deadline ends its run; the best member is the one with the lowest objective,
/// the lowest index among those that tie. The members differ by their parameters; [`PortfolioMember::varying_lambda`]
/// builds members that differ by the penalty weight of their guided local search.
///
/// A portfolio keeps its engines from one run to the next, as an [`Engine`] keeps its memory, and adds an engine
/// only where a run has more members than any before: a caller that keeps the portfolio allocates no engine on a
/// later run of as many members. A deadline shared by every member is a deadline in the limits of each.
///
/// ```
/// # use talos_core::ClosedOpenInterval;
/// # use talos_ls::{
/// #     Execution, Lambda, LocalSearchParameters, Portfolio, PortfolioMember, SearchLimits, StaticLambda,
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
/// let base = LocalSearchParameters {
///     limits: SearchLimits { iterations: Some(1000), ..SearchLimits::default() },
///     ..LocalSearchParameters::default()
/// };
/// let lambdas = [0.5, 1.0, 2.0].map(|weight| Lambda::Static(StaticLambda::new(weight)));
/// let mut members = PortfolioMember::varying_lambda(&base, lambdas);
///
/// let mut portfolio = Portfolio::new();
/// let outcome = portfolio.run(&model, initial, &mut members, Execution::Serial)?;
/// assert_eq!(outcome.members().len(), 3);
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
    /// The checks at the door cost one pass over the vessels per member, and every member is checked before any
    /// runs. Each run allocates what a lone run allocates, so a portfolio of `n` members holds `n` times the
    /// memory of one run while they run side by side; a run with more members than any before also makes the
    /// engines it lacks.
    ///
    /// # Errors
    ///
    /// [`PortfolioError::NoMember`] if `members` is empty; [`PortfolioError::Member`] with the member of lowest
    /// index whose run [`Engine::run`] refuses, and its error.
    ///
    /// # Panics
    ///
    /// Panics with the panic of a member's run or monitor; where the members run on a pool, every other member
    /// stops at its next scan or checkpoint before the panic reaches the caller.
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
        check_members(model, initial, members)?;
        self.provide_engines(members.len());
        let cancellation = Cancellation::default();
        let engines = &mut self.engines[..members.len()];
        let runs = match execution {
            Execution::Serial => engines
                .iter_mut()
                .zip(members.iter_mut())
                .map(|(engine, member)| run_member(engine, model, initial, member, &cancellation))
                .collect(),
            Execution::Pool(pool) => pool.install(|| {
                engines
                    .par_iter_mut()
                    .zip(members.par_iter_mut())
                    .map(|(engine, member)| {
                        run_member(engine, model, initial, member, &cancellation)
                    })
                    .collect()
            }),
        };
        PortfolioOutcome::from_runs(runs)
    }

    /// Makes engines until there is one for each of `members` members.
    fn provide_engines(&mut self, members: usize) {
        if self.engines.len() < members {
            self.engines.resize_with(members, Engine::new);
        }
    }
}

/// Refuses a portfolio without members, or one with a member whose run [`Engine::run`] refuses at its door, the
/// member of lowest index; it costs one pass over the vessels per member.
///
/// # Errors
///
/// [`PortfolioError::NoMember`] if `members` is empty; [`PortfolioError::Member`] with the first member whose run
/// is refused, and its error.
fn check_members<T, M>(
    model: &Model<T>,
    initial: SolutionView<'_, T>,
    members: &[PortfolioMember<M>],
) -> Result<(), PortfolioError>
where
    T: SolverNumeric,
{
    if members.is_empty() {
        return Err(PortfolioError::NoMember);
    }
    members
        .iter()
        .enumerate()
        .try_for_each(|(member, PortfolioMember { parameters, .. })| {
            check_run(model, initial, parameters)
                .map_err(|error| PortfolioError::Member { member, error })
        })
}

/// The run of `member` on `engine`, which ends early once `cancellation` is raised and raises it if the run
/// panics.
///
/// # Errors
///
/// The error of [`Engine::run`] for the member's parameters.
fn run_member<T, M>(
    engine: &mut Engine<T>,
    model: &Model<T>,
    initial: SolutionView<'_, T>,
    member: &mut PortfolioMember<M>,
    cancellation: &Cancellation,
) -> Result<LocalSearchOutcome<T>, LocalSearchError>
where
    T: SolverNumeric,
    M: LocalSearchMonitor<T>,
{
    let _guard = CancelOnPanic::new(cancellation);
    let mut monitor = CancellableMonitor::new(&mut member.monitor, cancellation);
    engine.run(model, initial, &member.parameters, &mut monitor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        limits::SearchLimits,
        metaheuristic::{GreedyDescent, Lambda, Metaheuristic, StaticLambda},
        monitor::{NoMonitor, SearchProgress},
        parameters::LocalSearchParameters,
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

    /// The members of `PortfolioMember::varying_lambda` on `base`, each observed by an empty trajectory.
    fn traced(base: &LocalSearchParameters, members: usize) -> Vec<PortfolioMember<Trajectory>> {
        PortfolioMember::varying_lambda(base, lambdas(members))
            .into_iter()
            .map(|member| PortfolioMember {
                parameters: member.parameters,
                monitor: Trajectory::default(),
            })
            .collect()
    }

    #[test]
    fn test_each_member_answers_what_a_lone_run_of_it_answers() {
        let (model, berths, starts) = staggered_instance(16);
        let initial = SolutionView::new(&berths, &starts, 0);
        let thread_pool = pool(2);
        for execution in [Execution::Serial, Execution::Pool(&thread_pool)] {
            let mut members = traced(&guided(4000), 5);
            let outcome = Portfolio::new()
                .run(&model, initial, &mut members, execution)
                .expect("every member should run");
            for (member, answered) in members.iter().zip(outcome.members()) {
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
        let mut members = PortfolioMember::varying_lambda(&guided(5000), lambdas(5));
        members.extend(PortfolioMember::varying_lambda(&by_cycles, lambdas(2)));
        members.push(PortfolioMember::from(LocalSearchParameters {
            metaheuristic: Metaheuristic::GreedyDescent(GreedyDescent::default()),
            ..guided(5000)
        }));
        let serial = Portfolio::new()
            .run(&model, initial, &mut members, Execution::Serial)
            .expect("every member should run");
        assert_eq!(serial.members().len(), 8);
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

    /// A monitor that panics at the start of its run if told to, and otherwise records the candidates its run made.
    #[derive(Debug, Default)]
    struct PanicProbe {
        panics: bool,
        ended_after: Option<u64>,
    }

    impl LocalSearchMonitor<i64> for PanicProbe {
        fn on_start(&mut self, _: &Model<i64>, _: SearchProgress<'_, i64>) {
            assert!(!self.panics, "the probe panics");
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
                        ended_after: None,
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
            for member in &members[1..] {
                assert!(
                    member
                        .monitor
                        .ended_after
                        .is_none_or(|iterations| iterations < SIBLING_BUDGET),
                    "{execution:?}: a sibling ran to its budget"
                );
            }
        }

        // The engines of a run that panicked are loaded afresh by the next run.
        let mut members = PortfolioMember::varying_lambda(&guided(3000), lambdas(6));
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
    fn test_a_refused_member_is_named_before_any_member_runs() {
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
            Portfolio::new().run(&model, initial, &mut members, Execution::Serial),
            Err(PortfolioError::Member {
                member: 2,
                error: LocalSearchError::NoNeighborhood
            })
        );
        assert!(
            members
                .iter()
                .all(|member| member.monitor.ended_after.is_none())
        );

        let shorter = SolutionView::new(&berths[1..], &starts[1..], 0);
        let mut members = PortfolioMember::varying_lambda(&guided(100), lambdas(2));
        assert_eq!(
            Portfolio::new().run(&model, shorter, &mut members, Execution::Serial),
            Err(PortfolioError::Member {
                member: 0,
                error: LocalSearchError::VesselCountMismatch {
                    model_vessels: 4,
                    solution_vessels: 3
                }
            })
        );
    }

    // Every member decodes the initial order when its run starts, so an order without a feasible schedule refuses
    // every member, and the first is named.
    #[test]
    fn test_an_initial_order_without_a_feasible_schedule_refuses_the_first_member() {
        let model = Model::new(
            vec![0, 0],
            vec![10, 10],
            vec![1, 1],
            vec![Some(8), Some(8), Some(8), Some(8)],
            vec![vec![ClosedOpenInterval::new(0, 100)]; 2],
        );
        let berths = [BerthIndex::new(0), BerthIndex::new(0)];
        let initial = SolutionView::new(&berths, &[0, 8], 0);
        let mut members = PortfolioMember::varying_lambda(&guided(100), lambdas(3));
        assert_eq!(
            Portfolio::new().run(&model, initial, &mut members, Execution::Pool(&pool(2))),
            Err(PortfolioError::Member {
                member: 0,
                error: LocalSearchError::InfeasibleInitialSolution
            })
        );
    }

    // A kept portfolio keeps its engines and nothing else: after runs of more members on another model, it answers
    // what a fresh portfolio answers, and holds no engine beyond the most members it ran.
    #[test]
    fn test_a_kept_portfolio_answers_as_a_fresh_one_and_keeps_its_engines() {
        let (model, berths, starts) = staggered_instance(16);
        let initial = SolutionView::new(&berths, &starts, 0);
        let (other, other_berths, other_starts) = staggered_instance(30);
        let thread_pool = pool(2);
        let mut members = PortfolioMember::varying_lambda(&guided(3000), lambdas(3));
        let fresh = Portfolio::new()
            .run(&model, initial, &mut members, Execution::Serial)
            .expect("every member should run");

        let mut portfolio = Portfolio::new();
        let _ = portfolio
            .run(&model, initial, &mut members, Execution::Pool(&thread_pool))
            .expect("every member should run");
        let mut more = PortfolioMember::varying_lambda(&guided(3000), lambdas(5));
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
