// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! What one run of a portfolio is asked to do, and who observes it.

use crate::{
    metaheuristic::{GuidedLocalSearch, Lambda, Metaheuristic},
    monitor::NoMonitor,
    parameters::LocalSearchParameters,
};

/// One run of a portfolio: the parameters it runs with and the monitor told of its events.
///
/// A member's run is the run [`Engine::run`](crate::Engine::run) makes of its parameters and its monitor, from the
/// model and the initial schedule of the portfolio, on an engine of its own; members share nothing while they run.
/// The members of a portfolio differ by their parameters: under a round robin of the neighbourhoods and a guided
/// local search no choice of a run is random, so members that differ by their seed alone run alike, and
/// [`varying_lambda`](Self::varying_lambda) builds members that differ by the penalty weight of their guided local
/// search. A member that nobody observes has the monitor [`NoMonitor`], which costs nothing.
#[derive(Debug, Clone, PartialEq)]
pub struct PortfolioMember<M> {
    /// The parameters of the run: what it explores, how it decides, what it minimises, its limits and its seed.
    pub parameters: LocalSearchParameters,
    /// The monitor of the run, told of its events and asked whether it goes on.
    pub monitor: M,
}

impl From<LocalSearchParameters> for PortfolioMember<NoMonitor> {
    /// The member that runs with `parameters` and that nobody observes.
    fn from(parameters: LocalSearchParameters) -> Self {
        Self {
            parameters,
            monitor: NoMonitor,
        }
    }
}

impl PortfolioMember<NoMonitor> {
    /// One member per weight of `lambdas`, in their order, that nobody observes: each has the parameters of `base`
    /// with the penalty weight of its guided local search, and how it evolves, replaced by the weight.
    ///
    /// ```
    /// # use talos_ls::{Lambda, LocalSearchParameters, PortfolioMember, StaticLambda};
    /// let base = LocalSearchParameters::default();
    /// let members = PortfolioMember::varying_lambda(
    ///     &base,
    ///     [0.5, 1.0, 2.0].map(|weight| Lambda::Static(StaticLambda::new(weight))),
    /// );
    /// assert_eq!(members.len(), 3);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if the metaheuristic of `base` is not a guided local search.
    pub fn varying_lambda(
        base: &LocalSearchParameters,
        lambdas: impl IntoIterator<Item = Lambda>,
    ) -> Vec<Self> {
        let Metaheuristic::GuidedLocalSearch(guided) = base.metaheuristic else {
            panic!(
                "called `PortfolioMember::varying_lambda` with base parameters of {:?}, which is not a guided \
                 local search",
                base.metaheuristic
            );
        };
        lambdas
            .into_iter()
            .map(|lambda| {
                // Each member owns its parameters, so the neighbourhoods of `base` are copied once per member.
                Self::from(LocalSearchParameters {
                    metaheuristic: Metaheuristic::GuidedLocalSearch(GuidedLocalSearch {
                        lambda,
                        ..guided
                    }),
                    ..base.clone()
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        limits::SearchLimits,
        metaheuristic::{
            AtNewBest, DynamicLambda, GreedyDescent, PenalizationTrigger, PenaltyDecay,
            StaticLambda,
        },
        operator::Neighborhood,
    };
    use std::num::NonZeroU64;

    #[test]
    fn test_varying_lambda_replaces_the_lambda_alone_in_the_order_given() {
        let guided = GuidedLocalSearch {
            lambda: Lambda::Static(StaticLambda::new(9.0)),
            decay: PenaltyDecay::None,
            trigger: PenalizationTrigger::AfterMoves(NonZeroU64::MIN),
        };
        let base = LocalSearchParameters {
            neighborhoods: vec![Neighborhood::InterBerthShift],
            metaheuristic: Metaheuristic::GuidedLocalSearch(guided),
            limits: SearchLimits {
                iterations: Some(7),
                ..SearchLimits::default()
            },
            seed: 5,
            ..LocalSearchParameters::default()
        };
        let lambdas = [
            Lambda::Static(StaticLambda::new(0.5)),
            Lambda::Dynamic(DynamicLambda::new(1.0, 0.1, AtNewBest::Restart, 0.5, 2.0)),
        ];
        let members = PortfolioMember::varying_lambda(&base, lambdas);
        assert_eq!(members.len(), 2);
        for (member, lambda) in members.iter().zip(lambdas) {
            let expected = LocalSearchParameters {
                metaheuristic: Metaheuristic::GuidedLocalSearch(GuidedLocalSearch {
                    lambda,
                    ..guided
                }),
                ..base.clone()
            };
            assert_eq!(member.parameters, expected);
            assert_eq!(member.monitor, NoMonitor);
        }
    }

    #[test]
    #[should_panic(
        expected = "called `PortfolioMember::varying_lambda` with base parameters of GreedyDescent"
    )]
    fn test_varying_lambda_refuses_a_base_without_a_guided_local_search() {
        let base = LocalSearchParameters {
            metaheuristic: Metaheuristic::GreedyDescent(GreedyDescent::default()),
            ..LocalSearchParameters::default()
        };
        let _ = PortfolioMember::varying_lambda(&base, [Lambda::default()]);
    }
}
