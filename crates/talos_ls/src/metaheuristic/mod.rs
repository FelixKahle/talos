// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The metaheuristics a run may follow, each as its settings and as the rule the engine asks during a run.
//!
//! - [`MoveSelection`] and [`TieBreaking`] say which improving move a descent takes; `TieBreaker` decides ties in
//!   a run.
//! - [`GreedyDescent`] accepts only improving moves; `GreedyDescentRule` is its run.
//! - [`GuidedLocalSearch`] descends on an objective augmented by penalties; `GuidedLocalSearchRule` is its run.
//! - [`Metaheuristic`] names one of them, and `AcceptanceRule` is what the engine asks of a run of either.

mod move_selection;
mod tie_breaking;

mod acceptance;

mod greedy_descent;
mod guided_local_search;

pub(super) use acceptance::{AcceptanceOutcome, AcceptanceRule, NeighborhoodExhaustionOutcome};
pub(super) use greedy_descent::GreedyDescentRule;
pub(super) use guided_local_search::GuidedLocalSearchRule;

pub use acceptance::Metaheuristic;
pub use greedy_descent::GreedyDescent;
pub use guided_local_search::{
    AdditiveDynamicLambda, AtNewBest, DynamicLambda, GeometricDecay, GuidedLocalSearch, Lambda,
    PenalizationTrigger, PenaltyDecay, StaticLambda, heuristic_lambda,
};
pub use move_selection::MoveSelection;
pub use tie_breaking::TieBreaking;
