// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A local-search engine for the dynamic berth allocation problem.
//!
//! An [`Engine`] improves a feasible schedule of a [`talos_model::Model`] by moves from a set of
//! [`Neighborhood`]s, under a [`Metaheuristic`], until a limit of [`SearchLimits`], the metaheuristic or a
//! [`LocalSearchMonitor`] ends the run. Its [`LocalSearchParameters`] say what a run explores, how it decides,
//! which [`Objective`] it minimises, when it ends and the seed of its random choices; a run is reproducible from
//! its model, its initial schedule, its parameters and its monitor, unless a deadline of its limits, which reads
//! the clock, ends it.
//!
//! A [`Portfolio`] runs several [`PortfolioMember`]s, each with parameters of its own, from one model and one
//! initial schedule on engines of their own, one after another or on a thread pool the caller hands in as an
//! [`Execution`], and answers each member's outcome and the best of them. The crate starts no thread of its own.
//!
//! The crate is a stack of layers, each on the ones before it:
//!
//! - `ring_arena`: closed rings of nodes spliced in constant time;
//! - `schedule`: the order of the vessels on each berth as rings, and the moves that change it with their records;
//! - `objective` and `decoding`: the cost of each service, and the schedule an order stands for;
//! - `operator`: the neighbourhoods and the operator that proposes their moves;
//! - `metaheuristic`: the rules that decide which candidates a run accepts;
//! - `statistics`, `termination`, `limits`, `monitor`, `parameters` and `outcome`: what a run counts, why it
//!   ends, what ends it, who observes it, what it is asked, and what it answers;
//! - `engine`: the run itself;
//! - `portfolio`: several runs with parameters of their own, side by side on engines of their own, whose best
//!   outcome is the answer.

mod ring_arena;

mod schedule;

mod decoding;
mod objective;

mod operator;

mod metaheuristic;

mod limits;
mod monitor;
mod outcome;
mod parameters;
mod statistics;
mod termination;

mod engine;

mod portfolio;

#[cfg(test)]
mod benchmarks;

pub use engine::Engine;
pub use limits::SearchLimits;
pub use metaheuristic::{
    AdditiveDynamicLambda, AtNewBest, DynamicLambda, GeometricDecay, GreedyDescent,
    GuidedLocalSearch, Lambda, Metaheuristic, MoveSelection, PenalizationTrigger, PenaltyDecay,
    StaticLambda, TieBreaking, heuristic_lambda,
};
pub use monitor::{LocalSearchMonitor, NoMonitor, SearchProgress};
pub use objective::Objective;
pub use operator::{MultiArmedBandit, Neighborhood, NeighborhoodSelection};
pub use outcome::LocalSearchOutcome;
pub use parameters::{InitialSolutionError, LocalSearchError, LocalSearchParameters};
pub use portfolio::{Execution, Portfolio, PortfolioError, PortfolioMember, PortfolioOutcome};
pub use statistics::LocalSearchStatistics;
pub use termination::{SearchCommand, TerminationReason};
