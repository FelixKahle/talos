// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! What one run of a portfolio is asked to do, and who observes it.

use crate::parameters::LocalSearchParameters;

/// One run of a portfolio: the parameters it runs with and the monitor told of its events.
///
/// A member's run is the run [`Engine::run`](crate::Engine::run) makes of its parameters and its monitor, from the
/// model and the initial schedule of the portfolio, on an engine of its own; members share nothing while they run.
/// The members of a portfolio differ by their parameters: under a round robin of the neighbourhoods and a guided
/// local search no choice of a run is random, so members that differ by their seed alone run alike, while members
/// that differ by the penalty weight of their guided local search, its [`Lambda`](crate::Lambda), take paths of
/// their own. A member that nobody observes has the monitor [`NoMonitor`](crate::NoMonitor), which costs nothing.
#[derive(Debug, Clone, PartialEq)]
pub struct PortfolioMember<M> {
    /// The parameters of the run: what it explores, how it decides, what it minimises, its limits and its seed.
    pub parameters: LocalSearchParameters,
    /// The monitor of the run, told of its events and asked whether it goes on.
    pub monitor: M,
}
