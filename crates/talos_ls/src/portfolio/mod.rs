// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The independent portfolio: several runs of the engine from one model and one initial schedule, each with
//! parameters of its own, on engines of their own, whose best outcome is the answer.
//!
//! - `member`: what one run of a portfolio is asked to do, and who observes it;
//! - `outcome`: what a portfolio answers, and why it refuses to start;
//! - `cancellation`: the flag by which a member that panics stops its siblings;
//! - `run`: the engines of a portfolio, and the runs of its members, one after another or on a thread pool.

mod member;
mod outcome;

mod cancellation;

mod run;

pub use member::PortfolioMember;
pub use outcome::{PortfolioError, PortfolioOutcome};
pub use run::{Execution, Portfolio};
