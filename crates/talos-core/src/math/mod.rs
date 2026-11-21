// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Numbers and the intervals between them: the integer types of a problem and closed-open intervals of them.

mod interval;
mod num;

pub use interval::{ClosedOpenInterval, ClosedOpenIntervalIter};
pub use num::SolverNumeric;
