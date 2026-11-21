// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{statistics::LocalSearchStatistics, termination::TerminationReason};
use talos_model::Solution;

/// What a run found: its best schedule, why it ended, and the work it did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalSearchOutcome<T> {
    /// The best schedule the run found, the initial one if it found none better.
    pub solution: Solution<T>,
    /// Why the run ended.
    pub termination_reason: TerminationReason,
    /// The work the run did.
    pub statistics: LocalSearchStatistics,
}
