// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Which of the improving moves of a scan a descent takes.

use super::tie_breaking::TieBreaking;

/// Which of the improving moves of a scan a descent takes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum MoveSelection {
    /// Scan the whole neighbourhood, keep the best improving move aside, two of equal objective decided by the
    /// tie-breaking, and take it once the scan is exhausted: steepest descent.
    BestImprovement(TieBreaking),
    /// Take the first improving move the scan finds.
    #[default]
    FirstImprovement,
}
