// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

/// Controls how the search selects among admissible moves in a
/// neighbourhood scan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SelectionStrategy {
    /// Evaluate the full neighbourhood, buffer every admissible move,
    /// and commit the best one. This is the standard Tabu Search
    /// behaviour from the literature.
    BestImprovement,

    /// Accept the first admissible move encountered and skip the
    /// rest of the neighbourhood.
    FirstImprovement,
}

impl std::fmt::Display for SelectionStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SelectionStrategy::BestImprovement => write!(f, "BestImprovement"),
            SelectionStrategy::FirstImprovement => write!(f, "FirstImprovement"),
        }
    }
}
