// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

use rand::{Rng, RngExt};

// ----------------------------------------------------------------
// Tie-Breaking Strategy
// ----------------------------------------------------------------

/// Decides which move to keep when two candidates have the same objective
/// value in best-improvement mode.
///
/// When an admissible candidate ties with the current buffer, this
/// strategy decides whether the new candidate replaces the buffer.
pub trait TieBreakingStrategy: std::fmt::Debug {
    /// Returns `true` if the new candidate should replace the current
    /// buffer when both have the same objective value.
    fn break_tie(&mut self) -> bool;
}

// ----------------------------------------------------------------
// KeepFirst
// ----------------------------------------------------------------

/// Always keeps the first-seen move on a tie (rejects the newer one).
///
/// This is the default tie-breaking strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct KeepFirst;

impl TieBreakingStrategy for KeepFirst {
    #[inline]
    fn break_tie(&mut self) -> bool {
        false
    }
}

// ----------------------------------------------------------------
// KeepLast
// ----------------------------------------------------------------

/// Always keeps the last-seen move on a tie (replaces the buffer).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct KeepLast;

impl TieBreakingStrategy for KeepLast {
    #[inline]
    fn break_tie(&mut self) -> bool {
        true
    }
}

// ----------------------------------------------------------------
// RandomTieBreak
// ----------------------------------------------------------------

/// Breaks ties randomly with a fair coin flip.
///
/// This helps avoid deterministic cycling on flat fitness landscapes.
#[derive(Debug)]
pub struct RandomTieBreak<R> {
    rng: R,
}

impl<R: Rng> RandomTieBreak<R> {
    /// Creates a new random tie-breaking strategy.
    #[inline]
    pub fn new(rng: R) -> Self {
        Self { rng }
    }
}

impl<R: Rng + std::fmt::Debug> TieBreakingStrategy for RandomTieBreak<R> {
    #[inline]
    fn break_tie(&mut self) -> bool {
        self.rng.random()
    }
}
