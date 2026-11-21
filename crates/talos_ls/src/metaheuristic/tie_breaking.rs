// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

use rand::RngExt;
use rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

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

/// Breaks ties randomly with a fair coin flip, drawn from a generator seeded with a given number.
///
/// This helps avoid deterministic cycling on flat fitness landscapes; the same seed flips the same coins.
#[derive(Debug, Clone)]
pub struct RandomTieBreak {
    generator: Xoshiro256PlusPlus,
}

impl RandomTieBreak {
    /// The tie-break whose coin flips are drawn from a generator seeded with `seed`.
    #[inline]
    pub fn new(seed: u64) -> Self {
        Self {
            generator: Xoshiro256PlusPlus::seed_from_u64(seed),
        }
    }
}

impl TieBreakingStrategy for RandomTieBreak {
    #[inline]
    fn break_tie(&mut self) -> bool {
        self.generator.random()
    }
}
