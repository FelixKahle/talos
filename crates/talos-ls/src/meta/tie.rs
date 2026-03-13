// Copyright (c) 2026 Felix Kahle.
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

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
