// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! How a best-improvement descent decides between two improving moves of equal objective.

use rand::RngExt;
use rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

/// Which of two improving moves of equal objective a best-improvement descent keeps aside.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum TieBreaking {
    /// Keep the move found first.
    #[default]
    KeepFirst,
    /// Keep the move found last.
    KeepLast,
    /// Flip a fair coin, drawn from the run's seeded generator.
    Random,
}

/// The tie-breaking of one run, with its generator where it draws.
#[derive(Debug, Clone)]
pub enum TieBreaker {
    KeepFirst,
    KeepLast,
    Random(Xoshiro256PlusPlus),
}

impl TieBreaker {
    /// The tie-breaking of `rule`, drawing from a generator seeded with `seed` where it draws.
    pub fn new(rule: TieBreaking, seed: u64) -> Self {
        match rule {
            TieBreaking::KeepFirst => Self::KeepFirst,
            TieBreaking::KeepLast => Self::KeepLast,
            TieBreaking::Random => Self::Random(Xoshiro256PlusPlus::seed_from_u64(seed)),
        }
    }

    /// Whether the newer of two moves of equal objective replaces the one kept aside.
    pub fn prefers_newer(&mut self) -> bool {
        match self {
            Self::KeepFirst => false,
            Self::KeepLast => true,
            Self::Random(generator) => generator.random(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keep_first_and_keep_last_decide_alike_every_time() {
        let mut first = TieBreaker::new(TieBreaking::KeepFirst, 0);
        let mut last = TieBreaker::new(TieBreaking::KeepLast, 0);
        assert!((0..8).all(|_| !first.prefers_newer() && last.prefers_newer()));
    }

    #[test]
    fn test_random_tie_breaking_repeats_its_flips_for_a_seed_and_varies_with_it() {
        let flips = |seed| {
            let mut breaker = TieBreaker::new(TieBreaking::Random, seed);
            (0..64)
                .map(|_| breaker.prefers_newer())
                .collect::<Vec<bool>>()
        };
        assert_eq!(flips(7), flips(7));
        assert_ne!(flips(7), flips(8));
        assert!(flips(7).contains(&true) && flips(7).contains(&false));
    }
}
