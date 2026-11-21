// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The moves from one schedule to its neighbours, and the operator that proposes them to a run.
//!
//! - `filter` holds the tests by which a neighbourhood skips a move no optimum needs.
//! - `swap` and `shift` hold the scans of the four neighbourhoods: the exchanges of two vessels, and the insertions
//!   of one vessel at another place.
//! - [`Neighborhood`] names a neighbourhood, and `LocalSearchOperator` is what the engine draws moves from.
//! - `CompoundOperator` proposes the moves of several neighbourhoods, chosen among by a [`NeighborhoodSelection`].

mod filter;

mod shift;
mod swap;

mod neighborhood;

mod composite;

pub(super) use composite::CompoundOperator;
pub(super) use neighborhood::{AcceptedMove, LocalSearchOperator};

pub use composite::{MultiArmedBandit, NeighborhoodSelection};
pub use neighborhood::Neighborhood;
