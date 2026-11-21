// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Where and when one vessel is served.

use crate::index::BerthIndex;

/// Where and when one vessel is served: the berth and the time its service there starts.
///
/// A solution answers one assignment per vessel. The record holds no invariant of its own: whether the berth
/// belongs to a model, the vessel may use it and the start keeps the model's conditions is what
/// [`Model::check_feasibility`](crate::Model::check_feasibility) answers for a whole schedule.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Assignment<T> {
    /// The time the vessel's service starts, in the model's unit of time.
    pub start_time: T,
    /// The berth the vessel is served at.
    pub berth: BerthIndex,
}
