// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::index::BerthIndex;

/// Assignment of a single vessel in a state.
///
/// An `Assignment` bundles the start time and berth chosen for a vessel.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Assignment<T> {
    /// The start time assigned to the vessel.
    pub start_time: T,

    /// The berth assigned to the vessel.
    pub berth: BerthIndex,
}

impl<T> Assignment<T> {
    /// Creates a new `Assignment` with the given start time and berth.
    #[inline(always)]
    pub fn new(start_time: T, berth: BerthIndex) -> Self {
        Self { start_time, berth }
    }
}

impl<T> std::fmt::Display for Assignment<T>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "start_time={}, berth={}", self.start_time, self.berth)
    }
}
