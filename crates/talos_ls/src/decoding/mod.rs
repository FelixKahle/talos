// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The schedule a vessel order on each berth stands for: each vessel's start and berth, and its cost.
//!
//! - [`ScheduleState`] holds a decoded schedule: each vessel's berth and start, each berth's cost and their sum.
//! - [`decode_full_unchecked`] decodes every berth of a graph, and [`decode_unchecked`] only the berths a move
//!   touched.

mod schedule_state;

mod decoder;

#[cfg(test)]
pub(super) use decoder::bench_mutation_cycles;
pub(super) use decoder::{decode_full_unchecked, decode_unchecked};
pub(super) use schedule_state::ScheduleState;
