// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The order of the vessels on each berth, and the moves that change it.
//!
//! - [`ScheduleGraph`] holds the vessels of each berth in order, as one ring per berth of a ring arena, and names
//!   the ends of the links between consecutive places of a berth as [`LinkEnd`]s.
//! - [`ScheduleGraphDiff`] records the links a sequence of moves broke and made and the vessels it moved between
//!   berths.
//! - `ScheduleGraphUndoLog` records how to undo a sequence of moves, and undoes it.
//! - [`TouchedBerths`] marks the berths whose order a sequence of moves changed.
//! - [`Mutator`] owns a graph and the records of the moves made on it: it makes each move and records it in a
//!   diff, an undo log and the touched berths.

mod graph;

mod diff;
mod touched_berths;
mod undo_log;

mod mutator;

pub(super) use diff::ScheduleGraphDiff;
pub(super) use graph::{LinkEnd, ScheduleGraph, VesselSequenceIter};
pub(super) use mutator::Mutator;
pub(super) use touched_berths::TouchedBerths;
