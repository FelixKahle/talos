// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The order of the vessels on each berth, and the moves that change it.
//!
//! - [`ScheduleGraph`] holds the vessels of each berth in order, as one ring per berth of a ring arena.
//! - [`ScheduleGraphDiff`] records the links a sequence of moves broke and made and the vessels it moved between
//!   berths.
//! - [`ScheduleGraphUndoLog`] records how to undo a sequence of moves, and undoes it.
//! - [`TouchedBerths`] marks the berths whose order a sequence of moves changed.
//! - [`Mutator`] makes the moves on a graph and records each in a diff, an undo log and the touched berths.

mod graph;
mod touched_berths;
mod undo_log;

mod mutator;

pub use graph::{
    BerthEdgeIter, BerthTopologyIter, DiffEdgeIter, DiffReallocationIter, ScheduleGraph,
    ScheduleGraphDiff, ScheduleGraphDiffEdge, ScheduleGraphDiffTracker, ScheduleGraphEdge,
    VesselSequenceIter, VesselSequenceRevIter,
};
pub use mutator::Mutator;
pub use touched_berths::{
    TouchedBerths, TouchedBerthsTracker, TouchedIndicesIter, UntouchedBerthsIter,
};
pub use undo_log::{ScheduleGraphUndoLog, ScheduleGraphUndoTracker, UndoInstruction, UndoLogIter};
