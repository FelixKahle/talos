// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The problem and the answer of the dynamic berth allocation problem.
//!
//! - [`VesselIndex`] and [`BerthIndex`] are the ids of a problem's vessels and berths, and [`VesselIndexIter`]
//!   and [`BerthIndexIter`] walk over all of one kind.
//! - [`Model`] is a problem instance: when each vessel arrives and must have left, its weight, its processing time
//!   at each berth it may use, and when each berth is open. Input that makes no model is refused as a
//!   [`ModelError`].
//! - [`Solution`] is a schedule of every vessel, a berth and a start time each as an [`Assignment`], and
//!   [`SolutionView`] the same schedule borrowed; tables of different lengths are refused as a
//!   [`SolutionLengthMismatch`].
//! - [`Model::check_feasibility`] tells whether a schedule keeps every condition of its model, or answers the
//!   first it breaks as an [`Infeasibility`].

mod assignment;
mod index;
mod infeasibility;
mod model;
mod solution;

pub use assignment::Assignment;
pub use index::{
    BerthIndex, BerthIndexIter, BerthIndexTag, VesselIndex, VesselIndexIter, VesselIndexTag,
};
pub use infeasibility::Infeasibility;
pub use model::{Model, ModelError};
pub use solution::{Solution, SolutionIter, SolutionLengthMismatch, SolutionView};
