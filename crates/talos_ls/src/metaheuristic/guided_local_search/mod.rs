// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Guided local search: a descent on the objective augmented by penalties of the features of the schedules it
//! stands in when it gets stuck.
//!
//! The features of a schedule are its links (a vessel next to a vessel or to the boundary of its berth, with the
//! boundaries of all berths taken as one) and its assignments (a vessel on a berth). The search accepts a move when
//! it lowers `objective + lambda * penalties`, the penalties of the features of the schedule summed. When the
//! trigger fires, it adds one penalty to every feature of the accepted schedule of the largest utility
//! `1 / (1 + penalty)`, which pushes the search out of the basin it stands in.
//!
//! - `settings` holds the settings of a search: its penalty weight, the decay of its penalties and its trigger.
//! - `penalty_memory` counts the penalties of every feature.
//! - `schedules` evolves the penalty weight and decays the penalties over a run.
//! - `rule` is a search under way, the rule the engine asks.

mod settings;

mod penalty_memory;

mod schedules;

mod rule;

pub use rule::GuidedLocalSearchRule;

pub use settings::{
    AdditiveDynamicLambda, DynamicLambda, GeometricDecay, GuidedLocalSearch, Lambda,
    PenalizationTrigger, PenaltyDecay, StaticLambda, heuristic_lambda,
};
