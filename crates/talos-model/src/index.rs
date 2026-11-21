// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

use talos_core::index::{TypedIndex, TypedIndexTag};

/// Marker type for indices into a vessel collection.
///
/// Use the `VesselIndex` type alias instead of referring to this marker
/// directly in most code.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VesselIndexTag;

impl TypedIndexTag for VesselIndexTag {
    const NAME: &'static str = "VesselIndex";
}

/// Strongly typed index for vessels.
///
/// This is a zero-cost wrapper around `usize` that prevents mixing
/// vessel indices with other index domains.
pub type VesselIndex = TypedIndex<VesselIndexTag>;

/// Marker type for indices into a berth collection.
///
/// Use the `BerthIndex` type alias instead of referring to this marker
/// directly in most code.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BerthIndexTag;

impl TypedIndexTag for BerthIndexTag {
    const NAME: &'static str = "BerthIndex";
}

/// Strongly typed index for berths.
///
/// This is a zero-cost wrapper around `usize` that prevents mixing
/// berth indices with other index domains.
pub type BerthIndex = TypedIndex<BerthIndexTag>;
