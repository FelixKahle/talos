// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The ids of a problem's vessels and berths, and the walks over all ids of each kind.

use talos_core::{TypedIndex, TypedIndexIter, TypedIndexTag};

/// The tag that makes a [`TypedIndex`] a [`VesselIndex`].
#[derive(Debug)]
pub struct VesselIndexTag;

impl TypedIndexTag for VesselIndexTag {
    const NAME: &'static str = "Vessel";
}

/// The id of a vessel: its position in the vessel tables of a model and of a solution.
pub type VesselIndex = TypedIndex<VesselIndexTag>;

/// The tag that makes a [`TypedIndex`] a [`BerthIndex`].
#[derive(Debug)]
pub struct BerthIndexTag;

impl TypedIndexTag for BerthIndexTag {
    const NAME: &'static str = "Berth";
}

/// The id of a berth: its position in the berth tables of a model.
pub type BerthIndex = TypedIndex<BerthIndexTag>;

/// The ids of every vessel, in increasing order.
pub type VesselIndexIter = TypedIndexIter<VesselIndexTag>;

/// The ids of every berth, in increasing order.
pub type BerthIndexIter = TypedIndexIter<BerthIndexTag>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ids_name_their_kind() {
        assert_eq!(format!("{}", VesselIndex::new(3)), "Vessel 3");
        assert_eq!(format!("{:?}", BerthIndex::new(1)), "Berth 1");
    }

    #[test]
    fn test_walks_yield_the_ids_of_their_kind() {
        assert_eq!(
            VesselIndexIter::new(2).next_back(),
            Some(VesselIndex::new(1))
        );
        assert_eq!(BerthIndexIter::new(1).next(), Some(BerthIndex::new(0)));
    }
}
