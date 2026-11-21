// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Why a schedule is not a feasible schedule of a model.

use crate::index::{BerthIndex, VesselIndex};
use std::fmt;

/// Why a schedule is not a feasible schedule of a [`Model`](crate::Model), as
/// [`Model::check_feasibility`](crate::Model::check_feasibility) answers it.
///
/// Each variant names the first condition a schedule breaks and the values that break it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Infeasibility<T> {
    /// The schedule does not hold one assignment per vessel of the model.
    VesselCount {
        /// The number of vessels the schedule assigns.
        scheduled: usize,
        /// The number of vessels of the model.
        num_vessels: usize,
    },
    /// A vessel is served at a berth the model does not have.
    BerthBeyondModel {
        /// The vessel.
        vessel: VesselIndex,
        /// The berth it is served at.
        berth: BerthIndex,
        /// The number of berths of the model.
        num_berths: usize,
    },
    /// A vessel is served at a berth it may not use.
    BerthNotAllowed {
        /// The vessel.
        vessel: VesselIndex,
        /// The berth it is served at.
        berth: BerthIndex,
    },
    /// A vessel's service starts before the vessel arrives.
    StartBeforeArrival {
        /// The vessel.
        vessel: VesselIndex,
        /// The time its service starts.
        start_time: T,
        /// The time it arrives.
        arrival_time: T,
    },
    /// A vessel's service fits no opening of its berth.
    OutsideOpenings {
        /// The vessel.
        vessel: VesselIndex,
        /// The berth it is served at.
        berth: BerthIndex,
        /// The time its service starts.
        start_time: T,
        /// How long its service at the berth takes.
        processing_time: T,
    },
    /// A vessel's service ends after the vessel must have left.
    LateDeparture {
        /// The vessel.
        vessel: VesselIndex,
        /// The time its service ends.
        departure_time: T,
        /// The time by which it must have left.
        latest_departure_time: T,
    },
    /// The services of two vessels at one berth share a moment of time.
    Overlap {
        /// The berth.
        berth: BerthIndex,
        /// The vessel whose service starts first, or the lower of the two where both start together.
        first: VesselIndex,
        /// The other vessel.
        second: VesselIndex,
    },
}

impl<T> fmt::Display for Infeasibility<T>
where
    T: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::VesselCount {
                scheduled,
                num_vessels,
            } => write!(
                f,
                "a schedule of {scheduled} vessels for a model of {num_vessels} vessels"
            ),
            Self::BerthBeyondModel {
                vessel,
                berth,
                num_berths,
            } => write!(
                f,
                "{vessel} is served at {berth}, beyond the {num_berths} berths"
            ),
            Self::BerthNotAllowed { vessel, berth } => {
                write!(f, "{vessel} is served at {berth}, which it may not use")
            }
            Self::StartBeforeArrival {
                vessel,
                start_time,
                arrival_time,
            } => write!(
                f,
                "{vessel} is served from {start_time}, before it arrives at {arrival_time}"
            ),
            Self::OutsideOpenings {
                vessel,
                berth,
                start_time,
                processing_time,
            } => write!(
                f,
                "{vessel} is served at {berth} from {start_time} for {processing_time}, within no opening of the berth"
            ),
            Self::LateDeparture {
                vessel,
                departure_time,
                latest_departure_time,
            } => write!(
                f,
                "{vessel} leaves at {departure_time}, after its latest departure time {latest_departure_time}"
            ),
            Self::Overlap {
                berth,
                first,
                second,
            } => write!(f, "{first} and {second} are served at {berth} at once"),
        }
    }
}

impl<T> std::error::Error for Infeasibility<T> where T: fmt::Debug + fmt::Display {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_every_infeasibility_states_what_breaks_with_its_values() {
        let (vessel, berth) = (VesselIndex::new(1), BerthIndex::new(2));
        let cases: [(Infeasibility<i64>, &str); 7] = [
            (
                Infeasibility::VesselCount {
                    scheduled: 3,
                    num_vessels: 2,
                },
                "a schedule of 3 vessels for a model of 2 vessels",
            ),
            (
                Infeasibility::BerthBeyondModel {
                    vessel,
                    berth,
                    num_berths: 2,
                },
                "Vessel 1 is served at Berth 2, beyond the 2 berths",
            ),
            (
                Infeasibility::BerthNotAllowed { vessel, berth },
                "Vessel 1 is served at Berth 2, which it may not use",
            ),
            (
                Infeasibility::StartBeforeArrival {
                    vessel,
                    start_time: 4,
                    arrival_time: 5,
                },
                "Vessel 1 is served from 4, before it arrives at 5",
            ),
            (
                Infeasibility::OutsideOpenings {
                    vessel,
                    berth,
                    start_time: 10,
                    processing_time: 0,
                },
                "Vessel 1 is served at Berth 2 from 10 for 0, within no opening of the berth",
            ),
            (
                Infeasibility::LateDeparture {
                    vessel,
                    departure_time: 30,
                    latest_departure_time: 25,
                },
                "Vessel 1 leaves at 30, after its latest departure time 25",
            ),
            (
                Infeasibility::Overlap {
                    berth,
                    first: VesselIndex::new(0),
                    second: vessel,
                },
                "Vessel 0 and Vessel 1 are served at Berth 2 at once",
            ),
        ];
        for (infeasibility, message) in cases {
            assert_eq!(infeasibility.to_string(), message);
        }
    }
}
