// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! What a run is asked to do, and why a run refuses to start.

use crate::{
    limits::SearchLimits,
    metaheuristic::Metaheuristic,
    objective::Objective,
    operator::{Neighborhood, NeighborhoodSelection},
};
use talos_core::SolverNumeric;
use talos_model::{BerthIndex, Model, SolutionView, VesselIndex};

/// The settings of a run: what it explores, how it decides, what it minimises, when it ends, and the seed of its
/// random choices.
///
/// A run is reproducible from its model, its initial schedule, these parameters and its monitor: the same inputs
/// make the same candidates in the same order, and only the deadlines of [`SearchLimits`] depend on the clock.
#[derive(Debug, Clone, PartialEq)]
pub struct LocalSearchParameters {
    /// The neighbourhoods the run explores, in the order a selection takes them; at least one.
    pub neighborhoods: Vec<Neighborhood>,
    /// The rule that chooses which neighbourhood proposes the next move.
    pub selection: NeighborhoodSelection,
    /// The strategy that decides which candidates the run accepts and when it ends.
    pub metaheuristic: Metaheuristic,
    /// The objective the run minimises.
    pub objective: Objective,
    /// The budgets and deadlines that end the run.
    pub limits: SearchLimits,
    /// The seed every random choice of the run is drawn from.
    pub seed: u64,
}

impl Default for LocalSearchParameters {
    /// Every neighbourhood under round robin, a guided local search with its default settings, the weighted
    /// turnaround time, no limit and seed zero.
    ///
    /// A guided local search never ends by itself, so a run of these parameters ends only by a limit set or by its
    /// monitor.
    fn default() -> Self {
        Self {
            neighborhoods: Neighborhood::ALL.to_vec(),
            selection: NeighborhoodSelection::default(),
            metaheuristic: Metaheuristic::default(),
            objective: Objective::default(),
            limits: SearchLimits::default(),
            seed: 0,
        }
    }
}

/// Why a run refused to start.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LocalSearchError {
    /// The parameters name no neighbourhood to explore.
    NoNeighborhood,
    /// The initial schedule is not one a run can start from.
    InitialSolution(InitialSolutionError),
}

impl From<InitialSolutionError> for LocalSearchError {
    fn from(error: InitialSolutionError) -> Self {
        Self::InitialSolution(error)
    }
}

// A refused initial schedule is the whole story of the refusal, so its message is the refusal's own.
impl std::fmt::Display for LocalSearchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoNeighborhood => write!(f, "the parameters name no neighborhood to explore"),
            Self::InitialSolution(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for LocalSearchError {}

/// Why an initial schedule is not one a run can start from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InitialSolutionError {
    /// The initial schedule has another number of vessels than the model.
    VesselCountMismatch {
        /// The vessels of the model.
        model_vessels: usize,
        /// The vessels of the initial schedule.
        solution_vessels: usize,
    },
    /// The initial schedule assigns a vessel to a berth the model does not have.
    BerthOutOfBounds {
        /// The vessel assigned.
        vessel: VesselIndex,
        /// The berth it is assigned to.
        berth: BerthIndex,
        /// The berths of the model.
        num_berths: usize,
    },
    /// The initial schedule assigns a vessel to a berth it may not use.
    IncompatibleBerth {
        /// The vessel assigned.
        vessel: VesselIndex,
        /// The berth it may not use.
        berth: BerthIndex,
    },
    /// The order of the vessels on the berths of the initial schedule has no feasible schedule: a vessel fits no
    /// opening of its berth before its latest departure.
    InfeasibleOrder,
}

impl std::fmt::Display for InitialSolutionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::VesselCountMismatch {
                model_vessels,
                solution_vessels,
            } => write!(
                f,
                "the initial solution has {solution_vessels} vessels but the model has {model_vessels}"
            ),
            Self::BerthOutOfBounds {
                vessel,
                berth,
                num_berths,
            } => write!(
                f,
                "the initial solution assigns {vessel} to {berth}, outside a model of {num_berths} berths"
            ),
            Self::IncompatibleBerth { vessel, berth } => {
                write!(
                    f,
                    "the initial solution assigns {vessel} to {berth}, which it may not use"
                )
            }
            Self::InfeasibleOrder => {
                write!(
                    f,
                    "the order of the initial solution has no feasible schedule"
                )
            }
        }
    }
}

impl std::error::Error for InitialSolutionError {}

/// Refuses parameters that name no neighbourhood, which [`Engine::run`](crate::Engine::run) refuses before it
/// looks at the initial schedule.
///
/// # Errors
///
/// [`LocalSearchError::NoNeighborhood`] if `parameters` name no neighbourhood.
pub(super) fn check_neighborhoods(
    parameters: &LocalSearchParameters,
) -> Result<(), LocalSearchError> {
    match parameters.neighborhoods.is_empty() {
        true => Err(LocalSearchError::NoNeighborhood),
        false => Ok(()),
    }
}

/// Refuses an initial schedule that is not a schedule of the vessels and berths of `model`: one berth per vessel,
/// each a berth of the model the vessel may use. It costs one pass over the vessels.
///
/// # Errors
///
/// [`InitialSolutionError::VesselCountMismatch`], [`InitialSolutionError::BerthOutOfBounds`] or
/// [`InitialSolutionError::IncompatibleBerth`], for the first vessel that breaks its condition.
pub(super) fn check_initial_solution<T>(
    model: &Model<T>,
    initial: SolutionView<'_, T>,
) -> Result<(), InitialSolutionError>
where
    T: SolverNumeric,
{
    if initial.num_vessels() != model.num_vessels() {
        return Err(InitialSolutionError::VesselCountMismatch {
            model_vessels: model.num_vessels(),
            solution_vessels: initial.num_vessels(),
        });
    }
    for (vessel, &berth) in initial.assigned_berths().iter_enumerated() {
        if berth.index() >= model.num_berths() {
            return Err(InitialSolutionError::BerthOutOfBounds {
                vessel,
                berth,
                num_berths: model.num_berths(),
            });
        }
        if model.vessel_processing_time(vessel, berth).is_none() {
            return Err(InitialSolutionError::IncompatibleBerth { vessel, berth });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use talos_core::ClosedOpenInterval;

    /// Two vessels; V0 may use only B0, V1 either berth.
    fn model() -> Model<i64> {
        Model::new(
            vec![0, 0],
            vec![100, 100],
            vec![1, 1],
            vec![Some(5), None, Some(5), Some(5)],
            vec![
                vec![ClosedOpenInterval::new(0, 200)],
                vec![ClosedOpenInterval::new(0, 200)],
            ],
        )
    }

    #[test]
    fn test_check_initial_solution_accepts_a_schedule_of_the_model() {
        let berths = [BerthIndex::new(0), BerthIndex::new(1)];
        let view = SolutionView::new(&berths, &[0, 0], 0);
        assert_eq!(check_initial_solution(&model(), view), Ok(()));
    }

    #[test]
    fn test_check_initial_solution_refuses_another_count_of_vessels() {
        let berths = [BerthIndex::new(0)];
        let view = SolutionView::new(&berths, &[0], 0);
        assert_eq!(
            check_initial_solution(&model(), view),
            Err(InitialSolutionError::VesselCountMismatch {
                model_vessels: 2,
                solution_vessels: 1
            })
        );
    }

    #[test]
    fn test_check_initial_solution_refuses_a_berth_outside_and_a_berth_not_allowed() {
        let berths = [BerthIndex::new(0), BerthIndex::new(2)];
        let view = SolutionView::new(&berths, &[0, 0], 0);
        assert_eq!(
            check_initial_solution(&model(), view),
            Err(InitialSolutionError::BerthOutOfBounds {
                vessel: VesselIndex::new(1),
                berth: BerthIndex::new(2),
                num_berths: 2
            })
        );
        let berths = [BerthIndex::new(1), BerthIndex::new(0)];
        let view = SolutionView::new(&berths, &[0, 0], 0);
        assert_eq!(
            check_initial_solution(&model(), view),
            Err(InitialSolutionError::IncompatibleBerth {
                vessel: VesselIndex::new(0),
                berth: BerthIndex::new(1)
            })
        );
    }

    #[test]
    fn test_check_neighborhoods_refuses_parameters_without_a_neighborhood() {
        let parameters = LocalSearchParameters {
            neighborhoods: Vec::new(),
            ..LocalSearchParameters::default()
        };
        assert_eq!(
            check_neighborhoods(&parameters),
            Err(LocalSearchError::NoNeighborhood)
        );
        assert_eq!(
            check_neighborhoods(&LocalSearchParameters::default()),
            Ok(())
        );
    }

    #[test]
    fn test_errors_display_their_values() {
        let error = InitialSolutionError::BerthOutOfBounds {
            vessel: VesselIndex::new(1),
            berth: BerthIndex::new(2),
            num_berths: 2,
        };
        let message =
            "the initial solution assigns Vessel 1 to Berth 2, outside a model of 2 berths";
        assert_eq!(error.to_string(), message);
        assert_eq!(LocalSearchError::from(error).to_string(), message);
        assert_eq!(
            LocalSearchError::NoNeighborhood.to_string(),
            "the parameters name no neighborhood to explore"
        );
    }
}
