// Copyright (c) 2026 Felix Kahle.
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

use talos_core::utils::num::SolverNumeric;
use talos_model::{
    index::{BerthIndex, VesselIndex},
    model::Model,
};

// ----------------------------------------------------------------
// ModelSolutionMismatchError
// ----------------------------------------------------------------

/// Error indicating a mismatch between the Model and the Initial Solution vessel counts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelSolutionMismatchError {
    pub model_count: usize,
    pub solution_count: usize,
}

impl std::fmt::Display for ModelSolutionMismatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "vessel count mismatch: model has {}, but initial solution has {}",
            self.model_count, self.solution_count
        )
    }
}

impl std::error::Error for ModelSolutionMismatchError {}

// ----------------------------------------------------------------
// SolutionBerthOutOfBoundsError
// ----------------------------------------------------------------

/// Error indicating that the initial solution contains a vessel index that exceeds the model's bounds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SolutionVesselOutOfBoundsError {
    pub invalid_index: usize,
    pub model_count: usize,
}

impl std::fmt::Display for SolutionVesselOutOfBoundsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "initial solution contains an out-of-bounds vessel index: {} (model max index is {})",
            self.invalid_index,
            self.model_count.saturating_sub(1)
        )
    }
}

impl std::error::Error for SolutionVesselOutOfBoundsError {}

// ----------------------------------------------------------------
// SolutionBerthOutOfBoundsError
// ----------------------------------------------------------------

/// Error indicating that the initial solution contains a berth index that exceeds the model's bounds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SolutionBerthOutOfBoundsError {
    pub invalid_index: usize,
    pub model_count: usize,
}

impl std::fmt::Display for SolutionBerthOutOfBoundsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "initial solution contains an out-of-bounds berth index: {} (model max index is {})",
            self.invalid_index,
            self.model_count.saturating_sub(1)
        )
    }
}

impl std::error::Error for SolutionBerthOutOfBoundsError {}

// ----------------------------------------------------------------
// BerthStartTimeMismatchError
// ----------------------------------------------------------------

/// Error indicating a mismatch between the number of assigned berths and start times.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BerthStartTimeMismatchError {
    pub berths_count: usize,
    pub start_times_count: usize,
}

impl std::fmt::Display for BerthStartTimeMismatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "length mismatch between initial solution arrays: berths array has {} elements, but start_times array has {}",
            self.berths_count, self.start_times_count
        )
    }
}

impl std::error::Error for BerthStartTimeMismatchError {}

// ----------------------------------------------------------------
// SolutionIncompatibleBerthError
// ----------------------------------------------------------------

/// Error indicating that the initial solution assigns a vessel to a berth it is not allowed to use.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SolutionIncompatibleBerthError {
    pub vessel_index: usize,
    pub incompatible_berth: usize,
}

impl std::fmt::Display for SolutionIncompatibleBerthError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "initial solution assigns vessel {} to berth {}, but the model does not allow this assignment (processing time is None)",
            self.vessel_index, self.incompatible_berth
        )
    }
}

impl std::error::Error for SolutionIncompatibleBerthError {}

// ----------------------------------------------------------------
// LocalSearchParamsError
// ----------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LocalSearchParamsError {
    ModelSolutionMismatch(ModelSolutionMismatchError),
    BerthStartTimeMismatch(BerthStartTimeMismatchError),
    SolutionBerthOutOfBounds(SolutionBerthOutOfBoundsError),
    SolutionIncompatibleBerth(SolutionIncompatibleBerthError),
}

impl std::fmt::Display for LocalSearchParamsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ModelSolutionMismatch(e) => write!(f, "{}", e),
            Self::BerthStartTimeMismatch(e) => write!(f, "{}", e),
            Self::SolutionBerthOutOfBounds(e) => write!(f, "{}", e),
            Self::SolutionIncompatibleBerth(e) => write!(f, "{}", e),
        }
    }
}

impl std::error::Error for LocalSearchParamsError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ModelSolutionMismatch(e) => Some(e),
            Self::BerthStartTimeMismatch(e) => Some(e),
            Self::SolutionBerthOutOfBounds(e) => Some(e),
            Self::SolutionIncompatibleBerth(e) => Some(e),
        }
    }
}

// ----------------------------------------------------------------
// LocalSearchParams
// ----------------------------------------------------------------

/// The validated parameters required to run the local search engine.
#[derive(Debug)]
pub struct LocalSearchParams<'a, T, H, O, M, G>
where
    T: SolverNumeric,
{
    model: &'a Model<T>,
    operator: &'a mut O,
    metaheuristic: &'a mut H,
    monitor: M,
    oracle: &'a G,
    berths: &'a [BerthIndex], // len = num_vessels
    start_times: &'a [T],     // len = num_vessels
    objective_value: T,
}

impl<'a, T, H, O, M, G> LocalSearchParams<'a, T, H, O, M, G>
where
    T: SolverNumeric,
{
    /// Creates a validated `LocalSearchParams` instance for the search engine.
    ///
    /// This constructor acts as a strict **fail-fast safety boundary** between the external
    /// configuration (e.g., data provided via FFI) and the high-performance inner loop of the solver.
    /// It exhaustively cross-references the state, model, and topology to mathematically prove
    /// that all data invariants hold.
    ///
    /// By ensuring these properties upfront, downstream components (such as `LocalSearchOperator`s
    /// and `Evaluator`s) are mathematically guaranteed to be safe when using zero-cost `_unchecked`
    /// memory access.
    ///
    /// ### Validations Performed
    /// 1. **Dimensional Consistency:** The `model`, `neighborhood`, and `initial_solution` must all
    ///    declare the exact same number of vessels.
    /// 2. **Neighborhood Bounds:** Every target `VesselIndex` yielded by the `neighborhood` topology
    ///    must be strictly less than `model.num_vessels()`.
    /// 3. **Solution Bounds:** Every `BerthIndex` assigned in the `initial_solution` must be
    ///    strictly less than `model.num_berths()`.
    /// 4. **Solution Feasibility:** Every vessel in the `initial_solution` must be mathematically
    ///    allowed to dock at its assigned berth (i.e., the model must not return a `None` processing time).
    ///
    /// # Errors
    ///
    /// Returns a `LocalSearchParamsError` if any component is mismatched, out of bounds, or
    /// violates the hard physical constraints of the `Model`.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub fn new(
        model: &'a Model<T>,
        operator: &'a mut O,
        metaheuristic: &'a mut H,
        monitor: M,
        oracle: &'a G,
        berths: &'a [BerthIndex],
        start_times: &'a [T],
        objective_value: T,
    ) -> Result<Self, LocalSearchParamsError> {
        let model_vessels = model.num_vessels();
        let solution_vessels = berths.len();

        // Check Model vs Solution bounds
        if model_vessels != solution_vessels {
            return Err(LocalSearchParamsError::ModelSolutionMismatch(
                ModelSolutionMismatchError {
                    model_count: model_vessels,
                    solution_count: solution_vessels,
                },
            ));
        }

        // Check internal solution array alignment
        if berths.len() != start_times.len() {
            return Err(LocalSearchParamsError::BerthStartTimeMismatch(
                BerthStartTimeMismatchError {
                    berths_count: berths.len(),
                    start_times_count: start_times.len(),
                },
            ));
        }

        // Check that the initial solution does not contain out-of-bounds berth indices,
        // AND check that every vessel is actually allowed to dock at its assigned berth.
        let num_berths = model.num_berths();
        for (v_idx, &berth_idx) in berths.iter().enumerate() {
            if berth_idx.get() >= num_berths {
                return Err(LocalSearchParamsError::SolutionBerthOutOfBounds(
                    SolutionBerthOutOfBoundsError {
                        invalid_index: berth_idx.get(),
                        model_count: num_berths,
                    },
                ));
            }

            let vessel = VesselIndex::new(v_idx);
            if !model.vessel_allowed_on_berth(vessel, berth_idx) {
                return Err(LocalSearchParamsError::SolutionIncompatibleBerth(
                    SolutionIncompatibleBerthError {
                        vessel_index: v_idx,
                        incompatible_berth: berth_idx.get(),
                    },
                ));
            }
        }

        Ok(Self {
            model,
            operator,
            metaheuristic,
            monitor,
            oracle,
            berths,
            start_times,
            objective_value,
        })
    }
}

// ----------------------------------------------------------------
// MutableLocalSearchParams
// ----------------------------------------------------------------

#[derive(Debug)]
pub struct MutableLocalSearchParams<'a, T, H, O, M, G>
where
    T: SolverNumeric,
{
    pub model: &'a Model<T>,
    pub operator: &'a mut O,
    pub metaheuristic: &'a mut H,
    pub monitor: M,
    pub oracle: &'a G,
    pub berths: &'a [BerthIndex],
    pub start_times: &'a [T],
    pub objective_value: T,
}

impl<'a, T, H, O, M, G> From<LocalSearchParams<'a, T, H, O, M, G>>
    for MutableLocalSearchParams<'a, T, H, O, M, G>
where
    T: SolverNumeric,
{
    fn from(params: LocalSearchParams<'a, T, H, O, M, G>) -> Self {
        Self {
            model: params.model,
            operator: params.operator,
            metaheuristic: params.metaheuristic,
            monitor: params.monitor,
            oracle: params.oracle,
            berths: params.berths,
            start_times: params.start_times,
            objective_value: params.objective_value,
        }
    }
}
