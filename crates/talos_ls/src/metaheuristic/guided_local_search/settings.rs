// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The settings of a guided local search: the penalty weight and how it evolves, the decay of the penalties, and
//! the trigger of a penalization.

/// The settings of a guided local search: how the penalty weight evolves, how penalties decay, and when the
/// search penalizes.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct GuidedLocalSearch {
    /// The weight of the penalties in the augmented objective, and how it evolves.
    pub lambda: Lambda,
    /// How the penalties erode over the run.
    pub decay: PenaltyDecay,
    /// When the search penalizes the features of the accepted schedule.
    pub trigger: PenalizationTrigger,
}

/// The weight `lambda` of the penalties in the augmented objective, and how it evolves over a run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Lambda {
    /// A weight that never changes.
    Static(StaticLambda),
    /// A weight multiplied up at each penalization and down at each new best.
    Dynamic(DynamicLambda),
    /// A weight stepped up at each penalization and down at each new best.
    AdditiveDynamic(AdditiveDynamicLambda),
}

impl Default for Lambda {
    /// A static weight of one.
    fn default() -> Self {
        Self::Static(StaticLambda::new(1.0))
    }
}

/// A penalty weight that never changes: positive.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StaticLambda {
    pub(super) value: f64,
}

impl StaticLambda {
    /// The weight `value`.
    ///
    /// # Panics
    ///
    /// Panics if `value` is not positive.
    pub fn new(value: f64) -> Self {
        assert!(
            value > 0.0,
            "called `StaticLambda::new` with {value}, which is not positive"
        );
        Self { value }
    }

    /// The weight.
    pub fn value(self) -> f64 {
        self.value
    }
}

/// A penalty weight that grows by a factor at each penalization and shrinks by a factor at each new best, kept
/// within its bounds.
///
/// The weight starts at `initial` clamped to `[min, max]`; a penalization multiplies it by `1 + increase`, and a
/// new best multiplies it by `1 - decrease` or, with `reset_on_best`, sets it back to its start.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DynamicLambda {
    pub(super) initial: f64,
    pub(super) increase: f64,
    pub(super) decrease: f64,
    pub(super) min: f64,
    pub(super) max: f64,
    pub(super) reset_on_best: bool,
}

impl DynamicLambda {
    /// The weight that starts at `initial` and is kept in `[min, max]`, grows by the factor `1 + increase` at each
    /// penalization and shrinks by the factor `1 - decrease` at each new best.
    ///
    /// # Panics
    ///
    /// Panics if `initial`, `increase` or `min` is not positive, if `decrease` is not in `(0, 1)`, or if `max` is
    /// below `min`.
    pub fn new(initial: f64, increase: f64, decrease: f64, min: f64, max: f64) -> Self {
        assert!(
            initial > 0.0 && increase > 0.0 && min > 0.0,
            "called `DynamicLambda::new` with initial {initial}, increase {increase} or min {min} not positive"
        );
        assert!(
            decrease > 0.0 && decrease < 1.0,
            "called `DynamicLambda::new` with decrease {decrease}, outside (0, 1)"
        );
        assert!(
            max >= min,
            "called `DynamicLambda::new` with max {max} below min {min}"
        );
        Self {
            initial,
            increase,
            decrease,
            min,
            max,
            reset_on_best: false,
        }
    }

    /// The same weight, which at a new best is set back to its start if `reset` holds, or shrinks otherwise.
    pub fn with_reset_on_best(mut self, reset: bool) -> Self {
        self.reset_on_best = reset;
        self
    }
}

/// A penalty weight that grows by a step at each penalization and shrinks by a step at each new best, kept within
/// its bounds.
///
/// The weight starts at `initial` clamped to `[min, max]`; a penalization adds `increase`, and a new best subtracts
/// `decrease` or, with `reset_on_best`, sets it back to its start.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdditiveDynamicLambda {
    pub(super) initial: f64,
    pub(super) increase: f64,
    pub(super) decrease: f64,
    pub(super) min: f64,
    pub(super) max: f64,
    pub(super) reset_on_best: bool,
}

impl AdditiveDynamicLambda {
    /// The weight that starts at `initial` and is kept in `[min, max]`, grows by `increase` at each penalization
    /// and shrinks by `decrease` at each new best.
    ///
    /// # Panics
    ///
    /// Panics if `initial`, `increase`, `decrease` or `min` is not positive, or if `max` is below `min`.
    pub fn new(initial: f64, increase: f64, decrease: f64, min: f64, max: f64) -> Self {
        assert!(
            initial > 0.0 && increase > 0.0 && decrease > 0.0 && min > 0.0,
            "called `AdditiveDynamicLambda::new` with initial {initial}, increase {increase}, decrease {decrease} \
             or min {min} not positive"
        );
        assert!(
            max >= min,
            "called `AdditiveDynamicLambda::new` with max {max} below min {min}"
        );
        Self {
            initial,
            increase,
            decrease,
            min,
            max,
            reset_on_best: false,
        }
    }

    /// The same weight, which at a new best is set back to its start if `reset` holds, or shrinks otherwise.
    pub fn with_reset_on_best(mut self, reset: bool) -> Self {
        self.reset_on_best = reset;
        self
    }
}

/// How the penalties erode over a run.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum PenaltyDecay {
    /// The penalties never erode.
    #[default]
    None,
    /// Every penalty is scaled down by a factor at every so many penalizations.
    Geometric(GeometricDecay),
}

/// A decay that multiplies every penalty by `factor`, rounding toward zero, at every `period`-th penalization.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeometricDecay {
    pub(super) factor: f64,
    pub(super) period: u64,
}

impl GeometricDecay {
    /// The decay by `factor` at every `period`-th penalization.
    ///
    /// # Panics
    ///
    /// Panics if `factor` is not in `(0, 1)` or `period` is zero.
    pub fn new(factor: f64, period: u64) -> Self {
        assert!(
            factor > 0.0 && factor < 1.0,
            "called `GeometricDecay::new` with factor {factor}, outside (0, 1)"
        );
        assert!(
            period >= 1,
            "called `GeometricDecay::new` with a period of zero"
        );
        Self { factor, period }
    }
}

/// When the search penalizes the features of the accepted schedule.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum PenalizationTrigger {
    /// When a scan finds no move that lowers the augmented objective: the classic guided local search.
    #[default]
    OnExhaustion,
    /// After this many steps in a row without a new best, each candidate and each exhausted scan a step.
    AfterNonImprovements(u64),
    /// After every this many accepted moves.
    AfterMoves(u64),
}

impl std::fmt::Display for PenalizationTrigger {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OnExhaustion => write!(f, "on exhaustion"),
            Self::AfterNonImprovements(steps) => write!(f, "after {steps} non-improving steps"),
            Self::AfterMoves(moves) => write!(f, "after {moves} accepted moves"),
        }
    }
}

/// A penalty weight from the objective of the initial schedule: `alpha * |objective| / features`, at least the
/// machine epsilon so that the penalties always count.
///
/// `features` is the number of features penalties may fall on, at least one; `alpha` is the share of the
/// objective, typically 0.1 to 0.5, that one penalty of every feature would add.
///
/// ```
/// # use talos_ls::heuristic_lambda;
/// let lambda = heuristic_lambda(1000.0, 50, 0.3);
/// assert!((lambda - 6.0).abs() < f64::EPSILON);
/// ```
#[expect(
    clippy::as_conversions,
    reason = "a count of features stays far below 2^53, where every count is an exact `f64`"
)]
pub fn heuristic_lambda(objective: f64, features: usize, alpha: f64) -> f64 {
    let features = features.max(1) as f64;
    (alpha * objective.abs() / features).max(f64::EPSILON)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heuristic_lambda_shares_the_objective_among_the_features() {
        assert!((heuristic_lambda(1000.0, 50, 0.3) - 6.0).abs() < 1e-12);
        assert!((heuristic_lambda(-1000.0, 0, 0.3) - 300.0).abs() < 1e-12);
        assert_eq!(heuristic_lambda(0.0, 10, 0.3), f64::EPSILON);
    }

    #[test]
    #[should_panic(expected = "called `StaticLambda::new` with 0, which is not positive")]
    fn test_static_lambda_refuses_zero() {
        let _ = StaticLambda::new(0.0);
    }

    #[test]
    #[should_panic(expected = "called `DynamicLambda::new` with decrease 1, outside (0, 1)")]
    fn test_dynamic_lambda_refuses_a_decrease_of_one() {
        let _ = DynamicLambda::new(1.0, 0.1, 1.0, 0.5, 2.0);
    }

    #[test]
    #[should_panic(expected = "called `AdditiveDynamicLambda::new` with max 1 below min 2")]
    fn test_additive_dynamic_lambda_refuses_max_below_min() {
        let _ = AdditiveDynamicLambda::new(1.0, 0.1, 0.1, 2.0, 1.0);
    }

    #[test]
    #[should_panic(expected = "called `GeometricDecay::new` with factor 1, outside (0, 1)")]
    fn test_geometric_decay_refuses_a_factor_of_one() {
        let _ = GeometricDecay::new(1.0, 1);
    }

    #[test]
    fn test_trigger_display_names_the_rule() {
        assert_eq!(
            PenalizationTrigger::OnExhaustion.to_string(),
            "on exhaustion"
        );
        assert_eq!(
            PenalizationTrigger::AfterNonImprovements(5).to_string(),
            "after 5 non-improving steps"
        );
        assert_eq!(
            PenalizationTrigger::AfterMoves(3).to_string(),
            "after 3 accepted moves"
        );
    }
}
