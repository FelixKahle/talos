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

/// A penalty weight that never changes: finite and positive.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StaticLambda {
    pub(super) value: f64,
}

impl StaticLambda {
    /// The weight `value`.
    ///
    /// # Panics
    ///
    /// Panics if `value` is not finite and positive.
    pub fn new(value: f64) -> Self {
        match Self::try_new(value) {
            Some(lambda) => lambda,
            None => {
                panic!("called `StaticLambda::new` with {value}, which is not finite and positive")
            }
        }
    }

    /// The weight `value`, or `None` if it is not finite and positive.
    pub fn try_new(value: f64) -> Option<Self> {
        is_finite_positive(value).then_some(Self { value })
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
    /// Panics if `initial`, `increase` or `min` is not finite and positive, if `decrease` is not in `(0, 1)`, or
    /// if `max` is not finite or below `min`.
    pub fn new(initial: f64, increase: f64, decrease: f64, min: f64, max: f64) -> Self {
        match Self::try_new(initial, increase, decrease, min, max) {
            Some(lambda) => lambda,
            None => panic!(
                "called `DynamicLambda::new` with initial {initial}, increase {increase}, decrease {decrease}, min \
                 {min} and max {max}, not finite and positive initial, increase and min, a decrease in (0, 1) and \
                 a finite max of at least min"
            ),
        }
    }

    /// The weight of [`new`](Self::new), or `None` if `initial`, `increase` or `min` is not finite and positive,
    /// if `decrease` is not in `(0, 1)`, or if `max` is not finite or below `min`.
    pub fn try_new(initial: f64, increase: f64, decrease: f64, min: f64, max: f64) -> Option<Self> {
        let valid = is_finite_positive(initial)
            && is_finite_positive(increase)
            && decrease > 0.0
            && decrease < 1.0
            && is_finite_positive(min)
            && max.is_finite()
            && max >= min;
        valid.then_some(Self {
            initial,
            increase,
            decrease,
            min,
            max,
            reset_on_best: false,
        })
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
    /// Panics if `initial`, `increase`, `decrease` or `min` is not finite and positive, or if `max` is not finite
    /// or below `min`.
    pub fn new(initial: f64, increase: f64, decrease: f64, min: f64, max: f64) -> Self {
        match Self::try_new(initial, increase, decrease, min, max) {
            Some(lambda) => lambda,
            None => panic!(
                "called `AdditiveDynamicLambda::new` with initial {initial}, increase {increase}, decrease \
                 {decrease}, min {min} and max {max}, not finite and positive initial, increase, decrease and min \
                 and a finite max of at least min"
            ),
        }
    }

    /// The weight of [`new`](Self::new), or `None` if `initial`, `increase`, `decrease` or `min` is not finite and
    /// positive, or if `max` is not finite or below `min`.
    pub fn try_new(initial: f64, increase: f64, decrease: f64, min: f64, max: f64) -> Option<Self> {
        let valid = is_finite_positive(initial)
            && is_finite_positive(increase)
            && is_finite_positive(decrease)
            && is_finite_positive(min)
            && max.is_finite()
            && max >= min;
        valid.then_some(Self {
            initial,
            increase,
            decrease,
            min,
            max,
            reset_on_best: false,
        })
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
        match Self::try_new(factor, period) {
            Some(decay) => decay,
            None => panic!(
                "called `GeometricDecay::new` with factor {factor} and period {period}, not a factor in (0, 1) and \
                 a positive period"
            ),
        }
    }

    /// The decay of [`new`](Self::new), or `None` if `factor` is not in `(0, 1)` or `period` is zero.
    pub fn try_new(factor: f64, period: u64) -> Option<Self> {
        (factor > 0.0 && factor < 1.0 && period >= 1).then_some(Self { factor, period })
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

/// Whether `value` is a finite number above zero; NaN is not.
fn is_finite_positive(value: f64) -> bool {
    value.is_finite() && value > 0.0
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
    #[should_panic(
        expected = "called `StaticLambda::new` with 0, which is not finite and positive"
    )]
    fn test_static_lambda_refuses_zero() {
        let _ = StaticLambda::new(0.0);
    }

    #[test]
    #[should_panic(
        expected = "called `DynamicLambda::new` with initial 1, increase 0.1, decrease 1,"
    )]
    fn test_dynamic_lambda_refuses_a_decrease_of_one() {
        let _ = DynamicLambda::new(1.0, 0.1, 1.0, 0.5, 2.0);
    }

    #[test]
    #[should_panic(
        expected = "called `AdditiveDynamicLambda::new` with initial 1, increase 0.1, decrease 0.1, min 2 and max 1,"
    )]
    fn test_additive_dynamic_lambda_refuses_max_below_min() {
        let _ = AdditiveDynamicLambda::new(1.0, 0.1, 0.1, 2.0, 1.0);
    }

    #[test]
    #[should_panic(expected = "called `GeometricDecay::new` with factor 1 and period 1,")]
    fn test_geometric_decay_refuses_a_factor_of_one() {
        let _ = GeometricDecay::new(1.0, 1);
    }

    // An infinite weight makes `lambda * 0` of an unpenalized move NaN, and a run that compares NaN rejects every
    // candidate: every constructor must refuse a value that is not finite.
    #[test]
    #[should_panic(expected = "called `StaticLambda::new` with inf")]
    fn test_static_lambda_refuses_infinity() {
        let _ = StaticLambda::new(f64::INFINITY);
    }

    #[test]
    #[should_panic(expected = "called `DynamicLambda::new` with")]
    fn test_dynamic_lambda_refuses_an_infinite_bound() {
        let _ = DynamicLambda::new(1.0, 0.1, 0.5, 0.5, f64::INFINITY);
    }

    #[test]
    #[should_panic(expected = "called `DynamicLambda::new` with")]
    fn test_dynamic_lambda_refuses_an_infinite_increase() {
        let _ = DynamicLambda::new(1.0, f64::INFINITY, 0.5, 0.5, 2.0);
    }

    #[test]
    #[should_panic(expected = "called `AdditiveDynamicLambda::new` with")]
    fn test_additive_dynamic_lambda_refuses_an_infinite_bound() {
        let _ = AdditiveDynamicLambda::new(1.0, 0.1, 0.1, 0.5, f64::INFINITY);
    }

    #[test]
    #[should_panic(expected = "called `AdditiveDynamicLambda::new` with")]
    fn test_additive_dynamic_lambda_refuses_an_infinite_increase() {
        let _ = AdditiveDynamicLambda::new(1.0, f64::INFINITY, 0.1, 0.5, 2.0);
    }

    #[test]
    fn test_try_new_refuses_what_new_panics_on_and_keeps_the_rest() {
        let (infinity, nan) = (f64::INFINITY, f64::NAN);
        assert_eq!(StaticLambda::try_new(2.0), Some(StaticLambda::new(2.0)));
        assert!(
            [0.0, -1.0, infinity, nan]
                .iter()
                .all(|&value| StaticLambda::try_new(value).is_none())
        );
        assert!(DynamicLambda::try_new(1.0, 0.1, 0.5, 0.5, 2.0).is_some());
        for (initial, increase, decrease, min, max) in [
            (infinity, 0.1, 0.5, 0.5, 2.0),
            (nan, 0.1, 0.5, 0.5, 2.0),
            (1.0, 0.0, 0.5, 0.5, 2.0),
            (1.0, 0.1, 1.0, 0.5, 2.0),
            (1.0, 0.1, nan, 0.5, 2.0),
            (1.0, 0.1, 0.5, 0.0, 2.0),
            (1.0, 0.1, 0.5, 0.5, nan),
            (1.0, 0.1, 0.5, 3.0, 2.0),
        ] {
            assert_eq!(
                DynamicLambda::try_new(initial, increase, decrease, min, max),
                None
            );
        }
        assert!(AdditiveDynamicLambda::try_new(1.0, 0.1, 2.0, 0.5, 2.0).is_some());
        for (initial, increase, decrease, min, max) in [
            (1.0, 0.1, infinity, 0.5, 2.0),
            (1.0, nan, 0.1, 0.5, 2.0),
            (1.0, 0.1, 0.0, 0.5, 2.0),
            (1.0, 0.1, 0.1, 0.5, infinity),
            (1.0, 0.1, 0.1, 3.0, 2.0),
        ] {
            assert_eq!(
                AdditiveDynamicLambda::try_new(initial, increase, decrease, min, max),
                None
            );
        }
        assert!(GeometricDecay::try_new(0.5, 1).is_some());
        assert_eq!(GeometricDecay::try_new(0.5, 0), None);
        assert_eq!(GeometricDecay::try_new(nan, 1), None);
        assert_eq!(GeometricDecay::try_new(infinity, 1), None);
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
