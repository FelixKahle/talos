// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! How the penalty weight and the penalties of a guided local search evolve over a run.

use super::{
    penalty_memory::PenaltyMemory,
    settings::{AtNewBest, Lambda, PenaltyDecay},
};

/// The penalty weight of a run as it evolves under its [`Lambda`].
#[derive(Debug, Clone, Copy)]
pub(super) struct LambdaSchedule {
    rule: Lambda,
    pub(super) current: f64,
}

impl LambdaSchedule {
    /// The schedule of `rule`, at its start.
    pub(super) fn new(rule: Lambda) -> Self {
        let mut schedule = Self { rule, current: 0.0 };
        schedule.restart();
        schedule
    }

    /// Sets the weight to its start.
    pub(super) fn restart(&mut self) {
        self.current = match self.rule {
            Lambda::Static(lambda) => lambda.value,
            Lambda::Dynamic(lambda) => lambda.initial.clamp(lambda.min, lambda.max),
            Lambda::AdditiveDynamic(lambda) => lambda.initial.clamp(lambda.min, lambda.max),
        };
    }

    /// Lowers the weight at a new best, or sets it to its start where the rule says so.
    pub(super) fn on_new_best(&mut self) {
        match self.rule {
            Lambda::Static(_) => {}
            Lambda::Dynamic(lambda) => match lambda.at_new_best {
                AtNewBest::Decrease(decrease) => {
                    self.current = (self.current * (1.0 - decrease)).clamp(lambda.min, lambda.max);
                }
                AtNewBest::Restart => self.restart(),
            },
            Lambda::AdditiveDynamic(lambda) => match lambda.at_new_best {
                AtNewBest::Decrease(decrease) => {
                    self.current = (self.current - decrease).clamp(lambda.min, lambda.max);
                }
                AtNewBest::Restart => self.restart(),
            },
        }
    }

    /// Raises the weight after a penalization.
    pub(super) fn on_penalization(&mut self) {
        match self.rule {
            Lambda::Static(_) => {}
            Lambda::Dynamic(lambda) => {
                self.current =
                    (self.current * (1.0 + lambda.increase)).clamp(lambda.min, lambda.max);
            }
            Lambda::AdditiveDynamic(lambda) => {
                self.current = (self.current + lambda.increase).clamp(lambda.min, lambda.max);
            }
        }
    }
}

/// The decay of the penalties of a run, with the penalizations counted since it last fired.
#[derive(Debug, Clone, Copy)]
pub(super) struct DecaySchedule {
    pub(super) rule: PenaltyDecay,
    pub(super) penalizations: u64,
}

impl DecaySchedule {
    /// Counts a penalization, and decays `memory` at every period of the rule.
    pub(super) fn after_penalization(&mut self, memory: &mut PenaltyMemory) {
        if let PenaltyDecay::Geometric(decay) = self.rule {
            self.penalizations += 1;
            if self.penalizations >= decay.period {
                self.penalizations = 0;
                memory.decay(decay.factor);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        metaheuristic::guided_local_search::settings::{
            AdditiveDynamicLambda, DynamicLambda, GeometricDecay, StaticLambda,
        },
        schedule::LinkEnd,
    };
    use talos_model::VesselIndex;

    /// The link of V0 to itself, a feature of a memory of one vessel.
    fn self_link() -> LinkEnd {
        LinkEnd::Vessel(VesselIndex::new(0))
    }

    /// The memory of `num_vessels` vessels and `num_berths` berths, every count zero.
    fn memory(num_vessels: usize, num_berths: usize) -> PenaltyMemory {
        let mut memory = PenaltyMemory::default();
        memory.reset(num_vessels, num_berths);
        memory
    }

    #[test]
    fn test_static_lambda_never_changes() {
        let mut schedule = LambdaSchedule::new(Lambda::Static(StaticLambda::new(2.5)));
        schedule.on_penalization();
        schedule.on_new_best();
        assert_eq!(schedule.current, 2.5);
    }

    #[test]
    fn test_dynamic_lambda_multiplies_within_its_bounds() {
        let mut schedule = LambdaSchedule::new(Lambda::Dynamic(DynamicLambda::new(
            1.0,
            0.5,
            AtNewBest::Decrease(0.5),
            0.4,
            2.0,
        )));
        schedule.on_penalization();
        assert_eq!(schedule.current, 1.5);
        schedule.on_penalization();
        assert_eq!(schedule.current, 2.0);
        schedule.on_new_best();
        assert_eq!(schedule.current, 1.0);
        schedule.on_new_best();
        schedule.on_new_best();
        assert_eq!(schedule.current, 0.4);
        schedule.restart();
        assert_eq!(schedule.current, 1.0);
    }

    #[test]
    fn test_dynamic_lambda_starts_clamped_and_resets_on_best_where_asked() {
        let lambda = DynamicLambda::new(5.0, 0.5, AtNewBest::Restart, 0.4, 2.0);
        let mut schedule = LambdaSchedule::new(Lambda::Dynamic(lambda));
        assert_eq!(schedule.current, 2.0);
        schedule.current = 0.5;
        schedule.on_new_best();
        assert_eq!(schedule.current, 2.0);
    }

    #[test]
    fn test_additive_dynamic_lambda_steps_within_its_bounds() {
        let mut schedule = LambdaSchedule::new(Lambda::AdditiveDynamic(
            AdditiveDynamicLambda::new(1.0, 0.25, AtNewBest::Decrease(0.5), 0.5, 1.5),
        ));
        schedule.on_penalization();
        assert_eq!(schedule.current, 1.25);
        schedule.on_penalization();
        schedule.on_penalization();
        assert_eq!(schedule.current, 1.5);
        schedule.on_new_best();
        schedule.on_new_best();
        schedule.on_new_best();
        assert_eq!(schedule.current, 0.5);
    }

    #[test]
    fn test_geometric_decay_fires_at_every_period() {
        let mut memory = memory(1, 1);
        memory.set_link_penalty(self_link(), self_link(), 100);
        let mut decay = DecaySchedule {
            rule: PenaltyDecay::Geometric(GeometricDecay::new(0.5, 3)),
            penalizations: 0,
        };
        decay.after_penalization(&mut memory);
        decay.after_penalization(&mut memory);
        assert_eq!(memory.link_penalty(self_link(), self_link()), 100);
        decay.after_penalization(&mut memory);
        assert_eq!(memory.link_penalty(self_link(), self_link()), 50);
        decay.after_penalization(&mut memory);
        decay.after_penalization(&mut memory);
        decay.after_penalization(&mut memory);
        assert_eq!(memory.link_penalty(self_link(), self_link()), 25);
    }

    #[test]
    fn test_no_decay_keeps_the_counts() {
        let mut memory = memory(1, 1);
        memory.set_link_penalty(self_link(), self_link(), 100);
        let mut decay = DecaySchedule {
            rule: PenaltyDecay::None,
            penalizations: 0,
        };
        decay.after_penalization(&mut memory);
        assert_eq!(memory.link_penalty(self_link(), self_link()), 100);
    }
}
