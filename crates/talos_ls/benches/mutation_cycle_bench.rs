// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use talos_core::ClosedOpenInterval;
use talos_ls::{
    decoding::{ScheduleState, decode_unchecked},
    objective::VesselCost,
    schedule::{Mutator, ScheduleGraph},
};
use talos_model::{BerthIndex, Model, VesselIndex};

/// The start of a vessel's service as its cost.
struct StartTimeCost;

impl VesselCost<i64> for StartTimeCost {
    unsafe fn cost_unchecked(
        _: &Model<i64>,
        _: VesselIndex,
        _: BerthIndex,
        start: i64,
    ) -> Option<i64> {
        Some(start)
    }
}

fn build_mock_model(num_vessels: usize, num_berths: usize) -> Model<i64> {
    let mut arrivals = Vec::with_capacity(num_vessels);
    let mut departures = Vec::with_capacity(num_vessels);
    let mut weights = Vec::with_capacity(num_vessels);
    let mut processing = Vec::with_capacity(num_vessels * num_berths);

    for v in 0..num_vessels {
        let arrival = (v * 10) as i64;
        arrivals.push(arrival);
        departures.push(arrival + 1000);
        weights.push(1);

        for _ in 0..num_berths {
            processing.push(Some(50));
        }
    }

    let mut opening = Vec::with_capacity(num_berths);
    for _ in 0..num_berths {
        opening.push(vec![ClosedOpenInterval::new(0, i64::MAX)]);
    }

    Model::new(arrivals, departures, weights, processing, opening)
}

fn setup_scenario(
    worst_case: bool,
) -> (
    ScheduleGraph,
    ScheduleState<i64>,
    ScheduleState<i64>,
    Model<i64>,
) {
    let num_vessels = 250;
    let num_berths = 20;

    let model = build_mock_model(num_vessels, num_berths);

    let mut berths = Vec::with_capacity(num_vessels);
    let mut starts = Vec::with_capacity(num_vessels);

    for v in 0..num_vessels {
        let b = if worst_case { 0 } else { (v / 50).min(4) };
        berths.push(BerthIndex::new(b));
        starts.push((v * 10) as i64);
    }

    let graph = ScheduleGraph::from_slices(&berths, &starts, num_berths);

    let state = ScheduleState::new(berths, starts.clone(), vec![0; num_berths], 0);
    let accepted = state.clone();

    (graph, state, accepted, model)
}

fn bench_mutation_cycle(c: &mut Criterion) {
    let mut group = c.benchmark_group("Mutation Cycle (Mutate -> Diff -> Decode -> Rollback)");

    let scenarios = [
        ("Average Case (50/berth)", false),
        ("Worst Case (250/berth)", true),
    ];

    for (name, is_worst_case) in scenarios.iter() {
        let (graph, mut candidate, accepted, model) = setup_scenario(*is_worst_case);

        let mut mutator = Mutator::new(graph);

        group.bench_with_input(
            BenchmarkId::new("Swap Intra-Berth", name),
            &is_worst_case,
            |b, _| {
                b.iter(|| {
                    mutator.clear_diff();
                    mutator.swap_vessels(VesselIndex::new(10), VesselIndex::new(40));

                    unsafe {
                        decode_unchecked::<_, StartTimeCost>(
                            mutator.touched(),
                            mutator.graph(),
                            &mut candidate,
                            &accepted,
                            &model,
                        )
                    };

                    mutator.rollback();
                });
            },
        );

        if !is_worst_case {
            group.bench_with_input(
                BenchmarkId::new("Swap Inter-Berth", name),
                &is_worst_case,
                |b, _| {
                    b.iter(|| {
                        mutator.clear_diff();
                        mutator.swap_vessels(VesselIndex::new(10), VesselIndex::new(60));

                        unsafe {
                            decode_unchecked::<_, StartTimeCost>(
                                mutator.touched(),
                                mutator.graph(),
                                &mut candidate,
                                &accepted,
                                &model,
                            )
                        };
                        mutator.rollback();
                    });
                },
            );
        }

        group.bench_with_input(
            BenchmarkId::new("Shift Intra-Berth", name),
            &is_worst_case,
            |b, _| {
                b.iter(|| {
                    mutator.clear_diff();
                    mutator.relocate_after(VesselIndex::new(10), VesselIndex::new(40));

                    unsafe {
                        decode_unchecked::<_, StartTimeCost>(
                            mutator.touched(),
                            mutator.graph(),
                            &mut candidate,
                            &accepted,
                            &model,
                        )
                    };
                    mutator.rollback();
                });
            },
        );

        if !is_worst_case {
            group.bench_with_input(
                BenchmarkId::new("Shift Inter-Berth", name),
                &is_worst_case,
                |b, _| {
                    b.iter(|| {
                        mutator.clear_diff();
                        mutator.relocate_after(VesselIndex::new(10), VesselIndex::new(60));

                        unsafe {
                            decode_unchecked::<_, StartTimeCost>(
                                mutator.touched(),
                                mutator.graph(),
                                &mut candidate,
                                &accepted,
                                &model,
                            )
                        };
                        mutator.rollback();
                    });
                },
            );
        }

        group.bench_with_input(
            BenchmarkId::new("Reverse Segment (5 vessels)", name),
            &is_worst_case,
            |b, _| {
                b.iter(|| {
                    mutator.clear_diff();
                    mutator.reverse_segment(VesselIndex::new(10), VesselIndex::new(15));

                    unsafe {
                        decode_unchecked::<_, StartTimeCost>(
                            mutator.touched(),
                            mutator.graph(),
                            &mut candidate,
                            &accepted,
                            &model,
                        )
                    };
                    mutator.rollback();
                });
            },
        );
    }

    group.finish();
}

fn bench_single_mutation_cycle(c: &mut Criterion) {
    let mut group =
        c.benchmark_group("Single Mutation Cycle (Mutate -> Diff -> Decode -> Rollback)");

    let scenarios = [
        ("Average Case (50/berth)", false),
        ("Worst Case (250/berth)", true),
    ];

    for (name, is_worst_case) in scenarios.iter() {
        let (graph, mut candidate, accepted, model) = setup_scenario(*is_worst_case);

        let mut mutator = Mutator::new(graph);

        group.bench_with_input(
            BenchmarkId::new("Swap Vessels", name),
            &is_worst_case,
            |b, _| {
                b.iter(|| {
                    mutator.clear_diff();
                    mutator.swap_vessels(VesselIndex::new(10), VesselIndex::new(40));

                    unsafe {
                        decode_unchecked::<_, StartTimeCost>(
                            mutator.touched(),
                            mutator.graph(),
                            &mut candidate,
                            &accepted,
                            &model,
                        )
                    };

                    black_box(candidate.objective());
                    mutator.rollback();
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_mutation_cycle, bench_single_mutation_cycle);
criterion_main!(benches);
