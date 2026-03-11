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

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use talos_core::math::interval::ClosedOpenInterval;
use talos_ls::{
    decoding::decode_unchecked,
    mutator::Mutator,
    sgraph::{ScheduleGraph, ScheduleGraphDiff},
    sgraphundo::ScheduleGraphUndoLog,
    state::ScheduleState,
    tberth::TouchedBerths,
};
use talos_model::{
    index::{BerthIndex, VesselIndex},
    model::{Model, ProcessingTime},
};

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
            processing.push(ProcessingTime::some(50));
        }
    }

    let mut opening = Vec::with_capacity(num_berths);
    for _ in 0..num_berths {
        opening.push(vec![ClosedOpenInterval::new(0, i64::MAX)]);
    }

    Model::new(
        num_vessels,
        num_berths,
        arrivals,
        departures,
        weights,
        processing,
        opening,
    )
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

    let state = ScheduleState::new(
        berths,
        starts.clone(),
        (0..num_vessels).collect(),
        vec![0; num_berths],
        0,
    );
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
        let (mut graph, mut candidate, accepted, model) = setup_scenario(*is_worst_case);

        let mut undo = ScheduleGraphUndoLog::preallocated(250);
        let mut diff = ScheduleGraphDiff::new(250);
        let mut touched = TouchedBerths::new(20);
        let evaluator = |_: &Model<i64>, _: VesselIndex, _: BerthIndex, start: i64| Some(start);

        group.bench_with_input(
            BenchmarkId::new("Swap Intra-Berth", name),
            &is_worst_case,
            |b, _| {
                b.iter(|| {
                    undo.clear();
                    diff.clear();
                    touched.reset();

                    let mut m = Mutator::new(&mut graph, &mut undo, &mut diff, &mut touched);
                    m.swap_vessels(VesselIndex::new(10), VesselIndex::new(40));

                    unsafe {
                        decode_unchecked(
                            &touched,
                            &graph,
                            &mut candidate,
                            &accepted,
                            &model,
                            evaluator,
                        )
                    };

                    undo.apply_rollback(&mut graph);
                });
            },
        );

        if !is_worst_case {
            group.bench_with_input(
                BenchmarkId::new("Swap Inter-Berth", name),
                &is_worst_case,
                |b, _| {
                    b.iter(|| {
                        undo.clear();
                        diff.clear();
                        touched.reset();

                        let mut m = Mutator::new(&mut graph, &mut undo, &mut diff, &mut touched);
                        m.swap_vessels(VesselIndex::new(10), VesselIndex::new(60));

                        unsafe {
                            decode_unchecked(
                                &touched,
                                &graph,
                                &mut candidate,
                                &accepted,
                                &model,
                                evaluator,
                            )
                        };
                        undo.apply_rollback(&mut graph);
                    });
                },
            );
        }

        group.bench_with_input(
            BenchmarkId::new("Shift Intra-Berth", name),
            &is_worst_case,
            |b, _| {
                b.iter(|| {
                    undo.clear();
                    diff.clear();
                    touched.reset();

                    let mut m = Mutator::new(&mut graph, &mut undo, &mut diff, &mut touched);
                    m.relocate_after(VesselIndex::new(10), VesselIndex::new(40));

                    unsafe {
                        decode_unchecked(
                            &touched,
                            &graph,
                            &mut candidate,
                            &accepted,
                            &model,
                            evaluator,
                        )
                    };
                    undo.apply_rollback(&mut graph);
                });
            },
        );

        if !is_worst_case {
            group.bench_with_input(
                BenchmarkId::new("Shift Inter-Berth", name),
                &is_worst_case,
                |b, _| {
                    b.iter(|| {
                        undo.clear();
                        diff.clear();
                        touched.reset();

                        let mut m = Mutator::new(&mut graph, &mut undo, &mut diff, &mut touched);
                        m.relocate_after(VesselIndex::new(10), VesselIndex::new(60));

                        unsafe {
                            decode_unchecked(
                                &touched,
                                &graph,
                                &mut candidate,
                                &accepted,
                                &model,
                                evaluator,
                            )
                        };
                        undo.apply_rollback(&mut graph);
                    });
                },
            );
        }

        group.bench_with_input(
            BenchmarkId::new("Reverse Segment (5 vessels)", name),
            &is_worst_case,
            |b, _| {
                b.iter(|| {
                    undo.clear();
                    diff.clear();
                    touched.reset();

                    let mut m = Mutator::new(&mut graph, &mut undo, &mut diff, &mut touched);
                    m.reverse_segment(VesselIndex::new(10), VesselIndex::new(15));

                    unsafe {
                        decode_unchecked(
                            &touched,
                            &graph,
                            &mut candidate,
                            &accepted,
                            &model,
                            evaluator,
                        )
                    };
                    undo.apply_rollback(&mut graph);
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
        let (mut graph, mut candidate, accepted, model) = setup_scenario(*is_worst_case);

        let mut undo = ScheduleGraphUndoLog::preallocated(250);
        let mut diff = ScheduleGraphDiff::new(250);
        let mut touched = TouchedBerths::new(20);
        let evaluator = |_: &Model<i64>, _: VesselIndex, _: BerthIndex, start: i64| Some(start);

        group.bench_with_input(
            BenchmarkId::new("Swap Vessels", name),
            &is_worst_case,
            |b, _| {
                b.iter(|| {
                    undo.clear();
                    diff.clear();
                    touched.reset();

                    let mut m = Mutator::new(&mut graph, &mut undo, &mut diff, &mut touched);
                    m.swap_vessels(VesselIndex::new(10), VesselIndex::new(40));

                    unsafe {
                        decode_unchecked(
                            &touched,
                            &graph,
                            &mut candidate,
                            &accepted,
                            &model,
                            evaluator,
                        )
                    };

                    black_box(candidate.objective());
                    undo.apply_rollback(&mut graph);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_mutation_cycle, bench_single_mutation_cycle);
criterion_main!(benches);
