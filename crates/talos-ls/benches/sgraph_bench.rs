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

//! Criterion benchmarks for [`ScheduleGraph`].
//!
//! Each mutation and query is benchmarked under two regimes:
//!
//! 1. **Typical** — 250 vessels across 20 berths, small neighborhood moves
//!    (single vessels, 2–3 element segments) that represent the average iteration
//!    of a Simulated Annealing / ALNS local search.
//!
//! 2. **Worst-case** — Same dimensions, but operating on the largest possible
//!    segments (full berth contents, cross-berth bulk relocations) to measure
//!    the upper bound of each operation.
//!
//! A minimal, seedable linear congruential generator (LCG) is used instead of
//! an external randomness crate so the benchmark has zero non-criterion dependencies.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use talos_ls::sgraph::ScheduleGraph;
use talos_model::index::{BerthIndex, VesselIndex};

const NUM_VESSELS: usize = 250;
const NUM_BERTHS: usize = 20;

struct Lcg {
    state: u64,
}

impl Lcg {
    #[inline]
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_mul(6364136223846793005).wrapping_add(1),
        }
    }

    /// Returns a value in `[0, bound)`.
    #[inline]
    fn next_bounded(&mut self, bound: usize) -> usize {
        // xorshift64
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        (self.state % bound as u64) as usize
    }
}

fn build_fixture() -> ScheduleGraph {
    let mut berths = Vec::with_capacity(NUM_VESSELS);
    let mut starts = Vec::with_capacity(NUM_VESSELS);

    for v in 0..NUM_VESSELS {
        let b = v % NUM_BERTHS;
        berths.push(BerthIndex::new(b));
        // Monotonically increasing per-berth start time so that
        // overwrite_from_slices preserves insertion order.
        starts.push(v as i64);
    }

    ScheduleGraph::from_slices(&berths, &starts, NUM_BERTHS)
}

/// Returns a pre-computed list of `(vessel, berth)` pairs for every vessel,
/// by iterating each berth.  Useful for picking operands by sequence position.
fn berth_sequences(graph: &ScheduleGraph) -> Vec<Vec<VesselIndex>> {
    (0..NUM_BERTHS)
        .map(|b| {
            graph
                .vessel_sequence_iter(BerthIndex::new(b))
                .collect::<Vec<_>>()
        })
        .collect()
}

fn intra_berth_pair(seqs: &[Vec<VesselIndex>]) -> (VesselIndex, VesselIndex) {
    // Berth 0 has ≥12 vessels in the even distribution.
    (seqs[0][0], seqs[0][1])
}

fn inter_berth_pair(seqs: &[Vec<VesselIndex>]) -> (VesselIndex, VesselIndex) {
    (seqs[0][0], seqs[1][0])
}

fn segment_in_berth(seqs: &[Vec<VesselIndex>], len: usize) -> (VesselIndex, VesselIndex) {
    let seq = &seqs[0];
    (seq[0], seq[len - 1])
}

fn full_berth_segment(seqs: &[Vec<VesselIndex>]) -> (VesselIndex, VesselIndex) {
    let seq = &seqs[0];
    (seq[0], *seq.last().unwrap())
}

fn bench_iteration(c: &mut Criterion) {
    let graph = build_fixture();
    let mut group = c.benchmark_group("iteration");

    group.bench_function("forward/full_berth", |b| {
        b.iter(|| {
            let mut sum = 0usize;
            for v in graph.vessel_sequence_iter(BerthIndex::new(0)) {
                sum = sum.wrapping_add(v.get());
            }
            black_box(sum);
        });
    });

    group.bench_function("reverse/full_berth", |b| {
        b.iter(|| {
            let mut sum = 0usize;
            for v in graph.vessel_sequence_rev_iter(BerthIndex::new(0)) {
                sum = sum.wrapping_add(v.get());
            }
            black_box(sum);
        });
    });

    group.bench_function("forward/all_berths", |b| {
        b.iter(|| {
            let mut sum = 0usize;
            for berth_idx in 0..NUM_BERTHS {
                for v in graph.vessel_sequence_iter(BerthIndex::new(berth_idx)) {
                    sum = sum.wrapping_add(v.get());
                }
            }
            black_box(sum);
        });
    });

    group.finish();
}

fn bench_lookups(c: &mut Criterion) {
    let graph = build_fixture();
    let mut group = c.benchmark_group("lookup");

    group.bench_function("vessel_berth/single", |b| {
        let v = VesselIndex::new(42);
        b.iter(|| black_box(unsafe { graph.vessel_berth_unchecked(black_box(v)) }));
    });

    group.bench_function("vessel_count/single", |b| {
        let berth = BerthIndex::new(5);
        b.iter(|| black_box(unsafe { graph.vessel_count_unchecked(black_box(berth)) }));
    });

    group.bench_function("first_vessel", |b| {
        let berth = BerthIndex::new(0);
        b.iter(|| black_box(unsafe { graph.first_vessel_unchecked(black_box(berth)) }));
    });

    group.bench_function("last_vessel", |b| {
        let berth = BerthIndex::new(0);
        b.iter(|| black_box(unsafe { graph.last_vessel_unchecked(black_box(berth)) }));
    });

    group.bench_function("vessel_successor", |b| {
        let v = VesselIndex::new(0);
        b.iter(|| black_box(unsafe { graph.vessel_successor_unchecked(black_box(v)) }));
    });

    group.bench_function("vessel_predecessor", |b| {
        let v = VesselIndex::new(0);
        b.iter(|| black_box(unsafe { graph.vessel_predecessor_unchecked(black_box(v)) }));
    });

    group.bench_function("is_empty", |b| {
        let berth = BerthIndex::new(0);
        b.iter(|| black_box(unsafe { graph.is_empty_unchecked(black_box(berth)) }));
    });

    group.finish();
}

fn bench_swap_vessels(c: &mut Criterion) {
    let base = build_fixture();
    let seqs = berth_sequences(&base);
    let mut group = c.benchmark_group("swap_vessels");

    // Typical: adjacent, same berth
    {
        let (a, b) = intra_berth_pair(&seqs);
        group.bench_function("typical/adjacent_same_berth", |bench| {
            let mut graph = base.clone();
            bench.iter(|| {
                unsafe { graph.swap_vessels_unchecked(black_box(a), black_box(b)) };
            });
        });
    }

    // Typical: non-adjacent, same berth
    {
        let a = seqs[0][0];
        let b = seqs[0][5];
        group.bench_function("typical/non_adjacent_same_berth", |bench| {
            let mut graph = base.clone();
            bench.iter(|| {
                unsafe { graph.swap_vessels_unchecked(black_box(a), black_box(b)) };
            });
        });
    }

    // Typical: different berths
    {
        let (a, b) = inter_berth_pair(&seqs);
        group.bench_function("typical/different_berths", |bench| {
            let mut graph = base.clone();
            bench.iter(|| {
                unsafe { graph.swap_vessels_unchecked(black_box(a), black_box(b)) };
            });
        });
    }

    // Throughput: many swaps back-to-back simulating a local search pass
    {
        let mut lcg = Lcg::new(0xDEAD);
        let pairs: Vec<(VesselIndex, VesselIndex)> = (0..1000)
            .map(|_| {
                let a = VesselIndex::new(lcg.next_bounded(NUM_VESSELS));
                let b = VesselIndex::new(lcg.next_bounded(NUM_VESSELS));
                (a, b)
            })
            .collect();

        group.bench_function("throughput/1000_mixed_swaps", |bench| {
            let mut graph = base.clone();
            bench.iter(|| {
                for &(a, b) in &pairs {
                    unsafe { graph.swap_vessels_unchecked(black_box(a), black_box(b)) };
                }
            });
        });
    }

    group.finish();
}

fn bench_swap_segments(c: &mut Criterion) {
    let base = build_fixture();
    let seqs = berth_sequences(&base);
    let mut group = c.benchmark_group("swap_segments");

    // Typical: 2-element segments, same berth
    // Berth 0 has ≥12 elements. Take [0..1] and [4..5].
    {
        let (af, al) = (seqs[0][0], seqs[0][1]);
        let (bf, bl) = (seqs[0][4], seqs[0][5]);
        group.bench_function("typical/small_same_berth", |bench| {
            let mut graph = base.clone();
            bench.iter(|| {
                unsafe {
                    graph.swap_segments_unchecked(
                        black_box(af),
                        black_box(al),
                        black_box(bf),
                        black_box(bl),
                    )
                };
            });
        });
    }

    // Typical: 2-element segments, different berths
    {
        let (af, al) = (seqs[0][0], seqs[0][1]);
        let (bf, bl) = (seqs[1][0], seqs[1][1]);
        group.bench_function("typical/small_different_berths", |bench| {
            let mut graph = base.clone();
            bench.iter(|| {
                unsafe {
                    graph.swap_segments_unchecked(
                        black_box(af),
                        black_box(al),
                        black_box(bf),
                        black_box(bl),
                    )
                };
            });
        });
    }

    // Worst case: full berth 0 vs full berth 1
    {
        let (af, al) = full_berth_segment(&seqs);
        let seq1 = &seqs[1];
        let (bf, bl) = (seq1[0], *seq1.last().unwrap());
        group.bench_function("worst/full_berth_vs_full_berth", |bench| {
            let mut graph = base.clone();
            bench.iter(|| {
                unsafe {
                    graph.swap_segments_unchecked(
                        black_box(af),
                        black_box(al),
                        black_box(bf),
                        black_box(bl),
                    )
                };
            });
        });
    }

    group.finish();
}

fn bench_relocate_after(c: &mut Criterion) {
    let base = build_fixture();
    let seqs = berth_sequences(&base);
    let mut group = c.benchmark_group("relocate_after");

    // Typical: intra-berth
    {
        let vessel = seqs[0][0];
        let anchor = seqs[0][5];
        group.bench_function("typical/intra_berth", |bench| {
            let mut graph = base.clone();
            bench.iter(|| {
                unsafe { graph.relocate_after_unchecked(black_box(vessel), black_box(anchor)) };
            });
        });
    }

    // Typical: inter-berth
    {
        let vessel = seqs[0][0];
        let anchor = seqs[1][0];
        group.bench_function("typical/inter_berth", |bench| {
            let mut graph = base.clone();
            bench.iter(|| {
                unsafe { graph.relocate_after_unchecked(black_box(vessel), black_box(anchor)) };
            });
        });
    }

    // Throughput: 1000 mixed relocations
    {
        let mut lcg = Lcg::new(0xBEEF);
        let ops: Vec<(VesselIndex, VesselIndex)> = (0..1000)
            .map(|_| {
                let v = VesselIndex::new(lcg.next_bounded(NUM_VESSELS));
                let a = VesselIndex::new(lcg.next_bounded(NUM_VESSELS));
                (v, a)
            })
            .collect();

        group.bench_function("throughput/1000_mixed", |bench| {
            let mut graph = base.clone();
            bench.iter(|| {
                for &(v, a) in &ops {
                    unsafe { graph.relocate_after_unchecked(black_box(v), black_box(a)) };
                }
            });
        });
    }

    group.finish();
}

fn bench_relocate_before(c: &mut Criterion) {
    let base = build_fixture();
    let seqs = berth_sequences(&base);
    let mut group = c.benchmark_group("relocate_before");

    // Typical: intra-berth
    {
        let vessel = seqs[0][5];
        let reference = seqs[0][0];
        group.bench_function("typical/intra_berth", |bench| {
            let mut graph = base.clone();
            bench.iter(|| {
                unsafe { graph.relocate_before_unchecked(black_box(vessel), black_box(reference)) };
            });
        });
    }

    // Typical: inter-berth (reference is first in its berth → triggers relocate_to_head path)
    {
        let vessel = seqs[0][0];
        let reference = seqs[1][0];
        group.bench_function("typical/inter_berth_head_path", |bench| {
            let mut graph = base.clone();
            bench.iter(|| {
                unsafe { graph.relocate_before_unchecked(black_box(vessel), black_box(reference)) };
            });
        });
    }

    group.finish();
}

fn bench_relocate_head_tail(c: &mut Criterion) {
    let base = build_fixture();
    let seqs = berth_sequences(&base);
    let mut group = c.benchmark_group("relocate_head_tail");

    // Typical: move one vessel to head of another berth
    {
        let vessel = seqs[0][0];
        let target = BerthIndex::new(5);
        group.bench_function("typical/to_head_inter_berth", |bench| {
            let mut graph = base.clone();
            bench.iter(|| {
                unsafe { graph.relocate_to_head_unchecked(black_box(vessel), black_box(target)) };
            });
        });
    }

    // Typical: move one vessel to tail of another berth
    {
        let vessel = seqs[0][0];
        let target = BerthIndex::new(5);
        group.bench_function("typical/to_tail_inter_berth", |bench| {
            let mut graph = base.clone();
            bench.iter(|| {
                unsafe { graph.relocate_to_tail_unchecked(black_box(vessel), black_box(target)) };
            });
        });
    }

    // Typical: move to head of same berth (no-op fast path when already at head)
    {
        let vessel = seqs[0][0];
        let target = BerthIndex::new(0);
        group.bench_function("typical/to_head_noop", |bench| {
            let graph = base.clone();
            bench.iter(|| {
                // Clone inside to avoid cumulative state changes, but this is a no-op
                // so the graph doesn't actually change.
                let mut g = graph.clone();
                unsafe { g.relocate_to_head_unchecked(black_box(vessel), black_box(target)) };
                black_box(&g);
            });
        });
    }

    group.finish();
}

fn bench_relocate_segment_after(c: &mut Criterion) {
    let base = build_fixture();
    let seqs = berth_sequences(&base);
    let mut group = c.benchmark_group("relocate_segment_after");

    // Typical: 3-element segment, intra-berth
    {
        let (sf, sl) = segment_in_berth(&seqs, 3);
        let anchor = *seqs[0].last().unwrap();
        group.bench_function("typical/3_elem_intra_berth", |bench| {
            let mut graph = base.clone();
            bench.iter(|| {
                unsafe {
                    graph.relocate_segment_after_unchecked(
                        black_box(sf),
                        black_box(sl),
                        black_box(anchor),
                    )
                };
            });
        });
    }

    // Typical: 3-element segment, inter-berth
    {
        let (sf, sl) = segment_in_berth(&seqs, 3);
        let anchor = seqs[1][0];
        group.bench_function("typical/3_elem_inter_berth", |bench| {
            let mut graph = base.clone();
            bench.iter(|| {
                unsafe {
                    graph.relocate_segment_after_unchecked(
                        black_box(sf),
                        black_box(sl),
                        black_box(anchor),
                    )
                };
            });
        });
    }

    // Worst case: entire berth 0 relocated after first vessel of berth 1
    {
        let (sf, sl) = full_berth_segment(&seqs);
        let anchor = seqs[1][0];
        group.bench_function("worst/full_berth_inter_berth", |bench| {
            let mut graph = base.clone();
            bench.iter(|| {
                unsafe {
                    graph.relocate_segment_after_unchecked(
                        black_box(sf),
                        black_box(sl),
                        black_box(anchor),
                    )
                };
            });
        });
    }

    group.finish();
}

fn bench_relocate_segment_head_tail(c: &mut Criterion) {
    let base = build_fixture();
    let seqs = berth_sequences(&base);
    let mut group = c.benchmark_group("relocate_segment_head_tail");

    // Typical: 3-element segment to head of different berth
    {
        let (sf, sl) = segment_in_berth(&seqs, 3);
        let target = BerthIndex::new(3);
        group.bench_function("typical/3_elem_to_head", |bench| {
            let mut graph = base.clone();
            bench.iter(|| {
                unsafe {
                    graph.relocate_segment_to_head_unchecked(
                        black_box(sf),
                        black_box(sl),
                        black_box(target),
                    )
                };
            });
        });
    }

    // Typical: 3-element segment to tail of different berth
    {
        let (sf, sl) = segment_in_berth(&seqs, 3);
        let target = BerthIndex::new(3);
        group.bench_function("typical/3_elem_to_tail", |bench| {
            let mut graph = base.clone();
            bench.iter(|| {
                unsafe {
                    graph.relocate_segment_to_tail_unchecked(
                        black_box(sf),
                        black_box(sl),
                        black_box(target),
                    )
                };
            });
        });
    }

    // Worst case: full berth moved to empty berth
    {
        // Berth 19 might have ~12 vessels; move full berth 0 to an initially-less-loaded berth.
        let (sf, sl) = full_berth_segment(&seqs);
        let target = BerthIndex::new(19);
        group.bench_function("worst/full_berth_to_head", |bench| {
            let mut graph = base.clone();
            bench.iter(|| {
                unsafe {
                    graph.relocate_segment_to_head_unchecked(
                        black_box(sf),
                        black_box(sl),
                        black_box(target),
                    )
                };
            });
        });
    }

    {
        let (sf, sl) = full_berth_segment(&seqs);
        let target = BerthIndex::new(19);
        group.bench_function("worst/full_berth_to_tail", |bench| {
            let mut graph = base.clone();
            bench.iter(|| {
                unsafe {
                    graph.relocate_segment_to_tail_unchecked(
                        black_box(sf),
                        black_box(sl),
                        black_box(target),
                    )
                };
            });
        });
    }

    group.finish();
}

fn bench_reverse_segment(c: &mut Criterion) {
    let base = build_fixture();
    let seqs = berth_sequences(&base);
    let mut group = c.benchmark_group("reverse_segment");

    // Typical: 3-element reversal
    {
        let (sf, sl) = segment_in_berth(&seqs, 3);
        group.bench_function("typical/3_elem", |bench| {
            // USE iter_batched to provide a fresh graph every time!
            bench.iter_batched(
                || base.clone(),
                |mut graph| {
                    unsafe { graph.reverse_segment_unchecked(black_box(sf), black_box(sl)) };
                    black_box(&graph);
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }

    // Typical: single element (no-op)
    {
        let v = seqs[0][0];
        group.bench_function("typical/single_noop", |bench| {
            bench.iter_batched(
                || base.clone(),
                |mut graph| {
                    unsafe { graph.reverse_segment_unchecked(black_box(v), black_box(v)) };
                    black_box(&graph);
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }

    // Worst case: full berth reversal
    {
        let (sf, sl) = full_berth_segment(&seqs);
        group.bench_function("worst/full_berth", |bench| {
            bench.iter_batched(
                || base.clone(),
                |mut graph| {
                    unsafe { graph.reverse_segment_unchecked(black_box(sf), black_box(sl)) };
                    black_box(&graph);
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

fn bench_overwrite(c: &mut Criterion) {
    let source = build_fixture();
    let mut group = c.benchmark_group("overwrite");

    // Typical: clone into a pre-allocated target (simulates best-solution snapshot)
    {
        group.bench_function("from_graph/preallocated", |bench| {
            let mut target = source.clone();
            bench.iter(|| {
                target.overwrite_from_graph(black_box(&source));
                black_box(&target);
            });
        });
    }

    // Cold: clone into an empty target (first-time allocation)
    {
        group.bench_function("from_graph/cold", |bench| {
            bench.iter(|| {
                let mut target = ScheduleGraph::from_slices::<i64>(&[], &[], 0);
                target.overwrite_from_graph(black_box(&source));
                black_box(&target);
            });
        });
    }

    group.finish();
}

fn bench_segment_scaling(c: &mut Criterion) {
    let base = build_fixture();
    let seqs = berth_sequences(&base);
    let berth0_len = seqs[0].len();
    let mut group = c.benchmark_group("segment_scaling");

    let sizes: Vec<usize> = vec![1, 2, 3, 5, 8, berth0_len / 2, berth0_len];

    for &size in &sizes {
        if size > berth0_len {
            continue;
        }

        let (sf, sl) = segment_in_berth(&seqs, size);
        let anchor = seqs[1][0];

        group.bench_with_input(
            BenchmarkId::new("relocate_segment_after/inter_berth", size),
            &size,
            |bench, _| {
                bench.iter_batched(
                    || base.clone(),
                    |mut graph| {
                        unsafe {
                            graph.relocate_segment_after_unchecked(
                                black_box(sf),
                                black_box(sl),
                                black_box(anchor),
                            )
                        };
                        black_box(&graph);
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );

        group.bench_with_input(
            BenchmarkId::new("reverse_segment", size),
            &size,
            |bench, _| {
                // FIXED: If you want to benchmark reversing back and forth,
                // you must flip the arguments on the second call!
                // OR just use iter_batched for a single clean call.
                // Using iter_batched is more accurate.
                bench.iter_batched(
                    || base.clone(),
                    |mut graph| {
                        unsafe { graph.reverse_segment_unchecked(black_box(sf), black_box(sl)) };
                        black_box(&graph);
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

fn bench_local_search_simulation(c: &mut Criterion) {
    let base = build_fixture();
    let mut group = c.benchmark_group("local_search_simulation");

    // Generate a deterministic sequence of 1000 mixed operations.
    // Mix: 40% relocate_after, 30% swap_vessels, 20% reverse 3-elem, 10% relocate_to_head
    let mut lcg = Lcg::new(0xCAFE);

    enum Op {
        RelocateAfter(VesselIndex, VesselIndex),
        SwapVessels(VesselIndex, VesselIndex),
        ReverseSmall(VesselIndex), // reverse 3 consecutive starting from this vessel
        RelocateToHead(VesselIndex, BerthIndex),
    }

    let ops: Vec<Op> = (0..1000)
        .map(|_| {
            let kind = lcg.next_bounded(100);
            if kind < 40 {
                let v = VesselIndex::new(lcg.next_bounded(NUM_VESSELS));
                let a = VesselIndex::new(lcg.next_bounded(NUM_VESSELS));
                Op::RelocateAfter(v, a)
            } else if kind < 70 {
                let a = VesselIndex::new(lcg.next_bounded(NUM_VESSELS));
                let b = VesselIndex::new(lcg.next_bounded(NUM_VESSELS));
                Op::SwapVessels(a, b)
            } else if kind < 90 {
                let v = VesselIndex::new(lcg.next_bounded(NUM_VESSELS));
                Op::ReverseSmall(v)
            } else {
                let v = VesselIndex::new(lcg.next_bounded(NUM_VESSELS));
                let b = BerthIndex::new(lcg.next_bounded(NUM_BERTHS));
                Op::RelocateToHead(v, b)
            }
        })
        .collect();

    group.bench_function("1000_mixed_ops", |bench| {
        bench.iter_batched(
            || base.clone(),
            |mut graph| {
                for op in &ops {
                    match *op {
                        Op::RelocateAfter(v, a) => unsafe {
                            graph.relocate_after_unchecked(v, a);
                        },
                        Op::SwapVessels(a, b) => unsafe {
                            graph.swap_vessels_unchecked(a, b);
                        },
                        Op::ReverseSmall(v) => {
                            // Reverse from v to its 2nd successor (if it exists),
                            // staying within the berth.
                            let next1 = graph.raw_next(v);
                            if next1.get() < NUM_VESSELS {
                                let next2 = graph.raw_next(next1);
                                if next2.get() < NUM_VESSELS {
                                    unsafe {
                                        graph.reverse_segment_unchecked(v, next2);
                                    }
                                }
                            }
                        }
                        Op::RelocateToHead(v, b) => unsafe {
                            graph.relocate_to_head_unchecked(v, b);
                        },
                    }
                }
                black_box(&graph);
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_iteration,
    bench_lookups,
    bench_swap_vessels,
    bench_swap_segments,
    bench_relocate_after,
    bench_relocate_before,
    bench_relocate_head_tail,
    bench_relocate_segment_after,
    bench_relocate_segment_head_tail,
    bench_reverse_segment,
    bench_overwrite,
    bench_segment_scaling,
    bench_local_search_simulation,
);

criterion_main!(benches);
