// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The benchmarks of the private kernels of the engine, run by the one ignored test of this module:
//! `cargo test --release -p talos_ls --lib -- --ignored --nocapture test_benchmarks`.

use criterion::Criterion;

#[test]
#[ignore = "benchmarks run on request, in a release build"]
fn test_benchmarks() {
    let mut criterion = Criterion::default();
    crate::decoding::bench_mutation_cycles(&mut criterion);
    criterion.final_summary();
}
