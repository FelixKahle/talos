// Copyright 2026 Felix Kahle
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The benchmarks of the private kernels of the engine, run by the one ignored test of this module:
//! `cargo test --release -p talos_ls -- --ignored test_benchmarks`.

use criterion::Criterion;

#[test]
#[ignore = "benchmarks run on request, in a release build"]
fn test_benchmarks() {
    let mut criterion = Criterion::default().configure_from_args();
    crate::decoding::benches::bench_mutation_cycles(&mut criterion);
    criterion.final_summary();
}
