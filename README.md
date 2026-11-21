# Talos 🐋

**A blazing-fast, zero-allocation local search engine for combinatorial optimization.**

Talos is a high-performance metaheuristic solver built in Rust, specifically tailored for complex resource scheduling and routing tasks — most notably the **Dynamic Berth Allocation Problem (DBAP)**. It is engineered from the ground up using strict **Data-Oriented Design (DOD)** principles to maximize CPU cache locality, defeat the memory allocator, and evaluate tens of millions of candidate solutions per second.

## Crates

| Crate | Description |
|---|---|
| **talos_core** | Shared foundational types and algorithms: intervals, ring arenas, typed indices, numeric traits. |
| **talos_model** | Problem definition and solution representation for the DBAP: model data, vessel/berth indices, SoA solutions. |
| **talos_ls** | The local search engine: mutation operators, schedule graph, decoding, and pluggable termination monitors. |

## Building

Requires the Rust **stable** toolchain (edition 2024).

```bash
cargo build --workspace
```

Run all tests:

```bash
cargo test --workspace
```

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT licence](LICENSE-MIT), at your
option.

## Citation

If you use Talos in academic work, please cite:

```bibtex
@software{kahle2026talos,
  author    = {Felix Kahle},
  title     = {Talos: A local search framework for the dynamic berth allocation problem},
  version   = {0.1.0},
  year      = {2026},
  url       = {https://github.com/FelixKahle/talos}
}
```
