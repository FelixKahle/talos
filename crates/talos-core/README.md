# talos-core

Shared foundational types and algorithms used across the Talos solver.

## Modules

- **math**
  - **interval** — Generic closed-open interval `[start, end)` with set operations
    (intersection, union, difference, gap, split), predicates, iteration, and
    operator sugar (`&` / `|`).

- **algorithm**
  - **interval** — Utilities for sorted, disjoint interval slices: validation
    (`are_disjoint_and_sorted`) and lower-bound searches (linear, binary, and
    threshold-based).

- **container**
  - **rarena** — Allocation-free ring arena for topological sequences. Stores
    closed rings as prev/next arrays with O(1) splicing and traversal iterators.

- **utils**
  - **index** — Zero-cost phantom-typed `usize` wrapper (`TypedIndex<T>`) that
    prevents accidental mixing of indices from different domains at compile time.
  - **num** — `SolverNumeric` trait bounding the numeric types the solver
    operates on (signed integers with saturating arithmetic, `Send + Sync`, etc.).
