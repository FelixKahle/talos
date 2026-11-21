# talos_core

Foundations shared by the Talos crates. Everything is re-exported at the crate root.

- **`TypedIndex<Tag>`** — one id type per kind of id, tagged by a `TypedIndexTag`. An id stores a `u32` and is
  made from and read as a `usize` (`new`, `try_new`, `index`).
- **`SolverNumeric`** — the signed integer types (`i8` to `i64`, `isize`) a problem's numbers may have.
- **`ClosedOpenInterval<T>`** — the interval `[start, end)` of integers, with intersection, union, gap, split,
  midpoint and iteration over its points (`ClosedOpenIntervalIter`).
