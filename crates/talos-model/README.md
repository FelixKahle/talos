# talos-model

Data model for the Dynamic Berth Allocation Problem (DBAP).

## Modules

- **index** — Strongly typed indices (`VesselIndex`, `BerthIndex`) built on
  `talos-core`'s `TypedIndex`. Prevents accidental mixing of vessel and berth
  indices at compile time with zero runtime cost.

- **model** — The problem definition (`Model`). Stores arrival times, latest
  departure times, vessel weights, processing times per vessel-berth pair, and
  berth opening intervals. Uses `ProcessingTime<T>`, a sentinel-encoded optional
  that avoids the `Option` discriminant overhead in hot loops.

- **assignment** — `Assignment<T>` pairs a start time with a berth index for a
  single vessel.

- **solution** — SoA (Structure of Arrays) solution representation. `Solution`
  owns the data; `SolutionView` borrows it. Both store per-vessel berth
  assignments, start times, and an objective value. All accessors are O(1).
