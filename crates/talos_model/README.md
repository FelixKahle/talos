# talos_model

The problem and the answer of the dynamic berth allocation problem (DBAP), on top of `talos_core`.

- **Ids.** `VesselIndex` and `BerthIndex` are typed ids, so a vessel cannot index a berth table.
  `VesselIndexIter` and `BerthIndexIter` walk over every id of one kind.
- **`Model`.** A problem instance: when each vessel arrives and must have left, its weight, its processing
  time at each berth it may use (`None` where it may not), and the intervals during which each berth is open.
  `Model::try_new` checks the input once and refuses it as a `ModelError`; each berth's openings are kept as
  their union, ordered and without overlap. Accessors answer in constant time, as a checked form and an
  `_unchecked` form for the engine's loops.
- **`Solution`.** A schedule: for each vessel the berth and the start time of its service, as tables keyed by
  the vessel, with the objective value. `SolutionView` is the same schedule borrowed from another owner's
  tables, and `Assignment` is the berth and start time of one vessel.

```rust
use talos_core::ClosedOpenInterval;
use talos_model::{BerthIndex, Model, VesselIndex};

let model = Model::new(
    vec![0, 10],         // arrival times
    vec![50, 60],        // latest departure times
    vec![1, 2],          // weights
    vec![Some(5), None], // processing times, one per vessel and berth
    vec![vec![ClosedOpenInterval::new(0, 100)]],
);
assert_eq!(model.vessel_processing_time(VesselIndex::new(1), BerthIndex::new(0)), None);
```
