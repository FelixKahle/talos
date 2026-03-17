# pytalos

Python bindings for **Talos**, a high-performance local-search solver for the
Dynamic Berth Allocation Problem (DBAP). Built with [PyO3](https://pyo3.rs) and
packaged via [Maturin](https://www.maturin.rs).

## Installation

Requires Python ≥ 3.8 and a Rust toolchain.

```bash
# Inside a virtual-env, from the repository root:
pip install maturin
maturin develop --release        # compile & install into the active venv
```

## Quick start

```python
import pytalos
from pytalos import (
    Model,
    Solution,
    Operator,
    LocalSearchConfig,
    GlsConfig,
    LambdaStrategy,
    Trigger,
    Decay,
)

# 1. Define the problem -------------------------------------------------------
num_vessels = 4
num_berths = 2

model = Model(
    num_vessels=num_vessels,
    num_berths=num_berths,
    arrivals=[0, 2, 5, 8],
    deadlines=[20, 25, 30, 35],
    weights=[1, 2, 1, 3],
    # Flat row-major list of length num_vessels * num_berths.
    # processing_times[v * num_berths + b] is the processing time of vessel v
    # at berth b, or None if that berth is infeasible for the vessel.
    processing_times=[5, 7, 6, None, 4, 8, 3, 5],
    # One list of (start, end) intervals per berth.
    time_windows=[
        [(0, 40)],  # berth 0 available in [0, 40)
        [(0, 40)],  # berth 1 available in [0, 40)
    ],
)

# 2. Provide an initial solution -----------------------------------------------
initial = Solution(
    berths=[0, 1, 0, 1],        # berth assignment per vessel
    start_times=[0, 2, 10, 8],  # start time per vessel
    objective=42,                # objective value of this schedule
)

# 3. Configure the local search ------------------------------------------------
config = LocalSearchConfig(
    operators=[
        Operator.IntraSwap,
        Operator.InterSwap,
        Operator.IntraShift,
        Operator.InterShift,
    ],
    time_limit_secs=10.0,
    max_non_improving_iterations=5000,
)

# 4. (Optional) Tune the GLS metaheuristic ------------------------------------
gls = GlsConfig(
    lambda_strategy=LambdaStrategy.Dynamic,
    lambda_inc_step=0.15,
    lambda_dec_step=0.10,
    trigger=Trigger.OnExhaustion,
    decay=Decay.NoDecay,
    reset_on_best=True,
)

# 5. (Optional) Callback on every new best solution ---------------------------
def on_new_best(objective, berths, start_times):
    print(f"  new best: {objective}")

# 6. Solve ---------------------------------------------------------------------
result = pytalos.solve(
    model=model,
    config=config,
    gls_config=gls,        # pass None for default GLS settings
    solution=initial,
    callback=on_new_best,  # or omit
)

# 7. Inspect the result --------------------------------------------------------
print(result)                        # SearchResult(objective=..., ...)
print(result.solution.objective)     # best objective value
print(result.solution.berths)        # list[int]
print(result.solution.start_times)   # list[int]
print(result.termination_reason)     # e.g. TerminationReason.TimeLimitReached
print(result.iterations)
print(result.time_total_secs)
```

## API reference

### `Model`

| Parameter | Type | Description |
|---|---|---|
| `num_vessels` | `int` | Number of vessels |
| `num_berths` | `int` | Number of berths |
| `arrivals` | `list[int]` | Arrival time per vessel |
| `deadlines` | `list[int]` | Deadline per vessel |
| `weights` | `list[int]` | Priority weight per vessel |
| `processing_times` | `list[int \| None]` | Flat row-major `(vessel, berth)` processing times; `None` = infeasible |
| `time_windows` | `list[list[tuple[int, int]]]` | Per-berth availability intervals `[start, end)` |

### `Solution`

| Parameter | Type | Description |
|---|---|---|
| `berths` | `list[int]` | Berth index assigned to each vessel |
| `start_times` | `list[int]` | Start time assigned to each vessel |
| `objective` | `int` | Objective value of the schedule |

### `LocalSearchConfig`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `operators` | `list[Operator]` | *(required)* | Neighbourhood operators to use |
| `time_limit_secs` | `float \| None` | `None` | Wall-clock time limit in seconds |
| `max_iterations` | `int \| None` | `None` | Hard iteration cap |
| `max_solutions` | `int \| None` | `None` | Stop after this many improving solutions |
| `max_cycles` | `int \| None` | `None` | Stop after this many operator cycles |
| `max_non_improving_iterations` | `int \| None` | `None` | Stagnation patience (iterations) |
| `max_non_improving_cycles` | `int \| None` | `None` | Stagnation patience (cycles) |
| `max_non_improving_time_secs` | `float \| None` | `None` | Stagnation patience (seconds) |
| `target_objective` | `int \| None` | `None` | Stop when objective ≤ target |

### `Operator`

| Variant | Description |
|---|---|
| `IntraSwap` | Swap two vessels on the same berth |
| `InterSwap` | Swap two vessels across different berths |
| `IntraShift` | Move a vessel to a new position on the same berth |
| `InterShift` | Move a vessel to a different berth |

### `GlsConfig`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `lambda_strategy` | `LambdaStrategy` | `Dynamic` | Lambda scaling strategy |
| `lambda_initial` | `float \| None` | `None` (heuristic) | Initial lambda value |
| `lambda_inc_step` | `float` | `0.1` | Lambda increase step |
| `lambda_dec_step` | `float` | `0.1` | Lambda decrease step |
| `lambda_min` | `float \| None` | `None` (heuristic) | Lambda lower bound |
| `lambda_max` | `float \| None` | `None` (heuristic) | Lambda upper bound |
| `trigger` | `Trigger` | `OnExhaustion` | When to apply penalization |
| `trigger_threshold` | `int` | `1000000` | Threshold for `AfterNonImprovements` / `AfterMoves` |
| `decay` | `Decay` | `NoDecay` | Penalty decay strategy |
| `decay_factor` | `float` | `0.9` | Geometric decay factor |
| `decay_period` | `int` | `10` | Decay application period |
| `reset_on_best` | `bool` | `False` | Reset lambda when a new best is found |

### `SearchResult`

| Attribute | Type | Description |
|---|---|---|
| `solution` | `Solution` | Best solution found |
| `termination_reason` | `TerminationReason` | Why the search stopped |
| `iterations` | `int` | Total iterations performed |
| `accepted_solutions` | `int` | Number of accepted moves |
| `total_solutions` | `int` | Number of improving solutions found |
| `infeasible_moves` | `int` | Infeasible moves encountered |
| `cycles` | `int` | Operator cycles completed |
| `time_total_secs` | `float` | Wall-clock time in seconds |

### `TerminationReason`

`TimeLimitReached`, `SolutionLimitReached`, `IterationLimitReached`,
`CycleLimitReached`, `MaxNonImprovingIterations`, `MaxNonImprovingCycles`,
`MaxNonImprovingTime`, `TargetObjectiveReached`, `NeighborhoodExhausted`,
`Interrupted`, `Aborted`.
