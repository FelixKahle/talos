# monitor

Monitoring and termination control for local search runs.

This module defines the `LocalSearchMonitor` trait and a set of concrete monitors
that observe the search lifecycle and decide when to stop. The trait provides
callbacks for start/end events, per-iteration updates, candidate evaluation,
and solution acceptance or rejection. Each monitor can issue a `SearchCommand`
to either continue or terminate the search.

## Modules

- **lsmonitor** — The `LocalSearchMonitor` trait. Defines all lifecycle callbacks
  and the `search_command` hook used by the engine to poll for termination.
- **composite** — `CompositeLocalSearchMonitor`. Fans out every callback to a list
  of inner monitors. For termination, the first monitor that fires wins.
- **iteration** — `IterationLimitMonitor`. Terminates after a fixed number of
  iterations.
- **cycle** — `CycleLimitMonitor`. Terminates after a fixed number of cycles
  (full neighborhood traversals).
- **solution** — `SolutionLimitMonitor`. Terminates after a fixed number of
  candidate solutions have been evaluated.
- **time** — `TimeLimitMonitor`. Terminates after a wall-clock duration. Uses a
  bitmask on the iteration counter to throttle clock reads.
- **nimpr** — `NoImprovementMonitor`. Terminates after a configurable stretch
  without improvement to the global best solution. Supports iteration-based,
  cycle-based, and duration-based patience simultaneously; whichever fires
  first wins. Duration checks use the same bitmask throttling as `TimeLimitMonitor`.
