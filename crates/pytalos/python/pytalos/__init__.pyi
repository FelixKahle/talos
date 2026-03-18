# Copyright (c) 2026 Felix Kahle.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""Type stubs for the pytalos native module."""

from typing import Callable
from enum import IntEnum
from .pytalos import *

class Model:
    """The DBAP problem model."""

    num_vessels: int
    num_berths: int

    def __init__(
        self,
        num_vessels: int,
        num_berths: int,
        arrivals: list[int],
        deadlines: list[int],
        weights: list[int],
        processing_times: list[int | None],
        time_windows: list[list[tuple[int, int]]],
    ) -> None: ...

class Solution:
    """A DBAP solution: berth assignments and start times for every vessel."""

    objective: int
    berths: list[int]
    start_times: list[int]
    num_vessels: int

    def __init__(
        self,
        berths: list[int],
        start_times: list[int],
        objective: int,
    ) -> None: ...
    def __repr__(self) -> str: ...

class Operator(IntEnum):
    """Neighbourhood operator."""

    IntraSwap = 0
    InterSwap = 1
    IntraShift = 2
    InterShift = 3

class LocalSearchConfig:
    """Configuration for the local search engine."""

    operators: list[Operator]
    max_iterations: int | None
    max_solutions: int | None
    max_cycles: int | None
    max_non_improving_iterations: int | None
    max_non_improving_cycles: int | None
    max_non_improving_time_secs: float | None
    time_limit_secs: float | None

    def __init__(
        self,
        operators: list[Operator],
        max_iterations: int | None = None,
        max_solutions: int | None = None,
        max_cycles: int | None = None,
        max_non_improving_iterations: int | None = None,
        max_non_improving_cycles: int | None = None,
        max_non_improving_time_secs: float | None = None,
        time_limit_secs: float | None = None,
    ) -> None: ...
    def __repr__(self) -> str: ...

class LambdaStrategy(IntEnum):
    """Lambda scaling strategy."""

    Static = 0
    Dynamic = 1
    Additive = 2

class Trigger(IntEnum):
    """When GLS fires its penalization step."""

    OnExhaustion = 0
    AfterNonImprovements = 1
    AfterMoves = 2

class Decay(IntEnum):
    """Penalty decay strategy."""

    NoDecay = 0
    Geometric = 1

class GlsConfig:
    """Guided Local Search configuration."""

    lambda_strategy: LambdaStrategy
    lambda_initial: float | None
    lambda_inc_step: float
    lambda_dec_step: float
    lambda_min: float | None
    lambda_max: float | None
    trigger: Trigger
    trigger_threshold: int
    decay: Decay
    decay_factor: float
    decay_period: int
    reset_on_best: bool

    def __init__(
        self,
        lambda_strategy: LambdaStrategy = ...,
        lambda_initial: float | None = None,
        lambda_inc_step: float = 0.1,
        lambda_dec_step: float = 0.1,
        lambda_min: float | None = None,
        lambda_max: float | None = None,
        trigger: Trigger = ...,
        trigger_threshold: int = 1000000,
        decay: Decay = ...,
        decay_factor: float = 0.9,
        decay_period: int = 10,
        reset_on_best: bool = False,
    ) -> None: ...
    def __repr__(self) -> str: ...

class TerminationReason(IntEnum):
    """Reason the local search terminated."""

    TimeLimitReached = 0
    SolutionLimitReached = 1
    IterationLimitReached = 2
    CycleLimitReached = 3
    MaxNonImprovingIterations = 4
    MaxNonImprovingCycles = 5
    MaxNonImprovingTime = 6
    TargetObjectiveReached = 7
    NeighborhoodExhausted = 8
    Interrupted = 9
    Aborted = 10

class SearchResult:
    """Result returned by the solver."""

    solution: Solution
    termination_reason: TerminationReason
    iterations: int
    accepted_solutions: int
    total_solutions: int
    infeasible_moves: int
    cycles: int
    time_total_secs: float

    def __repr__(self) -> str: ...

class Solver:
    """Reusable solver that keeps its internal Engine across calls."""

    def __init__(
        self, num_vessels: int = 0, num_berths: int = 0
    ) -> None: ...
    def solve(
        self,
        model: Model,
        config: LocalSearchConfig,
        gls_config: GlsConfig | None,
        solution: Solution,
        callback: Callable[[int, list[int], list[int]], None] | None = None,
    ) -> SearchResult: ...

def heuristic_gls_lambda(
    objective: float, num_features: int, scale: float
) -> float:
    """Heuristic lambda calculation for GLS."""
    ...
