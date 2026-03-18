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

"""
Comprehensive test suite for the pytalos package.

This module tests all major components of the pytalos Python bindings:
- Model creation and validation
- Solution creation and validation
- Operator enums
- LocalSearchConfig
- GLS configuration (GlsConfig, LambdaStrategy, Trigger, Decay)
- Solver functionality
- Utility functions
- End-to-end integration tests
"""

import pytalos
import pytest

# ----------------------------------------------------------------------------
# Fixtures - Reusable test data
# ----------------------------------------------------------------------------


@pytest.fixture
def simple_model():
    """Create a simple 2-vessel, 2-berth problem for testing."""
    return pytalos.Model(
        num_vessels=2,
        num_berths=2,
        arrivals=[0, 2],
        deadlines=[20, 25],
        weights=[1, 2],
        processing_times=[5, 7, 6, 8],  # 2 vessels × 2 berths
        time_windows=[
            [(0, 30)],  # berth 0 available in [0, 30)
            [(0, 30)],  # berth 1 available in [0, 30)
        ],
    )


@pytest.fixture
def simple_solution():
    """Create a simple solution for the 2-vessel problem."""
    return pytalos.Solution(
        berths=[0, 1],
        start_times=[0, 2],
        objective=15,
    )


@pytest.fixture
def larger_model():
    """Create a larger 4-vessel, 2-berth problem for integration tests."""
    return pytalos.Model(
        num_vessels=4,
        num_berths=2,
        arrivals=[0, 2, 5, 8],
        deadlines=[20, 25, 30, 35],
        weights=[1, 2, 1, 3],
        processing_times=[5, 7, 6, None, 4, 8, 3, 5],  # None = infeasible
        time_windows=[
            [(0, 40)],
            [(0, 40)],
        ],
    )


@pytest.fixture
def larger_solution():
    """Create a solution for the 4-vessel problem."""
    # All vessels on berth 0 (to avoid infeasible combinations)
    # Sequential placement respecting arrivals and processing times
    return pytalos.Solution(
        berths=[0, 0, 0, 0],
        start_times=[0, 5, 11, 15],  # Sequential on same berth
        objective=60,
    )


# ----------------------------------------------------------------------------
# Model Creation Tests
# ----------------------------------------------------------------------------


def test_model_create_valid(simple_model):
    """Test creating a valid model with correct dimensions."""
    assert simple_model.num_vessels == 2
    assert simple_model.num_berths == 2


def test_model_create_larger():
    """Test creating a larger model with various parameters."""
    model = pytalos.Model(
        num_vessels=5,
        num_berths=3,
        arrivals=[0, 1, 2, 3, 4],
        deadlines=[10, 15, 20, 25, 30],
        weights=[1, 1, 2, 3, 1],
        processing_times=[5, 6, 7] * 5,  # 5 vessels × 3 berths
        time_windows=[[(0, 50)], [(5, 45)], [(0, 50)]],
    )
    assert model.num_vessels == 5
    assert model.num_berths == 3


def test_model_with_infeasible_berths():
    """Test creating a model with some infeasible vessel-berth combinations."""
    model = pytalos.Model(
        num_vessels=2,
        num_berths=2,
        arrivals=[0, 2],
        deadlines=[20, 25],
        weights=[1, 2],
        processing_times=[
            5,
            None,
            None,
            8,
        ],  # vessel 0 can't use berth 1, vessel 1 can't use berth 0
        time_windows=[[(0, 30)], [(0, 30)]],
    )
    assert model.num_vessels == 2
    assert model.num_berths == 2


def test_model_invalid_arrivals_length():
    """Test that model creation fails when arrivals length is wrong."""
    with pytest.raises(ValueError, match="arrivals length .* != num_vessels"):
        pytalos.Model(
            num_vessels=2,
            num_berths=2,
            arrivals=[0],  # Wrong length!
            deadlines=[20, 25],
            weights=[1, 2],
            processing_times=[5, 7, 6, 8],
            time_windows=[[(0, 30)], [(0, 30)]],
        )


def test_model_invalid_deadlines_length():
    """Test that model creation fails when deadlines length is wrong."""
    with pytest.raises(ValueError, match="deadlines length .* != num_vessels"):
        pytalos.Model(
            num_vessels=2,
            num_berths=2,
            arrivals=[0, 2],
            deadlines=[20],  # Wrong length!
            weights=[1, 2],
            processing_times=[5, 7, 6, 8],
            time_windows=[[(0, 30)], [(0, 30)]],
        )


def test_model_invalid_weights_length():
    """Test that model creation fails when weights length is wrong."""
    with pytest.raises(ValueError, match="weights length .* != num_vessels"):
        pytalos.Model(
            num_vessels=2,
            num_berths=2,
            arrivals=[0, 2],
            deadlines=[20, 25],
            weights=[1],  # Wrong length!
            processing_times=[5, 7, 6, 8],
            time_windows=[[(0, 30)], [(0, 30)]],
        )


def test_model_invalid_processing_times_length():
    """Test that model creation fails when processing_times length is wrong."""
    with pytest.raises(ValueError, match="processing_times length"):
        pytalos.Model(
            num_vessels=2,
            num_berths=2,
            arrivals=[0, 2],
            deadlines=[20, 25],
            weights=[1, 2],
            processing_times=[5, 7, 6],  # Wrong length! Should be 4
            time_windows=[[(0, 30)], [(0, 30)]],
        )


def test_model_invalid_time_windows_length():
    """Test that model creation fails when time_windows length is wrong."""
    with pytest.raises(ValueError, match="time_windows length .* != num_berths"):
        pytalos.Model(
            num_vessels=2,
            num_berths=2,
            arrivals=[0, 2],
            deadlines=[20, 25],
            weights=[1, 2],
            processing_times=[5, 7, 6, 8],
            time_windows=[[(0, 30)]],  # Wrong length! Should be 2
        )


def test_model_properties(simple_model):
    """Test accessing model properties."""
    assert simple_model.num_vessels == 2
    assert simple_model.num_berths == 2


# ----------------------------------------------------------------------------
# Solution Tests
# ----------------------------------------------------------------------------


def test_solution_create_valid(simple_solution):
    """Test creating a valid solution."""
    assert simple_solution.num_vessels == 2
    assert simple_solution.berths == [0, 1]
    assert simple_solution.start_times == [0, 2]
    assert simple_solution.objective == 15


def test_solution_properties_access():
    """Test accessing solution properties."""
    sol = pytalos.Solution(
        berths=[0, 1, 0],
        start_times=[0, 5, 10],
        objective=100,
    )
    assert sol.num_vessels == 3
    assert sol.berths == [0, 1, 0]
    assert sol.start_times == [0, 5, 10]
    assert sol.objective == 100


def test_solution_invalid_mismatched_lengths():
    """Test that solution creation fails when berths and start_times lengths differ."""
    with pytest.raises(ValueError, match="berths length .* != start_times length"):
        pytalos.Solution(
            berths=[0, 1],
            start_times=[0],  # Wrong length!
            objective=10,
        )


def test_solution_repr(simple_solution):
    """Test the string representation of a solution."""
    repr_str = repr(simple_solution)
    assert "Solution" in repr_str
    assert "num_vessels=2" in repr_str
    assert "objective=15" in repr_str


# ----------------------------------------------------------------------------
# Operator Tests
# ----------------------------------------------------------------------------


def test_operator_intra_swap_exists():
    """Test that IntraSwap operator exists."""
    op = pytalos.Operator.IntraSwap
    assert op is not None


def test_operator_inter_swap_exists():
    """Test that InterSwap operator exists."""
    op = pytalos.Operator.InterSwap
    assert op is not None


def test_operator_intra_shift_exists():
    """Test that IntraShift operator exists."""
    op = pytalos.Operator.IntraShift
    assert op is not None


def test_operator_inter_shift_exists():
    """Test that InterShift operator exists."""
    op = pytalos.Operator.InterShift
    assert op is not None


def test_operators_in_list():
    """Test that operators can be used in a list."""
    operators = [
        pytalos.Operator.IntraSwap,
        pytalos.Operator.InterSwap,
        pytalos.Operator.IntraShift,
        pytalos.Operator.InterShift,
    ]
    assert len(operators) == 4
    assert pytalos.Operator.IntraSwap in operators


def test_operators_equality():
    """Test that operator equality works."""
    op1 = pytalos.Operator.IntraSwap
    op2 = pytalos.Operator.IntraSwap
    op3 = pytalos.Operator.InterSwap
    assert op1 == op2
    assert op1 != op3


# ----------------------------------------------------------------------------
# LocalSearchConfig Tests
# ----------------------------------------------------------------------------


def test_config_create_minimal():
    """Test creating a config with only required parameters."""
    config = pytalos.LocalSearchConfig(
        operators=[pytalos.Operator.IntraSwap],
    )
    assert len(config.operators) == 1
    assert config.max_iterations is None
    assert config.time_limit_secs is None


def test_config_create_with_all_options():
    """Test creating a config with all optional parameters."""
    config = pytalos.LocalSearchConfig(
        operators=[
            pytalos.Operator.IntraSwap,
            pytalos.Operator.InterSwap,
        ],
        max_iterations=10000,
        max_solutions=100,
        max_cycles=500,
        max_non_improving_iterations=5000,
        max_non_improving_cycles=50,
        max_non_improving_time_secs=30.0,
        time_limit_secs=60.0,
    )
    assert len(config.operators) == 2
    assert config.max_iterations == 10000
    assert config.max_solutions == 100
    assert config.max_cycles == 500
    assert config.max_non_improving_iterations == 5000
    assert config.max_non_improving_cycles == 50
    assert config.max_non_improving_time_secs == 30.0
    assert config.time_limit_secs == 60.0


def test_config_time_limit_only():
    """Test creating a config with only time limit."""
    config = pytalos.LocalSearchConfig(
        operators=[pytalos.Operator.IntraSwap],
        time_limit_secs=10.0,
    )
    assert config.time_limit_secs == 10.0
    assert config.max_iterations is None


def test_config_repr():
    """Test the string representation of a config."""
    config = pytalos.LocalSearchConfig(
        operators=[pytalos.Operator.IntraSwap],
        time_limit_secs=5.0,
    )
    repr_str = repr(config)
    assert "LocalSearchConfig" in repr_str
    assert "operators" in repr_str


def test_config_duplicate_operators():
    """Test that duplicate operators are deduplicated."""
    config = pytalos.LocalSearchConfig(
        operators=[
            pytalos.Operator.IntraSwap,
            pytalos.Operator.IntraSwap,  # Duplicate
            pytalos.Operator.InterSwap,
        ],
    )
    # Duplicates should be removed
    assert len(config.operators) <= 2


# ----------------------------------------------------------------------------
# GLS Configuration Tests
# ----------------------------------------------------------------------------


def test_gls_config_lambda_strategy_static():
    """Test creating GlsConfig with Static lambda strategy."""
    gls = pytalos.GlsConfig(lambda_strategy=pytalos.LambdaStrategy.Static)
    assert gls.lambda_strategy == pytalos.LambdaStrategy.Static


def test_gls_config_lambda_strategy_dynamic():
    """Test creating GlsConfig with Dynamic lambda strategy."""
    gls = pytalos.GlsConfig(lambda_strategy=pytalos.LambdaStrategy.Dynamic)
    assert gls.lambda_strategy == pytalos.LambdaStrategy.Dynamic


def test_gls_config_lambda_strategy_additive():
    """Test creating GlsConfig with Additive lambda strategy."""
    gls = pytalos.GlsConfig(lambda_strategy=pytalos.LambdaStrategy.Additive)
    assert gls.lambda_strategy == pytalos.LambdaStrategy.Additive


def test_gls_config_trigger_on_exhaustion():
    """Test creating GlsConfig with OnExhaustion trigger."""
    gls = pytalos.GlsConfig(trigger=pytalos.Trigger.OnExhaustion)
    assert gls.trigger == pytalos.Trigger.OnExhaustion


def test_gls_config_trigger_after_non_improvements():
    """Test creating GlsConfig with AfterNonImprovements trigger."""
    gls = pytalos.GlsConfig(trigger=pytalos.Trigger.AfterNonImprovements)
    assert gls.trigger == pytalos.Trigger.AfterNonImprovements


def test_gls_config_trigger_after_moves():
    """Test creating GlsConfig with AfterMoves trigger."""
    gls = pytalos.GlsConfig(trigger=pytalos.Trigger.AfterMoves)
    assert gls.trigger == pytalos.Trigger.AfterMoves


def test_gls_config_decay_no_decay():
    """Test creating GlsConfig with NoDecay."""
    gls = pytalos.GlsConfig(decay=pytalos.Decay.NoDecay)
    assert gls.decay == pytalos.Decay.NoDecay


def test_gls_config_decay_geometric():
    """Test creating GlsConfig with Geometric decay."""
    gls = pytalos.GlsConfig(decay=pytalos.Decay.Geometric)
    assert gls.decay == pytalos.Decay.Geometric


def test_gls_config_create_with_all_options():
    """Test creating GlsConfig with all parameters specified."""
    gls = pytalos.GlsConfig(
        lambda_strategy=pytalos.LambdaStrategy.Dynamic,
        lambda_initial=0.5,
        lambda_inc_step=0.15,
        lambda_dec_step=0.10,
        lambda_min=0.01,
        lambda_max=10.0,
        trigger=pytalos.Trigger.OnExhaustion,
        trigger_threshold=500,
        decay=pytalos.Decay.Geometric,
        decay_factor=0.95,
        decay_period=20,
        reset_on_best=True,
    )
    assert gls.lambda_strategy == pytalos.LambdaStrategy.Dynamic
    assert gls.lambda_initial == 0.5
    assert gls.lambda_inc_step == 0.15
    assert gls.lambda_dec_step == 0.10
    assert gls.lambda_min == 0.01
    assert gls.lambda_max == 10.0
    assert gls.trigger == pytalos.Trigger.OnExhaustion
    assert gls.trigger_threshold == 500
    assert gls.decay == pytalos.Decay.Geometric
    assert gls.decay_factor == 0.95
    assert gls.decay_period == 20
    assert gls.reset_on_best is True


def test_gls_config_defaults():
    """Test GlsConfig default values."""
    gls = pytalos.GlsConfig()
    assert gls.lambda_strategy == pytalos.LambdaStrategy.Dynamic
    assert gls.lambda_initial is None  # Uses heuristic
    assert gls.lambda_inc_step == 0.1
    assert gls.lambda_dec_step == 0.1
    assert gls.trigger == pytalos.Trigger.OnExhaustion
    assert gls.decay == pytalos.Decay.NoDecay
    assert gls.reset_on_best is False


def test_gls_config_repr():
    """Test the string representation of GlsConfig."""
    gls = pytalos.GlsConfig(
        lambda_strategy=pytalos.LambdaStrategy.Dynamic,
        trigger=pytalos.Trigger.OnExhaustion,
        decay=pytalos.Decay.NoDecay,
    )
    repr_str = repr(gls)
    assert "GlsConfig" in repr_str
    assert "strategy" in repr_str
    assert "trigger" in repr_str
    assert "decay" in repr_str


# ----------------------------------------------------------------------------
# Solver Tests
# ----------------------------------------------------------------------------


def test_solver_create_default():
    """Test creating a solver with default parameters."""
    solver = pytalos.Solver()
    assert solver is not None


def test_solver_create_with_preallocation():
    """Test creating a solver with pre-allocation hints."""
    solver = pytalos.Solver(num_vessels=10, num_berths=5)
    assert solver is not None


def test_solver_solve_simple(simple_model, simple_solution):
    """Test running a simple solve operation."""
    solver = pytalos.Solver()
    config = pytalos.LocalSearchConfig(
        operators=[pytalos.Operator.IntraSwap, pytalos.Operator.InterSwap],
        max_iterations=100,
    )

    result = solver.solve(
        model=simple_model,
        config=config,
        gls_config=None,
        solution=simple_solution,
    )

    assert result is not None
    assert result.solution is not None
    assert result.solution.num_vessels == 2
    assert result.iterations > 0


def test_solver_with_gls_config(simple_model, simple_solution):
    """Test running solve with GLS configuration."""
    solver = pytalos.Solver()
    config = pytalos.LocalSearchConfig(
        operators=[pytalos.Operator.IntraSwap],
        max_iterations=50,
        time_limit_secs=1.0,
    )
    gls = pytalos.GlsConfig(
        lambda_strategy=pytalos.LambdaStrategy.Dynamic,
        trigger=pytalos.Trigger.OnExhaustion,
    )

    result = solver.solve(
        model=simple_model,
        config=config,
        gls_config=gls,
        solution=simple_solution,
    )

    assert result is not None
    assert result.solution.objective >= 0


def test_solver_mismatched_solution_vessels(simple_model):
    """Test that solver rejects solution with wrong number of vessels."""
    solver = pytalos.Solver()
    config = pytalos.LocalSearchConfig(
        operators=[pytalos.Operator.IntraSwap],
        max_iterations=10,
    )

    # Create a solution with wrong number of vessels
    wrong_solution = pytalos.Solution(
        berths=[0, 1, 0],  # 3 vessels instead of 2
        start_times=[0, 2, 5],
        objective=20,
    )

    with pytest.raises(ValueError, match="solution has .* vessels but model has"):
        solver.solve(
            model=simple_model,
            config=config,
            gls_config=None,
            solution=wrong_solution,
        )


def test_solver_empty_operators(simple_model, simple_solution):
    """Test that solver rejects config with empty operators list."""
    solver = pytalos.Solver()
    config = pytalos.LocalSearchConfig(operators=[])

    with pytest.raises(ValueError, match="operators list must not be empty"):
        solver.solve(
            model=simple_model,
            config=config,
            gls_config=None,
            solution=simple_solution,
        )


def test_solver_out_of_bounds_berth(simple_model):
    """Test that solver rejects solution with out-of-bounds berth index."""
    solver = pytalos.Solver()
    config = pytalos.LocalSearchConfig(
        operators=[pytalos.Operator.IntraSwap],
        max_iterations=10,
    )

    # Create a solution with invalid berth index
    invalid_solution = pytalos.Solution(
        berths=[0, 5],  # Berth 5 doesn't exist (only 0 and 1)
        start_times=[0, 2],
        objective=20,
    )

    with pytest.raises(ValueError, match="berth index .* out of bounds"):
        solver.solve(
            model=simple_model,
            config=config,
            gls_config=None,
            solution=invalid_solution,
        )


# ----------------------------------------------------------------------------
# Utility Function Tests
# ----------------------------------------------------------------------------


def test_heuristic_gls_lambda_basic():
    """Test the heuristic_gls_lambda function with basic inputs."""
    lambda_value = pytalos.heuristic_gls_lambda(
        objective=100.0,
        num_features=10,
        scale=0.1,
    )
    assert isinstance(lambda_value, float)
    assert lambda_value > 0


def test_heuristic_gls_lambda_different_scales():
    """Test heuristic_gls_lambda with different scale values."""
    lambda1 = pytalos.heuristic_gls_lambda(100.0, 10, 0.1)
    lambda2 = pytalos.heuristic_gls_lambda(100.0, 10, 0.2)

    # Different scales should produce different lambda values
    assert lambda1 != lambda2


def test_heuristic_gls_lambda_different_objectives():
    """Test heuristic_gls_lambda with different objective values."""
    lambda1 = pytalos.heuristic_gls_lambda(100.0, 10, 0.1)
    lambda2 = pytalos.heuristic_gls_lambda(200.0, 10, 0.1)

    # Different objectives should produce different lambda values
    assert lambda1 != lambda2


# ----------------------------------------------------------------------------
# Integration Tests
# ----------------------------------------------------------------------------


def test_integration_end_to_end_simple(simple_model, simple_solution):
    """Test end-to-end solve with a simple problem."""
    solver = pytalos.Solver()

    config = pytalos.LocalSearchConfig(
        operators=[
            pytalos.Operator.IntraSwap,
            pytalos.Operator.InterSwap,
            pytalos.Operator.IntraShift,
            pytalos.Operator.InterShift,
        ],
        time_limit_secs=1.0,
        max_non_improving_iterations=500,
    )

    gls = pytalos.GlsConfig(
        lambda_strategy=pytalos.LambdaStrategy.Dynamic,
        trigger=pytalos.Trigger.OnExhaustion,
        reset_on_best=True,
    )

    result = solver.solve(
        model=simple_model,
        config=config,
        gls_config=gls,
        solution=simple_solution,
    )

    # Verify result structure
    assert result is not None
    assert result.solution is not None
    assert result.solution.num_vessels == 2
    assert len(result.solution.berths) == 2
    assert len(result.solution.start_times) == 2
    assert result.termination_reason is not None
    assert result.iterations > 0
    assert result.time_total_secs > 0


def test_integration_larger_problem(larger_model, larger_solution):
    """Test end-to-end solve with a larger problem."""
    solver = pytalos.Solver(num_vessels=4, num_berths=2)

    config = pytalos.LocalSearchConfig(
        operators=[
            pytalos.Operator.IntraSwap,
            pytalos.Operator.InterSwap,
        ],
        max_iterations=1000,
        time_limit_secs=2.0,
    )

    result = solver.solve(
        model=larger_model,
        config=config,
        gls_config=None,
        solution=larger_solution,
    )

    assert result.solution.num_vessels == 4
    assert result.iterations <= 1000


def test_integration_multiple_solves(simple_model, simple_solution):
    """Test running multiple solves with the same solver instance."""
    solver = pytalos.Solver()

    config = pytalos.LocalSearchConfig(
        operators=[pytalos.Operator.IntraSwap],
        max_iterations=50,
        time_limit_secs=1.0,
    )

    # First solve
    result1 = solver.solve(
        model=simple_model,
        config=config,
        gls_config=None,
        solution=simple_solution,
    )

    # Second solve with same solver
    result2 = solver.solve(
        model=simple_model,
        config=config,
        gls_config=None,
        solution=simple_solution,
    )

    # Both should succeed
    assert result1 is not None
    assert result2 is not None
    assert result1.solution.num_vessels == 2
    assert result2.solution.num_vessels == 2


def test_integration_with_callback(simple_model, simple_solution):
    """Test solve with a callback function."""
    solver = pytalos.Solver()
    callback_calls = []

    def on_new_best(objective, berths, start_times):
        callback_calls.append(
            {
                "objective": objective,
                "berths": berths,
                "start_times": start_times,
            }
        )

    config = pytalos.LocalSearchConfig(
        operators=[pytalos.Operator.IntraSwap, pytalos.Operator.InterSwap],
        max_iterations=100,
        time_limit_secs=1.0,
    )

    result = solver.solve(
        model=simple_model,
        config=config,
        gls_config=None,
        solution=simple_solution,
        callback=on_new_best,
    )

    assert result is not None
    # Callback may or may not be called depending on whether improvements were found
    # Just verify it doesn't crash


def test_integration_all_operators(larger_model, larger_solution):
    """Test using all available operators together."""
    solver = pytalos.Solver()

    config = pytalos.LocalSearchConfig(
        operators=[
            pytalos.Operator.IntraSwap,
            pytalos.Operator.InterSwap,
            pytalos.Operator.IntraShift,
            pytalos.Operator.InterShift,
        ],
        max_iterations=200,
    )

    result = solver.solve(
        model=larger_model,
        config=config,
        gls_config=None,
        solution=larger_solution,
    )

    assert result is not None
    assert result.iterations <= 200


def test_integration_search_result_properties(simple_model, simple_solution):
    """Test all properties of SearchResult."""
    solver = pytalos.Solver()

    config = pytalos.LocalSearchConfig(
        operators=[pytalos.Operator.IntraSwap],
        max_iterations=50,
        time_limit_secs=1.0,
    )

    result = solver.solve(
        model=simple_model,
        config=config,
        gls_config=None,
        solution=simple_solution,
    )

    # Test all result properties
    assert hasattr(result, "solution")
    assert hasattr(result, "termination_reason")
    assert hasattr(result, "iterations")
    assert hasattr(result, "accepted_solutions")
    assert hasattr(result, "total_solutions")
    assert hasattr(result, "infeasible_moves")
    assert hasattr(result, "cycles")
    assert hasattr(result, "time_total_secs")

    # Verify types
    assert isinstance(result.iterations, int)
    assert isinstance(result.accepted_solutions, int)
    assert isinstance(result.total_solutions, int)
    assert isinstance(result.infeasible_moves, int)
    assert isinstance(result.cycles, int)
    assert isinstance(result.time_total_secs, float)


def test_integration_termination_reasons(simple_model, simple_solution):
    """Test that termination reasons exist and can be accessed."""
    solver = pytalos.Solver()

    # Test with iteration limit
    config = pytalos.LocalSearchConfig(
        operators=[pytalos.Operator.IntraSwap],
        max_iterations=10,
        time_limit_secs=1.0,
    )

    result = solver.solve(
        model=simple_model,
        config=config,
        gls_config=None,
        solution=simple_solution,
    )

    # Should terminate for some reason
    assert result.termination_reason is not None

    # Verify termination reason enum values exist
    assert hasattr(pytalos.TerminationReason, "TimeLimitReached")
    assert hasattr(pytalos.TerminationReason, "IterationLimitReached")
    assert hasattr(pytalos.TerminationReason, "SolutionLimitReached")
    assert hasattr(pytalos.TerminationReason, "CycleLimitReached")


# ----------------------------------------------------------------------------
# EDF Schedule Tests
# ----------------------------------------------------------------------------


def test_edf_schedule_simple(simple_model):
    """Test EDF scheduling on a simple feasible model."""
    sol = pytalos.edf_schedule(simple_model)
    assert sol is not None
    assert isinstance(sol, pytalos.Solution)
    assert sol.num_vessels == simple_model.num_vessels
    assert len(sol.berths) == simple_model.num_vessels
    assert len(sol.start_times) == simple_model.num_vessels


def test_edf_schedule_larger(larger_model):
    """Test EDF scheduling on a larger model with some infeasible berths."""
    sol = pytalos.edf_schedule(larger_model)
    # May or may not find a feasible schedule depending on the model.
    if sol is not None:
        assert isinstance(sol, pytalos.Solution)
        assert sol.num_vessels == larger_model.num_vessels
        assert len(sol.berths) == larger_model.num_vessels
        assert len(sol.start_times) == larger_model.num_vessels


def test_edf_schedule_returns_valid_objective(simple_model):
    """Test that EDF returns a solution with a non-negative objective."""
    sol = pytalos.edf_schedule(simple_model)
    assert sol is not None
    assert sol.objective >= 0


def test_edf_schedule_respects_arrivals(simple_model):
    """Test that EDF start times are at or after vessel arrival times."""
    sol = pytalos.edf_schedule(simple_model)
    assert sol is not None
    arrivals = [0, 2]  # from simple_model fixture
    for i, start in enumerate(sol.start_times):
        assert start >= arrivals[i], (
            f"Vessel {i}: start_time {start} < arrival {arrivals[i]}"
        )
