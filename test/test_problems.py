# test/test_problems.py

import pytest
import numpy as np
from cilpy.problem.mpb import MovingPeaksBenchmark
from cilpy.problem.cmpb import ConstrainedMovingPeaksBenchmark

# =============================================================================
# Fixtures for MovingPeaksBenchmark (MPB)
# =============================================================================

@pytest.fixture
def static_mpb():
    """
    Provides a static, predictable MovingPeaksBenchmark instance for testing.
    - One peak, static environment.
    """
    problem = MovingPeaksBenchmark(
        dimension=2,
        num_peaks=1,
        change_frequency=0
    )
    # Manually set peak properties for predictable tests
    peak = problem.peaks[0]
    peak.v = np.array([50.0, 50.0])
    peak.h = 100.0
    peak.w = 10.0
    return problem

@pytest.fixture
def dynamic_mpb():
    """
    Provides a dynamic MovingPeaksBenchmark instance.
    - Change frequency is set to a low value for easy testing.
    """
    problem = MovingPeaksBenchmark(
        dimension=2,
        num_peaks=1,
        change_frequency=5
    )
    return problem

# =============================================================================
# Tests for MovingPeaksBenchmark
# =============================================================================

def test_mpb_static_evaluation_at_peak_center(static_mpb):
    """
    Verifies that the fitness at the exact center of a peak is its height.
    """
    evaluation = static_mpb.evaluate(np.array([50.0, 50.0]))
    # MPB is maximized, but your class negates it for minimization solvers.
    # We use pytest.approx for safe floating-point comparison.
    assert -evaluation.fitness == pytest.approx(100.0)

def test_mpb_static_evaluation_off_center(static_mpb):
    """
    Verifies the fitness calculation at a point away from the peak's center.
    """
    # Distance = ||[51, 50] - [50, 50]|| = 1.0
    # Expected value = h - w * dist = 100 - 10 * 1 = 90.0
    evaluation = static_mpb.evaluate(np.array([51.0, 50.0]))
    assert -evaluation.fitness == pytest.approx(90.0)

def test_mpb_dynamic_state_change(dynamic_mpb):
    """
    Verifies that the peak's properties change after the specified
    number of evaluations.
    """
    initial_pos = dynamic_mpb.peaks[0].v.copy()
    initial_height = dynamic_mpb.peaks[0].h

    # Perform enough evaluations to trigger one change
    for _ in range(dynamic_mpb._change_frequency):
        dynamic_mpb.evaluate(np.array([0.0, 0.0]))

    new_pos = dynamic_mpb.peaks[0].v
    new_height = dynamic_mpb.peaks[0].h

    # Assert that the state has actually changed
    assert not np.array_equal(initial_pos, new_pos)
    assert initial_height != new_height

def test_mpb_boundary_reflection(static_mpb):
    """
    Verifies that a peak reflects off the boundaries correctly.
    """
    problem = static_mpb
    peak = problem.peaks[0]
    
    # Place the peak very close to the lower boundary (0,0)
    peak.v = np.array([1.0, 1.0])
    # Give it a shift vector that will push it over the boundary
    peak.s_v = np.array([-2.0, 0.0])
    
    # Manually trigger a single peak update
    peak.update(
        height_sev=0.0, width_sev=0.0, # Don't change height/width
        change_sev=np.linalg.norm(peak.s_v),
        lambda_param=1.0, # Purely correlated movement
        bounds=problem.bounds
    )
    
    # The new position should be reflected: 2*min - old_pos = 2*0 - (-1) = 1
    # The new x-position should be 0.0 - (-1.0) = 1.0
    assert peak.v[0] == pytest.approx(1.0)
    # The shift vector's x-component should be inverted
    assert peak.s_v[0] > 0

# =============================================================================
# Fixtures for ConstrainedMovingPeaksBenchmark (CMPB)
# =============================================================================

@pytest.fixture
def static_cmpb():
    """
    Provides a static, predictable ConstrainedMovingPeaksBenchmark instance.
    """
    f_params = {"dimension": 2, "num_peaks": 1, "change_frequency": 0}
    g_params = {"dimension": 2, "num_peaks": 1, "change_frequency": 0}
    problem = ConstrainedMovingPeaksBenchmark(f_params, g_params)

    # Configure f_landscape peak
    problem.f_landscape.peaks[0].v = np.array([50.0, 50.0])
    problem.f_landscape.peaks[0].h = 100.0
    
    # Configure g_landscape peak
    problem.g_landscape.peaks[0].v = np.array([50.0, 50.0])
    # We will adjust g_peak height in tests to control feasibility
    problem.g_landscape.peaks[0].h = 80.0
    
    return problem

# =============================================================================
# Tests for ConstrainedMovingPeaksBenchmark
# =============================================================================

def test_cmpb_feasible_solution(static_cmpb):
    """
    Verifies correct evaluation for a feasible solution (f(x) > g(x)).
    """
    # At [50, 50]: f(x)=100, g(x)=80. Feasible.
    # Objective (minimization): g(x) - f(x) = 80 - 100 = -20
    # Constraint violation: g(x) - f(x) = -20. (<= 0 is feasible)
    eval = static_cmpb.evaluate(np.array([50.0, 50.0]))

    assert eval.fitness == pytest.approx(-20.0)
    assert len(eval.constraints_inequality) == 1
    assert eval.constraints_inequality[0] == pytest.approx(-20.0)
    assert eval.constraints_inequality[0] <= 0 # Check feasibility

def test_cmpb_infeasible_solution(static_cmpb):
    """
    Verifies correct evaluation for an infeasible solution (g(x) > f(x)).
    """
    # Make the solution infeasible by increasing g's height
    static_cmpb.g_landscape.peaks[0].h = 120.0

    # At [50, 50]: f(x)=100, g(x)=120. Infeasible.
    # Objective (minimization): g(x) - f(x) = 120 - 100 = 20
    # Constraint violation: g(x) - f(x) = 20. (> 0 is infeasible)
    eval = static_cmpb.evaluate(np.array([50.0, 50.0]))

    assert eval.fitness == pytest.approx(20.0)
    assert len(eval.constraints_inequality) == 1
    assert eval.constraints_inequality[0] == pytest.approx(20.0)
    assert eval.constraints_inequality[0] > 0 # Check infeasibility

def test_cmpb_dynamic_interaction():
    """
    Verifies that the dynamic behavior of one landscape works independently.
    """
    f_params = {"dimension": 2, "num_peaks": 1, "change_frequency": 5}
    g_params = {"dimension": 2, "num_peaks": 1, "change_frequency": 0} # g is static
    problem = ConstrainedMovingPeaksBenchmark(f_params, g_params)
    
    f_peak = problem.f_landscape.peaks[0]
    g_peak = problem.g_landscape.peaks[0]
    
    initial_f_pos = f_peak.v.copy()
    initial_g_pos = g_peak.v.copy()
    
    # Trigger a change in the f_landscape
    for _ in range(problem.f_landscape._change_frequency):
        problem.evaluate(np.array([0.0, 0.0]))
        
    final_f_pos = f_peak.v
    final_g_pos = g_peak.v
    
    # Assert that only the dynamic landscape's peak has moved
    assert not np.array_equal(initial_f_pos, final_f_pos)
    assert np.array_equal(initial_g_pos, final_g_pos)
