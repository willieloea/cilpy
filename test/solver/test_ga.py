# test/solver/test_ga.py
"""
Unit tests for the Genetic Algorithm (GA) solvers.

This suite tests the canonical GA and its variants (RIGA, HyperMGA).
It uses a mocked problem to control the fitness landscape and test the
solver's internal mechanisms (selection, crossover, mutation, elitism, etc.)
in isolation.
"""
import pytest
import random
from unittest.mock import MagicMock, patch

from cilpy.problem import Problem, Evaluation
from cilpy.solver.ga import GA, RIGA, HyperMGA
from cilpy.solver.chm import DefaultComparator

# --- Fixtures and Mocks ---

@pytest.fixture
def mock_problem() -> MagicMock:
    """
    Creates a mock problem that can be used to test solvers.
    - The `evaluate` method is a MagicMock, so we can control its return value.
    - It has a fixed dimension and bounds.
    """
    problem = MagicMock(spec=Problem)
    problem.dimension = 4
    problem.bounds = ([-10.0] * 4, [10.0] * 4)
    # Default behavior: fitness is the sum of the solution's values
    problem.evaluate.side_effect = lambda sol: Evaluation(fitness=sum(sol))
    return problem


# --- Tests for the base GA class ---

class TestGA:
    """Tests the canonical Genetic Algorithm."""

    def test_initialization(self, mock_problem):
        """Tests if the GA initializes with the correct population size and properties."""
        solver = GA(
            problem=mock_problem,
            name="TestGA",
            population_size=50,
            crossover_rate=0.8,
            mutation_rate=0.1
        )
        assert solver.population_size == 50
        assert len(solver.population) == 50
        assert len(solver.evaluations) == 50
        # Check if an individual is within bounds
        for gene in solver.population[0]:
            assert mock_problem.bounds[0][0] <= gene <= mock_problem.bounds[1][0]

    def test_selection(self, mock_problem):
        """
        Tests if tournament selection correctly favors individuals with better fitness.
        """
        solver = GA(mock_problem, "TestGA", 10, 0.8, 0.1, tournament_size=2)

        # Create a predictable population and evaluations
        # Individual 3 has the best (lowest) fitness
        solver.population = [[i]*4 for i in range(10)] # pop[3] = [3,3,3,3]
        solver.evaluations = [Evaluation(fitness=i*4) for i in range(10)] # eval[3].fitness = 12

        # Rig the tournament to always include the best individual (index 3)
        # This guarantees it should be selected most often.
        with patch('random.sample', return_value=[3, 9]): # Tournament between best and worst
            parents = solver._selection()
            # The winner of this tournament should always be individual 3
            assert all(parent == [3, 3, 3, 3] for parent in parents)

    def test_reproduction(self, mock_problem):
        """Tests single-point crossover."""
        solver = GA(mock_problem, "TestGA", 2, 1.0, 0.0) # Crossover rate = 100%
        parents = [
            [1.0, 1.0, 1.0, 1.0],
            [9.0, 9.0, 9.0, 9.0]
        ]
        
        # Force the crossover point to be 2
        with patch('random.randint', return_value=2):
            offspring = solver._reproduction(parents)
            assert offspring[0] == [1.0, 1.0, 9.0, 9.0]
            assert offspring[1] == [9.0, 9.0, 1.0, 1.0]

    def test_mutation(self, mock_problem):
        """Tests that mutation modifies individuals and clamps them to bounds."""
        solver = GA(mock_problem, "TestGA", 2, 0.0, 1.0) # Mutation rate = 100%
        offspring = [[0.0] * 4, [0.0] * 4]
        
        # Mock gauss to return a large value that will go out of bounds
        with patch('random.gauss', return_value=100.0):
            mutated = solver._mutation(offspring)
            # All genes should be clamped to the upper bound of 10.0
            assert all(gene == 10.0 for gene in mutated[0])
            assert all(gene == 10.0 for gene in mutated[1])

    def test_elitism_in_step(self, mock_problem):
        """
        Tests that the `step` method correctly preserves the best individual
        from the previous generation (elitism).
        """
        solver = GA(mock_problem, "TestGA", 10, 1.0, 0.0)

        # Create a population where one individual is clearly the best
        best_individual = [-5.0] * 4
        solver.population = [[10.0]*4] * 9 + [best_individual]
        solver.evaluations = [Evaluation(fitness=40.0)]*9 + [Evaluation(fitness=-20.0)]
        
        # Mock the operators to produce only terrible offspring
        terrible_offspring = [[100.0] * 4] * 10
        with patch.object(solver, '_selection', return_value=solver.population), \
             patch.object(solver, '_reproduction', return_value=terrible_offspring), \
             patch.object(solver, '_mutation', return_value=terrible_offspring):
            
            # The problem will evaluate the terrible offspring to have high fitness
            mock_problem.evaluate.return_value = Evaluation(fitness=400.0)

            solver.step()

            # Despite all new offspring being terrible, the elite individual
            # from the previous generation should have been preserved.
            assert best_individual in solver.population


# --- Tests for the RIGA class ---

class TestRIGA:
    """Tests the Random Immigrants Genetic Algorithm."""

    def test_step_replaces_worst_with_immigrants(self, mock_problem):
        """
        Tests that RIGA's step correctly identifies the worst individuals and
        replaces them with new random immigrants.
        """
        solver = RIGA(mock_problem, "TestRIGA", 10, 0.8, 0.1, immigrant_rate=0.2)
        num_immigrants = 2

        # Create a clearly ranked population
        # Individuals with higher values are worse
        solver.population = [[i]*4 for i in range(10)]
        solver.evaluations = [Evaluation(fitness=i*4) for i in range(10)]
        
        worst_individuals = [[8]*4, [9]*4]
        assert worst_individuals[0] in solver.population
        assert worst_individuals[1] in solver.population

        # Generate predictable immigrants
        new_immigrants = [[-100.0]*4, [-200.0]*4]
        with patch.object(solver, '_generate_immigrants', return_value=new_immigrants):
            solver.step()

            # Assert that the two worst individuals have been replaced
            assert worst_individuals[0] not in solver.population
            assert worst_individuals[1] not in solver.population

            # Assert that the new immigrants are now in the population
            assert new_immigrants[0] in solver.population
            assert new_immigrants[1] in solver.population

# --- Tests for the HyperMGA class ---

class TestHyperMGA:
    """Tests the Hyper-mutation Genetic Algorithm."""

    def test_step_uses_normal_mutation_when_stable(self, mock_problem):
        """
        Tests that HyperMGA uses the standard mutation operator when the
        environment is stable or improving.
        """
        # We patch the methods on the CLASS, not the instance.
        with patch.object(GA, '_mutation', autospec=True) as mock_normal_mutation, \
             patch.object(HyperMGA, '_hyper_mutation', autospec=True) as mock_hyper_mutation:

            # Configure the mocks to return the list they receive. This simulates
            # a passthrough, ensuring that the list of offspring is not empty.
            mock_normal_mutation.side_effect = lambda self, offspring: offspring
            mock_hyper_mutation.side_effect = lambda self, offspring: offspring

            # Create the instance *inside* the patch context manager.
            # This ensures __init__ sees the mocked methods.
            solver = HyperMGA(mock_problem, "HyperMGA", 10, 0.8, 0.1, 0.9, 5)
            
            # The mock should be called to evaluate the initial population
            # during initialization, so we reset it to test the `step` call cleanly.
            mock_problem.evaluate.reset_mock()

            # First step initializes f_best
            solver.step()
            
            # Second step, environment is stable
            solver.step()

            # The standard mutation method should be called, hyper should not
            assert mock_normal_mutation.called
            assert not mock_hyper_mutation.called

            # Assert that the current mutation operator is the normal one.
            assert solver.m_current == solver._mutation
            assert solver.m_current != solver._hyper_mutation


    def test_step_switches_to_hyper_mutation_on_change(self, mock_problem):
        """
        Tests that HyperMGA switches to the hyper-mutation operator when
        the best fitness degrades (environment changes).
        """
        solver = HyperMGA(mock_problem, "HyperMGA", 10, 0.8, 0.1, 0.9, 5)

        # First step, establish a good f_best
        solver.f_best = -100.0

        # Simulate an environmental change by making all fitness values worse
        mock_problem.evaluate.side_effect = lambda sol: Evaluation(fitness=sum(sol) + 200.0)

        solver.step()

        # The solver should detect the degradation and switch its operator
        assert solver.m_current == solver._hyper_mutation
        assert solver.hyper_count == 1

    def test_step_switches_back_from_hyper_mutation(self, mock_problem):
        """
        Tests that HyperMGA switches back to normal mutation after the
        hyper_total counter is exceeded.
        """
        hyper_period = 3
        solver = HyperMGA(mock_problem, "HyperMGA", 10, 0.8, 0.1, 0.9, hyper_period)

        # Manually trigger hyper-mutation mode
        solver.m_current = solver._hyper_mutation
        solver.hyper_count = hyper_period # Set counter to the limit

        # The next step should still use hyper-mutation
        solver.step()
        assert solver.m_current == solver._hyper_mutation
        assert solver.hyper_count == hyper_period + 1

        # The step *after* that should switch back to normal
        solver.step()
        assert solver.m_current == solver._mutation
        assert solver.hyper_count == 0
