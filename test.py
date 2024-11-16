"""Unit tests for the optimates library."""

from dataclasses import dataclass
import random
from typing import Iterable

import pytest

from optimates.search import BlindRandomSearch, EmptyNeighborSetError, ExhaustiveSearch, GreedyLocalSearch, HillClimb, SearchProblem, SimulatedAnnealing, StochasticLocalSearch


@dataclass
class RangeProblem(SearchProblem[int]):
    """Trivial search space where i = {0, ..., n - 1} and score(i) = i."""
    n: int

    def score(self, node: int) -> float:
        return float(node)

    def initial_nodes(self) -> Iterable[int]:
        return [0]

    def is_solution(self, node: int) -> bool:
        return True

    def iter_nodes(self) -> Iterable[int]:
        return range(self.n)

    def random_node(self) -> int:
        return random.choice(range(self.n))

    def get_neighbors(self, node: int) -> Iterable[int]:
        nbrs = []
        if node > 0:
            nbrs.append(node - 1)
        if node < self.n - 1:
            nbrs.append(node + 1)
        return nbrs

    def random_neighbor(self, node: int) -> int:
        nbrs = self.get_neighbors(node)
        if nbrs:
            return random.choice(list(nbrs))
        raise EmptyNeighborSetError()


class ReverseRangeProblem(RangeProblem):
    """Trivial search space where i = {0, ..., n - 1} and score(i) = -i."""

    def score(self, node: int) -> float:
        return -float(node)

    def initial_nodes(self) -> Iterable[int]:
        return [5]


range_problem = RangeProblem(10)
rev_range_problem = ReverseRangeProblem(10)

TESTS = [
    (0, ExhaustiveSearch(range_problem, max_iters = None),
        {'num_steps' : 10, 'monotonic' : True, 'score' : 9, 'solutions' : {9}}),
    (0, BlindRandomSearch(range_problem, max_iters = 5),
        {'num_steps' : 6, 'monotonic' : False, 'score' : 8, 'solutions' : {8}}),
    (0, StochasticLocalSearch(range_problem, max_iters = 20),
        {'num_steps' : 21, 'monotonic' : True, 'score' : 9, 'solutions' : {9}}),
    (0, StochasticLocalSearch(range_problem, max_iters = 10),
        {'num_steps' : 11, 'monotonic' : True, 'score' : 6, 'solutions' : {6}}),
    (0, GreedyLocalSearch(range_problem, max_iters = None),
        {'num_steps' : 10, 'monotonic' : True, 'score' : 9, 'solutions' : {9}}),
    (0, SimulatedAnnealing(range_problem, max_iters = 10),
        {'num_steps' : 11, 'monotonic' : False, 'score' : 5, 'solutions' : {5}}),
    (0, ExhaustiveSearch(rev_range_problem, max_iters = None),
        {'num_steps' : 11, 'monotonic' : False, 'score' : 0, 'solutions' : {0}}),
    (0, BlindRandomSearch(rev_range_problem, max_iters = 10),
        {'num_steps' : 11, 'monotonic' : False, 'score' : 0, 'solutions' : {0}}),
    (0, StochasticLocalSearch(rev_range_problem, max_iters = 20),
        {'num_steps' : 21, 'monotonic' : True, 'score' : 0, 'solutions' : {0}}),
    (0, StochasticLocalSearch(rev_range_problem, max_iters = 10),
        {'num_steps' : 11, 'monotonic' : True, 'score' : -3, 'solutions' : {3}}),
    (0, GreedyLocalSearch(rev_range_problem, max_iters = None),
        {'num_steps' : 6, 'monotonic' : True, 'score' : 0, 'solutions' : {0}}),
    (0, SimulatedAnnealing(rev_range_problem, max_iters = 10),
        {'num_steps' : 11, 'monotonic' : False, 'score' : -4, 'solutions' : {4}}),
    (0, SimulatedAnnealing(rev_range_problem, max_iters = 22),
        {'num_steps' : 23, 'monotonic' : False, 'score' : 0, 'solutions' : {0}}),
]

@pytest.mark.parametrize(['seed', 'search_obj', 'result'], TESTS)
def test_search(seed, search_obj, result):
    """Tests the expected result for various search problems."""
    random.seed(seed)
    if isinstance(search_obj, HillClimb):
        initial = search_obj.problem.default_initial_node()
        (res, steps) = search_obj.iterate_search(initial)
        if 'num_steps' in result:
            assert (len(steps) == result['num_steps'])
        if 'monotonic' in result:  # check monotonicity
            monotonic = result['monotonic']
            is_mon = True
            for i in range(len(steps) - 1):
                if monotonic:  # score sequence must be non-decreasing
                    assert (steps[i][1] <= steps[i + 1][1])
                else:
                    is_mon &= (steps[i][1] <= steps[i + 1][1])
            # if expecting not monotonic, the sequence must be non-monotonic
            assert monotonic or (not is_mon)
    assert result['score'] == res.score
    assert result['solutions'] == res.solutions
