"""A generic API for solving discrete search problems."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from math import exp, inf
import random
from typing import Generic, Optional, TypeVar

from optimates.utils import TopNHeap, logger


T = TypeVar('T')
IterData = tuple[T, float]  # node and score


##########
# ERRORS #
##########

class EmptyNeighborSetError(ValueError):
    """Error for when a node has no neighbors."""


##########
# SEARCH #
##########

@dataclass
class Solutions(Generic[T]):
    """Stores a set of best solutions to a search problem, and the maximum score."""
    score: Optional[float]
    solutions: set[T]

    @classmethod
    def empty(cls) -> Solutions[T]:
        """Constructs an empty set of solutions."""
        return Solutions(None, set())

    def add(self, solution: T, score: float) -> None:
        """Adds a new solution to the set, along with its score."""
        if (self.score is None) or (score > self.score):
            self.score = score
            self.solutions = {solution}
        elif score == self.score:
            self.solutions.add(solution)

    def merge(self, other: Solutions[T]) -> Solutions[T]:
        """Merges another Solutions set into this one (in-place)."""
        assert isinstance(other, Solutions)
        if other.score is not None:
            if (self.score is None) or (other.score > self.score):
                self.score = other.score
                self.solutions = other.solutions
            elif other.score == self.score:
                self.solutions |= other.solutions
        return self

    def __len__(self) -> int:
        """Gets the total number of stored solutions."""
        return len(self.solutions)


class SearchProblem(ABC, Generic[T]):
    """Generic search problem.
    This can be viewed as a directed graph of elements (nodes) to search.
    A subset of the nodes are considered "solutions."
    Each node has a score, and we wish to find a solution node with the maximum score.
    Furthermore, nodes may have directed edges to other "neighbor" nodes which are related in such a way that neighbor nodes are similar to each other in some way (hopefully in score as well).
    A variety of algorithms may be applied to try to search for an optimal solution in the search graph."""

    def score(self, node: T) -> float:
        """Scores a node of the search graph."""
        raise NotImplementedError

    @abstractmethod
    def initial_nodes(self) -> Iterable[T]:
        """Gets the set of initial nodes of the search graph (i.e. the set of nodes with no predecessor)."""

    def default_initial_node(self) -> T:
        """Gets some "canonical default" initial node of the search graph."""
        return next(iter(self.initial_nodes()))

    @abstractmethod
    def is_solution(self, node: T) -> bool:
        """Returns True if a node is a solution."""

    @abstractmethod
    def iter_nodes(self) -> Iterable[T]:
        """Gets an iterable over all nodes in the search space."""

    @abstractmethod
    def random_node(self) -> T:
        """Gets a random node in the search space."""

    @abstractmethod
    def get_neighbors(self, node: T) -> Iterable[T]:
        """Gets the neighbors of a node."""

    def num_neighbors(self, node: T) -> int:
        """Counts the number of neighbors of a node."""
        ctr = 0
        for _ in self.get_neighbors(node):
            ctr += 1
        return ctr

    def random_neighbor(self, node: T) -> T:
        """Gets a random neighbor of a node.
        By default, this will be distributed uniformly over the neighbor set.
        If no neighbors exist, raises an EmptyNeighborSetError."""
        nbrs = list(self.get_neighbors(node))
        if nbrs:
            return random.choice(nbrs)
        raise EmptyNeighborSetError(f'{node} has no neighbors')


@dataclass
class FilteredSearchProblem(SearchProblem[T]):
    """Class for a modified search problem where the search space is filtered by some predicate.
    A subclass should override the `is_element` method."""
    problem: SearchProblem[T]

    @abstractmethod
    def is_element(self, node: T) -> bool:
        """This method checks whether a node is a valid element of the search problem."""

    def score(self, node: T) -> float:
        return self.problem.score(node)

    def initial_nodes(self) -> Iterable[T]:
        return filter(self.is_element, self.problem.initial_nodes())

    def is_solution(self, node: T) -> bool:
        return self.is_element(node) and self.problem.is_solution(node)

    def iter_nodes(self) -> Iterable[T]:
        # NOTE: this can be inefficient, so you should preferably override it
        return filter(self.is_element, self.problem.iter_nodes())

    def random_node(self) -> T:
        # NOTE: this can be inefficient (or non-terminating), so you should preferably override it
        while True:
            node = self.problem.random_node()
            if self.is_element(node):
                return node

    def get_neighbors(self, node: T) -> Iterable[T]:
        return filter(self.is_element, self.problem.get_neighbors(node))

    def random_neighbor(self, node: T) -> T:
        while True:
            nbr = self.problem.random_neighbor(node)
            if self.is_element(nbr):
                return nbr


class _Search(ABC, Generic[T]):

    @abstractmethod
    def run(self, initial: Optional[T] = None) -> Solutions[T]:
        """Performs the search, starting from an initial node."""


@dataclass
class Search(_Search[T]):
    """Base class for a search algorithm.
    Starting from some initial state, it will attempt to find a global maximum, possibly using the neighborhood structure of ths earch problem."""
    problem: SearchProblem[T]

    @abstractmethod
    def _run(self, node: T) -> Solutions[T]:
        """Override this method to perform the search from a given node.
        This should return a Solutions object containing a set of best solutions and their score."""

    def run(self, initial: Optional[T] = None) -> Solutions[T]:
        """Performs the search, starting from an initial node.
        If none is provided, uses the SearchProblem's default initial node."""
        if initial is None:
            initial = self.problem.default_initial_node()
        return self._run(initial)


@dataclass
class SearchWithRestarts(_Search[T]):
    """Wraps another search problem.
    Runs that search multiple times, taking the best solution over all iterations."""
    search: Search[T]
    num_restarts: int = 25  # number of random restarts
    random_restart: bool = True  # whether to use random restarts

    def run(self, initial: Optional[T] = None) -> Solutions[T]:
        solutions: Solutions[T] = Solutions.empty()
        for t in range(self.num_restarts):
            logger.verbose(f'RESTART #{t + 1}', 1)
            node = self.search.problem.random_node() if self.random_restart else initial
            solutions.merge(self.search.run(node))
        return solutions


@dataclass
class HillClimb(Search[T]):
    """A general framework for discrete optimization which captures many well-known algorithms.
    Starting from the initial node, this will generate neighbors in some way, scoring each of them.
    Once they are scored, an acceptance criterion is applied to determine an accepted subset.
    If this subset is empty, remains at the current node; otherwise, transitions to one of the highest-scoring accepted nodes at random.
    Proceeds in this way until a stopping criterion is met (e.g. max number of iterations reached, no neighbors were accepted, etc.)."""
    max_iters: Optional[int] = None

    def reset(self) -> None:
        """Resets the state of the hill climb."""
        # maintain the set of best solutions
        self.solutions: Solutions[T] = Solutions.empty()

    def terminate_early(self) -> bool:
        """Returns True if the algorithm should terminate when no acceptable neighbors are found."""
        return True

    def get_neighbors(self, node: T) -> Iterable[T]:
        """Gets the set of neighbors for a node.
        By default, it will simply get the neighborhood from the underlying problem, but the search algorithm is free to modify this in some way."""
        return self.problem.get_neighbors(node)

    @abstractmethod
    def accept(self, cur_score: float, nbr_score: float) -> bool:
        """Given the current node's score and a neighbor node's score, returns True if the neighbor is accepted."""

    def iterate_search(self, initial: T) -> tuple[Solutions[T], list[IterData[T]]]:
        """Runs the optimization, returning a pair (best solutions, node sequence)."""
        self.reset()
        prob, solns = self.problem, self.solutions
        max_iters = inf if (self.max_iters is None) else self.max_iters
        cur_node = initial
        cur_score = prob.score(initial)
        pairs = [(cur_node, cur_score)]
        if prob.is_solution(cur_node):
            solns.add(cur_node, cur_score)
        t = 1
        while t <= max_iters:
            logger.verbose(f'\tIteration #{t}', 1)
            logger.verbose(f'\t\tcurrent node = {cur_node}, score = {cur_score}', 2)
            # store highest-scoring neighbors that are accepted
            local_solns: Solutions[T] = Solutions.empty()
            num_nbrs, num_accepted = 0, 0
            for nbr in self.get_neighbors(cur_node):
                num_nbrs += 1
                nbr_score = prob.score(nbr)
                if prob.is_solution(nbr):
                    solns.add(nbr, nbr_score)
                if self.accept(cur_score, nbr_score):
                    num_accepted += 1
                    local_solns.add(nbr, nbr_score)
            num_best_accepted = len(local_solns)
            logger.verbose(f'\t\tnum_neighbors = {num_nbrs}, num_accepted = {num_accepted}, num_best accepted = {num_best_accepted}', 3)
            if num_best_accepted == 0:
                if self.terminate_early():
                    logger.verbose(f'\tNo neighbors accepted: terminating at iteration #{t}.', 1)
                    break
                # otherwise, remain at the current node
            else:
                # choose randomly from the set of best accepted solutions
                cur_node = random.choice(list(local_solns.solutions))
                cur_score = local_solns.score  # type: ignore[assignment]
            pairs.append((cur_node, cur_score))
            t += 1
        else:
            logger.verbose(f'\tTerminating after max_iters ({max_iters}) iterations reached.', 1)
            logger.verbose(f'\t\tcurrent node = {cur_node}, score = {cur_score}', 1)
        return (solns, pairs)

    def _run(self, node: T) -> Solutions[T]:
        (solns, _) = self.iterate_search(node)
        return solns


class ExhaustiveSearch(HillClimb[T]):
    """An exhaustive search checks every node in the search space."""

    def reset(self) -> None:
        super().reset()
        # create a generator over all nodes in the search space
        self._node_gen = iter(self.problem.iter_nodes())

    def get_neighbors(self, node: T) -> Iterable[T]:
        # retrieve the next node from the stored generator
        while True:
            try:
                nbr = next(self._node_gen)
                if nbr != node:
                    return [nbr]
            except StopIteration:
                return []

    def accept(self, cur_score: float, nbr_score: float) -> bool:
        return True


class BlindRandomSearch(HillClimb[T]):
    """A blind random search randomly chooses a new node to search at each step."""

    def get_neighbors(self, node: T) -> Iterable[T]:
        return [self.problem.random_node()]

    def accept(self, cur_score: float, nbr_score: float) -> bool:
        return True


@dataclass
class StochasticLocalSearch(HillClimb[T]):
    """A stochastic local search randomly chooses a neighbor node at each step.
    A neighbor is accepted if its score is at least that of the current node.
    If strict_improvement = True, requires that the score be strictly higher."""
    strict_improvement: bool = False

    def get_neighbors(self, node: T) -> Iterable[T]:
        return [self.problem.random_neighbor(node)]

    def accept(self, cur_score: float, nbr_score: float) -> bool:
        if self.strict_improvement:
            return nbr_score > cur_score
        return nbr_score >= cur_score

    def terminate_early(self) -> bool:
        return False


class GreedyLocalSearch(HillClimb[T]):
    """A greedy local search selects the best-scoring neighbor from among the set of neighbors.
    If there is a tie, chooses one at random."""

    def accept(self, cur_score: float, nbr_score: float) -> bool:
        return nbr_score > cur_score


class SimulatedAnnealing(HillClimb[T]):
    """Simulated annealing attempts to find a global maximum by starting off in a more stochastic phase, allowing balances starts at a temperature T0, then gradually cools off the temperature via some exponential decay schedule."""
    T0: float = 1.0  # initial temperature
    decay: float = 0.99  # exponential decay coefficient (higher means mo

    def __post_init__(self) -> None:
        assert (self.decay > 0.0) and (self.decay < 1.0), 'temperature decay coefficient must be in (0, 1)'  # noqa: PT018

    def reset(self) -> None:
        super().reset()
        self.T = self.T0

    def get_neighbors(self, node: T) -> Iterable[T]:
        return [self.problem.random_neighbor(node)]

    def accept(self, cur_score: float, nbr_score: float) -> bool:
        delta = nbr_score - cur_score
        logger.verbose(f'\t\tcurrent temperature = {self.T}', 3)
        logger.verbose(f'\t\tneighbor score = {nbr_score}, delta = {delta}', 3)
        if delta > 0:  # accept any improvement
            acc = True
            logger.verbose('\t\tscore increased', 2)
        else:  # accept a worse solution with some probability (likelier with high temperature)
            p = exp(delta / self.T)
            acc = random.random() < p
            logger.verbose('\t\tscore decreased', 3)
            logger.verbose(f'\t\tacceptance probability = {p}', 3)
        logger.verbose('\t\t' + ('accepted' if acc else 'rejected') + ' neighbor', 3)
        # decay the temperature
        self.T *= self.decay
        return acc

    def terminate_early(self) -> bool:
        return False


class ExhaustiveDFS(Search[T]):
    """An exhaustive depth-first search.
    This employs dynamic programming: starting from the initial node, recursively finds best solutions from each descendant node.
    NOTE: this may fail to terminate if the search graph has cycles, and it will visit nodes repeatedly if the graph is not a tree."""

    def _run(self, node: T) -> Solutions[T]:
        score = self.problem.score(node)
        if self.problem.is_solution(node):
            solns = Solutions(score, {node})
        else:
            solns = Solutions.empty()
        # compute solutions for each neighbor
        for nbr in self.problem.get_neighbors(node):
            solns.merge(self._run(nbr))
        return solns


class BeamSearch(Search[T]):
    """A beam search."""
    beam_width: int = 10

    def _run(self, node: T) -> Solutions[T]:
        score = self.problem.score(node)
        if self.problem.is_solution(node):
            solns = Solutions(score, {node})
        else:
            solns = Solutions.empty()
        # compute heap of top-scoring neighbors (not just the best ones)
        best_nbrs: TopNHeap[tuple[float, T]] = TopNHeap(N=self.beam_width)
        for nbr in self.problem.get_neighbors(node):
            score = self.problem.score(nbr)
            best_nbrs.push((score, nbr))
        # for the top-scoring neighbors, compute their best solutions, then merge
        for (_, nbr) in best_nbrs:
            solns.merge(self._run(nbr))
        return solns
