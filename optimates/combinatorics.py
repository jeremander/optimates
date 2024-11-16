"""This module defines classes for search problems on spaces of standard combinatorial objects (subsets, permutations, etc.)."""

from collections.abc import Iterable
from dataclasses import dataclass
import itertools
import math
import random

from optimates.search import EmptyNeighborSetError, SearchProblem


Perm = tuple[int, ...]

def num_permutations(n: int, k: int) -> int:
    """Gets the number of permutations of n of size k."""
    return math.factorial(n) // math.factorial(n - k)

def random_combo2(n: int) -> tuple[int, int]:
    """Gets a random ordered pair (i, j), where i != j, and 0 <= i, j < n."""
    i = random.randint(0, n - 1)
    j = random.randint(0, n - 2)
    if j >= i:
        j += 1
    return tuple(sorted([i, j]))  # type: ignore[return-value]


@dataclass
class SubsetSearchProblem(SearchProblem[int]):
    """A search problem defined on the set of subsets of {0, 1, ..., n - 1}.
    Encodes a subset as an n-bit unsigned integer.
    A neighbor of a subset is considered to be any subset with "a bit flipped" (i.e. either one element was added or removed from the set)."""
    n: int
    initial: int = 0  # by default, start off empty

    def initial_nodes(self) -> Iterable[int]:
        return [self.initial]

    def is_solution(self, node: int) -> bool:
        return True

    def iter_nodes(self) -> Iterable[int]:
        return range(2 ** self.n)

    def random_node(self) -> int:
        return random.randint(0, 2 ** self.n - 1)

    def get_neighbors(self, node: int) -> Iterable[int]:
        return (node ^ (1 << i) for i in range(self.n))

    def num_neighbors(self, node: int) -> int:
        return self.n

    def random_neighbor(self, node: int) -> int:
        i = random.randint(0, self.n - 1)
        return node ^ (1 << i)


@dataclass
class PermutationSearchProblem(SearchProblem[Perm]):
    """A search problem defined on the set of length-n permutations.
    A permutation is represented as an integer vector x, where x[i] = j means that i is mapped to j under the permutation.
    A neighbor of a permutation is one where a single swap has occurred (n choose 2 total).
    If adjacent_only = True, only includes neighbors where the swap is adjacent (n - 1 total)."""
    n: int
    adjacent_only: bool = False

    def initial_nodes(self) -> Iterable[Perm]:
        return [tuple(range(self.n))]

    def is_solution(self, node: Perm) -> bool:
        return True

    def iter_nodes(self) -> Iterable[Perm]:
        return itertools.permutations(range(self.n))

    def random_node(self) -> Perm:
        perm = list(range(self.n))
        random.shuffle(perm)
        return tuple(perm)

    def get_neighbors(self, node: Perm) -> Iterable[Perm]:
        if self.adjacent_only:
            pair_gen: Iterable[tuple[int, int]] = ((i, i + 1) for i in range(self.n - 1))
        else:
            pair_gen = itertools.combinations(range(self.n), 2)
        for (i, j) in pair_gen:
            nbr = list(node)
            nbr[i], nbr[j] = nbr[j], nbr[i]
            yield tuple(nbr)

    def num_neighbors(self, node: Perm) -> int:
        return (self.n - 1) if self.adjacent_only else math.comb(self.n, 2)

    def random_neighbor(self, node: Perm) -> Perm:
        if (self.n <= 1):
            raise EmptyNeighborSetError()
        if self.adjacent_only:
            i = random.randint(0, self.n - 2)
            j = i + 1
        else:
            (i, j) = random_combo2(self.n)
        nbr = list(node)
        nbr[i], nbr[j] = nbr[j], nbr[i]
        return tuple(nbr)


@dataclass
class PermutedSubsequenceSearchProblem(SearchProblem[Perm]):
    """A search problem defined on permutations of subsequences of (0, 1, ..., n - 1).
    Each node is a sequence (i_1, ..., i_k), where k = 0, ..., n, and each i_j is a distinct element in {0, 1, ..., n - 1}.
    A neighbor of a permuted sequence is a sequence that swaps one element of the original sequence with another element of either the sequence or its complement, discards one element, or inserts one element."""
    n: int

    def initial_nodes(self) -> Iterable[Perm]:
        return [()]

    def is_solution(self, node: Perm) -> bool:
        return True

    def iter_nodes(self) -> Iterable[Perm]:
        return itertools.chain.from_iterable(itertools.permutations(range(self.n), k) for k in range(self.n + 1))

    def random_node(self) -> Perm:
        # choose k in proportion to the number of permutations of that size
        num_perms = [num_permutations(self.n, k) for k in range(self.n + 1)]
        k = random.choices(range(self.n + 1), num_perms)[0]
        # get a random permutation of n elements, then take the first k
        vals = list(range(self.n))
        random.shuffle(vals)
        return tuple(vals)[:k]

    def get_complement(self, node: Perm) -> list[int]:
        """Gets the complement of a node (as a list of integers, ordered, which are not in the subsequence)."""
        node_set = set(node)
        return [i for i in range(self.n) if (i not in node_set)]

    def get_neighbors(self, node: Perm) -> Iterable[Perm]:
        k = len(node)
        complement = self.get_complement(node)
        # internal swaps
        for (i, j) in itertools.combinations(range(k), 2):
            nbr = list(node)
            nbr[i], nbr[j] = nbr[j], nbr[i]
            yield tuple(nbr)
        # external swaps
        for (i, j) in itertools.product(range(k), range(self.n - k)):
            nbr = list(node)
            nbr[i] = complement[j]
            yield tuple(nbr)
        # discards
        for i in range(k):
            yield node[:i] + node[i + 1:]
        # insertions
        for (i, j) in itertools.product(range(k + 1), range(self.n - k)):
            nbr = list(node)
            nbr.insert(i, complement[j])
            yield tuple(nbr)

    def num_neighbors(self, node: Perm) -> int:
        k = len(node)
        return math.comb(k, 2) + k * (self.n - k + 1) + (k + 1) * (self.n - k)

    def random_neighbor(self, node: Perm) -> Perm:
        k = len(node)
        num_internal_swaps = math.comb(k, 2)
        num_external_swaps = k * (self.n - k)
        num_discards = k
        num_insertions = (1 + k) * (self.n - k)
        r = random.choices(range(4), [num_internal_swaps, num_external_swaps, num_discards, num_insertions])[0]
        nbr = list(node)
        if (r == 0):  # internal swap
            (i, j) = random_combo2(k)
            nbr[i], nbr[j] = nbr[j], nbr[i]
        elif (r == 1):  # external swap
            i = random.randint(0, k - 1)
            j = random.randint(0, self.n - k - 1)
            nbr[i] = self.get_complement(node)[j]
        elif (r == 2):  # discard
            i = random.randint(0, k - 1)
            nbr = nbr[:i] + nbr[i + 1:]
        else:  # insert
            i = random.randint(0, k)
            j = random.randint(0, self.n - k - 1)
            nbr.insert(i, self.get_complement(node)[j])
        return tuple(nbr)
