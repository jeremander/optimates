"""Utility module."""

import heapq
from typing import Optional, TypeVar


T = TypeVar('T')


class TopNHeap(list[T]):
    """Maintains the largest N elements on a heap."""

    def __init__(self, N: Optional[int] = None) -> None:
        super().__init__()
        self.N = N

    def empty(self) -> bool:
        """Returns True if the heap is empty."""
        return (len(self) == 0)

    def top(self) -> T:
        """Returns the top (minimum) element on the heap."""
        if self.empty():
            raise ValueError("heap is empty")
        return heapq.nsmallest(1, self)[0]

    def push(self, elt: T) -> Optional[T]:
        """Pushes a new element onto the heap.
        Returns the element that was removed, if one exists."""
        if (self.N is None) or (len(self) < self.N):
            heapq.heappush(self, elt)
            return None
        return heapq.heappushpop(self, elt)

    def pop(self) -> T:  # type: ignore[override]
        """Pops off the smallest element from the heap and returns it."""
        return heapq.heappop(self)
