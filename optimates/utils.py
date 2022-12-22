import heapq
from typing import List, Optional, TypeVar

T = TypeVar('T')


class TopNHeap(List[T]):
    """Maintains the largest N items on a heap."""
    def __init__(self, N: Optional[int] = None):
        super().__init__()
        self.N = N
    def empty(self) -> bool:
        return (len(self) == 0)
    def top(self) -> T:
        if self.empty():
            raise ValueError("heap is empty")
        return heapq.nsmallest(1, self)[0]
    def push(self, elt: T) -> Optional[T]:
        if ((self.N is None) or (len(self) < self.N)):
            heapq.heappush(self, elt)
            return None
        else:
            return heapq.heappushpop(self, elt)
    def pop(self) -> T:  # type: ignore[override]
        return heapq.heappop(self)