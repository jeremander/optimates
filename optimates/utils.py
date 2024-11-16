"""Utility module."""

import heapq
import logging
from typing import Any, Optional, TypeVar, cast


T = TypeVar('T')


###########
# LOGGING #
###########

class Logger(logging.Logger):
    """Custom subclass of logging.Logger."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.verbosity = 0

    def level_for_verbosity(self, verbosity: int) -> int:
        """Converts a verbosity level to a logging level."""
        return logging.INFO - verbosity

    def set_verbosity(self, verbosity: int) -> None:
        """Sets the verbosity level of the logger.
        A level of 0 is the normal (INFO) logging level.
        A higher number means more verbose."""
        self.setLevel(self.level_for_verbosity(verbosity))

    def verbose(self, msg: str, level: int = 1) -> None:
        """Logs a message at the given verbosity level."""
        self.log(self.level_for_verbosity(level), msg)


LOG_FMT = '%(name)s - %(message)s'
logging.basicConfig(format=LOG_FMT, level=logging.INFO)
logging.setLoggerClass(Logger)
logger = cast(Logger, logging.getLogger('optimates'))


########
# HEAP #
########

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
