# TODO

- Auto-tuned SAN
- beam search
    - exhaustive DFS as special case (explore all neighbors, i.e. beam width infinity)
    - greedy search as special case (only explore best neighbor, with random tiebreak, i.e. beam width 1)
- params to limit depth for the above two searches
- Dynamic programming (see `old/dp.py`)
- more unit tests
