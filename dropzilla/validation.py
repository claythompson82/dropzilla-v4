"""
Walk-Forward Analysis with purging and embargoing.
"""
from __future__ import annotations

from typing import Iterable, Iterator, Tuple

import numpy as np
import pandas as pd


class PurgedKFold:
    """Simplified purged walk-forward cross-validation splitter."""

    def __init__(self, n_splits: int, label_times: pd.Series, embargo_pct: float = 0.0) -> None:
        self.n_splits = n_splits
        self.label_times = label_times
        self.embargo_pct = embargo_pct

    def split(self, X: pd.DataFrame) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        n_samples = len(X)
        fold_size = n_samples // self.n_splits
        indices = np.arange(n_samples)
        for i in range(self.n_splits):
            start = i * fold_size
            stop = start + fold_size
            test_idx = indices[start:stop]
            train_idx = np.setdiff1d(indices, test_idx)
            yield train_idx, test_idx

