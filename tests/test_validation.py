"""
Unit tests for the time-series cross-validation module.
"""
import numpy as np
import pandas as pd
import pytest

# Attempt to import the class, skip if it fails
try:
    from dropzilla.validation import PurgedKFold
except ImportError:
    PurgedKFold = None

@pytest.mark.skipif(PurgedKFold is None, reason="PurgedKFold class not found in dropzilla.validation")
def test_purged_kfold_instantiation_and_split():
    """Tests that PurgedKFold can be instantiated and that its split method
    generates the correct number of splits without raising an error."""
    # 1. Create mock time-series data
    n_samples = 100
    date_range = pd.date_range(start="2023-01-01", periods=n_samples, freq="D")
    X = pd.DataFrame(index=date_range, data={'feature1': np.arange(n_samples)})

    # 2. Create mock label times (e.g., labels are determined 5 days later)
    label_times = pd.Series(X.index + pd.Timedelta(days=5), index=X.index)

    # 3. Instantiate the validator
    n_splits = 5
    pkf = PurgedKFold(n_splits=n_splits, label_times=label_times, embargo_pct=0.01)

    # 4. Run the split generator
    splits = list(pkf.split(X))

    # 5. Assertions
    assert len(splits) == n_splits, f"Expected {n_splits} splits, but got {len(splits)}"
    for train_idx, test_idx in splits:
        assert isinstance(train_idx, np.ndarray)
        assert isinstance(test_idx, np.ndarray)
        assert len(np.intersect1d(train_idx, test_idx)) == 0

    print("\nPurgedKFold test passed: Instantiation and split generation are successful.")
