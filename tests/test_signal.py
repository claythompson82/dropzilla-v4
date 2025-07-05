"""Unit tests for the signal processing module."""
import numpy as np

from dropzilla.signal import smooth_probabilities_kalman


def test_smooth_probabilities_kalman_basic():
    probabilities = np.array([0.1, 0.5, 0.9, 0.3, 0.6])
    result = smooth_probabilities_kalman(probabilities)
    expected = np.array([0.1, 0.30090498, 0.52609432, 0.45319263, 0.49679182])
    assert result.shape == probabilities.shape
    assert np.allclose(result, expected)
