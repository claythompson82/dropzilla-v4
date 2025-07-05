""""
Handles advanced signal processing, including confidence modeling and filtering.
"""
import numpy as np
from pykalman import KalmanFilter


def smooth_probabilities_kalman(probabilities: np.ndarray) -> np.ndarray:
    """Applies a Kalman Filter to a time series of probabilities to smooth them.

    The Kalman Filter estimates the 'true' underlying conviction by treating the
    raw model probabilities as noisy measurements.

    Args:
        probabilities (np.ndarray): A 1D numpy array of raw probability scores.

    Returns:
        np.ndarray: A 1D numpy array of smoothed probability scores.
    """

    if probabilities is None or len(probabilities) == 0:
        return np.array([])

    # Configure the Kalman Filter. These parameters are chosen to create a
    # simple, stable filter that tracks the input signal.
    # transition_matrices: How the state is expected to change (we assume it's stable).
    # observation_matrices: How the measurement relates to the state.
    # process_covariance: The noise in the process itself (a small value).
    # observation_covariance: The noise in our measurement (the model's probability).
    kf = KalmanFilter(
        transition_matrices=[1],
        observation_matrices=[1],
        initial_state_mean=probabilities[0],
        initial_state_covariance=1,
        transition_covariance=0.01,
        observation_covariance=0.1,
    )

    # Apply the filter to the data
    smoothed_states, _ = kf.filter(probabilities)

    # The output is the smoothed estimate of the underlying conviction
    return smoothed_states.flatten()

