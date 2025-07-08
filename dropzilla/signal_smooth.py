import numpy as np
from pykalman import KalmanFilter


def smooth_probabilities(raw: np.ndarray, obs_cov: float = 0.25) -> np.ndarray:
    """1-D Kalman smoother for probability series.

    Parameters
    ----------
    raw : array-like, shape (n_samples,)
        Unsmoothed 0-1 probabilities.
    obs_cov : float
        Observation covariance; lower ⇒ tighter smoothing.

    Returns
    -------
    np.ndarray shape (n_samples,)
    """
    kf = KalmanFilter(
        transition_matrices=[1],
        observation_matrices=[1],
        transition_covariance=0.01,
        observation_covariance=obs_cov,
        initial_state_mean=raw[0],
    )
    smoothed, _ = kf.smooth(raw.reshape(-1, 1))
    return smoothed.ravel().clip(0, 1)
