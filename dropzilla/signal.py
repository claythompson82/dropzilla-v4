"""
Handles the meta-model training and conviction score generation.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import pandas as pd
import numpy as np
from pykalman import KalmanFilter

def train_meta_model(meta_dataset: pd.DataFrame) -> CalibratedClassifierCV:
    """
    Trains and calibrates the secondary meta-model.

    Args:
        meta_dataset (pd.DataFrame): The dataset of candidate signals from the primary model.

    Returns:
        CalibratedClassifierCV: The trained and calibrated conviction model.
    """
    print("Training conviction meta-model...")

    # Define features and target for the meta-model
    meta_features_to_use = [
        'primary_model_probability',
        'relative_volume',
        'market_regime',
        'model_uncertainty',
        'sar_score',
        'vra_score'
    ]
    # Ensure all required columns exist
    for col in meta_features_to_use:
        if col not in meta_dataset.columns:
            raise ValueError(f"Meta-dataset is missing required feature: {col}")

    X_meta = meta_dataset[meta_features_to_use]
    y_meta = meta_dataset['meta_target']

    print(f"Meta-model training data shape: {X_meta.shape}")
    print(f"Meta-model target distribution:\n{y_meta.value_counts(normalize=True)}")

    # --- THE FIX: Handle class imbalance in the meta-model ---
    # The base estimator is a Logistic Regression that now balances class weights.
    lr = LogisticRegression(class_weight='balanced', random_state=42)
    # --- END FIX ---

    # We still calibrate this model to ensure its probabilities are reliable.
    calibrated_meta_model = CalibratedClassifierCV(
        estimator=lr,
        method='isotonic', # Isotonic is fine for the simpler meta-model
        cv=3
    )

    calibrated_meta_model.fit(X_meta, y_meta)

    return calibrated_meta_model


def smooth_probabilities_kalman(
    prob_series,
    transition_var: float = 1e-5,
    observation_var: float = 1e-2,
) -> np.ndarray:
    """Smooth probability estimates using a simple Kalman filter.

    Parameters
    ----------
    prob_series : array-like
        Sequence of probability values to smooth.
    transition_var : float, optional
        Variance of the state transition noise.
    observation_var : float, optional
        Variance of the observation noise.

    Returns
    -------
    np.ndarray
        Smoothed probability values with the same length as the input.
    """
    probs = np.asarray(prob_series, dtype=float)
    if probs.ndim != 1:
        probs = probs.ravel()

    kf = KalmanFilter(
        transition_matrices=1.0,
        observation_matrices=1.0,
        transition_covariance=transition_var * 1000,
        observation_covariance=observation_var * 10,
        initial_state_mean=probs[0],
        initial_state_covariance=1.0,
    )

    state_means, _ = kf.filter(probs)
    return state_means.ravel()
