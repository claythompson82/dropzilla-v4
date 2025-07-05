"""
Handles the training, optimization, and prediction logic for Dropzilla v4 models.

This module will contain:
- The primary model (LightGBM) training function.
- The Bayesian Optimization pipeline using Hyperopt.
- The final prediction function.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd  # <-- THIS IS THE FIX
from typing import Dict, Any, Tuple
from sklearn.metrics import roc_auc_score
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

def train_lightgbm_model(X_train: np.ndarray,
                         y_train: np.ndarray,
                         params: Dict[str, Any] | None = None) -> lgb.LGBMClassifier:
    """
    Trains a LightGBM classifier with a given set of parameters.

    Args:
        X_train (np.ndarray): The training feature data.
        y_train (np.ndarray): The training target labels.
        params (Dict[str, Any] | None, optional): A dictionary of parameters to override
                                                  the defaults. Defaults to None.

    Returns:
        lgb.LGBMClassifier: The trained LightGBM model object.
    """
    default_params: Dict[str, Any] = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'n_jobs': -1,
        'is_unbalance': True,
        'verbose': -1,
    }

    if params:
        default_params.update(params)

    model = lgb.LGBMClassifier(**default_params)
    model.fit(X_train, y_train)
    return model

def optimize_hyperparameters(X: pd.DataFrame,
                             y: pd.Series,
                             cv_validator,
                             max_evals: int = 50) -> Tuple[Dict[str, Any], Trials]:
    """
    Performs Bayesian hyperparameter optimization for the LightGBM model.
    """
    search_space = {
        'n_estimators': hp.quniform('n_estimators', 100, 1000, 50),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
        'num_leaves': hp.quniform('num_leaves', 20, 150, 5),
        'max_depth': hp.quniform('max_depth', 3, 15, 1),
        'min_child_samples': hp.quniform('min_child_samples', 20, 100, 5),
        'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
        'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
    }

    def objective(params: Dict[str, Any]) -> Dict[str, Any]:
        """The objective function that hyperopt will minimize."""
        params['n_estimators'] = int(params['n_estimators'])
        params['num_leaves'] = int(params['num_leaves'])
        params['max_depth'] = int(params['max_depth'])
        params['min_child_samples'] = int(params['min_child_samples'])

        scores = []
        for train_idx, test_idx in cv_validator.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model = train_lightgbm_model(X_train.values, y_train.values, params)
            
            y_proba = model.predict_proba(X_test.values)[:, 1]
            score = roc_auc_score(y_test, y_proba)
            scores.append(score)

        avg_score = np.mean(scores)
        
        return {'loss': -avg_score, 'status': STATUS_OK, 'params': params}

    trials = Trials()
    best_params = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
        rstate=np.random.default_rng(42)
    )

    print(f"\nOptimization Complete. Best validation ROC AUC: {-trials.best_trial['result']['loss']:.4f}")
    print(f"Best parameters: {best_params}")

    return best_params, trials
