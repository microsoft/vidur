import hashlib
import os
import pickle
from abc import abstractmethod
from itertools import product
from typing import Any, Dict, List, Tuple
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from vidur.config import (
    HybridForestExecutionTimePredictorConfig,
    BaseReplicaSchedulerConfig,
    MetricsConfig,
    ReplicaConfig,
)
from vidur.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
)

from vidur.logger import init_logger

logger = init_logger(__name__)

class HybridForestLinearRegressor(BaseEstimator, RegressorMixin):
    """
    A hybrid model that uses RandomForestRegressor for interpolation within the training
    data range and LinearRegression for extrapolation beyond specified boundaries.
    
    Parameters
    ----------
    boundary_method : str, default='auto'
        Method to determine boundaries:
        - 'auto': Use min/max values from training data
        - 'percentile': Use percentiles of training data
        - 'manual': Use manually specified boundaries
    
    lower_boundary : float or None, default=None
        Lower boundary for switching to linear regression.
        Required if boundary_method='manual'.
    
    upper_boundary : float or None, default=None
        Upper boundary for switching to linear regression.
        Required if boundary_method='manual'.
    
    percentile_range : tuple, default=(1, 99)
        Percentile range to use if boundary_method='percentile'.
    
    n_estimators : int, default=100
        Number of trees in the RandomForestRegressor.
    
    max_depth : int or None, default=None
        Maximum depth of the trees in the RandomForestRegressor.
    
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node in the RandomForestRegressor.
    
    lr_params : dict, default=None
        Parameters to pass to LinearRegression.
    
    feature_idx : int, default=0
        Index of the feature to use for boundary condition when X is multidimensional.

    polynomial_degree : int, default=2
        Degree of the polynomial features.
        
    polynomial_include_bias : bool, default=True
        Whether to include a bias column in the polynomial features.
    
    polynomial_interaction_only : bool, default=True
        Whether to include only interaction features in the polynomial features.
    
    fit_intercept : bool, default=True
        Whether to fit an intercept in the LinearRegression model.
 
    """
    fit_intercept = True
    # add all parameters
    boundary_method = 'auto'
    lower_boundary = None
    upper_boundary = None
    percentile_range = (1, 99)
    n_estimators = None
    max_depth = None
    min_samples_split = 100
    lr_params = None
    feature_idx = 0
    polynomial_degree = 2
    polynomial_include_bias = True
    polynomial_interaction_only = True
    fit_intercept = True
    
    
    def __init__(self, boundary_method='auto', lower_boundary=None, upper_boundary=None,
                 percentile_range=(1, 99), n_estimators=None, max_depth=None, 
                 min_samples_split=None, feature_idx=0, polynomial_degree=2,
                 polynomial_include_bias=True, polynomial_interaction_only=True, fit_intercept=True):
        
        self.boundary_method = boundary_method
        self.lower_boundary = lower_boundary
        self.upper_boundary = upper_boundary
        self.percentile_range = percentile_range
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        
        self.feature_idx = feature_idx
        self.polynomial_degree = polynomial_degree
        self.polynomial_include_bias = polynomial_include_bias
        self.polynomial_interaction_only = polynomial_interaction_only
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        """
        Fit the hybrid model to training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input samples.
        y : array-like of shape (n_samples,)
            Target values.
        
        Returns
        -------
        self : object
            Returns self.
        """
        # Initialize models
        self.rf_model_ = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split
        )
        

        self.poly_params =                                                       {
            'degree': self.polynomial_degree,
            'include_bias': self.polynomial_include_bias,
            'interaction_only': self.polynomial_interaction_only
        }
        self.lr_params = {
            'fit_intercept': self.fit_intercept
        }
        
        self.lr_model_lower_ = make_pipeline(PolynomialFeatures(**self.poly_params),
                                             LinearRegression(**self.lr_params))
        
        self.lr_model_upper_ = make_pipeline(PolynomialFeatures(**self.poly_params),
                                             LinearRegression(**self.lr_params))

        if isinstance(X, np.ndarray):
            X_feature = X[:, self.feature_idx]

        else:
            col = X.columns[self.feature_idx]
            X_feature = X[col]
        
        if self.boundary_method == 'auto':
            self.lower_bound_ = np.min(X_feature)
            self.upper_bound_ = np.max(X_feature)
        elif self.boundary_method == 'percentile':
            self.lower_bound_ = np.percentile(X_feature, self.percentile_range[0])
            self.upper_bound_ = np.percentile(X_feature, self.percentile_range[1])
        elif self.boundary_method == 'manual':
            if self.lower_boundary is None or self.upper_boundary is None:
                raise ValueError("When boundary_method='manual', both lower_boundary and "
                                 "upper_boundary must be provided.")
            self.lower_bound_ = self.lower_boundary
            self.upper_bound_ = self.upper_boundary
        else:
            raise ValueError(f"Unknown boundary_method: {self.boundary_method}")
        
        # Fit Random Forest on all data
        self.rf_model_.fit(X, y)
        
        # For lower boundary extrapolation
        if isinstance(X, np.ndarray):
            mask_lower = X_feature <= np.percentile(X_feature, 20)  # Use bottom 20% of data
            X_lower = X[mask_lower]
            y_lower = y[mask_lower]
            # get all data, not just the feature column of X
            
            # self.lr_model_lower_ = LinearRegression(**self.lr_params)
            self.lr_model_lower_.fit(X_lower, y_lower)
            
            # For upper boundary extrapolation
            mask_upper = X_feature >= np.percentile(X_feature, 60)  # Use top 20% of data
            X_upper = X[mask_upper]
            y_upper = y[mask_upper]
            self.lr_model_upper_.fit(X_upper, y_upper)
            
        # handle for dataframe X
            
        else:
            mask_lower = X_feature <= np.percentile(X_feature, 20)  # Use bottom 20% of data
            X_lower = X[mask_lower]
            y_lower = y[mask_lower.to_list()]
            self.lr_model_lower_.fit(X_lower, y_lower)
            
            # For upper boundary extrapolation
            mask_upper = X_feature >= np.percentile(X_feature, 60)  # Use top 20% of data
            X_upper = X[mask_upper]
            y_upper = y[mask_upper.to_list()]
            self.lr_model_upper_.fit(X_upper, y_upper)
        return self


    def predict(self, X):
        """
        Predict using the hybrid model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.
            
        Returns
        -------
        y : array-like of shape (n_samples,)
            Returns predicted values.
        """
        # Check if the model has been fitted
        if not hasattr(self, 'rf_model_'):
            raise ValueError("This model has not been fitted yet. Call 'fit' before using 'predict'.")
        
        # Get the feature used for boundary condition
        if isinstance(X, np.ndarray):
            X_feature = X[:, self.feature_idx]
        else:
            X_feature = X.iloc[:, self.feature_idx]
        
        # Initialize prediction array
        y_pred = np.zeros(X_feature.shape[0])
        # Identify which samples fall into which region
        mask_lower = X_feature < self.lower_bound_
        mask_upper = X_feature > self.upper_bound_
        mask_middle = ~(mask_lower | mask_upper)
        
        # Predict with appropriate model for each region
        if np.any(mask_lower):
            y_pred[mask_lower.to_list()] = self.lr_model_lower_.predict(X[mask_lower])
        
        if np.any(mask_upper):
            y_pred[mask_upper.to_list()] = self.lr_model_upper_.predict(X[mask_upper])
        
        if np.any(mask_middle):
            y_pred[mask_middle.to_list()] = self.rf_model_.predict(X[mask_middle])
        
        return y_pred
    
    def get_boundaries(self):
        """Return the computed boundaries."""
        if not hasattr(self, 'lower_bound_'):
            raise ValueError("Boundaries not set. Call 'fit' first.")
        return self.lower_bound_, self.upper_bound_


class HybridForestExecutionTimePredictor(SklearnExecutionTimePredictor):
    def __init__(
        self,
        predictor_config: HybridForestExecutionTimePredictorConfig,
        replica_config: ReplicaConfig,
        replica_scheduler_config: BaseReplicaSchedulerConfig,
        metrics_config: MetricsConfig,
    ) -> None:
        # will trigger model training
        super().__init__(
            predictor_config=predictor_config,
            replica_config=replica_config,
            replica_scheduler_config=replica_scheduler_config,
            metrics_config=metrics_config,
        )

    def _get_grid_search_params(self):
        return {
            "n_estimators": self._config.num_estimators,
            "max_depth": self._config.max_depth,
            "min_samples_split": self._config.min_samples_split,
            "boundary_method": self._config.boundary_method,
            "lower_boundary": self._config.lower_boundary,
            "upper_boundary": self._config.upper_boundary,
            "percentile_range": self._config.percentile_range,
            "feature_idx": self._config.feature_idx,
            "polynomial_degree": self._config.polynomial_degree,
            "polynomial_include_bias": self._config.polynomial_include_bias,
            "polynomial_interaction_only": self._config.polynomial_interaction_only,
            "fit_intercept": self._config.fit_intercept,
        }

    def _get_estimator(self):
        return HybridForestLinearRegressor()