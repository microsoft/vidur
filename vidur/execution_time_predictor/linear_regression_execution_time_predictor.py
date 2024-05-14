from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from vidur.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
)


class LinearRegressionExecutionTimePredictor(SklearnExecutionTimePredictor):
    def __init__(self, config):
        self._polynomial_degree = (
            config.linear_regression_execution_time_predictor_polynomial_degree
        )
        self._polynomial_include_bias = (
            config.linear_regression_execution_time_predictor_polynomial_include_bias
        )
        self._polynomial_interaction_only = (
            config.linear_regression_execution_time_predictor_polynomial_interaction_only
        )
        self._fit_intercept = (
            config.linear_regression_execution_time_predictor_fit_intercept
        )

        # will trigger model training
        super().__init__(config)

    def _get_grid_search_params(self):
        return {
            "polynomialfeatures__degree": self._polynomial_degree,
            "polynomialfeatures__include_bias": self._polynomial_include_bias,
            "polynomialfeatures__interaction_only": self._polynomial_interaction_only,
            "linearregression__fit_intercept": self._fit_intercept,
        }

    def _get_estimator(self):
        return make_pipeline(PolynomialFeatures(), LinearRegression())
