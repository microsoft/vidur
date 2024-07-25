from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from vidur.config import SimulationConfig
from vidur.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
)


class LinearRegressionExecutionTimePredictor(SklearnExecutionTimePredictor):
    def __init__(self, config: SimulationConfig):
        predictor_config = config.cluster_config.execution_time_predictor_config
        self._polynomial_degree = (
            predictor_config.polynomial_degree
        )
        self._polynomial_include_bias = (
            predictor_config.polynomial_include_bias
        )
        self._polynomial_interaction_only = (
            predictor_config.polynomial_interaction_only
        )
        self._fit_intercept = (
            predictor_config.fit_intercept
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
