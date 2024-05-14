from sklearn.ensemble import RandomForestRegressor

from vidur.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
)


class RandomForrestExecutionTimePredictor(SklearnExecutionTimePredictor):
    def __init__(self, config):
        self._num_estimators = (
            config.random_forrest_execution_time_predictor_num_estimators
        )
        self._max_depth = config.random_forrest_execution_time_predictor_max_depth
        self._min_samples_split = (
            config.random_forrest_execution_time_predictor_min_samples_split
        )

        # will trigger model training
        super().__init__(config)

    def _get_grid_search_params(self):
        return {
            "n_estimators": self._num_estimators,
            "max_depth": self._max_depth,
            "min_samples_split": self._min_samples_split,
        }

    def _get_estimator(self):
        return RandomForestRegressor()
