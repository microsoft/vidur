from sklearn.ensemble import RandomForestRegressor

from vidur.config import SimulationConfig
from vidur.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
)


class RandomForrestExecutionTimePredictor(SklearnExecutionTimePredictor):
    def __init__(self, config: SimulationConfig):
        predictor_config = config.cluster_config.execution_time_predictor_config
        self._num_estimators = (
            predictor_config.num_estimators
        )
        self._max_depth = predictor_config.max_depth
        self._min_samples_split = predictor_config.min_samples_split

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
