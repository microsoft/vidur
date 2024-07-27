from sklearn.ensemble import RandomForestRegressor

from vidur.config import RandomForrestExecutionTimePredictorConfig
from vidur.config.model_config import BaseModelConfig
from vidur.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
)


class RandomForrestExecutionTimePredictor(SklearnExecutionTimePredictor):
    def __init__(self, config: RandomForrestExecutionTimePredictorConfig, model_config: BaseModelConfig):
        # will trigger model training
        super().__init__(config, model_config)

    def _get_grid_search_params(self):
        return {
            "n_estimators": self._config.num_estimators,
            "max_depth": self._config.max_depth,
            "min_samples_split": self._config.min_samples_split,
        }

    def _get_estimator(self):
        return RandomForestRegressor()
