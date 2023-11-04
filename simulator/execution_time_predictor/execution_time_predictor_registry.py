from simulator.execution_time_predictor.dummy_execution_time_predictor import (
    DummyExecutionTimePredictor,
)
from simulator.execution_time_predictor.linear_regression_execution_time_predictor import (
    LinearRegressionExecutionTimePredictor,
)
from simulator.execution_time_predictor.random_forrest_execution_time_predictor import (
    RandomForrestExecutionTimePredictor,
)
from simulator.types import ExecutionTimePredictorType
from simulator.utils.base_registry import BaseRegistry


class ExecutionTimePredictorRegistry(BaseRegistry):
    @classmethod
    def get_key_from_str(cls, key_str: str) -> ExecutionTimePredictorType:
        return ExecutionTimePredictorType.from_str(key_str)


ExecutionTimePredictorRegistry.register(
    ExecutionTimePredictorType.DUMMY, DummyExecutionTimePredictor
)
ExecutionTimePredictorRegistry.register(
    ExecutionTimePredictorType.RANDOM_FORREST, RandomForrestExecutionTimePredictor
)
ExecutionTimePredictorRegistry.register(
    ExecutionTimePredictorType.LINEAR_REGRESSION, LinearRegressionExecutionTimePredictor
)
