import pandas as pd

from simulator.logger import init_logger
from simulator.request_generator.base_request_interval_generator import (
    BaseRequestIntervalGenerator,
)

logger = init_logger(__name__)


class TraceRequestIntervalGenerator(BaseRequestIntervalGenerator):
    """
    Reads a trace csv file containing request arrival time, its prompt and completion token values to generate
    inter-request times, number of tokens.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        trace_file = self._config.trace_request_interval_generator_trace_file
        # load into a pd dataframe
        self._trace_df = pd.read_csv(trace_file)

        self._trace_df["arrival_time"] = pd.to_datetime(self._trace_df["arrival_time"])
        # restrict trace_df to be a subset of rows that have the same date
        self._trace_df = self._trace_df[
            (
                self._trace_df["arrival_time"]
                > self._config.trace_request_interval_generator_start_time
            )
            & (
                self._trace_df["arrival_time"]
                < self._config.trace_request_interval_generator_end_time
            )
        ]

        # change back to seconds
        self._trace_df["arrival_time"] = (
            self._trace_df["arrival_time"] - self._trace_df["arrival_time"].min()
        ) // pd.Timedelta("1s")

        # rescale the time to change QPS
        self._trace_df["arrival_time"] = (
            self._trace_df["arrival_time"]
            * self._config.trace_request_interval_generator_time_scale_factor
        )

        # compute the inter-request time
        self._trace_df["inter_request_time"] = self._trace_df["arrival_time"].diff()

        self._next_request_idx = 1

        logger.info(
            f"Loaded interval trace file {trace_file} with {len(self._trace_df)} requests"
        )

    def get_next_inter_request_time(self) -> float:
        if self._next_request_idx >= len(self._trace_df):
            return None

        inter_request_time = self._trace_df.iloc[self._next_request_idx][
            "inter_request_time"
        ]
        self._next_request_idx += 1
        return inter_request_time
