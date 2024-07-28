import logging
from typing import List

import pandas as pd

from vidur.config import TraceRequestGeneratorConfig
from vidur.entities import Request
from vidur.request_generator.base_request_generator import BaseRequestGenerator

logger = logging.getLogger(__name__)


class TraceReplayRequestGenerator(BaseRequestGenerator):
    """
    Reads a trace csv file containing request arrival time, its prompt and completion token values to generate
    inter-request times, number of tokens.
    """

    def __init__(self, config: TraceRequestGeneratorConfig):
        super().__init__(config)

        # load into a pd dataframe
        self.trace_df = pd.read_csv(config.trace_file)
        # restrict trace_df to be a subset of rows that have the same date
        self.trace_df = self.trace_df[self.trace_df["Date"] == config.date]

        # scale prefill and decode tokens
        self.trace_df["PromptTokenCount"] = (
            self.trace_df["PromptTokenCount"] * config.prefill_scale_factor
        )
        self.trace_df["CompletionTokenCount"] = (
            self.trace_df["CompletionTokenCount"] * config.decode_scale_factor
        )

        # make sure all the prefill and decode counts are integers
        self.trace_df["PromptTokenCount"] = self.trace_df["PromptTokenCount"].astype(
            int
        )
        self.trace_df["CompletionTokenCount"] = self.trace_df[
            "CompletionTokenCount"
        ].astype(int)

        # make sure that there is at least one prefill and decode token
        self.trace_df["PromptTokenCount"] = self.trace_df["PromptTokenCount"].clip(
            lower=1
        )
        self.trace_df["CompletionTokenCount"] = self.trace_df[
            "CompletionTokenCount"
        ].clip(lower=1)

        # make sure the total does not exceed the max tokens, adjust the prefill tokens if needed
        total_tokens = (
            self.trace_df["PromptTokenCount"] + self.trace_df["CompletionTokenCount"]
        )
        diff_tokens = total_tokens - config.max_tokens
        diff_tokens = diff_tokens.clip(lower=0)
        self.trace_df["PromptTokenCount"] = (
            self.trace_df["PromptTokenCount"] - diff_tokens
        )

        assert all(
            self.trace_df["PromptTokenCount"] + self.trace_df["CompletionTokenCount"]
            <= config.max_tokens
        )

        # rescale the time to change QPS
        self.trace_df["Time"] = self.trace_df["Time"] * config.time_scale_factor

        # compute pd ratio and log the 25, 50, 75, 90, 95, 99 percentiles
        pd_ratio = (
            self.trace_df["PromptTokenCount"] / self.trace_df["CompletionTokenCount"]
        )
        logger.info(
            f"Loaded trace file {config.trace_file} with {len(self.trace_df)} requests"
        )
        logger.info(
            f"Prompt/decode token ratio stats\n:{pd_ratio.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99])}"
        )

    def generate_requests(self) -> List[Request]:
        requests = []

        for _, row in self.trace_df.iterrows():
            request = Request(
                arrived_at=row["Time"],
                num_prefill_tokens=row["PromptTokenCount"],
                num_decode_tokens=row["CompletionTokenCount"],
            )

            requests.append(request)

        return requests
