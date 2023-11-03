""" File to store names for different metrics captured """

import enum


class RequestMetricsTimeDistributions(enum.Enum):
    REQUEST_E2E_TIME = "request_e2e_time"
    REQUEST_EXECUTION_TIME = "request_execution_time"
    REQUEST_PREEMPTION_TIME = "request_preemption_time"
    REQUEST_SCHEDULING_DELAY = "request_scheduling_delay"
    REQUEST_EXECUTION_PLUS_PREEMPTION_TIME = "request_execution_plus_preemption_time"
    PREFILL_TIME_E2E = "prefill_e2e_time"
    PREFILL_TIME_EXECUTION_PLUS_PREEMPTION = "prefill_time_execution_plus_preemption"
    PREFILL_TIME_EXECUTION_PLUS_PREEMPTION_NORMALIZED = (
        "prefill_time_execution_plus_preemption_normalized"
    )
    DECODE_TIME_EXECUTION_PLUS_PREEMPTION_NORMALIZED = (
        "decode_time_execution_plus_preemption_normalized"
    )


class DecodeTimeDistribution(enum.Enum):
    DECODE_TOKEN_EXECUTION_PLUS_PREMPTION_TIME = "decode_token_execution_plus_preemption_time"


class RequestMetricsHistogram(enum.Enum):
    REQUEST_INTER_ARRIVAL_DELAY = "request_inter_arrival_delay"
    REQUEST_NUM_TOKENS = "request_num_tokens"
    REQUEST_PREFILL_TOKENS = "request_num_prefill_tokens"
    REQUEST_DECODE_TOKENS = "request_num_decode_tokens"
    REQUEST_PD_RATIO = "request_pd_ratio"
    REQUEST_NUM_RESTARTS = "request_num_restarts"


class BatchMetricsCountDistribution(enum.Enum):
    BATCH_NUM_TOKENS = "batch_num_tokens"
    BATCH_NUM_PREFILL_TOKENS = "batch_num_prefill_tokens"
    BATCH_NUM_DECODE_TOKENS = "batch_num_decode_tokens"
    BATCH_SIZE = "batch_size"


class BatchMetricsTimeDistribution(enum.Enum):
    BATCH_EXECUTION_TIME = "batch_execution_time"

class CompletionMetricsTimeSeries(enum.Enum):
    REQUEST_ARRIVAL = "request_arrival"
    REQUEST_COMPLETION = "request_completion"
    PREFILL_COMPLETIONS = "prefill_completion"
    DECODE_COMPLETIONS = "decode_completion"
