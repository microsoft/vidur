from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List

from vidur.config.base_poly_config import BasePolyConfig
from vidur.config.flat_dataclass import create_flat_dataclass
from vidur.logger import init_logger
from vidur.types import ReplicaSchedulerType, GlobalSchedulerType, ExecutionTimePredictorType, RequestGeneratorType, RequestIntervalGeneratorType, RequestLengthGeneratorType

logger = init_logger(__name__)


@dataclass
class BaseRequestIntervalGeneratorConfig(BasePolyConfig):
    seed: int = 42


@dataclass
class BaseRequestLengthGeneratorConfig(BasePolyConfig):
    seed: int = 42


@dataclass
class TraceRequestIntervalGeneratorConfig(BaseRequestIntervalGeneratorConfig):
    trace_file: str = (
        "data/processed_traces/AzureFunctionsInvocationTraceForTwoWeeksJan2021Processed.csv"
    )
    start_time: str = "1970-01-04 12:00:00"
    end_time: str = "1970-01-04 15:00:00"
    time_scale_factor: float = 0.3

    @staticmethod
    def get_type():
        return RequestIntervalGeneratorType.TRACE


@dataclass
class PoissonRequestIntervalGeneratorConfig(BaseRequestIntervalGeneratorConfig):
    qps: float = 1.0

    @staticmethod
    def get_type():
        return RequestIntervalGeneratorType.POISSON


@dataclass
class GammaRequestIntervalGeneratorConfig(BaseRequestIntervalGeneratorConfig):
    qps: float = 1.0
    cv: float = 0.5

    @staticmethod
    def get_type():
        return RequestIntervalGeneratorType.GAMMA


@dataclass
class StaticRequestIntervalGeneratorConfig(BaseRequestIntervalGeneratorConfig):
    @staticmethod
    def get_type():
        return RequestIntervalGeneratorType.STATIC


@dataclass
class TraceRequestLengthGeneratorConfig(BaseRequestLengthGeneratorConfig):
    trace_file: str = (
        "data/processed_traces/sharegpt_8k_filtered_stats_llama2_tokenizer.csv"
    )
    prefill_scale_factor: float = 1
    decode_scale_factor: float = 1
    max_tokens: int = 4096

    @staticmethod
    def get_type():
        return RequestLengthGeneratorType.TRACE


@dataclass
class ZipfRequestLengthGeneratorConfig(BaseRequestLengthGeneratorConfig):
    theta: float = 0.6
    scramble: bool = False
    min_tokens: int = 1024
    max_tokens: int = 4096
    prefill_to_decode_ratio: float = 20.0

    @staticmethod
    def get_type():
        return RequestLengthGeneratorType.ZIPF


@dataclass
class UniformRequestLengthGeneratorConfig(BaseRequestLengthGeneratorConfig):
    min_tokens: int = 1024
    max_tokens: int = 4096
    prefill_to_decode_ratio: float = 20.0

    @staticmethod
    def get_type():
        return RequestLengthGeneratorType.UNIFORM


@dataclass
class FixedRequestLengthGeneratorConfig(BaseRequestLengthGeneratorConfig):
    prefill_tokens: int = 4096
    decode_tokens: int = 512

    @staticmethod
    def get_type():
        return RequestLengthGeneratorType.FIXED


@dataclass
class BaseRequestGeneratorConfig(BasePolyConfig):
    seed: int = 42


@dataclass
class SyntheticRequestGeneratorConfig(BaseRequestGeneratorConfig):
    length_generator_config: BaseRequestLengthGeneratorConfig = field(
        default_factory=FixedRequestLengthGeneratorConfig
    )
    interval_generator_config: BaseRequestIntervalGeneratorConfig = field(
        default_factory=PoissonRequestIntervalGeneratorConfig
    )
    num_requests: int = 64
    duration: float = None

    @staticmethod
    def get_type():
        return RequestGeneratorType.SYNTHETIC


@dataclass
class TraceRequestGeneratorConfig(BaseRequestGeneratorConfig):
    trace_file: str = "data/processed_traces/sydney_enterprise.csv"
    date: str = "2023-08-21"
    prefill_scale_factor: float = 0.3
    decode_scale_factor: float = 1
    time_scale_factor: float = 0.04
    max_tokens: int = 4096

    @staticmethod
    def get_type():
        return RequestGeneratorType.TRACE


@dataclass
class BaseReplicaSchedulerConfig(BasePolyConfig):
    max_num_seqs: int = 128
    num_pipeline_stages: int = 1
    watermark_blocks_fraction: float = 0.01
    block_size: int = 16
    num_blocks: Optional[int] = None

    @abstractmethod
    def get_max_num_batched_tokens(self):
        pass


@dataclass
class VllmSchedulerConfig(BaseReplicaSchedulerConfig):
    max_batched_tokens: int = None

    @staticmethod
    def get_type():
        return ReplicaSchedulerType.VLLM


@dataclass
class LightLLMSchedulerConfig(BaseReplicaSchedulerConfig):
    max_batched_tokens: int = None
    max_tokens_in_batch: int = None

    @staticmethod
    def get_type():
        return ReplicaSchedulerType.SIMPLE_CHUNKING


@dataclass
class OrcaSchedulerConfig(BaseReplicaSchedulerConfig):

    @staticmethod
    def get_type():
        return ReplicaSchedulerType.ORCA


@dataclass
class FasterTransformerSchedulerConfig(BaseReplicaSchedulerConfig):

    @staticmethod
    def get_type():
        return ReplicaSchedulerType.FASTER_TRANSFORMER


@dataclass
class SarathiSchedulerConfig(BaseReplicaSchedulerConfig):
    chunk_size: int = 512

    @staticmethod
    def get_type():
        return ReplicaSchedulerType.SARATHI


@dataclass
class MetricsConfig:
    """Metric configuration."""

    write_metrics: bool = True
    write_json_trace: bool = False
    wandb_project: Optional[str] = None
    wandb_group: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_sweep_id: Optional[str] = None
    wandb_run_id: Optional[str] = None
    enable_chrome_trace: bool = True
    save_table_to_wandb: bool = False
    store_plots: bool = True
    store_operation_metrics: bool = False
    store_token_completion_metrics: bool = False
    store_request_metrics: bool = True
    store_batch_metrics: bool = True
    store_utilization_metrics: bool = True
    keep_individual_batch_metrics: bool = False


@dataclass
class ReplicaConfig:
    model_name: str = "meta-llama/Llama-2-7b-hf"
    gpu_memory_utilization: float = 0.8
    pipeline_parallel_size: int = 1
    tensor_parallel_size: int = 1
    device: str = "a100"
    network_device: str = "a100_pair_nvlink"

    def __post_init__(self):
        self.world_size = self.pipeline_parallel_size * self.tensor_parallel_size


@dataclass
class BaseGlobalSchedulerConfig(BasePolyConfig):
    pass


@dataclass
class RandomGlobalSchedulerConfig(BaseGlobalSchedulerConfig):
    @staticmethod
    def get_type():
        return GlobalSchedulerType.RANDOM


@dataclass
class RoundRobinGlobalSchedulerConfig(BaseGlobalSchedulerConfig):
    @staticmethod
    def get_type():
        return GlobalSchedulerType.ROUND_ROBIN


@dataclass
class LORGlobalSchedulerConfig(BaseGlobalSchedulerConfig):
    @staticmethod
    def get_type():
        return GlobalSchedulerType.LOR


@dataclass
class BaseExecutionTimePredictorConfig(BasePolyConfig):
    compute_input_file: str = "./data/profiling/compute/{DEVICE}/{MODEL}/mlp.csv"
    attention_input_file: str = "./data/profiling/compute/{DEVICE}/{MODEL}/attention.csv"
    all_reduce_input_file: str = "./data/profiling/network/{NETWORK_DEVICE}/all_reduce.csv"
    send_recv_input_file: str = "./data/profiling/network/{NETWORK_DEVICE}/send_recv.csv"
    cpu_overhead_input_file: str = "./data/profiling/cpu_overhead/{NETWORK_DEVICE}/{MODEL}/cpu_overheads.csv"
    k_fold_cv_splits: int = 10
    no_cache: bool = False
    kv_cache_prediction_granularity: int = 64
    prediction_max_prefill_chunk_size: int = 4096
    prediction_max_batch_size: int = 128
    prediction_max_tokens_per_request: int = 4096
    attention_decode_batching_overhead_fraction: float = 0.1
    attention_prefill_batching_overhead_fraction: float = 0.1
    nccl_cpu_launch_overhead_ms: float =  0.02
    nccl_cpu_skew_overhead_per_device_ms: float = 0.0
    num_training_job_threads: int = -1
    skip_cpu_overhead_modeling: bool = True


@dataclass
class LinearRegressionExecutionTimePredictorConfig(BaseExecutionTimePredictorConfig):
    polynomial_degree: List[int] = field(default_factory=lambda: list(range(1, 6)))
    polynomial_include_bias: List[bool] = field(default_factory=lambda: [True, False])
    polynomial_interaction_only: List[bool] = field(default_factory=lambda: [True, False])
    fit_intercept: List[bool] = field(default_factory=lambda: [True, False])

    @staticmethod
    def get_type():
        return ExecutionTimePredictorType.LINEAR_REGRESSION


@dataclass
class RandomForrestExecutionTimePredictorConfig(BaseExecutionTimePredictorConfig):
    num_estimators: List[int] = field(default_factory=lambda: [250, 500, 750])
    max_depth: List[int] = field(default_factory=lambda: [8, 16, 32])
    min_samples_split: List[int] = field(default_factory=lambda: [2, 5, 10])

    @staticmethod
    def get_type():
        return ExecutionTimePredictorType.RANDOM_FORREST


@dataclass
class ClusterConfig:
    num_replicas: int = 1
    replica_config: ReplicaConfig = field(default_factory=ReplicaConfig)
    execution_time_predictor_config: BaseExecutionTimePredictorConfig = field(
        default_factory=RandomForrestExecutionTimePredictorConfig
    )
    global_scheduler_config: BaseGlobalSchedulerConfig = field(
        default_factory=RoundRobinGlobalSchedulerConfig
    )
    replica_scheduler_config: BaseReplicaSchedulerConfig = field(
        default_factory=SarathiSchedulerConfig
    )
    metrics_config: MetricsConfig = field(default_factory=MetricsConfig)


@dataclass
class SimulationConfig(ABC):
    log_level: str = "info"
    output_dir: str = "simulator_output"
    cache_dir: str = "cache"
    time_limit: int = 0 # in seconds, 0 is no limit
    cluster_config: ClusterConfig = field(default_factory=ClusterConfig)
    request_generator_config: BaseRequestGeneratorConfig = field(
        default_factory=SyntheticRequestGeneratorConfig
    )

    def __post_init__(self):
        self.output_dir = (
            f"{self.output_dir}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')}"
        )

    @classmethod
    def create_from_cli_args(cls):
        flat_config = create_flat_dataclass(cls).create_from_cli_args()
        instance = flat_config.reconstruct_original_dataclass()
        instance.__flat_config__ = flat_config
        return instance

    def to_dict(self):
        if not hasattr(self, "__flat_config__"):
            logger.warning("Flat config not found. Returning the original config.")
            return self.__dict__

        return self.__flat_config__.__dict__
