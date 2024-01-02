import logging
import os
from functools import reduce
from typing import Dict, List

import pandas as pd
import plotly_express as px
import wandb

from simulator.config import Config
from simulator.entities import Batch, BatchStage, ExecutionTime, Request
from simulator.plotting.cdf_sketch import CDFSketch
from simulator.plotting.constants import (
    BatchMetricsCountDistribution,
    BatchMetricsTimeDistribution,
    CompletionMetricsTimeSeries,
    CpuOperationMetrics,
    OperationMetrics,
    RequestMetricsHistogram,
    RequestMetricsTimeDistributions,
    TokenMetricsTimeDistribution,
)
from simulator.plotting.data_series import DataSeries
from simulator.plotting.series_average_meter import SeriesAverageMeter
from simulator.utils.mfu_calculator import MFUCalculator

logger = logging.getLogger(__name__)


def if_write_metrics(func):
    def wrapper(self, *args, **kwargs):
        if self._should_write_metrics:
            return func(self, *args, **kwargs)

    return wrapper


REQUEST_ID_STR = "Request Id"
COUNT_STR = "Count"
TIME_STR = "Time (sec)"
BATCH_ID_STR = "Batch Id"
MEMORY_USAGE_STR = "Memory Usage (%)"
BUSY_TIME_PERCENT = "Busy Time (%)"
UTILIZATION_STR = "Utilization (%)"
OPERATION_STR = "Operation"
TIME_STR_MS = "Time (ms)"


class MetricsStore:
    def __init__(self, config: Config):
        self._config = config
        self._num_replicas = config.cluster_num_replicas
        self._num_stages = config.replica_num_pipeline_stages
        self._should_write_metrics = config.write_metrics
        self._subsamples = config.metrics_store_subsamples
        self._save_table_to_wandb = config.metrics_store_save_table_to_wandb

        self._wandb_project = config.metrics_store_wandb_project
        self._wandb_group = config.metrics_store_wandb_group
        self._wandb_run_name = config.metrics_store_wandb_run_name

        self._min_batch_idx = config.metrics_store_min_batch_idx
        self._max_batch_idx = config.metrics_store_max_batch_idx

        # Initialise request metrics
        self._request_metrics_time_distributions: Dict[
            RequestMetricsTimeDistributions, DataSeries
        ] = {}
        for metric_name in RequestMetricsTimeDistributions:
            self._request_metrics_time_distributions[metric_name] = DataSeries(
                REQUEST_ID_STR,
                metric_name.value,
                self._subsamples,
                self._save_table_to_wandb,
            )

        self._token_metrics_time_distribution: Dict[
            TokenMetricsTimeDistribution, DataSeries
        ] = {}
        for metric_name in TokenMetricsTimeDistribution:
            self._token_metrics_time_distribution[metric_name] = CDFSketch(
                metric_name.value,
                self._save_table_to_wandb,
            )

        self._request_metrics_histogram: Dict[RequestMetricsHistogram, DataSeries] = {}
        for metric_name in RequestMetricsHistogram:
            self._request_metrics_histogram[metric_name] = DataSeries(
                REQUEST_ID_STR,
                metric_name.value,
                self._subsamples,
                self._save_table_to_wandb,
            )

        # Initialise batch metrics
        self._batch_metrics_count_distribution: Dict[
            BatchMetricsCountDistribution, DataSeries
        ] = {}
        for metric_name in BatchMetricsCountDistribution:
            self._batch_metrics_count_distribution[metric_name] = CDFSketch(
                metric_name.value,
                self._save_table_to_wandb,
            )

        self._batch_metrics_time_distribution: Dict[
            BatchMetricsTimeDistribution, DataSeries
        ] = {}
        for metric_name in BatchMetricsTimeDistribution:
            self._batch_metrics_time_distribution[metric_name] = CDFSketch(
                metric_name.value,
                self._save_table_to_wandb,
            )

        # Initialise completion metrics
        self._completion_metrics_time_series: Dict[
            CompletionMetricsTimeSeries, DataSeries
        ] = {}
        for metric_name in CompletionMetricsTimeSeries:
            self._completion_metrics_time_series[metric_name] = DataSeries(
                TIME_STR,
                metric_name.value,
                self._subsamples,
                self._save_table_to_wandb,
            )

        # Initialise operation metrics
        self._operation_metrics: Dict[OperationMetrics, DataSeries] = {}
        for metric_name in OperationMetrics:
            self._operation_metrics[metric_name] = CDFSketch(
                metric_name.value,
                self._save_table_to_wandb,
            )

        self._cpu_operation_metrics: Dict[CpuOperationMetrics, DataSeries] = {}
        for metric_name in CpuOperationMetrics:
            self._cpu_operation_metrics[metric_name] = CDFSketch(
                metric_name.value,
                self._save_table_to_wandb,
            )

        # per replica metrics
        self._replica_memory_usage = []
        # per replica stage metrics
        self._replica_busy_time = []
        self._replica_mfu = []
        self._mfu_calculator = MFUCalculator(config)

        for replica_idx in range(self._num_replicas):
            self._replica_memory_usage.append(
                SeriesAverageMeter(
                    TIME_STR,
                    MEMORY_USAGE_STR,
                    self._save_table_to_wandb,
                )
            )
            self._replica_memory_usage[replica_idx].put(0, 0)

            self._replica_busy_time.append([])
            self._replica_mfu.append([])

            for stage_idx in range(self._num_stages):
                self._replica_busy_time[replica_idx].append(
                    SeriesAverageMeter(
                        TIME_STR,
                        BUSY_TIME_PERCENT,
                        save_table_to_wandb=self._save_table_to_wandb,
                    )
                )
                self._replica_busy_time[replica_idx][stage_idx].put(0, 0)

                self._replica_mfu[replica_idx].append(
                    SeriesAverageMeter(
                        TIME_STR,
                        UTILIZATION_STR,
                        save_table_to_wandb=self._save_table_to_wandb,
                    )
                )
                self._replica_mfu[replica_idx][stage_idx].put(0, 0)

        self._init_wandb()

    def _init_wandb(self):
        if (
            not self._should_write_metrics
            or not self._wandb_project
            or not self._wandb_group
        ):
            return

        wandb.init(
            project=self._wandb_project,
            group=self._wandb_group,
            name=self._wandb_run_name,
            config=self._config.to_dict(),
        )

    def _save_as_csv(
        self,
        dataseries_list: List[DataSeries],
        key_to_join: str,
        base_path: str,
        file_name: str,
    ):
        os.makedirs(base_path, exist_ok=True)

        merged_request_df = reduce(
            lambda left, right: pd.merge(left, right, on=[key_to_join], how="outer"),
            [dataseries._to_df() for dataseries in dataseries_list],
        )
        merged_request_df.to_csv(f"{base_path}/{file_name}.csv", index=False)

    def _store_bar_plot(
        self,
        base_path: str,
        plot_name: str,
        x_label: str,
        y_label: str,
        data: Dict[str, float],
    ):
        fig = px.bar(
            x=list(data.keys()),
            y=list(data.values()),
            labels={"x": x_label, "y": y_label},
        )

        if wandb.run:
            wandb.log(
                {
                    plot_name: wandb.plot.bar(
                        wandb.Table(
                            dataframe=pd.DataFrame(
                                data=data.items(), columns=[x_label, y_label]
                            )
                        ),
                        x_label,
                        y_label,
                        title=plot_name,
                    )
                },
                step=0,
            )

        fig.write_image(f"{base_path}/{plot_name}.png")

    def _store_operation_metrics(self, base_plot_path: str):
        total_operation_runtimes: Dict[str, float] = {}

        total_operation_runtimes["model_execution_e2e"] = 0
        for dataseries in self._operation_metrics.values():
            dataseries.plot_cdf(
                base_plot_path, f"{dataseries._metric_name}_execution_time", TIME_STR_MS
            )
            total_operation_runtimes[dataseries._metric_name] = dataseries.sum
            total_operation_runtimes["model_execution_e2e"] += dataseries.sum

        total_operation_runtimes["cpu"] = 0
        for dataseries in self._cpu_operation_metrics.values():
            dataseries.plot_cdf(
                base_plot_path, f"{dataseries._metric_name}_execution_time", TIME_STR_MS
            )
            total_operation_runtimes[dataseries._metric_name] = dataseries.sum
            if dataseries._metric_name != "ray_comm_time":
                total_operation_runtimes["cpu"] += dataseries.sum

        total_operation_runtimes["cpu"] += total_operation_runtimes[
            "model_execution_e2e"
        ]

        self._store_bar_plot(
            base_plot_path,
            "total_operation_runtimes",
            OPERATION_STR,
            TIME_STR_MS,
            total_operation_runtimes,
        )

    def _store_request_metrics(self, base_plot_path: str):
        all_request_metrics = list(
            self._request_metrics_time_distributions.values()
        ) + list(self._request_metrics_histogram.values())

        self._save_as_csv(
            dataseries_list=all_request_metrics,
            key_to_join=REQUEST_ID_STR,
            base_path=self._config.output_dir,
            file_name="request_metrics",
        )

        for dataseries in self._request_metrics_histogram.values():
            dataseries.plot_histogram(base_plot_path, dataseries._y_name)

        for dataseries in self._request_metrics_time_distributions.values():
            dataseries.plot_cdf(base_plot_path, dataseries._y_name, TIME_STR)

    def _store_batch_metrics(self, base_plot_path: str):
        for dataseries in self._batch_metrics_time_distribution.values():
            dataseries.plot_cdf(base_plot_path, dataseries._metric_name, TIME_STR)

        for dataseries in self._batch_metrics_count_distribution.values():
            dataseries.plot_cdf(base_plot_path, dataseries._metric_name, COUNT_STR)

    def _store_completion_metrics(self, base_plot_path: str):
        for dataseries in self._token_metrics_time_distribution.values():
            dataseries.plot_cdf(base_plot_path, dataseries._metric_name, TIME_STR)

        for dataseries in self._completion_metrics_time_series.values():
            dataseries.plot_step(
                base_plot_path, f"{dataseries._y_name}_time_series", COUNT_STR
            )

    def _store_utilization_metrics(self):
        for replica_idx in range(self._num_replicas):
            self._replica_memory_usage[replica_idx].print_stats(
                f"replica_{replica_idx + 1}_memory_usage"
            )
            for stage_idx in range(self._num_stages):
                self._replica_busy_time[replica_idx][stage_idx].print_stats(
                    f"replica_{replica_idx + 1}_stage_{stage_idx + 1}_busy_time_percent"
                )
                self._replica_mfu[replica_idx][stage_idx].print_stats(
                    f"replica_{replica_idx + 1}_stage_{stage_idx + 1}_mfu"
                )

    @if_write_metrics
    def plot(self) -> None:
        dir_plot_path = f"{self._config.output_dir}/plots"
        os.makedirs(dir_plot_path, exist_ok=True)

        self._store_request_metrics(dir_plot_path)
        self._store_batch_metrics(dir_plot_path)
        self._store_completion_metrics(dir_plot_path)
        self._store_operation_metrics(dir_plot_path)
        self._store_utilization_metrics()

    @if_write_metrics
    def on_request_arrival(self, time: float, request: Request) -> None:
        self._completion_metrics_time_series[
            CompletionMetricsTimeSeries.REQUEST_ARRIVAL
        ].put_delta(time, 1)
        self._request_metrics_histogram[RequestMetricsHistogram.REQUEST_NUM_TOKENS].put(
            request.id, request.total_tokens
        )
        self._request_metrics_histogram[
            RequestMetricsHistogram.REQUEST_PREFILL_TOKENS
        ].put(request.id, request.num_prefill_tokens)
        self._request_metrics_histogram[
            RequestMetricsHistogram.REQUEST_DECODE_TOKENS
        ].put(request.id, request.num_decode_tokens)
        self._request_metrics_histogram[RequestMetricsHistogram.REQUEST_PD_RATIO].put(
            request.id, request.pd_ratio
        )
        self._request_metrics_histogram[
            RequestMetricsHistogram.REQUEST_INTER_ARRIVAL_DELAY
        ].put(request.id, request.arrived_at)

    @if_write_metrics
    def _on_request_end(self, time: float, request: Request) -> None:
        self._completion_metrics_time_series[
            CompletionMetricsTimeSeries.REQUEST_COMPLETION
        ].put_delta(request.completed_at, 1)
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_E2E_TIME
        ].put(request.id, request.e2e_time)
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_E2E_TIME_NORMALIZED
        ].put(request.id, request.e2e_time_normalized)
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_E2E_TIME_PIECEWISE_NORMALIZED
        ].put(request.id, request.e2e_time_piecewise_normalized)
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_EXECUTION_TIME
        ].put(request.id, request.execution_time)
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_EXECUTION_TIME_NORMALIZED
        ].put(request.id, request.execution_time_normalized)
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_MODEL_EXECUTION_TIME
        ].put(request.id, request.model_execution_time)
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_MODEL_EXECUTION_TIME_NORMALIZED
        ].put(request.id, request.model_execution_time_normalized)
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_PREEMPTION_TIME
        ].put(request.id, request.preempted_time)
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_SCHEDULING_DELAY
        ].put(request.id, request.scheduling_delay)
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_EXECUTION_PLUS_PREEMPTION_TIME
        ].put(request.id, request.execution_time + request.preempted_time)
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.PREFILL_TIME_E2E
        ].put(request.id, request.prefill_completed_at - request.arrived_at)
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.PREFILL_TIME_EXECUTION_PLUS_PREEMPTION
        ].put(request.id, request.prefill_completed_at - request.scheduled_at)
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.PREFILL_TIME_EXECUTION_PLUS_PREEMPTION_NORMALIZED
        ].put(
            request.id,
            (request.prefill_completed_at - request.scheduled_at)
            / request.num_prefill_tokens,
        )
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.DECODE_TIME_EXECUTION_PLUS_PREEMPTION_NORMALIZED
        ].put(
            request.id,
            (request.completed_at - request.prefill_completed_at)
            / request.num_decode_tokens,
        )

        self._request_metrics_histogram[
            RequestMetricsHistogram.REQUEST_NUM_RESTARTS
        ].put(request.id, request.num_restarts)

    def _update_per_token_execution_times(
        self, time: float, request: Request, batch: Batch
    ) -> None:
        # if prefill has just finished in this iteration, update the prefill completion timeseries
        if time == request.prefill_completed_at:
            self._completion_metrics_time_series[
                CompletionMetricsTimeSeries.PREFILL_COMPLETIONS
            ].put_delta(
                time,
                request.num_prefill_tokens,
            )

        # determine if this was prefill or decode token
        if not request.has_started_decode:
            return

        self._token_metrics_time_distribution[
            TokenMetricsTimeDistribution.DECODE_TOKEN_EXECUTION_PLUS_PREMPTION_TIME
        ].put(
            time - batch.scheduled_at + request.latest_iteration_scheduling_delay,
        )

        self._completion_metrics_time_series[
            CompletionMetricsTimeSeries.DECODE_COMPLETIONS
        ].put_delta(time, 1)

    @if_write_metrics
    def on_batch_end(
        self, time: float, batch: Batch, replica_id: int, memory_usage_percent: int
    ) -> None:
        if (self._min_batch_idx and batch.id < self._min_batch_idx) or (
            self._max_batch_idx and batch.id > self._max_batch_idx
        ):
            return

        for request in batch.completed_requests:
            self._on_request_end(time, request)

        self._batch_metrics_time_distribution[
            BatchMetricsTimeDistribution.BATCH_EXECUTION_TIME
        ].put(time - batch.scheduled_at)

        self._batch_metrics_time_distribution[
            BatchMetricsTimeDistribution.MODEL_EXECUTION_E2E
        ].put(time - batch.scheduled_at)

        self._batch_metrics_count_distribution[
            BatchMetricsCountDistribution.BATCH_NUM_TOKENS
        ].put(batch.total_num_tokens)

        self._batch_metrics_count_distribution[
            BatchMetricsCountDistribution.BATCH_NUM_PREFILL_TOKENS
        ].put(batch.num_prefill_tokens)

        self._batch_metrics_count_distribution[
            BatchMetricsCountDistribution.BATCH_NUM_DECODE_TOKENS
        ].put(batch.num_decode_tokens)

        self._batch_metrics_count_distribution[
            BatchMetricsCountDistribution.BATCH_SIZE
        ].put(batch.size)

        self._replica_memory_usage[replica_id - 1].put(time, memory_usage_percent)

        for request in batch.requests:
            self._update_per_token_execution_times(time, request, batch)

    @if_write_metrics
    def on_replica_schedule(
        self, time: float, replica_id: int, memory_usage_percent: int
    ) -> None:
        self._replica_memory_usage[replica_id - 1].put(time, memory_usage_percent)

    @if_write_metrics
    def on_replica_stage_schedule(
        self,
        time: float,
        replica_id: int,
        stage_id: int,
        batch_stage: BatchStage,
        execution_time: ExecutionTime,
    ) -> None:
        pass
        self._replica_busy_time[replica_id - 1][stage_id - 1].put(time, 100)
        mfu = self._mfu_calculator.get_mfu(batch_stage)
        self._replica_mfu[replica_id - 1][stage_id - 1].put(time, mfu)

        for _ in range(execution_time.num_blocks):
            self._operation_metrics[OperationMetrics.MLP_UP_PROJ].put(
                execution_time.mlp_layer_up_proj_execution_time
            )
            self._operation_metrics[OperationMetrics.MLP_ACTIVATION].put(
                execution_time.mlp_layer_act_execution_time
            )
            self._operation_metrics[OperationMetrics.MLP_DOWN_PROJ].put(
                execution_time.mlp_layer_down_proj_execution_time
            )
            self._operation_metrics[OperationMetrics.MLP_DOWN_PROJ_ALL_REDUCE].put(
                execution_time.mlp_all_reduce_time
            )
            self._operation_metrics[OperationMetrics.ATTN_PRE_PROJ].put(
                execution_time.attention_pre_proj_time
            )
            self._operation_metrics[OperationMetrics.ATTN_POST_PROJ].put(
                execution_time.attention_post_proj_time
            )
            self._operation_metrics[OperationMetrics.ATTN_POST_PROJ_ALL_REDUCE].put(
                execution_time.attention_all_reduce_time
            )
            self._operation_metrics[OperationMetrics.ATTN_PREFILL_KV_CACHE_PREP].put(
                execution_time.attention_prefill_kv_cache_prep_execution_time
            )
            self._operation_metrics[
                OperationMetrics.ATTN_PREFILL_OUTPUT_RESHAPE_COPY
            ].put(execution_time.attention_prefill_output_reshape_copy_execution_time)
            self._operation_metrics[OperationMetrics.ATTN_PREFILL].put(
                execution_time.attention_prefill_execution_time
            )
            self._operation_metrics[OperationMetrics.ATTN_KV_CACHE_SAVE].put(
                execution_time.attention_kv_cache_save_execution_time
            )
            self._operation_metrics[OperationMetrics.ATTN_DECODE].put(
                execution_time.attention_decode_execution_time
            )
            self._operation_metrics[OperationMetrics.ATTN_ROPE].put(
                execution_time.attention_rope_execution_time
            )
            self._operation_metrics[OperationMetrics.ADD].put(
                execution_time.add_time * 2
            )
            self._operation_metrics[OperationMetrics.RMS_NORM].put(
                execution_time.rms_norm_time * 2
            )
        self._operation_metrics[OperationMetrics.PIPELINE_SEND_RECV].put(
            execution_time.pipeline_parallel_communication_time
        )

        self._cpu_operation_metrics[CpuOperationMetrics.SCHEDULE].put(
            execution_time.schedule_time
        )
        self._cpu_operation_metrics[CpuOperationMetrics.SAMPLER_E2E].put(
            execution_time.sampler_e2e_time
        )
        self._cpu_operation_metrics[CpuOperationMetrics.PREPARE_INPUTS_E2E].put(
            execution_time.prepare_inputs_e2e_time
        )
        self._cpu_operation_metrics[CpuOperationMetrics.PROCESS_MODEL_OUTPUTS].put(
            execution_time.process_model_outputs_time
        )
        self._cpu_operation_metrics[
            CpuOperationMetrics.POST_PREPARE_INPUTS_BARRIER
        ].put(execution_time.post_prepare_inputs_barrier_time)
        self._cpu_operation_metrics[CpuOperationMetrics.RAY_COMM_TIME].put(
            execution_time.ray_comm_time
        )

    @if_write_metrics
    def on_batch_stage_end(self, time: float, replica_id: int, stage_id: int) -> None:
        self._replica_busy_time[replica_id - 1][stage_id - 1].put(time, 0)
        self._replica_mfu[replica_id - 1][stage_id - 1].put(time, 0)
