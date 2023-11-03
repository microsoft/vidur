import logging
import os
from functools import reduce
from typing import Dict, List

import pandas as pd

import wandb
from simulator.config import Config
from simulator.entities import Batch, BatchStage, Request
from simulator.plotting.constants import (
    BatchMetricsCountDistribution,
    BatchMetricsTimeDistribution,
    CompletionMetricsTimeSeries,
    DecodeTimeDistribution,
    RequestMetricsHistogram,
    RequestMetricsTimeDistributions,
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
        self._req_metrics_time_distributions: Dict[
            RequestMetricsTimeDistributions, DataSeries
        ] = {}
        for metric_name in RequestMetricsTimeDistributions:
            self._req_metrics_time_distributions[metric_name] = DataSeries(
                REQUEST_ID_STR,
                metric_name.value,
                self._subsamples,
                self._save_table_to_wandb,
            )

        self._decode_time_distributions: Dict[
            DecodeTimeDistribution, DataSeries
        ] = {}
        for metric_name in DecodeTimeDistribution:
            self._decode_time_distributions[metric_name] = DataSeries(
                REQUEST_ID_STR,
                metric_name.value,
                self._subsamples,
                self._save_table_to_wandb,
            )        

        self._req_metrics_histogram: Dict[RequestMetricsHistogram, DataSeries] = {}
        for metric_name in RequestMetricsHistogram:
            self._req_metrics_histogram[metric_name] = DataSeries(
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
            self._batch_metrics_count_distribution[metric_name] = DataSeries(
                BATCH_ID_STR,
                metric_name.value,
                self._subsamples,
                self._save_table_to_wandb,
            )

        self._batch_metrics_time_distribution: Dict[
            BatchMetricsTimeDistribution, DataSeries
        ] = {}
        for metric_name in BatchMetricsTimeDistribution:
            self._batch_metrics_time_distribution[metric_name] = DataSeries(
                BATCH_ID_STR,
                metric_name.value,
                self._subsamples,
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

    @if_write_metrics
    def plot(self) -> None:
        dir_plot_path = f"{self._config.output_dir}/plots"
        os.makedirs(dir_plot_path, exist_ok=True)

        all_req_metrics = list(self._req_metrics_time_distributions.values()) + list(
            self._req_metrics_histogram.values()
        )

        self._save_as_csv(
            dataseries_list=all_req_metrics,
            key_to_join=REQUEST_ID_STR,
            base_path=self._config.output_dir,
            file_name="request_metrics",
        )

        all_batch_metrics = list(
            self._batch_metrics_count_distribution.values()
        ) + list(self._batch_metrics_time_distribution.values())

        self._save_as_csv(
            dataseries_list=all_batch_metrics,
            key_to_join=BATCH_ID_STR,
            base_path=self._config.output_dir,
            file_name="batch_metrics",
        )

        all_histogram_metrics = list(self._req_metrics_histogram.values())
        all_time_distribution_metrics = list(
            self._req_metrics_time_distributions.values()
        ) + list(self._batch_metrics_time_distribution.values()
        ) + list(self._decode_time_distributions.values())
        all_count_distribution_metrics = list(
            self._batch_metrics_count_distribution.values()
        )
        all_time_series_metrics = list(self._completion_metrics_time_series.values())

        for dataseries in all_histogram_metrics:
            dataseries.plot_histogram(dir_plot_path, dataseries._y_name)

        for dataseries in all_time_distribution_metrics:
            dataseries.plot_cdf(dir_plot_path, dataseries._y_name, TIME_STR)

        for dataseries in all_count_distribution_metrics:
            dataseries.plot_cdf(dir_plot_path, dataseries._y_name, COUNT_STR)

        for dataseries in all_time_series_metrics:
            dataseries.plot_step(
                dir_plot_path, f"{dataseries._y_name}_time_series", COUNT_STR
            )

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
    def on_request_arrival(self, time: float, request: Request) -> None:
        self._completion_metrics_time_series[
            CompletionMetricsTimeSeries.REQUEST_ARRIVAL
        ].put_delta(time, 1)
        self._req_metrics_histogram[RequestMetricsHistogram.REQUEST_NUM_TOKENS].put(
            request.id, request.total_tokens
        )
        self._req_metrics_histogram[RequestMetricsHistogram.REQUEST_PREFILL_TOKENS].put(
            request.id, request.num_prefill_tokens
        )
        self._req_metrics_histogram[RequestMetricsHistogram.REQUEST_DECODE_TOKENS].put(
            request.id, request.num_decode_tokens
        )
        self._req_metrics_histogram[RequestMetricsHistogram.REQUEST_PD_RATIO].put(
            request.id, request.pd_ratio
        )
        self._req_metrics_histogram[
            RequestMetricsHistogram.REQUEST_INTER_ARRIVAL_DELAY
        ].put(request.id, request.arrived_at)

    @if_write_metrics
    def _on_request_end(self, time: float, request: Request) -> None:
        self._completion_metrics_time_series[
            CompletionMetricsTimeSeries.REQUEST_COMPLETION
        ].put_delta(request.completed_at, 1)
        self._req_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_E2E_TIME
        ].put(request.id, request.completed_at - request.arrived_at)
        self._req_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_EXECUTION_TIME
        ].put(request.id, request.execution_time)
        self._req_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_PREEMPTION_TIME
        ].put(request.id, request.preempted_time)
        self._req_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_SCHEDULING_DELAY
        ].put(request.id, request.scheduling_delay)
        self._req_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_EXECUTION_PLUS_PREEMPTION_TIME
        ].put(request.id, request.execution_time + request.preempted_time)
        self._req_metrics_time_distributions[
            RequestMetricsTimeDistributions.PREFILL_TIME_E2E
        ].put(request.id, request.prefill_completed_at - request.arrived_at)
        self._req_metrics_time_distributions[
            RequestMetricsTimeDistributions.PREFILL_TIME_EXECUTION_PLUS_PREEMPTION
        ].put(request.id, request.prefill_completed_at - request.scheduled_at)
        self._req_metrics_time_distributions[
            RequestMetricsTimeDistributions.PREFILL_TIME_EXECUTION_PLUS_PREEMPTION_NORMALIZED
        ].put(
            request.id,
            (request.prefill_completed_at - request.scheduled_at)
            / request.num_prefill_tokens,
        )
        self._req_metrics_time_distributions[
            RequestMetricsTimeDistributions.DECODE_TIME_EXECUTION_PLUS_PREEMPTION_NORMALIZED
        ].put(
            request.id,
            (request.completed_at - request.prefill_completed_at)
            / request.num_decode_tokens,
        )

        self._req_metrics_histogram[RequestMetricsHistogram.REQUEST_NUM_RESTARTS].put(
            request.id, request.num_restarts
        )

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

        self._decode_time_distributions[
            DecodeTimeDistribution.DECODE_TOKEN_EXECUTION_PLUS_PREMPTION_TIME
        ].put(
            request.id,
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
        ].put(batch.id, time - batch.scheduled_at)

        self._batch_metrics_count_distribution[
            BatchMetricsCountDistribution.BATCH_NUM_TOKENS
        ].put(batch.id, batch.total_num_tokens)

        self._batch_metrics_count_distribution[
            BatchMetricsCountDistribution.BATCH_NUM_PREFILL_TOKENS
        ].put(batch.id, batch.num_prefill_tokens)

        self._batch_metrics_count_distribution[
            BatchMetricsCountDistribution.BATCH_NUM_DECODE_TOKENS
        ].put(batch.id, batch.num_decode_tokens)

        self._batch_metrics_count_distribution[
            BatchMetricsCountDistribution.BATCH_SIZE
        ].put(batch.id, batch.size)

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
        self, time: float, replica_id: int, stage_id: int, batch_stage: BatchStage
    ) -> None:
        self._replica_busy_time[replica_id - 1][stage_id - 1].put(time, 100)
        mfu = self._mfu_calculator.get_mfu(batch_stage)
        self._replica_mfu[replica_id - 1][stage_id - 1].put(time, mfu)

    @if_write_metrics
    def on_batch_stage_end(self, time: float, replica_id: int, stage_id: int) -> None:
        self._replica_busy_time[replica_id - 1][stage_id - 1].put(time, 0)
        self._replica_mfu[replica_id - 1][stage_id - 1].put(time, 0)
