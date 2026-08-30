import atexit
import heapq
import json
from typing import List

from tqdm import tqdm

from vidur.config import SimulationConfig
from vidur.entities import Cluster
from vidur.events import BaseEvent, RequestArrivalEvent, RebalanceEvent
from vidur.logger import init_logger
from vidur.metrics import MetricsStore
from vidur.request_generator import RequestGeneratorRegistry
from vidur.scheduler import BaseGlobalScheduler, GlobalSchedulerRegistry
from vidur.types import GlobalSchedulerType
from vidur.events.base_event import BaseEvent

logger = init_logger(__name__)


class Simulator:
    def __init__(self, config: SimulationConfig) -> None:
        self._config: SimulationConfig = config

        self._time = 0
        self._terminate = False
        self._time_limit = self._config.time_limit
        if not self._time_limit:
            self._time_limit = float("inf")

        self._event_queue = []

        self._event_trace = []
        self._event_chrome_trace = []

        self._cluster = Cluster(
            self._config.cluster_config,
            self._config.metrics_config,
            self._config.request_generator_config,
        )
        self._metric_store = MetricsStore(self._config)
        self._request_generator = RequestGeneratorRegistry.get(
            self._config.request_generator_config.get_type(),
            self._config.request_generator_config,
        )
        self._scheduler = GlobalSchedulerRegistry.get(
            self._config.cluster_config.global_scheduler_config.get_type(),
            self._config,
            self._cluster.replicas,
        )
        BaseEvent.global_scheduler_ref = self._scheduler
        BaseEvent.cluster_ref = self._cluster

        self._init_event_queue()

    @property
    def scheduler(self) -> BaseGlobalScheduler:
        return self._scheduler

    @property
    def metric_store(self) -> MetricsStore:
        return self._metric_store

    def run(self) -> None:
        logger.info(
            f"Starting simulation with cluster: {self._cluster} and {len(self._event_queue)} requests"
        )

        # Create progress bar based on time limit or event count
        if self._time_limit != float("inf"):
            pbar = tqdm(total=100, desc="Simulation Progress", unit="%", 
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
            last_progress = 0
        else:
            # If no time limit, use event count
            total_events = len(self._event_queue)
            pbar = tqdm(total=total_events, desc="Processing Events", unit="events")
            last_progress = 0

        event_count = 0
        try:
            while self._event_queue and not self._terminate:
                _, event = heapq.heappop(self._event_queue)
                self._set_time(event._time)
                new_events = event.handle_event(self._scheduler, self._metric_store)
                self._add_events(new_events)

                if self._config.metrics_config.write_json_trace:
                    self._event_trace.append(event.to_dict())

                if self._config.metrics_config.enable_chrome_trace:
                    chrome_events = event.to_chrome_trace()
                    if chrome_events:
                        self._event_chrome_trace.extend(chrome_events)

                # Update progress bar
                event_count += 1
                if self._time_limit != float("inf"):
                    # Update based on time progress
                    progress = min(100, int((self._time / self._time_limit) * 100))
                    if progress > last_progress:
                        pbar.update(progress - last_progress)
                        last_progress = progress
                        pbar.set_postfix({'time': f'{self._time:.2f}s', 'events': event_count})
                else:
                    # Update based on event count
                    pbar.update(1)
                    pbar.set_postfix({'time': f'{self._time:.2f}s'})

        finally:
            pbar.close()

        if not self._scheduler.is_empty() and not self._terminate:
            logger.warning("Simulation ended but scheduler still has pending work. Draining...")

            # Let the scheduler process outstanding decode/prefill steps
            while not self._scheduler.is_empty():
                leftover_events = self._scheduler.step()
                
                # Safety check: if no events produced, scheduler is stuck; break to avoid infinite loop
                if not leftover_events:
                    logger.warning("Drain loop produced no events; breaking to avoid infinite loop")
                    break
                
                self._add_events(leftover_events)

                # ================================
                # DEBUG STATE DUMP FOR DRAIN LOOP
                # ================================
                print("\n==== GLOBAL EMPTY CHECK ====")
                print("global request_queue:", len(self._scheduler._request_queue))

                for rid, rs in self._scheduler._replica_schedulers.items():
                    print(f"Replica {rid}:")
                    print("  local_queue:", len(rs._priority_queue))
                    print("  allocations:", len(rs._allocation_map))
                    print("  migrations_out:", len(rs._migrations_out))
                    print("  running_batches:", rs._num_running_batches)
                    print("  reservations:", len(rs._reservations))
                    print("  is_empty():", rs.is_empty())

                print("event_queue:", len(self._event_queue))
                print("================================\n")



        logger.info(f"Simulation ended at: {self._time}s")

    def _write_output(self) -> None:
        logger.info("Writing output")

        self._metric_store.plot()
        logger.info("Metrics written")

        if self._config.metrics_config.write_json_trace:
            self._write_event_trace()
            logger.info("Json event trace written")

        if self._config.metrics_config.enable_chrome_trace:
            self._write_chrome_trace()
            logger.info("Chrome event trace written")

    def _add_event(self, event: BaseEvent) -> None:
        heapq.heappush(self._event_queue, (event._priority_number, event))

    def _add_events(self, events: List[BaseEvent]) -> None:
        for event in events:
            self._add_event(event)

    def _init_event_queue(self) -> None:
        requests = self._request_generator.generate()

        for request in requests:
            self._add_event(RequestArrivalEvent(request.arrived_at, request))
        
        # Initialize rebalancing and auto-scaling for Llumnix scheduler
        if self._config.cluster_config.global_scheduler_config.get_type() == GlobalSchedulerType.LLUMNIX:
            llumnix_config = self._config.cluster_config.global_scheduler_config
            if (hasattr(llumnix_config, 'enable_migration') and 
                llumnix_config.enable_migration and 
                self._config.cluster_config.num_replicas > 1):
                # Schedule first rebalance event
                initial_rebalance_time = llumnix_config.rebalance_interval
                self._add_event(RebalanceEvent(initial_rebalance_time))
                logger.info(
                    f"Llumnix rebalancing enabled with interval {llumnix_config.rebalance_interval}s"
                )
                
                # Schedule first auto-scale check event (every 1 second)
                # Includes warm-up period to prevent premature scale-in during load stabilization
                from vidur.events.autoscale_event import AutoScaleEvent
                autoscale_interval = getattr(llumnix_config, 'autoscale_interval', 1.0)
                autoscale_warmup = getattr(llumnix_config, 'autoscale_warmup_period', 5.0)
                max_replicas = self._config.cluster_config.num_replicas
                self._add_event(AutoScaleEvent(autoscale_interval, autoscale_interval, autoscale_warmup, max_replicas))
                logger.info(
                    f"Llumnix auto-scaling enabled with interval {autoscale_interval}s "
                    f"(scale_out at avgF<{llumnix_config.autoscale_low}, "
                    f"scale_in at avgF>{llumnix_config.autoscale_high}, "
                    f"warmup={autoscale_warmup}s, max_replicas={max_replicas})"
                )
            elif hasattr(llumnix_config, 'enable_migration') and llumnix_config.enable_migration:
                logger.warning(
                    f"Llumnix rebalancing disabled: requires at least 2 replicas (found {self._config.cluster_config.num_replicas})"
                )

    def _set_time(self, time: float) -> None:
        self._time = time
        if self._time > self._time_limit:
            logger.info(
                f"Time limit reached: {self._time_limit}s terminating the simulation."
            )
            self._terminate = True

    def _write_event_trace(self) -> None:
        trace_file = f"{self._config.metrics_config.output_dir}/event_trace.json"
        with open(trace_file, "w") as f:
            json.dump(self._event_trace, f)

    def _write_chrome_trace(self) -> None:
        trace_file = f"{self._config.metrics_config.output_dir}/chrome_trace.json"

        chrome_trace = {"traceEvents": self._event_chrome_trace}

        with open(trace_file, "w") as f:
            json.dump(chrome_trace, f)
