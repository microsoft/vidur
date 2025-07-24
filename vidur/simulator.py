import atexit
import heapq
import json
from typing import List

from vidur.config import SimulationConfig
from vidur.entities import Cluster
from vidur.events import BaseEvent, RequestArrivalEvent
from vidur.logger import init_logger
from vidur.metrics import MetricsStore
from vidur.request_generator import RequestGeneratorRegistry
from vidur.scheduler import BaseGlobalScheduler, GlobalSchedulerRegistry

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

        self._init_event_queue()
        atexit.register(self._write_output)

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

        while self._event_queue and not self._terminate:
            _, event = heapq.heappop(self._event_queue)
            self._set_time(event._time)
            new_events = event.handle_event(self._scheduler, self._metric_store)
            self._add_events(new_events)

            if self._config.metrics_config.write_json_trace:
                self._event_trace.append(event.to_dict())

            if self._config.metrics_config.enable_chrome_trace:
                chrome_trace = event.to_chrome_trace()
                if chrome_trace:
                    self._event_chrome_trace.append(chrome_trace)

        assert self._scheduler.is_empty() or self._terminate

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
