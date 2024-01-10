import json
import logging
from queue import PriorityQueue
from typing import List

from simulator.config import Config
from simulator.entities import Cluster
from simulator.events import BaseEvent, RequestArrivalEvent
from simulator.plotting import MetricsStore
from simulator.request_generator import RequestGeneratorRegistry
from simulator.scheduler import BaseGlobalScheduler, GlobalSchedulerRegistry

logger = logging.getLogger(__name__)


class Simulator:
    def __init__(self, config: Config) -> None:
        self._config = config

        self._time = 0
        self.terminate = False
        self._time_limit = self._config.simulator_time_limit
        self._event_queue = PriorityQueue()

        self._should_write_json_trace = self._config.write_json_trace
        self._should_write_chrome_trace = self._config.write_chrome_trace

        self._event_trace = []
        self._event_chrome_trace = []

        self._cluster = Cluster(self._config)
        self._metric_store = MetricsStore(self._config)
        self._request_generator = RequestGeneratorRegistry.get_from_str(
            self._config.request_generator_provider, self._config
        )
        self._scheduler = GlobalSchedulerRegistry.get_from_str(
            self._config.global_scheduler_provider, self._config, self._cluster.replicas
        )

        self._init_event_queue()

    @property
    def scheduler(self) -> BaseGlobalScheduler:
        return self._scheduler

    @property
    def metric_store(self) -> MetricsStore:
        return self._metric_store

    def run(self) -> None:
        logger.info(
            f"Starting simulation with cluster: {self._cluster} and {self._event_queue.qsize() - 1} requests"
        )

        while not self._event_queue.empty() and not self.terminate:
            event = self._event_queue.get()
            self._set_time(event.time)
            new_events = event.handle_event(self._scheduler, self._metric_store)
            self._add_events(new_events)
            if self._should_write_json_trace:
                self._event_trace.append(event.to_dict())

            if self._should_write_chrome_trace:
                chrome_trace = event.to_chrome_trace()
                if chrome_trace:
                    self._event_chrome_trace.append(chrome_trace)

        assert self._scheduler.is_empty()

        logger.info(f"Simulation ended at: {self._time}s")
        self._write_output()

    def _write_output(self) -> None:
        logger.info("Writing output")

        self._metric_store.plot()
        logger.info("Metrics written")

        if self._should_write_json_trace:
            self._write_event_trace()
            self._scheduler.write_batching_history()
            logger.info("Json event trace written")

        if self._should_write_chrome_trace:
            self._write_chrome_trace()
            logger.info("Chrome event trace written")

    def _add_event(self, event: BaseEvent) -> None:
        self._event_queue.put(event)

    def _add_events(self, events: List[BaseEvent]) -> None:
        for event in events:
            self._add_event(event)

    def _init_event_queue(self) -> None:
        requests = self._request_generator.generate()

        for request in requests:
            self._add_event(RequestArrivalEvent(request.arrived_at, request))

    def _set_time(self, time: float) -> None:
        self._time = time
        if self._time_limit and self._time > self._time_limit:
            self.terminate = True

    def _write_event_trace(self) -> None:
        trace_file = f"{self._config.output_dir}/event_trace.json"
        with open(trace_file, "w") as f:
            json.dump(self._event_trace, f)

    def _write_chrome_trace(self) -> None:
        trace_file = f"{self._config.output_dir}/chrome_trace.json"

        chrome_trace = {"traceEvents": self._event_chrome_trace}

        with open(trace_file, "w") as f:
            json.dump(chrome_trace, f)
