from typing import List

from vidur.events import BaseEvent
from vidur.logger import init_logger
from vidur.metrics import MetricsStore
from vidur.scheduler import BaseGlobalScheduler
from vidur.types import EventType

logger = init_logger(__name__)


class RebalanceEvent(BaseEvent):
    """
    Event for periodic load rebalancing in Llumnix scheduler.
    
    Triggers the scheduler to check load imbalance and migrate requests
    between replicas if beneficial.
    """
    
    def __init__(self, time: float):
        super().__init__(time, EventType.REBALANCE)
        self._migrations = []

    def handle_event(
        self, scheduler: BaseGlobalScheduler, metrics_store: MetricsStore
    ) -> List[BaseEvent]:
        from vidur.events.migration_event import MigrationEvent
        
        # Check if scheduler supports rebalancing (Llumnix)
        if not hasattr(scheduler, 'should_rebalance'):
            return []
        
        # Always schedule the next rebalance check first
        events = []
        if hasattr(scheduler, '_rebalance_interval'):
            events.append(RebalanceEvent(self.time + scheduler._rebalance_interval))
        
        # Check if rebalancing should occur
        if not scheduler.should_rebalance(self.time):
            return events
        
        # Perform rebalancing
        self._migrations = scheduler.rebalance(self.time)
        
        logger.info(
            f"Rebalance at {self.time}s: {len(self._migrations)} migrations planned"
        )
        
        # Add migration events
        for request_id, source_id, target_id in self._migrations:
            events.append(MigrationEvent(self.time, request_id, source_id, target_id))
        
        return events

    def to_dict(self):
        return {
            "time": self.time,
            "event_type": self.event_type,
            "num_migrations": len(self._migrations),
            "migrations": [
                {
                    "request_id": req_id,
                    "source_replica": src,
                    "target_replica": tgt
                }
                for req_id, src, tgt in self._migrations
            ],
        }

    def to_chrome_trace(self):
        """
        Generate Chrome trace events for rebalancing decisions.
        Shows when and how many migrations were planned.
        """
        if not self._migrations:
            return []
        
        return [{
            "name": f"Rebalance ({len(self._migrations)} migrations)",
            "cat": "rebalance",
            "ph": "i",  # Instant event
            "ts": self.time * 1e6,
            "pid": -1,  # Special PID for global scheduler events
            "tid": 0,
            "s": "g",  # Global scope
            "args": {
                "num_migrations": len(self._migrations),
                "migrations": [
                    {
                        "request_id": req_id,
                        "source": src,
                        "target": tgt
                    }
                    for req_id, src, tgt in self._migrations
                ],
            }
        }]
