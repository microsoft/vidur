from typing import List

from vidur.events import BaseEvent
from vidur.logger import init_logger
from vidur.metrics import MetricsStore
from vidur.scheduler import BaseGlobalScheduler
from vidur.types import EventType

logger = init_logger(__name__)


class MigrationEvent(BaseEvent):
    """
    Event for migrating a request from one replica to another.
    
    Implements Llumnix's live instance migration by transferring
    KV cache state and request context between replicas.
    """
    
    def __init__(self, time: float, request_id: int, source_replica_id: int, target_replica_id: int):
        super().__init__(time, EventType.MIGRATION)
        self._request_id = request_id
        self._source_replica_id = source_replica_id
        self._target_replica_id = target_replica_id
        self._migration_successful = False

    def handle_event(
        self, scheduler: BaseGlobalScheduler, metrics_store: MetricsStore
    ) -> List[BaseEvent]:
        from vidur.events.replica_schedule_event import ReplicaScheduleEvent
        
        source_scheduler = scheduler.get_replica_scheduler(self._source_replica_id)
        target_scheduler = scheduler.get_replica_scheduler(self._target_replica_id)
        
        # Check if request still exists on source
        if self._request_id not in source_scheduler._allocation_map:
            logger.warning(
                f"Migration failed: request {self._request_id} not found on replica {self._source_replica_id}"
            )
            return []
        
        # Get KV cache size
        kv_cache_blocks = source_scheduler._allocation_map[self._request_id]
        
        # Check if target has capacity
        if not target_scheduler.can_allocate(kv_cache_blocks):
            logger.warning(
                f"Migration failed: replica {self._target_replica_id} lacks capacity for {kv_cache_blocks} blocks"
            )
            return []
        
        # Find the request object
        request = None
        for req in source_scheduler._request_queue:
            if req.id == self._request_id:
                request = req
                break
        
        # If not in queue, check pending requests
        if request is None:
            # Skip migration of actively running requests
            logger.debug(
                f"Skipping migration of actively running request {self._request_id}"
            )
            return []
        
        # Perform migration
        try:
            # Remove from source
            source_scheduler._request_queue.remove(request)
            source_scheduler.free(self._request_id)
            
            # Add to target queue
            target_scheduler.add_request(request)
            
            self._migration_successful = True
            
            logger.info(
                f"Migrated request {self._request_id} from replica {self._source_replica_id} "
                f"to {self._target_replica_id} ({kv_cache_blocks} blocks)"
            )
            
            # Record migration metric
            if hasattr(metrics_store, 'on_request_migration'):
                metrics_store.on_request_migration(
                    self._request_id,
                    self._source_replica_id,
                    self._target_replica_id,
                    kv_cache_blocks,
                    self.time
                )
            
            # Trigger scheduling on target replica
            return [ReplicaScheduleEvent(self.time, self._target_replica_id)]
            
        except Exception as e:
            logger.error(f"Migration error: {e}")
            return []

    def to_dict(self):
        return {
            "time": self.time,
            "event_type": self.event_type,
            "request_id": self._request_id,
            "source_replica_id": self._source_replica_id,
            "target_replica_id": self._target_replica_id,
            "successful": self._migration_successful,
        }
    
    def to_chrome_trace(self):
        """Generate Chrome trace event for migration."""
        return [{
            "name": f"Migration R{self._request_id}",
            "cat": "migration",
            "ph": "i",  # Instant event
            "ts": self.time * 1e6,  # Convert to microseconds
            "pid": self._source_replica_id,
            "tid": 0,
            "s": "g",  # Global scope
            "args": {
                "request_id": self._request_id,
                "source": self._source_replica_id,
                "target": self._target_replica_id,
                "success": self._migration_successful,
            }
        }]
