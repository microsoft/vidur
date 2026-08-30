from typing import List

from vidur.events import BaseEvent
from vidur.logger import init_logger
from vidur.metrics import MetricsStore
from vidur.scheduler import BaseGlobalScheduler
from vidur.types import EventType

logger = init_logger(__name__)


class AutoScaleEvent(BaseEvent):
    """
    Periodic event that checks cluster load and triggers auto-scaling decisions.
    
    When average freeness crosses thresholds:
    - avgF < autoscale_low: Scale out (add replicas)
    - avgF > autoscale_high: Scale in (drain replicas)
    
    Currently only supports scale-in (draining existing replicas).
    
    Includes warm-up period to prevent premature scale-in decisions before
    load has properly distributed across replicas.
    """
    
    # Class variable to track simulation start time (shared across all instances)
    _simulation_start_time = None
    # Track maximum replicas (set from initial config) and drained replicas pool
    _max_replicas = None
    _drained_replicas_pool = []  # Stack of drained replica IDs available for restoration
    # Cooldown tracking to prevent thrashing
    _last_scale_in_time = None
    _last_scale_out_time = None
    _scale_cooldown = 10.0  # Minimum seconds between scale operations
    
    def __init__(self, time: float, interval: float = 1.0, warmup_period: float = 5.0, max_replicas: int = None):
        super().__init__(time, EventType.AUTOSCALE)
        self._interval = interval
        self._warmup_period = warmup_period  # Seconds to wait before allowing scale-in
        self._is_scaling_in = False
        self._is_scaling_out = False
        self._draining_replicas = []
        self._restored_replicas = []  # Track restored replicas for Chrome trace
        
        # Track simulation start time on first instance
        if AutoScaleEvent._simulation_start_time is None:
            AutoScaleEvent._simulation_start_time = time
        
        # Set maximum replicas from initial configuration
        if AutoScaleEvent._max_replicas is None and max_replicas is not None:
            AutoScaleEvent._max_replicas = max_replicas
            logger.info(f"[AutoScale] Maximum replica cap set to {max_replicas}")
    
    def handle_event(
        self, scheduler: BaseGlobalScheduler, metrics_store: MetricsStore
    ) -> List[BaseEvent]:
        """
        Check autoscaling conditions and trigger draining if needed.
        
        Scale-in is blocked during warm-up period to allow load to stabilize.
        """
        # Only Llumnix scheduler supports autoscaling
        if not hasattr(scheduler, 'autoscale_recommendation'):
            return [AutoScaleEvent(self.time + self._interval, self._interval, self._warmup_period, AutoScaleEvent._max_replicas)]
        
        # Calculate elapsed time since simulation start
        elapsed_time = self.time - AutoScaleEvent._simulation_start_time
        
        # Get autoscale recommendation
        recommendation = scheduler.autoscale_recommendation()
        
        if recommendation == "scale_in":
            # Get normal-priority freeness for decision-making
            normal_freeness = scheduler._all_normal_priority_freeness()
            avg_freeness = sum(f for _, f in normal_freeness) / len(normal_freeness)
            
            # Check if still in warm-up period
            if elapsed_time < self._warmup_period:
                logger.debug(
                    f"[AutoScale] Scale-in blocked during warm-up period "
                    f"(elapsed={elapsed_time:.1f}s < warmup={self._warmup_period}s, "
                    f"avgF_normal={avg_freeness:.3f})"
                )
            # Check cooldown period
            elif (AutoScaleEvent._last_scale_in_time is not None and 
                  self.time - AutoScaleEvent._last_scale_in_time < AutoScaleEvent._scale_cooldown):
                logger.debug(
                    f"[AutoScale] Scale-in blocked by cooldown "
                    f"(last={AutoScaleEvent._last_scale_in_time:.1f}s, "
                    f"cooldown={AutoScaleEvent._scale_cooldown}s)"
                )
            # Don't scale-in if system is actually overloaded (prevents thrashing)
            elif avg_freeness < 0:
                logger.debug(
                    f"[AutoScale] Scale-in blocked: system overloaded "
                    f"(avgF_normal={avg_freeness:.3f} < 0)"
                )
            else:
                # Paper-compliant scale-in: select instance with fewest running requests
                # Per Section 4.4.3: "Llumnix chooses the instance with fewest running requests for termination"
                running_counts = scheduler._all_running_request_counts()
                if running_counts:
                    # Sort by running request count (ascending) and drain the one with fewest
                    running_counts.sort(key=lambda x: x[1])
                    fewest_requests_rid, request_count = running_counts[0]
                    
                    # Only trigger draining if there's capacity elsewhere
                    if len(running_counts) > 1:
                        scheduler.set_draining([fewest_requests_rid], draining=True)
                        self._draining_replicas = [fewest_requests_rid]
                        self._is_scaling_in = True
                        
                        # Add drained replica to pool for potential restoration
                        AutoScaleEvent._drained_replicas_pool.append(fewest_requests_rid)
                        
                        # Record scale-in timestamp
                        AutoScaleEvent._last_scale_in_time = self.time
                        
                        logger.info(
                            f"[AutoScale] Scale-in triggered: Replica {fewest_requests_rid} "
                            f"marked for draining ({request_count} running requests, "
                            f"avgF_normal={avg_freeness:.3f}, high={scheduler._autoscale_high}, "
                            f"drained pool size: {len(AutoScaleEvent._drained_replicas_pool)})"
                        )
        
        elif recommendation == "scale_out":
            # Use normal-priority freeness for logging (paper-compliant metric)
            normal_freeness = scheduler._all_normal_priority_freeness()
            avg_freeness = sum(f for _, f in normal_freeness) / len(normal_freeness)
            
            # Check cooldown period
            if (AutoScaleEvent._last_scale_out_time is not None and 
                self.time - AutoScaleEvent._last_scale_out_time < AutoScaleEvent._scale_cooldown):
                logger.debug(
                    f"[AutoScale] Scale-out blocked by cooldown "
                    f"(last={AutoScaleEvent._last_scale_out_time:.1f}s, "
                    f"cooldown={AutoScaleEvent._scale_cooldown}s, avgF_normal={avg_freeness:.3f})"
                )
            # Check if we have drained replicas available for restoration
            elif AutoScaleEvent._drained_replicas_pool:
                # Mark for trace emission
                self._is_scaling_out = True
                # Restore a previously drained replica
                restored_rid = AutoScaleEvent._drained_replicas_pool.pop()
                
                # Un-drain the replica in the scheduler
                if hasattr(scheduler, 'set_draining'):
                    scheduler.set_draining([restored_rid], draining=False)
                    
                    # Record scale-out timestamp and restored replica
                    AutoScaleEvent._last_scale_out_time = self.time
                    self._restored_replicas = [restored_rid]
                    
                    logger.info(
                        f"[AutoScale] Scale-out executed: Restored replica {restored_rid} "
                        f"(avgF_normal={avg_freeness:.3f}, low={scheduler._autoscale_low}, "
                        f"active replicas: {len(scheduler._replicas)}, "
                        f"drained pool: {len(AutoScaleEvent._drained_replicas_pool)})"
                    )
                else:
                    logger.warning(
                        f"[AutoScale] Cannot restore replica {restored_rid}: "
                        f"scheduler lacks set_draining method"
                    )
            else:
                # No drained replicas available - we're at maximum capacity
                # Only log occasionally to avoid spam
                if AutoScaleEvent._last_scale_out_time is None or self.time - AutoScaleEvent._last_scale_out_time > 30.0:
                    logger.warning(
                        f"[AutoScale] Scale-out recommended at avgF_normal={avg_freeness:.3f} "
                        f"(low={scheduler._autoscale_low}) but already at maximum capacity "
                        f"({AutoScaleEvent._max_replicas} replicas, drained pool empty)"
                    )
                    AutoScaleEvent._last_scale_out_time = self.time  # Prevent spam
        
        # Schedule next autoscale check (preserve warmup_period and max_replicas)
        return [AutoScaleEvent(self.time + self._interval, self._interval, self._warmup_period, AutoScaleEvent._max_replicas)]
    
    def to_dict(self):
        recommendation = "none"
        if self._is_scaling_in:
            recommendation = "scale_in"
        elif self._is_scaling_out:
            recommendation = "scale_out"
        
        return {
            "time": self.time,
            "event_type": self.event_type,
            "recommendation": recommendation,
            "draining_replicas": self._draining_replicas,
            "restored_replicas": self._restored_replicas,
        }
    
    def to_chrome_trace(self):
        """
        Emit autoscale events to Chrome trace for visibility.
        """
        events = []
        
        if self._is_scaling_in:
            events.append({
                "name": f"AutoScale: Scale-In (Drain Replica {self._draining_replicas[0]})",
                "cat": "autoscale",
                "ph": "i",  # Instant event
                "ts": self.time * 1e6,
                "pid": -1,  # Global scope
                "tid": 0,
                "s": "g",
                "args": {
                    "action": "scale_in",
                    "draining_replicas": self._draining_replicas,
                }
            })
        
        if self._is_scaling_out:
            if self._restored_replicas:
                # Scale-out executed: restored a drained replica
                events.append({
                    "name": f"AutoScale: Scale-Out (Restore Replica {self._restored_replicas[0]})",
                    "cat": "autoscale",
                    "ph": "i",  # Instant event
                    "ts": self.time * 1e6,
                    "pid": -1,  # Global scope
                    "tid": 0,
                    "s": "g",
                    "args": {
                        "action": "scale_out",
                        "restored_replicas": self._restored_replicas,
                        "note": "Restored previously drained replica"
                    }
                })
            else:
                # Scale-out recommended but at max capacity
                events.append({
                    "name": "AutoScale: Scale-Out Blocked (Max Capacity)",
                    "cat": "autoscale",
                    "ph": "i",  # Instant event
                    "ts": self.time * 1e6,
                    "pid": -1,  # Global scope
                    "tid": 0,
                    "s": "g",
                    "args": {
                        "action": "scale_out_blocked",
                        "note": "No drained replicas available to restore"
                    }
                })
        
        return events
