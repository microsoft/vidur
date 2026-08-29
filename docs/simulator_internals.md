# Vidur Simulator: Internal Architecture and Event-Driven Execution Overview

## Simulation Architecture

Vidur is a discrete event-driven simulator with a hierarchical scheduler architecture:

* Global Scheduler: Handles request routing and load balancing across replicas
* Replica Scheduler: Manages batching and memory allocation per replica
* Stage Scheduler: Controls pipeline stage execution within each replica

> Replica is an instance of LLM model deployed on one or more GPU devices

## Event Queue System

The simulator maintains a priority queue (`Simulator._event_queue`) that
sorts events based on arrival time, event type, and their id. To ensure proper
causality and deterministic execution order.

## Event Flow and Processing

**Request Arrival Event**

The simulation begins by adding a `RequestArrivalEvent` to the event queue.
By processing this event simulator:

* Creates a new `Request` object and adds it to the global scheduler
* Returns a `GlobalScheduleEvent` with the same timestamp as the request arrival (means that request arrival and global scheduler running happens instantaneously)

**Global Schedule Event**

The `GlobalScheduleEvent` triggers the global scheduler to create request
mappings - a list of tuples specifying which replica should process each
request in the schedulers queue.  Requests are added to global scheduler's
queue when `RequestArrivalEvent`.

The scheduler:

* Dequeues pending requests from its internal queue
* Assigns requests to replicas based on the configured scheduling policy (there are multiple)
* Adds assigned requests to each replica's local scheduler
* Returns `ReplicaScheduleEvent` events for each replica receiving new requests

**Replica Schedule Event**

A `ReplicaScheduleEvent` notifies a replica's scheduler to create execution
batches for each pipeline stage. The scheduling process is:

* Check there is memory available for processing request (e.g., there is a `KVCacheManager` class that tracks memory footprint)
* Creates `BatchStageArrivalEvent` events for processable batches
* Generates new `ReplicaScheduleEvent` events for requests that cannot be processed due to memory constraints

**Batch Stage Arrival Event**

This event adds batches to the replica's stage scheduler queue and creates corresponding `ReplicaStageScheduleEvent` events for each batch.

**Replica Stage Schedule Event**

The replica's stage scheduler processes this event by:

* Checking if the target stage is available (not busy)
* Using performance estimators (such as random forest models) to predict execution time
* Creating a `BatchStageEndEvent` scheduled after the estimated execution duration

**Batch Stage End Event**

Triggered when a pipeline stage completes processing a batch, this event:

* Generates a `ReplicaStageScheduleEvent` since the stage is available for processing new work (if there be enough memory or some sort of preemption)
* At the final pipeline stages: Creates a `BatchEndEvent` signaling batch completion
* At intermediate stages: Creates a `BatchStageArrivalEvent` for the next stage of the pipeline

**Batch End Event**

Occurs when a batch completes all pipeline stages. This event:

* Generates either a `PrefillEndEvent` (if transitioning from prefill to decode) or a `RequestEndEvent` (if request is complete)
* Creates a new `ReplicaScheduleEvent` which will cause the replica's scheduler to schedule new batches to be processed

This event removes request IDs from the replica scheduler's `scheduled_req_ids`
list so the new invocation schedule them for next iteration of going through
the model.

**Prefill End Event**

Marks the completion of the prefill phase of a request.

**Request End Event**

Signals completion of a request. This event updates performance metrics and cleans up associated resources.

