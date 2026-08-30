from vidur.types.base_int_enum import BaseIntEnum


class PriorityDistributionType(BaseIntEnum):
    """
    Priority distribution types for request generation.
    
    - ROUND_ROBIN: Cycle through priority levels sequentially
    - UNIFORM: Equal probability for each priority level
    - NORMAL: Normal distribution centered on middle priority
    - POWER_LAW: Most requests at normal priority, few at high/critical
    - ENTERPRISE: 60% normal, 30% high, 10% critical (enterprise workload)
    - BURSTIER: 70% normal, 20% high, 10% critical (bursty workload)
    - TIME_OF_DAY: Vary distribution based on time simulation time
    - TRAFFIC_CLASS: 80% background, 15% normal, 5% high (web traffic)
    """
    ROUND_ROBIN = 1
    UNIFORM = 2
    NORMAL = 3
    POWER_LAW = 4
    ENTERPRISE = 5
    BURSTIER = 6
    TIME_OF_DAY = 7
    TRAFFIC_CLASS = 8
