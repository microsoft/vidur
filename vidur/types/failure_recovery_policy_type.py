from vidur.types.base_int_enum import BaseIntEnum


class FailureRecoveryPolicyType(BaseIntEnum):
    NONE = 1
    STOP_AND_RESTART = 2
    FIXED_CHECKPOINTING = 3
    LUMEN = 4
