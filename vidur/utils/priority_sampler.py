"""
Priority sampler for request generation.

Maps distribution types to categorical sampling strategies aligned with
Llumnix priority-aware scheduling and headroom policies.
"""
import math
import random
from typing import List, Optional

from vidur.types import PriorityDistributionType


class PrioritySampler:
    """
    Samples request priorities from configurable distributions.
    
    Priority semantics (Llumnix-consistent):
    - 0 = critical (highest priority, largest headroom)
    - 1 = high
    - 2 = normal
    - 3 = low
    - 4 = background (lowest priority, no headroom)
    
    For 5-level configurations, default headrooms are:
    [2400, 1600, 0, 0, 0] tokens respectively.
    """
    
    def __init__(
        self,
        num_levels: int,
        distribution_type: int,
        custom_weights: Optional[List[float]] = None,
        seed: Optional[int] = None,
    ):
        """
        Args:
            num_levels: Number of priority levels (1-5 typical).
            distribution_type: PriorityDistributionType enum value.
            custom_weights: Optional custom weights (must sum to ~1.0).
            seed: Random seed for reproducibility.
        """
        self.num_levels = max(1, num_levels)
        self.distribution_type = distribution_type
        self.custom_weights = custom_weights
        self._counter = 0
        
        if seed is not None:
            random.seed(seed)
        
        # Build sampling weights
        self.weights = self._build_weights()
        
        # Cumulative distribution for efficient sampling
        self._build_cdf()
    
    def _build_weights(self) -> List[float]:
        """Build probability weights for each priority level."""
        if self.custom_weights is not None:
            # Validate and normalize
            if len(self.custom_weights) != self.num_levels:
                raise ValueError(
                    f"custom_weights length {len(self.custom_weights)} != num_levels {self.num_levels}"
                )
            total = sum(self.custom_weights)
            return [w / total for w in self.custom_weights]
        
        # Use distribution-specific defaults
        if self.distribution_type == PriorityDistributionType.ROUND_ROBIN:
            # Not used for round-robin, but provide uniform as fallback
            return [1.0 / self.num_levels] * self.num_levels
        
        elif self.distribution_type == PriorityDistributionType.UNIFORM:
            # Equal probability for all levels
            return [1.0 / self.num_levels] * self.num_levels
        
        elif self.distribution_type == PriorityDistributionType.NORMAL:
            # Gaussian-like: peak at middle priority
            if self.num_levels == 1:
                return [1.0]
            # General case: approximate normal via Gaussian weights
            # Works well for 2-10 priority levels
            mid = (self.num_levels - 1) / 2.0
            sigma = self.num_levels / 4.0  # Spread across ~4 sigma
            weights = []
            for i in range(self.num_levels):
                dist = (i - mid) / sigma
                weights.append(math.exp(-0.5 * dist * dist))
            total = sum(weights)
            return [w / total for w in weights]
        
        elif self.distribution_type == PriorityDistributionType.POWER_LAW:
            # Heavy tail: most requests at lower priorities, few at high
            # Works for 1-10 priority levels
            if self.num_levels == 1:
                return [1.0]
            # Power-law decay: P(priority=k) ∝ 1/(k+1)^α
            # α=1.5 gives good differentiation across priority levels
            alpha = 1.5
            weights = [1.0 / (i + 1) ** alpha for i in range(self.num_levels)]
            total = sum(weights)
            return [w / total for w in weights]
        
        elif self.distribution_type == PriorityDistributionType.ENTERPRISE:
            # Enterprise mix: most at middle priorities, some high, few critical
            # Works for 1-10 priority levels
            if self.num_levels == 1:
                return [1.0]
            elif self.num_levels == 2:
                return [0.25, 0.75]  # 25% critical, 75% normal
            # General case: exponential decay from middle toward extremes
            # Critical gets 10%, then exponential decay
            weights = []
            weights.append(0.10)  # Critical priority
            if self.num_levels > 2:
                # Middle priorities get bulk of traffic
                middle_weight = 0.70 / max(1, self.num_levels - 2)
                for i in range(1, self.num_levels - 1):
                    weights.append(middle_weight)
                # Lowest priority gets remainder
                weights.append(max(0.05, 1.0 - sum(weights)))
            else:
                weights.append(0.90)
            total = sum(weights)
            return [w / total for w in weights]
        
        elif self.distribution_type == PriorityDistributionType.BURSTIER:
            # Burstier mix: heavy concentration at middle, with some high-priority bursts
            # Works for 1-10 priority levels
            if self.num_levels == 1:
                return [1.0]
            elif self.num_levels == 2:
                return [0.30, 0.70]
            # General case: concentrate at middle priorities
            weights = []
            weights.append(0.10)  # Critical (burst)
            if self.num_levels > 2:
                weights.append(0.20)  # High (burst)
                # Middle/low priorities share remainder
                middle_weight = 0.70 / max(1, self.num_levels - 2)
                for i in range(2, self.num_levels):
                    weights.append(middle_weight)
            else:
                weights.append(0.90)
            total = sum(weights)
            return [w / total for w in weights]
        
        elif self.distribution_type == PriorityDistributionType.TIME_OF_DAY:
            # For now, default to enterprise; actual time-varying logic can be added in sample()
            if self.num_levels == 5:
                return [0.10, 0.30, 0.50, 0.08, 0.02]
            else:
                return self._build_weights_for_type(PriorityDistributionType.ENTERPRISE)
        
        elif self.distribution_type == PriorityDistributionType.TRAFFIC_CLASS:
            # Traffic class: heavy concentration on background/low priority
            # Works for 1-10 priority levels
            if self.num_levels == 1:
                return [1.0]
            # General case: inverse priority weighting (low priority = high probability)
            # Background (lowest) gets 60-75% of traffic
            weights = []
            background_weight = min(0.75, 0.60 + 0.05 * (self.num_levels - 2))
            high_priority_weight = (1.0 - background_weight) / max(1, self.num_levels - 1)
            for i in range(self.num_levels - 1):
                weights.append(high_priority_weight)
            weights.append(background_weight)  # Lowest priority
            total = sum(weights)
            return [w / total for w in weights]
        
        else:
            # Default: uniform
            return [1.0 / self.num_levels] * self.num_levels
    
    def _build_weights_for_type(self, dist_type: int) -> List[float]:
        """Helper to build weights for a specific distribution type (recursive fallback)."""
        old_type = self.distribution_type
        self.distribution_type = dist_type
        weights = self._build_weights()
        self.distribution_type = old_type
        return weights
    
    def _build_cdf(self):
        """Build cumulative distribution for efficient sampling."""
        self.cdf = []
        cumulative = 0.0
        for w in self.weights:
            cumulative += w
            self.cdf.append(cumulative)
        # Normalize to exactly 1.0 to avoid floating point issues
        if self.cdf:
            self.cdf[-1] = 1.0
    
    def sample(self, current_time: Optional[float] = None) -> int:
        """
        Sample a priority level.
        
        Args:
            current_time: Current simulation time (for time-varying distributions).
        
        Returns:
            Priority level (0 = highest, num_levels-1 = lowest).
        """
        if self.num_levels == 1:
            return 0
        
        if self.distribution_type == PriorityDistributionType.ROUND_ROBIN:
            # Cycle through levels
            priority = self._counter % self.num_levels
            self._counter += 1
            return priority
        
        # For time-of-day, adjust weights dynamically
        if (
            self.distribution_type == PriorityDistributionType.TIME_OF_DAY
            and current_time is not None
        ):
            # Simulate time-of-day variation:
            # - Peak hours (e.g., time % 100 in [40, 60]): more high-priority requests
            # - Off-peak: more background requests
            cycle_pos = (current_time % 100.0) / 100.0  # normalized to [0, 1)
            if 0.4 <= cycle_pos < 0.6:
                # Peak: shift toward high priority
                temp_weights = self._build_weights_for_type(PriorityDistributionType.ENTERPRISE)
            else:
                # Off-peak: shift toward lower priority
                temp_weights = self._build_weights_for_type(PriorityDistributionType.TRAFFIC_CLASS)
            
            # Build temp CDF
            cdf = []
            cumulative = 0.0
            for w in temp_weights:
                cumulative += w
                cdf.append(cumulative)
            if cdf:
                cdf[-1] = 1.0
            
            # Sample
            r = random.random()
            for i, cum_prob in enumerate(cdf):
                if r < cum_prob:
                    return i
            return self.num_levels - 1
        
        # Standard categorical sampling
        r = random.random()
        for i, cum_prob in enumerate(self.cdf):
            if r < cum_prob:
                return i
        
        # Fallback (should not reach here)
        return self.num_levels - 1
