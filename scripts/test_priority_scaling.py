#!/usr/bin/env python3
"""Test script to verify priority headroom and distribution scaling."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vidur.utils.priority_sampler import PrioritySampler
from vidur.types import PriorityDistributionType


def test_headroom_calculation():
    """Test headroom calculation for different priority levels."""
    print("=" * 60)
    print("Testing Headroom Calculation")
    print("=" * 60)
    
    # Mock the calculation method
    import math
    
    def calculate_headroom(capacity, num_levels, decay_mode):
        if num_levels <= 0:
            return []
        if num_levels == 1:
            return [0]
        
        headroom = []
        if decay_mode == "linear":
            max_headroom_fraction = 0.25
            for p in range(num_levels):
                fraction = max_headroom_fraction * (1.0 - p / (num_levels - 1))
                headroom.append(int(capacity * fraction))
        elif decay_mode == "exponential":
            max_headroom_fraction = 0.30
            decay_constant = 2.5 / max(1, num_levels - 1)
            for p in range(num_levels):
                fraction = max_headroom_fraction * math.exp(-decay_constant * p)
                headroom.append(int(capacity * fraction))
        return headroom
    
    capacity = 1000  # Example: 1000 KV blocks
    
    for num_levels in [1, 2, 3, 5, 7, 10]:
        print(f"\n{num_levels} Priority Levels (Capacity={capacity}):")
        print("-" * 60)
        
        for mode in ["linear", "exponential"]:
            headroom = calculate_headroom(capacity, num_levels, mode)
            print(f"  {mode.capitalize():12} decay: {headroom}")
            if headroom:
                percentages = [f"{h/capacity*100:.1f}%" for h in headroom]
                print(f"  {'':12}         {percentages}")


def test_priority_distributions():
    """Test priority distribution for different numbers of levels."""
    print("\n" + "=" * 60)
    print("Testing Priority Distributions")
    print("=" * 60)
    
    distributions = [
        ("UNIFORM", PriorityDistributionType.UNIFORM),
        ("NORMAL", PriorityDistributionType.NORMAL),
        ("POWER_LAW", PriorityDistributionType.POWER_LAW),
        ("ENTERPRISE", PriorityDistributionType.ENTERPRISE),
    ]
    
    for num_levels in [1, 2, 3, 5, 7, 10]:
        print(f"\n{num_levels} Priority Levels:")
        print("-" * 60)
        
        for name, dist_type in distributions:
            sampler = PrioritySampler(
                num_levels=num_levels,
                distribution_type=dist_type,
                seed=42
            )
            
            # Show weights as percentages
            percentages = [f"{w*100:.1f}%" for w in sampler.weights]
            print(f"  {name:12}: {percentages}")


def test_sampling():
    """Test actual sampling from distributions."""
    print("\n" + "=" * 60)
    print("Testing Sampling (10000 samples)")
    print("=" * 60)
    
    num_levels = 5
    num_samples = 10000
    
    for dist_type in [PriorityDistributionType.NORMAL, PriorityDistributionType.POWER_LAW]:
        sampler = PrioritySampler(
            num_levels=num_levels,
            distribution_type=dist_type,
            seed=42
        )
        
        # Sample and count
        counts = [0] * num_levels
        for _ in range(num_samples):
            priority = sampler.sample()
            counts[priority] += 1
        
        # Show results
        dist_name = "NORMAL" if dist_type == PriorityDistributionType.NORMAL else "POWER_LAW"
        print(f"\n{dist_name} distribution ({num_levels} levels):")
        print(f"  Expected: {[f'{w*100:.1f}%' for w in sampler.weights]}")
        print(f"  Observed: {[f'{c/num_samples*100:.1f}%' for c in counts]}")


if __name__ == "__main__":
    test_headroom_calculation()
    test_priority_distributions()
    test_sampling()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
