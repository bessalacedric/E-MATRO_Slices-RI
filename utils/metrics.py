"""
Metrics — helper functions for evaluation and comparison.
"""

import numpy as np
from typing import List, Dict


def compute_energy_savings(
    baseline_energy: float,
    policy_energy:   float,
) -> float:
    """Energy savings (%) relative to baseline."""
    if baseline_energy == 0:
        return 0.0
    return 100.0 * (baseline_energy - policy_energy) / baseline_energy


def compute_episode_stats(rewards: List[float]) -> Dict[str, float]:
    arr = np.array(rewards, dtype=np.float32)
    return {
        "mean":   float(arr.mean()),
        "std":    float(arr.std()),
        "min":    float(arr.min()),
        "max":    float(arr.max()),
        "total":  float(arr.sum()),
    }


def constraint_satisfaction_rate(constraint_log: List[dict], key: str) -> float:
    """Fraction of steps where a constraint was satisfied."""
    if not constraint_log:
        return 0.0
    return sum(1 for c in constraint_log if c.get(key, False)) / len(constraint_log)


def average_active_servers(node_state_log: List[np.ndarray]) -> float:
    """Mean number of active (state=2) servers over an episode."""
    if not node_state_log:
        return 0.0
    return float(np.mean([np.sum(s == 2) for s in node_state_log]))


def consolidation_ratio(
    x_ri_log: List[np.ndarray],   # list of x_ri_DU matrices
    num_nodes: int,
) -> float:
    """
    Mean fraction of nodes that are idle (no VNF hosted) over time.
    Higher = more consolidated.
    """
    if not x_ri_log:
        return 0.0
    ratios = []
    for x in x_ri_log:
        active = np.sum(x.sum(axis=0) > 0)
        ratios.append(1.0 - active / num_nodes)
    return float(np.mean(ratios))
