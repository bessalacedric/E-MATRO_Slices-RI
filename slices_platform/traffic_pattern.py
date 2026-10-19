"""
Traffic Pattern Driver
======================
Runs in a background thread during online training.
Controls iperf3 target bitrate per UE to produce realistic
load variations that force the MADRL to learn both
consolidation (low traffic) and scale-out (high traffic).

Pattern (cyclic, one cycle = 24 steps ≈ 24 minutes):
    steps  0- 4 : ramp up   (10 → 40 Mbps per RU)
    steps  5-10 : peak       (40 Mbps)
    steps 11-14 : ramp down  (40 → 10 Mbps)
    steps 15-20 : trough     (10 Mbps)
    steps 21-23 : ramp up again

Each RU can be offset by a phase shift so they don't all
peak simultaneously — more realistic heterogeneous load.
"""

import numpy as np
import threading
import time
import logging
from typing import Dict

log = logging.getLogger(__name__)

# Mbps per UE at each pattern level
MIN_MBPS  = 2.0
MID_MBPS  = 15.0
PEAK_MBPS = 40.0


def _diurnal_level(step: int, phase: int = 0, cycle: int = 24) -> float:
    """
    Returns target Mbps for one RU at a given step.
    phase offsets the RU so not all RUs peak together.
    """
    t = (step + phase) % cycle
    if t < 5:
        # ramp up
        return MIN_MBPS + (PEAK_MBPS - MIN_MBPS) * (t / 4.0)
    elif t < 11:
        # peak
        return PEAK_MBPS
    elif t < 15:
        # ramp down
        return PEAK_MBPS - (PEAK_MBPS - MIN_MBPS) * ((t - 10) / 4.0)
    else:
        # trough
        return MIN_MBPS


def get_target_mbps(step: int, n_rus: int, ues_per_ru: int = 3) -> np.ndarray:
    """
    Returns target UL Mbps per UE for each UE.
    Shape: (n_rus * ues_per_ru,)

    Each RU has a phase offset of cycle/n_rus to spread the peaks.
    """
    cycle       = 24
    phase_step  = cycle // n_rus
    targets     = np.zeros(n_rus * ues_per_ru, dtype=np.float32)
    for ru in range(n_rus):
        mbps_per_ru  = _diurnal_level(step, phase=ru * phase_step)
        mbps_per_ue  = mbps_per_ru / ues_per_ru
        for ue in range(ues_per_ru):
            targets[ru * ues_per_ru + ue] = mbps_per_ue
    return targets


class TrafficPatternDriver:
    """
    Runs a background thread that continuously cycles through
    the traffic pattern. The online training loop reads
    current_targets at each step before firing iperf3.

    Usage:
        driver = TrafficPatternDriver(n_rus=4, ues_per_ru=3)
        driver.start()
        # In training loop:
        targets = driver.get_targets()
        traffic_gen.run_traffic_step(target_mbps_per_ue=targets)
        driver.advance()   # move to next step
        driver.stop()
    """

    def __init__(self, n_rus: int = 4, ues_per_ru: int = 3):
        self.n_rus       = n_rus
        self.ues_per_ru  = ues_per_ru
        self._step       = 0
        self._lock       = threading.Lock()

    def get_targets(self) -> np.ndarray:
        with self._lock:
            return get_target_mbps(self._step, self.n_rus, self.ues_per_ru)

    def advance(self):
        with self._lock:
            self._step += 1

    @property
    def step(self):
        with self._lock:
            return self._step

    def ru_load_summary(self) -> Dict[int, float]:
        """Total expected Mbps per RU at current step."""
        targets = self.get_targets()
        return {
            r: float(np.sum(targets[r*self.ues_per_ru:(r+1)*self.ues_per_ru]))
            for r in range(self.n_rus)
        }