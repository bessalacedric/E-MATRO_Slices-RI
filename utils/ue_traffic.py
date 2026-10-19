"""
UE Traffic Generator
====================
Models 6 RUs each serving 3 UEs over a 24-hour period (1440 steps at 1 min).

Each UE follows a 3-state Markov chain:
    IDLE       — no traffic (browsing pause, screen off)
    BACKGROUND — low-rate traffic (push notifications, sync)
    ACTIVE     — full session (video, file transfer, call)

State transition probabilities follow a diurnal pattern:
    Night (00–07h): high idle probability
    Morning (07–12h): increasing activity
    Peak (12–22h): sustained high activity
    Late night (22–24h): tapering off

UL bitrate per state (kbps, realistic 5G NR values):
    IDLE:       0 – 10
    BACKGROUND: 50 – 500
    ACTIVE:     1000 – 50000  (1–50 Mbps)

Output: np.ndarray shape (T, N_rus) — total UL bitrate per RU in Mbps.
"""

import numpy as np
from typing import Optional

# UE states
IDLE       = 0
BACKGROUND = 1
ACTIVE     = 2

# UL bitrate ranges per state (kbps)
RATE_RANGES = {
    IDLE:       (0,     10),
    BACKGROUND: (50,    500),
    ACTIVE:     (1000,  50000),
}


def _transition_matrix(hour: int) -> np.ndarray:
    """
    3×3 Markov transition matrix P[i,j] = P(next=j | current=i)
    varies with time of day to produce diurnal traffic patterns.

    Night   00–06: mostly idle
    Morning 06–09: waking up, moderate activity
    Day     09–22: peak activity
    Evening 22–24: winding down
    """
    if 0 <= hour < 6:        # night
        return np.array([
            [0.95, 0.04, 0.01],   # idle → mostly stay idle
            [0.30, 0.65, 0.05],   # background → often return to idle
            [0.40, 0.40, 0.20],   # active → quickly return to idle
        ])
    elif 6 <= hour < 9:      # morning ramp-up
        return np.array([
            [0.80, 0.12, 0.08],
            [0.20, 0.60, 0.20],
            [0.15, 0.30, 0.55],
        ])
    elif 9 <= hour < 22:     # peak hours
        return np.array([
            [0.70, 0.15, 0.15],   # idle → often becomes active
            [0.10, 0.55, 0.35],   # background → often becomes active
            [0.10, 0.20, 0.70],   # active → tends to stay active
        ])
    else:                    # late evening
        return np.array([
            [0.85, 0.10, 0.05],
            [0.25, 0.60, 0.15],
            [0.25, 0.35, 0.40],
        ])


def _sample_rate(state: int, rng: np.random.Generator) -> float:
    """Sample UL bitrate (kbps) for a given UE state."""
    lo, hi = RATE_RANGES[state]
    if state == IDLE:
        return float(rng.uniform(lo, hi))
    elif state == BACKGROUND:
        return float(rng.uniform(lo, hi))
    else:
        # Active: log-uniform to get realistic heavy-tail distribution
        return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def generate_ue_traffic(
    n_rus:     int   = 6,
    ues_per_ru: int  = 3,
    T:         int   = 1440,   # minutes (1 day)
    seed:      Optional[int] = 42,
) -> np.ndarray:
    """
    Generate per-RU UL traffic by simulating UE state machines.

    Parameters
    ----------
    n_rus      : number of RUs
    ues_per_ru : number of UEs per RU
    T          : number of time steps (minutes)
    seed       : RNG seed for reproducibility

    Returns
    -------
    traffic : np.ndarray shape (T, n_rus), UL bitrate per RU in Mbps
    """
    rng     = np.random.default_rng(seed)
    traffic = np.zeros((T, n_rus), dtype=np.float32)

    for ru in range(n_rus):
        ru_traffic = np.zeros(T, dtype=np.float32)

        for ue in range(ues_per_ru):
            # Each UE starts in a random state weighted by time-of-day
            state = int(rng.choice([IDLE, BACKGROUND, ACTIVE],
                                    p=[0.6, 0.3, 0.1]))
            ue_trace = np.zeros(T, dtype=np.float32)

            for t in range(T):
                hour = (t // 60) % 24
                P    = _transition_matrix(hour)

                # Sample rate for current state
                rate        = _sample_rate(state, rng)
                ue_trace[t] = rate

                # Transition to next state
                state = int(rng.choice([IDLE, BACKGROUND, ACTIVE],
                                        p=P[state]))

            ru_traffic += ue_trace

        # Convert kbps → Mbps
        traffic[:, ru] = ru_traffic / 1e3

    return traffic


def generate_multi_episode_traffic(
    n_rus:      int  = 6,
    ues_per_ru: int  = 3,
    T:          int  = 1440,
    n_episodes: int  = 1,
    base_seed:  int  = 42,
) -> list:
    """
    Generate a list of traffic matrices, one per episode.
    Each episode gets a different seed so traffic varies across episodes.

    Returns: list of np.ndarray shape (T, n_rus)
    """
    return [
        generate_ue_traffic(n_rus, ues_per_ru, T, seed=base_seed + ep)
        for ep in range(n_episodes)
    ]