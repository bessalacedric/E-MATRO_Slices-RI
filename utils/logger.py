"""
Logger — episode metrics to CSV + formatted console output.
"""

import csv
import os
import logging
from typing import Optional
import numpy as np


log = logging.getLogger(__name__)


class EpisodeLogger:
    """
    Logs per-episode metrics to a CSV file and prints formatted
    summaries to stdout matching the target evaluation format.
    """

    # FIELDS = [
    #     "episode", "avg_reward", "objective", "total_migrations",
    #     "avg_loss", "epsilon",
    #     "optimality_score", "link_capacity", "du_placement",
    #     "cu_placement", "load_balance",
    #     "max_link_util", "active_DUs", "active_CUs",
    #     "du_changes", "cu_changes", "platform",
    # ]

    FIELDS = [
        "episode", "avg_reward", "objective", "total_migrations",
        "avg_loss", "epsilon", "optimality_score", "node_states", "max_link_util",
        "rus_always_on", "DUs_placed", "CUs_placed", "raw_DU_changes", "raw_CU_changes", "total_raw_changes", "actual_migrations", "platform",
    ]



    def __init__(self, results_dir: str, filename: str = "training_log.csv"):
        os.makedirs(results_dir, exist_ok=True)
        path = os.path.join(results_dir, filename)
        self._f      = open(path, "w", newline="")
        self._writer = csv.DictWriter(self._f, fieldnames=self.FIELDS, extrasaction="ignore")
        self._writer.writeheader()
        log.info(f"Logging to {path}")

    def log(self, row: dict):
        self._writer.writerow(row)
        self._f.flush()

    def close(self):
        self._f.close()

    @staticmethod
    def print_episode(
        ep:          int,
        n_episodes:  int,
        rewards:     list,
        losses:      list,
        migrations:  list,
        energy:      float,
        constraints: dict,
        du_changes:  float,
        cu_changes:  float,
        epsilon:     float,
    ):
        
        yn      = lambda v: "Yes" if v else "No"
        avg_r   = float(np.mean(rewards))  if rewards  else 0.0
        avg_l   = float(np.mean(losses))   if losses   else 0.0
        tot_mig = float(sum(migrations))

        print(
            f"\nEpisode: {ep}/{n_episodes}, "
            f"Average Reward: {avg_r:.2f}, "
            f"Objective: {energy:.4f}, "
            f"Total_migrations : {tot_mig:.1f}, "
            f"Average Loss: {avg_l:.4f} , "
            f"Epsilon: {epsilon:.2f}"
        )
        print("Network State Analysis:")
        print(f"  Optimality Score: {constraints.get('optimality_score', 0):.2f}")
        print(f"  link_capacity: {yn(constraints.get('link_capacity', False))}")
        print(f"  du_placement:  {yn(constraints.get('du_placement',  False))}")
        print(f"  cu_placement:  {yn(constraints.get('cu_placement',  False))}")
        print(f"  load_balance:  {yn(constraints.get('load_balance',  False))}")
        print("Detailed Metrics:")
        print(f"  Max Link Utilization: {constraints.get('max_link_util', 0):.2f}")
        print(f"  Active DUs: {constraints.get('active_DUs', 0)}")
        print(f"  Active CUs: {constraints.get('active_CUs', 0)}")
        print(f"Episode {ep}:")
        print(f"  Raw DU changes: {du_changes:.1f}")
        print(f"  Raw CU changes: {cu_changes:.1f}")
        print(f"  Total raw changes: {du_changes + cu_changes:.1f}")
        print(f"  Actual migrations: {tot_mig:.1f}")
