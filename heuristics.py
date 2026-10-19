"""
heuristics.py — Static and random baseline policies for comparison
===========================================================================
All three heuristics use the same OCloudEnv environment as. 
They are evaluated in inference mode only (no training)
by running N_EVAL episodes and logging the same metrics as the learned methods.

Usage:
  python3 heuristics.py --heuristic group_all --episodes 1000 --max-steps 10
  python3 heuristics.py --heuristic split_all --episodes 1000 --max-steps 10
  python3 heuristics.py --heuristic random    --episodes 1000 --max-steps 10
  python3 heuristics.py --heuristic all       --episodes 1000 --max-steps 10
"""

import os
import csv
import random
import argparse
import numpy as np

from env_ocloud import OCloudEnv, N_NODES, N_RUS


# =============================================================================
# Helpers
# =============================================================================

class CSVLogger:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.f      = open(path, "w", newline="")
        self.writer = None

    def log(self, row: dict):
        if self.writer is None:
            self.writer = csv.DictWriter(self.f, fieldnames=list(row.keys()))
            self.writer.writeheader()
        self.writer.writerow(row)
        self.f.flush()

    def close(self):
        self.f.close()


def make_env(cfg: dict) -> OCloudEnv:
    return OCloudEnv(
        ssh_user         = cfg["ssh_user"],
        max_steps        = cfg["max_steps"],
        use_real_traffic = False,
        step_interval    = 0.0,
    )


def collect_episode_metrics(info: dict, rewards: np.ndarray, flat_actions: np.ndarray, per_agent_act_dim: int, ep_store: dict):
    """Accumulate per-step metrics into episode lists."""
    CAPACITY = 0.85

    node_loads   = info["node_loads"]
    nodes_active = max(info["nodes_active"], 1)
    ocloud_util  = min(sum(node_loads) / nodes_active, 1.0)

    n_proposed = sum( 1 for n in range(N_NODES) if any(flat_actions[n*per_agent_act_dim + 1:
                                    n*per_agent_act_dim + per_agent_act_dim] != n))   # agents proposing at least one migration

    n_executed = info.get("n_migrations", 0)
    n_failed   = info.get("n_failed_mig", 0)
    n_conflicts = info.get("n_conflicts", 0)

    n_delay_viol = info.get("delay_violation", 0.0)
    n_delay_sla_ok = info.get("delay_sla_ok", 0.0)

    active_delays = [d for d in info["node_delays_ms"] if d < float('inf')]
    n_active_delays = (max(active_delays) if active_delays else 0.0)

    success_rate = (n_executed / n_proposed if n_proposed > 0 else 1.0)
    

    eff = info["energy_saving"] / (ocloud_util + 1e-6)

    ep_store["rewards"].append(float(np.mean(rewards)))
    ep_store["saving"].append(info["energy_saving"])
    ep_store["avg_energy"].append(info["instant_energy_W"])
    ep_store["nodes"].append(info["nodes_active"])
    ep_store["ocloud_util"].append(ocloud_util)
    ep_store["energy_eff"].append(info["energy_saving"] * ocloud_util)
    ep_store["migrations"].append(info.get("n_migrations", 0))
    ep_store["conflicts"].append(info.get("n_conflicts", 0))
    ep_store["failed"].append(info.get("n_failed_mig", 0))
    ep_store["cap_viol"].append(
        sum(1 for l in node_loads if l > CAPACITY)
    )
    ep_store["peak_load"].append(max(node_loads) if node_loads else 0.0)
    ep_store["qos_viol"].append(info.get("qos_violation", 0.0))
    ep_store["energy_eff"].append(eff)

    ep_store["ep_proposed"].append(n_proposed)
    ep_store["ep_migration_success"].append(success_rate)
    ep_store["ep_conflicts_step"].append(n_conflicts)

    ep_store["ep_delay_viol"].append(n_delay_viol)
    ep_store["ep_delay_sla_ok"].append(n_delay_sla_ok)
    ep_store["ep_active_delay"].append(n_active_delays)
    
def episode_summary(ep_store: dict, ep: int, epsilon: float = 0.0) -> dict:
    return {
        "episode":                ep,
        "avg_reward":             round(float(np.mean(ep_store["rewards"])),   6),
        "energy_saving_%":        round(float(np.mean(ep_store["saving"]))*100, 2),
        "avg_energy":             round(float(np.mean(ep_store["avg_energy"])), 6),
        "avg_nodes_active":       round(float(np.mean(ep_store["nodes"])),     2),
        "ocloud_utilization":     round(float(np.mean(ep_store["ocloud_util"])),4),
        "energy_efficiency":      round(float(np.mean(ep_store["energy_eff"])),4),
        "total_migrations":       int(sum(ep_store["migrations"])),
        "failed_migrations":      int(sum(ep_store["failed"])),
        "total_conflicts":        int(sum(ep_store["conflicts"])),
        "capacity_violations":    int(sum(ep_store["cap_viol"])),
        "capacity_violation_rate":round(
            sum(1 for v in ep_store["cap_viol"] if v > 0)
            / max(len(ep_store["cap_viol"]), 1), 4),
        "peak_load":         round(float(np.max(ep_store["peak_load"])),  4),
        "avg_qos_violation":      round(float(np.mean(ep_store["qos_viol"])),  4),
        "epsilon":                epsilon,

        "delay_violation_rate": round(float(np.mean(ep_store["ep_delay_sla_ok"]) < 1.0), 4),
        "avg_delay_violation":  round(float(np.mean(ep_store["ep_delay_viol"])),  4),
        "worst_delay_ms":       round(float(np.max(ep_store["ep_active_delay"])),  4),

        "conflict_rate":        round(float(np.mean(ep_store["ep_conflicts_step"])) / max(float(np.mean(ep_store["ep_proposed"])), 1e-6), 4),
        "migration_success_rate": round(float(np.mean(ep_store["ep_migration_success"])), 4),

    }


# =============================================================================
# Heuristic 1 — Group-All
# =============================================================================

class GroupAllPolicy:
    """
    Static policy: at the start of each episode, consolidate ALL VNFs
    onto the single node with the lowest current load (node 0 by default
    since loads are initialized identically). Keep all VNFs there for
    the entire episode — never migrate, never wake sleeping nodes.

    Energy saving: maximum (only 1 node active)
    QoS risk:      high (all load on one node, may exceed C=0.85)
    Migrations:    0 per episode after initial placement
    """

    def __init__(self, env: OCloudEnv):
        self.env              = env
        self.per_agent_act_dim = env.per_agent_act_dim  # 1 + 2*N_RUS
        self.target_node      = 0   # always consolidate onto node 0

        self.initialized       = False

    def reset(self):
        """Call at the start of each episode."""
        self.initialized = False


    

    def act(self, obs: np.ndarray, info: dict) -> np.ndarray:
        """
        For each agent i:
          a_i = ON  if i == target_node else OFF
          m_DU[r] = target_node for all r  (migrate everything to target)
          m_CU[r] = target_node for all r
        After the first step the placement is stable and no migrations occur.
        """
        flat = np.zeros(N_NODES * self.per_agent_act_dim, dtype=np.int64)
        for n in range(N_NODES):
            base = n * self.per_agent_act_dim
            # Activation: only target node stays ON
            flat[base] = 1 if n == self.target_node else 0
            # All DU and CU migrations point to target node
            for k in range(1, self.per_agent_act_dim):
                flat[base + k] = self.target_node
        return flat


# =============================================================================
# Heuristic 2 — Split-All
# =============================================================================

class SplitAllPolicy:
    """
    Static policy: keep ALL nodes active and distribute VNFs as evenly
    as possible — each node hosts at most ceil(N_RUS/N_NODES) DUs and
    ceil(N_RUS/N_NODES) CUs. Never migrates after the initial placement.

    Energy saving: zero (all nodes always active)
    QoS risk:      minimal (load spread thin across all nodes)
    Migrations:    0 per episode after initial placement
    """

    def __init__(self, env: OCloudEnv):
        self.env               = env
        self.per_agent_act_dim = env.per_agent_act_dim
        self.initialized       = False

        # Pre-compute target node for each RU's DU and CU
        # Round-robin assignment: DU r → node (r % N_NODES)
        #                         CU r → node ((r + N_NODES//2) % N_NODES)
        self.du_target = {r: r % N_NODES           for r in range(N_RUS)}
        self.cu_target = {r: (r + N_NODES//2) % N_NODES for r in range(N_RUS)}

    def reset(self):
        """Call at the start of each episode."""
        self.initialized = False


    def act(self, obs: np.ndarray, info: dict) -> np.ndarray:
        """
        For each agent i:
          a_i = ON  (all nodes always active)
          m_DU[r] = du_target[r]  (spread DUs round-robin)
          m_CU[r] = cu_target[r]  (spread CUs round-robin, offset)
        """
        flat = np.zeros(N_NODES * self.per_agent_act_dim, dtype=np.int64)
        for n in range(N_NODES):
            base = n * self.per_agent_act_dim
            flat[base] = 1  # all nodes active
            for r in range(N_RUS):
                flat[base + 1 + r]         = self.du_target[r]
                flat[base + 1 + N_RUS + r] = self.cu_target[r]
        return flat


# =============================================================================
# Heuristic 3 — Random
# =============================================================================

class RandomPolicy:
    """
    Random policy: at each step independently sample a uniformly random
    valid action for each node — random activation and random migration
    destinations for each VNF. No learning, no memory.

    Energy saving: ~25-50% on average by chance
    QoS risk:      variable — random migrations often conflict
    Migrations:    high and random
    """

    def __init__(self, env: OCloudEnv):
        self.env               = env
        self.per_agent_act_dim = env.per_agent_act_dim

    def reset(self):
        """Call at the start of each episode."""
        pass  # no internal state to reset

    def act(self, obs: np.ndarray, info: dict) -> np.ndarray:
        """Sample directly from the environment action space."""
        return self.env.action_space.sample()


# =============================================================================
# Evaluation loop — shared across all heuristics
# =============================================================================

def evaluate(policy, env: OCloudEnv, n_episodes: int,
             logger: CSVLogger, name: str):

    per_agent_act_dim = env.per_agent_act_dim
    print(f"\n{'='*50}")
    print(f"  Evaluating: {name}")
    print(f"  Episodes  : {n_episodes}")
    print(f"  Nodes     : {N_NODES}  |  RUs: {N_RUS}")
    print(f"{'='*50}")

    for ep in range(1, n_episodes + 1):
        obs, info = env.reset()

        #policy.reset()  # reset any internal state if needed

        ep_store = {k: [] for k in [
            "rewards", "saving", "nodes", "ocloud_util",
            "energy_eff", "migrations", "conflicts",
            "failed", "cap_viol", "peak_load", "qos_viol",  "avg_energy" , "ep_proposed", "ep_migration_success", 
            "ep_conflicts_step", "ep_delay_viol", "ep_delay_sla_ok", "ep_active_delay", 
        ]}

        for step in range(env.max_steps):
            flat_actions = policy.act(obs, info)
            next_obs, rewards, done, trunc, info = env.step(flat_actions)
            obs = next_obs

            collect_episode_metrics(
                info, rewards, flat_actions,
                per_agent_act_dim, ep_store
            )
            if done or trunc:
                break

        row = episode_summary(ep_store, ep)
        logger.log(row)

        if ep % 100 == 0:
            print(
                f"  ep {ep:5d}/{n_episodes} | "
                f"reward={row['avg_reward']:+.4f} | "
                f"saving={row['energy_saving_%']:.1f}% | "
                f"util={row['ocloud_utilization']:.3f} | "
                f"cap_viol={row['capacity_violations']}"
            )

    print(f"  Done. Results saved.")


# =============================================================================
# Entry point
# =============================================================================

DEFAULT_CFG = {
    "ssh_user":  "ubuntu",
    "max_steps": 10,
}

HEURISTICS = {
    "group_all": (GroupAllPolicy, "group_all_training_log_8rus_6nodes_10000_episodes.csv"),
    "split_all": (SplitAllPolicy, "split_all_training_log_8rus_6nodes_10000_episodes.csv"),
    "random":    (RandomPolicy,   "random_training_log_8rus_6nodes_10000_episodes.csv"),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate static/random heuristic baselines"
    )
    parser.add_argument("--heuristic",   default="all",
                        choices=["group_all","split_all","random","all"])
    parser.add_argument("--episodes",    type=int, default=1000)
    parser.add_argument("--max-steps",   type=int, default=10)
    parser.add_argument("--n-rus",       type=int, default=None)
    parser.add_argument("--results-dir", default="data_comparison/results_8rus_6nodes") 
    args = parser.parse_args()

    # Topology override
    if args.n_rus is not None:
        import topology as _t
        _t.N_RUS   = args.n_rus
        _t.N_DUS   = max(2, args.n_rus // 2)
        _t.N_CUS   = max(2, (args.n_rus + 3) // 4)
        _t.N_NODES = min(_t.N_DUS + _t.N_CUS, _t.N_VMS)
        _t.RUS_PER_DU = _t.N_RUS // _t.N_DUS
        _t.NODE_CONFIGS          = _t.build_node_configs()
        _t.RU_TO_DU, _t.RU_TO_CU = _t.build_ru_mapping()
        _t.P_BASE   = _t.N_NODES * _t.P_IDLE
        _t.INITIAL_PLACEMENT_DU = dict(_t.RU_TO_DU)
        _t.INITIAL_PLACEMENT_CU = dict(_t.RU_TO_CU)
        import env_ocloud as _env
        _env.N_NODES = _t.N_NODES
        _env.N_RUS   = _t.N_RUS

    DEFAULT_CFG["max_steps"] = args.max_steps
    env = make_env(DEFAULT_CFG)

    to_run = (list(HEURISTICS.keys())
              if args.heuristic == "all"
              else [args.heuristic])

    for name in to_run:
        PolicyClass, fname = HEURISTICS[name]
        policy = PolicyClass(env)
        logger = CSVLogger(os.path.join(args.results_dir, fname))
        evaluate(policy, env, args.episodes, logger, name)
        logger.close()

    env.close()
    print("\nAll heuristics evaluated.")
