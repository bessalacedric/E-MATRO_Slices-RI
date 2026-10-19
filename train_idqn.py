"""
train_idqn.py — Independent Deep-Q Networks Baseline
===========================================================
Naive multi-agent baseline: N_NODES independent DQN agents, each trying
to solve the GLOBAL objective on its own, with NO coordination mechanism:

  - No xApp guidance vector (no centralized coordination signal)
  - No shared/centralized training context
  - Each agent observes the same global state and acts independently
  - Each agent receives the SAME global reward (credit assignment is implicit/naive — every agent is blamed/rewarded for the joint outcome
    even though it only controls its own slice of the action)

This is the classic "Independent Deep-Q Networks" (IDQN) MARL baseline — agents are independent learners with no awareness of each other's
policies, which is known to suffer from:
  1. Non-stationarity: each agent's environment looks non-stationary because other agents are simultaneously changing their policy
  2. High conflict rate: agents propose SLEEP/migrations without knowing whether a partner agent will accept — no orchestration
  3. Poor scalability: conflict probability grows with N_NODES since coordination is purely accidental

Architecture (same network class as E-MATRO++ for fair comparison):
  E-MATRO MA-DQN:  DAppDQN(obs + xApp guidance) → factorized heads
  IDQN (this file):        DAppDQN(obs + zero-guidance)  → factorized heads (guidance vector always zero — no coordination)

Output: results/idqn_log.csv 

Usage:
  python3 train_idqn.py --episodes 5000 --max-steps 10
  python3 train_idqn.py --n-rus 8 --episodes 5000 --max-steps 10
"""

import os
import csv
import random
import argparse
import time
import numpy as np
import torch

from env_ocloud import OCloudEnv, N_NODES, N_RUS, N_ACTIONS, CAPACITY
from madqn.networks import MADQNAgent, ReplayBuffer

# =============================================================================
# Config
# =============================================================================

DEFAULT_CFG = {
    "ssh_user":        "ubuntu",
    "use_real":        False,
    "step_interval":   2.0,
    "max_steps":       10,

    "n_episodes":      5000,
    "batch_size":      128,
    "replay_capacity": 50_000,
    "target_sync":     200,
    "tau":             0.005,

    "eps_start":       1.0,
    "eps_end":         0.05,
    "eps_decay":       0.9995, #0.9995

    "lr":              1e-4, #1e-4
    "gamma":           0.99, #0.99
    "hidden":          128,
    "device":          "cpu",

    "log_every":       10,
    "save_every":      500,
    "results_dir":     "results",
}


# =============================================================================
# CSV Logger
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


# =============================================================================
# Training loop — Independent Q-Learning, no coordination
# =============================================================================

def train(cfg: dict):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # ── Topology override (same mechanism as train.py / train_centralized.py)
    if cfg.get("n_rus") is not None:
        import topology as _t
        _t.N_RUS      = cfg["n_rus"]
        _t.N_UES      = cfg["n_rus"]
        _t.N_DUS      = max(2, cfg["n_rus"] // 2)
        _t.N_CUS      = max(2, (cfg["n_rus"] + 3) // 4)
        _t.N_NODES    = min(_t.N_DUS + _t.N_CUS, _t.N_VMS)
        _t.RUS_PER_DU = _t.N_RUS // _t.N_DUS
        _t.NODE_CONFIGS          = _t.build_node_configs()
        _t.RU_TO_DU, _t.RU_TO_CU = _t.build_ru_mapping()
        _t.UE_NAMES   = _t.build_ue_names()
        _t.P_BASE     = _t.N_NODES * _t.P_IDLE
        _t.INITIAL_PLACEMENT_DU = dict(_t.RU_TO_DU)
        _t.INITIAL_PLACEMENT_CU = dict(_t.RU_TO_CU)
        import env_ocloud as _env
        _env.N_NODES = _t.N_NODES
        _env.N_RUS   = _t.N_RUS
        _env.N_DUS   = _t.N_DUS
        _env.N_CUS   = _t.N_CUS
        _env.NODE_CONFIGS         = _t.NODE_CONFIGS
        _env.INITIAL_PLACEMENT_DU = _t.INITIAL_PLACEMENT_DU
        _env.INITIAL_PLACEMENT_CU = _t.INITIAL_PLACEMENT_CU
        _env.RU_TO_DU = _t.RU_TO_DU
        _env.RU_TO_CU = _t.RU_TO_CU
        _env.P_BASE   = _t.P_BASE

    # ── Environment — SAME env as BRAVE-O and centralized baselines ────────
    env = OCloudEnv(
        ssh_user         = cfg["ssh_user"],
        max_steps        = cfg["max_steps"],
        use_real_traffic = cfg["use_real"],
        step_interval    = cfg["step_interval"] if cfg["use_real"] else 0.0,
    )

    obs_dim           = env.observation_space.shape[1]
    n_nodes            = N_NODES
    per_agent_act_dim  = env.per_agent_act_dim     # 1 + 2*N_RUS
    guidance_dim       = 3 * n_nodes                # kept for shape compat

    print(f"{'='*60}")
    print(f"  Independent Q-Learning (No-CTDE) Baseline")
    print(f"  Mode       : {'REAL' if cfg['use_real'] else 'SYNTHETIC'}")
    print(f"  Episodes   : {cfg['n_episodes']}")
    print(f"  Obs dim    : {obs_dim}  |  Nodes: {n_nodes}  |  RUs: {N_RUS}")
    print(f"  Per-agent act dim : {per_agent_act_dim}  |  "
          f"Flat action dim: {n_nodes*per_agent_act_dim}")
    print(f"  Coordination signal : NONE (guidance vector fixed at zero)")
    print(f"  Reward : global (same reward broadcast to all agents,")
    print(f"           but no mechanism prevents conflicting actions)")
    print(f"{'='*60}\n")

    # ── Agents — N_NODES INDEPENDENT learners, no shared coordination ──────
    # Same network class as E-MATRO++ (DAppDQN via MADQNAgent) for architectural
    # fairness — the only difference is the guidance input is always zero,
    # i.e. there is no xApp.
    agents = [
        MADQNAgent(
            agent_id = n,
            obs_dim  = obs_dim,
            n_nodes  = n_nodes,
            n_rus    = N_RUS,
            lr       = cfg["lr"],
            gamma    = cfg["gamma"],
            hidden   = cfg["hidden"],
            device   = cfg["device"],
        )
        for n in range(n_nodes)
    ]
    # Each agent gets its OWN independent replay buffer — no shared
    # centralized replay.
    replays = [ReplayBuffer(cfg["replay_capacity"]) for _ in range(n_nodes)]

    logger = CSVLogger(os.path.join(cfg["results_dir"], "idqn_training_log_8rus_6nodes_10000_episodes.csv"))

    zero_guidance = np.zeros(guidance_dim, dtype=np.float32)  # NO coordination

    epsilon     = cfg["eps_start"]
    global_step = 0
    best_reward = -np.inf

    os.makedirs(cfg["results_dir"], exist_ok=True)

    start_time = time.time()


    for ep in range(1, cfg["n_episodes"] + 1):

        obs, info = env.reset()

        ep_rewards        = []
        ep_losses          = []
        ep_energy    = []
        ep_saving           = []
        ep_nodes             = []
        ep_migrations         = 0
        ep_conflicts           = 0
        ep_failed_mig           = 0
        ep_orphan_accept          = 0
        ep_inference_ms            = []
        ep_max_inference_ms        = []
        ep_ocloud_util            = []
        ep_peak_load              = []
        ep_energy_eff             = []
        ep_capacity_viol          = []

        ep_qos        = []

        ep_proposed          = []
        ep_conflicts_step    = []
        ep_migration_success = []

        ep_delay_viol   = []
        ep_delay_sla_ok = []
        ep_worst_delay  = []


        for step in range(cfg["max_steps"]):
            train_start = time.time()

            # ── Each agent acts INDEPENDENTLY — no shared guidance ─────────
            per_agent    = []
            inference_ms = []
            for n in range(n_nodes):
                act_n, t_ms = agents[n].act(obs[n], zero_guidance, epsilon)
                per_agent.append(act_n)
                inference_ms.append(t_ms)

            flat_actions = np.concatenate(per_agent)
            ep_inference_ms.append(sum(inference_ms))

            max_inference_ms  = max(inference_ms)   # bottleneck agent

            ep_max_inference_ms.append(max_inference_ms)

            train_end = time.time()

            # ── Environment step (same env, same conflict resolution) ──────
            next_obs, rewards, done, trunc, info = env.step(flat_actions)

            # ── Each agent stores its OWN transition in its OWN buffer ──────
            for n in range(n_nodes):
                replays[n].push(
                    obs[n], per_agent[n], rewards[n], next_obs[n],
                    done or trunc
                )

            obs = next_obs
            global_step += 1

            
            node_loads   = info["node_loads"]                          # L_i × a_i already
            # nodes_active = info["nodes_active"]                          # active nodes 
            # ocloud_util  = (sum(node_loads) / nodes_active if nodes_active > 0 else 0.0)

            nodes_active = max(info["nodes_active"], 1)
            ocloud_util  = sum(info["node_loads"]) / nodes_active
            ocloud_util  = min(ocloud_util, 1.0)         # clip to valid range
            
            ep_delay_viol.append(info["delay_violation"])
            ep_delay_sla_ok.append(float(info["delay_sla_ok"]))

            # Worst-case delay across active nodes
            active_delays = [d for d in info["node_delays_ms"] if d < float('inf')]
            ep_worst_delay.append(max(active_delays) if active_delays else 0.0)
          
            

            peak_load = max(info["node_loads"])
            

            eff = info["energy_saving"] / (ocloud_util + 1e-6)

            capacity_viol = sum(
                1 for l in info["node_loads"] if l > CAPACITY
            )

            # ── Conflict and migration success ─────────────────────────────
            n_proposed = sum(
                1 for n in range(N_NODES) if any(flat_actions[n*per_agent_act_dim + 1:
                                    n*per_agent_act_dim + per_agent_act_dim] != n))   # agents proposing at least one migration
            n_executed = info.get("n_migrations", 0)
            n_failed   = info.get("n_failed_mig", 0)
            n_conflicts = info.get("n_conflicts", 0)

            ep_proposed.append(n_proposed)

            ep_conflicts_step.append(n_conflicts)
            success_rate = (n_executed / n_proposed if n_proposed > 0 else 1.0)
            ep_migration_success.append(success_rate)

            ep_delay_viol.append(info["delay_violation"])
            ep_delay_sla_ok.append(float(info["delay_sla_ok"]))

            # Worst-case delay across active nodes
            active_delays = [d for d in info["node_delays_ms"] if d < float('inf')]
            ep_worst_delay.append(max(active_delays) if active_delays else 0.0)
          


            ep_capacity_viol.append(capacity_viol)
            ep_ocloud_util.append(ocloud_util)
            ep_energy_eff.append(eff)
            ep_peak_load.append(peak_load)

            ep_rewards.append(float(np.mean(rewards)))
            ep_energy.append(info["instant_energy_W"])
            ep_saving.append(info["energy_saving"])
            ep_nodes.append(info["nodes_active"])
            ep_migrations    += info.get("n_migrations", 0)
            ep_conflicts      += info.get("n_conflicts", 0)
            ep_failed_mig      += info.get("n_failed_mig", 0)
            ep_orphan_accept     += info.get("n_orphan_accept", 0)
            ep_qos.append(info["qos_violation"])


            # ── Independent learning — each agent updates from ITS OWN
            #    buffer only, with zero guidance (no coordination signal)
            if len(replays[0]) >= cfg["batch_size"]:
                step_loss = 0.0
                for n in range(n_nodes):
                    batch = replays[n].sample(cfg["batch_size"])
                    g_batch = torch.zeros(
                        cfg["batch_size"], guidance_dim, dtype=torch.float32
                    )
                    loss = agents[n].update(batch, g_batch)
                    step_loss += loss
                ep_losses.append(step_loss / n_nodes)

                if global_step % cfg["target_sync"] == 0:
                    for n in range(n_nodes):
                        agents[n].sync_target(tau=cfg["tau"])

            if done or trunc:
                break

        epsilon = max(cfg["eps_end"], epsilon * cfg["eps_decay"])

        end_time = time.time()
        training_time = train_end - train_start

        avg_reward = float(np.mean(ep_rewards))
        avg_energy = float(np.mean(ep_energy))
        avg_loss   = float(np.mean(ep_losses)) if ep_losses else 0.0
        avg_saving = float(np.mean(ep_saving))
        avg_nodes  = float(np.mean(ep_nodes))
        avg_qos    = float(np.mean(ep_qos))

        logger.log({
            "episode":           ep,
            "avg_reward":        round(avg_reward, 6),
            "avg_energy":       round(avg_energy, 6),
            "avg_loss":          round(avg_loss, 6),
            "avg_qos":         round(avg_qos, 6),
            "energy_saving_%":   round(avg_saving * 100, 2),
            "avg_nodes_active":  round(avg_nodes, 2),
            "total_migrations":  ep_migrations,
            "ocloud_utilization":  round(float(np.mean(ep_ocloud_util)), 4),
            "peak_load":          round(float(np.mean(ep_peak_load)), 4),
            "avg_energy_eff":    round(float(np.mean(ep_energy_eff)), 4),
            "capacity_violations": int(sum(ep_capacity_viol)),

            "delay_violation_rate": round(float(np.mean(ep_delay_sla_ok) < 1.0), 4),
            "avg_delay_violation":  round(float(np.mean(ep_delay_viol)),  4),
            "worst_delay_ms":       round(float(np.max(ep_worst_delay)),  4),

            "failed_migrations": ep_failed_mig,
            "orphan_accepts":    ep_orphan_accept,
            "total_conflicts":   ep_conflicts,
            "avg_inference_ms":  round(float(np.mean(ep_inference_ms)) if ep_inference_ms else 0, 4),
            "max_inference_ms":     round(float(np.max(ep_max_inference_ms)) if ep_max_inference_ms else 0, 4),
            "total_inference_ms":   round(float(np.sum(ep_inference_ms)) if ep_inference_ms else 0, 4),
            "training_time":  round(training_time, 4),
            "epsilon":           round(epsilon, 4),
            "global_step":       global_step,

            "conflict_rate":        round(float(np.mean(ep_conflicts_step)) / max(float(np.mean(ep_proposed)), 1e-6), 4),
            "migration_success_rate": round(float(np.mean(ep_migration_success)), 4),



        })

        if ep % cfg["log_every"] == 0:
            print(
                f"\nEpisode: {ep}/{cfg['n_episodes']} | "
                f"Reward: {avg_reward:.4f} | "
                f"Saving: {avg_saving*100:.1f}% | "
                f"Avg Nodes ON: {avg_nodes:.1f}/{n_nodes} | "
                f"Loss: {avg_loss:.4f} | "
                f"Epsilon: {epsilon:.3f} | "
                f"OCloud util: {np.mean(ep_ocloud_util)*100:.1f}% |"
                f"Peak load: {np.mean(ep_peak_load):.1f} | "
                f"Avg energy eff: {np.mean(ep_energy_eff):.1f}"
                f" | Capacity violations: {sum(ep_capacity_viol)}"
            )
            print(
                f"  Migrations: {ep_migrations} | "
                f"Failed: {ep_failed_mig} | "
                f"Conflicts: {ep_conflicts} | "
                f"(NO coordination — independent agents)"
            )

        if ep % cfg["save_every"] == 0:
            for n in range(n_nodes):
                agents[n].save(
                    os.path.join(cfg["results_dir"], f"IDQN/idqn_agent_{n}_ep{ep}.pt"))
            print(f"  [checkpoint @ ep {ep}]")

        if avg_reward > best_reward:
            best_reward = avg_reward
            for n in range(n_nodes):
                agents[n].save(
                    os.path.join(cfg["results_dir"], f"IDQN/idqn_agent_{n}_best.pt"))


        if ep == cfg["n_episodes"]:
            print(f"\n  Training complete. Best avg reward: {best_reward:.4f}")
            print(f"\n Energy saving: {avg_saving*100:.2f}% | Avg nodes active: {avg_nodes:.2f} | Avg energy: {avg_energy:.1f}Wh")
            print(f"  Avg QoS violation: {avg_qos:.4f} | Avg OCloud util: {np.mean(ep_ocloud_util)*100:.1f}% | Peak load: {np.mean(ep_peak_load):.1f} | Avg energy eff: {np.mean(ep_energy_eff):.1f}")
            print(f"  Capacity violations: {sum(ep_capacity_viol)} | Total migrations: {ep_migrations} | Failed migrations: {ep_failed_mig} | Orphan accepts: {ep_orphan_accept} | Total conflicts: {ep_conflicts}")
      

    logger.close()
    env.close()

    print(f"\n{'='*60}")
    print(f"  IDQN baseline complete.")
    print(f"  Best avg reward : {best_reward:.4f}")
    print(f"  Results: {cfg['results_dir']}/idqn_training_log.csv")
    print(f"{'='*60}")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Independent Deep-Q Networks  baseline for E-MATRO++ comparison"
    )
    parser.add_argument("--n-rus",         type=int, default=None,  help="Override N_RUS (e.g. 4, 8, 12)")
    parser.add_argument("--real",          action="store_true")
    parser.add_argument("--ssh-user",      default="ubuntu")
    parser.add_argument("--step-interval", type=float, default=2.0)
    parser.add_argument("--max-steps",     type=int,   default=10)
    parser.add_argument("--episodes",      type=int,   default=5000)
    parser.add_argument("--batch-size",    type=int,   default=128)
    parser.add_argument("--lr",            type=float, default=1e-3) #1e-3
    parser.add_argument("--gamma",         type=float, default=0.99) #0.99
    parser.add_argument("--hidden",        type=int,   default=128)
    parser.add_argument("--eps-start",     type=float, default=1.0)
    parser.add_argument("--eps-end",       type=float, default=0.05)
    parser.add_argument("--eps-decay",     type=float, default=0.9999) #0.995
    parser.add_argument("--target-sync",   type=int,   default=200)
    parser.add_argument("--tau",           type=float, default=0.005)
    parser.add_argument("--log-every",     type=int,   default=10)
    parser.add_argument("--save-every",    type=int,   default=500)
    parser.add_argument("--results-dir",   default="results")
    parser.add_argument("--device",        default="cpu")
    args = parser.parse_args()

    cfg = {**DEFAULT_CFG}
    cfg.update({
        "n_rus":           args.n_rus,
        "ssh_user":        args.ssh_user,
        "use_real":        args.real,
        "step_interval":   args.step_interval,
        "max_steps":       args.max_steps,
        "n_episodes":      args.episodes,
        "batch_size":      args.batch_size,
        "lr":              args.lr,
        "gamma":           args.gamma,
        "hidden":          args.hidden,
        "eps_start":       args.eps_start,
        "eps_end":         args.eps_end,
        "eps_decay":       args.eps_decay,
        "target_sync":     args.target_sync,
        "tau":             args.tau,
        "log_every":       args.log_every,
        "save_every":      args.save_every,
        "results_dir":     args.results_dir,
        "device":          args.device,
    })

    train(cfg)
