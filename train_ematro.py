"""
train.py — Training Loop for E-MATRO++
============================================
Matches new ocloud_env.py:
  - Actions: IDLE / SLEEP / ACCEPT / WAKE per node
  - Obs: (N_NODES=4, obs_dim=22)
  - Real traffic via SSH to SLICES-RI (--real flag)
  - Synthetic mode for offline training (default)

Usage:
  # Synthetic (offline)
 
  python3 train_ematro.py --episodes 5000 --max-steps 50
  python3 train_ematro.py --episodes 5000 --max-steps 10 --eps-start 1.0 --eps-end 0.05 --eps-decay 0.999

  # 8 RUs  — 6 nodes: python3 train.py --n-rus 8 --episodes 5000 --max-steps 10

  # Real traffic from testbed
  python3 train_ematro.py --real --ssh-user ubuntu --episodes 100 --step-interval 2.0 
  python3 train_ematro.py --real --ssh-user ubuntu \
    --results-dir ~/E-matro-evaluation/results \
    --metrics-file ~/E-matro-evaluation/results/e-matro_ran_metrics.csv



python3 train_ematro.py --episodes 3000 --max-steps 10 --results-dir results/madqn_3000

python3 train_dqn.py --episodes 3000 --max-steps 10 --results-dir results/dqn_3000

"""


import os
import csv
import random
import argparse
import numpy as np
import torch
import time


import sys as _sys
_nrus_override = None
for _i, _a in enumerate(_sys.argv):
    if _a == '--n-rus' and _i + 1 < len(_sys.argv):
        _nrus_override = int(_sys.argv[_i + 1])
        break

if _nrus_override is not None:
    import topology as _topo_pre
    _topo_pre.N_RUS      = _nrus_override
    _topo_pre.N_UES      = _nrus_override
    _topo_pre.N_DUS      = max(2, _nrus_override // 2)
    _topo_pre.N_CUS      = max(2, (_nrus_override + 3) // 4)
    _topo_pre.N_NODES    = min(_topo_pre.N_DUS + _topo_pre.N_CUS, _topo_pre.N_VMS)
    _topo_pre.RUS_PER_DU = _topo_pre.N_RUS // _topo_pre.N_DUS
    _topo_pre.NODE_CONFIGS           = _topo_pre.build_node_configs()
    _topo_pre.RU_TO_DU, _topo_pre.RU_TO_CU = _topo_pre.build_ru_mapping()
    _topo_pre.UE_NAMES               = _topo_pre.build_ue_names()
    _topo_pre.P_BASE                 = _topo_pre.N_NODES * _topo_pre.P_IDLE
    _topo_pre.INITIAL_PLACEMENT_DU   = dict(_topo_pre.RU_TO_DU)
    _topo_pre.INITIAL_PLACEMENT_CU   = dict(_topo_pre.RU_TO_CU)

from env_ocloud import OCloudEnv, N_NODES, N_RUS, N_ACTIONS, CAPACITY
from madqn.networks import MADQNAgent, XAppActor, ReplayBuffer
from ran_metrics import RANMetricsCollector

# =============================================================================
# Hyperparameters
# =============================================================================

DEFAULT_CFG = {
    # Environment
    "ssh_user":        "ubuntu",
    "use_real":        False,
    "step_interval":   2.0,
    "max_steps":       5,       # steps per episode (each = 1 env step)

    # Training
    "n_episodes":      500,
    "batch_size":      128,       # larger batch → stable gradients
    "replay_capacity": 50_000,    # larger buffer → diverse samples
    "target_sync":     200,       # steps between target sync
    "tau":             0.005,     # Polyak averaging (soft update)
    "use_soft_update": True,      # True=Polyak, False=hard copy

    # Exploration
    "eps_start":       1.0,
    "eps_end":         0.01,
    "eps_decay":       0.9995, #0.9995   # slower decay → more exploration

    # Guided exploration (warmup)
    "warmup_steps":    1000,     # global steps to apply guided injection
    "guided_prob":     0.3,      # probability of injecting SLEEP+ACCEPT per step

    # Network
    "lr":              1e-4,  #1e-4    # lower LR — key fix for divergence
    "gamma":           0.99,
    "hidden":          128,
    "device":          "cpu",

    # Logging
    "log_every":       10,
    "save_every":      100,
    "results_dir":     "results",

    # RAN metrics collection (real traffic only)
    "collect_ran":     True,      # collect RAN metrics during training
    "metrics_every":   5,         # collect every N steps (SSH overhead)
    
    "metrics_file":    os.path.join(os.path.dirname(os.path.abspath(__file__)), 
    "results", "e-matro_ran_metrics.csv"),

}

ACTION_NAMES  = ["IDLE", "SLEEP", "ACCEPT", "WAKE"]
ACTION_IDLE   = 0
ACTION_SLEEP  = 1
ACTION_ACCEPT = 2
ACTION_WAKE   = 3


# =============================================================================
# Guided exploration — forces coordinated SLEEP+ACCEPT during warmup
# =============================================================================

def guided_action(last_info: dict, global_step: int, n_nodes: int,
                  warmup_steps: int = 1000,
                  guided_prob: float = 0.3) -> list | None:
    """
    During warmup (global_step < warmup_steps): occasionally inject a
    coordinated SLEEP+ACCEPT pair so the agent sees successful migrations
    early in training, before epsilon decays too low.

    Returns a list of actions if injection fires, else None
    (caller falls back to normal epsilon-greedy).

    Args:
        last_info    : info dict from previous env.step() / reset()
        global_step  : total steps taken so far
        n_nodes      : number of agents
        warmup_steps : how long to apply guided exploration
        guided_prob  : probability of injecting a guided pair per step
    """
    if global_step >= warmup_steps:
        return None
    if random.random() >= guided_prob:
        return None

    actions = [ACTION_IDLE] * n_nodes

    # Pick source node: prefer nodes that currently host VNFs
    active_with_vnfs = list(range(n_nodes))
    if last_info and "placement" in last_info:
        hosting = set(last_info["placement"].values())
        if hosting:
            active_with_vnfs = list(hosting)

    if len(active_with_vnfs) < 1:
        return None

    src = random.choice(active_with_vnfs)
    candidates = [n for n in range(n_nodes) if n != src]
    if not candidates:
        return None
    dst = random.choice(candidates)

    actions[src] = ACTION_SLEEP
    actions[dst] = ACTION_ACCEPT
    return actions


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
# Episode printer
# =============================================================================

def print_episode(ep: int, cfg: dict, ep_info: dict):
    """Print one-line summary plus detail block every log_every episodes."""
    avg_reward  = ep_info["avg_reward"]
    avg_energy  = ep_info["avg_energy"]
    avg_loss    = ep_info["avg_loss"]
    saving      = ep_info["avg_saving"] * 100
    nodes_on    = ep_info["avg_nodes_active"]
    total_mig   = ep_info["total_migrations"]
    
    n_conflicts  = ep_info["total_conflicts"]
    failed_mig   = ep_info["failed_mig"]
    orphan_acc   = ep_info["orphan_accept"]
    eps          = ep_info["epsilon"]
    power        = ep_info["avg_power_W"]
   


    print(
        f"\nEpisode: {ep}/{cfg['n_episodes']} | "
        f"Reward: {avg_reward:.4f} | "
        f"Instant Energy: {avg_energy:.1f}W | "
        f"Saving: {saving:.1f}% | "
        f"Nodes ON: {nodes_on:.1f}/{N_NODES} | "
        f"Migrations: {total_mig:.0f} | "
        f"Conflicts: {n_conflicts:.0f} | "
        f"Loss: {avg_loss:.4f} | "
        f"Epsilon: {eps:.3f}"
        # f"OCloud util: {np.mean(ep_ocloud_util)*100:.1f}% |"
        # f"Peak load: {np.mean(ep_peak_load):.1f} | "
        # f"Avg energy eff: {np.mean(ep_energy_eff):.1f}"
        # f" | Capacity violations: {sum(ep_capacity_viol)}"
    )
    print(
        f"  Migrations: {total_mig} (executed, avg {total_mig/cfg['max_steps']:.1f}/step) | "
        f"Failed attempts: {failed_mig} | "
        f"Orphan accepts: {orphan_acc} | "
        f"Total conflicts: {n_conflicts}"
    )

    print(
        f"  Power: {power:.0f}W | "
        f"QoS viol: {ep_info['avg_qos']:.4f} | "
        f"Placement: {ep_info['placement_str']}"
    )


# =============================================================================
# Training loop
# =============================================================================

def train(cfg: dict):
    # ── Topology override ────────────────────────────────────────────────────
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
        _t.UE_NAMES  = _t.build_ue_names()
        _t.P_BASE    = _t.N_NODES * _t.P_IDLE
        _t.INITIAL_PLACEMENT_DU = dict(_t.RU_TO_DU)
        _t.INITIAL_PLACEMENT_CU = dict(_t.RU_TO_CU)
        # Reload env module constants from updated topology
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

    # ── Reproducibility ──────────────────────────────────────────────────────
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # ── Environment ───────────────────────────────────────────────────────────
    env = OCloudEnv(
        ssh_user         = cfg["ssh_user"],
        max_steps        = cfg["max_steps"],
        use_real_traffic = cfg["use_real"],
        step_interval    = cfg["step_interval"] if cfg["use_real"] else 0.0,
    )

    obs_dim = env.observation_space.shape[1]   
    n_nodes = N_NODES                           

    print(f"{'='*60}")
    print(f"  E-MATRO++ Training")
    print(f"  Mode    : {'REAL TRAFFIC' if cfg['use_real'] else 'SYNTHETIC'}")
    print(f"  Episodes: {cfg['n_episodes']}")
    print(f"  Steps/ep: {cfg['max_steps']}")
    print(f"  Obs dim : {obs_dim}  |  Nodes: {n_nodes}  |  RUs: {N_RUS}")
    print(f"  Device  : {cfg['device']}")
    print(f"{'='*60}\n")



    # ── RAN Metrics Collector ─────────────────────────────────────────────
    ran_collector = None
    if cfg["use_real"] and cfg.get("collect_ran", True):
        ran_collector = RANMetricsCollector(
            ssh_user    = cfg["ssh_user"],
            output_path = cfg["metrics_file"],
        )
        print(f"  RAN metrics: {cfg['metrics_file']}")


    # ── Networks ──────────────────────────────────────────────────────────────
    xapp = XAppActor(obs_dim, n_nodes, hidden=cfg["hidden"])

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

    per_agent_act_dim = 1 + 2 * N_RUS   # matches env.per_agent_act_dim

    replay  = ReplayBuffer(cfg["replay_capacity"])
    logger  = CSVLogger(os.path.join(cfg["results_dir"], "e-matro_training_log_8rus_6nodes_10000_episodes.csv"))


    epsilon      = cfg["eps_start"]
    global_step  = 0
    best_reward  = -np.inf

    os.makedirs(cfg["results_dir"], exist_ok=True)

    start_time = time.time()

    # ── Training ──────────────────────────────────────────────────────────────
    for ep in range(1, cfg["n_episodes"] + 1):

        obs, info = env.reset()
        # obs shape: (N_NODES, obs_dim) — same global obs for all agents
        # Use node 0's obs as global state for xApp (they're identical)
        guidance = xapp.act(obs[0])

        ep_rewards    = []
        ep_instant_energy = []
        ep_losses     = []
        ep_saving     = []
        ep_nodes      = []
        ep_qos        = []
        ep_power      = []
        ep_migrations = 0
        ep_conflicts  = 0

        ep_failed_mig    = 0
        ep_orphan_accept = 0

        ep_inference_ms     = []
        ep_max_inference_ms = []
        ep_ocloud_util = []
        ep_peak_load   = []
        ep_energy_eff  = []
        ep_capacity_viol = []


        ep_proposed          = []
        ep_conflicts_step    = []
        ep_migration_success = []

        ep_delay_viol   = []
        ep_delay_sla_ok = []
        ep_worst_delay  = []

        last_info = info   # carry info across steps for guided_action

        for step in range(cfg["max_steps"]):

            
            train_start = time.time()
            per_agent      = []
            inference_ms = []
            for n in range(n_nodes):
                act_n, t_ms = agents[n].act(obs[n], guidance, epsilon)
                per_agent.append(act_n)
                inference_ms.append(t_ms)

            flat_actions = np.concatenate(per_agent)  # (N_NODES*per_agent_act_dim,)

            step_inference_ms = sum(inference_ms)   # total for all agents this step
            max_inference_ms  = max(inference_ms)   # bottleneck agent

            ep_inference_ms.append(step_inference_ms)
            ep_max_inference_ms.append(max_inference_ms)

            train_end = time.time()

            # During warmup: override activation decisions only (a_i component)
            # guided_action returns [a_i per node] — inject into flat_actions
            guided = guided_action(
                last_info    = last_info,
                global_step  = global_step,
                n_nodes      = n_nodes,
                warmup_steps = cfg.get("warmup_steps", 1000),
                guided_prob  = cfg.get("guided_prob", 0.3),
            )
            if guided is not None:
                # guided: list of N_NODES a_i values ∈ {0=OFF, 1=ON}
                # clamp to valid range before injecting
                for n in range(n_nodes):
                    flat_actions[n * per_agent_act_dim] = min(guided[n], 1)


               
            # ── Environment step ──────────────────────────────────────────
            next_obs, rewards, done, trunc, info = env.step(flat_actions)

            #ocloud_util = sum(info["node_loads"]) / N_NODES

            node_loads   = info["node_loads"]                          # L_i × a_i already
            #nodes_active = info["nodes_active"]                          # active nodes 
            #ocloud_util  = (sum(node_loads) / nodes_active if nodes_active > 0 else 0.0)
            
            nodes_active = max(info["nodes_active"], 1)
            ocloud_util  = sum(info["node_loads"]) / nodes_active
            ocloud_util  = min(ocloud_util, 1.0)         # clip to valid range
            
                        

            ocloud_util  = sum(info["node_loads"]) / nodes_active

            peak_load = max(info["node_loads"])

            eff = info["energy_saving"] / (ocloud_util + 1e-6)

            capacity_viol = sum(
                1 for l in info["node_loads"] if l > CAPACITY
            )
            ep_capacity_viol.append(capacity_viol)


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
          

            # ── Update xApp guidance ──────────────────────────────────────
            next_guidance = xapp.act(next_obs[0])

            # ── Store in replay buffer ────────────────────────────────────
            
            # Store per-agent action slices in replay buffer
            # acts_per_agent: (N_NODES, per_agent_act_dim)
            acts_per_agent = flat_actions.reshape(n_nodes, per_agent_act_dim)
            replay.push(obs, acts_per_agent, rewards, next_obs, done or trunc)


            obs       = next_obs
            guidance  = next_guidance
            last_info = info   # update for next step's guided_action

            # ── Accumulate episode stats ──────────────────────────────────
            ep_rewards.append(float(np.mean(rewards)))
            ep_instant_energy.append(info["instant_energy_W"])
            ep_saving.append(info["energy_saving"])
            ep_ocloud_util.append(ocloud_util)
            ep_peak_load.append(peak_load)
            ep_energy_eff.append(eff)
            ep_nodes.append(info["nodes_active"])
            ep_qos.append(info["qos_violation"])
            ep_power.append(info["power_W"])
            ep_migrations += len(info["migrations"])
            ep_conflicts  += info["n_conflicts"]
            ep_failed_mig     += info["n_failed_mig"]
            ep_orphan_accept  += info["n_orphan_accept"]
            global_step   += 1

            # ── Learn ─────────────────────────────────────────────────────
            if len(replay) >= cfg["batch_size"]:
                obs_b, acts_b, rews_b, nobs_b, dones_b = \
                    replay.sample(cfg["batch_size"])
                # guidance batch: compute from sampled next_obs
                # Use first agent obs (global) to get guidance
                g_batch = torch.FloatTensor(
                    np.stack([
                        xapp.act(nobs_b[i, 0].numpy())
                        for i in range(cfg["batch_size"])
                    ])
                )

                step_loss = 0.0
                for n, agent in enumerate(agents):
                    agent_batch = (
                        obs_b[:, n, :],    # (B, obs_dim)
                        acts_b[:, n, :],   # (B, per_agent_act_dim)
                        rews_b[:, n],      # (B,)
                        nobs_b[:, n, :],   # (B, obs_dim)
                        dones_b,           # (B,)
                    )
                    loss = agent.update(agent_batch, g_batch)
                    step_loss += loss

                ep_losses.append(step_loss / n_nodes)

                # Sync target networks
                if global_step % cfg["target_sync"] == 0:
                    tau = cfg["tau"] if cfg["use_soft_update"] else 1.0
                    for agent in agents:
                        agent.sync_target(tau=tau)

            if done or trunc:
                break

        # ── Epsilon decay ─────────────────────────────────────────────────
        epsilon = max(cfg["eps_end"], epsilon * cfg["eps_decay"])

        end_time = time.time()
        training_time = train_end - train_start

        execution_time = end_time - start_time

        # ── Episode summary ───────────────────────────────────────────────
        avg_reward = float(np.mean(ep_rewards))
        avg_energy = float(np.mean(ep_instant_energy))
        avg_loss   = float(np.mean(ep_losses)) if ep_losses else 0.0
        avg_saving = float(np.mean(ep_saving))
        avg_nodes  = float(np.mean(ep_nodes))
        avg_qos    = float(np.mean(ep_qos))
        avg_power  = float(np.mean(ep_power))

        ep_info = {
            "avg_reward":      avg_reward,
            "avg_energy": avg_energy,
            "avg_loss":        avg_loss,
            "avg_saving":      avg_saving,
            "avg_nodes_active":avg_nodes,
            "avg_qos":         avg_qos,
            "avg_power_W":     avg_power,
            "total_migrations":ep_migrations,
            "total_conflicts": ep_conflicts,
            "failed_mig":       ep_failed_mig,
            "orphan_accept":    ep_orphan_accept,
            "epsilon":         epsilon,
            "placement_str":   info.get("placement_str", ""),
            "ocloud_utilization":  round(float(np.mean(ep_ocloud_util)), 4),
            "peak_load":          round(float(np.mean(ep_peak_load)), 4),
            "avg_energy_eff":    round(float(np.mean(ep_energy_eff)), 4),
            "capacity_violations": int(sum(ep_capacity_viol)),
        }

        # ── CSV log ───────────────────────────────────────────────────────
        logger.log({
            "episode":          ep,
            "avg_reward":       round(avg_reward, 6),
            "avg_energy":       round(avg_energy, 6),
            "avg_loss":         round(avg_loss, 6),
            "energy_saving_%":  round(avg_saving * 100, 2),
            "avg_nodes_active": round(avg_nodes, 2),
            "avg_power_W":      round(avg_power, 1),
            "ocloud_utilization":  round(float(np.mean(ep_ocloud_util)), 4),
            "peak_load":          round(float(np.mean(ep_peak_load)), 4),
            "avg_energy_eff":    round(float(np.mean(ep_energy_eff)), 4),
            "capacity_violations": int(sum(ep_capacity_viol)),
            "avg_qos_violation":round(avg_qos, 6),

            "delay_violation_rate": round(float(np.mean(ep_delay_sla_ok) < 1.0), 4),
            "avg_delay_violation":  round(float(np.mean(ep_delay_viol)),  4),
            "worst_delay_ms":       round(float(np.max(ep_worst_delay)),  4),

            "total_migrations": ep_migrations,
            "failed_migrations":ep_failed_mig,
            "orphan_accepts":   ep_orphan_accept,
            "total_conflicts":  ep_conflicts,
            "epsilon":          round(epsilon, 4),
            "global_step":      global_step,
            "avg_inference_ms":     round(float(np.mean(ep_inference_ms)) if ep_inference_ms else 0, 4),
            "max_inference_ms":     round(float(np.max(ep_max_inference_ms)) if ep_max_inference_ms else 0, 4),
            "total_inference_ms":   round(float(np.sum(ep_inference_ms)) if ep_inference_ms else 0, 4),

            "execution_time": round(execution_time, 4),
            "training_time":  round(training_time, 4),

            "conflict_rate":        round(float(np.mean(ep_conflicts_step)) / max(float(np.mean(ep_proposed)), 1e-6), 4),
            "migration_success_rate": round(float(np.mean(ep_migration_success)), 4),

        })

        # ── Console log ───────────────────────────────────────────────────
        if ep % cfg["log_every"] == 0:
            print_episode(ep, cfg, ep_info)

        # ── Checkpoints ───────────────────────────────────────────────────
        if ep % cfg["save_every"] == 0:
            for n, agent in enumerate(agents):
                agent.save(os.path.join(cfg["results_dir"], f"E-MATRO/agent_{n}_ep{ep}.pt"))
            print(f"  [checkpoint saved @ ep {ep}]")

        if avg_reward > best_reward:
            best_reward = avg_reward
            for n, agent in enumerate(agents):
                agent.save(os.path.join(cfg["results_dir"], f"E-MATRO/agent_{n}_best.pt"))

        if ep == cfg["n_episodes"]:
            print(f"\n  Training complete. Best avg reward: {best_reward:.4f}")
            print(f"\n Energy saving: {avg_saving*100:.2f}% | Avg nodes active: {avg_nodes:.2f} | Avg power: {avg_power:.1f}W | Avg energy: {avg_energy:.1f}Wh")
            print(f"  Avg QoS violation: {avg_qos:.4f} | Avg OCloud util: {np.mean(ep_ocloud_util)*100:.1f}% | Peak load: {np.mean(ep_peak_load):.1f} | Avg energy eff: {np.mean(ep_energy_eff):.1f}")
            print(f"  Capacity violations: {sum(ep_capacity_viol)} | Total migrations: {ep_migrations} | Failed migrations: {ep_failed_mig} | Orphan accepts: {ep_orphan_accept} | Total conflicts: {ep_conflicts}")
            #print(f"\n Total migrations: {np.mean}")

            

    # ── Done ─────────────────────────────────────────────────────────────────
    logger.close()
    env.close()

    if ran_collector is not None:
        ran_collector.close()
        print(f"  RAN metrics saved to: {cfg['metrics_file']}")


    print(f"\n{'='*60}")
    print(f"  Training complete.")
    print(f"  Best avg reward : {best_reward:.4f}")
    print(f"  Results saved to: {cfg['results_dir']}/")
    print(f"{'='*60}")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E-MATRO++ Training")

    parser.add_argument("--n-rus",         type=int, default=None, help="Override N_RUS (e.g. 4, 8, 12)")
    parser.add_argument("--real",          action="store_true", help="Use real traffic from SLICES-RI testbed")
    parser.add_argument("--ssh-user",      default="ubuntu", help="SSH user for vm-servers (default: ubuntu)")
    parser.add_argument("--step-interval", type=float, default=2.0, help="Seconds between load samples (real mode)")
    parser.add_argument("--max-steps",     type=int, default=2, help="Max steps per episode")
    parser.add_argument("--episodes",      type=int, default=500)
    parser.add_argument("--batch-size",    type=int, default=64)
    parser.add_argument("--lr",            type=float, default=1e-3) #1e-3
    parser.add_argument("--gamma",         type=float, default=0.99) #0.99
    parser.add_argument("--hidden",        type=int, default=128)
    parser.add_argument("--eps-start",     type=float, default=1.0)
    parser.add_argument("--eps-end",       type=float, default=0.01)
    parser.add_argument("--eps-decay",     type=float, default=0.9999) #0.995
    parser.add_argument("--target-sync",   type=int, default=200)
    parser.add_argument("--tau",           type=float, default=0.005, help="Polyak averaging coefficient for soft update")
    parser.add_argument("--hard-update",   action="store_true", help="Use hard target copy instead of soft update")
    parser.add_argument("--log-every",     type=int, default=10)
    parser.add_argument("--save-every",    type=int, default=100)
    parser.add_argument("--results-dir",   default="results")
    parser.add_argument("--device",        default="cpu")

    args = parser.parse_args()

    cfg = {**DEFAULT_CFG}
    cfg.update({
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
        "n_rus":           args.n_rus,
        "n_rus":           args.n_rus,
        "target_sync":     args.target_sync,
        "tau":             args.tau,
        "use_soft_update": not args.hard_update,
        "log_every":       args.log_every,
        "save_every":      args.save_every,
        "results_dir":     args.results_dir,
        "device":          args.device,
    })

    train(cfg)
