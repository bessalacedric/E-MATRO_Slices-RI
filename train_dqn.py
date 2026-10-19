"""
train_dqn.py — Centralized Single-Agent DQN Baseline
=============================================================
Implements a single xApp DQN agent deployed in the Near-RT RIC
that takes ALL node decisions simultaneously (joint action space).

This is the comparison baseline for E-MATRO's MA-DQN CTDE approach.
Expected result: converges but slower and to a worse policy due to:
  1. Exponential joint action space: 4^N_NODES vs N_NODES × 4
  2. No credit assignment — single reward for all nodes
  3. No inter-agent coordination signal (no XApp guidance vector)
  4. Scalability bottleneck: adding a node multiplies action space by 4

Architecture difference:
  E-MATRO MA-DQN:   N_NODES agents × (obs_dim → 4 Q-values)
  Centralized DQN:  1 agent × (obs_dim → 4^N_NODES Q-values)

For N_NODES=4: joint action space = 4^4 = 256 actions
For N_NODES=5: joint action space = 4^5 = 1024 actions  ← curse of dimensionality

Output: results/centralized_log.csv (same format as training_log.csv for
        direct comparison plotting)

Usage:
  python3 train_dqn.py --episodes 5000 --max-steps 50

  python3 train_dqn.py --n-rus 8 --episodes 5000 --max-steps 10
"""

import os
import csv
import random
import argparse
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import time

from env_ocloud import OCloudEnv, N_NODES, N_RUS, N_ACTIONS, CAPACITY

# =============================================================================
# Joint action space
# =============================================================================

def build_joint_action_space(n_nodes: int, n_actions: int) -> list:
    """
    Enumerate all possible joint actions.
    Returns list of length n_actions^n_nodes, each entry is a tuple
    of n_nodes individual actions.

    For N_NODES=4, N_ACTIONS=4: 256 joint actions
    """
    return list(itertools.product(range(n_actions), repeat=n_nodes))


# =============================================================================
# Centralized DQN — single network, joint action output
# =============================================================================

class CentralizedDQN(nn.Module):
    """
    Single DQN that selects a JOINT action for all N_NODES nodes.
    Input:  global observation (obs_dim,)
    Output: Q-value for each of 4^N_NODES joint actions

    Deployed as a single xApp in the Near-RT RIC — sees the full
    global state and issues all control decisions simultaneously.
    """

    def __init__(self, obs_dim: int, flat_dim: int,
                 hidden: int = 256):
        super().__init__()
        n_joint = flat_dim   # flat joint action vector length

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, n_joint),
        )

        # Orthogonal init for stability
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (B, obs_dim) → Q: (B, n_joint)"""
        return self.net(obs)


class CentralizedAgent:
    """
    Single centralized DQN agent (xApp).
    Wraps online + target networks with Double DQN update.
    """

    def __init__(self, obs_dim: int, n_nodes: int, per_agent_dim: int, hidden: int = 256, lr: float = 1e-4, gamma: float = 0.99, device: str = "cpu"):
        
        self.n_nodes       = n_nodes
        self.per_agent_dim = per_agent_dim
        self.flat_dim      = n_nodes * per_agent_dim
        self.n_joint       = self.flat_dim
        self.gamma         = gamma
        self.device        = torch.device(device)

        self.online = CentralizedDQN(obs_dim, self.flat_dim, hidden).to(self.device)
        self.target = CentralizedDQN(obs_dim, self.flat_dim, hidden).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)




    def act(self, obs: np.ndarray, epsilon: float, per_agent_nvec: np.ndarray) -> tuple:
        """
        Epsilon-greedy action selection over flat joint action vector.
        Returns (flat_actions, inference_ms):
        flat_actions : np.ndarray of shape (N_NODES * per_agent_dim,)
        inference_ms : float — forward pass time in ms (0.0 if random)
        """
        if random.random() < epsilon:
            # Random exploration — no network inference
            flat = np.array([ np.random.randint(0, per_agent_nvec[k % len(per_agent_nvec)]) for k in range(self.flat_dim)], dtype=np.int64)
            return flat, 0.0    # 0ms — no forward pass

        # Greedy — measure inference time
        t0 = time.perf_counter()
        # with torch.no_grad():
        #     x    = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        #     q    = self.online(x)           # (1, flat_dim)
        #     q_np = q.squeeze(0).cpu().numpy()
        #     flat = np.array([
        #         int(q_np[k] % per_agent_nvec[k % len(per_agent_nvec)])
        #         for k in range(self.flat_dim)
        #     ], dtype=np.int64)
        with torch.no_grad():
            x = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            q = self.online(x).squeeze(0).cpu().numpy()

            flat = np.zeros(self.flat_dim, dtype=np.int64)
            for agent in range(self.n_nodes):
                for k in range(self.per_agent_dim):
                    idx     = agent * self.per_agent_dim + k
                    max_val = 2 if k == 0 else self.n_nodes
                    # Clip Q-value-derived index into valid range (not modulo)
                    flat[idx] = int(np.clip(round(q[idx]), 0, max_val - 1))
        t1 = time.perf_counter()
        inference_ms = (t1 - t0) * 1000

        return flat, inference_ms




    def update(self, batch: tuple) -> float:
        """
            Double DQN update — per-component loss over flat action vector.
            Each output neuron k predicts Q for component k of the flat action.
        """

        obs_b, acts_b, rew_b, nobs_b, done_b = batch
        obs_b  = obs_b.to(self.device)
        acts_b = acts_b.to(self.device)
        rew_b  = rew_b.to(self.device)
        nobs_b = nobs_b.to(self.device)
        done_b = done_b.to(self.device)

        q_online      = self.online(obs_b)
        q_target_net  = self.target(nobs_b)
        with torch.no_grad():
            q_next_online = self.online(nobs_b)

        total_loss = torch.tensor(0.0, device=self.device)
        for agent in range(self.n_nodes):
            for k in range(self.per_agent_dim):
                idx = agent * self.per_agent_dim + k
                # Correct range per component: index 0 → binary, rest → N_NODES
                max_val = 2 if k == 0 else self.n_nodes
                act_k   = acts_b[:, idx].clamp(0, max_val - 1)

                q_k     = q_online[:, idx]
                q_nk_on = q_next_online[:, idx]
                q_nk_tg = q_target_net[:, idx]

                q_tgt   = rew_b + self.gamma * q_nk_tg * (1.0 - done_b)
                total_loss = total_loss + F.smooth_l1_loss(q_k, q_tgt)

        loss = total_loss / self.flat_dim
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 1.0)
        self.optimizer.step()
        return float(loss.item())

    def sync_target(self, tau: float = 0.005):
        for p_on, p_tg in zip(self.online.parameters(), self.target.parameters()):
            p_tg.data.copy_(tau * p_on.data + (1.0 - tau) * p_tg.data)

    def save(self, path: str):
        torch.save({"online": self.online.state_dict(), "target": self.target.state_dict()}, path)


# =============================================================================
# Centralized replay buffer — stores (obs, joint_idx, scalar_reward, next_obs, done)
# =============================================================================

class CentralizedReplayBuffer:
    def __init__(self, capacity: int = 50_000):
        self.buf = deque(maxlen=capacity)

    def push(self, obs, actions, reward: float, next_obs, done: bool):
        """
        obs/next_obs : (obs_dim,)
        actions      : np.ndarray (flat_dim,) — full flat action vector
        """
        self.buf.append((obs, np.array(actions, dtype=np.int64), float(reward), next_obs, float(done)))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        obs, acts, rews, nobs, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(obs)),    # (B, obs_dim)
            torch.LongTensor(np.array(acts)),    # (B, flat_dim)
            torch.FloatTensor(np.array(rews)),   # (B,)
            torch.FloatTensor(np.array(nobs)),   # (B, obs_dim)
            torch.FloatTensor(np.array(dones)),  # (B,)
        )

    def __len__(self):
        return len(self.buf)


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
# Training loop
# =============================================================================

DEFAULT_CFG = {
    "ssh_user":        "ubuntu",
    "use_real":        False,
    "step_interval":   2.0,
    "max_steps":       50,
    "n_episodes":      5000,
    "batch_size":      128,
    "replay_capacity": 50_000,
    "target_sync":     200,
    "tau":             0.005,
    "eps_start":       1.0,
    "eps_end":         0.05,
    "eps_decay":       0.9995,
    "lr":              1e-4,
    "gamma":           0.99,
    "hidden":          256,   # larger hidden — needs to handle 256 outputs
    "device":          "cpu",
    "log_every":       10,
    "save_every":      500,
    "results_dir":     "centralized_train_logs",
}


def train(cfg: dict):
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

    obs_dim   = env.observation_space.shape[1]
    per_agent_dim = env.per_agent_act_dim        # 1 + 2*N_RUS = 17
    #n_joint       = per_agent_dim ** N_NODES     # 17^4 — theoretical joint

    n_joint = (2 * (N_NODES ** (2 * N_RUS))) ** N_NODES

    flat_dim      = N_NODES * per_agent_dim      # 68 — actual flat vector
    true_joint    = (2 * (N_NODES ** (2 * N_RUS))) ** N_NODES   # true combinatorial space

    print(f"{'='*60}")
    print(f"  Centralized Single-Agent DRL Baseline")
    print(f"  Mode       : {'REAL' if cfg['use_real'] else 'SYNTHETIC'}")
    print(f"  Episodes   : {cfg['n_episodes']}")
    print(f"  Obs dim    : {obs_dim}  |  Nodes: {N_NODES}  |  RUs: {N_RUS}")
    print(f"  Per-agent act dim : {per_agent_dim}  |  Flat action dim: {flat_dim}")
    print(f"  True joint space   : {true_joint:.2e}  (intractable — combinatorial)")
    print(f"  Centralized output : {flat_dim}  (per-component independent selection)")
    print(f"  MA-DQN output      : {per_agent_dim} per agent × {N_NODES} agents = {flat_dim} total")
    print(f"{'='*60}\n")

    # ── Agent & replay ────────────────────────────────────────────────────────
    agent  = CentralizedAgent(
        obs_dim       = obs_dim,
        n_nodes       = N_NODES,
        per_agent_dim = per_agent_dim,
        hidden        = cfg["hidden"],
        lr            = cfg["lr"],
        gamma         = cfg["gamma"],
        device        = cfg["device"],
    )
    replay = CentralizedReplayBuffer(cfg["replay_capacity"])
    logger = CSVLogger(os.path.join(cfg["results_dir"], "dqn_training_log_12rus_9nodes_10000_episodes.csv"))

    # Pre-build joint action lookup for reverse mapping (actions → joint_idx)
    joint_actions = build_joint_action_space(N_NODES, N_ACTIONS)
    action_to_idx = {a: i for i, a in enumerate(joint_actions)}

    epsilon     = cfg["eps_start"]
    global_step = 0
    best_reward = -np.inf

    os.makedirs(cfg["results_dir"], exist_ok=True)

    start_time = time.time()

    for ep in range(1, cfg["n_episodes"] + 1):

        obs, info  = env.reset()
        global_obs = obs[0]   # all agents share same obs — take first

        ep_rewards    = []
        ep_energy    = []
        ep_losses     = []
        ep_saving     = []
        ep_nodes      = []
        ep_inference_ms = []
        ep_ocloud_util = []
        ep_peak_load   = []
        ep_energy_eff  = []
        ep_capacity_viol = []

        ep_proposed          = []
        ep_conflicts_step    = []
        ep_migration_success = []

        ep_qos        = []


        ep_migrations = 0
        ep_conflicts  = 0
        ep_failed     = 0
        ep_orphan     = 0


        ep_delay_viol   = []
        ep_delay_sla_ok = []
        ep_worst_delay  = []


        for step in range(cfg["max_steps"]):

            # ── Single agent selects JOINT action ─────────────────────────
            #actions   = agent.act(global_obs, epsilon, env.per_agent_nvec)
            train_start = time.time()

            actions, inference_ms = agent.act(global_obs, epsilon, env.per_agent_nvec)
            ep_inference_ms.append(inference_ms)
            # ── Environment step ──────────────────────────────────────────
            next_obs, rewards, done, trunc, info = env.step(actions)
            next_global = next_obs[0]

            
            # ── Scalar reward — mean across nodes ─────────────────────────
            scalar_reward = float(np.mean(rewards))

            # ── Store transition ──────────────────────────────────────────
            replay.push(global_obs, actions, scalar_reward, next_global, done or trunc)

            global_obs = next_global
            global_step += 1

            train_end = time.time()
            training_time = train_end - train_start

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

            capacity_viol = sum( 1 for l in info["node_loads"] if l > CAPACITY)

            # ── Conflict and migration success ─────────────────────────────
            n_proposed = sum(1 for n in range(N_NODES) if any(actions[n*per_agent_dim + 1:
                                    n*per_agent_dim + per_agent_dim] != n))   # agents proposing at least one migration
            
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
            ep_peak_load.append(peak_load)
            ep_energy_eff.append(eff)

            # ── Accumulate stats ──────────────────────────────────────────
            ep_rewards.append(scalar_reward)
            ep_energy.append(info["instant_energy_W"])
            ep_saving.append(info["energy_saving"])
            ep_nodes.append(info["nodes_active"])
            ep_qos.append(info["qos_violation"])

            ep_migrations += info.get("n_migrations", len(info.get("migrations", [])))
            ep_conflicts  += info.get("n_conflicts", 0)
            ep_failed     += info.get("n_failed_mig", 0)
            ep_orphan     += info.get("n_orphan_accept", 0)

            # ── Learn ─────────────────────────────────────────────────────
            if len(replay) >= cfg["batch_size"]:
                loss = agent.update(replay.sample(cfg["batch_size"]))
                ep_losses.append(loss)
                if global_step % cfg["target_sync"] == 0:
                    agent.sync_target(tau=cfg["tau"])

            if done or trunc:
                break

        # ── Epsilon decay ─────────────────────────────────────────────────
        epsilon = max(cfg["eps_end"], epsilon * cfg["eps_decay"])


        end_time = time.time()
            

        training_time = train_end - train_start
        

            
        execution_time = end_time - start_time

        # ── Episode summary ───────────────────────────────────────────────
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
            "avg_qos":           round(avg_qos, 6),
            "energy_saving_%":   round(avg_saving * 100, 2),
            "avg_nodes_active":  round(avg_nodes, 2),
            "ocloud_utilization":  round(float(np.mean(ep_ocloud_util)), 4),
            "peak_load":          round(float(np.mean(ep_peak_load)), 4),
            "avg_energy_eff":    round(float(np.mean(ep_energy_eff)), 4),
            "capacity_violations": int(sum(ep_capacity_viol)),

            "delay_violation_rate": round(float(np.mean(ep_delay_sla_ok) < 1.0), 4),
            "avg_delay_violation":  round(float(np.mean(ep_delay_viol)),  4),
            "worst_delay_ms":       round(float(np.max(ep_worst_delay)),  4),

            "total_migrations":  ep_migrations,
            "failed_migrations": ep_failed,
            "orphan_accepts":    ep_orphan,
            "total_conflicts":   ep_conflicts,
            "epsilon":           round(epsilon, 4),
            "global_step":       global_step,
            "avg_inference_ms":  round(np.mean(ep_inference_ms) if ep_inference_ms else 0, 4),
            "max_inference_ms":  round(np.max(ep_inference_ms)  if ep_inference_ms else 0, 4),
            "total_inference_ms":   round(np.sum(ep_inference_ms), 4),

            "execution_time": round(execution_time, 4),
            "training_time":  round(training_time, 4),

            "conflict_rate":        round(float(np.mean(ep_conflicts_step)) / max(float(np.mean(ep_proposed)), 1e-6), 4),
            "migration_success_rate": round(float(np.mean(ep_migration_success)), 4),

        })

        if ep % cfg["log_every"] == 0:
            print(
                f"\nEpisode: {ep}/{cfg['n_episodes']} | "
                f"Reward: {avg_reward:.4f} | "
                f"Energy: {avg_energy:.4f} | "
                f"Saving: {avg_saving*100:.1f}% | "
                f"Avg Nodes ON: {avg_nodes:.1f}/{N_NODES} | "
                f"Loss: {avg_loss:.4f} | "
                f"Epsilon: {epsilon:.3f} | "
                f"OCloud util: {np.mean(ep_ocloud_util)*100:.1f}% |"
                f"Peak load: {np.mean(ep_peak_load):.1f} | "
                f"Avg energy eff: {np.mean(ep_energy_eff):.1f}"
                f" | Capacity violations: {sum(ep_capacity_viol)}"
            )
            print(
                f"  Migrations: {ep_migrations} | "
                f"Failed: {ep_failed} | "
                f"Orphan: {ep_orphan} | "
                f"Joint output dim: {flat_dim}"
            )

        if ep % cfg["save_every"] == 0:
            agent.save(os.path.join(cfg["results_dir"], f"DQN/dqn_ep{ep}.pt"))
            print(f"  [checkpoint @ ep {ep}]")

        if avg_reward > best_reward:
            best_reward = avg_reward
            agent.save(os.path.join(cfg["results_dir"], "DQN/dqn_best.pt"))


        if ep == cfg["n_episodes"]:
            print(f"\n  Training complete. Best avg reward: {best_reward:.4f}")
            print(f"\n Energy saving: {avg_saving*100:.2f}% | Avg nodes active: {avg_nodes:.2f} | Avg energy: {avg_energy:.1f}Wh")
            print(f"  Avg QoS violation: {avg_qos:.4f} | Avg OCloud util: {np.mean(ep_ocloud_util)*100:.1f}% | Peak load: {np.mean(ep_peak_load):.1f} | Avg energy eff: {np.mean(ep_energy_eff):.1f}")
            print(f"  Capacity violations: {sum(ep_capacity_viol)} | Total migrations: {ep_migrations} | Failed migrations: {ep_failed} | Orphan accepts: {ep_orphan} | Total conflicts: {ep_conflicts}")
      

    logger.close()
    env.close()

    print(f"\n{'='*60}")
    print(f"  Centralized baseline complete.")
    print(f"  Best avg reward : {best_reward:.4f}")
    print(f"  True joint space: {true_joint:.2e}  |  Centralized output: {flat_dim}  |  MA-DQN: {N_NODES}×{per_agent_dim}")
    print(f"  Results: {cfg['results_dir']}/dqn_training_log.csv")
    print(f"{'='*60}")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Centralized single-agent DQN baseline for E-MATRO comparison")
    
    parser.add_argument("--n-rus", type=int, default=None)
    parser.add_argument("--real",          action="store_true")
    parser.add_argument("--ssh-user",      default="ubuntu")
    parser.add_argument("--step-interval", type=float, default=2.0)
    parser.add_argument("--max-steps",     type=int,   default=50)
    parser.add_argument("--episodes",      type=int,   default=5000)
    parser.add_argument("--batch-size",    type=int,   default=128)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--gamma",         type=float, default=0.99)
    parser.add_argument("--hidden",        type=int,   default=256)
    parser.add_argument("--eps-start",     type=float, default=1.0)
    parser.add_argument("--eps-end",       type=float, default=0.05)
    parser.add_argument("--eps-decay",     type=float, default=0.9995)
    parser.add_argument("--target-sync",   type=int,   default=200)
    parser.add_argument("--log-every",     type=int,   default=10)
    parser.add_argument("--save-every",    type=int,   default=500)
    parser.add_argument("--results-dir",   default="results")
    parser.add_argument("--device",        default="cpu")
    args = parser.parse_args()

    cfg = {**DEFAULT_CFG}
    cfg.update({
        "ssh_user":      args.ssh_user,
        "use_real":      args.real,
        "step_interval": args.step_interval,
        "max_steps":     args.max_steps,
        "n_episodes":    args.episodes,
        "batch_size":    args.batch_size,
        "lr":            args.lr,
        "gamma":         args.gamma,
        "hidden":        args.hidden,
        "eps_start":     args.eps_start,
        "eps_end":       args.eps_end,
        "eps_decay":     args.eps_decay,
        "target_sync":   args.target_sync,
        "log_every":     args.log_every,
        "save_every":    args.save_every,
        "results_dir":   args.results_dir,
        "device":        args.device,
    })

    train(cfg)