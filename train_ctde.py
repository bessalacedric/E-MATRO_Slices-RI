"""
train_ctde.py — E-MATRO MA-DQN with TRUE Centralized Training,
                 Decentralized Execution
                 (single-file version: networks + training loop)
=================================================================
This is the corrected CTDE implementation. The previous train.py had a
critical bug: the XAppActor (centralized coordination signal) was
instantiated with random weights and NEVER TRAINED — no optimizer, no
backward() call. As a result, "guidance" fed into every agent was
frozen random noise, which made CTDE perform WORSE than Independent
Q-Learning (IQL), since IQL agents don't have to filter out that noise.

Fix in this file:
  1. XApp guidance network gets its own optimizer.
  2. During the learning step, guidance is computed WITH gradients
     enabled (not under torch.no_grad()).
  3. Each agent's loss.backward() is called with retain_graph=True
     (except the very last agent), so gradients accumulate into the
     shared XApp's parameters from ALL agents that used its output.
  4. After all agents have backpropped, ONE xapp_optimizer.step() is
     taken — this is genuine joint/centralized training: the XApp
     learns to produce guidance that minimizes the agents' TD error,
     i.e. it learns what coordination signal actually helps.

This makes CTDE a fair, functioning baseline:
  - Training:  centralized (XApp sees global state, trained on
               aggregated gradient from all agents)
  - Execution: decentralized (each agent acts independently using
               only its local obs + the learned guidance — no
               agent-to-agent communication at inference time)

Usage:
  python3 train_ctde.py --episodes 5000 --max-steps 10
  python3 train_ctde.py --n-rus 8 --episodes 5000 --max-steps 10
"""

import os
import csv
import random
import argparse
import numpy as np
import torch
import time

from env_ocloud import OCloudEnv, N_NODES, N_RUS, N_ACTIONS, CAPACITY

# =============================================================================
# Networks — merged from networks.py (single-file version)
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

# =============================================================================
# Replay Buffer
# =============================================================================

class ReplayBuffer:
    """Shared experience replay for all agents."""

    def __init__(self, capacity: int = 50_000):
        self.buf = deque(maxlen=capacity)

    def push(self, obs, actions, rewards, next_obs, done):
        """
        obs       : np.ndarray (N_NODES, obs_dim)
        actions   : np.ndarray (N_NODES, per_agent_act_dim)  — factorized
        rewards   : np.ndarray (N_NODES,)
        next_obs  : np.ndarray (N_NODES, obs_dim)
        done      : bool
        """
        self.buf.append((obs, actions, rewards, next_obs, float(done)))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(obs)),           # (B, N, obs_dim)
            torch.LongTensor(np.array(actions)),         # (B, N, per_agent_act_dim)
            torch.FloatTensor(np.array(rewards)),        # (B, N)
            torch.FloatTensor(np.array(next_obs)),       # (B, N, obs_dim)
            torch.FloatTensor(np.array(dones)),          # (B,)
        )

    def __len__(self):
        return len(self.buf)


# =============================================================================
# XApp Actor — centralized coordinator
# =============================================================================

class XAppActor(nn.Module):
    """
    Takes the global observation (first agent's obs = global obs since
    all agents share the same 22-feature global observation) and outputs
    a guidance vector g_t ∈ [0,1]^(3*N_NODES):
        [target_load_n, consolidate_flag_n, wake_threshold_n]  per node
    """

    def __init__(self, obs_dim: int, n_nodes: int, hidden: int = 128):
        super().__init__()
        self.n_nodes = n_nodes
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3 * n_nodes),
            nn.Sigmoid(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        """obs: (obs_dim,) → guidance: (3*n_nodes,)"""
        x = torch.FloatTensor(obs).unsqueeze(0)
        return self.net(x).squeeze(0).numpy()


# =============================================================================
# DApp DQN — one per O-Cloud node
# =============================================================================

class DAppDQN(nn.Module):
    """
    Factorized DQN for a single O-Cloud node agent.
    Input:  obs (obs_dim,) + guidance (3*n_nodes,)
    Output: separate Q-heads per action component:
      head_a:       Q ∈ R^2         (OFF/ON)
      head_DU[r]:   Q ∈ R^N_NODES  per RU r  (DU destination)
      head_CU[r]:   Q ∈ R^N_NODES  per RU r  (CU destination)
    """

    def __init__(self, obs_dim: int, n_nodes: int, n_rus: int,
                 hidden: int = 128):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_rus   = n_rus
        guidance_dim = 3 * n_nodes
        in_dim       = obs_dim + guidance_dim

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        # Factorized output heads
        self.head_a  = nn.Linear(hidden, 2)            # ON/OFF
        self.heads_du = nn.ModuleList(
            [nn.Linear(hidden, n_nodes) for _ in range(n_rus)]
        )
        self.heads_cu = nn.ModuleList(
            [nn.Linear(hidden, n_nodes) for _ in range(n_rus)]
        )

        # Orthogonal init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, obs: torch.Tensor,
                guidance: torch.Tensor) -> list[torch.Tensor]:
        """
        obs      : (B, obs_dim)
        guidance : (B, 3*n_nodes)
        Returns list of Q-tensors:
          [q_a(B,2), q_DU_r0(B,N), .., q_DU_r(R-1)(B,N),
                     q_CU_r0(B,N), .., q_CU_r(R-1)(B,N)]
        Length: 1 + 2*N_RUS
        """
        h    = self.trunk(torch.cat([obs, guidance], dim=-1))
        qs   = [self.head_a(h)]
        qs  += [hd(h) for hd in self.heads_du]
        qs  += [hd(h) for hd in self.heads_cu]
        return qs

    @torch.no_grad()
    def act(self, obs: np.ndarray, guidance: np.ndarray,
            epsilon: float = 0.0) -> np.ndarray:
        """
        Epsilon-greedy factorized action selection.
        Returns flat action vector of length 1 + 2*N_RUS:
          [a_i, m_DU_r0,..,m_DU_r(R-1), m_CU_r0,..,m_CU_r(R-1)]
        """
        
        n_components = 1 + 2 * self.n_rus
        if random.random() < epsilon:
            act    = np.zeros(n_components, dtype=np.int64)
            act[0] = random.randint(0, 1)
            act[1:] = np.random.randint(0, self.n_nodes, size=2*self.n_rus)
            return act, 0.0
 
        t0 = time.perf_counter()
        o  = torch.FloatTensor(obs).unsqueeze(0)
        g  = torch.FloatTensor(guidance).unsqueeze(0)
        qs = self.forward(o, g)
        action = np.array([int(q.argmax(dim=1).item()) for q in qs],
                          dtype=np.int64)
        t1 = time.perf_counter()
        return action, (t1 - t0) * 1000



# =============================================================================
# MA-DQN Agent — wraps DAppDQN + target network + optimizer
# =============================================================================

class MADQNAgent:
    """
    Single dApp agent:
    - online DAppDQN (trained at each step)
    - target DAppDQN (synced every target_sync steps)
    - Adam optimizer
    """

    def __init__(
        self,
        agent_id:  int,
        obs_dim:   int,
        n_nodes:   int,
        n_rus:     int,
        lr:        float = 1e-3, #1e-3
        gamma:     float = 0.99, #0.99
        hidden:    int   = 128,
        device:    str   = "cpu",
    ):
        self.agent_id   = agent_id
        self.gamma      = gamma
        self.device     = torch.device(device)
        self.n_nodes    = n_nodes
        self.n_rus      = n_rus
        self.n_act_comp = 1 + 2 * n_rus   # components per agent action

        self.online = DAppDQN(obs_dim, n_nodes, n_rus, hidden).to(self.device)
        self.target = DAppDQN(obs_dim, n_nodes, n_rus, hidden).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)

    def act(self, obs: np.ndarray, guidance: np.ndarray, epsilon: float) -> int:

        return self.online.act(obs, guidance, epsilon)

    def update(self, batch: tuple, guidance_batch: torch.Tensor, retain_graph: bool = False) -> float:
        """
        One Double-DQN gradient step with factorized action heads.

        batch = (obs_B, actions_B, rewards_B, next_obs_B, dones_B)
          obs_B      : (B, obs_dim)
          actions_B  : (B, n_act_comp)  — factorized: [a_i, m_DU_r0.., m_CU_r0..]
          rewards_B  : (B,)
          next_obs_B : (B, obs_dim)
          dones_B    : (B,)
        guidance_batch : (B, 3*n_nodes) — grad-enabled if the XApp guidance
          network is trained jointly (true CTDE); pass a detached tensor
          for a frozen-guidance ablation.
        retain_graph : True when guidance_batch's graph is shared across
          several agents' backward() calls in the same training step
          (needed so gradients also flow back into the shared XApp).
        """
        obs_b, acts_b, rews_b, nobs_b, dones_b = batch
        obs_b   = obs_b.to(self.device)
        acts_b  = acts_b.to(self.device)          # (B, n_act_comp)
        rews_b  = rews_b.to(self.device)
        nobs_b  = nobs_b.to(self.device)
        dones_b = dones_b.to(self.device)
        g_b     = guidance_batch.to(self.device)

        # Current Q per head — guidance flows through here; if g_b requires
        # grad, this is where the XApp's parameters receive gradient.
        qs_online = self.online(obs_b, g_b)        # list of (B, sz) tensors

        # Double DQN target per head — target side always detached
        with torch.no_grad():
            g_b_det        = g_b.detach()
            qs_next_online = self.online(nobs_b, g_b_det)
            qs_next_target = self.target(nobs_b, g_b_det)

        total_loss = torch.tensor(0.0, device=self.device)
        for k, q_k in enumerate(qs_online):
            # k=0: a_i head (size 2); k=1..R: DU heads; k=R+1..2R: CU heads
            act_k    = acts_b[:, k]                            # (B,)
            q_taken  = q_k.gather(1, act_k.unsqueeze(1)).squeeze(1)  # (B,)

            with torch.no_grad():
                best_k  = qs_next_online[k].argmax(dim=1, keepdim=True)
                q_next  = qs_next_target[k].gather(1, best_k).squeeze(1)
                q_tgt   = rews_b + self.gamma * q_next * (1.0 - dones_b)

            total_loss = total_loss + F.smooth_l1_loss(q_taken, q_tgt)

        # Average over heads
        loss = total_loss / len(qs_online)
        self.optimizer.zero_grad()
        loss.backward(retain_graph=retain_graph)
        nn.utils.clip_grad_norm_(self.online.parameters(), 1.0)
        self.optimizer.step()
        return float(loss.item())

    def sync_target(self, tau: float = 1.0):
        """
        Sync target network.
        tau=1.0  → hard copy (original DQN)
        tau=0.01 → soft Polyak update (more stable)
        """
        for p_online, p_target in zip(
            self.online.parameters(), self.target.parameters()
        ):
            p_target.data.copy_(
                tau * p_online.data + (1.0 - tau) * p_target.data
            )

    def save(self, path: str):
        torch.save({
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optim":  self.optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.online.load_state_dict(ckpt["online"])
        self.target.load_state_dict(ckpt["target"])
        self.optimizer.load_state_dict(ckpt["optim"])



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
    "xapp_lr":         1e-4,   #1e-4 # XApp coordination network learning rate
    "gamma":           0.99, #0.99
    "hidden":          128,
    "device":          "cpu",

    "log_every":       10,
    "save_every":      500,
    "results_dir":     "results",

    "warmup_steps":    1000,
    "guided_prob":     0.3,
}


# =============================================================================
# Guided exploration (unchanged from original train.py)
# =============================================================================

def guided_action(last_info, global_step, n_nodes, warmup_steps=1000,
                  guided_prob=0.3):
    """During warmup, occasionally force a coordinated SLEEP+ACCEPT-style
    activation pattern by biasing a_i decisions — same helper as before."""
    if global_step >= warmup_steps:
        return None
    if random.random() >= guided_prob:
        return None
    # Bias: randomly pick one node to deactivate (encourages early
    # exploration of consolidation, same spirit as original helper)
    a_i = [1] * n_nodes
    if last_info and "placement" in last_info:
        pass
    src = random.randint(0, n_nodes - 1)
    a_i[src] = 0
    return a_i


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
# Training loop — TRUE CTDE
# =============================================================================

def train(cfg: dict):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # ── Topology override ───────────────────────────────────────────────────
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

    # ── Environment ───────────────────────────────────────────────────────────
    env = OCloudEnv(
        ssh_user         = cfg["ssh_user"],
        max_steps        = cfg["max_steps"],
        use_real_traffic = cfg["use_real"],
        step_interval    = cfg["step_interval"] if cfg["use_real"] else 0.0,
    )

    obs_dim           = env.observation_space.shape[1]
    n_nodes            = N_NODES
    per_agent_act_dim  = env.per_agent_act_dim
    guidance_dim       = 3 * n_nodes

    print(f"{'='*60}")
    print(f"  E-MATRO MA-DQN — TRUE CTDE Training")
    print(f"  Mode       : {'REAL' if cfg['use_real'] else 'SYNTHETIC'}")
    print(f"  Episodes   : {cfg['n_episodes']}")
    print(f"  Obs dim    : {obs_dim}  |  Nodes: {n_nodes}  |  RUs: {N_RUS}")
    print(f"  Per-agent act dim : {per_agent_act_dim}")
    print(f"  XApp guidance dim : {guidance_dim}  (NOW TRAINED — lr={cfg['xapp_lr']})")
    print(f"{'='*60}\n")

    # ── Networks ──────────────────────────────────────────────────────────────
    xapp = XAppActor(obs_dim, n_nodes, hidden=cfg["hidden"])
    xapp_optimizer = torch.optim.Adam(xapp.parameters(), lr=cfg["xapp_lr"])

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
    per_agent_act_dim_chk = per_agent_act_dim

    # Shared centralized replay buffer (CTDE — agents trained on joint
    # experience, even though guidance itself is what's shared/centralized)
    replay = ReplayBuffer(cfg["replay_capacity"])
    logger = CSVLogger(os.path.join(cfg["results_dir"], "ctde_training_log_8rus_6nodes_10000_episodes.csv"))

    epsilon     = cfg["eps_start"]
    global_step = 0
    best_reward = -np.inf

    os.makedirs(cfg["results_dir"], exist_ok=True)

    for ep in range(1, cfg["n_episodes"] + 1):

        obs, info = env.reset()
        with torch.no_grad():
            guidance = xapp.act(obs[0])

        ep_rewards        = []
        ep_energy    = []
        ep_losses           = []
        ep_xapp_losses        = []
        ep_saving               = []
        ep_nodes                  = []
        ep_migrations                = 0
        ep_conflicts                   = 0
        ep_failed_mig                     = 0
        ep_orphan_accept                     = 0
        ep_inference_ms                        = []
        ep_max_inference_ms        = []

        ep_ocloud_util = []
        ep_peak_load   = []
        ep_energy_eff  = []
        ep_capacity_viol = []

        ep_proposed          = []
        ep_conflicts_step    = []
        ep_migration_success = []

        ep_qos        = []


        ep_delay_viol   = []
        ep_delay_sla_ok = []
        ep_worst_delay  = []



        last_info = info

        for step in range(cfg["max_steps"]):

            # ── Action selection — guided warmup or epsilon-greedy ─────────
            guided = guided_action(
                last_info, global_step, n_nodes,
                cfg.get("warmup_steps", 1000), cfg.get("guided_prob", 0.3)
            )

            per_agent    = []
            inference_ms = []
            for n in range(n_nodes):
                act_n, t_ms = agents[n].act(obs[n], guidance, epsilon)
                per_agent.append(act_n)
                inference_ms.append(t_ms)

            flat_actions = np.concatenate(per_agent)
            if guided is not None:
                for n in range(n_nodes):
                    flat_actions[n * per_agent_act_dim] = min(guided[n], 1)

            ep_inference_ms.append(sum(inference_ms))

            max_inference_ms  = max(inference_ms)   # bottleneck agent

            ep_max_inference_ms.append(max_inference_ms)


            # ── Environment step ──────────────────────────────────────────
            next_obs, rewards, done, trunc, info = env.step(flat_actions)

            node_loads   = info["node_loads"]                          # L_i × a_i already
            # nodes_active = info["nodes_active"]                          # active nodes 
            # ocloud_util  = (sum(node_loads) / nodes_active if nodes_active > 0 else 0.0)

            nodes_active = max(info["nodes_active"], 1)
            ocloud_util  = sum(info["node_loads"]) / nodes_active
            ocloud_util  = min(ocloud_util, 1.0)         # clip to valid range
            

            peak_load = max(info["node_loads"])

            eff = info["energy_saving"] / (ocloud_util + 1e-6)

            capacity_viol = sum(
                1 for l in info["node_loads"] if l > CAPACITY
            )

            ep_delay_viol.append(info["delay_violation"])
            ep_delay_sla_ok.append(float(info["delay_sla_ok"]))

            # Worst-case delay across active nodes
            active_delays = [d for d in info["node_delays_ms"] if d < float('inf')]
            ep_worst_delay.append(max(active_delays) if active_delays else 0.0)
          

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
          


            with torch.no_grad():
                next_guidance = xapp.act(next_obs[0])

            # ── Store in shared replay buffer ───────────────────────────────
            acts_per_agent = flat_actions.reshape(n_nodes, per_agent_act_dim)
            replay.push(obs, acts_per_agent, rewards, next_obs, done or trunc)

            obs       = next_obs
            guidance  = next_guidance
            last_info = info
            global_step += 1

            ep_rewards.append(float(np.mean(rewards)))
            ep_saving.append(info["energy_saving"])
            ep_energy.append(info["instant_energy_W"])
            ep_nodes.append(info["nodes_active"])
            ep_qos.append(info["qos_violation"])
            ep_ocloud_util.append(ocloud_util)
            ep_peak_load.append(peak_load)
            ep_energy_eff.append(eff)
            ep_capacity_viol.append(capacity_viol)

            ep_migrations    += info.get("n_migrations", 0)
            ep_conflicts      += info.get("n_conflicts", 0)
            ep_failed_mig      += info.get("n_failed_mig", 0)
            ep_orphan_accept     += info.get("n_orphan_accept", 0)

            # ── TRUE CTDE learning step ─────────────────────────────────────
            if len(replay) >= cfg["batch_size"]:
                obs_b, acts_b, rews_b, nobs_b, dones_b = \
                    replay.sample(cfg["batch_size"])

                # Guidance computed WITH gradients — this is the fix.
                # Previously this was under @torch.no_grad() inside xapp.act(),
                # so the XApp never received gradient and stayed at its
                # random initialization forever.
                xapp_optimizer.zero_grad()
                g_batch = xapp.forward(obs_b[:, 0, :])   # (B, guidance_dim), grad-enabled

                step_loss = 0.0
                for n in range(n_nodes):
                    agent_batch = (
                        obs_b[:, n, :],
                        acts_b[:, n, :],
                        rews_b[:, n],
                        nobs_b[:, n, :],
                        dones_b,
                    )
                    is_last = (n == n_nodes - 1)
                    loss = agents[n].update(
                        agent_batch, g_batch,
                        retain_graph=not is_last,
                    )
                    step_loss += loss

                # One joint update for the XApp, using gradient accumulated
                # from ALL agents' backward() calls above.
                torch.nn.utils.clip_grad_norm_(xapp.parameters(), 1.0)
                xapp_optimizer.step()

                ep_losses.append(step_loss / n_nodes)

                if global_step % cfg["target_sync"] == 0:
                    for n in range(n_nodes):
                        agents[n].sync_target(tau=cfg["tau"])

            if done or trunc:
                break

        epsilon = max(cfg["eps_end"], epsilon * cfg["eps_decay"])

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
            "failed_migrations": ep_failed_mig,
            "orphan_accepts":    ep_orphan_accept,
            "total_conflicts":   ep_conflicts,
            "avg_inference_ms":  round(float(np.mean(ep_inference_ms)) if ep_inference_ms else 0, 4),
            "max_inference_ms":     round(float(np.max(ep_max_inference_ms)) if ep_max_inference_ms else 0, 4),
            "total_inference_ms":   round(float(np.sum(ep_inference_ms)) if ep_inference_ms else 0, 4),
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
                f"OCloud util: {np.mean(ep_ocloud_util)*100:.1f}% | "
                f"Peak load: {np.mean(ep_peak_load):.1f} | "
                f"Avg energy eff: {np.mean(ep_energy_eff):.1f}"
                f" | Capacity violations: {sum(ep_capacity_viol)}"
            )
            print(
                f"  Migrations: {ep_migrations} | "
                f"Failed: {ep_failed_mig} | "
                f"Conflicts: {ep_conflicts} | "
                f"XApp guidance: TRAINED (lr={cfg['xapp_lr']})"
            )

        if ep % cfg["save_every"] == 0:
            for n in range(n_nodes):
                agents[n].save(
                    os.path.join(cfg["results_dir"], f"CTDE/ctde_agent_{n}_ep{ep}.pt"))
            torch.save(xapp.state_dict(),
                      os.path.join(cfg["results_dir"], f"CTDE/ctde_xapp_ep{ep}.pt"))
            print(f"  [checkpoint @ ep {ep}]")

        if avg_reward > best_reward:
            best_reward = avg_reward
            for n in range(n_nodes):
                agents[n].save(
                    os.path.join(cfg["results_dir"], f"CTDE/ctde_agent_{n}_best.pt"))
            torch.save(xapp.state_dict(),
                      os.path.join(cfg["results_dir"], "CTDE/ctde_xapp_best.pt"))


        if ep == cfg["n_episodes"]:
            print(f"\n  Training complete. Best avg reward: {best_reward:.4f}")
            print(f"\n Energy saving: {avg_saving*100:.2f}% | Avg nodes active: {avg_nodes:.2f} | Avg energy: {avg_energy:.1f}Wh")
            print(f"  Avg QoS violation: {avg_qos:.4f} | Avg OCloud util: {np.mean(ep_ocloud_util)*100:.1f}% | Peak load: {np.mean(ep_peak_load):.1f} | Avg energy eff: {np.mean(ep_energy_eff):.1f}")
            print(f"  Capacity violations: {sum(ep_capacity_viol)} | Total migrations: {ep_migrations} | Failed migrations: {ep_failed_mig} | Orphan accepts: {ep_orphan_accept} | Total conflicts: {ep_conflicts}")
      

    logger.close()
    env.close()

    print(f"\n{'='*60}")
    print(f"  E-MATRO CTDE training complete.")
    print(f"  Best avg reward : {best_reward:.4f}")
    print(f"  Results: {cfg['results_dir']}/ctde_training_log.csv")
    print(f"{'='*60}")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="E-MATRO MA-DQN — true CTDE training (XApp jointly trained)"
    )
    parser.add_argument("--n-rus",         type=int, default=None)
    parser.add_argument("--real",          action="store_true")
    parser.add_argument("--ssh-user",      default="ubuntu")
    parser.add_argument("--step-interval", type=float, default=2.0)
    parser.add_argument("--max-steps",     type=int,   default=10)
    parser.add_argument("--episodes",      type=int,   default=5000)
    parser.add_argument("--batch-size",    type=int,   default=128)
    parser.add_argument("--lr",            type=float, default=1e-3) #1e-3
    parser.add_argument("--xapp-lr",       type=float, default=1e-3) #1e-3
    parser.add_argument("--gamma",         type=float, default=0.99) #0.99
    parser.add_argument("--hidden",        type=int,   default=128)
    parser.add_argument("--eps-start",     type=float, default=1.0)
    parser.add_argument("--eps-end",       type=float, default=0.05)
    parser.add_argument("--eps-decay",     type=float, default=0.9999) #0.9995
    parser.add_argument("--target-sync",   type=int,   default=200)
    parser.add_argument("--tau",           type=float, default=0.005)
    parser.add_argument("--warmup-steps",  type=int,   default=1000)
    parser.add_argument("--guided-prob",   type=float, default=0.3)
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
        "xapp_lr":         args.xapp_lr,
        "gamma":           args.gamma,
        "hidden":          args.hidden,
        "eps_start":       args.eps_start,
        "eps_end":         args.eps_end,
        "eps_decay":       args.eps_decay,
        "target_sync":     args.target_sync,
        "tau":             args.tau,
        "warmup_steps":    args.warmup_steps,
        "guided_prob":     args.guided_prob,
        "log_every":       args.log_every,
        "save_every":      args.save_every,
        "results_dir":     args.results_dir,
        "device":          args.device,
    })

    train(cfg)