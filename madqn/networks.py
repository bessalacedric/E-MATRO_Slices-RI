"""
network.py — MA-DQN Networks for E-MATRO++
==========================================
Matches new ocloud_env.py
"""

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
        actions   : np.ndarray (N_NODES,)
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

    def __init__(self, obs_dim: int, n_nodes: int, n_rus: int, hidden: int = 128):
        super().__init__()
        self.n_nodes    = n_nodes
        self.n_rus      = n_rus
        self.n_act_comp = 1 + 2 * n_rus   # [a_i, m_DU_r0..r(R-1), m_CU_r0..r(R-1)]
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
            epsilon: float = 0.0) -> tuple:
        """
        Epsilon-greedy factorized action selection.
        Returns (action_array, inference_ms):
          action_array : flat vector length 1 + 2*N_RUS
          inference_ms : float, 0.0 if random exploration
        """
        import time
        n_components = 1 + 2 * self.n_rus
        if random.random() < epsilon:
            act    = np.zeros(n_components, dtype=np.int64)
            act[0] = random.randint(0, 1)
            act[1:] = np.random.randint(0, self.n_nodes, size=2*self.n_rus)
            return act, 0.0   # no inference

        t0 = time.perf_counter()
        o  = torch.FloatTensor(obs).unsqueeze(0)
        g  = torch.FloatTensor(guidance).unsqueeze(0)
        qs = self.forward(o, g)
        action = np.array([int(q.argmax(dim=1).item()) for q in qs], dtype=np.int64)
        t1 = time.perf_counter()
        return action, (t1 - t0) * 1000
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
        lr:        float = 1e-3,
        gamma:     float = 0.99,
        hidden:    int   = 128,
        device:    str   = "cpu",
    ):
        self.agent_id = agent_id
        self.gamma    = gamma
        self.device   = torch.device(device)
        self.n_nodes    = n_nodes
        self.n_rus      = n_rus
        self.n_act_comp = 1 + 2 * self.n_rus
 
        self.online = DAppDQN(obs_dim, n_nodes, n_rus, hidden).to(self.device)
        self.target = DAppDQN(obs_dim, n_nodes, n_rus, hidden).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
 
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
 
    
    def act(self, obs: np.ndarray, guidance: np.ndarray, epsilon: float) -> tuple:
        return self.online.act(obs, guidance, epsilon)  # (action_array, inference_ms)
 

 
    def update(self, batch: tuple, guidance_batch: torch.Tensor) -> float:
        """
        One Double-DQN gradient step with factorized action heads.
 
        batch = (obs_B, actions_B, rewards_B, next_obs_B, dones_B)
          obs_B      : (B, obs_dim)
          actions_B  : (B, n_act_comp)  — factorized: [a_i, m_DU_r0.., m_CU_r0..]
          rewards_B  : (B,)
          next_obs_B : (B, obs_dim)
          dones_B    : (B,)
        guidance_batch : (B, 3*n_nodes)
        """
        obs_b, acts_b, rews_b, nobs_b, dones_b = batch
        obs_b   = obs_b.to(self.device)
        acts_b  = acts_b.to(self.device)          # (B, n_act_comp)
        rews_b  = rews_b.to(self.device)
        nobs_b  = nobs_b.to(self.device)
        dones_b = dones_b.to(self.device)
        g_b     = guidance_batch.to(self.device)
 
        # Current Q per head
        qs_online = self.online(obs_b, g_b)        # list of (B, sz) tensors
 
        # Double DQN target per head
        with torch.no_grad():
            qs_next_online = self.online(nobs_b, g_b)
            qs_next_target = self.target(nobs_b, g_b)
 
        total_loss = torch.tensor(0.0, device=self.device)
        for k, q_k in enumerate(qs_online):
            # k=0: a_i head (size 2); k=1..R: DU heads; k=R+1..2R: CU heads
            act_k    = acts_b[:, k].clamp(0, q_k.shape[1] - 1)  # guard OOB
            q_taken  = q_k.gather(1, act_k.unsqueeze(1)).squeeze(1)  # (B,)
 
            with torch.no_grad():
                best_k  = qs_next_online[k].argmax(dim=1, keepdim=True)
                q_next  = qs_next_target[k].gather(1, best_k).squeeze(1)
                q_tgt   = rews_b + self.gamma * q_next * (1.0 - dones_b)
            total_loss = total_loss + F.smooth_l1_loss(q_taken, q_tgt)
 
        # Average over heads
        loss = total_loss / len(qs_online)
        self.optimizer.zero_grad()
        loss.backward()
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
