"""
env_ocloud.py — E-MATRO MA-DQN Environment with Real Traffic Collection
===========================================
State variables (paper formulation):
  w_ri_DU[r, i] ∈ {0,1}  — RU r's DU is placed on node i
  w_ri_CU[r, i] ∈ {0,1}  — RU r's CU is placed on node i
  z_r_ij[r, i, j] ∈ {0,1} — RU r's DU is on node i and CU on node j

Deployment: SLICES-RI — 4 O-Cloud nodes, 4 RUs, 1 UE per RU

Consolidation target (energy saving):
  4 nodes (0%)  → baseline
  2 nodes (50%) → DUs consolidated on node0, CUs on node0
  1 node  (75%) → ALL VNFs on node0  (CU load ≈ 0, safe)
"""

import subprocess
import time
import logging
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

# ── Topology — single source of truth ─
import topology as _topo


# Offline test (no testbed needed)
#python3 env_ocloud.py --steps 5

# Real traffic from testbed
#python3 env_ocloud.py --real --user ubuntu --steps 5





N_NODES   = _topo.N_NODES
N_RUS     = _topo.N_RUS
N_DUS     = _topo.N_DUS
N_CUS     = _topo.N_CUS
N_CORES   = _topo.N_CORES
N_PRB_MAX = _topo.N_PRB_MAX

P_SLEEP   = _topo.P_SLEEP
P_IDLE    = _topo.P_IDLE
P_MAX     = _topo.P_MAX
P_BASE    = _topo.P_BASE
CAPACITY  = _topo.CAPACITY



P_active  = P_IDLE + (P_MAX - P_IDLE) * 0.25  # typical active power (25% load)



NODE_CONFIGS          = _topo.NODE_CONFIGS
INITIAL_PLACEMENT_DU  = _topo.INITIAL_PLACEMENT_DU
INITIAL_PLACEMENT_CU  = _topo.INITIAL_PLACEMENT_CU
RU_TO_DU              = _topo.RU_TO_DU
RU_TO_CU              = _topo.RU_TO_CU
# ─────────────────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)

# Actions per agent — factorized space:
#   [0]          a_i ∈ {0=OFF, 1=ON}           node activation
#   [1:N_RUS+1]  m_DU[r] ∈ {0..N_NODES-1}      DU migration destination per RU
#   [N_RUS+1:]   m_CU[r] ∈ {0..N_NODES-1}      CU migration destination per RU
# Interpretation: m_DU[r] = i means "keep on this node"; != i means "migrate to i"
# Per-agent action dim = 1 + 2*N_RUS
A_OFF = 0    # deactivate node
A_ON  = 1    # keep/activate node



# Actions (per agent/node)
ACTION_IDLE   = 0
ACTION_SLEEP  = 1
ACTION_ACCEPT = 2
ACTION_WAKE   = 3
N_ACTIONS     = 4





# Reward weights
W_ENERGY   = 1.0
W_QOS      = 2.0
W_CONFLICT = 0.1


W_LOAD = 1.0

# ── Delay model ───────────────────────────────────────────────────────────────
D_BASE_MS      = 2.0    # base uplink delay (ms): fronthaul + DU proc + F1-U
D_MAX_MS       = 10.0   # uplink delay SLA budget per UE-RU (ms)
# Load threshold derived from M/M/1: D_base/(1-rho) = D_max
# => rho_delay = 1 - D_base/D_max
RHO_DELAY      = 1.0 - D_BASE_MS / D_MAX_MS   # = 0.80
W_DELAY        = 0.5    # penalty weight for delay violation in reward


# RU type load coefficients (GOPS per Mbps, from paper)
ALPHA_DU   = 0.6
ALPHA_CU   = 0.1

SSH_OPTS   = "-o StrictHostKeyChecking=no -o ConnectTimeout=3"

# =============================================================================
# Node configuration
# =============================================================================

# NodeConfig defined in topology.py

# NODE_CONFIGS now imported from topology.py

# =============================================================================
# Traffic collector
# =============================================================================

from topology import build_node_configs
NodeConfig = type(list(_topo.NODE_CONFIGS.values())[0])



class TrafficCollector:
    """
    Polls real load metrics from SLICES-RI.
    DU nodes → NPRB / N_PRB_MAX  (from container logs)
    CU nodes → CPU% / (100×cores) (from docker stats)
    """

    def __init__(self, configs: dict, ssh_user: str = "ubuntu"):
        self.configs  = configs
        self.user     = ssh_user
        self._cache   = {i: 0.0 for i in range(N_NODES)}

    def _ssh(self, ip: str, cmd: str) -> str:
        full = f'ssh {SSH_OPTS} {self.user}@{ip} "{cmd}"'
        return subprocess.check_output(
            full, shell=True, timeout=5, stderr=subprocess.DEVNULL
        ).decode().strip()

    def _nprb(self, node) -> float:
        """NPRB-based load for DU nodes ∈ [0,1]."""
        try:
            out = self._ssh(
                node.ip,
                f"docker logs {node.container} 2>&1 | "
                f"grep -oP 'NPRB \\\\K[0-9]+' | tail -4"
            )
            vals = [int(x) for x in out.split() if x.isdigit()]
            return min(sum(vals) / N_PRB_MAX, 1.0) if vals else 0.0
        except Exception:
            return self._cache[node.node_id]

    def _cpu(self, node) -> float:
        """CPU-based load for CU nodes ∈ [0,1]."""
        try:
            out = self._ssh(
                node.ip,
                f"docker stats --no-stream --format '{{{{.CPUPerc}}}}' "
                f"{node.container} 2>/dev/null | head -1 | tr -d '%'"
            )
            return min(float(out) / (100.0 * node.n_cores), 1.0) if out else 0.0
        except Exception:
            return self._cache[node.node_id]

    def _poll(self, node) -> tuple:
        load = self._nprb(node) if node.role == "DU" else self._cpu(node)
        self._cache[node.node_id] = load
        return node.node_id, load

    def get_ru_loads(self) -> tuple:
        """
        Returns (du_loads, cu_loads) each shape (N_RUS,).
        du_loads[r] = load produced by RU r's DU on its physical host
        cu_loads[r] = load produced by RU r's CU on its physical host

        Mapping: RU0,RU1 → DU node0;  RU2,RU3 → DU node1
                 RU0,RU1 → CU node2;  RU2,RU3 → CU node3
        """
        du_loads = np.zeros(N_RUS, dtype=np.float32)
        cu_loads = np.zeros(N_RUS, dtype=np.float32)

        with ThreadPoolExecutor(max_workers=N_NODES) as ex:
            futures = {ex.submit(self._poll, n): nid
                       for nid, n in self.configs.items()}
            results = {}
            for f in as_completed(futures, timeout=8):
                try:
                    nid, load = f.result()
                    results[nid] = load
                except Exception:
                    pass

        # Distribute DU load to RUs using topology mapping
        for nid, node in NODE_CONFIGS.items():
            if node.role == "DU":
                rus = [r for r in range(N_RUS) if INITIAL_PLACEMENT_DU[r] == nid]
                n_rus = max(len(rus), 1)
                for r in rus:
                    du_loads[r] = results.get(nid, 0.0) / n_rus

        # Distribute CU load to RUs using topology mapping
        for nid, node in NODE_CONFIGS.items():
            if node.role == "CU":
                rus = [r for r in range(N_RUS) if INITIAL_PLACEMENT_CU[r] == nid]
                n_rus = max(len(rus), 1)
                for r in rus:
                    cu_loads[r] = results.get(nid, 0.0) / n_rus

        return du_loads, cu_loads


# =============================================================================
# O-Cloud state — paper formulation
# =============================================================================

class OCloudState:
    """
    State using paper variables:
      w_ri_DU[r, i] = 1 if RU r's DU is on node i
      w_ri_CU[r, i] = 1 if RU r's CU is on node i
      z_r_ij[r, i, j] = 1 if RU r's DU→node i and CU→node j
      node_active[i]   = True if node i is ON
    """

    def __init__(self):
        # Placement matrices — binary, each row sums to 1
        self.w_ri_DU = np.zeros((N_RUS, N_NODES), dtype=np.float32)
        self.w_ri_CU = np.zeros((N_RUS, N_NODES), dtype=np.float32)

        # Routing matrix derived from placements
        self.z_r_ij  = np.zeros((N_RUS, N_NODES, N_NODES), dtype=np.float32)

        # Node activation variable: a_i[i] = 1 if node i is ON
        # Invariant: a_i[i] = 1 iff node i hosts at least one VNF
        self.a_i = np.ones(N_NODES, dtype=np.float32)

        # Per-RU VNF loads (updated from measurements)
        self.du_loads = np.zeros(N_RUS, dtype=np.float32)
        self.cu_loads = np.zeros(N_RUS, dtype=np.float32)

        # Initialize placement from topology.py (works for any N_RUS)
        for r in range(N_RUS):
            du_node = INITIAL_PLACEMENT_DU[r]
            cu_node = INITIAL_PLACEMENT_CU[r]
            self.w_ri_DU[r, du_node] = 1.0
            self.w_ri_CU[r, cu_node] = 1.0

        self._update_routing()

    def _update_routing(self):
        """z_r_ij[r,i,j] = w_ri_DU[r,i] × w_ri_CU[r,j]"""
        for r in range(N_RUS):
            self.z_r_ij[r] = np.outer(self.w_ri_DU[r], self.w_ri_CU[r])

    def node_load(self, i: int) -> float:
        """
        Total load on node i:
          L_i = Σ_r [ w_ri_DU[r,i] * du_loads[r] + w_ri_CU[r,i] * cu_loads[r] ]
        """
        if self.a_i[i] < 0.5:
            return 0.0
        return float( np.dot(self.w_ri_DU[:, i], self.du_loads) + np.dot(self.w_ri_CU[:, i], self.cu_loads))

    def all_node_loads(self) -> np.ndarray:
        return np.array([self.node_load(i) for i in range(N_NODES)], dtype=np.float32)

    def nodes_active_count(self) -> int:
        return int(self.a_i.sum())

    def vnfs_on_node(self, i: int) -> dict:
        """Returns {'DU': [ru_ids], 'CU': [ru_ids]} hosted on node i."""
        du_rus = [r for r in range(N_RUS) if self.w_ri_DU[r, i] > 0.5]
        cu_rus = [r for r in range(N_RUS) if self.w_ri_CU[r, i] > 0.5]
        return {"DU": du_rus, "CU": cu_rus}

    def can_accept(self, dst: int, src_vnfs: dict) -> bool:
        """Check if node dst has capacity to absorb VNFs from src."""
        if self.a_i[dst] < 0.5:
            return False
        current = self.node_load(dst)
        # Estimate added load from migrating VNFs
        added = (len(src_vnfs["DU"]) * np.mean(self.du_loads + 1e-6) / max(1, len(src_vnfs["DU"]))
                + len(src_vnfs["CU"]) * np.mean(self.cu_loads + 1e-6))
        # Hard block — reject migration that would exceed capacity
        return (current + added) < CAPACITY



    def migrate_all(self, src: int, dst: int):
        """
        Move all VNFs from node src to node dst.
        Updates w_ri_DU, w_ri_CU, z_r_ij atomically.
        """
        for r in range(N_RUS):
            if self.w_ri_DU[r, src] > 0.5:
                self.w_ri_DU[r, src] = 0.0
                self.w_ri_DU[r, dst] = 1.0
            if self.w_ri_CU[r, src] > 0.5:
                self.w_ri_CU[r, src] = 0.0
                self.w_ri_CU[r, dst] = 1.0
        self._update_routing()

    def enforce_activation(self):
        """
        Hard constraint: a_i[i] = 1 iff node i hosts at least one VNF.
        Called after every placement change (migration or wake).
        """
        for i in range(N_NODES):
            vnfs = self.vnfs_on_node(i)
            has_vnf = bool(vnfs["DU"] or vnfs["CU"])
            self.a_i[i] = 1.0 if has_vnf else 0.0

    @property
    def node_active(self) -> np.ndarray:
        """Boolean view of a_i for backward compatibility."""
        return self.a_i > 0.5

    def placement_str(self) -> str:
        """Human-readable per-node layout."""
        parts = []
        for i in range(N_NODES):
            vnfs = self.vnfs_on_node(i)
            du_s = f"DU{vnfs['DU']}" if vnfs["DU"] else ""
            cu_s = f"CU{vnfs['CU']}" if vnfs["CU"] else ""
            content = "+".join(filter(None, [du_s, cu_s])) or "—"
            status  = "ON" if self.a_i[i] > 0.5 else "OFF"
            parts.append(f"node{i}[{status}]:{content}")
        return "  ".join(parts)


# =============================================================================
# Environment
# =============================================================================

class OCloudEnv(gym.Env):
    """
    E-MATRO MA-DQN environment.

    Observation — 26 features per agent (global, shared):
      [0:4]    node_loads[0..3]          ∈ [0,1]
      [4:8]    node_active[0..3]         ∈ {0,1}
      [8:24]   w_ri_DU.flatten()         (4×4) ∈ {0,1}
      [24:40]  w_ri_CU.flatten()         (4×4) ∈ {0,1}
      [40:44]  du_loads[0..3]            per-RU DU load
      [44:48]  cu_loads[0..3]            per-RU CU load
      [48]     nodes_active / N_NODES    consolidation level
      [49]     energy_saving             current saving ∈ [0,1]
    Total: 50 features

    Actions: MultiDiscrete([4,4,4,4])
      0=IDLE  1=SLEEP  2=ACCEPT  3=WAKE
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        ssh_user:         str   = "ubuntu",
        max_steps:        int   = 50,
        use_real_traffic: bool  = True,
        step_interval:    float = 2.0,
    ):
        super().__init__()
        self.max_steps     = max_steps
        self.step_interval = step_interval
        self.use_real      = use_real_traffic

        self.collector = TrafficCollector(NODE_CONFIGS, ssh_user)
        self.state     = OCloudState()
        self._step     = 0

        self.OBS_DIM = (
            N_NODES +              # a_i   — node activation
            N_RUS * N_NODES +      # w_ri_DU — DU placement
            N_RUS * N_NODES +      # w_ri_CU — CU placement
            N_NODES +              # L_i   — node load
            N_NODES * N_NODES      # z_ij  — link status matrix
        )


        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(N_NODES, self.OBS_DIM),
            dtype=np.float32,
        )


        # Factorized per-agent action space:
        # each agent decides (a_i, m_DU[0..R-1], m_CU[0..R-1])
        self.per_agent_nvec = np.array(
            [2]                    # a_i: 0=OFF, 1=ON
            + [N_NODES] * N_RUS    # m_DU[r]: destination node for RU r's DU
            + [N_NODES] * N_RUS,   # m_CU[r]: destination node for RU r's CU
            dtype=np.int64
        )
        self.per_agent_act_dim = len(self.per_agent_nvec)   # 1 + 2*N_RUS
        # Joint: all agents concatenated
        self.action_space = spaces.MultiDiscrete( np.tile(self.per_agent_nvec, N_NODES))



        #self.action_space = spaces.MultiDiscrete([N_ACTIONS] * N_NODES)





    # ── Observation ───────────────────────────────────────────────────────────

    def _build_obs(self) -> np.ndarray:
        # a_i — node activation
        a_i      = self.state.a_i.copy()                          # (N_NODES,)

        # w_ri_DU, w_ri_CU — placement matrices
        w_du     = self.state.w_ri_DU.flatten()                   # (N_RUS×N_NODES,)
        w_cu     = self.state.w_ri_CU.flatten()                   # (N_RUS×N_NODES,)

        # L_i — node load (combined, normalized)
        L_i      = self.state.all_node_loads()                    # (N_NODES,)

        # z_ij — link status: 1 if any RU's routing uses link (i→j)
        # Derived from z_r_ij: z_ij = max over r of z_r_ij[r,i,j]
        z_ij     = self.state.z_r_ij.max(axis=0).flatten()       # (N_NODES×N_NODES,)

        global_obs = np.concatenate([
            a_i,     # N_NODES
            w_du,    # N_RUS × N_NODES
            w_cu,    # N_RUS × N_NODES
            L_i,     # N_NODES
            z_ij,    # N_NODES × N_NODES
        ]).astype(np.float32)

        return np.tile(global_obs, (N_NODES, 1))

    
    # ── Load collection ───────────────────────────────────────────────────────

    def _collect_loads(self):
        if self.use_real:
            du_l, cu_l = self.collector.get_ru_loads()
        else:
            # Synthetic: DU idle ~0.25, CU idle ~0.001
            du_l = np.clip(np.random.normal(0.25, 0.03, N_RUS), 0, 1)
            cu_l = np.clip(np.random.normal(0.001, 5e-4, N_RUS), 0, 1)
        self.state.du_loads = du_l.astype(np.float32)
        self.state.cu_loads = cu_l.astype(np.float32)

    # ── Energy model ──────────────────────────────────────────────────────────


    def _total_power(self) -> float:
        p = 0.0
        for i in range(N_NODES):
            if self.state.a_i[i] < 0.5:
                p += P_SLEEP
            else:
                load = self.state.node_load(i)
                p += P_IDLE + (P_MAX - P_IDLE) * load #* self.state.a_i[i]
                #p += P_active * load  # add active power component #P_IDLE + (P_MAX - P_IDLE) * float(self.state.raw_loads[n])
        return p




    def _instant_energy(self) -> float:
        energy = 0.0
        for i in range(N_NODES):
            if self.state.a_i[i] < 0.5:
                energy += P_SLEEP
            else:
                energy += P_active
        return energy

    def _energy_saving(self) -> float:
        """
        Fraction of nodes sleeping = direct energy saving signal.
        Cleaner than power difference (avoids large-scale reward issues).
        """
        # n_sleeping = N_NODES - int(self.state.a_i.sum())
        # return float(n_sleeping / N_NODES)   # ∈ [0, 0.75] for 1→3 sleeping

        n_sleeping = N_NODES - self.state.nodes_active_count()
        return float(n_sleeping / N_NODES)   # ∈ [0, 0.75] for 1→3 sleeping


    # ── QoS ───────────────────────────────────────────────────────────────────

    def _qos_violation(self) -> float:
        
        total = 0.0
        for i in range(N_NODES):
            if self.state.a_i[i] > 0.5:
                load = self.state.node_load(i)
                if load > CAPACITY:
                    # much worse than marginal overload
                    total += (load - CAPACITY) ** 2 * 10.0
        return min(total / N_NODES, 1.0)


    
    def _delay_violation(self) -> float:
        """
        Fraction of active nodes whose estimated uplink delay exceeds D_MAX_MS.
        Uses M/M/1 queuing approximation:
        D(L_i) = D_BASE_MS / (1 - L_i)
        Violation when L_i > RHO_DELAY = 1 - D_BASE_MS/D_MAX_MS = 0.80

        Per-RU delay guarantee: every RU served by an overloaded node
        is counted as a delay violation. Normalized by total RUs.
        """
        violated_rus = 0
        for i in range(N_NODES):
            if self.state.a_i[i] > 0.5:
                load = self.state.node_load(i)
                if load > RHO_DELAY:
                    # Count RUs served by this overloaded node
                    for r in range(N_RUS):
                        if (self.state.w_ri_DU[r, i] > 0.5 or
                                self.state.w_ri_CU[r, i] > 0.5):
                            violated_rus += 1
        return min(violated_rus / max(N_RUS, 1), 1.0)


    def _estimated_delay_ms(self, node_i: int) -> float:
        """Estimated uplink delay for a UE served by node_i (M/M/1 model)."""
        if self.state.a_i[node_i] < 0.5:
            return float('inf')   # sleeping node — undefined
        load = self.state.node_load(node_i)
        load = min(load, 0.999)   # prevent division by zero
        return D_BASE_MS / (1.0 - load)



    def _parse_agent_action(self, flat_actions: np.ndarray,
                               agent_id: int) -> dict:
        """
        Parse flat joint action vector for one agent.
 
        flat_actions: full joint action array (N_NODES * per_agent_act_dim,)
        Returns dict with:
          a_i:        0=OFF / 1=ON
          mig_du:     list of (r, dst) for DU migrations
          mig_cu:     list of (r, dst) for CU migrations
        """
        pad  = agent_id * self.per_agent_act_dim
        acts = flat_actions[pad : pad + self.per_agent_act_dim]
 
        a_i   = int(acts[0])
        m_du  = acts[1 : N_RUS + 1]             # destination per RU — DU
        m_cu  = acts[N_RUS + 1 : 2 * N_RUS + 1] # destination per RU — CU
 
        mig_du, mig_cu = [], []
        for r in range(N_RUS):
            # Only migrate if this agent currently hosts the VNF
            if self.state.w_ri_DU[r, agent_id] > 0.5:
                dst = int(m_du[r])
                if dst != agent_id:             # actual migration
                    mig_du.append((r, dst))
            if self.state.w_ri_CU[r, agent_id] > 0.5:
                dst = int(m_cu[r])
                if dst != agent_id:
                    mig_cu.append((r, dst))
 
        return {"a_i": a_i, "mig_du": mig_du, "mig_cu": mig_cu}


    # ── Conflict resolution ───────────────────────────────────────────────────

    def _resolve(self, flat_actions: np.ndarray) -> tuple:
        """
        Parse all agents' actions and resolve conflicts.
        Conflict: two agents try to migrate the same VNF simultaneously.
        Returns (vnf_migrations, conflict_info).
          vnf_migrations: list of (vnf_type, r, src, dst)
        """
        parsed      = [self._parse_agent_action(flat_actions, i)
                       for i in range(N_NODES)]
        conflicts   = [False] * N_NODES
        vnf_migs    = []   # (type, ru, src_node, dst_node)
        claimed_vnf = {}   # (type, r) → first src that claimed it
 
        # Phase 1: collect migration intents per VNF
        for src in range(N_NODES):
            p = parsed[src]
            # DU migrations
            for r, dst in p["mig_du"]:
                key = ("DU", r)
                if key in claimed_vnf:
                    # Conflict: two agents want to move same VNF
                    conflicts[src] = True
                else:
                    claimed_vnf[key] = (src, dst)
            # CU migrations
            for r, dst in p["mig_cu"]:
                key = ("CU", r)
                if key in claimed_vnf:
                    conflicts[src] = True
                else:
                    claimed_vnf[key] = (src, dst)
 
        # Phase 2: validate destination capacity
        for (vtype, r), (src, dst) in claimed_vnf.items():
            vnfs = self.state.vnfs_on_node(src)
            if dst == src or not self.state.a_i[dst] > 0.5:
                conflicts[src] = True
                continue
            if not self.state.can_accept(dst, {"DU": [r] if vtype=="DU" else [], "CU": [r] if vtype=="CU" else []}):
                conflicts[src] = True
                continue
            vnf_migs.append((vtype, r, src, dst))
 
        # Phase 3: activation intents
        a_intents = [parsed[i]["a_i"] for i in range(N_NODES)]
 
        n_failed = sum(conflicts)
        n_orphan = 0   # no longer an "orphan accept" concept
 
        conflict_info = {
            "mask":     conflicts,
            "n_failed": n_failed,
            "n_orphan": n_orphan,
            "n_total":  n_failed,
        }
        return vnf_migs, a_intents, conflict_info


    # ── Step ─────────────────────────────────────────────────────────────────

    def step(self, actions):
        flat_actions = np.asarray(actions, dtype=np.int64)
        assert len(flat_actions) == N_NODES * self.per_agent_act_dim


        #assert len(actions) == N_NODES

        # 1. Collect loads
        self._collect_loads()

        # 2. Resolve migrations
        vnf_migs, a_intents, cinfo = self._resolve(flat_actions)
        conflicts = cinfo["mask"]
        n_failed  = cinfo["n_failed"]
        n_orphan  = cinfo["n_orphan"]


        # 3. Execute per-VNF migrations (w_ri_DU, w_ri_CU, z_r_ij updated)
        for vtype, r, src, dst in vnf_migs:
            if vtype == "DU":
                self.state.w_ri_DU[r, src] = 0.0
                self.state.w_ri_DU[r, dst] = 1.0
            else:
                self.state.w_ri_CU[r, src] = 0.0
                self.state.w_ri_CU[r, dst] = 1.0
            self.state._update_routing()
            logger.info(f"[{self._step}] {vtype} RU{r}: node{src}→node{dst}")


        # 4. Apply activation intents (a_i) then enforce hard constraint
        for n in range(N_NODES):
            self.state.a_i[n] = float(a_intents[n])
        self.state.enforce_activation()   # a_i=1 only if VNFs present
 

        # 5. Compute rewards — bounded to [-1, 1]
        r_energy   = W_ENERGY * self._energy_saving()
        r_qos      = W_QOS   * self._qos_violation()
        r_delay    = W_DELAY * self._delay_violation()
        # Penalize failed migration attempts (not orphan accepts separately)
        r_conflict = W_CONFLICT * n_failed / N_NODES

        n_migs      = len(vnf_migs)
        r_migration = -0.30 * n_migs / (N_RUS * 2)   # normalized by max possible

        r_load = W_LOAD * sum( max(0.0, self.state.node_load(i) - 1.0) for i in range(N_NODES) if self.state.a_i[i] > 0.5 )

        n_sleeping = N_NODES - self.state.nodes_active_count()
        r_consolidation = 0.5 * (n_sleeping / N_NODES)  # +0.5 max for full consolidation


        base = np.clip(r_energy - r_qos - r_delay - r_conflict - r_load - r_consolidation, -1.0, 1.0)

        #base = np.clip(-r_energy +r_qos + r_conflict + r_load - r_consolidation, -1.0, 1.0)
        rewards = np.full(N_NODES, base, dtype=np.float32)

        # Individual penalty: nodes that caused a failed migration
        for n in range(N_NODES):
            if conflicts[n]:
                rewards[n] = np.clip(rewards[n] - W_CONFLICT, -1.0, 1.0)

        # 6. Build observation
        obs = self._build_obs()
        self._step += 1
        done  = False
        trunc = self._step >= self.max_steps

        info = {
            "step":             self._step,
            "node_loads":       self.state.all_node_loads().tolist(),
            "du_loads":         self.state.du_loads.tolist(),
            "cu_loads":         self.state.cu_loads.tolist(),
            "w_ri_DU":          self.state.w_ri_DU.tolist(),
            "w_ri_CU":          self.state.w_ri_CU.tolist(),
            "nodes_active":     self.state.nodes_active_count(),
            "energy_saving":    self._energy_saving(),
            "power_W":          self._total_power(),
            "instant_energy_W": self._instant_energy(),
            "qos_violation":    self._qos_violation(),

            "delay_violation":  self._delay_violation(),
            "node_delays_ms":   [self._estimated_delay_ms(i) for i in range(N_NODES)],
            "delay_sla_ok":     self._delay_violation() == 0.0,
            # Conflict breakdown
            "n_conflicts":      n_failed + n_orphan,  # total
            "n_failed_mig":     n_failed,   # SLEEP with no ACCEPT partner
            "n_orphan_accept":  n_orphan,   # ACCEPT with no SLEEP partner
            "n_migrations":     len(vnf_migs),
            "migrations":       vnf_migs,
            "placement_str":    self.state.placement_str(),

        }


        if self.use_real and self.step_interval > 0:
            time.sleep(self.step_interval)

        return obs, rewards, done, trunc, info

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = OCloudState()
        self._step = 0
        self._collect_loads()
        obs  = self._build_obs()
        info = {
            "node_loads":   self.state.all_node_loads().tolist(),
            "du_loads":     self.state.du_loads.tolist(),
            "cu_loads":     self.state.cu_loads.tolist(),
            "nodes_active": self.state.nodes_active_count(),
            "placement_str":self.state.placement_str(),
        }
        return obs, info

    # ── Render ────────────────────────────────────────────────────────────────

    def render(self):
        loads    = self.state.all_node_loads()
        n_active = self.state.nodes_active_count()
        print(f"\n{'='*70}")
        print(
            f"  Step {self._step}/{self.max_steps} | "
            f"Active: {n_active}/{N_NODES} | "
            f"Power: {self._total_power():.0f}W | "
            f"Saving: {self._energy_saving()*100:.1f}%"
        )
        print(f"  {'Node':<6} {'Status':<6} {'Load':>6}  "
              f"{'w_ri_DU':>10}  {'w_ri_CU':>10}  VNFs")
        print(f"  {'-'*65}")
        for i in range(N_NODES):
            s  = "ON " if self.state.a_i[i] > 0.5 else "OFF"
            ld = loads[i]
            du_col = str(np.where(self.state.w_ri_DU[:, i] > 0.5)[0].tolist())
            cu_col = str(np.where(self.state.w_ri_CU[:, i] > 0.5)[0].tolist())
            vnfs   = self.state.vnfs_on_node(i)
            label  = (f"DU{vnfs['DU']}+" if vnfs["DU"] else "") + \
                     (f"CU{vnfs['CU']}" if vnfs["CU"] else "")
            print(f"  node{i:<3} {s:<6} {ld:>6.3f}  "
                  f"{'DUs:'+du_col:>12}  {'CUs:'+cu_col:>12}  {label or '—'}")
        print(f"\n  DU loads: {[f'{v:.3f}' for v in self.state.du_loads]}")
        print(f"  CU loads: {[f'{v:.4f}' for v in self.state.cu_loads]}")
        print(f"\n  w_ri_DU:\n{self.state.w_ri_DU}")
        print(f"  w_ri_CU:\n{self.state.w_ri_CU}")
        print(f"{'='*70}")

    def close(self):
        pass


# =============================================================================
# Quick test
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--real",     action="store_true")
    parser.add_argument("--user",     default="ubuntu")
    parser.add_argument("--steps",    type=int, default=4)
    parser.add_argument("--interval", type=float, default=2.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    env = OCloudEnv(
        ssh_user         = args.user,
        max_steps        = args.steps,
        use_real_traffic = args.real,
        step_interval    = args.interval if args.real else 0.0,
    )

    print(f"Obs shape : {env.observation_space.shape}")
    print(f"Act shape : {env.action_space.nvec.tolist()}")

    obs, info = env.reset()
    print(f"\nReset:")
    print(f"  {info['placement_str']}")
    env.render()

    # Consolidation sequence:
    # step1: node1→node0 (DU1 migrates), node3→node2 (CU3 migrates)
    # step2: node2→node0 (CU2+CU3 migrate → everything on node0)
    scenarios = [
        [ACTION_ACCEPT, ACTION_SLEEP, ACTION_ACCEPT, ACTION_SLEEP],
        [ACTION_ACCEPT, ACTION_IDLE,  ACTION_SLEEP,  ACTION_IDLE ],
        [ACTION_IDLE,   ACTION_IDLE,  ACTION_IDLE,   ACTION_IDLE ],
        [ACTION_IDLE,   ACTION_WAKE,  ACTION_IDLE,   ACTION_IDLE ],
    ]

    names = ["IDLE","SLEEP","ACCEPT","WAKE"]
    for i, actions in enumerate(scenarios[:args.steps]):
        obs, rewards, done, trunc, info = env.step(actions)
        print(f"\n--- Step {i+1}: {[names[a] for a in actions]} ---")
        print(f"  Rewards   : {[round(r,3) for r in rewards.tolist()]}")
        print(f"  Saving    : {info['energy_saving']*100:.1f}%")
        print(f"  Nodes ON  : {info['nodes_active']}/4")
        print(f"  Conflicts : {info['n_conflicts']}")
        print(f"  Migrations: {info['migrations']}")
        print(f"  Placement : {info['placement_str']}")
        env.render()
        if done or trunc:
            break

    env.close()
    print("\nDone.")