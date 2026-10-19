"""
topology_multi_gnb.py — Multi-gNB E-MATRO Topology
====================================================
Architecture: N_GNB independent gNBs, each with 4 UEs (4 RUs)
  → avoids OAI single-gNB scalability issues beyond 4 UEs
  → each DU container = 1 gNB with 4 UEs
  → each CU container serves N_GNB_PER_CU gNBs

Usage:
  import topology_multi_gnb as topo
  python3 train.py --topology multi_gnb --n-gnb 4   # 4 gNBs × 4 UEs = 16 UEs

Scenarios available:
  Scenario A (baseline):  1 gNB,  4 UEs,  2 nodes  ← current setup
  Scenario B:             2 gNBs, 8 UEs,  4 nodes
  Scenario C:             4 gNBs, 16 UEs, 8 nodes
  Scenario D:             8 gNBs, 32 UEs, 12 nodes
  Scenario E (max):       15 gNBs,60 UEs, 15 nodes
"""

# =============================================================================
# Primary scaling parameter — change N_GNB to scale
# =============================================================================

N_GNB        = 2    # number of independent gNBs (each has UES_PER_GNB UEs)
UES_PER_GNB  = 4    # fixed: 4 UEs per gNB (proven to work with OAI rfsim)

# Derived
N_UES  = N_GNB * UES_PER_GNB    # total UEs
N_RUS  = N_UES                    # 1 RU per UE

# Physical constraint
N_VMS  = 15    # vm-server-1..15 on SLICES-RI

# 1 DU container per gNB, CUs shared (1 CU per 2 DUs max)
N_DUS  = N_GNB
N_CUS  = max(1, (N_GNB + 1) // 2)
N_NODES = min(N_DUS + N_CUS, N_VMS)

N_CORES   = 4
N_PRB_MAX = 106

# Energy model
P_SLEEP = 10.0
P_IDLE  = 100.0
P_MAX   = 400.0
P_BASE  = N_NODES * P_IDLE
CAPACITY = 0.85

# =============================================================================
# Physical VM IPs
# =============================================================================

VM_IPS = [
    "10.10.208.109",   # vm-server-1
    "10.10.210.195",   # vm-server-2
    "10.10.211.225",   # vm-server-3
    "10.10.211.11",    # vm-server-4
    "10.10.208.178",   # vm-server-5
    "10.10.209.189",   # vm-server-6
    "10.10.210.208",   # vm-server-7
    "10.10.214.191",   # vm-server-8
    "10.10.215.173",   # vm-server-9
    "10.10.210.73",    # vm-server-10
    "10.10.209.31",    # vm-server-11
    "10.10.209.143",   # vm-server-12
    "10.10.211.241",   # vm-server-13
    "10.10.211.195",   # vm-server-14
    "10.10.215.211",   # vm-server-15
]

CTL_IP   = "10.10.208.188"
SSH_USER = "ubuntu"

# =============================================================================
# Node configs
# =============================================================================

from dataclasses import dataclass

@dataclass
class NodeConfig:
    node_id:   int
    role:      str    # "DU" or "CU"
    ip:        str
    container: str
    gnb_id:    int    # which gNB this DU/CU belongs to
    vm_index:  int
    n_cores:   int = N_CORES


def build_node_configs() -> dict:
    configs = {}
    # DU nodes: one per gNB
    for gnb in range(N_GNB):
        vm_idx = gnb % len(VM_IPS)
        configs[gnb] = NodeConfig(
            node_id   = gnb,
            role      = "DU",
            ip        = VM_IPS[vm_idx],
            container = f"oai-du-gnb{gnb}",
            gnb_id    = gnb,
            vm_index  = vm_idx,
        )
    # CU nodes: one per 2 DUs
    for cu_idx in range(N_CUS):
        node_id = N_DUS + cu_idx
        vm_idx  = node_id % len(VM_IPS)
        configs[node_id] = NodeConfig(
            node_id   = node_id,
            role      = "CU",
            ip        = VM_IPS[vm_idx],
            container = f"oai-cu-gnb{cu_idx}",
            gnb_id    = cu_idx,
            vm_index  = vm_idx,
        )
    return configs


NODE_CONFIGS = build_node_configs()


def build_ru_mapping() -> tuple:
    """
    Each gNB has UES_PER_GNB RUs.
    RU r → DU of gNB (r // UES_PER_GNB)
    RU r → CU serving that DU
    """
    ru_to_du = {}
    ru_to_cu = {}
    for r in range(N_RUS):
        gnb    = r // UES_PER_GNB
        du_node = gnb % N_NODES
        cu_node = (N_DUS + gnb // 2) % N_NODES
        ru_to_du[r] = du_node
        ru_to_cu[r] = cu_node
    return ru_to_du, ru_to_cu


RU_TO_DU, RU_TO_CU = build_ru_mapping()
INITIAL_PLACEMENT_DU = dict(RU_TO_DU)
INITIAL_PLACEMENT_CU = dict(RU_TO_CU)
RUS_PER_DU = UES_PER_GNB


def build_ue_names() -> dict:
    names = {}
    for i in range(N_UES):
        names[i] = "oai-nr-ue" if i == 0 else f"oai-nr-ue{i+1}"
    return names


UE_NAMES = build_ue_names()


# =============================================================================
# Scenario table
# =============================================================================

SCENARIOS = {
    "A": {"n_gnb": 1,  "label": "1 gNB  × 4 UEs = 4 UEs  (baseline)"},
    "B": {"n_gnb": 2,  "label": "2 gNBs × 4 UEs = 8 UEs"},
    "C": {"n_gnb": 4,  "label": "4 gNBs × 4 UEs = 16 UEs"},
    "D": {"n_gnb": 8,  "label": "8 gNBs × 4 UEs = 32 UEs"},
    "E": {"n_gnb": 15, "label": "15 gNBs × 4 UEs = 60 UEs (full testbed)"},
}


def validate():
    assert N_GNB >= 1
    assert UES_PER_GNB == 4, "UES_PER_GNB must be 4 (proven OAI rfsim limit)"
    assert N_NODES <= N_VMS
    assert all(v < N_NODES for v in RU_TO_DU.values()), \
        f"DU node out of range: {RU_TO_DU}"
    assert all(v < N_NODES for v in RU_TO_CU.values()), \
        f"CU node out of range: {RU_TO_CU}"

    print(f"Multi-gNB topology — Scenario {'?' if N_GNB not in [s['n_gnb'] for s in SCENARIOS.values()] else next(k for k,v in SCENARIOS.items() if v['n_gnb']==N_GNB)}:")
    print(f"  N_GNB    = {N_GNB}  (each with {UES_PER_GNB} UEs)")
    print(f"  N_UES    = {N_UES}")
    print(f"  N_DUS    = {N_DUS}   N_CUS = {N_CUS}   N_NODES = {N_NODES}")
    print(f"  RU→DU    = {RU_TO_DU}")
    print(f"  RU→CU    = {RU_TO_CU}")
    obs_dim = N_NODES+N_RUS*N_NODES+N_RUS*N_NODES+N_NODES+N_NODES*N_NODES
    print(f"  OBS_DIM  = {obs_dim}")
    print(f"  Per-agent act dim = 1 + 2×{N_RUS} = {1+2*N_RUS}")
    print(f"\n  Available scenarios:")
    for k, v in SCENARIOS.items():
        marker = " ← current" if v["n_gnb"] == N_GNB else ""
        print(f"    {k}: {v['label']}{marker}")


if __name__ == "__main__":
    validate()