"""
topology.py — E-MATRO Deployment Topology Configuration
========================================================
Single source of truth for all deployment parameters.
Change N_RUS here to scale the entire system.

Current physical constraints (SLICES-RI):
  - 4 vm-servers (2 DU-VMs + 2 CU-VMs)
  - 4 vCPUs per VM
  - ~7.7 GB RAM per VM (OOM risk above 2 DUs/VM)
  - 12 UEs provisioned in DB

Scaling limits:
  N_RUS=4  → 2 DUs × 2 CUs  → safe (current)
  N_RUS=8  → 4 DUs × 2 CUs  → 2 DUs/DU-VM (monitor RAM)
  N_RUS=12 → 4 DUs × 4 CUs  → 3 DUs/DU-VM (OOM risk!)
"""

# =============================================================================
# Primary scaling parameter — we change only this to scale
# =============================================================================

N_RUS = 8   # number of RUs = number of UEs (1 UE per RU)
                   # supported: 4, 6, 8  (12 is risky for OOM)

# =============================================================================
# Derived topology 
# =============================================================================

# UEs = RUs (1 UE per RU, 1 gNB per RU)
N_UES = N_RUS

# Physical VM constraint — never exceed available VMs
N_VMS = 15  # vm-server-1..n on SLICES-RI


# DU/CU counts — scale with RUs, bounded by VM capacity
# Rule: 1 DU per 2 RUs, 1 CU per 4 RUs (CUs serve more RUs)
N_DUS = max(2, N_RUS // 2)       # 4RU→2DU, 6RU→3DU, 8RU→4DU
N_CUS = max(2, (N_RUS + 3) // 4) # 4RU→2CU, 8RU→2CU, 12RU→3CU

# Total O-Cloud nodes = DU nodes + CU nodes (dynamic) or give a hard limit
N_NODES = 6#min(N_DUS + N_CUS, N_VMS)   # N

# RUs per DU (balanced distribution)
RUS_PER_DU = N_RUS // N_DUS      # 4RU/2DU=2, 8RU/4DU=2

# Physical VM assignment
N_CORES    = 4     # vCPUs per vm-server
N_PRB_MAX  = 106   # max PRBs (40 MHz, numerology 1)

# =============================================================================
# Physical VM → node mapping
# =============================================================================
# vm-servers are assigned roles based on node count:
# Nodes 0..N_DUS-1      → DU nodes (vm-server-1, vm-server-2, ...)
# Nodes N_DUS..N_NODES-1 → CU nodes (remaining vm-servers)

# All vm-server IPs (in order: will be assigned to nodes 0,1,2,3,...) or adapted to your own reserved IPs
CTL_IP   = "10.10.213.129"   # was 10.10.208.188
SSH_USER = "ubuntu"

VM_IPS = [
    "10.10.212.103",   # vm-server-1
    "10.10.214.110",   # vm-server-2
    "10.10.212.33",    # vm-server-3
    "10.10.212.196",   # vm-server-4
    "10.10.209.122",   # vm-server-5
    "10.10.210.7",     # vm-server-6
    "10.10.214.49",    # vm-server-7
    "10.10.212.215",   # vm-server-8
    "10.10.209.93",    # vm-server-9
    "10.10.210.241",   # vm-server-10
    "10.10.210.203",   # vm-server-11
    "10.10.215.22",    # vm-server-12
    "10.10.211.154",   # vm-server-13
    "10.10.213.26",    # vm-server-14
    "10.10.209.253",   # vm-server-15
]


# =============================================================================
# Node configuration (auto-generated from topology)
# =============================================================================

def build_node_configs() -> dict:
    """
    Build NodeConfig dict dynamically from topology.
    Returns {node_id: NodeConfig} for all N_NODES nodes.
    """
    from dataclasses import dataclass

    @dataclass
    class NodeConfig:
        node_id:   int
        role:      str      # "DU" or "CU"
        ip:        str
        container: str
        vm_index:  int      # which physical VM (0-based)
        n_cores:   int = N_CORES

    configs = {}

    # DU nodes: nodes 0 .. N_DUS-1
    for i in range(N_DUS):
        vm_idx = i % len(VM_IPS)   # distribute across VMs
        configs[i] = NodeConfig(
            node_id   = i,
            role      = "DU",
            ip        = VM_IPS[vm_idx],
            container = f"oai-du-node{i}",
            vm_index  = vm_idx,
        )

    # CU nodes: nodes N_DUS .. N_NODES-1
    for j in range(N_CUS):
        node_id = N_DUS + j
        vm_idx  = (N_DUS + j) % len(VM_IPS)
        configs[node_id] = NodeConfig(
            node_id   = node_id,
            role      = "CU",
            ip        = VM_IPS[vm_idx],
            container = f"oai-cu-node{node_id}",
            vm_index  = vm_idx,
        )

    return configs


# Build at import time
NODE_CONFIGS = build_node_configs()


# =============================================================================
# RU → DU/CU mapping (auto-generated)
# =============================================================================

def build_ru_mapping() -> tuple[dict, dict]:
    """
    Build RU → DU and RU → CU assignments.
    Returns (ru_to_du, ru_to_cu):
      ru_to_du[r] = node_id of DU serving RU r  ∈ [0, N_NODES)
      ru_to_cu[r] = node_id of CU serving RU r  ∈ [0, N_NODES)

    Node IDs are always < N_NODES (capped at N_VMS=4).
    DUs occupy the first N_DUS logical slots (wrapped to N_NODES).
    CUs occupy the remaining slots (also wrapped to N_NODES).
    """
    ru_to_du = {}
    ru_to_cu = {}

    for r in range(N_RUS):
        du_idx  = r // RUS_PER_DU               # logical DU index 0..N_DUS-1
        cu_idx  = du_idx % N_CUS                # logical CU index 0..N_CUS-1

        # Map logical index → physical node id (wrapped to N_NODES)
        du_node = du_idx % N_NODES
        cu_node = (N_DUS + cu_idx) % N_NODES    # CUs after DUs, wrapped

        ru_to_du[r] = du_node
        ru_to_cu[r] = cu_node

    return ru_to_du, ru_to_cu


RU_TO_DU, RU_TO_CU = build_ru_mapping()

# Initial placement: VNF → node (used by OCloudState)
INITIAL_PLACEMENT_DU = dict(RU_TO_DU)   # ru_id → du_node_id
INITIAL_PLACEMENT_CU = dict(RU_TO_CU)   # ru_id → cu_node_id


# =============================================================================
# UE container names (auto-generated)
# =============================================================================

def build_ue_names() -> dict:
    """Returns {ue_idx (0-based): container_name}"""
    names = {}
    for i in range(N_UES):
        if i == 0:
            names[i] = "oai-nr-ue"
        else:
            names[i] = f"oai-nr-ue{i+1}"
    return names


UE_NAMES = build_ue_names()


# =============================================================================
# Energy model
# =============================================================================

P_SLEEP    = 10.0
P_IDLE     = 100.0
P_MAX      = 400.0
P_BASE     = N_NODES * P_IDLE
CAPACITY   = 0.85






# =============================================================================
# Validation
# =============================================================================

def validate_topology():
    """Assert topology is internally consistent."""
    assert N_RUS >= 2,           "Need at least 2 RUs"
    assert N_DUS >= 1,           "Need at least 1 DU"
    assert N_CUS >= 1,           "Need at least 1 CU"
    assert N_NODES <= len(VM_IPS), \
        f"N_NODES={N_NODES} exceeds available VMs={len(VM_IPS)}"
    assert N_RUS % N_DUS == 0,   "N_RUS must be divisible by N_DUS"
    assert N_UES == N_RUS,       "1 UE per RU required"

    # Check load feasibility at full traffic
    # Each DU handles RUS_PER_DU UEs at max NPRB=106 → load=1.0 per DU
    # Consolidation: max 2 DUs on one node → load=2.0 (violates capacity)
    # So agent must keep DUs on separate nodes when all are loaded
    max_du_load_consolidated = RUS_PER_DU * 1.0
    print(f"Topology validation:")
    print(f"  N_RUS={N_RUS}  N_DUS={N_DUS}  N_CUS={N_CUS}  N_NODES={N_NODES}")
    print(f"  RUs/DU={RUS_PER_DU}")
    print(f"  Max load if 2 DUs consolidated: {2*max_du_load_consolidated/N_DUS:.2f}")
    print(f"  Safe to consolidate DUs when load < {CAPACITY/2:.2f} each")
    print(f"  RU→DU: {RU_TO_DU}")
    print(f"  RU→CU: {RU_TO_CU}")
    print(f"  Nodes: { {nid: f'{c.role}@{c.ip}' for nid,c in NODE_CONFIGS.items()} }")
    print(f"  UEs: {UE_NAMES}")
    return True


if __name__ == "__main__":
    validate_topology()
