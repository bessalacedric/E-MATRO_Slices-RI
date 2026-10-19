# load_collector.py — runs on vm-controller, polls all nodes

import subprocess, re, time
#import paramiko

NODES = {
    0: {"ip": "10.10.208.109", "role": "DU", "container": "oai-du-node0"},
    1: {"ip": "10.10.210.195", "role": "DU", "container": "oai-du-node1"},
    2: {"ip": "10.10.211.225", "role": "CU", "container": "oai-cu-node2"},
    3: {"ip": "10.10.211.11",  "role": "CU", "container": "oai-cu-node3"},
}

NODE_CAPACITY_CORES = 4   # vCPUs per vm-server
POLL_INTERVAL = 1.0       # seconds



def get_cpu_load(node_id: int) -> float:
    """
    Returns normalized CPU load ∈ [0.0, 1.0] for a node.
    Uses docker stats for the OAI container on that vm-server.
    """
    node = NODES[node_id]
    cmd = (
        f"ssh -o StrictHostKeyChecking=no ubuntu@{node['ip']} "
        f"\"docker stats --no-stream --format '{{{{.CPUPerc}}}}' "
        f"{node['container']} 2>/dev/null\""
    )
    try:
        out = subprocess.check_output(cmd, shell=True, timeout=3).decode().strip()
        # output like "12.34%"
        pct = float(out.replace("%", ""))
        return min(pct / 100.0, 1.0)
    except Exception:
        return 0.0

def get_prb_utilization(node_id: int) -> float:
    """
    Returns PRB utilization ∈ [0.0, 1.0] from DU logs.
    Only meaningful for DU nodes.
    """
    node = NODES[node_id]
    if node["role"] != "DU":
        return get_cpu_load(node_id)   # fallback for CU

    cmd = (
        f"ssh -o StrictHostKeyChecking=no ubuntu@{node['ip']} "
        f"\"docker logs {node['container']} 2>&1 | "
        f"grep -oP 'NPRB \\K[0-9]+' | tail -4\""
    )
    try:
        out = subprocess.check_output(cmd, shell=True, timeout=3).decode().strip()
        prbs = [int(x) for x in out.split() if x.isdigit()]
        if prbs:
            MAX_PRB = 106   # for 40 MHz bandwidth, numerology 1
            return min(sum(prbs) / (len(prbs) * MAX_PRB), 1.0)
    except Exception:
        pass
    return get_cpu_load(node_id)   # fallback

def get_all_loads() -> dict[int, float]:
    """Poll all 4 nodes simultaneously."""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as ex:
        results = {
            nid: ex.submit(get_prb_utilization, nid)
            for nid in NODES
        }
    return {nid: f.result() for nid, f in results.items()}


if __name__ == "__main__":
    while True:
        loads = get_all_loads()
        print(" | ".join(
            f"node{i}({NODES[i]['role']}): {v:.2f}"
            for i, v in loads.items()
        ))
        time.sleep(POLL_INTERVAL)