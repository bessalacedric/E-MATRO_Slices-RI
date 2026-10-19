"""
MetricCollector
===============
Polls real CPU/RAM from SLICES-RI VMs via SSH + psutil.
Feeds real measurements back to OCloudEnv when real_platform=True.

Energy estimate:
    E_i^C = eps_C_i * (cpu_util_i * NODE_CAPACITY_GOPS) + eps_Base_i
"""

import subprocess
import threading
import numpy as np
import logging
from typing import Dict, Optional, Tuple

log = logging.getLogger(__name__)

NODE_CAPACITY_GOPS = 10.0
BASTION            = "proxy@bastion2.slices-be.eu"
SSH_TIMEOUT        = 5


class MetricCollector:
    def __init__(self, vm_ips: Dict[int, str], ssh_key: str = "~/.ssh/id_ed25519", dry_run: bool = False):

        self.vm_ips  = vm_ips
        self.ssh_key = ssh_key
        self.dry_run = dry_run
        self._last_cpu: Dict[int, float] = {i: 0.0 for i in vm_ips}
        self._last_mem: Dict[int, float] = {i: 0.0 for i in vm_ips}

    def _ssh_read(self, node: int, cmd: str) -> Optional[str]:
        if self.dry_run:
            return f"{np.random.uniform(10, 80):.1f},{np.random.uniform(20, 60):.1f}"
        ip   = self.vm_ips[node]
        full = (f"ssh -i {self.ssh_key} "f"-o StrictHostKeyChecking=no "
        f"-o ConnectTimeout={SSH_TIMEOUT} " f"-o BatchMode=yes " f"-J {BASTION} " f"ubuntu@{ip} '{cmd}'")

        try:
            r = subprocess.run(full, shell=True, capture_output=True, text=True, timeout=SSH_TIMEOUT + 2)

            return r.stdout.strip() if r.returncode == 0 else None
        except Exception as e:
            log.warning(f"MetricCollector node={node}: {e}")
            return None

    def get_node_metrics(self, node: int) -> Tuple[float, float]:
        cmd = ("python3 -c \""
               "import psutil; "
               "cpu=psutil.cpu_percent(interval=0.5); "
               "mem=psutil.virtual_memory().percent; "
               "print(f'{cpu},{mem}')\"")
        out = self._ssh_read(node, cmd)
        if out is None:
            return self._last_cpu[node], self._last_mem[node]
        try:
            cpu, mem = (float(x) / 100.0 for x in out.split(","))
            self._last_cpu[node], self._last_mem[node] = cpu, mem
            return cpu, mem
        except Exception:
            return self._last_cpu[node], self._last_mem[node]

    def poll_all(self) -> Tuple[np.ndarray, np.ndarray]:
        results: Dict[int, Tuple[float, float]] = {}
        lock = threading.Lock()

        def _poll(idx):
            cpu, mem = self.get_node_metrics(idx)
            with lock: results[idx] = (cpu, mem)

        threads = [threading.Thread(target=_poll, args=(i,)) for i in self.vm_ips]
        for t in threads: t.start()
        for t in threads: t.join()

        cpu_arr = np.array([results[i][0] for i in sorted(self.vm_ips)], dtype=np.float32)
        mem_arr = np.array([results[i][1] for i in sorted(self.vm_ips)], dtype=np.float32)
        return cpu_arr, mem_arr

    def estimate_energy(self, cpu_utils, node_states, eps_c, eps_base) -> Tuple[float, np.ndarray]:

        per_node = np.zeros(len(node_states), dtype=np.float32)
        for i, (u, s) in enumerate(zip(cpu_utils, node_states)):
            if s == 0:
                per_node[i] = 0.0
            elif s == 1:
                per_node[i] = float(eps_base[i] * 0.1)
            else:
                W = u * NODE_CAPACITY_GOPS
                per_node[i] = float(eps_c[i] * W + eps_base[i])
        return float(per_node.sum()), per_node