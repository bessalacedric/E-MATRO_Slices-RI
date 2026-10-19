"""
PlacementExecutor
=================
Translates x_ri_DU / x_ri_CU decisions into real stress-ng processes
on SLICES-RI VMs via SSH.

VNF naming: e-matro_du_ru{r}  /  e-matro_cu_ru{r}
CPU load:   alpha * L_r / NODE_CAPACITY_GOPS  (clamped 1-95%)
"""

import subprocess
import threading
import numpy as np
import logging
from typing import Dict, Set

log = logging.getLogger(__name__)

ALPHA_DU           = 0.6
ALPHA_CU           = 0.3
NODE_CAPACITY_GOPS = 10.0
BASTION            = "proxy@bastion2.slices-be.eu"
SSH_TIMEOUT        = 8


class PlacementExecutor:
    def __init__(self, vm_ips: Dict[int, str], ssh_key: str = "~/.ssh/id_ed25519", dry_run: bool = False):
        self.vm_ips  = vm_ips
        self.ssh_key = ssh_key
        self.dry_run = dry_run
        self.current_du: Dict[int, Set[int]] = {i: set() for i in vm_ips}
        self.current_cu: Dict[int, Set[int]] = {i: set() for i in vm_ips}
        self.node_power: Dict[int, int]      = {i: 2 for i in vm_ips}
        self.migration_count = 0

    def _ssh(self, node: int, cmd: str, background=False) -> bool:
        ip   = self.vm_ips[node]
        tail = " &" if background else ""
        full = (f"ssh -i {self.ssh_key} " f"-o StrictHostKeyChecking=no " f"-o ConnectTimeout={SSH_TIMEOUT} "
                f"-o BatchMode=yes " f"-J {BASTION} " f"ubuntu@{ip} '{cmd}{tail}'")
        if self.dry_run:
            log.info(f"[DRY] node={node}: {cmd}")
            return True
        try:
            r = subprocess.run(full, shell=True, capture_output=True, text=True, timeout=SSH_TIMEOUT + 2)
            return r.returncode == 0
        except Exception as e:
            log.warning(f"SSH node={node}: {e}")
            return False

    def _ssh_parallel(self, tasks):
        threads = [threading.Thread(target=self._ssh, args=t) for t in tasks]
        for t in threads: t.start()
        for t in threads: t.join()

    def _cpu_pct(self, kind, load_mbps):
        alpha = ALPHA_DU if kind == "du" else ALPHA_CU
        return int(np.clip(alpha * load_mbps / NODE_CAPACITY_GOPS * 100, 1, 95))

    def _spawn_cmd(self, kind, ru, cpu_pct):
        name = f"e-matro_{kind}_ru{ru}"
        return (f"pkill -f '{name}' 2>/dev/null; " f"nohup stress-ng --cpu 1 --cpu-load {cpu_pct} " f"--cpu-method matrixprod --taskname {name} "f"--timeout 0 > /tmp/{name}.log 2>&1")

    def _kill_cmd(self, kind, ru):
        return f"pkill -f 'e-matro_{kind}_ru{ru}' 2>/dev/null; true"

    def apply(self, x_ri_DU, x_ri_CU, traffic):
        tasks, n_mig = [], 0
        for node in self.vm_ips:
            want_du = set(np.where(x_ri_DU[:, node] > 0.5)[0])
            want_cu = set(np.where(x_ri_CU[:, node] > 0.5)[0])

            for ru in self.current_du[node] - want_du:
                tasks.append((node, self._kill_cmd("du", ru), False)); n_mig += 1
            for ru in want_du - self.current_du[node]:
                pct = self._cpu_pct("du", float(traffic[ru]))
                tasks.append((node, self._spawn_cmd("du", ru, pct), True)); n_mig += 1
            for ru in want_du & self.current_du[node]:
                pct = self._cpu_pct("du", float(traffic[ru]))
                tasks.append((node, self._spawn_cmd("du", ru, pct), True))

            for ru in self.current_cu[node] - want_cu:
                tasks.append((node, self._kill_cmd("cu", ru), False)); n_mig += 1
            for ru in want_cu - self.current_cu[node]:
                pct = self._cpu_pct("cu", float(traffic[ru]))
                tasks.append((node, self._spawn_cmd("cu", ru, pct), True)); n_mig += 1
            for ru in want_cu & self.current_cu[node]:
                pct = self._cpu_pct("cu", float(traffic[ru]))
                tasks.append((node, self._spawn_cmd("cu", ru, pct), True))

            self.current_du[node] = want_du
            self.current_cu[node] = want_cu

        if tasks:
            self._ssh_parallel(tasks)
        self.migration_count += n_mig
        return n_mig

    def set_node_power(self, node, state):
        if state == self.node_power[node]: return
        if state == 0:
            self._ssh(node, "pkill stress-ng 2>/dev/null; true")
        elif state == 1:
            self._ssh(node, "pkill stress-ng 2>/dev/null; ""nohup stress-ng --cpu 1 --cpu-load 2 --timeout 0 " "> /tmp/idle.log 2>&1", background=True)
        elif state == 2:
            self._ssh(node, "pkill stress-ng 2>/dev/null; true")
        self.node_power[node] = state

    def cleanup_all(self):
        tasks = [(n, "pkill stress-ng 2>/dev/null; true", False) for n in self.vm_ips]
        self._ssh_parallel(tasks)
        for n in self.vm_ips:
            self.current_du[n] = set()
            self.current_cu[n] = set()

    def reset_counters(self):
        self.migration_count = 0
