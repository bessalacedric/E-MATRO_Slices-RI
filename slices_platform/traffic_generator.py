"""
Traffic Generator
=================
Drives real UL traffic through OAI UE containers via iperf3.

For each UE container:
  - Runs iperf3 client toward oai-ext-dn (UPF side)
  - Bitrate is set proportionally to the desired load level
  - Results are parsed and returned as real UL measurements

Mapping:
    UE containers oai-nr-ue1 … oai-nr-ue18
    ↓
    Grouped 3 per RU:
        RU 0 → ue1, ue2,  ue3
        RU 1 → ue4, ue5,  ue6
        ...
        RU 5 → ue16, ue17, ue18
"""

import subprocess
import json
import threading
import numpy as np
import logging
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

BASTION       = "proxy@bastion2.slices-be.eu"
IPERF_SERVER  = "192.168.70.135"    # oai-ext-dn IP from docker-compose
IPERF_PORT    = 5201  # base port; each UE uses 5201+ue_idx
IPERF_DURATION= 55                  # seconds per measurement (< 60s step)
UE_INTERFACE  = "oaitun_ue1"        # tunnel interface inside UE container


def _ue_name(ue_idx: int) -> str:
    """Container name for UE index (0-based)."""
    if ue_idx == 0:   return "oai-nr-ue"
    if ue_idx == 1:   return "oai-nr-ue2"
    return f"oai-nr-ue{ue_idx + 1}"


def _ru_to_ues(ru_idx: int, ues_per_ru: int = 3) -> List[int]:
    """Return 0-based UE indices for a given RU."""
    start = ru_idx * ues_per_ru
    return list(range(start, start + ues_per_ru))


class TrafficGenerator:
    """
    Orchestrates iperf3 traffic across all UE containers.

    Parameters
    ----------
    controller_ip : IP of vm-controller where UE containers run
    ssh_key       : path to SSH key
    n_rus         : number of RUs
    ues_per_ru    : UEs per RU
    dry_run       : return synthetic values without SSH
    """

    def __init__(
        self,
        controller_ip: str,
        ssh_key:       str  = "~/.ssh/id_ed25519",
        n_rus:         int  = 6,
        ues_per_ru:    int  = 3,
        dry_run:       bool = False,
    ):
        self.controller_ip = controller_ip
        self.ssh_key       = ssh_key
        self.n_rus         = n_rus
        self.ues_per_ru    = ues_per_ru
        self.dry_run       = dry_run
        self.n_ues         = n_rus * ues_per_ru

        # Last measured UL bitrate per UE (kbps)
        self._last_rate: Dict[int, float] = {i: 0.0 for i in range(self.n_ues)}

    def _docker_exec(self, container: str, cmd: str,
                     timeout: int = 70) -> Optional[str]:
        """Run cmd inside a container on vm-controller via SSH."""
        ssh_cmd = (f"ssh -i {self.ssh_key} "
                   f"-o StrictHostKeyChecking=no "
                   f"-o ConnectTimeout=10 "
                   f"-o BatchMode=yes "
                   f"-J {BASTION} "
                   f"ubuntu@{self.controller_ip} "
                   f"\"docker exec {container} {cmd}\"")
        if self.dry_run:
            return json.dumps({"end": {"sum_sent": {"bits_per_second": 
                               np.random.uniform(1e6, 20e6)}}})
        try:
            r = subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True, timeout=timeout)
            return r.stdout.strip() if r.returncode == 0 else None
        except Exception as e:
            log.warning(f"docker exec {container}: {e}")
            return None

    def _run_iperf_ue(self, ue_idx: int,
                      target_mbps: float = 10.0) -> float:
        """
        Run one iperf3 UDP uplink burst from a UE container.
        Returns measured UL bitrate in kbps.
        """
        name    = _ue_name(ue_idx)
        bitrate = f"{target_mbps:.1f}M"

        # iperf3 UDP client: UL (client→server), JSON output
        cmd = (f"iperf3 -c {IPERF_SERVER} -p {IPERF_PORT} " f"-u -b {bitrate} -t {IPERF_DURATION} " f"-B {UE_INTERFACE} " f"-J 2>/dev/null")

        out = self._docker_exec(name, cmd, timeout=IPERF_DURATION + 10)
        if out is None:
            return self._last_rate[ue_idx]

        try:
            data    = json.loads(out)
            bps     = data["end"]["sum_sent"]["bits_per_second"]
            kbps    = bps / 1e3
            self._last_rate[ue_idx] = kbps
            return kbps
        except (json.JSONDecodeError, KeyError):
            log.warning(f"iperf3 parse error for {name}")
            return self._last_rate[ue_idx]

    def run_traffic_step(
        self,
        target_mbps_per_ue: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Run one 55-second traffic measurement step across all UEs in parallel.

        Parameters
        ----------
        target_mbps_per_ue : (n_ues,) target UL rate per UE in Mbps.
                              If None, uses 10 Mbps for all UEs.

        Returns
        -------
        ru_traffic : (n_rus,) total UL bitrate per RU in Mbps
        """
        if target_mbps_per_ue is None:
            target_mbps_per_ue = np.full(self.n_ues, 10.0, dtype=np.float32)

        results: Dict[int, float] = {}
        lock = threading.Lock()

        def _measure(ue_idx):
            rate = self._run_iperf_ue(ue_idx, target_mbps_per_ue[ue_idx])
            with lock:
                results[ue_idx] = rate

        threads = [threading.Thread(target=_measure, args=(i,))
                   for i in range(self.n_ues)]
        for t in threads: t.start()
        for t in threads: t.join()

        # Aggregate per RU (sum UEs in the same RU group)
        ru_traffic = np.zeros(self.n_rus, dtype=np.float32)
        for ru in range(self.n_rus):
            ues = _ru_to_ues(ru, self.ues_per_ru)
            ru_traffic[ru] = sum(results.get(u, 0.0) for u in ues) / 1e3  # kbps→Mbps

        log.info(f"RU traffic (Mbps): {np.round(ru_traffic, 2).tolist()}")
        return ru_traffic

    def get_last_ru_traffic(self) -> np.ndarray:
        """Return last measured RU traffic without running new iperf."""
        ru_traffic = np.zeros(self.n_rus, dtype=np.float32)
        for ru in range(self.n_rus):
            ues = _ru_to_ues(ru, self.ues_per_ru)
            ru_traffic[ru] = sum(
                self._last_rate.get(u, 0.0) for u in ues
            ) / 1e3
        return ru_traffic

    def stop_all_traffic(self):
        """Kill any running iperf3 clients in all UE containers."""
        for i in range(self.n_ues):
            name = _ue_name(i)
            self._docker_exec(name, "pkill iperf3 2>/dev/null; true", timeout=5)