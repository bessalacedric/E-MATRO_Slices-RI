"""
Traffic Collector
=================
Reads real UL bitrate measurements from the live OAI 5G network.

Two collection methods (use whichever is available):

Method A — iperf3 results (primary):
    Reads JSON output from iperf3 runs already managed by TrafficGenerator.
    Most accurate — direct measurement.

Method B — OAI UPF GTP counters (fallback):
    Reads per-TEID byte counters from the UPF via docker exec.
    Less accurate (coarser granularity) but works without iperf3.

Method C — AMF log parsing (last resort):
    Parses AMF log for UE registration and last seen data rates.
    Only available if OAI AMF exposes rate info in logs.
"""

import subprocess
import re
import time
import numpy as np
import logging
from typing import Dict, Optional

log = logging.getLogger(__name__)

BASTION = "proxy@bastion2.slices-be.eu"


class TrafficCollector:
    """
    Collects per-RU UL bitrate from the live OAI 5G deployment.

    Parameters
    ----------
    controller_ip : vm-controller IP
    ssh_key       : SSH key path
    n_rus         : number of RUs
    ues_per_ru    : UEs per RU
    dry_run       : return synthetic values
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
        self._prev_bytes:  Dict[str, int] = {}
        self._prev_time:   float = time.time()

    def _ssh(self, cmd: str, timeout: int = 10) -> Optional[str]:
        full = (f"ssh -i {self.ssh_key} "
                f"-o StrictHostKeyChecking=no "
                f"-o ConnectTimeout={timeout} "
                f"-o BatchMode=yes "
                f"-J {BASTION} "
                f"ubuntu@{self.controller_ip} '{cmd}'")
        if self.dry_run:
            return None
        try:
            r = subprocess.run(full, shell=True, capture_output=True, text=True, timeout=timeout + 2)
            return r.stdout.strip() if r.returncode == 0 else None
        except Exception:
            return None

    # ── Method A: read from TrafficGenerator results directly ────────────────

    def from_traffic_generator(self, generator) -> np.ndarray:
        """
        Get RU traffic from a TrafficGenerator instance.
        This is the primary method — no SSH needed.
        """
        return generator.get_last_ru_traffic()

    # ── Method B: UPF GTP byte counter diff ──────────────────────────────────

    def from_upf_counters(self) -> Optional[np.ndarray]:
        """
        Read per-TEID byte counters from OAI UPF and compute bitrate.
        Returns (n_rus,) array in Mbps, or None on failure.
        """
        # OAI UPF exposes GTP-U stats via the metrics endpoint
        out = self._ssh(
            "docker exec oai-upf "
            "cat /proc/net/dev 2>/dev/null | grep -E 'tun|gtp' || echo ''")
        if not out:
            return None

        now   = time.time()
        dt    = now - self._prev_time
        rates = {}

        for line in out.splitlines():
            parts = line.split()
            if len(parts) < 10:
                continue
            iface = parts[0].rstrip(":")
            try:
                rx_bytes = int(parts[1])   # received = UL from UE perspective
                prev     = self._prev_bytes.get(iface, rx_bytes)
                delta    = max(0, rx_bytes - prev)
                rates[iface] = (delta * 8) / (dt * 1e6) if dt > 0 else 0.0
                self._prev_bytes[iface] = rx_bytes
            except (ValueError, IndexError):
                continue

        self._prev_time = now

        if not rates:
            return None

        # Map interfaces to RUs (round-robin if count differs)
        ifaces     = sorted(rates.keys())
        ru_traffic = np.zeros(self.n_rus, dtype=np.float32)
        for i, iface in enumerate(ifaces):
            ru_idx = i % self.n_rus
            ru_traffic[ru_idx] += rates[iface]

        return ru_traffic

    # ── Method C: AMF log parsing ─────────────────────────────────────────────

    def from_amf_logs(self) -> Optional[np.ndarray]:
        """
        Parse AMF logs for registered UEs.
        Returns uniform traffic estimate if UEs are registered, None otherwise.
        """
        out = self._ssh(
            "docker logs oai-amf 2>&1 | grep '5GMM-REGISTERED' | "
            "awk '{print $NF}' | sort -u | wc -l")
        if not out:
            return None

        try:
            n_registered = int(out.strip())
        except ValueError:
            return None

        if n_registered == 0:
            return np.zeros(self.n_rus, dtype=np.float32)

        # Estimate: registered UEs → assume 10 Mbps average each
        ues_per_ru   = self.ues_per_ru
        ru_traffic   = np.full(self.n_rus, 10.0 * ues_per_ru, dtype=np.float32)
        log.info(f"AMF: {n_registered} UEs registered → "
                 f"estimated {10.0 * ues_per_ru:.1f} Mbps/RU")
        return ru_traffic

    # ── Primary collection method ─────────────────────────────────────────────

    def collect(self, generator=None) -> np.ndarray:
        """
        Collect real UL bitrate per RU using best available method.

        Priority:
            1. TrafficGenerator results (if provided)
            2. UPF GTP counters
            3. AMF log estimate
            4. Zeros (fallback)
        """
        if self.dry_run:
            # Synthetic diurnal pattern for dry-run testing
            import datetime
            hour    = datetime.datetime.now().hour
            base    = 5.0 + 30.0 * np.exp(-((hour - 14) ** 2) / 18.0)
            traffic = np.abs(np.random.normal(base, 3.0, size=self.n_rus))
            return traffic.astype(np.float32)

        if generator is not None:
            return self.from_traffic_generator(generator)

        result = self.from_upf_counters()
        if result is not None:
            return result

        result = self.from_amf_logs()
        if result is not None:
            return result

        log.warning("All collection methods failed — returning zeros")
        return np.zeros(self.n_rus, dtype=np.float32)

    # ── UE registration check ─────────────────────────────────────────────────

    def check_ues_registered(self, expected: int) -> bool:
        """Return True if at least `expected` UEs are registered."""
        out = self._ssh(
            "docker logs oai-amf 2>&1 | grep -c '5GMM-REGISTERED' || echo 0")
        if out is None:
            return False
        try:
            return int(out.strip()) >= expected
        except ValueError:
            return False

    def get_registered_ue_count(self) -> int:
        """Return number of currently registered UEs."""
        out = self._ssh(
            "docker logs oai-amf 2>&1 | grep '5GMM-REGISTERED' | "
            "awk '{print $NF}' | sort -u | wc -l")
        try:
            return int(out.strip()) if out else 0
        except ValueError:
            return 0