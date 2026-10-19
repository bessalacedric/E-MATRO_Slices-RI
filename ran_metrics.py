"""
ran_metrics.py — RAN Metrics Collector for E-MATRO
====================================================
Collects per-RU RAN metrics from OAI DU container logs during training.

Metrics collected per RU per step:
  - MCS (uplink/downlink)
  - SNR (dB)
  - BLER (uplink/downlink)
  - NPRB (Physical Resource Blocks used)
  - Throughput estimated (Mbps)
  - HARQ rounds distribution
  - UE RNTI

Stored in: results/e-matro_ran_metrics.csv
"""

import re
import csv
import os
import subprocess
import logging
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

SSH_OPTS = "-o StrictHostKeyChecking=no -o ConnectTimeout=3"

# Mbps per PRB at given MCS (rough estimate, numerology 1, 30kHz SCS)
# Based on: PRB = 12 subcarriers × 14 symbols × bits_per_symbol × coding_rate
MBPS_PER_PRB = {
    0:  0.02,   # QPSK 1/8
    5:  0.07,   # QPSK 1/2
    9:  0.18,   # QPSK ~3/4
    14: 0.35,   # 16QAM 1/2
    19: 0.55,   # 16QAM 3/4
    24: 0.85,   # 64QAM 3/4
    28: 1.10,   # 256QAM ~5/6
}

def _mcs_to_mbps(mcs: int, nprb: int) -> float:
    """Estimate throughput from MCS and NPRB."""
    # Find closest MCS entry
    closest = min(MBPS_PER_PRB.keys(), key=lambda k: abs(k - mcs))
    return round(MBPS_PER_PRB[closest] * nprb, 4)


# =============================================================================
# Per-UE metrics dataclass
# =============================================================================

@dataclass
class UEMetrics:
    """Metrics for one UE at one collection step."""

    # Identifiers
    episode:     int   = 0
    step:        int   = 0
    ru_id:       int   = 0    # RU index (0-3)
    ue_rnti:     str   = ""   # UE RNTI (hex)
    du_node:     int   = 0    # physical DU node (0 or 1)

    # Uplink metrics (most informative — UE → gNB direction)
    ul_mcs:      int   = 0    # MCS index 0-28 (higher = more bits/symbol)
    ul_nprb:     int   = 0    # PRBs allocated for UL
    ul_bler:     float = 0.0  # UL block error rate ∈ [0,1]
    ul_snr_db:   float = 0.0  # UL SNR in dB (RFSim: ~51dB ideal)
    ul_rounds_1: int   = 0    # HARQ 1st round successes
    ul_rounds_2: int   = 0    # HARQ 2nd round (1 retransmission)
    ul_rounds_3: int   = 0    # HARQ 3rd round (2 retransmissions)
    ul_rounds_4: int   = 0    # HARQ 4th round (3 retransmissions)
    ul_errors:   int   = 0    # UL packets failed all 4 rounds

    # Downlink metrics (gNB → UE direction)
    dl_mcs:      int   = 0    # DL MCS index
    dl_bler:     float = 0.0  # DL block error rate
    dl_rounds_1: int   = 0    # DL HARQ 1st round
    dl_rounds_2: int   = 0    # DL HARQ 2nd round
    dl_errors:   int   = 0    # DL packets failed

    # Derived metrics
    ul_tput_mbps:  float = 0.0  # estimated UL throughput
    dl_tput_mbps:  float = 0.0  # estimated DL throughput

    # Context (from env)
    node_load:     float = 0.0  # combined load on hosting node
    nodes_active:  int   = 4    # number of active nodes
    energy_saving: float = 0.0  # current energy saving ∈ [0,1]


# =============================================================================
# Log parser
# =============================================================================

# Patterns matching OAI DU log lines:
# UE 2540: ulsch_rounds 434266/3639/86/0, ulsch_errors 0, ulsch_DTX 0,
#          BLER 0.01174 MCS (0) 28 (Qm 6  dB) NPRB 5  SNR 51.0 dB
UL_PATTERN = re.compile(
    r"UE\s+(?P<rnti>[0-9a-f]+):\s+ulsch_rounds\s+"
    r"(?P<r1>\d+)/(?P<r2>\d+)/(?P<r3>\d+)/(?P<r4>\d+),\s+"
    r"ulsch_errors\s+(?P<errors>\d+).*?"
    r"BLER\s+(?P<bler>[0-9.]+)\s+MCS\s+\(\d+\)\s+(?P<mcs>\d+).*?"
    r"NPRB\s+(?P<nprb>\d+)\s+SNR\s+(?P<snr>[0-9.-]+)"
)

# UE 2540: dlsch_rounds 4422/0/0/0, dlsch_errors 0, pucch0_DTX 0,
#          BLER 0.00000 MCS (0) 9
DL_PATTERN = re.compile(
    r"UE\s+(?P<rnti>[0-9a-f]+):\s+dlsch_rounds\s+"
    r"(?P<r1>\d+)/(?P<r2>\d+)/(?P<r3>\d+)/(?P<r4>\d+),\s+"
    r"dlsch_errors\s+(?P<errors>\d+).*?"
    r"BLER\s+(?P<bler>[0-9.]+)\s+MCS\s+\(\d+\)\s+(?P<mcs>\d+)"
)


def parse_du_logs(log_text: str, n_ues: int = 2) -> list[dict]:
    """
    Parse OAI DU log output and extract per-UE metrics.
    Returns list of dicts with ul_* and dl_* fields per UE.
    """
    results = {}

    # Parse uplink
    for m in UL_PATTERN.finditer(log_text):
        rnti = m.group("rnti")
        if rnti not in results:
            results[rnti] = {"rnti": rnti}
        results[rnti].update({
            "ul_mcs":      int(m.group("mcs")),
            "ul_nprb":     int(m.group("nprb")),
            "ul_bler":     float(m.group("bler")),
            "ul_snr_db":   float(m.group("snr")),
            "ul_rounds_1": int(m.group("r1")),
            "ul_rounds_2": int(m.group("r2")),
            "ul_rounds_3": int(m.group("r3")),
            "ul_rounds_4": int(m.group("r4")),
            "ul_errors":   int(m.group("errors")),
        })

    # Parse downlink
    for m in DL_PATTERN.finditer(log_text):
        rnti = m.group("rnti")
        if rnti not in results:
            results[rnti] = {"rnti": rnti}
        results[rnti].update({
            "dl_mcs":      int(m.group("mcs")),
            "dl_bler":     float(m.group("bler")),
            "dl_rounds_1": int(m.group("r1")),
            "dl_rounds_2": int(m.group("r2")),
            "dl_errors":   int(m.group("errors")),
        })

    return list(results.values())


# =============================================================================
# RAN Metrics Collector
# =============================================================================

class RANMetricsCollector:
    """
    Collects RAN metrics from OAI DU nodes via SSH during training.
    Writes to a separate CSV file from the training log.

    DU0 (node0): serves RU0 (UE1) and RU1 (UE2)
    DU1 (node1): serves RU2 (UE3) and RU3 (UE4)
    """

    DU_NODES = {
        0: {"ip": "10.10.208.109", "container": "oai-du-node0", "rus": [0, 1]},
        1: {"ip": "10.10.210.195", "container": "oai-du-node1", "rus": [2, 3]},
    }

    def __init__(self, ssh_user: str = "ubuntu", output_path: str = "results/e-matro_ran_metrics.csv"):
        self.ssh_user    = ssh_user
        self.output_path = output_path
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        # Open file immediately so it exists even if no metrics collected
        self._file   = open(output_path, "w", newline="")
        self._writer = None   # writer created on first row (needs fieldnames)
        print(f"  RAN metrics file opened: {output_path}")

        # self._writer     = None
        # self._file       = None
        # self._headers_written = False
        # os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    def _ssh(self, ip: str, cmd: str) -> str:
        full = f'ssh {SSH_OPTS} {self.ssh_user}@{ip} "{cmd}"'
        return subprocess.check_output(
            full, shell=True, timeout=5, stderr=subprocess.DEVNULL
        ).decode()

    def _fetch_du_logs(self, du_id: int) -> str:
        """Fetch last N lines of DU container logs."""
        node = self.DU_NODES[du_id]
        try:
            # Get last 20 lines — enough for 2 UEs × (dl + ul) lines
            return self._ssh(
                node["ip"],
                f"docker logs {node['container']} 2>&1 | "
                f"grep -E 'ulsch_rounds|dlsch_rounds' | tail -8"
            )
        except Exception as e:
            logger.debug(f"DU{du_id} log fetch failed: {e}")
            return ""

    def collect(self, episode: int, step: int, env_info: dict) -> list[UEMetrics]:
        """
        Collect RAN metrics for all RUs at current training step.

        Args:
            episode:  current training episode
            step:     current step within episode
            env_info: info dict from env.step() — provides node context

        Returns:
            list of UEMetrics (one per RU/UE)
        """
        metrics = []
        node_loads    = env_info.get("node_loads", [0.0] * 4)
        nodes_active  = env_info.get("nodes_active", 4)
        energy_saving = env_info.get("energy_saving", 0.0)

        # Fetch logs in parallel
        with ThreadPoolExecutor(max_workers=2) as ex:
            futures = {ex.submit(self._fetch_du_logs, du_id): du_id
                       for du_id in self.DU_NODES}
            logs = {}
            for f in as_completed(futures, timeout=8):
                du_id = futures[f]
                try:
                    logs[du_id] = f.result()
                except Exception:
                    logs[du_id] = ""

        # Parse logs and build UEMetrics
        for du_id, node_cfg in self.DU_NODES.items():
            log_text = logs.get(du_id, "")
            ue_data  = parse_du_logs(log_text)

            for ru_offset, ue_dict in enumerate(ue_data[:2]):
                ru_id = node_cfg["rus"][ru_offset]

                m = UEMetrics(
                    episode      = episode,
                    step         = step,
                    ru_id        = ru_id,
                    ue_rnti      = ue_dict.get("rnti", ""),
                    du_node      = du_id,
                    node_load    = float(node_loads[du_id]) if du_id < len(node_loads) else 0.0,
                    nodes_active = nodes_active,
                    energy_saving= energy_saving,
                )

                # UL metrics
                m.ul_mcs      = ue_dict.get("ul_mcs", 0)
                m.ul_nprb     = ue_dict.get("ul_nprb", 0)
                m.ul_bler     = ue_dict.get("ul_bler", 0.0)
                m.ul_snr_db   = ue_dict.get("ul_snr_db", 0.0)
                m.ul_rounds_1 = ue_dict.get("ul_rounds_1", 0)
                m.ul_rounds_2 = ue_dict.get("ul_rounds_2", 0)
                m.ul_rounds_3 = ue_dict.get("ul_rounds_3", 0)
                m.ul_rounds_4 = ue_dict.get("ul_rounds_4", 0)
                m.ul_errors   = ue_dict.get("ul_errors", 0)

                # DL metrics
                m.dl_mcs      = ue_dict.get("dl_mcs", 0)
                m.dl_bler     = ue_dict.get("dl_bler", 0.0)
                m.dl_rounds_1 = ue_dict.get("dl_rounds_1", 0)
                m.dl_rounds_2 = ue_dict.get("dl_rounds_2", 0)
                m.dl_errors   = ue_dict.get("dl_errors", 0)

                # Derived throughput
                m.ul_tput_mbps = _mcs_to_mbps(m.ul_mcs, m.ul_nprb)
                m.dl_tput_mbps = _mcs_to_mbps(m.dl_mcs, m.ul_nprb)

                metrics.append(m)

                # If SSH failed and no metrics collected, write empty placeholder
                if not metrics:
                    for du_id, node_cfg in self.DU_NODES.items():
                        for ru_offset in range(2):
                            ru_id = node_cfg["rus"][ru_offset]
                            metrics.append(UEMetrics(
                                episode      = episode,
                                step         = step,
                                ru_id        = ru_id,
                                du_node      = du_id,
                                ue_rnti      = "unreachable",
                                node_load    = float(node_loads[du_id]) if du_id < len(node_loads) else 0.0,
                                nodes_active = nodes_active,
                                energy_saving= energy_saving,
                            ))

        return metrics

    def write(self, metrics: list[UEMetrics]):
        """Append metrics to CSV file."""
        if not metrics:
            return

        if self._file is None:
            self._file   = open(self.output_path, "w", newline="")
            self._writer = csv.DictWriter(
                self._file, fieldnames=list(asdict(metrics[0]).keys())
            )
            self._writer.writeheader()

        for m in metrics:
            self._writer.writerow(asdict(m))
        self._file.flush()

    def collect_and_write(self, episode: int, step: int, env_info: dict) -> list[UEMetrics]:
        """Convenience: collect + write in one call."""
        metrics = self.collect(episode, step, env_info)
        self.write(metrics)
        return metrics

    def close(self):
        if self._file:
            self._file.close()
            self._file = None


# =============================================================================
# Relevance reference
# =============================================================================

METRIC_RELEVANCE = """
RAN Metrics — Relevance for E-MATRO Learning and Evaluation
============================================================

1. NPRB (Physical Resource Blocks used)
   - PRIMARY load signal for DU nodes
   - Directly drives w_ri_DU load: L_DU = NPRB / N_PRB_MAX
   - Higher NPRB → agent should NOT consolidate (node would overload)
   - Lower NPRB → agent CAN migrate VNFs to free up a node

2. MCS (Modulation and Coding Scheme, 0-28)
   - Reflects link quality and scheduler aggressiveness
   - High MCS (≥20) → good channel, UE far from saturation
   - Low MCS (≤9)  → weak channel, scheduler already compensating
   - MCS drop after migration = QoS degradation indicator
   - Use as secondary reward signal: MCS drop → penalize migration

3. SNR/SINR (Signal-to-Noise Ratio, dB)
   - In RFSim: always ~51dB (ideal channel) — not useful for E-MATRO
   - On real RF: would drop if RU moves or interference increases
   - Useful as QoS guard: if SNR < threshold, block migration

4. BLER (Block Error Rate, 0-1)
   - Target: ~0.1 (10%) for UL — adaptive MCS tries to maintain this
   - BLER spike after migration = temporary QoS degradation
   - Track BLER before/after each migration to measure migration cost
   - r_qos in E-MATRO should increase when BLER > 0.15

5. HARQ rounds (r1/r2/r3/r4)
   - r1 only: perfect → most transmissions decoded on first try
   - r2/r3/r4 > 0: retransmissions → link stress or scheduler overload
   - r3+r4 spike after consolidation → node overloaded
   - r2/(r1+r2+r3+r4) ratio = practical BLER proxy

6. UE Throughput (Mbps, estimated from MCS × NPRB)
   - Most intuitive QoS metric for evaluation plots
   - Before migration: baseline throughput per UE
   - After migration: should remain ≥ 95% of baseline
   - Use in paper: "E-MATRO achieves X% energy saving with
     only Y% throughput degradation"

7. UL errors (ulsch_errors)
   - Packets that failed all 4 HARQ rounds → irrecoverable loss
   - Should be 0 at all times in RFSim
   - Non-zero after migration = severe overload

E-MATRO learning relevance summary:
  NPRB     → state input (load signal)
  MCS      → state input (link quality context)
  BLER     → reward shaping (QoS penalty)
  Throughput → evaluation metric (not in reward)
  HARQ     → evaluation metric (migration cost measurement)
"""


if __name__ == "__main__":
    print(METRIC_RELEVANCE)

    # Test parser on sample log
    sample = """
UE 2540: dlsch_rounds 4422/0/0/0, dlsch_errors 0, pucch0_DTX 0, BLER 0.00000 MCS (0) 9
UE 2540: ulsch_rounds 434266/3639/86/0, ulsch_errors 0, ulsch_DTX 0, BLER 0.01174 MCS (0) 28 (Qm 6  dB) NPRB 5  SNR 51.0 dB
UE fae4: dlsch_rounds 4110/0/0/0, dlsch_errors 0, pucch0_DTX 0, BLER 0.00000 MCS (0) 9
UE fae4: ulsch_rounds 403709/3177/34/0, ulsch_errors 0, ulsch_DTX 0, BLER 0.01952 MCS (0) 28 (Qm 6  dB) NPRB 5  SNR 51.0 dB
"""
    parsed = parse_du_logs(sample)
    print("\nParser test:")
    for ue in parsed:
        print(f"  UE {ue['rnti']}: UL MCS={ue.get('ul_mcs')} NPRB={ue.get('ul_nprb')} "
              f"BLER={ue.get('ul_bler')} SNR={ue.get('ul_snr_db')}dB "
              f"DL MCS={ue.get('dl_mcs')} BLER={ue.get('dl_bler')}")
        print(f"    Throughput: UL={_mcs_to_mbps(ue.get('ul_mcs',0), ue.get('ul_nprb',0))} Mbps")