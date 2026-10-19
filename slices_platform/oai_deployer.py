"""
OAI Deployer
============
Deploys OpenAirInterface 5G Core + gNB + UEs on vm-controller via SSH.
Uses the latincom-2025 Docker Compose setup from SLICES-RI.

Topology:
    1 gNB (RFSimulator) → represents 6 logical RUs
    18 UEs (oai-nr-ue1 … oai-nr-ue18) → 3 UEs per RU
    iperf3 server on UPF side → UEs generate UL traffic toward it

All containers run on vm-controller.
"""

import subprocess
import time
import logging
from typing import Optional

log = logging.getLogger(__name__)

BASTION  = "proxy@bastion2.slices-be.eu"
SSH_KEY  = "~/.ssh/id_ed25519"
CTL_IP   = "10.10.211.165"          # vm-controller — update per reservation
REPO_URL = "https://gitlab.inria.fr/slices-ri/latincom-2025.git"
REPO_DIR = "~/latincom-2025"
CN_DIR   = f"{REPO_DIR}/cn"


class OAIDeployer:
    """
    Manages the full OAI 5G stack lifecycle on vm-controller.
    """

    def __init__(self, controller_ip: str = CTL_IP,
                 ssh_key: str = SSH_KEY,
                 dry_run: bool = False):
        self.controller_ip = controller_ip
        self.ssh_key       = ssh_key
        self.dry_run       = dry_run

    # ── SSH helper ────────────────────────────────────────────────────────────

    def _ssh(self, cmd: str, timeout: int = 60) -> tuple:
        full = (f"ssh -i {self.ssh_key} "
                f"-o StrictHostKeyChecking=no "
                f"-o ConnectTimeout=10 "
                f"-o BatchMode=yes "
                f"-J {BASTION} "
                f"ubuntu@{self.controller_ip} '{cmd}'")
        if self.dry_run:
            log.info(f"[DRY] {cmd}")
            return 0, ""
        try:
            r = subprocess.run(full, shell=True, capture_output=True,
                               text=True, timeout=timeout)
            if r.returncode != 0 and r.stderr:
                log.warning(f"SSH stderr: {r.stderr.strip()[:200]}")
            return r.returncode, r.stdout.strip()
        except subprocess.TimeoutExpired:
            log.warning(f"SSH timeout: {cmd[:80]}")
            return -1, ""

    # ── Setup ─────────────────────────────────────────────────────────────────

    def setup(self):
        """Clone repo and install dependencies on vm-controller."""
        log.info("Setting up OAI on vm-controller...")

        # Install Docker if needed
        self._ssh("which docker || (curl -fsSL https://get.docker.com | sh && "
                  "sudo usermod -aG docker ubuntu)", timeout=120)

        # Clone the SLICES-RI tutorial repo
        rc, _ = self._ssh(f"test -d {REPO_DIR}")
        if rc != 0:
            self._ssh(f"git clone {REPO_URL} {REPO_DIR}", timeout=120)
            log.info("Repo cloned")
        else:
            self._ssh(f"cd {REPO_DIR} && git pull", timeout=30)
            log.info("Repo updated")

        # Install dependencies
        self._ssh(f"cd {REPO_DIR} && bash utils/install_dependencies.sh",
                  timeout=300)

        # Pull Docker images
        self._ssh(f"cd {CN_DIR} && docker compose -f docker-compose.yml pull",
                  timeout=600)
        log.info("Docker images pulled")

    # ── 5G Core ───────────────────────────────────────────────────────────────

    def start_core(self) -> bool:
        """Start OAI 5G Core network."""
        log.info("Starting 5G Core...")

        # Create network by starting mysql first
        self._ssh(f"cd {CN_DIR} && docker compose -f docker-compose.yml up -d mysql",
                  timeout=60)
        time.sleep(5)

        # Start full core
        self._ssh(f"cd {CN_DIR} && docker compose -f docker-compose.yml up -d",
                  timeout=120)

        # Wait for all components to be healthy
        log.info("Waiting for core network (up to 120s)...")
        for attempt in range(24):
            time.sleep(5)
            rc, out = self._ssh(
                f"cd {CN_DIR} && docker compose -f docker-compose.yml ps --format json "
                f"| python3 -c \"import sys,json; "
                f"data=[json.loads(l) for l in sys.stdin if l.strip()]; "
                f"unhealthy=[d['Name'] for d in data if d.get('Health','') not in ('healthy','')]; "
                f"print('READY' if not unhealthy else 'WAITING:'+','.join(unhealthy))\"")
            if "READY" in out:
                log.info("5G Core is healthy ✓")
                return True
            log.info(f"  Still waiting... ({attempt+1}/24): {out}")

        log.warning("Core did not become healthy in time")
        return False

    # ── gNB ───────────────────────────────────────────────────────────────────

    def start_gnb(self) -> bool:
        """Start OAI gNB (RFSimulator)."""
        log.info("Starting gNB...")
        self._ssh(f"cd {CN_DIR} && "
                  f"docker compose -f docker-compose-ran.yml up -d oai-gnb",
                  timeout=60)

        for attempt in range(12):
            time.sleep(5)
            rc, out = self._ssh(
                f"docker logs oai-gnb 2>&1 | grep -c 'Connected' || echo 0")
            if out.strip() != "0":
                log.info("gNB connected to core ✓")
                return True
        log.warning("gNB may not be connected")
        return False

    # ── UEs ───────────────────────────────────────────────────────────────────

    def start_ues(self, n_ues: int = 6) -> list:
        """
        Start N UE containers.
        Uses oai-nr-ue and oai-nr-ue2 from docker-compose-ran.yml
        for the first 2, then spawns additional UEs from the same image.

        Returns list of started container names.
        """
        log.info(f"Starting {n_ues} UEs...")
        started = []

        # Start the two preconfigured UEs
        for ue in ["oai-nr-ue", "oai-nr-ue2"]:
            self._ssh(f"cd {CN_DIR} && "
                      f"docker compose -f docker-compose-ran.yml up -d {ue}",
                      timeout=60)
            started.append(ue)
            time.sleep(3)

        # For additional UEs, reuse oai-nr-ue image with modified IMSI
        ue_image  = "oaisoftwarealliance/oai-nr-ue:develop"
        gnb_ip    = "192.168.71.140"   # from docker-compose-ran.yml

        for i in range(3, n_ues + 1):
            name = f"oai-nr-ue{i}"
            imsi = f"00101000000010{i:d}"   # unique IMSI per UE
            cmd  = (f"docker run -d --name {name} "
                    f"--network oaiworkshop "
                    f"--cap-add=NET_ADMIN "
                    f"-e USE_ADDITIONAL_OPTIONS=\"--rfsim "
                    f"--log_config.global_log_options level,nocolor,time "
                    f"-r 106 --numerology 1 --band 78 -C 3619200000 "
                    f"--rfsimulator.serveraddr {gnb_ip} "
                    f"-O /opt/oai-nr-ue/etc/ue.conf\" "
                    f"{ue_image}")
            self._ssh(f"docker rm -f {name} 2>/dev/null; {cmd}", timeout=30)
            started.append(name)
            time.sleep(2)

        # Wait for UEs to register
        log.info("Waiting for UEs to register (30s)...")
        time.sleep(30)

        # Report registered UEs
        rc, out = self._ssh(
            "docker logs oai-amf 2>&1 | grep '5GMM-REGISTERED' | wc -l")
        log.info(f"  {out.strip()} UEs registered with AMF")

        return started

    # ── iperf3 server ─────────────────────────────────────────────────────────

    def start_iperf_server(self, port: int = 5201) -> bool:
        """Start iperf3 server on the external data network container."""
        log.info("Starting iperf3 server on oai-ext-dn...")
        self._ssh(f"docker exec -d oai-ext-dn "
                  f"iperf3 -s -p {port} -D --logfile /tmp/iperf3_server.log",
                  timeout=15)
        time.sleep(2)
        rc, out = self._ssh(
            f"docker exec oai-ext-dn pgrep -c iperf3 || echo 0")
        ok = out.strip() != "0"
        if ok:
            log.info("iperf3 server running ✓")
        return ok

    # ── Teardown ──────────────────────────────────────────────────────────────

    def stop_all(self):
        """Stop all OAI containers."""
        log.info("Stopping all OAI containers...")
        self._ssh(f"cd {CN_DIR} && "
                  f"docker compose -f docker-compose-ran.yml down 2>/dev/null")
        self._ssh(f"cd {CN_DIR} && "
                  f"docker compose -f docker-compose.yml down 2>/dev/null")
        # Stop any extra UE containers
        self._ssh("docker ps --format '{{.Names}}' | grep oai-nr-ue | "
                  "xargs -r docker stop 2>/dev/null")
        log.info("All containers stopped")

    def full_deploy(self, n_ues: int = 6) -> bool:
        """
        Full deployment sequence:
            setup → core → gnb → ues → iperf3
        Returns True if all steps succeeded.
        """
        self.setup()
        ok = self.start_core()
        if not ok:
            log.error("Core failed to start")
            return False
        ok = self.start_gnb()
        if not ok:
            log.warning("gNB may not be connected — continuing anyway")
        self.start_ues(n_ues)
        self.start_iperf_server()
        log.info("Full OAI deployment complete ✓")
        return True