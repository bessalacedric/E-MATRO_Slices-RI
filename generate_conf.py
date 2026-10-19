"""
generate_conf.py — Dynamic DU/CU Configuration Generator
=========================================================
Generates OAI DU/CU conf files and docker-compose files
from topology.py without hardcoding node counts.

Usage:
  python3 generate_conf.py              # generates for current topology
  python3 generate_conf.py --n-rus 8   # override N_RUS and regenerate
  python3 generate_conf.py --dry-run   # print without writing files
  python3 generate_conf.py --deploy    # generate + scp to vm-servers
"""

import os
import yaml
import argparse
import subprocess
from pathlib import Path

# =============================================================================
# Templates
# =============================================================================

DU_CONF_TEMPLATE = """########################################
# DU Node {du_id} — vm-server (node {du_id})
# Serves RUs: {ru_list}
# F1 → CU node {cu_node_id} ({cu_ip})
# N2/N3 via vm-controller ({ctl_ip})
########################################
Active_gNBs = ( "gNB-DU-node{du_id}" );
Asn1_verbosity = "none";

gNBs = (
{{
  gNB_ID    = 0xe{du_id:02x}0;
  gNB_name  = "gNB-DU-node{du_id}";
  tracking_area_code  = 1;

  plmn_list = ({{
    mcc = 001; mnc = 01; mnc_length = 2;
    snssaiList = (
      {{ sst = 1; sd = 0xffffff }},
      {{ sst = 1; sd = 0x1      }}
    );
  }});

  nr_cellid = {cell_id}L;
  tr_s_preference = "f1";

  local_n_address = "{du_ip}";
  remote_n_address = "{cu_ip}";
  local_n_portc  = {portc};
  remote_n_portc = 500;
  local_n_portd  = {portd};
  remote_n_portd = 2153;

  servingCellConfigCommon = (
  {{
    physCellId = {cell_id_short};
    dl_frequencyBand = 78;
    dl_absoluteFrequencySSB = 641280;
    dl_absoluteFrequencyPointA = 640008;
    dl_offstToCarrier = 0;
    dl_subcarrierSpacing = 1;
    dl_carrierBandwidth = 106;
    initialDLBWPlocationAndBandwidth = 28875;
    initialDLBWPsubcarrierSpacing = 1;
    ul_frequencyBand = 78;
    ul_absoluteFrequencyPointA = 640008;
    ul_offstToCarrier = 0;
    ul_subcarrierSpacing = 1;
    ul_carrierBandwidth = 106;
    initialULBWPlocationAndBandwidth = 28875;
    initialULBWPsubcarrierSpacing = 1;
    pMax = 20;
    dl_AntennaPortsCount = 1;
    ul_AntennaPortsCount = 1;
    ssPBCH_BlockPower = -25;
  }}
  );
}});

MACRLCs = (
{{
  num_cc = 1;
  tr_s_preference = "f1";
  local_s_address = "{du_ip}";
  remote_s_address = "{cu_ip}";
  local_s_portc = {portc};
  remote_s_portc = 500;
  local_s_portd = {portd};
  remote_s_portd = 2153;
  pusch_TargetSNRx10 = 200;
  pucch_TargetSNRx10 = 200;
}});

rfsimulator: {{
  serveraddr = "server";
  serverport = "4043";
  options = ();
  modelname = "AWGN";
  IQfile = "/tmp/rfsimulator";
}};

security = {{
  # don't enable mandatory integrity/ciphering — rfsim only
  ciphering_algorithms = ( "nea0" );
  integrity_algorithms = ( "nia2", "nia0" );
  drb_ciphering = "no";
  drb_integrity = "no";
}};

log_config: {{
  global_log_level = "info";
  global_log_options = "level,nocolor,time";
}};
"""

CU_CONF_TEMPLATE = """########################################
# CU Node {cu_node_id} — vm-server (node {cu_node_id})
# Serves DUs: {du_list}
# N2 → AMF on vm-controller ({ctl_ip}:38412)
# N3 → UPF on vm-controller ({ctl_ip}:2152)
########################################
Active_gNBs = ( "gNB-CU-node{cu_node_id}" );
Asn1_verbosity = "none";

gNBs = (
{{
  gNB_ID    = 0xe{gNb_id:02x}0;
  gNB_name  = "gNB-CU-node{cu_node_id}";
  tracking_area_code  = 1;

  plmn_list = ({{
    mcc = 001; mnc = 01; mnc_length = 2;
    snssaiList = (
      {{ sst = 1; sd = 0xffffff }},
      {{ sst = 1; sd = 0x1      }}
    );
  }});

  nr_cellid = {cell_id}L;
  tr_s_preference = "f1";

  local_s_address  = "{cu_ip}";
  local_s_portc    = 500;
  local_s_portd    = 2153;

  amf_ip_address = ({{
    ipv4       = "{ctl_ip}";
    ipv6       = "::1";
    active     = "yes";
    preference = "ipv4";
  }});

  NETWORK_INTERFACES: {{
    GNB_INTERFACE_NAME_FOR_NG_AMF = "eth0";
    GNB_IPV4_ADDRESS_FOR_NG_AMF   = "{cu_ip}";
    GNB_INTERFACE_NAME_FOR_NGU    = "eth0";
    GNB_IPV4_ADDRESS_FOR_NGU      = "{cu_ip}";
    GNB_PORT_FOR_S1U              = 2153;
  }};

  e1_ipaddress = "127.0.0.1";
  e1_portc     = 38462;
}});

log_config: {{
  global_log_level = "info";
  global_log_options = "level,nocolor,time";
}};
"""

UE_CONF_TEMPLATE = """uicc0 = {{
  imsi = "00101000000010{ue_id_padded}";
  key = "fec86ba6eb707ed08905757b1bb44b8f";
  opc = "C42449363BBAD02B66D16BC975D77CC1";
  dnn = "oai";
  nssai_sst = 1;
  nssai_sd = 0xffffff;
}};

rfsimulator: {{
  serveraddr = "{du_ip}";
  serverport = "4043";
  options = ();
  modelname = "AWGN";
  IQfile = "/tmp/rfsimulator";
}};

log_config: {{
  global_log_level = "info";
}};
"""


# =============================================================================
# Compose file generators
# =============================================================================

def generate_du_compose(du_id: int, du_ip: str) -> dict:
    """Generate docker-compose service entry for one DU container."""
    return {
        f"oai-du-node{du_id}": {
            "image": "oaisoftwarealliance/oai-gnb:2024.w26",
            "privileged": True,
            "container_name": f"oai-du-node{du_id}",
            "environment": {"TZ": "Europe/Paris"},
            "volumes": [
                f"./conf/du.node{du_id}.conf:/opt/oai-gnb/etc/gnb.conf"
            ],
            "command": (
                "/opt/oai-gnb/bin/nr-softmodem "
                "-O /opt/oai-gnb/etc/gnb.conf "
                "-E --sa --rfsim "
                "--log_config.global_log_options level,nocolor,time"
            ),
            "healthcheck": {
                "test": '/bin/bash -c "pgrep nr-softmodem"',
                "interval": "10s",
                "timeout": "5s",
                "retries": 5,
            },
            "networks": {"public_net": {}},
        }
    }


def generate_cu_compose(cu_node_id: int, cu_ip: str) -> dict:
    """Generate docker-compose service entry for one CU container."""
    return {
        f"oai-cu-node{cu_node_id}": {
            "image": "oaisoftwarealliance/oai-gnb:2024.w26",
            "privileged": True,
            "container_name": f"oai-cu-node{cu_node_id}",
            "environment": {"TZ": "Europe/Paris"},
            "volumes": [
                f"./conf/cu.node{cu_node_id}.conf:/opt/oai-gnb/etc/gnb.conf"
            ],
            "command": (
                "/opt/oai-gnb/bin/nr-softmodem "
                "-O /opt/oai-gnb/etc/gnb.conf "
                "--sa "
                "--log_config.global_log_options level,nocolor,time"
            ),
            "healthcheck": {
                "test": '/bin/bash -c "pgrep nr-softmodem"',
                "interval": "10s",
                "timeout": "5s",
                "retries": 5,
            },
            "networks": {"public_net": {}},
        }
    }


def generate_ue_compose(ue_id: int, ue_ip_suffix: int) -> dict:
    """Generate docker-compose service entry for one UE container."""
    name = "oai-nr-ue" if ue_id == 0 else f"oai-nr-ue{ue_id+1}"
    return {
        name: {
            "image": "oaisoftwarealliance/oai-nr-ue:2024.w26",
            "privileged": True,
            "container_name": name,
            "cap_add": ["NET_ADMIN"],
            "environment": {"TZ": "Europe/Paris"},
            "volumes": [
                f"./conf/ue{ue_id+1}.conf:/opt/oai-nr-ue/etc/nr-ue.conf"
            ],
            "command": (
                "/opt/oai-nr-ue/bin/nr-uesoftmodem "
                f"-O /opt/oai-nr-ue/etc/nr-ue.conf "
                "-E --sa --rfsim -r 106 --numerology 1 "
                "--uicc0.imsi 00101000000010{:02d} ".format(ue_id+1) +
                f"--rfsimulator.serveraddr {ue_ip_suffix}"
            ),
            "networks": {
                "public_net": {
                    "ipv4_address": f"192.168.70.{150+ue_id}"
                }
            },
        }
    }


# =============================================================================
# Main generator
# =============================================================================

class TopologyGenerator:
    """
    Generates all configuration files from topology.py.
    Works with any N_RUS value without manual editing.
    """

    def __init__(self, topology_module, output_dir: str = "."):
        self.topo   = topology_module
        self.outdir = Path(output_dir)
        self.conf_dir = self.outdir / "conf"

    def _write(self, path: Path, content: str, dry_run: bool = False):
        if dry_run:
            print(f"\n{'='*60}")
            print(f"  FILE: {path}")
            print(f"{'='*60}")
            print(content[:500] + ("..." if len(content) > 500 else ""))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            print(f"  Written: {path}")

    def generate_du_confs(self, dry_run: bool = False):
        """Generate du.nodeN.conf for each DU node."""
        print("\n── DU conf files ──────────────────────────────────")
        for du_id in range(self.topo.N_DUS):
            node = self.topo.NODE_CONFIGS[du_id]
            # Find CU for this DU (first RU served by this DU)
            first_ru = du_id * self.topo.RUS_PER_DU
            cu_node_id = self.topo.RU_TO_CU[first_ru]
            cu_node    = self.topo.NODE_CONFIGS[cu_node_id]
            ru_list    = list(range(first_ru, first_ru + self.topo.RUS_PER_DU))

            content = DU_CONF_TEMPLATE.format(
                du_id       = du_id,
                ru_list     = ru_list,
                cu_node_id  = cu_node_id,
                cu_ip       = cu_node.ip,
                ctl_ip      = self.topo.CTL_IP,
                du_ip       = node.ip,
                cell_id     = 12345678 + du_id,
                cell_id_short = du_id,
                portc       = 500 + du_id * 10,
                portd       = 2153 + du_id * 10,
            )
            self._write(self.conf_dir / f"du.node{du_id}.conf", content, dry_run)

    def generate_cu_confs(self, dry_run: bool = False):
        """Generate cu.nodeN.conf for each CU node."""
        print("\n── CU conf files ──────────────────────────────────")
        for j in range(self.topo.N_CUS):
            cu_node_id = self.topo.N_DUS + j
            node = self.topo.NODE_CONFIGS[cu_node_id]
            du_list = [du_id for du_id in range(self.topo.N_DUS)
                       if self.topo.RU_TO_CU[du_id * self.topo.RUS_PER_DU] == cu_node_id]

            content = CU_CONF_TEMPLATE.format(
                cu_node_id  = cu_node_id,
                gNb_id      = cu_node_id,
                du_list     = du_list,
                ctl_ip      = self.topo.CTL_IP,
                cu_ip       = node.ip,
                cell_id     = 12345679 + j,
            )
            self._write(self.conf_dir / f"cu.node{cu_node_id}.conf", content, dry_run)

    def generate_ue_confs(self, dry_run: bool = False):
        """Generate ueN.conf for each UE."""
        print("\n── UE conf files ──────────────────────────────────")
        for ue_id in range(self.topo.N_UES):
            du_node_id = self.topo.RU_TO_DU[ue_id]
            du_node    = self.topo.NODE_CONFIGS[du_node_id]
            content = UE_CONF_TEMPLATE.format(
                ue_id_padded = f"{ue_id+1:02d}",
                du_ip        = du_node.ip,
            )
            self._write(self.conf_dir / f"ue{ue_id+1}.conf", content, dry_run)

    def generate_du_compose(self, dry_run: bool = False):
        """Generate docker-compose-du.yml for each DU VM."""
        print("\n── docker-compose-du.yml ──────────────────────────")
        # Group DU nodes by VM
        vm_to_dus = {}
        for du_id in range(self.topo.N_DUS):
            node = self.topo.NODE_CONFIGS[du_id]
            vm_to_dus.setdefault(node.ip, []).append(du_id)

        for vm_ip, du_ids in vm_to_dus.items():
            services = {}
            for du_id in du_ids:
                node = self.topo.NODE_CONFIGS[du_id]
                services.update(generate_du_compose(du_id, node.ip))

            compose = {
                "services": services,
                "networks": {
                    "public_net": {
                        "external": True,
                        "name": "oai_e_matro"
                    }
                }
            }
            # Use vm index for filename
            vm_idx = self.topo.VM_IPS.index(vm_ip)
            path   = self.outdir / f"docker-compose-du-vm{vm_idx}.yml"
            self._write(path, yaml.dump(compose, default_flow_style=False), dry_run)

    def generate_cu_compose(self, dry_run: bool = False):
        """Generate docker-compose-cu.yml for each CU VM."""
        print("\n── docker-compose-cu.yml ──────────────────────────")
        vm_to_cus = {}
        for j in range(self.topo.N_CUS):
            cu_node_id = self.topo.N_DUS + j
            node = self.topo.NODE_CONFIGS[cu_node_id]
            vm_to_cus.setdefault(node.ip, []).append(cu_node_id)

        for vm_ip, cu_ids in vm_to_cus.items():
            services = {}
            for cu_id in cu_ids:
                node = self.topo.NODE_CONFIGS[cu_id]
                services.update(generate_cu_compose(cu_id, node.ip))

            compose = {
                "services": services,
                "networks": {
                    "public_net": {
                        "external": True,
                        "name": "oai_e_matro"
                    }
                }
            }
            vm_idx = self.topo.VM_IPS.index(vm_ip)
            path   = self.outdir / f"docker-compose-cu-vm{vm_idx}.yml"
            self._write(path, yaml.dump(compose, default_flow_style=False), dry_run)

    def generate_ue_compose(self, dry_run: bool = False):
        """Generate docker-compose-ran.yml for all UEs."""
        print("\n── docker-compose-ran.yml ─────────────────────────")
        services = {}
        for ue_id in range(self.topo.N_UES):
            du_node = self.topo.NODE_CONFIGS[self.topo.RU_TO_DU[ue_id]]
            services.update(generate_ue_compose(ue_id, du_node.ip))

        compose = {
            "services": services,
            "networks": {
                "public_net": {
                    "external": True,
                    "name": "oai_e_matro"
                }
            }
        }
        path = self.outdir / "docker-compose-ran.yml"
        self._write(path, yaml.dump(compose, default_flow_style=False), dry_run)

    def generate_all(self, dry_run: bool = False, deploy: bool = False):
        """Generate all conf and compose files."""
        print(f"\nGenerating topology: N_RUS={self.topo.N_RUS} "
              f"N_DUS={self.topo.N_DUS} N_CUS={self.topo.N_CUS} "
              f"N_NODES={self.topo.N_NODES}")

        self.generate_du_confs(dry_run)
        self.generate_cu_confs(dry_run)
        self.generate_ue_confs(dry_run)
        self.generate_du_compose(dry_run)
        self.generate_cu_compose(dry_run)
        self.generate_ue_compose(dry_run)

        if deploy and not dry_run:
            self._deploy()

        print("\nDone ✓")

    def _deploy(self):
        """SCP generated conf files to each vm-server."""
        print("\n── Deploying to vm-servers ────────────────────────")
        ssh_opts = "-o StrictHostKeyChecking=no"
        user     = self.topo.SSH_USER

        for node_id, node in self.topo.NODE_CONFIGS.items():
            ip = node.ip
            # Create remote directory
            subprocess.run(
                f"ssh {ssh_opts} {user}@{ip} 'mkdir -p ~/oai/conf'",
                shell=True, check=False
            )
            # Copy relevant conf file
            if node.role == "DU":
                conf = self.conf_dir / f"du.node{node_id}.conf"
                remote = f"~/oai/conf/du.node{node_id}.conf"
            else:
                conf = self.conf_dir / f"cu.node{node_id}.conf"
                remote = f"~/oai/conf/cu.node{node_id}.conf"

            if conf.exists():
                result = subprocess.run(
                    f"scp {ssh_opts} {conf} {user}@{ip}:{remote}",
                    shell=True, capture_output=True
                )
                status = "✓" if result.returncode == 0 else "✗"
                print(f"  {node.role}{node_id} → {ip}:{remote} {status}")

            # Copy compose file
            vm_idx     = self.topo.VM_IPS.index(ip)
            role_lower = node.role.lower()
            compose    = self.outdir / f"docker-compose-{role_lower}-vm{vm_idx}.yml"
            if compose.exists():
                result = subprocess.run(
                    f"scp {ssh_opts} {compose} "
                    f"{user}@{ip}:~/oai/docker-compose-{role_lower}.yml",
                    shell=True, capture_output=True
                )
                status = "✓" if result.returncode == 0 else "✗"
                print(f"  compose → {ip}:~/oai/docker-compose-{role_lower}.yml {status}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser( description="Generate OAI conf and compose files from topology")
    parser.add_argument("--n-rus",    type=int, default=None, help="Override N_RUS in topology.py")
    parser.add_argument("--dry-run",  action="store_true", help="Print generated files without writing")
    parser.add_argument("--deploy",   action="store_true", help="SCP files to vm-servers after generation")
    parser.add_argument("--out-dir",  default=".", help="Output directory (default: current)")
    args = parser.parse_args()

    import topology as topo

    if args.n_rus is not None:
        # Override N_RUS at runtime
        topo.N_RUS  = args.n_rus
        topo.N_UES  = args.n_rus
        topo.N_DUS  = max(2, args.n_rus // 2)
        topo.N_CUS  = max(2, (args.n_rus + 3) // 4)
        topo.N_NODES = topo.N_DUS + topo.N_CUS
        topo.RUS_PER_DU = topo.N_RUS // topo.N_DUS
        topo.NODE_CONFIGS = topo.build_node_configs()
        topo.RU_TO_DU, topo.RU_TO_CU = topo.build_ru_mapping()
        topo.UE_NAMES = topo.build_ue_names()

    topo.validate_topology()

    gen = TopologyGenerator(topo, output_dir=args.out_dir)
    gen.generate_all(dry_run=args.dry_run, deploy=args.deploy)