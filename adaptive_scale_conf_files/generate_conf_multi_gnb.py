"""
generate_conf_multi_gnb.py — Multi-gNB Configuration Generator
===============================================================
Generates OAI conf files for N_GNB independent gNBs,
each with UES_PER_GNB=4 UEs. Different cell IDs and port
offsets ensure gNBs don't interfere.

Usage:
  # Scenario A: 1 gNB (baseline, same as before)
  python3 generate_conf_multi_gnb.py --n-gnb 1 --deploy

  # Scenario B: 2 gNBs, 8 UEs total
  python3 generate_conf_multi_gnb.py --n-gnb 2 --deploy

  # Scenario C: 4 gNBs, 16 UEs total
  python3 generate_conf_multi_gnb.py --n-gnb 4 --deploy

  # Dry run — print without writing
  python3 generate_conf_multi_gnb.py --n-gnb 4 --dry-run
"""

import os
import yaml
import argparse
import subprocess
from pathlib import Path


# =============================================================================
# Templates
# =============================================================================

DU_CONF = """########################################
# DU gNB{gnb_id} — node{node_id} ({du_ip})
# Serves UEs {ue_start}..{ue_end} (RUs {ru_start}..{ru_end})
# F1 → CU gNB{cu_gnb_id} ({cu_ip})
########################################
Active_gNBs = ( "gNB-DU-{gnb_id}" );
Asn1_verbosity = "none";

gNBs = (
{{
  gNB_ID    = 0xe{gnb_id:02x}0;
  gNB_name  = "gNB-DU-{gnb_id}";
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

  local_n_address  = "{du_ip}";
  remote_n_address = "{cu_ip}";
  local_n_portc    = {portc};
  remote_n_portc   = {remote_portc};
  local_n_portd    = {portd};
  remote_n_portd   = {remote_portd};

  servingCellConfigCommon = (
  {{
    physCellId                  = {phys_cell_id};
    dl_frequencyBand            = 78;
    dl_absoluteFrequencySSB     = {ssb_freq};
    dl_absoluteFrequencyPointA  = {point_a};
    dl_offstToCarrier           = 0;
    dl_subcarrierSpacing        = 1;
    dl_carrierBandwidth         = 106;
    initialDLBWPlocationAndBandwidth = 28875;
    initialDLBWPsubcarrierSpacing    = 1;
    ul_frequencyBand            = 78;
    ul_absoluteFrequencyPointA  = {point_a};
    ul_offstToCarrier           = 0;
    ul_subcarrierSpacing        = 1;
    ul_carrierBandwidth         = 106;
    initialULBWPlocationAndBandwidth = 28875;
    initialULBWPsubcarrierSpacing    = 1;
    pMax                        = 20;
    dl_AntennaPortsCount        = 1;
    ul_AntennaPortsCount        = 1;
    ssPBCH_BlockPower           = -25;
  }}
  );
}});

MACRLCs = (
{{
  num_cc             = 1;
  tr_s_preference    = "f1";
  local_s_address    = "{du_ip}";
  remote_s_address   = "{cu_ip}";
  local_s_portc      = {portc};
  remote_s_portc     = {remote_portc};
  local_s_portd      = {portd};
  remote_s_portd     = {remote_portd};
  pusch_TargetSNRx10 = 200;
  pucch_TargetSNRx10 = 200;
}});

rfsimulator: {{
  serveraddr = "server";
  serverport = "{rfsim_port}";
  options    = ();
  modelname  = "AWGN";
  IQfile     = "/tmp/rfsimulator";
}};

security = {{
  ciphering_algorithms = ( "nea0" );
  integrity_algorithms = ( "nia2", "nia0" );
  drb_ciphering        = "no";
  drb_integrity        = "no";
}};

log_config: {{
  global_log_level   = "info";
  global_log_options = "level,nocolor,time";
}};
"""

CU_CONF = """########################################
# CU gNB{cu_gnb_id} — node{node_id} ({cu_ip})
# Serves DUs: {du_list}
# N2 → AMF ({ctl_ip}:38412)
# N3 → UPF ({ctl_ip}:2152)
########################################
Active_gNBs = ( "gNB-CU-{cu_gnb_id}" );
Asn1_verbosity = "none";

gNBs = (
{{
  gNB_ID    = 0xe{gnb_hex:02x}0;
  gNB_name  = "gNB-CU-{cu_gnb_id}";
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
  local_s_portc    = {portc};
  local_s_portd    = {portd};

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
    GNB_PORT_FOR_S1U              = {s1u_port};
  }};

  e1_ipaddress = "127.0.0.1";
  e1_portc     = 38462;
}});

log_config: {{
  global_log_level   = "info";
  global_log_options = "level,nocolor,time";
}};
"""

UE_CONF = """uicc0 = {{
  imsi = "0010100000001{ue_id_padded}";
  key  = "fec86ba6eb707ed08905757b1bb44b8f";
  opc  = "C42449363BBAD02B66D16BC975D77CC1";
  dnn  = "oai";
  nssai_sst = 1;
  nssai_sd  = 0xffffff;
}};

rfsimulator: {{
  serveraddr = "{du_ip}";
  serverport = "{rfsim_port}";
  options    = ();
  modelname  = "AWGN";
  IQfile     = "/tmp/rfsimulator";
}};

log_config: {{
  global_log_level = "info";
}};
"""


# =============================================================================
# Port / frequency helpers — each gNB gets unique offsets
# =============================================================================

BASE_PORTC    = 500
BASE_PORTD    = 2153
BASE_RFSIM    = 4043
BASE_SSB_FREQ = 641280
BASE_POINT_A  = 640008
PORT_STEP     = 20    # offset between gNBs

def gnb_ports(gnb_id: int) -> dict:
    return {
        "portc":        BASE_PORTC  + gnb_id * PORT_STEP,
        "portd":        BASE_PORTD  + gnb_id * PORT_STEP,
        "remote_portc": BASE_PORTC  + gnb_id * PORT_STEP,
        "remote_portd": BASE_PORTD  + gnb_id * PORT_STEP,
        "rfsim_port":   str(BASE_RFSIM + gnb_id),
        "s1u_port":     2153 + gnb_id * PORT_STEP,
        "ssb_freq":     BASE_SSB_FREQ + gnb_id * 100,
        "point_a":      BASE_POINT_A  + gnb_id * 100,
        "cell_id":      12345678 + gnb_id,
        "phys_cell_id": gnb_id % 504,
    }


# =============================================================================
# Generator
# =============================================================================

class MultiGNBGenerator:

    def __init__(self, n_gnb: int, topo_module, out_dir: str = "."):
        self.n_gnb  = n_gnb
        self.t      = topo_module
        self.outdir = Path(out_dir)
        self.conf   = self.outdir / "conf"
        self.conf.mkdir(parents=True, exist_ok=True)

    def _write(self, path: Path, content: str, dry: bool):
        if dry:
            print(f"\n── {path} ──")
            print(content[:300] + "..." if len(content) > 300 else content)
        else:
            path.write_text(content)
            print(f"  Written: {path.name}")

    def gen_du_confs(self, dry=False):
        print("\n── DU conf files ──────────────────────────────────")
        for gnb_id in range(self.n_gnb):
            node     = self.t.NODE_CONFIGS[gnb_id]
            cu_gnb   = gnb_id // 2
            cu_nid   = self.t.N_DUS + cu_gnb
            cu_node  = self.t.NODE_CONFIGS[cu_nid]
            ports    = gnb_ports(gnb_id)
            ue_start = gnb_id * self.t.UES_PER_GNB + 1
            ue_end   = ue_start + self.t.UES_PER_GNB - 1

            content = DU_CONF.format(
                gnb_id      = gnb_id,
                node_id     = node.node_id,
                du_ip       = node.ip,
                cu_gnb_id   = cu_gnb,
                cu_ip       = cu_node.ip,
                ue_start    = ue_start,
                ue_end      = ue_end,
                ru_start    = gnb_id * self.t.UES_PER_GNB,
                ru_end      = gnb_id * self.t.UES_PER_GNB + self.t.UES_PER_GNB - 1,
                **ports,
            )
            self._write(self.conf / f"du.gnb{gnb_id}.conf", content, dry)

    def gen_cu_confs(self, dry=False):
        print("\n── CU conf files ──────────────────────────────────")
        for cu_idx in range(self.t.N_CUS):
            cu_nid  = self.t.N_DUS + cu_idx
            node    = self.t.NODE_CONFIGS[cu_nid]
            ports   = gnb_ports(cu_idx)
            du_list = [g for g in range(self.n_gnb) if g // 2 == cu_idx]

            content = CU_CONF.format(
                cu_gnb_id = cu_idx,
                node_id   = node.node_id,
                cu_ip     = node.ip,
                du_list   = du_list,
                ctl_ip    = self.t.CTL_IP,
                gnb_hex   = self.t.N_DUS + cu_idx,
                cell_id   = 12345679 + cu_idx,
                portc     = ports["portc"],
                portd     = ports["portd"],
                s1u_port  = ports["s1u_port"],
            )
            self._write(self.conf / f"cu.gnb{cu_idx}.conf", content, dry)

    def gen_ue_confs(self, dry=False):
        print("\n── UE conf files ──────────────────────────────────")
        for ue_id in range(self.t.N_UES):
            gnb_id  = ue_id // self.t.UES_PER_GNB
            du_node = self.t.NODE_CONFIGS[gnb_id]
            ports   = gnb_ports(gnb_id)

            content = UE_CONF.format(
                ue_id_padded = f"{ue_id+1:02d}",
                du_ip        = du_node.ip,
                rfsim_port   = ports["rfsim_port"],
            )
            self._write(self.conf / f"ue{ue_id+1}.conf", content, dry)

    def gen_du_compose(self, dry=False):
        """One docker-compose per DU VM."""
        print("\n── docker-compose-du-*.yml ────────────────────────")
        vm_to_gnbs: dict = {}
        for gnb_id in range(self.n_gnb):
            node = self.t.NODE_CONFIGS[gnb_id]
            vm_to_gnbs.setdefault(node.ip, []).append(gnb_id)

        for vm_ip, gnbs in vm_to_gnbs.items():
            services = {}
            for gnb_id in gnbs:
                ports = gnb_ports(gnb_id)
                services[f"oai-du-gnb{gnb_id}"] = {
                    "image":          "oaisoftwarealliance/oai-gnb:2024.w26",
                    "privileged":     True,
                    "container_name": f"oai-du-gnb{gnb_id}",
                    "environment":    {"TZ": "Europe/Paris"},
                    "volumes":        [f"./conf/du.gnb{gnb_id}.conf:/opt/oai-gnb/etc/gnb.conf"],
                    "command": (
                        f"/opt/oai-gnb/bin/nr-softmodem "
                        f"-O /opt/oai-gnb/etc/gnb.conf "
                        f"-E --sa --rfsim "
                        f"--log_config.global_log_options level,nocolor,time"
                    ),
                    "networks": {"public_net": {}},
                }
            compose = {
                "services": services,
                "networks": {"public_net": {"external": True, "name": "oai_e_matro"}},
            }
            vm_idx = self.t.VM_IPS.index(vm_ip)
            self._write(
                self.outdir / f"docker-compose-du-vm{vm_idx}.yml",
                yaml.dump(compose, default_flow_style=False), dry
            )

    def gen_cu_compose(self, dry=False):
        """One docker-compose per CU VM."""
        print("\n── docker-compose-cu-*.yml ────────────────────────")
        for cu_idx in range(self.t.N_CUS):
            cu_nid = self.t.N_DUS + cu_idx
            node   = self.t.NODE_CONFIGS[cu_nid]
            services = {
                f"oai-cu-gnb{cu_idx}": {
                    "image":          "oaisoftwarealliance/oai-gnb:2024.w26",
                    "privileged":     True,
                    "container_name": f"oai-cu-gnb{cu_idx}",
                    "environment":    {"TZ": "Europe/Paris"},
                    "volumes":        [f"./conf/cu.gnb{cu_idx}.conf:/opt/oai-gnb/etc/gnb.conf"],
                    "command": (
                        "/opt/oai-gnb/bin/nr-softmodem "
                        "-O /opt/oai-gnb/etc/gnb.conf "
                        "--sa "
                        "--log_config.global_log_options level,nocolor,time"
                    ),
                    "networks": {"public_net": {}},
                }
            }
            compose = {
                "services": services,
                "networks": {"public_net": {"external": True, "name": "oai_e_matro"}},
            }
            vm_idx = self.t.VM_IPS.index(node.ip)
            self._write(
                self.outdir / f"docker-compose-cu-vm{vm_idx}.yml",
                yaml.dump(compose, default_flow_style=False), dry
            )

    def gen_ue_compose(self, dry=False):
        """One docker-compose-ran.yml for all UEs."""
        print("\n── docker-compose-ran-multi.yml ───────────────────")
        services = {}
        for ue_id in range(self.t.N_UES):
            gnb_id  = ue_id // self.t.UES_PER_GNB
            du_node = self.t.NODE_CONFIGS[gnb_id]
            ports   = gnb_ports(gnb_id)
            name    = "oai-nr-ue" if ue_id == 0 else f"oai-nr-ue{ue_id+1}"
            services[name] = {
                "image":          "oaisoftwarealliance/oai-nr-ue:2024.w26",
                "privileged":     True,
                "container_name": name,
                "cap_add":        ["NET_ADMIN"],
                "environment":    {"TZ": "Europe/Paris"},
                "volumes":        [f"./conf/ue{ue_id+1}.conf:/opt/oai-nr-ue/etc/nr-ue.conf"],
                "command": (
                    "/opt/oai-nr-ue/bin/nr-uesoftmodem "
                    f"-O /opt/oai-nr-ue/etc/nr-ue.conf "
                    f"-E --sa --rfsim -r 106 --numerology 1 "
                    f"--uicc0.imsi 001010000000{ue_id+1:02d} "
                    f"--rfsimulator.serveraddr {du_node.ip} "
                    f"--rfsimulator.serverport {ports['rfsim_port']}"
                ),
                "networks": {
                    "public_net": {"ipv4_address": f"192.168.70.{150+ue_id}"}
                },
            }
        compose = {
            "services": services,
            "networks": {"public_net": {"external": True, "name": "oai_e_matro"}},
        }
        self._write(
            self.outdir / "docker-compose-ran-multi.yml",
            yaml.dump(compose, default_flow_style=False), dry
        )

    def deploy(self):
        """SCP conf files to each VM."""
        print("\n── Deploying to vm-servers ────────────────────────")
        ssh_opts = "-o StrictHostKeyChecking=no"
        user     = self.t.SSH_USER

        for gnb_id in range(self.n_gnb):
            node = self.t.NODE_CONFIGS[gnb_id]
            subprocess.run(
                f"ssh {ssh_opts} {user}@{node.ip} 'mkdir -p ~/oai/conf'",
                shell=True, check=False
            )
            conf = self.conf / f"du.gnb{gnb_id}.conf"
            r = subprocess.run(
                f"scp {ssh_opts} {conf} {user}@{node.ip}:~/oai/conf/",
                shell=True, capture_output=True
            )
            print(f"  DU{gnb_id} → {node.ip} {'✓' if r.returncode==0 else '✗'}")

            vm_idx   = self.t.VM_IPS.index(node.ip)
            compose  = self.outdir / f"docker-compose-du-vm{vm_idx}.yml"
            if compose.exists():
                subprocess.run(
                    f"scp {ssh_opts} {compose} {user}@{node.ip}:~/oai/docker-compose-du.yml",
                    shell=True, capture_output=True
                )

        for cu_idx in range(self.t.N_CUS):
            cu_nid = self.t.N_DUS + cu_idx
            node   = self.t.NODE_CONFIGS[cu_nid]
            subprocess.run(
                f"ssh {ssh_opts} {user}@{node.ip} 'mkdir -p ~/oai/conf'",
                shell=True, check=False
            )
            conf = self.conf / f"cu.gnb{cu_idx}.conf"
            r = subprocess.run(
                f"scp {ssh_opts} {conf} {user}@{node.ip}:~/oai/conf/",
                shell=True, capture_output=True
            )
            print(f"  CU{cu_idx} → {node.ip} {'✓' if r.returncode==0 else '✗'}")

            vm_idx  = self.t.VM_IPS.index(node.ip)
            compose = self.outdir / f"docker-compose-cu-vm{vm_idx}.yml"
            if compose.exists():
                subprocess.run(
                    f"scp {ssh_opts} {compose} {user}@{node.ip}:~/oai/docker-compose-cu.yml",
                    shell=True, capture_output=True
                )

    def generate_all(self, dry=False, deploy=False):
        print(f"\nGenerating multi-gNB config: {self.n_gnb} gNBs × {self.t.UES_PER_GNB} UEs = {self.t.N_UES} UEs")
        self.gen_du_confs(dry)
        self.gen_cu_confs(dry)
        self.gen_ue_confs(dry)
        self.gen_du_compose(dry)
        self.gen_cu_compose(dry)
        self.gen_ue_compose(dry)
        if deploy and not dry:
            self.deploy()
        print("\nDone ✓")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate multi-gNB OAI conf files"
    )
    parser.add_argument("--n-gnb",   type=int, default=2,
                        help="Number of gNBs (each with 4 UEs). Default: 2")
    parser.add_argument("--out-dir", default=".",
                        help="Output directory")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--deploy",  action="store_true",
                        help="SCP files to vm-servers")
    args = parser.parse_args()

    # Override topology module
    import sys
    sys.path.insert(0, args.out_dir)
    import topology_multi_gnb as topo

    topo.N_GNB   = args.n_gnb
    topo.N_UES   = args.n_gnb * topo.UES_PER_GNB
    topo.N_RUS   = topo.N_UES
    topo.N_DUS   = args.n_gnb
    topo.N_CUS   = max(1, (args.n_gnb + 1) // 2)
    topo.N_NODES = min(topo.N_DUS + topo.N_CUS, topo.N_VMS)
    topo.NODE_CONFIGS = topo.build_node_configs()
    topo.RU_TO_DU, topo.RU_TO_CU = topo.build_ru_mapping()
    topo.UE_NAMES = topo.build_ue_names()
    topo.validate()

    gen = MultiGNBGenerator(args.n_gnb, topo, args.out_dir)
    gen.generate_all(dry=args.dry_run, deploy=args.deploy)