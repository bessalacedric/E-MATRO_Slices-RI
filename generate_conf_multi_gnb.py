#!/usr/bin/env python3
"""
generate_conf_multi_gnb.py
==========================
Generates OAI DU/CU configuration files, UE docker-compose,
and a deploy script for the E-MATRO multi-gNB setup on SLICES-RI.

Reads topology from topology.py (single source of truth).

VM assignment per gNB k  (k = 0 .. N_GNB-1):
  DU  ->  VM_IPS[2*k]       (vm-server-2k+1)
  CU  ->  VM_IPS[2*k + 1]   (vm-server-2k+2)

Per gNB k:
  nr_cellid  = 12345678 + k
  F1-C port  = 500 + 20*k
  F1-U port  = 501 + 20*k
  RFSim port = 4043 + k
  4 UEs per gNB (IMSI sequentially allocated)

Usage:
  python3 generate_conf_multi_gnb.py --n-gnb 2 --out-dir cn/
  python3 generate_conf_multi_gnb.py --n-rus 8 --out-dir cn/ --deploy
"""

import os
import sys
import argparse
import subprocess

# ── topology ──────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import topology as T
    CTL_IP   = T.CTL_IP
    VM_IPS   = T.VM_IPS
    SSH_USER = T.SSH_USER
except ImportError:
    print("ERROR: topology.py not found. Create it with CTL_IP, VM_IPS, SSH_USER.")
    sys.exit(1)

# ── constants ─────────────────────────────────────────────────────────────────
BASE_CELLID  = 12345678
BASE_F1C     = 500          # F1-C (control plane) base port
F1_STEP      = 20           # port offset per gNB
BASE_RFSIM   = 4043         # RFSim server base port
UES_PER_GNB  = 4
OAI_IMAGE    = "oaisoftwarealliance/oai-gnb:2024.w26"
OAI_UE_IMAGE = "oaisoftwarealliance/oai-nr-ue:2024.w26"

# Docker bridge subnet used by OAI core (for UPF route on CU nodes)
UPF_DOCKER_IP = "192.168.70.134"

# ── DU config template ────────────────────────────────────────────────────────
# OAI 5G SA gNB-DU with CU/DU split and RFSim
DU_CONF = """\
Active_gNBs = ( "gNB-DU-{k}" );
Asn1_verbosity = "none";

gNBs = (
{{
  gNB_ID    = {gnb_id};
  gNB_DU_ID = {k};
  cell_type =  "CELL_MACRO_GNB";
  gNB_name  =  "gNB-DU-{k}";

  tracking_area_code = 1;
  plmn_list = ({{
    mcc = 001; mnc = 01; mnc_length = 2;
    snssaiList = ( {{ sst = 1; }} );
  }});

  nr_cellid = {nr_cellid}L;
  min_rxtxtime = 6;

  servingCellConfigCommon = (
  {{
    physCellId                              = {physCellId};

    dl_frequencyBand                        = 78;
    dl_absoluteFrequencySSB                 = 632628;
    dl_offstToCarrier                       = 0;
    dl_subcarrierSpacing                    = 1;
    dl_carrierBandwidth                     = 106;

    ul_frequencyBand                        = 78;
    ul_absoluteFrequencyPointA              = 620000;
    ul_offstToCarrier                       = 0;
    ul_subcarrierSpacing                    = 1;
    ul_carrierBandwidth                     = 106;
    ul_additionalSpectrumEmission           = 0;

    prach_ConfigurationIndex                = 98;
    prach_msg1_FDM                          = 0;
    prach_msg1_FrequencyStart               = 0;
    zeroCorrelationZoneConfig               = 13;
    preambleReceivedTargetPower             = -96;
    preamblTransMax                         = 6;
    powerRampingStep                        = 1;
    ra_ResponseWindow                       = 4;
    ssb_perRACH_OccasionAndCB_PreamblesPerSSB_PR = 4;
    ssb_perRACH_OccasionAndCB_PreamblesPerSSB    = 14;
    ra_ContentionResolutionTimer            = 7;
    rsrp_ThresholdSSB                       = 19;
    prach_RootSequenceIndex_PR              = 2;
    prach_RootSequenceIndex                 = 1;
    msg1_SubcarrierSpacing                  = 1;
    restrictedSetConfig                     = 0;
    msg3_DeltaPreamble                      = 1;
    p0_NominalWithGrant                     = -90;

    pucch_GroupHopping                      = 0;
    p0_nominal                              = -90;
    cyclicPrefix                            = 0;

    ssb_PositionsInBurst_Bitmap             = 1;
    ssb_periodicityServingCell              = 2;
    dmrs_TypeA_Position                     = 0;
    subcarrierSpacing                       = 1;
    referenceSubcarrierSpacing              = 1;

    dl_UL_TransmissionPeriodicity           = 6;
    nrofDownlinkSlots                       = 7;
    nrofDownlinkSymbols                     = 6;
    nrofUplinkSlots                         = 2;
    nrofUplinkSymbols                       = 4;

    ssPBCH_BlockPower                       = -25;
  }}
  );

  # F1 interface — DU side
  local_s_if_name  = "lo";
  remote_s_address = "{cu_ip}";
  local_s_address  = "{du_ip}";
  local_s_portc    = {f1c};
  remote_s_portc   = {f1c};
  local_s_portd    = {f1d};
  remote_s_portd   = {f1d};

  rfsimulator : {{
    serveraddr = "server";
    serverport = {rfsim_port};
    options    = ();
    modelname  = "AWGN";
    IQfile     = "/tmp/rfsimulator.IQ";
  }};

  log_config : {{
    global_log_level = "info";
    hw_log_level     = "info";
    phy_log_level    = "info";
    mac_log_level    = "info";
    rlc_log_level    = "info";
    pdcp_log_level   = "info";
    rrc_log_level    = "info";
    f1ap_log_level   = "info";
  }};
}}
);
"""

# ── CU config template ────────────────────────────────────────────────────────
CU_CONF = """\
Active_gNBs = ( "gNB-CU-{k}" );
Asn1_verbosity = "none";

gNBs = (
{{
  gNB_ID   = {gnb_id};
  cell_type =  "CELL_MACRO_GNB";
  gNB_name  =  "gNB-CU-{k}";

  tracking_area_code = 1;
  plmn_list = ({{
    mcc = 001; mnc = 01; mnc_length = 2;
    snssaiList = ( {{ sst = 1; }} );
  }});

  nr_cellid = {nr_cellid}L;

  # F1 termination — CU listens for incoming DU connections
  local_s_if_name  = "lo";
  local_s_address  = "{cu_ip}";
  local_s_portc    = {f1c};
  local_s_portd    = {f1d};

  # N2/N3 interfaces (toward AMF and UPF on vm-controller)
  NETWORK_INTERFACES : {{
    GNB_INTERFACE_NAME_FOR_NG_AMF = "lo";
    GNB_IPV4_ADDRESS_FOR_NG_AMF   = "{cu_ip}";
    GNB_INTERFACE_NAME_FOR_NGU    = "lo";
    GNB_IPV4_ADDRESS_FOR_NGU      = "{cu_ip}";
    GNB_PORT_FOR_S1U              = 2152;
  }};

  amf_ip_address = ({{
    ipv4       = "{ctl_ip}";
    ipv6       = "192:168:30::17";
    active     = "yes";
    preference = "ipv4";
  }});

  SCTP : {{
    SCTP_INSTREAMS  = 2;
    SCTP_OUTSTREAMS = 2;
  }};

  log_config : {{
    global_log_level = "info";
    hw_log_level     = "info";
    pdcp_log_level   = "info";
    rrc_log_level    = "info";
    ngap_log_level   = "info";
    f1ap_log_level   = "info";
  }};
}}
);
"""

# ── UE config template ────────────────────────────────────────────────────────
UE_CONF = """\
uicc0 = {{
  imsi      = "{imsi}";
  key       = "fec86ba6eb707ed08905757b1bb44b8f";
  opc       = "C42449363BBAD02B66D16BC975D77CC1";
  dnn       = "oai";
  nssai_sst = 1;
}};

rfsimulator : {{
  serveraddr = "{du_ip}";
  serverport = {rfsim_port};
  options    = ();
  modelname  = "AWGN";
  IQfile     = "/tmp/rfsimulator.IQ";
}};

log_config : {{
  global_log_level = "info";
  phy_log_level    = "info";
  mac_log_level    = "info";
  rlc_log_level    = "info";
  nas_log_level    = "info";
  rrc_log_level    = "info";
}};
"""


# ── generators ────────────────────────────────────────────────────────────────

def generate_du_conf(k, out_dir):
    du_ip      = VM_IPS[2 * k]
    cu_ip      = VM_IPS[2 * k + 1]
    nr_cellid  = BASE_CELLID + k
    f1c        = BASE_F1C + F1_STEP * k
    f1d        = f1c + 1
    rfsim_port = BASE_RFSIM + k
    gnb_id     = hex(0xe000 + k)

    content = DU_CONF.format(
        k=k, gnb_id=gnb_id, nr_cellid=nr_cellid,
        physCellId=k,
        du_ip=du_ip, cu_ip=cu_ip,
        f1c=f1c, f1d=f1d, rfsim_port=rfsim_port,
    )
    path = os.path.join(out_dir, f"du.gnb{k}.conf")
    with open(path, "w") as f:
        f.write(content)
    return path, du_ip


def generate_cu_conf(k, out_dir):
    cu_ip      = VM_IPS[2 * k + 1]
    nr_cellid  = BASE_CELLID + k
    f1c        = BASE_F1C + F1_STEP * k
    f1d        = f1c + 1
    gnb_id     = hex(0xe000 + k)

    content = CU_CONF.format(
        k=k, gnb_id=gnb_id, nr_cellid=nr_cellid,
        cu_ip=cu_ip, ctl_ip=CTL_IP,
        f1c=f1c, f1d=f1d,
    )
    path = os.path.join(out_dir, f"cu.gnb{k}.conf")
    with open(path, "w") as f:
        f.write(content)
    return path, cu_ip


def generate_ue_confs(n_gnb, out_dir):
    """Generate one config file per UE and return list of (path, du_ip)."""
    paths = []
    ue_idx = 1
    for k in range(n_gnb):
        du_ip      = VM_IPS[2 * k]
        rfsim_port = BASE_RFSIM + k
        for _ in range(UES_PER_GNB):
            imsi    = f"001010000000{ue_idx:03d}"
            content = UE_CONF.format(
                imsi=imsi, du_ip=du_ip, rfsim_port=rfsim_port
            )
            path = os.path.join(out_dir, f"ue{ue_idx}.conf")
            with open(path, "w") as f:
                f.write(content)
            paths.append((path, ue_idx))
            ue_idx += 1
    return paths


def generate_ue_compose(n_gnb, out_dir):
    """Generate docker-compose-ran-multi.yml for all UE containers."""
    lines = [
        "version: '3.8'",
        "services:",
    ]
    ue_idx = 1
    for k in range(n_gnb):
        du_ip      = VM_IPS[2 * k]
        rfsim_port = BASE_RFSIM + k
        for _ in range(UES_PER_GNB):
            imsi    = f"001010000000{ue_idx:03d}"
            svc     = "oai-nr-ue" if ue_idx == 1 else f"oai-nr-ue{ue_idx}"
            opts    = (
                f"--sa -r 106 --numerology 1 -C 3619200000 "
                f"--ssb 516 --band 78 "
                f"--uicc0.imsi {imsi} "
                f"--uicc0.key fec86ba6eb707ed08905757b1bb44b8f "
                f"--uicc0.opc C42449363BBAD02B66D16BC975D77CC1 "
                f"--uicc0.dnn oai "
                f"--uicc0.nssai_sst 1 "
                f"--rfsimulator.serveraddr {du_ip} "
                f"--rfsimulator.serverport {rfsim_port} "
                f"--log_config.global_log_options level,nocolor,time"
            )
            lines += [
                f"  {svc}:",
                f"    image: {OAI_UE_IMAGE}",
                f"    network_mode: host",
                f"    privileged: true",
                f"    cap_add:",
                f"      - NET_ADMIN",
                f"    restart: unless-stopped",
                f"    environment:",
                f"      USE_ADDITIONAL_OPTIONS: '{opts}'",
                f"    # gNB {k} — IMSI {imsi} — RFSim {du_ip}:{rfsim_port}",
                "",
            ]
            ue_idx += 1

    path = os.path.join(out_dir, "docker-compose-ran-multi.yml")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def generate_deploy_script(n_gnb, out_dir):
    """Generate deploy_multi_gnb.sh — starts CU then DU for each gNB."""
    lines = [
        "#!/bin/bash",
        "# deploy_multi_gnb.sh — start all gNB DU/CU containers",
        f"# Auto-generated for N_GNB={n_gnb}  CTL={CTL_IP}",
        "set -e",
        "",
        f"CTL_IP={CTL_IP}",
        f"SSH_USER={SSH_USER}",
        "",
    ]
    for k in range(n_gnb):
        du_ip   = VM_IPS[2 * k]
        cu_ip   = VM_IPS[2 * k + 1]
        f1c     = BASE_F1C + F1_STEP * k
        rfsim   = BASE_RFSIM + k
        cellid  = BASE_CELLID + k
        lines += [
            f"# ── gNB {k}  DU={du_ip}  CU={cu_ip}  "
            f"cellid={cellid}  F1={f1c}  RFSim={rfsim} ─────────",
            f"echo '==> gNB {k}: adding UPF route on CU node...'",
            f"ssh {SSH_USER}@{cu_ip} 'sudo ip route del {UPF_DOCKER_IP}/32 2>/dev/null || true'",
            f"ssh {SSH_USER}@{cu_ip} 'sudo ip route add {UPF_DOCKER_IP}/32 via {CTL_IP} dev enp6s18'",
            "",
            f"echo '==> gNB {k}: SCP configs...'",
            f"scp {out_dir}/cu.gnb{k}.conf {SSH_USER}@{cu_ip}:~/oai/conf/cu.gnb{k}.conf",
            f"scp {out_dir}/du.gnb{k}.conf {SSH_USER}@{du_ip}:~/oai/conf/du.gnb{k}.conf",
            "",
            f"echo '==> gNB {k}: starting CU on {cu_ip}...'",
            f"ssh {SSH_USER}@{cu_ip} 'mkdir -p ~/oai/conf'",
            f"ssh {SSH_USER}@{cu_ip} '",
            f"  docker rm -f oai-cu-gnb{k} 2>/dev/null || true",
            f"  docker run -d --rm --privileged --network host \\",
            f"    --name oai-cu-gnb{k} \\",
            f"    -v ~/oai/conf/cu.gnb{k}.conf:/opt/oai-gnb/etc/gnb.conf \\",
            f"    {OAI_IMAGE} \\",
            f"    /opt/oai-gnb/bin/nr-softmodem \\",
            f"    -O /opt/oai-gnb/etc/gnb.conf --sa \\",
            f"    --log_config.global_log_options level,nocolor,time",
            f"'",
            f"sleep 20",
            f"ssh {SSH_USER}@{cu_ip} \"docker logs oai-cu-gnb{k} 2>&1 | grep -i 'ng setup\\|amf\\|error' | tail -5\"",
            "",
            f"echo '==> gNB {k}: starting DU on {du_ip}...'",
            f"ssh {SSH_USER}@{du_ip} 'mkdir -p ~/oai/conf'",
            f"ssh {SSH_USER}@{du_ip} '",
            f"  docker rm -f oai-du-gnb{k} 2>/dev/null || true",
            f"  docker run -d --rm --privileged --network host \\",
            f"    --name oai-du-gnb{k} \\",
            f"    -v ~/oai/conf/du.gnb{k}.conf:/opt/oai-gnb/etc/gnb.conf \\",
            f"    {OAI_IMAGE} \\",
            f"    /opt/oai-gnb/bin/nr-softmodem \\",
            f"    -O /opt/oai-gnb/etc/gnb.conf -E --sa --rfsim \\",
            f"    --log_config.global_log_options level,nocolor,time",
            f"'",
            f"sleep 20",
            f"ssh {SSH_USER}@{du_ip} \"docker logs oai-du-gnb{k} 2>&1 | grep -i 'f1 setup\\|error' | tail -5\"",
            "",
        ]
    lines += [
        "echo ''",
        "echo '==> All gNBs deployed. Verify on AMF:'",
        "echo \"docker logs oai-amf 2>&1 | grep -i 'gnb\\|ng setup'\"",
    ]
    path = os.path.join(out_dir, "deploy_multi_gnb.sh")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.chmod(path, 0o755)
    return path


def generate_restore_script(n_gnb, out_dir):
    """Generate restore_network_multi_gnb.sh for daily restart."""
    lines = [
        "#!/bin/bash",
        "# restore_network_multi_gnb.sh — daily restore after testbed restart",
        f"# Auto-generated for N_GNB={n_gnb}",
        "set -e",
        "",
        f"CTL_IP={CTL_IP}",
        f"SSH_USER={SSH_USER}",
        "",
        "echo '==> Step 1: GTP-U tc flower redirect...'",
        "UPF_PID=$(docker inspect oai-upf --format '{{{{.State.Pid}}}}')",
        "UPF_VETH=$(ip link | grep -B1 \"if${UPF_PID}:\" | grep -oP 'veth\\w+' | head -1)",
        "UPF_MAC=$(cat /sys/class/net/${UPF_VETH}/address)",
        "sudo tc qdisc add dev enp6s18 ingress 2>/dev/null || true",
        "sudo tc filter del dev enp6s18 ingress 2>/dev/null || true",
        "sudo tc filter add dev enp6s18 ingress \\",
        "  protocol ip flower ip_proto udp dst_port 2152 \\",
        "  action pedit ex munge eth dst set ${UPF_MAC} pipe \\",
        "  action mirred egress redirect dev ${UPF_VETH}",
        "echo \"  GTP-U redirect: enp6s18 -> ${UPF_VETH} (${UPF_MAC})\"",
        "",
        "echo '==> Step 2: Remove Docker FORWARD DROP rules...'",
        "sudo nft -a list chain ip filter DOCKER 2>/dev/null \\",
        "  | grep 'drop' | grep -oP 'handle \\d+' | awk '{print $2}' \\",
        "  | while read h; do",
        "    sudo nft delete rule ip filter DOCKER handle $h 2>/dev/null && echo \"  Removed DROP handle $h\"",
        "  done",
        "",
        "echo '==> Step 3: Flush stale UE auth state...'",
        "docker exec mysql mysql -u root --password=linux oai_db \\",
        "  -e 'DELETE FROM AuthenticationStatus;' 2>/dev/null || true",
        "docker compose -f cn/docker-compose.yml restart oai-ausf oai-udr oai-udm",
        "sleep 20",
        "",
        "echo '==> Step 4: Deploy gNBs...'",
        f"bash {out_dir}/deploy_multi_gnb.sh",
        "",
        "echo '==> Step 5: Attach UEs (90s gap between each)...'",
        "docker compose -f cn/docker-compose-ran-multi.yml down 2>/dev/null || true",
    ]
    ue_idx = 1
    for k in range(n_gnb):
        for _ in range(UES_PER_GNB):
            svc = "oai-nr-ue" if ue_idx == 1 else f"oai-nr-ue{ue_idx}"
            lines += [
                f"docker compose -f cn/docker-compose-ran-multi.yml up -d {svc}",
                f"sleep 90",
                f"IP=$(docker exec {svc} ip addr show oaitun_ue1 2>/dev/null "
                f"| grep 'inet ' | awk '{{print $2}}' | cut -d/ -f1)",
                f"echo \"  {svc}: ${{IP:-NO IP}}\"",
            ]
            ue_idx += 1
    lines += [
        "",
        "echo '==> Restore complete.'",
    ]
    path = os.path.join(out_dir, "restore_network_multi_gnb.sh")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.chmod(path, 0o755)
    return path


def scp(local_path, remote_ip, remote_path):
    subprocess.run(
        ["ssh", f"{SSH_USER}@{remote_ip}", f"mkdir -p {os.path.dirname(remote_path)}"],
        check=True
    )
    subprocess.run(
        ["scp", local_path, f"{SSH_USER}@{remote_ip}:{remote_path}"],
        check=True
    )
    print(f"    SCP -> {remote_ip}:{remote_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate OAI DU/CU configs for E-MATRO multi-gNB deployment"
    )
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--n-gnb", type=int, help="Number of gNBs")
    grp.add_argument("--n-rus", type=int, help="Number of RUs (auto: n_gnb = n_rus // 4)")
    parser.add_argument("--out-dir", default="cn",
                        help="Output directory (default: cn/)")
    parser.add_argument("--deploy", action="store_true",
                        help="SCP DU/CU configs to vm-servers after generation")
    args = parser.parse_args()

    n_gnb = args.n_gnb if args.n_gnb else args.n_rus // 4
    if n_gnb < 1:
        print("ERROR: n_gnb must be >= 1")
        sys.exit(1)

    needed = 2 * n_gnb
    if needed > len(VM_IPS):
        print(f"ERROR: {n_gnb} gNBs need {needed} vm-servers "
              f"but topology.py only has {len(VM_IPS)}")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\nE-MATRO config generator — N_GNB={n_gnb}  CTL={CTL_IP}")
    print(f"Output: {args.out_dir}/\n")

    scp_list = []
    for k in range(n_gnb):
        du_ip = VM_IPS[2 * k]
        cu_ip = VM_IPS[2 * k + 1]
        print(f"  gNB {k}: DU={du_ip}  CU={cu_ip}  "
              f"cellid={BASE_CELLID+k}  F1={BASE_F1C+F1_STEP*k}  "
              f"RFSim={BASE_RFSIM+k}")

        du_path, _ = generate_du_conf(k, args.out_dir)
        cu_path, _ = generate_cu_conf(k, args.out_dir)
        print(f"    {os.path.basename(du_path)}  {os.path.basename(cu_path)}")
        scp_list.append((du_path, du_ip, f"~/oai/conf/du.gnb{k}.conf"))
        scp_list.append((cu_path, cu_ip, f"~/oai/conf/cu.gnb{k}.conf"))

    ue_paths = generate_ue_confs(n_gnb, args.out_dir)
    print(f"\n  UE configs: {len(ue_paths)} files")

    ue_compose = generate_ue_compose(n_gnb, args.out_dir)
    print(f"  UE compose: {ue_compose}")

    deploy_sh  = generate_deploy_script(n_gnb, args.out_dir)
    restore_sh = generate_restore_script(n_gnb, args.out_dir)
    print(f"  Scripts:    {deploy_sh}")
    print(f"              {restore_sh}")

    if args.deploy:
        print(f"\nDeploying configs via SCP...")
        for local_path, remote_ip, remote_path in scp_list:
            try:
                scp(local_path, remote_ip, remote_path)
            except subprocess.CalledProcessError as e:
                print(f"    WARNING: SCP failed {remote_ip}: {e}")
        print("  Deploy done.")

    print(f"\nNext steps:")
    print(f"  bash {deploy_sh}")
    print(f"  docker compose -f {args.out_dir}/docker-compose-ran-multi.yml up -d oai-nr-ue")


if __name__ == "__main__":
    main()