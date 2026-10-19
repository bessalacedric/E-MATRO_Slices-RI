#!/bin/bash
# =============================================================================
# deploy_disaggregated.sh
# Full disaggregated O-RAN deployment on SLICES-RI
#
# Architecture:
#   vm-controller : 5G Core + RFSim RUs + UE containers
#   vm-server-1   : DU node 0 (RU0, RU1)
#   vm-server-2   : DU node 1 (RU2, RU3)
#   vm-server-3   : CU-CP + CU-UP node 2
#   vm-server-4   : CU-CP + CU-UP node 3
#
# Usage: bash deploy_disaggregated.sh
# =============================================================================
set -euo pipefail

# ── IPs — update after each reservation ──────────────────────────────────────
CTL_IP="10.10.212.98"     # vm-controller
DU0_IP="10.10.208.89"     # vm-server-1
DU1_IP="10.10.209.228"    # vm-server-2
CU2_IP="10.10.210.51"     # vm-server-3
CU3_IP="10.10.209.200"    # vm-server-4

CN_DIR="$HOME/E-matro-evaluation/cn"
SSH="ssh -o StrictHostKeyChecking=no"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[✓]${NC} $*"; }
info() { echo -e "${YELLOW}[→]${NC} $*"; }
err()  { echo -e "\033[0;31m[✗]\033[0m $*"; }

# ── Step 1: Enable IP forwarding on vm-controller ────────────────────────────
info "Step 1 — Enabling IP forwarding on vm-controller..."
sudo sysctl -w net.ipv4.ip_forward=1 > /dev/null

# Expose AMF N2 (SCTP 38412) on management interface
# Expose UPF N3 (UDP 2152) on management interface
AMF_DOCKER_IP="192.168.70.132"
UPF_DOCKER_IP="192.168.70.134"

sudo iptables -t nat -D PREROUTING -d $CTL_IP -p sctp --dport 38412 \
    -j DNAT --to-destination $AMF_DOCKER_IP:38412 2>/dev/null || true
sudo iptables -t nat -A PREROUTING -d $CTL_IP -p sctp --dport 38412 \
    -j DNAT --to-destination $AMF_DOCKER_IP:38412

sudo iptables -t nat -D PREROUTING -d $CTL_IP -p udp --dport 2152 \
    -j DNAT --to-destination $UPF_DOCKER_IP:2152 2>/dev/null || true
sudo iptables -t nat -A PREROUTING -d $CTL_IP -p udp --dport 2152 \
    -j DNAT --to-destination $UPF_DOCKER_IP:2152

sudo iptables -t nat -D POSTROUTING -s 10.10.0.0/16 \
    -d 192.168.70.0/26 -j MASQUERADE 2>/dev/null || true
sudo iptables -t nat -A POSTROUTING -s 10.10.0.0/16 \
    -d 192.168.70.0/26 -j MASQUERADE

ok "IP forwarding and port forwarding configured"

# ── Step 2: Install Docker on vm-servers ────────────────────────────────────
info "Step 2 — Installing Docker on vm-servers..."
for IP in $DU0_IP $DU1_IP $CU2_IP $CU3_IP; do
    $SSH ubuntu@$IP "which docker > /dev/null 2>&1 || \
        (curl -fsSL https://get.docker.com | sudo sh && \
         sudo usermod -aG docker ubuntu)" &
done
wait
ok "Docker installed on all vm-servers"

# ── Step 3: Copy config files and compose files to vm-servers ────────────────
info "Step 3 — Copying config files to vm-servers..."

for IP in $DU0_IP $DU1_IP $CU2_IP $CU3_IP; do
    $SSH ubuntu@$IP "mkdir -p ~/oai/conf"
    scp -o StrictHostKeyChecking=no \
        $CN_DIR/conf/du.node0.conf \
        $CN_DIR/conf/du.node1.conf \
        $CN_DIR/conf/cucp.node2.conf \
        $CN_DIR/conf/cucp.node3.conf \
        $CN_DIR/conf/cuup.node2.conf \
        $CN_DIR/conf/cuup.node3.conf \
        ubuntu@$IP:~/oai/conf/
    scp -o StrictHostKeyChecking=no \
        $CN_DIR/docker-compose-du.yml \
        $CN_DIR/docker-compose-cu.yml \
        ubuntu@$IP:~/oai/
done
ok "Config files copied to all vm-servers"

# ── Step 4: Pull images on vm-servers ────────────────────────────────────────
info "Step 4 — Pulling OAI images on vm-servers (background)..."
for IP in $DU0_IP $DU1_IP; do
    $SSH ubuntu@$IP "docker pull oaisoftwarealliance/oai-gnb:2024.w26 -q" &
done
for IP in $CU2_IP $CU3_IP; do
    $SSH ubuntu@$IP \
        "docker pull oaisoftwarealliance/oai-gnb:2024.w26 -q && \
         docker pull oaisoftwarealliance/oai-nr-cuup:2024.w26 -q" &
done
wait
ok "Images pulled on all vm-servers"

# ── Step 5: Deploy CU first (CU must be up before DU connects via F1) ────────
info "Step 5 — Deploying CU-CP + CU-UP on CU nodes..."

$SSH ubuntu@$CU2_IP "cd ~/oai && \
    docker compose -f docker-compose-cu.yml up -d cucp-node2
    sleep 10
    docker compose -f docker-compose-cu.yml up -d cuup-node2"
ok "CU node 2 deployed ($CU2_IP)"

$SSH ubuntu@$CU3_IP "cd ~/oai && \
    docker compose -f docker-compose-cu.yml up -d cucp-node3
    sleep 10
    docker compose -f docker-compose-cu.yml up -d cuup-node3"
ok "CU node 3 deployed ($CU3_IP)"

echo "Waiting 30s for CU to connect to AMF..."
sleep 30

# Check CU-CP N2 connection
info "Checking CU-CP → AMF connections..."
docker logs oai-amf 2>&1 | grep -A3 "gNBs' Information" | tail -5

# ── Step 6: Deploy DUs ───────────────────────────────────────────────────────
info "Step 6 — Deploying DU on DU nodes..."

$SSH ubuntu@$DU0_IP "cd ~/oai && \
    docker compose -f docker-compose-du.yml up -d oai-du-node0"
ok "DU node 0 deployed ($DU0_IP)"

sleep 5

$SSH ubuntu@$DU1_IP "cd ~/oai && \
    docker compose -f docker-compose-du.yml up -d oai-du-node1"
ok "DU node 1 deployed ($DU1_IP)"

echo "Waiting 30s for DU → CU F1 connection..."
sleep 30

# ── Step 7: Update UE configs to point to DUs ────────────────────────────────
info "Step 7 — Updating UE rfsimulator addresses..."

cd $CN_DIR

# UE1, UE2 → DU node 0 (vm-server-1)
python3 - << PYEOF
import pathlib, re

for ue in [1, 2]:
    conf = pathlib.Path(f"conf/ue{ue}.conf")
    if conf.exists():
        print(f"  ue{ue}.conf already exists")
    # The rfsimulator address is set via USE_ADDITIONAL_OPTIONS
    # in docker-compose-ran.yml, so update that instead

# Update docker-compose-ran.yml
p   = pathlib.Path("docker-compose-ran.yml")
src = p.read_text()

# UE1 and UE2 → DU0 (10.10.208.89)
src = src.replace(
    "--rfsimulator.serveraddr 192.168.70.160",
    "--rfsimulator.serveraddr $DU0_IP"
)
# After split we update per-UE below

print("docker-compose-ran.yml: will be updated per UE")
PYEOF

# Patch docker-compose-ran.yml: UE1+UE2 → DU0, UE3+UE4 → DU1
python3 - << PYEOF
import pathlib, re

p   = pathlib.Path("docker-compose-ran.yml")
src = p.read_text()

# UE1 and UE2 connect to DU node 0
for ue_svc in ["oai-nr-ue:", "oai-nr-ue2:"]:
    # Find the USE_ADDITIONAL_OPTIONS for this UE and replace serveraddr
    pass  # complex to do with regex — use sed instead
PYEOF

sed -i "s/--rfsimulator.serveraddr 192.168.70.160/--rfsimulator.serveraddr $DU0_IP/g" \
    docker-compose-ran.yml

# UE3+UE4 need DU1 address — patch their specific sections
python3 - << PYEOF
import pathlib

p   = pathlib.Path("docker-compose-ran.yml")
src = p.read_text()

# For UE3 and UE4 sections, replace DU0 with DU1
import re

def fix_ue_section(src, ue_service, old_addr, new_addr):
    pattern = rf'(container_name:\s*{re.escape(ue_service)}.*?)(--rfsimulator\.serveraddr\s+){re.escape(old_addr)}'
    return re.sub(pattern, rf'\1\g<2>{new_addr}', src, flags=re.DOTALL)

src = fix_ue_section(src, "oai-nr-ue3", "$DU0_IP", "$DU1_IP")
src = fix_ue_section(src, "oai-nr-ue4", "$DU0_IP", "$DU1_IP")
p.write_text(src)
print("  UE3+UE4 → DU1 address set")
PYEOF

ok "UE configs updated"

# ── Step 8: Restart UEs ───────────────────────────────────────────────────────
info "Step 8 — Starting UEs..."
docker compose -f docker-compose-ran.yml down 2>/dev/null || true
sleep 5

for SVC in oai-nr-ue oai-nr-ue2 oai-nr-ue3 oai-nr-ue4; do
    docker compose -f docker-compose-ran.yml up -d $SVC
    echo -n "  $SVC... "; sleep 60
    IP=$(docker exec $SVC ip addr show oaitun_ue1 2>/dev/null \
        | grep "inet " | awk '{print $2}' | cut -d/ -f1 || echo "")
    [[ -n "$IP" ]] && echo "$IP ✓" || echo "no IP yet"
done

# ── Step 9: Status report ─────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Disaggregated O-RAN Status"
echo "═══════════════════════════════════════════════════════"

echo ""
echo "  CU nodes:"
for IP_ROLE in "$CU2_IP:node2" "$CU3_IP:node3"; do
    IP="${IP_ROLE%%:*}"; ROLE="${IP_ROLE##*:}"
    CUCP=$($SSH ubuntu@$IP "docker ps --format '{{.Names}}\t{{.Status}}' \
        | grep cucp" 2>/dev/null || echo "not running")
    CUUP=$($SSH ubuntu@$IP "docker ps --format '{{.Names}}\t{{.Status}}' \
        | grep cuup" 2>/dev/null || echo "not running")
    echo "  vm-server $ROLE ($IP):"
    echo "    CU-CP: $CUCP"
    echo "    CU-UP: $CUUP"
done

echo ""
echo "  DU nodes:"
for IP_ROLE in "$DU0_IP:node0" "$DU1_IP:node1"; do
    IP="${IP_ROLE%%:*}"; ROLE="${IP_ROLE##*:}"
    DU=$($SSH ubuntu@$IP "docker ps --format '{{.Names}}\t{{.Status}}' \
        | grep du" 2>/dev/null || echo "not running")
    echo "  vm-server $ROLE ($IP): $DU"
done

echo ""
echo "  AMF — connected gNBs:"
docker logs oai-amf 2>&1 | grep -A5 "gNBs' Information" | tail -6

echo ""
echo "  UE tunnel IPs:"
for i in 1 2 3 4; do
    [[ $i -eq 1 ]] && NAME="oai-nr-ue"  || NAME="oai-nr-ue${i}"
    [[ $i -eq 2 ]] && NAME="oai-nr-ue2"
    IP=$(docker exec $NAME ip addr show oaitun_ue1 2>/dev/null \
        | grep "inet " | awk '{print $2}' | cut -d/ -f1 || echo "NO IP")
    DU_TARGET=$([[ $i -le 2 ]] && echo "$DU0_IP" || echo "$DU1_IP")
    printf "  RU%d  %-14s → DU(%s)  tunnel: %s\n" \
        $((i-1)) "$NAME" "$DU_TARGET" "$IP"
done
echo "═══════════════════════════════════════════════════════"
