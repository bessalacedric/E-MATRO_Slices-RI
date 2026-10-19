#!/bin/bash
# =============================================================================
# restore_network.sh — Full restart sequence for BRAVE-O on SLICES-RI
# Run this every morning or after any crash/reboot
# Usage: bash ~/E-matro-evaluation/restore_network.sh
# =============================================================================
set -euo pipefail

CN_DIR="$HOME/E-matro-evaluation/cn"
LOG="$HOME/E-matro-evaluation/restore.log"
SSH="ssh -o StrictHostKeyChecking=no"

CU2_IP="10.10.211.225"
CU3_IP="10.10.211.11"
DU0_IP="10.10.208.109"
DU1_IP="10.10.210.195"
CTL_IP="10.10.208.188"
UPF_DOCKER_IP="192.168.70.134"
UPF_GW="192.168.70.129"
EXT_IF="enp6s18"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[✓]${NC} $*" | tee -a $LOG; }
info() { echo -e "${YELLOW}[→]${NC} $*" | tee -a $LOG; }

echo "" >> $LOG
echo "======================================" >> $LOG
echo "$(date) — restore_network.sh started" >> $LOG

cd $CN_DIR

# ── Step 1: Stop everything cleanly ──────────────────────────────────────────
info "Step 1 — Stopping all containers..."
docker compose -f docker-compose-ran.yml down 2>/dev/null || true
docker compose -f docker-compose.yml down 2>/dev/null || true
sleep 5
ok "Containers stopped"

# ── Step 2: Start core (core must come before ran) ────────────────────────────
info "Step 2 — Starting 5G Core..."
docker compose -f docker-compose.yml up -d
info "Waiting 60s for core to initialize..."
sleep 60

# Wait for mysql healthy
for i in $(seq 1 12); do
    docker exec mysql mysqladmin -u root --password=linux ping &>/dev/null && break
    sleep 5
done
ok "5G Core started"

# ── Step 3: Load and provision database ──────────────────────────────────────
info "Step 3 — Loading database..."
chmod +r $CN_DIR/database/oai_db.sql
docker exec -i mysql mysql -u root --password=linux oai_db \
    < $CN_DIR/database/oai_db.sql 2>/dev/null
python3 $HOME/E-matro-evaluation/provision_db.py
ok "Database provisioned"

# ── Step 4: Fix container routes ─────────────────────────────────────────────
info "Step 4 — Fixing routes..."
docker exec oai-upf ip route add 10.10.0.0/16 via $UPF_GW 2>/dev/null || true
docker exec oai-ext-dn ip route del 10.0.0.0/16 2>/dev/null || true
docker exec oai-ext-dn ip route add 10.0.0.0/16 via $UPF_DOCKER_IP dev eth0
ok "Routes fixed"

# ── Step 5: tc GTP-U redirect ────────────────────────────────────────────────
info "Step 5 — Setting up tc redirect..."
bash $HOME/E-matro-evaluation/setup_tc.sh
ok "tc redirect done"

# ── Step 6: CU routes to UPF ─────────────────────────────────────────────────
info "Step 6 — Fixing CU routes..."
for CU_IP in $CU2_IP $CU3_IP; do
    $SSH ubuntu@$CU_IP "
        sudo ip route del $UPF_DOCKER_IP/32 2>/dev/null || true
        sudo ip route add $UPF_DOCKER_IP/32 via $CTL_IP dev $EXT_IF
        sudo ip route flush cache && echo $CU_IP OK"
done
ok "CU routes fixed"

# ── Step 7: Delete Docker DROP rules ─────────────────────────────────────────
info "Step 7 — Clearing Docker DROP rules..."
sudo nft -a list chain ip filter DOCKER 2>/dev/null | grep "drop" | \
    grep -oP "handle \d+" | awk '{print $2}' | while read h; do
    sudo nft delete rule ip filter DOCKER handle $h 2>/dev/null \
        && echo "  Deleted handle $h"
done
ok "DROP rules cleared"

# ── Step 8: Restart CUs then DUs ─────────────────────────────────────────────
info "Step 8 — Restarting CUs..."
$SSH ubuntu@$CU2_IP "cd ~/oai && docker restart oai-cu-node2"
$SSH ubuntu@$CU3_IP "cd ~/oai && docker restart oai-cu-node3"
sleep 30

info "Step 9 — Restarting DUs..."
$SSH ubuntu@$DU0_IP "cd ~/oai && docker restart oai-du-node0"
$SSH ubuntu@$DU1_IP "cd ~/oai && docker restart oai-du-node1"
sleep 20

docker logs oai-amf 2>&1 | grep -A5 "gNBs' Information" | tail -7

# ── Step 10: Restart auth NFs to clear cache ─────────────────────────────────
info "Step 10 — Restarting auth NFs..."
docker compose -f docker-compose.yml restart oai-ausf oai-udr oai-udm
sleep 20
ok "Auth NFs restarted"

# ── Step 11: Start UEs one by one ────────────────────────────────────────────
info "Step 11 — Starting UEs..."
declare -A UE_NAMES=([1]="oai-nr-ue" [2]="oai-nr-ue2" [3]="oai-nr-ue3" [4]="oai-nr-ue4")

for i in 1 2 3 4; do
    NAME=${UE_NAMES[$i]}
    docker stop $NAME && docker rm $NAME 2>/dev/null || true
    docker compose -f docker-compose-ran.yml up -d $NAME
    echo -n "  $NAME... "
    sleep 90
    IP=$(docker exec $NAME ip addr show oaitun_ue1 2>/dev/null \
        | grep "inet " | awk '{print $2}' | cut -d/ -f1)
    if [[ -n "$IP" ]]; then
        echo "$IP ✓"
    else
        echo "NO IP — clearing auth and retrying..."
        docker exec mysql mysql -u root --password=linux oai_db -e "
            DELETE FROM AuthenticationStatus;
            DELETE FROM Amf3GppAccessRegistration;" 2>/dev/null
        docker compose -f docker-compose-ran.yml restart $NAME
        sleep 90
        IP=$(docker exec $NAME ip addr show oaitun_ue1 2>/dev/null \
            | grep "inet " | awk '{print $2}' | cut -d/ -f1)
        [[ -n "$IP" ]] && echo "$IP ✓ (retry)" || echo "FAILED ✗"
    fi
done

ok "All UEs started"

# ── Final check ───────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════"
echo "  Quick E2E check:"
for i in 1 2 3 4; do
    NAME=${UE_NAMES[$i]}
    IP=$(docker exec $NAME ip addr show oaitun_ue1 2>/dev/null \
        | grep "inet " | awk '{print $2}' | cut -d/ -f1)
    LOSS=$(docker exec $NAME ping -I oaitun_ue1 192.168.70.135 -c2 -q 2>/dev/null \
        | grep -oP '\d+(?=% packet loss)')
    [[ "$LOSS" == "0" ]] \
        && printf "  UE%d %s ✓\n" $i "$IP" \
        || printf "  UE%d %s ✗\n" $i "${IP:-NO IP}"
done
echo "══════════════════════════════════════"
echo "$(date) — restore complete" >> $LOG
echo ""
echo "Run: bash ~/E-matro-evaluation/vnf_connectivity.sh"