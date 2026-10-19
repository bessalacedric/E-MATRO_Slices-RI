#!/bin/bash
# =============================================================================
# restore_network_multi_gnb.sh — Daily restore for multi-gNB scenario
#
# Run every morning when the testbed loses state (tc rules, DHCP, etc.)
# Keeps the same scenario as last deploy (reads N_GNB from conf files)
#
# Usage:
#   bash restore_network_multi_gnb.sh           # auto-detect N_GNB
#   bash restore_network_multi_gnb.sh --n-gnb 4
#   bash restore_network_multi_gnb.sh --scenario B
# =============================================================================
set -euo pipefail

BASE_DIR="$HOME/E-matro-evaluation"
CN_DIR="$BASE_DIR/cn"
SSH="ssh -o StrictHostKeyChecking=no"
CTL_IP="10.10.208.188"
SSH_USER="ubuntu"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[✓]${NC} $*"; }
info() { echo -e "${YELLOW}[→]${NC} $*"; }
err()  { echo -e "${RED}[✗]${NC} $*"; }
now_ms() { date +%s%3N; }

VM_IPS=(
    "10.10.208.109" "10.10.210.195" "10.10.211.225" "10.10.211.11"
    "10.10.208.178" "10.10.209.189" "10.10.210.208" "10.10.214.191"
    "10.10.215.173" "10.10.210.73"  "10.10.209.31"  "10.10.209.143"
    "10.10.211.241" "10.10.211.195" "10.10.215.211"
)

# Parse args
N_GNB=""
SCENARIO=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --scenario) SCENARIO=$2; shift 2 ;;
        --n-gnb)    N_GNB=$2;    shift 2 ;;
        *) shift ;;
    esac
done

# Map scenario
case $SCENARIO in
    A) N_GNB=1 ;; B) N_GNB=2 ;; C) N_GNB=4 ;;
    D) N_GNB=8 ;; E) N_GNB=15 ;;
esac

# Auto-detect from conf files if not specified
if [[ -z "$N_GNB" ]]; then
    N_GNB=$(ls $CN_DIR/conf/du.gnb*.conf 2>/dev/null | wc -l)
    if [[ $N_GNB -eq 0 ]]; then
        echo "ERROR: no du.gnb*.conf found and --n-gnb not specified"
        echo "Run: bash deploy_multi_gnb.sh --scenario B"
        exit 1
    fi
    echo "  Auto-detected N_GNB=$N_GNB from conf files"
fi

N_UES=$((N_GNB * 4))
N_DUS=$N_GNB
N_CUS=$(( (N_GNB + 1) / 2 ))

T0=$(now_ms)

echo "════════════════════════════════════════════════════════════"
echo "  Multi-gNB Network Restore — $(date '+%Y-%m-%d %H:%M:%S')"
echo "  N_GNB=${N_GNB}  N_UES=${N_UES}  N_DUS=${N_DUS}  N_CUS=${N_CUS}"
echo "════════════════════════════════════════════════════════════"

# =============================================================================
# 1. Restart 5G core
# =============================================================================
info "1 — Restarting 5G core..."
cd $CN_DIR
docker compose -f docker-compose.yml down 2>/dev/null || true
sleep 5
docker compose -f docker-compose.yml up -d
echo -n "  Waiting 60s..."
sleep 60
echo " done"
ok "Core up"

# =============================================================================
# 2. UPF and ext-dn routes
# =============================================================================
info "2 — Fixing UPF/ext-dn routes..."
docker exec oai-upf ip route del 10.10.0.0/16  2>/dev/null || true
docker exec oai-upf ip route add 10.10.0.0/16 via 192.168.70.129
docker exec oai-ext-dn ip route del 10.0.0.0/16 2>/dev/null || true
docker exec oai-ext-dn ip route add 10.0.0.0/16 via 192.168.70.134 dev eth0
ok "UPF/ext-dn routes fixed"

# =============================================================================
# 3. tc GTP-U redirect
# =============================================================================
info "3 — Setting up GTP-U tc redirect..."
bash $BASE_DIR/setup_tc.sh
ok "GTP-U redirect active"

# =============================================================================
# 4. CU routes + restart CUs
# =============================================================================
info "4 — Restarting CU containers (${N_CUS})..."
for ((cu_idx=0; cu_idx<N_CUS; cu_idx++)); do
    NODE_ID=$((N_DUS + cu_idx))
    VM_IDX=$((NODE_ID % ${#VM_IPS[@]}))
    CU_IP="${VM_IPS[$VM_IDX]}"

    echo -n "  CU${cu_idx} (${CU_IP}): "
    $SSH ${SSH_USER}@${CU_IP} "
        sudo ip route del 192.168.70.134/32 2>/dev/null || true
        sudo ip route add 192.168.70.134/32 via ${CTL_IP} dev enp6s18
        sudo ip route flush cache
        docker restart oai-cu-gnb${cu_idx} 2>/dev/null \
            || echo 'container not found — run deploy first'
    " && echo "restarted ✓" || echo "FAILED ✗"
done

# Delete DROP rules
sudo nft -a list chain ip filter DOCKER 2>/dev/null | grep "drop" | \
    grep -oP "handle \d+" | awk '{print $2}' | while read h; do
    sudo nft delete rule ip filter DOCKER handle $h 2>/dev/null && echo "  Deleted nft handle $h"
done

sleep 30
echo "  AMF gNB registrations:"
docker logs oai-amf 2>&1 | grep "gNBs' Information" -A $((N_CUS+3)) | tail -$((N_CUS+5)) || true
ok "CUs restarted"

# =============================================================================
# 5. Restart DUs
# =============================================================================
info "5 — Restarting DU containers (${N_DUS})..."
for ((gnb_id=0; gnb_id<N_GNB; gnb_id++)); do
    VM_IDX=$((gnb_id % ${#VM_IPS[@]}))
    DU_IP="${VM_IPS[$VM_IDX]}"
    echo -n "  DU(gNB${gnb_id}) (${DU_IP}): "
    $SSH ${SSH_USER}@${DU_IP} "docker restart oai-du-gnb${gnb_id}" \
        && echo "restarted ✓" || echo "FAILED ✗"
done
sleep 20
ok "DUs restarted"

# =============================================================================
# 6. DB + auth NFs
# =============================================================================
info "6 — Resetting auth state + restarting NFs..."
python3 $BASE_DIR/provision_db.py
docker exec mysql mysql -u root --password=linux oai_db -e \
    "DELETE FROM AuthenticationStatus;" 2>/dev/null || true
docker compose -f docker-compose.yml restart oai-ausf oai-udr oai-udm
sleep 20
ok "Auth NFs ready"

# =============================================================================
# 7. Start UEs (grouped by gNB, 4 at a time)
# =============================================================================
info "7 — Starting ${N_UES} UE containers..."
for ((gnb_id=0; gnb_id<N_GNB; gnb_id++)); do
    echo "  gNB${gnb_id} UEs:"
    for ((ue_in_gnb=0; ue_in_gnb<4; ue_in_gnb++)); do
        UE_IDX=$((gnb_id * 4 + ue_in_gnb))
        if [[ $UE_IDX -eq 0 ]]; then
            UE_NAME="oai-nr-ue"
        else
            UE_NAME="oai-nr-ue$((UE_IDX+1))"
        fi
        echo -n "    ${UE_NAME}... "
        docker stop $UE_NAME 2>/dev/null || true
        docker rm   $UE_NAME 2>/dev/null || true
        docker compose -f docker-compose-ran-multi.yml up -d $UE_NAME
        sleep 90
        IP=$(docker exec $UE_NAME ip addr show oaitun_ue1 2>/dev/null \
            | grep "inet " | awk '{print $2}' | cut -d/ -f1)
        [[ -n "$IP" ]] && echo "$IP ✓" || echo "NO IP ✗"
    done
done

# =============================================================================
# 8. E2E verification
# =============================================================================
info "8 — E2E ping check..."
PASS=0; FAIL=0
for ((ue_idx=0; ue_idx<N_UES; ue_idx++)); do
    if [[ $ue_idx -eq 0 ]]; then
        NAME="oai-nr-ue"
    else
        NAME="oai-nr-ue$((ue_idx+1))"
    fi
    IP=$(docker exec $NAME ip addr show oaitun_ue1 2>/dev/null \
        | grep "inet " | awk '{print $2}' | cut -d/ -f1)
    if [[ -z "$IP" ]]; then
        echo "  $NAME: NO IP ✗"
        ((FAIL++))
        continue
    fi
    if docker exec $NAME ping -I oaitun_ue1 192.168.70.135 -c2 -q \
            &>/dev/null 2>&1; then
        echo "  $NAME ($IP): E2E ✓"
        ((PASS++))
    else
        echo "  $NAME ($IP): ping FAILED ✗"
        ((FAIL++))
    fi
done

T1=$(now_ms)

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Restore complete in $(( (T1-T0)/1000 ))s"
echo "  E2E: ${PASS}/${N_UES} UEs reachable"
if [[ $FAIL -gt 0 ]]; then
    echo "  ${FAIL} UE(s) failed — check setup_tc.sh and CU routes"
fi
echo "════════════════════════════════════════════════════════════"