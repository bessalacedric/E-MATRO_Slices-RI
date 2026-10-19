#!/bin/bash
# =============================================================================
# deploy_multi_gnb.sh — Deploy multi-gNB scenario to SLICES-RI
#
# Architecture: N_GNB independent gNBs, each with 4 UEs
#   Scenario A: 1 gNB  × 4 UEs = 4 UEs  (baseline)
#   Scenario B: 2 gNBs × 4 UEs = 8 UEs
#   Scenario C: 4 gNBs × 4 UEs = 16 UEs
#   Scenario D: 8 gNBs × 4 UEs = 32 UEs
#
# Usage: bash deploy_multi_gnb.sh --scenario B
#        bash deploy_multi_gnb.sh --n-gnb 4
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

# Parse args
N_GNB=2
SCENARIO=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --scenario) SCENARIO=$2; shift 2 ;;
        --n-gnb)    N_GNB=$2;    shift 2 ;;
        --dry-run)  DRY_RUN=true; shift  ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Map scenario letter to N_GNB
case $SCENARIO in
    A) N_GNB=1  ;;
    B) N_GNB=2  ;;
    C) N_GNB=4  ;;
    D) N_GNB=8  ;;
    E) N_GNB=15 ;;
    "") ;;  # use --n-gnb directly
    *) echo "Unknown scenario $SCENARIO. Use A/B/C/D/E or --n-gnb"; exit 1 ;;
esac

N_UES=$((N_GNB * 4))
N_DUS=$N_GNB
N_CUS=$(( (N_GNB + 1) / 2 ))
N_NODES=$((N_DUS + N_CUS))

# VM IPs in order
VM_IPS=(
    "10.10.208.109"   # vm-server-1
    "10.10.210.195"   # vm-server-2
    "10.10.211.225"   # vm-server-3
    "10.10.211.11"    # vm-server-4
    "10.10.208.178"   # vm-server-5
    "10.10.209.189"   # vm-server-6
    "10.10.210.208"   # vm-server-7
    "10.10.214.191"   # vm-server-8
    "10.10.215.173"   # vm-server-9
    "10.10.210.73"    # vm-server-10
    "10.10.209.31"    # vm-server-11
    "10.10.209.143"   # vm-server-12
    "10.10.211.241"   # vm-server-13
    "10.10.211.195"   # vm-server-14
    "10.10.215.211"   # vm-server-15
)

echo "════════════════════════════════════════════════════════════"
echo "  Multi-gNB Deploy — Scenario ${SCENARIO:-custom}"
echo "  N_GNB=${N_GNB}  N_UES=${N_UES}  N_DUS=${N_DUS}  N_CUS=${N_CUS}  N_NODES=${N_NODES}"
echo "════════════════════════════════════════════════════════════"

if [[ $DRY_RUN == true ]]; then
    echo "  [DRY RUN — no changes applied]"
fi

# =============================================================================
# Step 1 — Generate conf files
# =============================================================================
info "Step 1 — Generating conf files for ${N_GNB} gNBs..."
if [[ $DRY_RUN == false ]]; then
    cd $BASE_DIR
    python3 generate_conf_multi_gnb.py --n-gnb $N_GNB --out-dir $CN_DIR
fi
ok "Conf files generated"

# =============================================================================
# Step 2 — Start 5G core (vm-controller only — same for all scenarios)
# =============================================================================
info "Step 2 — Starting 5G core on vm-controller..."
if [[ $DRY_RUN == false ]]; then
    cd $CN_DIR
    docker compose -f docker-compose.yml down 2>/dev/null || true
    sleep 3
    docker compose -f docker-compose.yml up -d
    echo -n "  Waiting 60s for core to stabilize..."
    sleep 60
    echo " done"
    python3 $BASE_DIR/provision_db.py
fi
ok "5G core running"

# =============================================================================
# Step 3 — Setup tc redirect (GTP-U path)
# =============================================================================
info "Step 3 — Setting up GTP-U tc redirect..."
if [[ $DRY_RUN == false ]]; then
    bash $BASE_DIR/setup_tc.sh
fi
ok "GTP-U redirect configured"

# =============================================================================
# Step 4 — Start CU containers
# =============================================================================
info "Step 4 — Starting ${N_CUS} CU containers..."
for ((cu_idx=0; cu_idx<N_CUS; cu_idx++)); do
    NODE_ID=$((N_DUS + cu_idx))
    VM_IDX=$((NODE_ID % ${#VM_IPS[@]}))
    CU_IP="${VM_IPS[$VM_IDX]}"

    echo "  CU${cu_idx} → ${CU_IP}"
    if [[ $DRY_RUN == false ]]; then
        # Add route to UPF on CU node
        $SSH ${SSH_USER}@${CU_IP} "
            sudo ip route del 192.168.70.134/32 2>/dev/null || true
            sudo ip route add 192.168.70.134/32 via ${CTL_IP} dev enp6s18
            sudo ip route flush cache
            cd ~/oai
            docker compose -f docker-compose-cu.yml up -d oai-cu-gnb${cu_idx} 2>/dev/null \
                || docker run -d --rm --privileged \
                    --name oai-cu-gnb${cu_idx} \
                    --network host \
                    -v ~/oai/conf/cu.gnb${cu_idx}.conf:/opt/oai-gnb/etc/gnb.conf \
                    oaisoftwarealliance/oai-gnb:2024.w26 \
                    /opt/oai-gnb/bin/nr-softmodem -O /opt/oai-gnb/etc/gnb.conf \
                    --sa --log_config.global_log_options level,nocolor,time
        "
    fi
done
ok "CU containers started"

# Delete Docker DROP rules that block N2/N3
info "  Removing Docker nft DROP rules..."
if [[ $DRY_RUN == false ]]; then
    sudo nft -a list chain ip filter DOCKER 2>/dev/null | grep "drop" | \
        grep -oP "handle \d+" | awk '{print $2}' | while read h; do
        sudo nft delete rule ip filter DOCKER handle $h 2>/dev/null && echo "  Deleted handle $h"
    done
fi

# Wait for CUs to register to AMF
sleep 30
if [[ $DRY_RUN == false ]]; then
    echo "  Checking CU→AMF N2 sessions:"
    docker logs oai-amf 2>&1 | grep "gNBs' Information" -A $((N_CUS+3)) | tail -$((N_CUS+5))
fi

# =============================================================================
# Step 5 — Start DU containers
# =============================================================================
info "Step 5 — Starting ${N_DUS} DU containers..."
for ((gnb_id=0; gnb_id<N_GNB; gnb_id++)); do
    VM_IDX=$((gnb_id % ${#VM_IPS[@]}))
    DU_IP="${VM_IPS[$VM_IDX]}"

    echo "  DU(gNB${gnb_id}) → ${DU_IP}"
    if [[ $DRY_RUN == false ]]; then
        $SSH ${SSH_USER}@${DU_IP} "
            cd ~/oai
            docker compose -f docker-compose-du.yml up -d oai-du-gnb${gnb_id} 2>/dev/null \
                || docker run -d --rm --privileged \
                    --name oai-du-gnb${gnb_id} \
                    --network host \
                    -v ~/oai/conf/du.gnb${gnb_id}.conf:/opt/oai-gnb/etc/gnb.conf \
                    oaisoftwarealliance/oai-gnb:2024.w26 \
                    /opt/oai-gnb/bin/nr-softmodem -O /opt/oai-gnb/etc/gnb.conf \
                    -E --sa --rfsim \
                    --log_config.global_log_options level,nocolor,time
        "
    fi
done
ok "DU containers started"

sleep 20

# =============================================================================
# Step 6 — Start UE containers (4 per gNB, with gaps)
# =============================================================================
info "Step 6 — Starting ${N_UES} UE containers..."
if [[ $DRY_RUN == false ]]; then
    docker compose -f docker-compose.yml restart oai-ausf oai-udr oai-udm
    sleep 20

    docker exec mysql mysql -u root --password=linux oai_db -e \
        "DELETE FROM AuthenticationStatus;" 2>/dev/null || true
fi

for ((gnb_id=0; gnb_id<N_GNB; gnb_id++)); do
    echo "  Starting UEs for gNB${gnb_id}..."
    for ((ue_in_gnb=0; ue_in_gnb<4; ue_in_gnb++)); do
        UE_IDX=$((gnb_id * 4 + ue_in_gnb))
        if [[ $UE_IDX -eq 0 ]]; then
            UE_NAME="oai-nr-ue"
        else
            UE_NAME="oai-nr-ue$((UE_IDX+1))"
        fi

        echo -n "    ${UE_NAME}... "
        if [[ $DRY_RUN == false ]]; then
            docker stop $UE_NAME 2>/dev/null || true
            docker rm   $UE_NAME 2>/dev/null || true
            docker compose -f docker-compose-ran-multi.yml up -d $UE_NAME
            sleep 90
            IP=$(docker exec $UE_NAME ip addr show oaitun_ue1 2>/dev/null \
                | grep "inet " | awk '{print $2}' | cut -d/ -f1)
            [[ -n "$IP" ]] && echo "$IP ✓" || echo "NO IP ✗"
        else
            echo "(dry run)"
        fi
    done
done

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Deploy complete — Multi-gNB Scenario ${SCENARIO:-N_GNB=$N_GNB}"
echo "  ${N_GNB} gNBs × 4 UEs = ${N_UES} UEs total"
echo ""
echo "  Train with this scenario:"
echo "    python3 train.py --n-gnb ${N_GNB} --topology multi_gnb \\"
echo "        --episodes 5000 --max-steps 10"
echo ""
echo "  Connectivity test:"
echo "    bash test_vnf_connectivity.sh"
echo "════════════════════════════════════════════════════════════"