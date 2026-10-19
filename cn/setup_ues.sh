#!/bin/bash
# =============================================================================
# setup_ues.sh — Configure 4 RUs × 3 UEs = 12 UE containers on vm-controller
#
# Run from: ~/E-matro-evaluation/cn/
# Usage:    bash setup_ues.sh
# =============================================================================
set -euo pipefail

CN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GNB_IP="192.168.70.160"
EXT_DN_IP="192.168.70.135"
DB_PASS="linux"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
ok()  { echo -e "${GREEN}[✓]${NC} $*"; }
warn(){ echo -e "${YELLOW}[!]${NC} $*"; }
err() { echo -e "${RED}[✗]${NC} $*"; }
section() { echo -e "\n${YELLOW}━━━ $* ━━━${NC}"; }

# ── UE → RU mapping ──────────────────────────────────────────────────────────
# RU 0 → UE 1,  2,  3   (IMSI ...101, ...102, ...103)
# RU 1 → UE 4,  5,  6   (IMSI ...104, ...105, ...106)
# RU 2 → UE 7,  8,  9   (IMSI ...107, ...108, ...109)
# RU 3 → UE 10, 11, 12  (IMSI ...110, ...111, ...112)

ue_container() {
    local i=$1
    [[ $i -eq 1 ]] && echo "oai-nr-ue"  && return
    [[ $i -eq 2 ]] && echo "oai-nr-ue2" && return
    echo "oai-nr-ue${i}"
}

ue_imsi() {
    local i=$1
    printf "00101000000010%d" $i
}

ue_ru() {
    local i=$1
    echo $(( (i - 1) / 3 ))
}

# ── Step 1: Verify MySQL is healthy ──────────────────────────────────────────
section "Step 1 — MySQL check"
if docker exec mysql mysqladmin -u root --password=$DB_PASS ping &>/dev/null; then
    ok "MySQL is alive"
else
    err "MySQL not reachable — run: docker compose -f docker-compose.yml restart mysql"
    exit 1
fi

# ── Step 2: Insert all 12 IMSIs ──────────────────────────────────────────────
section "Step 2 — Inserting 12 UE subscribers into DB"

TEMPLATE_IMSI="001010000000101"

for i in $(seq 1 12); do
    IMSI=$(ue_imsi $i)
    COUNT=$(docker exec mysql mysql -u root --password=$DB_PASS oai_db \
        -sNe "SELECT COUNT(*) FROM AuthenticationSubscription WHERE ueid='${IMSI}';" 2>/dev/null)
    if [[ "$COUNT" -eq 0 ]]; then
        docker exec mysql mysql -u root --password=$DB_PASS oai_db 2>/dev/null -e "
        INSERT INTO AuthenticationSubscription
          (ueid, authenticationMethod, encPermanentKey, protectionParameterId,
           sequenceNumber, authenticationManagementField, algorithmId,
           encOpcKey, encTopcKey, vectorGenerationInHss, n5gcAuthMethod,
           rgAuthenticationInd, supi)
        SELECT '${IMSI}', authenticationMethod, encPermanentKey,
           protectionParameterId, sequenceNumber, authenticationManagementField,
           algorithmId, encOpcKey, encTopcKey, vectorGenerationInHss,
           n5gcAuthMethod, rgAuthenticationInd, '${IMSI}'
        FROM AuthenticationSubscription
        WHERE ueid='${TEMPLATE_IMSI}';"
        ok "IMSI ${IMSI} inserted (RU$(ue_ru $i)-UE$i)"
    else
        ok "IMSI ${IMSI} already exists (RU$(ue_ru $i)-UE$i)"
    fi
done

# Verify
COUNT_TOTAL=$(docker exec mysql mysql -u root --password=$DB_PASS oai_db \
    -sNe "SELECT COUNT(*) FROM AuthenticationSubscription;" 2>/dev/null)
ok "Total subscribers in DB: $COUNT_TOTAL"

# ── Step 3: Verify core is healthy ───────────────────────────────────────────
section "Step 3 — Core network health"
UNHEALTHY=$(docker compose -f docker-compose.yml ps -a 2>/dev/null \
    | grep -v "healthy" | grep "oai-\|mysql" | grep -v "Up.*healthy" || true)
if [[ -n "$UNHEALTHY" ]]; then
    warn "Some containers not healthy:"
    echo "$UNHEALTHY"
else
    ok "All core containers healthy"
fi

# ── Step 4: Start gNB ────────────────────────────────────────────────────────
section "Step 4 — gNB"
if ! docker ps --format "{{.Names}}" | grep -q "^oai-gnb$"; then
    docker compose -f docker-compose-ran.yml up -d oai-gnb
    echo -n "  Waiting for gNB to start"
    for _ in $(seq 1 12); do sleep 5; echo -n "."; done
    echo ""
fi
GNB_STATUS=$(docker inspect oai-gnb --format "{{.State.Health.Status}}" 2>/dev/null)
ok "gNB status: $GNB_STATUS"

# ── Step 5: Start UE 1 and 2 (from docker-compose) ───────────────────────────
section "Step 5 — UE 1 & 2 (from docker-compose-ran.yml)"

for UE_SVC in "oai-nr-ue" "oai-nr-ue2"; do
    if ! docker ps --format "{{.Names}}" | grep -q "^${UE_SVC}$"; then
        docker compose -f docker-compose-ran.yml up -d $UE_SVC
        sleep 20
    fi
    STATUS=$(docker inspect $UE_SVC --format "{{.State.Health.Status}}" 2>/dev/null || echo "unknown")
    ok "$UE_SVC status: $STATUS"
done

# ── Step 6: Start UE 3-12 (extra containers) ─────────────────────────────────
section "Step 6 — UE 3–12 (extra containers)"

UE_IMAGE="oaisoftwarealliance/oai-nr-ue:2024-june"

for i in $(seq 3 12); do
    NAME=$(ue_container $i)
    IMSI=$(ue_imsi $i)

    if docker ps -a --format "{{.Names}}" | grep -q "^${NAME}$"; then
        STATUS=$(docker inspect $NAME --format "{{.State.Status}}" 2>/dev/null)
        if [[ "$STATUS" == "running" ]]; then
            ok "$NAME already running (IMSI $IMSI)"
            continue
        else
            docker rm -f $NAME &>/dev/null
        fi
    fi

    docker run -d \
        --name $NAME \
        --network oaiworkshop \
        --cap-add=NET_ADMIN \
        --cap-add=SYS_ADMIN \
        --privileged \
        -e USE_ADDITIONAL_OPTIONS="\
--rfsim \
--rfsimulator.serveraddr ${GNB_IP} \
-r 106 --numerology 1 --band 78 \
-C 3619200000 --nokrnmod \
--uicc0.imsi ${IMSI} \
--uicc0.key fec86ba6eb707ed08905757b1bb44b8f \
--uicc0.opc C42449363BBAD02B66D16BC975D77CC1 \
--log_config.global_log_options level,nocolor,time" \
        $UE_IMAGE &>/dev/null

    echo -n "  Starting $NAME (IMSI $IMSI)"
    for _ in $(seq 1 6); do sleep 5; echo -n "."; done
    echo " done"
    ok "$NAME started (RU$(ue_ru $i))"
done

# ── Step 7: Check tunnel IPs ──────────────────────────────────────────────────
section "Step 7 — Tunnel IP assignment"
echo ""
printf "  %-14s %-6s %-6s %-20s\n" "Container" "RU" "UE#" "Tunnel IP"
echo "  ──────────────────────────────────────────────"

CONNECTED=0
for i in $(seq 1 12); do
    NAME=$(ue_container $i)
    RU=$(ue_ru $i)
    UE_NUM=$(( (i-1) % 3 + 1 ))
    IP=$(docker exec $NAME ip addr show oaitun_ue1 2>/dev/null | grep "inet " | awk '{print $2}' | cut -d/ -f1 || echo "")
    if [[ -n "$IP" ]]; then
        printf "  %-14s %-6s %-6s %-20s\n" "$NAME" "RU${RU}" "UE${UE_NUM}" "$IP"
        CONNECTED=$((CONNECTED + 1))
    else
        printf "  %-14s %-6s %-6s %-20s\n" "$NAME" "RU${RU}" "UE${UE_NUM}" "NOT CONNECTED"
    fi
done

echo ""
ok "$CONNECTED / 12 UEs have tunnel IPs"

# ── Step 8: Start iperf3 server ───────────────────────────────────────────────
section "Step 8 — iperf3 server"
if docker exec oai-ext-dn pgrep -f "iperf3 -s" &>/dev/null; then
    ok "iperf3 server already running on oai-ext-dn"
else
    docker exec -d oai-ext-dn iperf3 -s -p 5201 --daemon \
        --logfile /tmp/iperf3_server.log 2>/dev/null || \
    docker exec -d oai-ext-dn iperf3 -s -p 5201
    sleep 2
    ok "iperf3 server started on oai-ext-dn:5201"
fi

# ── Step 9: Quick traffic test on connected UEs ───────────────────────────────
section "Step 9 — Quick traffic test"
for i in 1 2; do
    NAME=$(ue_container $i)
    IP=$(docker exec $NAME ip addr show oaitun_ue1 2>/dev/null | grep "inet " | awk '{print $2}' | cut -d/ -f1 || echo "")
    if [[ -n "$IP" ]]; then
        echo -n "  Testing $NAME ($IP)... "
        RESULT=$(docker exec $NAME iperf3 -c $EXT_DN_IP -p 5201 -u -b 2M -t 3 -B $IP --get-server-output 2>/dev/null | grep -E "Mbits|sender" | tail -1 || echo "failed")
        echo "$RESULT"
    fi
done

# ── Summary ───────────────────────────────────────────────────────────────────
section "Setup complete"
echo ""
echo "  4 RUs × 3 UEs = 12 UE containers"
echo "  $CONNECTED currently connected with tunnel IPs"
echo ""
echo "  RU 0 → oai-nr-ue    oai-nr-ue2   oai-nr-ue3"
echo "  RU 1 → oai-nr-ue4   oai-nr-ue5   oai-nr-ue6"
echo "  RU 2 → oai-nr-ue7   oai-nr-ue8   oai-nr-ue9"
echo "  RU 3 → oai-nr-ue10  oai-nr-ue11  oai-nr-ue12"
echo ""
echo "  Watch live:  bash ~/watch_ues.sh"
echo "  Start train: cd ~/E-matro-evaluation && python train_online.py --dry-run --steps 5"



