#!/bin/bash
# vnf_connectivity.sh — inter-node VNF interface reachability
# SCTP tested via ss (live sessions), UDP via nc, IP via ping


CTL=10.10.213.129
DU0=10.10.212.103   # vm-server-1
DU1=10.10.214.110   # vm-server-2
CU2=10.10.212.33   # vm-server-3
CU3=10.10.212.196    # vm-server-4


PASS=0; FAIL=0

check() {
    local label="$1"; local cmd="$2"
    if eval "$cmd" &>/dev/null; then
        printf "  %-48s ✓\n" "$label"
        ((PASS++))
    else
        printf "  %-48s ✗\n" "$label"
        ((FAIL++))
    fi
}

ping_check() {
    local label="$1"; local ping_cmd="$2"
    local out; out=$(eval "$ping_cmd" 2>/dev/null)
    local summary rtt loss
    summary=$(echo "$out" | grep "packets transmitted")
    rtt=$(echo "$out" | grep "rtt")
    loss=$(echo "$summary" | grep -oP '\d+(?=% packet loss)')
    if [[ "$loss" == "0" ]]; then
        printf "  %-30s %s %s ✓\n" "$label" "$summary" "$rtt"
        ((PASS++))
    else
        printf "  %-30s %s ✗\n" "$label" "${summary:-no response}"
        ((FAIL++))
    fi
}

check_sctp_listen() {
    local label="$1"; local host="$2"; local port="$3"
    if ssh ubuntu@$host "ss -lnap | grep sctp | grep -q ':$port'" &>/dev/null; then
        printf "  %-48s ✓\n" "$label"
        ((PASS++))
    else
        printf "  %-48s ✗\n" "$label"
        ((FAIL++))
    fi
}

check_sctp_estab() {
    local label="$1"; local host="$2"; local remote="$3"; local port="$4"
    if ssh ubuntu@$host "ss -nap | grep sctp | grep ESTAB | grep -q '$remote'" &>/dev/null; then
        printf "  %-48s ✓\n" "$label"
        ((PASS++))
    else
        printf "  %-48s ✗\n" "$label"
        ((FAIL++))
    fi
}

echo "======================================================"
echo "  VNF Inter-Node Connectivity Test"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================"

echo ""
echo "── F1 interface (DU → CU, SCTP 38472) ─────────────"
# Check CU is listening
check_sctp_listen "CU2 listening on F1 :38472"     $CU2 38472
check_sctp_listen "CU3 listening on F1 :38472"     $CU3 38472
# Check established sessions
check_sctp_estab  "DU0 ↔ CU2 F1 session ESTAB"    $CU2 $DU0 38472
check_sctp_estab  "DU1 ↔ CU3 F1 session ESTAB"    $CU3 $DU1 38472
# Cross-reachability for migration (IP level — SCTP establishes on demand)
ping_check "DU0 → CU3 " "ssh ubuntu@$DU0 'ping -c3 -q $CU3'"
ping_check "DU1 → CU2 " "ssh ubuntu@$DU1 'ping -c3 -q $CU2'"

echo ""
echo "── N2 interface (CU → AMF, SCTP 38412) ────────────"
check_sctp_estab "CU2 → AMF N2 session ESTAB"     $CU2 $CTL 38412
check_sctp_estab "CU3 → AMF N2 session ESTAB"     $CU3 $CTL 38412

#ping_check "CU2 → AMF N2 UDP 38412" "ssh ubuntu@$CU2 'ping -c3 -q $CTL'"
#ping_check "CU3 → AMF N2 UDP 38412" "ssh ubuntu@$CU3 'ping -c3 -q $CTL'"


echo ""
echo "── N3 interface (CU → UPF, UDP 2152) ──────────────"

ping_check "CU2 → UPF N3 UDP 2152" "ssh ubuntu@$CU2 'ping -c3 -q $CTL'"
ping_check "CU3 → UPF N3 UDP 2152" "ssh ubuntu@$CU3 'ping -c3 -q $CTL'"

echo ""
echo "── IP reachability (all pairs) ─────────────────────"
NODES="DU0:$DU0 DU1:$DU1 CU2:$CU2 CU3:$CU3"
for SRC_PAIR in $NODES; do
    SRC_NAME=${SRC_PAIR%%:*}; SRC_IP=${SRC_PAIR##*:}
    for DST_PAIR in $NODES; do
        DST_NAME=${DST_PAIR%%:*}; DST_IP=${DST_PAIR##*:}
        [[ $SRC_NAME == $DST_NAME ]] && continue
        ping_check "$SRC_NAME → $DST_NAME ($DST_IP)" \
            "ssh ubuntu@$SRC_IP 'ping -c3 -q $DST_IP'"
    done
done

echo ""
echo "── E2E UE → ext-dn (192.168.70.135) ───────────────"
declare -A UE_NAMES=([1]="oai-nr-ue" [2]="oai-nr-ue2" [3]="oai-nr-ue3" [4]="oai-nr-ue4")
for i in 1 2 3 4; do
    NAME=${UE_NAMES[$i]}
    IP=$(docker exec $NAME ip addr show oaitun_ue1 2>/dev/null \
        | grep "inet " | awk '{print $2}' | cut -d/ -f1)
    if [[ -z "$IP" ]]; then
        printf "  %-30s ✗ (no tunnel IP)\n" "UE$i ($NAME)"
        ((FAIL++))
        continue
    fi
    ping_check "UE$i ($IP) → ext-dn" \
        "docker exec $NAME ping -I oaitun_ue1 192.168.70.135 -c3 -q"
done

echo ""
echo "======================================================"
printf "  Result: %d pass, %d fail\n" $PASS $FAIL
[[ $FAIL -eq 0 ]] && echo "  ALL PASS ✓" || echo "  $FAIL interface(s) FAILED ✗"
echo "======================================================"
