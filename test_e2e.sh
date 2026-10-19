#!/bin/bash
# E2E Connectivity Test — 4 UEs → ext-dn (192.168.70.135)

EXT_DN="192.168.70.135"
PASS=0
FAIL=0

declare -A UE_NAMES=(
    [1]="oai-nr-ue"
    [2]="oai-nr-ue2"
    [3]="oai-nr-ue3"
    [4]="oai-nr-ue4"
)

echo "======================================================"
echo "  E2E Connectivity Test — $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================"
echo ""

for i in 1 2 3 4; do
    NAME=${UE_NAMES[$i]}
    IP=$(docker exec $NAME ip addr show oaitun_ue1 2>/dev/null \
        | grep "inet " | awk '{print $2}' | cut -d/ -f1)

    if [[ -z "$IP" ]]; then
        echo "UE$i ($NAME): NO TUNNEL IP — SKIP"
        ((FAIL++))
        continue
    fi

    RESULT=$(docker exec $NAME ping -I oaitun_ue1 $EXT_DN -c3 -q 2>/dev/null)
    SUMMARY=$(echo "$RESULT" | grep "packets transmitted")
    RTT=$(echo "$RESULT" | grep "rtt")
    LOSS=$(echo "$SUMMARY" | grep -oP '\d+(?=% packet loss)')

    printf "UE%d %-14s [%s → %s]\n" $i "$NAME" "$IP" "$EXT_DN"
    if [[ -n "$SUMMARY" ]]; then
        echo "  $SUMMARY"
        [[ -n "$RTT" ]] && echo "  $RTT"
    fi

    if [[ "$LOSS" == "0" ]]; then
        echo "  E2E ✓"
        ((PASS++))
    else
        echo "  E2E ✗ (${LOSS}% loss)"
        ((FAIL++))
    fi
    echo ""
done

echo "======================================================"
echo "  Result: $PASS/4 UEs OK"
[[ $FAIL -eq 0 ]] && echo "  ALL PASS ✓" || echo "  $FAIL FAILED ✗"
echo "======================================================"

