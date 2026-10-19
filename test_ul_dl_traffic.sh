#!/bin/bash
# =============================================================================
# test_ul_dl_traffic.sh — UL + DL iperf3 traffic test for all UEs
# Usage: bash test_ul_dl_traffic.sh [MBPS] [DURATION]
#   MBPS     : target bitrate per UE in Mbps (default: 5)
#   DURATION : test duration in seconds     (default: 20)
#
# Example: bash test_ul_dl_traffic.sh 10 30
# =============================================================================


MBPS=${1:-5}
DURATION=${2:-20}
EXT_DN="192.168.70.135"
BASE_PORT=5200
RESULT_DIR="/tmp/iperf_results"

# UE containers in order
UE_CONTAINERS=(
    "oai-nr-ue"
    "oai-nr-ue2"
    "oai-nr-ue3"
    "oai-nr-ue4"
    "oai-nr-ue5"
    "oai-nr-ue6"
    "oai-nr-ue7"
    "oai-nr-ue8"
    "oai-nr-ue9"
    "oai-nr-ue10"
    "oai-nr-ue11"
    "oai-nr-ue12"
)
N_UES=${#UE_CONTAINERS[@]}

# RU labels (1 UE per RU in current setup)
RU_LABELS=("RU0" "RU1" "RU2" "RU3" "RU4" "RU5" "RU6" "RU7" "RU8" "RU9" "RU10" "RU11")

# =============================================================================
# Helper
# =============================================================================
sep() { printf '═%.0s' $(seq 1 60); echo; }

cleanup() {
    # Kill iperf3 servers on ext-dn
    docker exec oai-ext-dn pkill iperf3 2>/dev/null || true
    rm -f $RESULT_DIR/*.json $RESULT_DIR/*.log 2>/dev/null
}
trap cleanup EXIT

# =============================================================================
# Step 1 — Start iperf3 servers on ext-dn (one port per UE)
# =============================================================================
mkdir -p $RESULT_DIR

docker exec oai-ext-dn pkill iperf3 2>/dev/null || true
sleep 1

for i in $(seq 0 $((N_UES-1))); do
    PORT=$((BASE_PORT + i + 1))
    docker exec -d oai-ext-dn iperf3 -s -p $PORT -1 2>/dev/null
done
sleep 2   # let servers start

# =============================================================================
# Step 2 — Collect UE IPs
# =============================================================================
declare -A UE_IPS
declare -A UE_ACTIVE

for i in $(seq 0 $((N_UES-1))); do
    NAME=${UE_CONTAINERS[$i]}
    IP=$(docker exec $NAME ip addr show oaitun_ue1 2>/dev/null \
         | grep "inet " | awk '{print $2}' | cut -d/ -f1)
    UE_IPS[$i]="$IP"
    [[ -n "$IP" ]] && UE_ACTIVE[$i]=1 || UE_ACTIVE[$i]=0
done

# =============================================================================
# UL TEST
# =============================================================================
run_ul_test() {
    sep
    printf "  Parallel iperf3 UL test: %s Mbps × %ss per UE\n" "$MBPS" "$DURATION"
    printf "  UE → ext-dn (%s)\n" "$EXT_DN"
    sep

    # Start all UEs in parallel
    declare -a UL_PIDS
    for i in $(seq 0 $((N_UES-1))); do
        NAME=${UE_CONTAINERS[$i]}
        RU=${RU_LABELS[$i]}
        PORT=$((BASE_PORT + i + 1))
        IP=${UE_IPS[$i]}

        if [[ ${UE_ACTIVE[$i]} -eq 0 ]]; then
            printf "  %-8s %-18s no tunnel IP — skipping\n" "$RU" "$NAME"
            continue
        fi

        printf "  Starting %-8s (%s) from %s → port %d...\n" "$RU" "$NAME" "$IP" "$PORT"
        timeout $((DURATION + 10)) docker exec $NAME iperf3 \
            -c $EXT_DN -p $PORT \
            -B $IP \
            -t $DURATION \
            -b ${MBPS}M \
            -u \
            -J \
            > $RESULT_DIR/ul_ue${i}.json 2>/dev/null &
        UL_PIDS[$i]=$!
    done

    printf "\n  Waiting %ss for UL tests to complete...\n" "$DURATION"
    sleep $((DURATION + 3))

    # Kill any hung clients and collect
    for pid in "${UL_PIDS[@]}"; do
        kill $pid 2>/dev/null || true
    done
    wait 2>/dev/null || true

    # Print UL results
    sep
    printf "  %-10s %-20s %-20s %-15s %-10s\n" \
        "UE" "Container" "Tunnel IP" "UL bitrate" "Loss"
    printf "  %s\n" "$(printf '─%.0s' $(seq 1 78))"

    declare -A RU_UL_TOTAL
    for i in "${!RU_LABELS[@]}"; do RU_UL_TOTAL[${RU_LABELS[$i]}]=0; done
    TOTAL_UL=0
    ACTIVE_UES=0

    for i in $(seq 0 $((N_UES-1))); do
        NAME=${UE_CONTAINERS[$i]}
        RU=${RU_LABELS[$i]}
        IP=${UE_IPS[$i]:-"—"}

        if [[ ${UE_ACTIVE[$i]} -eq 0 ]]; then
            printf "  %-10s %-20s %-20s %-15s %-10s\n" \
                "$RU" "$NAME" "—" "—" "—"
            continue
        fi

        RESULT=$RESULT_DIR/ul_ue${i}.json
        if [[ -f "$RESULT" ]]; then
            BITRATE=$(python3 -c "
import json, sys
try:
    d = json.load(open('$RESULT'))
    bps = d['end']['sum']['bits_per_second']
    print(f'{bps/1e6:.2f}')
except:
    print('0')
" 2>/dev/null)
            LOSS=$(python3 -c "
import json, sys
try:
    d = json.load(open('$RESULT'))
    lost  = d['end']['sum']['lost_packets']
    total = d['end']['sum']['packets']
    pct   = lost/total*100 if total>0 else 0
    print(f'{pct:.1f}%')
except:
    print('—')
" 2>/dev/null)
        else
            BITRATE="0"
            LOSS="—"
        fi

        printf "  %-10s %-20s %-20s %-15s %-10s\n" \
            "$RU" "$NAME" "$IP" "${BITRATE} Mbps" "$LOSS"

        TOTAL_UL=$(echo "$TOTAL_UL + ${BITRATE:-0}" | bc -l 2>/dev/null || echo $TOTAL_UL)
        RU_UL_TOTAL[$RU]=$(echo "${RU_UL_TOTAL[$RU]:-0} + ${BITRATE:-0}" | bc -l 2>/dev/null || echo 0)
        ((ACTIVE_UES++)) || true
    done

    printf "  %s\n" "$(printf '─%.0s' $(seq 1 78))"
    printf "  Total UL throughput: %.2f Mbps across %d UEs\n" "$TOTAL_UL" "$ACTIVE_UES"
    printf "  Per-RU aggregate (UL):\n"
    for RU in "${RU_LABELS[@]}"; do
        VAL=${RU_UL_TOTAL[$RU]:-0}
        if (( $(echo "$VAL > 0" | bc -l) )); then
            printf "    %s: %.2f Mbps\n" "$RU" "$VAL"
        fi
    done
}

# =============================================================================
# DL TEST
# =============================================================================
run_dl_test() {
    sep
    printf "  Parallel iperf3 DL test: %s Mbps × %ss per UE\n" "$MBPS" "$DURATION"
    printf "  ext-dn → UE (%s)\n" "$EXT_DN"
    sep

    # For DL: iperf3 client runs on ext-dn, server on UE side
    # Start iperf3 servers inside UE containers — bind to tunnel IP
    declare -a UE_SERVER_PIDS
    for i in $(seq 0 $((N_UES-1))); do
        [[ ${UE_ACTIVE[$i]} -eq 0 ]] && continue
        NAME=${UE_CONTAINERS[$i]}
        PORT=$((BASE_PORT + 100 + i + 1))
        IP=${UE_IPS[$i]}
        # -B binds server to tunnel interface (oaitun_ue1) not Docker bridge
        docker exec -d $NAME iperf3 -s -p $PORT -B $IP 2>/dev/null
        UE_SERVER_PIDS[$i]=$PORT
    done
    sleep 2

    # Run DL clients from ext-dn
    declare -a DL_PIDS
    for i in $(seq 0 $((N_UES-1))); do
        [[ ${UE_ACTIVE[$i]} -eq 0 ]] && continue
        NAME=${UE_CONTAINERS[$i]}
        RU=${RU_LABELS[$i]}
        PORT=$((BASE_PORT + 100 + i + 1))
        IP=${UE_IPS[$i]}

        printf "  Starting DL %-8s (%s) to %s port %d...\n" "$RU" "$NAME" "$IP" "$PORT"
        timeout $((DURATION + 10)) docker exec oai-ext-dn iperf3 \
            -c $IP -p $PORT \
            -t $DURATION \
            -b ${MBPS}M \
            -u \
            -J \
            > $RESULT_DIR/dl_ue${i}.json 2>/dev/null &
        DL_PIDS[$i]=$!
    done

    printf "\n  Waiting %ss for DL tests to complete...\n" "$DURATION"
    sleep $((DURATION + 3))

    # Kill any hung clients
    for pid in "${DL_PIDS[@]}"; do
        kill $pid 2>/dev/null || true
    done
    wait 2>/dev/null || true

    # Stop UE iperf3 servers
    for i in $(seq 0 $((N_UES-1))); do
        [[ ${UE_ACTIVE[$i]} -eq 0 ]] && continue
        docker exec ${UE_CONTAINERS[$i]} pkill iperf3 2>/dev/null || true
    done

    # Print DL results
    sep
    printf "  %-10s %-20s %-20s %-15s %-10s\n" \
        "UE" "Container" "Tunnel IP" "DL bitrate" "Loss"
    printf "  %s\n" "$(printf '─%.0s' $(seq 1 78))"

    TOTAL_DL=0
    ACTIVE_UES=0

    for i in $(seq 0 $((N_UES-1))); do
        NAME=${UE_CONTAINERS[$i]}
        RU=${RU_LABELS[$i]}
        IP=${UE_IPS[$i]:-"—"}

        if [[ ${UE_ACTIVE[$i]} -eq 0 ]]; then
            printf "  %-10s %-20s %-20s %-15s %-10s\n" \
                "$RU" "$NAME" "—" "—" "—"
            continue
        fi

        RESULT=$RESULT_DIR/dl_ue${i}.json
        if [[ -f "$RESULT" ]]; then
            BITRATE=$(python3 -c "
import json
try:
    d = json.load(open('$RESULT'))
    bps = d['end']['sum']['bits_per_second']
    print(f'{bps/1e6:.2f}')
except:
    print('0')
" 2>/dev/null)
            LOSS=$(python3 -c "
import json
try:
    d = json.load(open('$RESULT'))
    lost  = d['end']['sum']['lost_packets']
    total = d['end']['sum']['packets']
    pct   = lost/total*100 if total>0 else 0
    print(f'{pct:.1f}%')
except:
    print('—')
" 2>/dev/null)
        else
            BITRATE="0"
            LOSS="—"
        fi

        printf "  %-10s %-20s %-20s %-15s %-10s\n" \
            "$RU" "$NAME" "$IP" "${BITRATE} Mbps" "$LOSS"

        TOTAL_DL=$(echo "$TOTAL_DL + ${BITRATE:-0}" | bc -l 2>/dev/null || echo $TOTAL_DL)
        ((ACTIVE_UES++)) || true
    done

    printf "  %s\n" "$(printf '─%.0s' $(seq 1 78))"
    printf "  Total DL throughput: %.2f Mbps across %d UEs\n" "$TOTAL_DL" "$ACTIVE_UES"
}

# =============================================================================
# Main
# =============================================================================
sep
printf "  Traffic Test — %s Mbps × %ss | %s\n" \
    "$MBPS" "$DURATION" "$(date '+%Y-%m-%d %H:%M:%S')"
printf "  UEs configured: %d | Active: %d\n" \
    "$N_UES" "$(for i in $(seq 0 $((N_UES-1))); do echo ${UE_ACTIVE[$i]}; done | grep -c 1)"
sep

echo ""
echo "╔══ UPLINK ══════════════════════════════════════════╗"
run_ul_test

echo ""
echo "╔══ DOWNLINK ════════════════════════════════════════╗"
run_dl_test

sep
printf "  Test complete — %s\n" "$(date '+%H:%M:%S')"
sep