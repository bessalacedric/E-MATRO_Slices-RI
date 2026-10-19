#!/bin/bash
# =============================================================================
# test_traffic.sh — Fire iperf3 on all connected UEs in parallel
#
# Usage: bash test_traffic.sh [bitrate_mbps] [duration_sec]
# =============================================================================

BITRATE=${1:-5}   # Mbps per UE
DURATION=${2:-10} # seconds
EXT_DN_IP="192.168.70.135"
PORT=5201

ue_container() {
    local i=$1
    [[ $i -eq 1 ]] && echo "oai-nr-ue"  && return
    [[ $i -eq 2 ]] && echo "oai-nr-ue2" && return
    echo "oai-nr-ue${i}"
}

echo "════════════════════════════════════════════════════════"
echo "  Parallel iperf3 UL test: ${BITRATE} Mbps × ${DURATION}s per UE"
echo "  Target: oai-ext-dn ${EXT_DN_IP}:${PORT}"
echo "════════════════════════════════════════════════════════"

# Start iperf3 server if not running
docker exec oai-ext-dn pgrep -f "iperf3 -s" &>/dev/null || docker exec -d oai-ext-dn iperf3 -s -p $PORT
sleep 1

# Fire all UEs in parallel, save results to tmp files
PIDS=()
for i in $(seq 1 12); do
    NAME=$(ue_container $i)
    RU=$(( (i-1) / 3 ))
    UE_N=$(( (i-1) % 3 + 1 ))
    TMPFILE="/tmp/iperf3_ue${i}.json"

    IP=$(docker exec $NAME ip addr show oaitun_ue1 2>/dev/null | grep "inet " | awk '{print $2}' | cut -d/ -f1 || echo "")

    if [[ -z "$IP" ]]; then
        echo "  UE${i} (RU${RU}-UE${UE_N}): no tunnel IP — skipping"
        continue
    fi

    echo "  Starting UE${i} (RU${RU}-UE${UE_N}) from ${IP}..."

    docker exec $NAME iperf3 -c $EXT_DN_IP -p $PORT -u -b ${BITRATE}M -t $DURATION -B $IP -J \
        > $TMPFILE 2>/dev/null &
    PIDS+=($!)
done

echo ""
echo "  Waiting ${DURATION}s for all tests to complete..."
for PID in "${PIDS[@]}"; do wait $PID 2>/dev/null || true; done

# Parse results
echo ""
echo "════════════════════════════════════════════════════════"
printf "  %-6s %-14s %-20s %-15s\n" "UE" "Container" "Tunnel IP" "UL bitrate"
echo "  ──────────────────────────────────────────────────────"

TOTAL=0
COUNT=0
for i in $(seq 1 12); do
    NAME=$(ue_container $i)
    RU=$(( (i-1) / 3 ))
    UE_N=$(( (i-1) % 3 + 1 ))
    TMPFILE="/tmp/iperf3_ue${i}.json"

    [[ ! -f "$TMPFILE" ]] && continue

    IP=$(docker exec $NAME ip addr show oaitun_ue1 2>/dev/null | grep "inet " | awk '{print $2}' | cut -d/ -f1 || echo "no-ip")

    BPS=$(python3 -c "
import json, sys
try:
    d = json.load(open('$TMPFILE'))
    print(d['end']['sum_sent']['bits_per_second'])
except:
    print(0)
" 2>/dev/null || echo "0")

    MBPS=$(echo "scale=2; $BPS / 1000000" | bc 2>/dev/null || echo "0.00")
    TOTAL=$(echo "scale=2; $TOTAL + $MBPS" | bc 2>/dev/null || echo "$TOTAL")
    COUNT=$((COUNT + 1))

    printf "  RU%d-UE%d %-14s %-20s %s Mbps\n" \
        $RU $UE_N "$NAME" "$IP" "$MBPS"
done

echo "  ──────────────────────────────────────────────────────"
echo "  Total UL throughput: ${TOTAL} Mbps across ${COUNT} UEs"
echo ""

# Also show per-RU totals
echo "  Per-RU aggregate:"
for r in 0 1 2 3; do
    RU_TOTAL=0
    for ue_offset in 1 2 3; do
        i=$(( r * 3 + ue_offset ))
        TMPFILE="/tmp/iperf3_ue${i}.json"
        [[ ! -f "$TMPFILE" ]] && continue
        BPS=$(python3 -c "
import json
try:
    d=json.load(open('$TMPFILE'))
    print(d['end']['sum_sent']['bits_per_second'])
except: print(0)
" 2>/dev/null || echo "0")
        M=$(echo "scale=2; $BPS/1000000" | bc 2>/dev/null || echo "0")
        RU_TOTAL=$(echo "scale=2; $RU_TOTAL + $M" | bc 2>/dev/null || echo "$RU_TOTAL")
    done
    printf "    RU%d: %s Mbps\n" $r "$RU_TOTAL"
done
echo "════════════════════════════════════════════════════════"