#!/bin/bash
# =============================================================================
# watch_ues.sh — Live monitor: 4 RUs × 3 UEs + real-time traffic counters
#
# Run in a separate terminal alongside train_online.py
# Usage: bash ~/watch_ues.sh
# =============================================================================

EXT_DN_IP="192.168.70.135"
IPERF_PORT=5201
INTERVAL=5

ue_container() {
    local i=$1
    [[ $i -eq 1 ]] && echo "oai-nr-ue"  && return
    [[ $i -eq 2 ]] && echo "oai-nr-ue2" && return
    echo "oai-nr-ue${i}"
}

ue_ru()  { echo $(( ($1 - 1) / 3 )); }
ue_num() { echo $(( ($1 - 1) % 3 + 1 )); }

bar() {
    local val=$1 max=${2:-100} width=${3:-12}
    local filled=$(( val * width / (max > 0 ? max : 1) ))
    [[ $filled -gt $width ]] && filled=$width
    local empty=$(( width - filled ))
    printf '%s%s' "$(printf '█%.0s' $(seq 1 $filled 2>/dev/null || true))" \
                  "$(printf '░%.0s' $(seq 1 $empty  2>/dev/null || true))"
}

# Declare associative arrays for previous byte counts
declare -A PREV_RX
declare -A PREV_TX
declare -A PREV_TIME

for i in $(seq 1 12); do
    PREV_RX[$i]=0; PREV_TX[$i]=0; PREV_TIME[$i]=$(date +%s%N)
done

while true; do
    clear

    NOW=$(date +"%H:%M:%S")
    EPOCH=$(date +%s)

    # ── Header ────────────────────────────────────────────────────────────────
    echo "╔══════════════════════════════════════════════════════════════════════╗"
    echo "║  E-MATRO Live Traffic Monitor   O-RAN: 4 RUs × 3 UEs   ${NOW}  ║"
    echo "╠══════════════════════════════════════════════════════════════════════╣"
    echo "║  IMSI range: 001010000000101 → 001010000000112                       ║"
    echo "╠═══════╦═══════════════╦══════════╦══════════╦════════════════════════╣"
    printf "║ %-5s ║ %-13s ║ %-8s ║ %-8s ║ %-22s ║\n" \
        "UE" "Container" "Tunnel IP" "Status" "UL traffic (live)"
    echo "╠═══════╬═══════════════╬══════════╬══════════╬════════════════════════╣"

    declare -A RU_MBPS
    for r in 0 1 2 3; do RU_MBPS[$r]="0.0"; done

    for i in $(seq 1 12); do
        NAME=$(ue_container $i)
        RU=$(ue_ru $i)
        UE_N=$(ue_num $i)

        # Print RU separator
        if [[ $UE_N -eq 1 ]]; then
            [[ $i -gt 1 ]] && \
            echo "╠═══════╬═══════════════╬══════════╬══════════╬════════════════════════╣"
            echo "║ RU ${RU}  ║               ║          ║          ║                        ║"
        fi

        # Get tunnel IP
        IP=$(docker exec $NAME ip addr show oaitun_ue1 2>/dev/null \
            | grep "inet " | awk '{print $2}' | cut -d/ -f1 2>/dev/null || echo "")

        if [[ -z "$IP" ]]; then
            printf "║ UE%2d  ║ %-13s ║ %-8s ║ %-8s ║ %-22s ║\n" \
                $i "$NAME" "no IP" "DOWN" "not connected"
            continue
        fi

        # Get byte counters from /proc/net/dev inside container
        RAW=$(docker exec $NAME cat /proc/net/dev 2>/dev/null \
            | grep "oaitun" || echo "")

        if [[ -z "$RAW" ]]; then
            STATUS="UP/nodata"
            MBPS_STR="0.0 Mbps"
            BAR_STR="░░░░░░░░░░░░"
        else
            CUR_RX=$(echo "$RAW" | awk '{gsub(/:/,"",$1); print $10}')
            CUR_TX=$(echo "$RAW" | awk '{gsub(/:/,"",$1); print $2}')
            CUR_TIME=$(date +%s%N)

            PREV=${PREV_TX[$i]:-0}
            PREV_T=${PREV_TIME[$i]:-$CUR_TIME}
            DT=$(( (CUR_TIME - PREV_T) / 1000000 ))  # ms

            if [[ $DT -gt 0 && $PREV -gt 0 ]]; then
                BYTES_DIFF=$(( CUR_TX - PREV ))
                [[ $BYTES_DIFF -lt 0 ]] && BYTES_DIFF=0
                # Mbps = bytes * 8 / (dt_ms / 1000) / 1e6
                MBPS=$(echo "scale=2; $BYTES_DIFF * 8 / $DT / 1000" | bc 2>/dev/null || echo "0.0")
            else
                MBPS="0.0"
            fi

            PREV_TX[$i]=$CUR_TX
            PREV_TIME[$i]=$CUR_TIME

            # Cap bar at 20 Mbps for display
            MBPS_INT=$(echo "$MBPS" | cut -d. -f1)
            BAR_STR=$(bar ${MBPS_INT:-0} 20 12)
            MBPS_STR="${MBPS} Mbps"

            # Accumulate per-RU
            OLD=${RU_MBPS[$RU]}
            RU_MBPS[$RU]=$(echo "scale=2; $OLD + $MBPS" | bc 2>/dev/null || echo "$OLD")

            # iperf3 running?
            IPERF=$(docker exec $NAME pgrep -c iperf3 2>/dev/null || echo "0")
            [[ "$IPERF" -gt 0 ]] && STATUS="TX:iperf" || STATUS="UP"
        fi

        printf "║ UE%2d  ║ %-13s ║ %-8s ║ %-8s ║ %s %-10s║\n" \
            $i "$NAME" \
            "${IP:0:8}" "$STATUS" \
            "$BAR_STR" "$MBPS_STR"
    done

    # ── Per-RU summary ────────────────────────────────────────────────────────
    echo "╠═══════╩═══════════════╩══════════╩══════════╩════════════════════════╣"
    echo "║  Per-RU aggregate UL load (sum of 3 UEs each)                        ║"
    echo "╠══════════════════════════════════════════════════════════════════════╣"
    for r in 0 1 2 3; do
        TOTAL=${RU_MBPS[$r]}
        TOTAL_INT=$(echo "$TOTAL" | cut -d. -f1)
        BAR_R=$(bar ${TOTAL_INT:-0} 60 24)
        printf "║  RU %d : %s  %6s Mbps                     ║\n" \
            $r "$BAR_R" "$TOTAL"
    done

    # ── Core status ───────────────────────────────────────────────────────────
    echo "╠══════════════════════════════════════════════════════════════════════╣"
    REGISTERED=$(docker logs oai-amf 2>&1 | grep -c "5GMM-REGISTERED" 2>/dev/null || echo "?")
    GNB_OK=$(docker inspect oai-gnb --format "{{.State.Health.Status}}" 2>/dev/null || echo "?")
    MYSQL_OK=$(docker inspect mysql --format "{{.State.Health.Status}}" 2>/dev/null || echo "?")
    IPERF_SRV=$(docker exec oai-ext-dn pgrep -c iperf3 2>/dev/null || echo "0")
    printf "║  gNB:%-9s  MySQL:%-10s  UEs registered:%-4s  iperf3-srv:%-3s    ║\n" \
        "$GNB_OK" "$MYSQL_OK" "$REGISTERED" "$IPERF_SRV"
    echo "╠══════════════════════════════════════════════════════════════════════╣"
    echo "║  Refresh every ${INTERVAL}s   Ctrl+C to stop   UL bar scale: 0─────────20 Mbps ║"
    echo "╚══════════════════════════════════════════════════════════════════════╝"

    sleep $INTERVAL
done