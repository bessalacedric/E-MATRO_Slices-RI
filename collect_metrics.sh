#!/bin/bash
# =============================================================================
# collect_metrics.sh — Parallel metrics collection during E-MATRO training
#
# Collects 8 metrics in background during training, writes separate CSVs.
# Run BEFORE starting train.py, stops automatically when training ends
# (or kill with Ctrl-C / kill $(cat /tmp/e-matro_metrics.pid))
#
# Usage:
#   # Terminal 1 — start metrics collection
#   bash collect_metrics.sh --n-gnb 2 --interval 5 --out-dir results/metrics
#
#   # Terminal 2 — run training
#   python3 train.py --n-gnb 2 --episodes 5000 --max-steps 10
#
# Output files (all in --out-dir):
#   energy_saving.csv       — nodes active, energy saving % per step
#   du_cpu.csv              — DU container CPU% and memory per VM
#   cu_cpu.csv              — CU container CPU% and memory per VM
#   migration_timing.csv    — migration time, interruption, state size
#   ue_throughput.csv       — UL/DL throughput per UE
#   bler_snr.csv            — BLER, SNR, MCS, NPRB per RU from DU logs
#   vm_resources.csv        — per-VM host CPU, memory, network bytes
#   link_utilization.csv    — bytes/s on enp6s18 per VM
# =============================================================================
set -uo pipefail

# =============================================================================
# Args
# =============================================================================
N_GNB=2
INTERVAL=5       # seconds between samples
OUT_DIR="results/metrics"
SSH_USER="ubuntu"
MULTI_GNB=true   # set false to use original single-gNB topology

while [[ $# -gt 0 ]]; do
    case $1 in
        --n-gnb)    N_GNB=$2;     shift 2 ;;
        --interval) INTERVAL=$2;  shift 2 ;;
        --out-dir)  OUT_DIR=$2;   shift 2 ;;
        --user)     SSH_USER=$2;  shift 2 ;;
        *) shift ;;
    esac
done

N_UES=$((N_GNB * 4))
N_DUS=$N_GNB
N_CUS=$(( (N_GNB + 1) / 2 ))

VM_IPS=(
    "10.10.212.103" "10.10.214.110" "10.10.212.33" "10.10.212.196"
    "10.10.209.122" "10.10.210.7" "10.10.214.49" "10.10.212.215"
    "10.10.209.93" "10.10.210.241"  "10.10.210.203"  "10.10.215.22"
    "10.10.211.154" "10.10.213.26" "10.10.209.253"
)

SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=3"
mkdir -p "$OUT_DIR"

# PID file to track all background processes
PID_FILE="/tmp/e-matro_metrics.pid"
> "$PID_FILE"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[✓]${NC} $*"; }
info() { echo -e "${YELLOW}[→]${NC} $*"; }

echo "════════════════════════════════════════════════════════════"
echo "  E-MATRO Metrics Collection"
echo "  N_GNB=${N_GNB}  N_UES=${N_UES}  interval=${INTERVAL}s"
echo "  Output: ${OUT_DIR}/"
echo "════════════════════════════════════════════════════════════"

# Cleanup on exit
cleanup() {
    echo ""
    info "Stopping all metric collectors..."
    while read pid; do
        kill "$pid" 2>/dev/null || true
    done < "$PID_FILE"
    ok "Collection stopped. Results in ${OUT_DIR}/"
}
trap cleanup EXIT INT TERM

# =============================================================================
# Helper: timestamp
# =============================================================================
ts() { date +%s; }
ts_iso() { date '+%Y-%m-%d %H:%M:%S'; }

# =============================================================================
# Metric 1 — Energy saving + nodes active
# (reads training_log.csv which train.py writes live)
# =============================================================================
collect_energy() {
    local f="$OUT_DIR/energy_saving.csv"
    echo "timestamp,episode,energy_saving_pct,nodes_active,avg_reward,power_W,qos_violation" > "$f"

    local last_ep=0
    while true; do
        if [[ -f "results/training_log.csv" ]]; then
            tail -1 results/training_log.csv | while IFS=, read ep rew loss sav nodes pow qos mig fail orph conf eps step; do
                if [[ "$ep" != "$last_ep" && "$ep" =~ ^[0-9]+$ ]]; then
                    echo "$(ts_iso),$ep,$sav,$nodes,$rew,$pow,$qos" >> "$f"
                    last_ep=$ep
                fi
            done
        fi
        sleep "$INTERVAL"
    done
}

# =============================================================================
# Metric 2 — DU CPU and memory (docker stats per DU container)
# =============================================================================
collect_du_cpu() {
    local f="$OUT_DIR/du_cpu.csv"
    echo "timestamp,gnb_id,vm_ip,container,cpu_pct,mem_usage_mb,mem_limit_mb,mem_pct" > "$f"

    while true; do
        for ((gnb=0; gnb<N_DUS; gnb++)); do
            vm_idx=$((gnb % ${#VM_IPS[@]}))
            vm_ip="${VM_IPS[$vm_idx]}"
            container="oai-du-gnb${gnb}"

            result=$($SSH ${SSH_USER}@${vm_ip} \
                "docker stats $container --no-stream \
                --format '{{.CPUPerc}},{{.MemUsage}},{{.MemPerc}}' 2>/dev/null" \
                2>/dev/null || echo "N/A,N/A,N/A")

            # Parse "cpu%, X.XMiB / Y.YGiB, mem%"
            cpu=$(echo "$result" | cut -d',' -f1 | tr -d '%')
            mem_raw=$(echo "$result" | cut -d',' -f2)
            mem_pct=$(echo "$result" | cut -d',' -f3 | tr -d '%')
            mem_mb=$(echo "$mem_raw" | python3 -c "
import sys; s=sys.stdin.read().strip().split(' / ')
def to_mb(x):
    x=x.strip()
    if 'GiB' in x: return float(x.replace('GiB',''))*1024
    if 'MiB' in x: return float(x.replace('MiB',''))
    if 'kB'  in x: return float(x.replace('kB',''))/1024
    return 0
print(f'{to_mb(s[0]):.1f},{to_mb(s[1]):.1f}' if len(s)==2 else '0,0')
" 2>/dev/null || echo "0,0")

            echo "$(ts_iso),${gnb},${vm_ip},${container},${cpu},${mem_mb},${mem_pct}" >> "$f"
        done
        sleep "$INTERVAL"
    done
}

# =============================================================================
# Metric 3 — CU CPU and memory
# =============================================================================
collect_cu_cpu() {
    local f="$OUT_DIR/cu_cpu.csv"
    echo "timestamp,cu_id,vm_ip,container,cpu_pct,mem_usage_mb,mem_limit_mb,mem_pct" > "$f"

    while true; do
        for ((cu=0; cu<N_CUS; cu++)); do
            node_id=$((N_DUS + cu))
            vm_idx=$((node_id % ${#VM_IPS[@]}))
            vm_ip="${VM_IPS[$vm_idx]}"
            container="oai-cu-gnb${cu}"

            result=$($SSH ${SSH_USER}@${vm_ip} \
                "docker stats $container --no-stream \
                --format '{{.CPUPerc}},{{.MemUsage}},{{.MemPerc}}' 2>/dev/null" \
                2>/dev/null || echo "N/A,N/A,N/A")

            cpu=$(echo "$result" | cut -d',' -f1 | tr -d '%')
            mem_raw=$(echo "$result" | cut -d',' -f2)
            mem_pct=$(echo "$result" | cut -d',' -f3 | tr -d '%')
            mem_mb=$(echo "$mem_raw" | python3 -c "
import sys; s=sys.stdin.read().strip().split(' / ')
def to_mb(x):
    x=x.strip()
    if 'GiB' in x: return float(x.replace('GiB',''))*1024
    if 'MiB' in x: return float(x.replace('MiB',''))
    if 'kB'  in x: return float(x.replace('kB',''))/1024
    return 0
print(f'{to_mb(s[0]):.1f},{to_mb(s[1]):.1f}' if len(s)==2 else '0,0')
" 2>/dev/null || echo "0,0")

            echo "$(ts_iso),${cu},${vm_ip},${container},${cpu},${mem_mb},${mem_pct}" >> "$f"
        done
        sleep "$INTERVAL"
    done
}

# =============================================================================
# Metric 4 — UE throughput (periodic iperf3 spot tests)
# Runs a 5s UL+DL test every 60s to not interfere with training
# =============================================================================
collect_ue_throughput() {
    local f="$OUT_DIR/ue_throughput.csv"
    echo "timestamp,ue_id,ue_name,tunnel_ip,ul_mbps,ul_loss_pct,dl_mbps,dl_loss_pct" > "$f"
    local EXT_DN="192.168.70.135"

    while true; do
        sleep 60   # run every 60s

        for ((ue_idx=0; ue_idx<N_UES; ue_idx++)); do
            if [[ $ue_idx -eq 0 ]]; then
                UE_NAME="oai-nr-ue"
            else
                UE_NAME="oai-nr-ue$((ue_idx+1))"
            fi

            # Get tunnel IP
            TUN_IP=$(docker exec $UE_NAME ip addr show oaitun_ue1 2>/dev/null \
                | grep "inet " | awk '{print $2}' | cut -d/ -f1)
            [[ -z "$TUN_IP" ]] && {
                echo "$(ts_iso),${ue_idx},${UE_NAME},NO_IP,0,0,0,0" >> "$f"
                continue
            }

            UL_PORT=$((5200 + ue_idx + 1))
            DL_PORT=$((5300 + ue_idx + 1))

            # UL test (5s, 5Mbps)
            docker exec oai-ext-dn iperf3 -s -p $UL_PORT -1 &>/dev/null &
            sleep 0.5
            UL=$(docker exec $UE_NAME iperf3 \
                -c $EXT_DN -p $UL_PORT -B $TUN_IP \
                -t 5 -b 5M -u -J 2>/dev/null | python3 -c "
import json,sys
try:
    d=json.load(sys.stdin)
    bps=d['end']['sum']['bits_per_second']
    lost=d['end']['sum']['lost_packets']
    total=d['end']['sum']['packets']
    pct=lost/total*100 if total>0 else 0
    print(f'{bps/1e6:.2f},{pct:.1f}')
except: print('0,0')
" 2>/dev/null || echo "0,0")

            # DL test (5s, 5Mbps)
            docker exec $UE_NAME iperf3 -s -p $DL_PORT -B $TUN_IP -D &>/dev/null
            sleep 0.5
            DL=$(docker exec oai-ext-dn iperf3 \
                -c $TUN_IP -p $DL_PORT \
                -t 5 -b 5M -u -J 2>/dev/null | python3 -c "
import json,sys
try:
    d=json.load(sys.stdin)
    bps=d['end']['sum']['bits_per_second']
    lost=d['end']['sum']['lost_packets']
    total=d['end']['sum']['packets']
    pct=lost/total*100 if total>0 else 0
    print(f'{bps/1e6:.2f},{pct:.1f}')
except: print('0,0')
" 2>/dev/null || echo "0,0")
            docker exec $UE_NAME pkill iperf3 &>/dev/null || true

            echo "$(ts_iso),${ue_idx},${UE_NAME},${TUN_IP},${UL},${DL}" >> "$f"
        done
    done
}

# =============================================================================
# Metric 5 — BLER, SNR, MCS, NPRB from DU logs
# =============================================================================
collect_bler_snr() {
    local f="$OUT_DIR/bler_snr.csv"
    echo "timestamp,gnb_id,ue_rnti,ul_mcs,ul_nprb,ul_bler,ul_snr_db,dl_mcs,dl_bler,ul_tput_est_mbps" > "$f"

    while true; do
        for ((gnb=0; gnb<N_DUS; gnb++)); do
            vm_idx=$((gnb % ${#VM_IPS[@]}))
            vm_ip="${VM_IPS[$vm_idx]}"

            logs=$($SSH ${SSH_USER}@${vm_ip} \
                "docker logs oai-du-gnb${gnb} 2>&1 \
                | grep -E 'ulsch_rounds|dlsch_rounds' | tail -8" \
                2>/dev/null || echo "")

            [[ -z "$logs" ]] && continue

            echo "$logs" | python3 - << PYEOF >> "$f"
import re, sys
from datetime import datetime

log = sys.stdin.read()
ts  = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
gnb = ${gnb}

ul_pat = re.compile(
    r'UE\s+(?P<rnti>[0-9a-f]+):.*?BLER\s+(?P<bler>[0-9.]+)\s+MCS\s+\(\d+\)\s+(?P<mcs>\d+)'
    r'.*?NPRB\s+(?P<nprb>\d+)\s+SNR\s+(?P<snr>[0-9.-]+)',
    re.DOTALL
)
dl_pat = re.compile(
    r'UE\s+(?P<rnti>[0-9a-f]+):.*?dlsch.*?BLER\s+(?P<bler>[0-9.]+)\s+MCS\s+\(\d+\)\s+(?P<mcs>\d+)'
)

ul = {m['rnti']: m.groupdict() for m in ul_pat.finditer(log)}
dl = {m['rnti']: m.groupdict() for m in dl_pat.finditer(log)}

MBPS_MAP = {0:0.02,5:0.07,9:0.18,14:0.35,19:0.55,24:0.85,28:1.10}
def mcs2mbps(mcs, nprb):
    c = min(MBPS_MAP, key=lambda k: abs(k-mcs))
    return round(MBPS_MAP[c]*nprb, 3)

for rnti, u in ul.items():
    d = dl.get(rnti, {})
    mcs=int(u['mcs']); nprb=int(u['nprb'])
    print(f"{ts},{gnb},{rnti},{mcs},{nprb},{u['bler']},{u['snr']},"
          f"{d.get('mcs','0')},{d.get('bler','0')},"
          f"{mcs2mbps(mcs,nprb)}")
PYEOF
        done
        sleep "$INTERVAL"
    done
}

# =============================================================================
# Metric 6 — VM host resources (CPU, memory, network)
# =============================================================================
collect_vm_resources() {
    local f="$OUT_DIR/vm_resources.csv"
    echo "timestamp,vm_ip,vm_idx,host_cpu_pct,host_mem_pct,host_mem_used_mb,net_rx_bytes,net_tx_bytes" > "$f"

    while true; do
        N_ACTIVE=$((N_DUS + N_CUS))
        for ((i=0; i<N_ACTIVE; i++)); do
            vm_idx=$((i % ${#VM_IPS[@]}))
            vm_ip="${VM_IPS[$vm_idx]}"

            result=$($SSH ${SSH_USER}@${vm_ip} "
                CPU=\$(top -bn1 | grep 'Cpu(s)' | awk '{print \$2}' | tr -d '%us,')
                MEM=\$(free -m | awk '/^Mem/{print \$3,\$2}')
                NET=\$(cat /proc/net/dev | grep enp6s18 | awk '{print \$2,\$10}')
                echo \$CPU \$MEM \$NET
            " 2>/dev/null || echo "0 0 0 0 0")

            read cpu mem_used mem_total rx tx <<< "$result"
            mem_pct=$(python3 -c "print(round($mem_used/$mem_total*100,1))" 2>/dev/null || echo 0)
            echo "$(ts_iso),${vm_ip},${vm_idx},${cpu},${mem_pct},${mem_used},${rx:-0},${tx:-0}" >> "$f"
        done
        sleep "$INTERVAL"
    done
}

# =============================================================================
# Metric 7 — Link utilization (bytes/s on enp6s18)
# =============================================================================
collect_link_util() {
    local f="$OUT_DIR/link_utilization.csv"
    echo "timestamp,vm_ip,rx_bytes_per_s,tx_bytes_per_s,rx_mbps,tx_mbps" > "$f"

    # Store previous values
    declare -A prev_rx prev_tx

    N_ACTIVE=$((N_DUS + N_CUS))
    for ((i=0; i<N_ACTIVE; i++)); do
        vm_idx=$((i % ${#VM_IPS[@]}))
        vm_ip="${VM_IPS[$vm_idx]}"
        prev_rx[$vm_ip]=0
        prev_tx[$vm_ip]=0
    done

    while true; do
        sleep "$INTERVAL"
        for ((i=0; i<N_ACTIVE; i++)); do
            vm_idx=$((i % ${#VM_IPS[@]}))
            vm_ip="${VM_IPS[$vm_idx]}"

            counters=$($SSH ${SSH_USER}@${vm_ip} \
                "cat /proc/net/dev | grep enp6s18 | awk '{print \$2,\$10}'" \
                2>/dev/null || echo "0 0")
            read rx tx <<< "$counters"

            drx=$(( rx - ${prev_rx[$vm_ip]:-0} ))
            dtx=$(( tx - ${prev_tx[$vm_ip]:-0} ))
            rx_mbps=$(python3 -c "print(round($drx/$INTERVAL/1e6,3))" 2>/dev/null || echo 0)
            tx_mbps=$(python3 -c "print(round($dtx/$INTERVAL/1e6,3))" 2>/dev/null || echo 0)

            echo "$(ts_iso),${vm_ip},${drx},${dtx},${rx_mbps},${tx_mbps}" >> "$f"
            prev_rx[$vm_ip]=$rx
            prev_tx[$vm_ip]=$tx
        done
    done
}

# =============================================================================
# Metric 8 — Migration events (parse from training log)
# Written by train.py already in training_log.csv; we extract migration events
# =============================================================================
collect_migration_events() {
    local f="$OUT_DIR/migration_timing.csv"
    echo "timestamp,episode,n_migrations,failed_migrations,energy_saving_pct,migration_cost_est_s" > "$f"

    local last_ep=0
    while true; do
        if [[ -f "results/training_log.csv" ]]; then
            tail -1 results/training_log.csv | while IFS=, read ep rew loss sav nodes pow qos mig fail orph conf eps step; do
                if [[ "$ep" != "$last_ep" && "$ep" =~ ^[0-9]+$ && "$mig" =~ ^[0-9]+$ ]]; then
                    # Estimate migration cost: 60s per DU, 8s per CU (from measurements)
                    cost=$(python3 -c "print(round(int('$mig')*0.5*60,1))" 2>/dev/null || echo 0)
                    echo "$(ts_iso),$ep,$mig,$fail,$sav,$cost" >> "$f"
                    last_ep=$ep
                fi
            done
        fi
        sleep "$INTERVAL"
    done
}

# =============================================================================
# Launch all collectors in background
# =============================================================================
info "Starting metric collectors..."

collect_energy          & echo $! >> "$PID_FILE"; ok "Metric 1: energy_saving.csv"
collect_du_cpu          & echo $! >> "$PID_FILE"; ok "Metric 2: du_cpu.csv"
collect_cu_cpu          & echo $! >> "$PID_FILE"; ok "Metric 3: cu_cpu.csv"
collect_ue_throughput   & echo $! >> "$PID_FILE"; ok "Metric 4: ue_throughput.csv (every 60s)"
collect_bler_snr        & echo $! >> "$PID_FILE"; ok "Metric 5: bler_snr.csv"
collect_vm_resources    & echo $! >> "$PID_FILE"; ok "Metric 6: vm_resources.csv"
collect_link_util       & echo $! >> "$PID_FILE"; ok "Metric 7: link_utilization.csv"
collect_migration_events& echo $! >> "$PID_FILE"; ok "Metric 8: migration_timing.csv"

echo ""
echo "  All collectors running. PIDs in $PID_FILE"
echo "  Start training now in another terminal:"
echo "    python3 train.py --episodes 5000 --max-steps 10"
echo ""
echo "  Stop with: Ctrl-C  or  kill \$(cat $PID_FILE | tr '\\n' ' ')"
echo "════════════════════════════════════════════════════════════"

# Wait forever (until Ctrl-C or training ends)
wait
