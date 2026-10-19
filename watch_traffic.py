"""
watch_traffic.py — Live monitor during online training.
Run in a SEPARATE terminal alongside train_online.py.

Reads online_log.csv and tails it, printing a live dashboard.

Usage:
    # In terminal 1 (inside tmux pane 1):
    python train_online.py --steps 200

    # In terminal 2 (tmux pane 2):
    python watch_traffic.py --log results/online/online_log.csv
"""

import csv
import time
import os
import argparse


BAR = "█"
EMPTY = "░"
BAR_WIDTH = 20


def mbps_bar(mbps: float, peak: float = 40.0) -> str:
    filled = int(round(BAR * BAR_WIDTH * min(mbps / peak, 1.0)))
    empty  = BAR_WIDTH - len(filled)
    return filled + EMPTY * empty


def node_icons(active_str: str) -> str:
    icons = {"1": "■ ON ", "0": "□ OFF"}
    return "  ".join(
        f"node{i}:{icons.get(c,'?')}"
        for i, c in enumerate(active_str)
    )


def tail_log(path: str, interval: float = 5.0):
    print(f"Watching: {path}")
    print("Press Ctrl+C to stop\n")

    last_pos = 0
    last_row = {}

    while True:
        try:
            size = os.path.getsize(path)
        except FileNotFoundError:
            #print(f"Waiting for log file: {path}")
            time.sleep(2)
            continue

        if size > last_pos:
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                rows   = list(reader)
            if rows:
                last_row = rows[-1]
                last_pos = size

        if last_row:
            _print_dashboard(last_row)

        time.sleep(interval)


def _print_dashboard(row: dict):
    os.system("clear")

    step  = row.get("step", "?")
    ts    = row.get("timestamp", "")
    eps   = float(row.get("epsilon", 0))
    loss  = float(row.get("avg_loss", 0))
    E     = float(row.get("energy", 0))
    r     = float(row.get("reward", 0))
    sla   = float(row.get("sla_viol", 0))
    mig   = float(row.get("n_migrations", 0))
    sleep = int(row.get("sleeping_nodes", 0))
    opt   = float(row.get("optimality_score", 0))
    nodes = row.get("node_active", "????")

    print("╔══════════════════════════════════════════════════════════╗")
    print(f"║  E-MATRO Online Training Monitor   {ts:>8}  step {step:>5}  ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  ε = {eps:.3f}   loss = {loss:.4f}                              ║")
    print("╠═══════════════════ Traffic ══════════════════════════════╣")

    for r_idx in range(4):
        key  = f"data_flow_ru{r_idx}"
        mbps = float(row.get(key, 0))
        bar  = mbps_bar(mbps)
        print(f"║  RU {r_idx}  {bar}  {mbps:5.1f} Mbps          ║")

    print("╠═══════════════════ O-Cloud ══════════════════════════════╣")
    print(f"║  {node_icons(nodes)}          ║")
    print(f"║  Sleeping: {sleep}/4 nodes                                   ║")
    print("╠═══════════════════ Reward ═══════════════════════════════╣")
    print(f"║  Energy : {E:.4f} kWh                                    ║")
    print(f"║  SLA viol: {sla:.4f}   Migrations: {mig:.1f}                  ║")
    print(f"║  Reward  : {r:+.4f}                                      ║")
    print(f"║  Optimality: {opt:.2f}                                      ║")
    print("╚══════════════════════════════════════════════════════════╝")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log",      default="results/online/online_log.csv")
    p.add_argument("--interval", type=float, default=5.0, help="Refresh interval in seconds")
    args = p.parse_args()
    try:
        tail_log(args.log, args.interval)
    except KeyboardInterrupt:
        print("\nMonitor stopped.")


if __name__ == "__main__":
    main()