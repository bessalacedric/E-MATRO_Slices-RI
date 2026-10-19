# E-MATRO++ on SLICES-RI — Artifact Appendix

## A.1 Abstract

This artifact contains the deployment scripts, configuration files and evaluation code needed to reproduce the E-MATRO++ experiments on the [SLICES-RI](https://portal.slices-ri.eu) research infrastructure. E-MATRO++ learns how to schedule the workload of O-RAN virtual network functions (CUs and DUs) across a pool of servers in order to reduce energy consumption during operation.

The setup deploys an OpenAirInterface (OAI) 5G core and a disaggregated RAN (2 CUs split into CU-CP/CU-UP, and 2 DUs) across 16 virtual machines on the SLICES `be-gent1-bi-vm1` site (Ghent, Belgium):

- **1 controller VM** hosts the 5G core, the simulated UEs, the E-MATRO++ controller and the gateway between the RAN and the core.
- **15 server VMs** host the RAN VNFs and form the pool of hosts that E-MATRO++ schedules workload on.

> **Note:** The IP addresses shown in examples come from our own reservation. Yours will be different, so always replace them with the values from `slices bi list-resources`.

## A.2 Artifact check-list (meta-information)

| Item | Value |
|---|---|
| **Algorithm** | E-MATRO++: energy-aware scheduling of O-RAN VNF workloads, trained with centralized training and decentralized execution (CTDE) |
| **Program** | Python (PyTorch, Gymnasium, ZeroMQ) controller; OAI 5G core, RAN and UEs in Docker |
| **Compilation** | None. Python dependencies are installed with pip, and OAI is used as prebuilt Docker images |
| **RAN software** | `oaisoftwarealliance/oai-gnb:2024.w26` and `oaisoftwarealliance/oai-nr-cuup:2024.w26` |
| **Run-time environment** | Ubuntu 24.04.1, Python 3, Docker (installed by the steps below) |
| **Hardware** | SLICES-RI virtual machines: 1 × `large` (controller) and 15 × `medium` (servers) |
| **Execution** | Bash commands run from the controller VM over SSH |
| **Metrics** | _TODO: e.g. total energy consumption, number of active hosts, VNF migration count, training reward_ |
| **Output** | Infrastructure metrics in `results/metrics/`; training log in `results/8rus_6nodes/training_log.csv` |
| **Experiments** | _TODO: list one item per figure or table reproduced_ |
| **Disk space required** | _TODO_ |
| **Workflow preparation time** | About 1–2 hours (VM creation, dependency installation, image pulls, deployment) |
| **Experiment duration** | _TODO: duration of a 10,000-episode training run_ |
| **Publicly available** | Yes |
| **Code license** | _TODO: e.g. Apache-2.0_ |
| **Archived (DOI)** | _TODO: Zenodo DOI after the first release_ |

## A.3 Description

### A.3.1 How the artifact is delivered

The artifact is distributed through the public GitHub repository:
<https://github.com/bessalacedric/E-MATRO_Slices-RI>

The SLICES-RI tooling used for reservation and access comes from the LATINCOM 2025 hands-on tutorial:
<https://gitlab.inria.fr/slices-ri/latincom-2025>

### A.3.2 Hardware dependencies

The experiments require a SLICES-RI account and an experiment on the `be-gent1-bi-vm1` site with the following VMs.

| VM | Flavor | Count | Role |
|---|---|---|---|
| `vm-controller` | `large` | 1 | 5G core, UEs, E-MATRO++ controller, gateway |
| `vm-server-1` … `vm-server-15` | `medium` | 15 | RAN VNF hosts (CU-CP, CU-UP, DU) |

The setup can be scaled to more or fewer server VMs by changing the loop bound in [A.4.3](#a43-create-the-vms) and the list in `hosts.txt`.

### A.3.3 Software dependencies

**On your local machine**

- Linux with Python 3, `python3-venv` and `git`
- `slices-cli` and `publish_results`, installed in [A.4.1](#a41-install-the-slices-cli-on-your-local-machine)
- An SSH key pair, for example `~/.ssh/id_ed25519`

**On the VMs** (installed by the steps below)

- Ubuntu 24.04.1
- Docker
- `stress-ng`
- On the controller only: Python 3 virtual environment with `torch`, `numpy`, `pandas`, `gymnasium`, `pyzmq`, `psutil`, `matplotlib`, `tqdm`, `scikit-learn`

## A.4 Installation

### A.4.1 Install the SLICES CLI on your local machine

```bash
sudo apt-get update -qq && sudo apt-get install -y python3-pip python3-venv git

python3 -m venv ~/slices-venv
source ~/slices-venv/bin/activate

pip install slices-cli --extra-index-url=https://doc.slices-ri.eu/pypi/
pip install git+https://gitlab.inria.fr/slices-ri/publish_results.git
```

Link the CLI to your SLICES-RI account:

```bash
slices auth login
```

Open the URL that the command prints and authorize the CLI on the SLICES-RI portal.

### A.4.2 Join the project and create the experiment

The experiments run in the `post5g-ran` project. First, request to join it at
<https://portal.slices-ri.eu/invite/post5g-ran>. Once you are accepted, run:

```bash
slices project use post5g-ran
```

If you are reusing our existing reservation, skip the next command. Otherwise, create the experiment:

```bash
slices experiment create e-matro-eval --duration 5d
```

Point the CLI at the site and experiment so that later commands don't need extra flags:

```bash
export SLICES_BI_SITE_ID=be-gent1-bi-vm1
export SLICES_EXPERIMENT=e-matro-eval
```

> **Tip:** Reservations expire. To extend one before it ends, run
> `slices bi extend --experiment e-matro-eval --duration 5d`.

Register your SSH public key so you can reach the VMs. If you don't have a key yet, create one first with `ssh-keygen -t ed25519`.

```bash
slices pubkey register ~/.ssh/id_ed25519.pub
```

### A.4.3 Create the VMs

The controller needs more resources, so it uses the `large` flavor:

```bash
slices bi create vm-controller --duration 5d --image "Ubuntu 24.04.1" --flavor large
```

The 15 servers use the `medium` flavor:

```bash
for i in {1..15}; do
    echo "Creating vm-server-$i..."
    slices bi create "vm-server-$i" --duration 5d --image "Ubuntu 24.04.1" --flavor medium
done
```

Check the status of the resources until every VM is `up`:

```bash
slices bi list-resources
# Compact view:
slices bi list-resources | sed 's/[┃│┡┘└┐┌├┤┬┴┼━]/ /g' | grep -E "vm-|up"
```

Write down the IP address of `vm-controller` and of each `vm-server-N`. They are needed in the next steps.

### A.4.4 Connect to the controller

Open an SSH session from your local machine to the controller **with agent forwarding (`-A`)**. Agent forwarding lets the controller reach the servers using your registered key during setup.

```bash
ssh -A -J proxy@bastion2.slices-be.eu ubuntu@<CONTROLLER_IP>
```

You can also use `slices bi ssh vm-controller`, but it does not forward your SSH agent, so the key-distribution step in [A.4.6](#a46-configure-the-server-vms-from-the-controller) would fail.

**All remaining commands run on `vm-controller` unless stated otherwise.**

### A.4.5 Set up the controller

Install system packages, the Python environment and Docker:

```bash
sudo apt-get update -qq && \
sudo apt-get install -y python3-pip python3-venv git tmux stress-ng nmap && \
python3 -m venv ~/e-matro-env && \
source ~/e-matro-env/bin/activate && \
pip install torch numpy pandas gymnasium pyzmq psutil matplotlib tqdm scikit-learn

curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker ubuntu
```

Log out and back in so that the `docker` group membership takes effect.

Clone the artifact and install its dependencies:

```bash
git clone https://github.com/bessalacedric/E-MATRO_Slices-RI.git
cd E-MATRO_Slices-RI
./utils/install_dependencies.sh
```

Set the variables used throughout the rest of this guide:

```bash
export REPO=~/E-MATRO_Slices-RI
export CONTROLLER_IP=<CONTROLLER_IP>      # e.g. 10.10.213.129
```

#### The `hosts.txt` file

Every loop in this guide reads the server IP addresses from `~/hosts.txt` instead of hard-coding them. Create this file on `vm-controller` with the IP addresses of the 15 servers, **one per line, in order** (line 1 = `vm-server-1`, line 15 = `vm-server-15`):

```bash
cat > ~/hosts.txt << 'EOF'
10.10.212.103
10.10.214.110
10.10.212.33
...
10.10.209.253
EOF
```

The order matters, because later steps find the DU and CU hosts by line number. When you make a new reservation, update this file and nothing else. The repository contains `hosts.example.txt` as a format reference.

### A.4.6 Configure the server VMs from the controller

**1. Check network reachability.** All VMs share the same network, so every server should answer:

```bash
i=1
while read -r IP; do
    echo -n "vm-server-$i ($IP): "
    ping -c2 -q "$IP" 2>/dev/null | grep rtt || echo "UNREACHABLE"
    i=$((i+1))
done < ~/hosts.txt
```

**2. Authorize the controller's key on every server.** Create a key on the controller if it does not have one, then copy it to each server. This step relies on the agent forwarding set up in [A.4.4](#a44-connect-to-the-controller).

```bash
[ -f ~/.ssh/id_ed25519 ] || ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519
PUBKEY=$(cat ~/.ssh/id_ed25519.pub)

while read -r IP; do
    echo -n "$IP: "
    ssh -n -o StrictHostKeyChecking=accept-new ubuntu@"$IP" \
        "mkdir -p ~/.ssh && echo '$PUBKEY' >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys && echo OK"
done < ~/hosts.txt
```

Every line should end with `OK`.

**3. Verify passwordless SSH.** This check runs without the forwarded agent, so it confirms that the controller's own key now works:

```bash
while read -r IP; do
    echo -n "$IP: "
    SSH_AUTH_SOCK= ssh -n ubuntu@"$IP" "echo OK" 2>/dev/null || echo "FAILED"
done < ~/hosts.txt
```

**4. Install `stress-ng` and Docker on all servers, in parallel:**

```bash
while read -r IP; do
    ssh -n ubuntu@"$IP" "
        sudo apt-get update -qq && sudo apt-get install -y -qq stress-ng >/dev/null
        if which docker >/dev/null 2>&1; then
            echo '$IP: docker already installed'
        else
            curl -fsSL https://get.docker.com | sudo sh >/dev/null && sudo usermod -aG docker ubuntu && echo '$IP: docker installed'
        fi
    " &
done < ~/hosts.txt
wait
```

## A.5 Experiment workflow

Deploy step by step and check the result of each step before moving to the next one. Don't run everything in one go.

| Step | What it does | Re-run when |
|---|---|---|
| [A.5.2](#a52-start-the-5g-core) | Start the 5G core and provision subscribers | — |
| [A.5.3](#a53-route-ran-traffic-to-the-core) | NAT rules exposing the AMF and UPF | Controller reboot |
| [A.5.4](#a54-fix-the-gtp-u-dataplane) | Redirect GTP-U traffic to the UPF | **Every UPF restart** |
| [A.5.5](#a55-fix-the-docker-forward-rules) | Remove Docker's DROP rules | **Every core restart** |
| [A.5.6](#a56-generate-the-ran-configuration) | Generate the CU/DU configuration files | Topology change |
| [A.5.7](#a57-distribute-the-ran-configuration-and-images)–[A.5.9](#a59-deploy-the-dus) | Deploy the CUs and DUs | — |
| [A.5.10](#a510-attach-the-ues) | Attach the UEs | — |
| [A.5.11](#a511-run-e-matro) | Run E-MATRO++ | — |

### A.5.1 Initial RAN layout

The experiment starts from a small initial deployment. E-MATRO++ then learns how to schedule the workload of these VNFs across the server pool to save energy.

| Host | IP (line in `hosts.txt`) | Function |
|---|---|---|
| `vm-controller` | `$CONTROLLER_IP` | 5G core (AMF, UPF, …), 4 UEs, E-MATRO++ |
| `vm-server-1` | line 1 | DU node 0 → attached to CU node 2 |
| `vm-server-2` | line 2 | DU node 1 → attached to CU node 3 |
| `vm-server-3` | line 3 | CU node 2 (CU-CP + CU-UP) |
| `vm-server-4` | line 4 | CU node 3 (CU-CP + CU-UP) |
| `vm-server-5` … `15` | lines 5–15 | Initially idle; available to E-MATRO++ |

Define shortcuts for these hosts:

```bash
DU0=$(sed -n 1p ~/hosts.txt)
DU1=$(sed -n 2p ~/hosts.txt)
CU2=$(sed -n 3p ~/hosts.txt)
CU3=$(sed -n 4p ~/hosts.txt)
```

### A.5.2 Start the 5G core

Start the OAI core functions on the controller and give them time to come up:

```bash
cd "$REPO/cn"
docker compose -f docker-compose.yml up -d
sleep 60
```

Provision the subscriber database with the UE credentials:

```bash
python3 provision_db.py
```

**Check:** the AMF has started and is listening on SCTP port 38412 (N2).

```bash
docker logs oai-amf 2>&1 | grep "Started"
docker exec oai-amf ss -lnp | grep 38412
```

### A.5.3 Route RAN traffic to the core

The core containers sit on the Docker network `192.168.70.0/26` inside the controller. These rules expose the AMF (N2, SCTP 38412) and the UPF (N3, GTP-U UDP 2152) on the controller's address, so that the servers can reach them.

```bash
sudo sysctl -w net.ipv4.ip_forward=1

# Forward N2 (SCTP 38412) to the AMF and N3 (UDP 2152) to the UPF
sudo iptables -t nat -A PREROUTING -d "$CONTROLLER_IP" -p sctp --dport 38412 -j DNAT --to-destination 192.168.70.132:38412
sudo iptables -t nat -A PREROUTING -d "$CONTROLLER_IP" -p udp  --dport 2152  -j DNAT --to-destination 192.168.70.134:2152
sudo iptables -t nat -A POSTROUTING -s 10.10.0.0/16 -d 192.168.70.0/26 -j MASQUERADE

# Allow forwarded traffic to and from the AMF and UPF
for DST in 192.168.70.132 192.168.70.134; do
    sudo iptables -I FORWARD -d "$DST" -j ACCEPT
    sudo iptables -I FORWARD -s "$DST" -j ACCEPT
done
```

> These rules are not persistent. Re-apply them after a reboot of the controller.

Check from a CU host that the AMF port is reachable through the controller:

```bash
ssh ubuntu@"$CU2" "sudo apt-get install -y -qq nmap >/dev/null; sudo nmap -sY -p 38412 $CONTROLLER_IP | grep 38412"
```

The port should show as `open`.

### A.5.4 Fix the GTP-U dataplane

> **Re-run this step after every UPF restart.** The UPF gets a new virtual interface each time its container starts, so the redirect rule must be rebuilt.

User-plane traffic (GTP-U, UDP 2152) arriving from the CU-UPs on the controller's network interface is redirected straight to the UPF container's virtual interface, with the destination MAC rewritten to the UPF's.

First, find the controller interface that faces the servers. In our reservation it was `enp6s18`.

```bash
NIC=$(ip -o route get "$(head -1 ~/hosts.txt)" | grep -oP 'dev \K\S+')
echo "Server-facing interface: $NIC"
```

Find the host-side virtual interface of the UPF container and its MAC address. This assumes the UPF's N3 interface is `eth0` inside the container.

```bash
UPF_PID=$(docker inspect oai-upf --format '{{.State.Pid}}')
PEER_IDX=$(sudo nsenter -t "$UPF_PID" -n ip -o link show eth0 | grep -oP '@if\K[0-9]+')
UPF_VETH=$(ip -o link | awk -F': ' -v i="$PEER_IDX" '$1 == i {print $2}' | cut -d@ -f1)
UPF_MAC=$(sudo nsenter -t "$UPF_PID" -n cat /sys/class/net/eth0/address 2>/dev/null \
          || docker exec oai-upf cat /sys/class/net/eth0/address)
echo "UPF veth: $UPF_VETH, UPF MAC: $UPF_MAC"
```

Install the redirect rule. Recreating the ingress qdisc first removes any stale rule from a previous UPF instance.

```bash
sudo tc qdisc del dev "$NIC" ingress 2>/dev/null || true
sudo tc qdisc add dev "$NIC" ingress
sudo tc filter add dev "$NIC" ingress \
    protocol ip flower ip_proto udp dst_port 2152 \
    action pedit ex munge eth dst set "$UPF_MAC" pipe \
    action mirred egress redirect dev "$UPF_VETH"

echo "GTP-U redirect installed → $UPF_VETH ($UPF_MAC)"
```

**Check:** the filter is listed.

```bash
sudo tc filter show dev "$NIC" ingress
```

### A.5.5 Fix the Docker FORWARD rules

> **Re-run this step after every core restart.** Docker re-creates its DROP rules whenever the core's containers are recreated.

Docker's isolation rules drop traffic forwarded from the servers to the core containers. Remove the DROP rules from Docker's chain:

```bash
sudo nft -a list chain ip filter DOCKER 2>/dev/null \
    | grep "drop" | grep -oP "handle \d+" | awk '{print $2}' \
    | while read -r h; do
        sudo nft delete rule ip filter DOCKER handle "$h"
        echo "Removed DROP rule handle $h"
    done
```

> This relaxes Docker's network isolation on the controller. That's acceptable on an isolated testbed VM, but don't apply it on a shared or production host.

### A.5.6 Generate the RAN configuration

Generate the CU and DU configuration files from the topology defined in `topology.py`. The number of gNBs is read from `topology.N_GNB`.

```bash
cd "$REPO"
source ~/e-matro-env/bin/activate
python3 generate_conf_multi_gnb.py \
    --n-gnb "$(python3 -c 'import topology as t; print(t.N_GNB)')" \
    --out-dir cn/ --deploy
```

_TODO: state exactly what `--deploy` does. If it already copies the files to the servers and starts the CUs and DUs, [A.5.7](#a57-distribute-the-ran-configuration-and-images)–[A.5.9](#a59-deploy-the-dus) describe the manual equivalent and should be marked as such._

### A.5.7 Distribute the RAN configuration and images

Copy the CU and DU configuration files and Compose files to every server:

```bash
while read -r IP; do
    ssh -n ubuntu@"$IP" "mkdir -p ~/oai/conf"
    scp -q "$REPO"/cn/conf/du.node*.conf "$REPO"/cn/conf/cu.node*.conf ubuntu@"$IP":~/oai/conf/
    scp -q "$REPO"/cn/docker-compose-du.yml "$REPO"/cn/docker-compose-cu.yml ubuntu@"$IP":~/oai/
    echo "Files copied to $IP"
done < ~/hosts.txt
```

Pull the images in parallel. The gNB image (used by the DUs and the CU-CPs) goes on every server, and the CU-UP image goes on the CU hosts:

```bash
while read -r IP; do
    ssh -n ubuntu@"$IP" "docker pull -q oaisoftwarealliance/oai-gnb:2024.w26 && echo 'gNB image ready on $IP'" &
done < ~/hosts.txt
for IP in "$CU2" "$CU3"; do
    ssh -n ubuntu@"$IP" "docker pull -q oaisoftwarealliance/oai-nr-cuup:2024.w26 && echo 'CU-UP image ready on $IP'" &
done
wait
echo "All images pulled"
```

### A.5.8 Deploy the CUs

CUs must start **before** the DUs. On each CU host, start the CU-CP first, then the CU-UP.

```bash
# CU node 2 on vm-server-3
ssh ubuntu@"$CU2" "cd ~/oai && docker compose -f docker-compose-cu.yml up -d cucp-node2"
sleep 20
ssh ubuntu@"$CU2" "cd ~/oai && docker compose -f docker-compose-cu.yml up -d cuup-node2"

# CU node 3 on vm-server-4
ssh ubuntu@"$CU3" "cd ~/oai && docker compose -f docker-compose-cu.yml up -d cucp-node3"
sleep 20
ssh ubuntu@"$CU3" "cd ~/oai && docker compose -f docker-compose-cu.yml up -d cuup-node3"

echo "Waiting 30 s for the CU-CPs to register with the AMF..."
sleep 30
```

**Check:** both CU-CPs should appear in the AMF's gNB table, and the CU-CP logs should show a successful N2 connection.

```bash
docker logs oai-amf 2>&1 | grep -A8 "gNBs' Information" | tail -10
ssh ubuntu@"$CU2" "docker logs oai-cucp-node2 2>&1 | grep -E 'AMF|N2|connect|error' | tail -10"
ssh ubuntu@"$CU3" "docker logs oai-cucp-node3 2>&1 | grep -E 'AMF|N2|connect|error' | tail -10"
```

### A.5.9 Deploy the DUs

Only continue once both CU-CPs are connected to the AMF.

```bash
# DU node 0 on vm-server-1, attached to CU node 2
ssh ubuntu@"$DU0" "cd ~/oai && docker compose -f docker-compose-du.yml up -d oai-du-node0"
sleep 20
ssh ubuntu@"$DU0" "docker logs oai-du-node0 2>&1 | grep -E 'F1|CU|connect|error|sctp' | tail -10"

# DU node 1 on vm-server-2, attached to CU node 3
ssh ubuntu@"$DU1" "cd ~/oai && docker compose -f docker-compose-du.yml up -d oai-du-node1"
sleep 20
ssh ubuntu@"$DU1" "docker logs oai-du-node1 2>&1 | grep -E 'F1|CU|connect|error|sctp' | tail -10"
```

**Check:** the F1 setup between each DU and its CU-CP should have succeeded.

```bash
ssh ubuntu@"$CU2" "docker logs oai-cucp-node2 2>&1 | grep -E 'F1_SETUP|f1ap|SETUP' | tail -10"
ssh ubuntu@"$CU3" "docker logs oai-cucp-node3 2>&1 | grep -E 'F1_SETUP|f1ap|SETUP' | tail -10"
docker logs oai-amf 2>&1 | grep -A8 "gNBs' Information" | tail -10
```

The RAN is now deployed, and the initial layout from [A.5.1](#a51-initial-ran-layout) is running.

### A.5.10 Attach the UEs

First, clear stale authentication state left by earlier runs and restart the authentication functions of the core:

```bash
cd "$REPO"
docker exec mysql mysql -u root --password=linux oai_db \
    -e "DELETE FROM AuthenticationStatus;"
docker compose -f cn/docker-compose.yml restart oai-ausf oai-udr oai-udm
sleep 20
```

Attach the 4 UEs one at a time, 90 seconds apart. Attaching them simultaneously can cause registration failures.

```bash
for UE in oai-nr-ue oai-nr-ue2 oai-nr-ue3 oai-nr-ue4; do
    docker compose -f cn/docker-compose-ran-multi.yml up -d "$UE"
    sleep 90
    IP=$(docker exec "$UE" ip addr show oaitun_ue1 2>/dev/null \
        | grep "inet " | awk '{print $2}' | cut -d/ -f1)
    echo "$UE tunnel IP: ${IP:-NOT ATTACHED}"
done
```

**Check:** each UE should print a tunnel IP address. `NOT ATTACHED` means the UE failed to register. In that case, check the AMF logs and the dataplane steps [A.5.4](#a54-fix-the-gtp-u-dataplane) and [A.5.5](#a55-fix-the-docker-forward-rules).

### A.5.11 Run E-MATRO++

Run the commands in a `tmux` session so that training survives a dropped SSH connection:

```bash
tmux new -s ematro
cd "$REPO"
source ~/e-matro-env/bin/activate
```

Start the infrastructure metrics collection in the background, sampling every 5 seconds:

```bash
bash collect_metrics.sh \
    --n-gnb 2 --interval 5 --out-dir results/metrics &
METRICS_PID=$!
```

Train E-MATRO++:

```bash
python3 train_ctde.py \
    --episodes 10000 --max-steps 10 --n-rus 8 \
    --results-dir results/8rus_6nodes
```

Follow training progress from another terminal or `tmux` window:

```bash
tail -f "$REPO"/results/8rus_6nodes/training_log.csv
```

When training finishes, stop the metrics collection:

```bash
kill "$METRICS_PID"
```

**Outputs:**

- `results/metrics/`: infrastructure metrics collected during the run
- `results/8rus_6nodes/training_log.csv`: per-episode training log

_TODO: list the scripts that turn these outputs into the paper's figures and tables._

## A.6 Evaluation and expected results

_TODO: reference values for each experiment (tables or figures from the paper), and how much they are expected to vary between runs. Results on shared virtual infrastructure can fluctuate, so state which trends should hold even when absolute values differ._

## A.7 Experiment customization

- **Number of servers:** change the loop bound in [A.4.3](#a43-create-the-vms) and the number of lines in `hosts.txt`.
- **Number of gNBs:** change `N_GNB` in `topology.py`, regenerate the configuration ([A.5.6](#a56-generate-the-ran-configuration)) and pass the same value to `collect_metrics.sh --n-gnb`.
- **Training run:** `train_ctde.py` takes `--episodes`, `--max-steps`, `--n-rus` and `--results-dir`. Use a different `--results-dir` for each configuration so runs don't overwrite each other.
- **Reservation length:** change `--duration`, or extend a running reservation with `slices bi extend`.
- **RAN layout:** edit the `cu.node*.conf` and `du.node*.conf` files and the Compose files in `$REPO/cn/`.

## A.8 Clean-up

_TODO: commands to stop the containers and release the VMs at the end of the experiment._

## A.9 Methodology

Submission, reviewing and badging methodology:

- <https://ctuning.org/ae/submission-20190109.html>
- <https://ctuning.org/ae/reviewing-20190109.html>
- <https://www.acm.org/publications/policies/artifact-review-badging>
