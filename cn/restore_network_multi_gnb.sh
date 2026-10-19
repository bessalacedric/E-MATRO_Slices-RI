#!/bin/bash
# restore_network_multi_gnb.sh — daily restore after testbed restart
# Auto-generated for N_GNB=2
set -e

CTL_IP=10.10.213.129
SSH_USER=ubuntu

echo '==> Step 1: GTP-U tc flower redirect...'
UPF_PID=$(docker inspect oai-upf --format '{{{{.State.Pid}}}}')
UPF_VETH=$(ip link | grep -B1 "if${UPF_PID}:" | grep -oP 'veth\w+' | head -1)
UPF_MAC=$(cat /sys/class/net/${UPF_VETH}/address)
sudo tc qdisc add dev enp6s18 ingress 2>/dev/null || true
sudo tc filter del dev enp6s18 ingress 2>/dev/null || true
sudo tc filter add dev enp6s18 ingress \
  protocol ip flower ip_proto udp dst_port 2152 \
  action pedit ex munge eth dst set ${UPF_MAC} pipe \
  action mirred egress redirect dev ${UPF_VETH}
echo "  GTP-U redirect: enp6s18 -> ${UPF_VETH} (${UPF_MAC})"

echo '==> Step 2: Remove Docker FORWARD DROP rules...'
sudo nft -a list chain ip filter DOCKER 2>/dev/null \
  | grep 'drop' | grep -oP 'handle \d+' | awk '{print $2}' \
  | while read h; do
    sudo nft delete rule ip filter DOCKER handle $h 2>/dev/null && echo "  Removed DROP handle $h"
  done

echo '==> Step 3: Flush stale UE auth state...'
docker exec mysql mysql -u root --password=linux oai_db \
  -e 'DELETE FROM AuthenticationStatus;' 2>/dev/null || true
docker compose -f cn/docker-compose.yml restart oai-ausf oai-udr oai-udm
sleep 20

echo '==> Step 4: Deploy gNBs...'
bash cn//deploy_multi_gnb.sh

echo '==> Step 5: Attach UEs (90s gap between each)...'
docker compose -f cn/docker-compose-ran-multi.yml down 2>/dev/null || true
docker compose -f cn/docker-compose-ran-multi.yml up -d oai-nr-ue
sleep 90
IP=$(docker exec oai-nr-ue ip addr show oaitun_ue1 2>/dev/null | grep 'inet ' | awk '{print $2}' | cut -d/ -f1)
echo "  oai-nr-ue: ${IP:-NO IP}"
docker compose -f cn/docker-compose-ran-multi.yml up -d oai-nr-ue2
sleep 90
IP=$(docker exec oai-nr-ue2 ip addr show oaitun_ue1 2>/dev/null | grep 'inet ' | awk '{print $2}' | cut -d/ -f1)
echo "  oai-nr-ue2: ${IP:-NO IP}"
docker compose -f cn/docker-compose-ran-multi.yml up -d oai-nr-ue3
sleep 90
IP=$(docker exec oai-nr-ue3 ip addr show oaitun_ue1 2>/dev/null | grep 'inet ' | awk '{print $2}' | cut -d/ -f1)
echo "  oai-nr-ue3: ${IP:-NO IP}"
docker compose -f cn/docker-compose-ran-multi.yml up -d oai-nr-ue4
sleep 90
IP=$(docker exec oai-nr-ue4 ip addr show oaitun_ue1 2>/dev/null | grep 'inet ' | awk '{print $2}' | cut -d/ -f1)
echo "  oai-nr-ue4: ${IP:-NO IP}"
docker compose -f cn/docker-compose-ran-multi.yml up -d oai-nr-ue5
sleep 90
IP=$(docker exec oai-nr-ue5 ip addr show oaitun_ue1 2>/dev/null | grep 'inet ' | awk '{print $2}' | cut -d/ -f1)
echo "  oai-nr-ue5: ${IP:-NO IP}"
docker compose -f cn/docker-compose-ran-multi.yml up -d oai-nr-ue6
sleep 90
IP=$(docker exec oai-nr-ue6 ip addr show oaitun_ue1 2>/dev/null | grep 'inet ' | awk '{print $2}' | cut -d/ -f1)
echo "  oai-nr-ue6: ${IP:-NO IP}"
docker compose -f cn/docker-compose-ran-multi.yml up -d oai-nr-ue7
sleep 90
IP=$(docker exec oai-nr-ue7 ip addr show oaitun_ue1 2>/dev/null | grep 'inet ' | awk '{print $2}' | cut -d/ -f1)
echo "  oai-nr-ue7: ${IP:-NO IP}"
docker compose -f cn/docker-compose-ran-multi.yml up -d oai-nr-ue8
sleep 90
IP=$(docker exec oai-nr-ue8 ip addr show oaitun_ue1 2>/dev/null | grep 'inet ' | awk '{print $2}' | cut -d/ -f1)
echo "  oai-nr-ue8: ${IP:-NO IP}"

echo '==> Restore complete.'
