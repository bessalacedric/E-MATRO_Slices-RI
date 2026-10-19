#!/bin/bash
# deploy_multi_gnb.sh — start all gNB DU/CU containers
# Auto-generated for N_GNB=2  CTL=10.10.213.129
set -e

CTL_IP=10.10.213.129
SSH_USER=ubuntu

# ── gNB 0  DU=10.10.212.103  CU=10.10.214.110  cellid=12345678  F1=500  RFSim=4043 ─────────
echo '==> gNB 0: adding UPF route on CU node...'
ssh ubuntu@10.10.214.110 'sudo ip route del 192.168.70.134/32 2>/dev/null || true'
ssh ubuntu@10.10.214.110 'sudo ip route add 192.168.70.134/32 via 10.10.213.129 dev enp6s18'

echo '==> gNB 0: SCP configs...'
scp cn//cu.gnb0.conf ubuntu@10.10.214.110:~/oai/conf/cu.gnb0.conf
scp cn//du.gnb0.conf ubuntu@10.10.212.103:~/oai/conf/du.gnb0.conf

echo '==> gNB 0: starting CU on 10.10.214.110...'
ssh ubuntu@10.10.214.110 'mkdir -p ~/oai/conf'
ssh ubuntu@10.10.214.110 '
  docker rm -f oai-cu-gnb0 2>/dev/null || true
  docker run -d --rm --privileged --network host \
    --name oai-cu-gnb0 \
    -v ~/oai/conf/cu.gnb0.conf:/opt/oai-gnb/etc/gnb.conf \
    oaisoftwarealliance/oai-gnb:2024.w26 \
    /opt/oai-gnb/bin/nr-softmodem \
    -O /opt/oai-gnb/etc/gnb.conf --sa \
    --log_config.global_log_options level,nocolor,time
'
sleep 20
ssh ubuntu@10.10.214.110 "docker logs oai-cu-gnb0 2>&1 | grep -i 'ng setup\|amf\|error' | tail -5"

echo '==> gNB 0: starting DU on 10.10.212.103...'
ssh ubuntu@10.10.212.103 'mkdir -p ~/oai/conf'
ssh ubuntu@10.10.212.103 '
  docker rm -f oai-du-gnb0 2>/dev/null || true
  docker run -d --rm --privileged --network host \
    --name oai-du-gnb0 \
    -v ~/oai/conf/du.gnb0.conf:/opt/oai-gnb/etc/gnb.conf \
    oaisoftwarealliance/oai-gnb:2024.w26 \
    /opt/oai-gnb/bin/nr-softmodem \
    -O /opt/oai-gnb/etc/gnb.conf -E --sa --rfsim \
    --log_config.global_log_options level,nocolor,time
'
sleep 20
ssh ubuntu@10.10.212.103 "docker logs oai-du-gnb0 2>&1 | grep -i 'f1 setup\|error' | tail -5"

# ── gNB 1  DU=10.10.212.33  CU=10.10.212.196  cellid=12345679  F1=520  RFSim=4044 ─────────
echo '==> gNB 1: adding UPF route on CU node...'
ssh ubuntu@10.10.212.196 'sudo ip route del 192.168.70.134/32 2>/dev/null || true'
ssh ubuntu@10.10.212.196 'sudo ip route add 192.168.70.134/32 via 10.10.213.129 dev enp6s18'

echo '==> gNB 1: SCP configs...'
scp cn//cu.gnb1.conf ubuntu@10.10.212.196:~/oai/conf/cu.gnb1.conf
scp cn//du.gnb1.conf ubuntu@10.10.212.33:~/oai/conf/du.gnb1.conf

echo '==> gNB 1: starting CU on 10.10.212.196...'
ssh ubuntu@10.10.212.196 'mkdir -p ~/oai/conf'
ssh ubuntu@10.10.212.196 '
  docker rm -f oai-cu-gnb1 2>/dev/null || true
  docker run -d --rm --privileged --network host \
    --name oai-cu-gnb1 \
    -v ~/oai/conf/cu.gnb1.conf:/opt/oai-gnb/etc/gnb.conf \
    oaisoftwarealliance/oai-gnb:2024.w26 \
    /opt/oai-gnb/bin/nr-softmodem \
    -O /opt/oai-gnb/etc/gnb.conf --sa \
    --log_config.global_log_options level,nocolor,time
'
sleep 20
ssh ubuntu@10.10.212.196 "docker logs oai-cu-gnb1 2>&1 | grep -i 'ng setup\|amf\|error' | tail -5"

echo '==> gNB 1: starting DU on 10.10.212.33...'
ssh ubuntu@10.10.212.33 'mkdir -p ~/oai/conf'
ssh ubuntu@10.10.212.33 '
  docker rm -f oai-du-gnb1 2>/dev/null || true
  docker run -d --rm --privileged --network host \
    --name oai-du-gnb1 \
    -v ~/oai/conf/du.gnb1.conf:/opt/oai-gnb/etc/gnb.conf \
    oaisoftwarealliance/oai-gnb:2024.w26 \
    /opt/oai-gnb/bin/nr-softmodem \
    -O /opt/oai-gnb/etc/gnb.conf -E --sa --rfsim \
    --log_config.global_log_options level,nocolor,time
'
sleep 20
ssh ubuntu@10.10.212.33 "docker logs oai-du-gnb1 2>&1 | grep -i 'f1 setup\|error' | tail -5"

echo ''
echo '==> All gNBs deployed. Verify on AMF:'
echo "docker logs oai-amf 2>&1 | grep -i 'gnb\|ng setup'"
