#!/bin/bash
UPF_MAC=$(docker inspect oai-upf \
    --format '{{range .NetworkSettings.Networks}}{{.MacAddress}}{{end}}')
DST_VETH=$(bridge fdb show br oai_e_matro \
    | grep "$UPF_MAC" | awk '{print $3}')
UPF_PID=$(docker inspect oai-upf --format '{{.State.Pid}}')
sudo nsenter -t $UPF_PID -n ip link set eth0 promisc on
sudo tc filter del dev enp6s18 ingress 2>/dev/null || true
sudo tc qdisc add dev enp6s18 ingress 2>/dev/null || true
sudo tc filter add dev enp6s18 ingress \
    protocol ip prio 1 \
    flower ip_proto udp dst_ip 192.168.70.134 dst_port 2152 \
    action pedit ex munge eth dst set $UPF_MAC pipe \
    action mirred egress redirect dev $DST_VETH
echo "tc redirect: enp6s18 → $DST_VETH (MAC: $UPF_MAC) ✓"
