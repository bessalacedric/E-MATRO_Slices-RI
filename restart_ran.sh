#!/bin/bash
CN_DIR="$HOME/E-matro-evaluation/cn"
cd "$CN_DIR"
echo "=== Stopping RAN ==="
docker compose -f docker-compose-ran.yml down 2>/dev/null || true
sleep 5

echo "=== Resetting DB auth state ==="
docker exec mysql mysql -u root --password=linux oai_db -e "
UPDATE AuthenticationSubscription
SET sequenceNumber='{\"sqn\": \"000000000020\", \"sqnScheme\": \"NON_TIME_BASED\", \"lastIndexes\": {\"ausf\": 0}}'
WHERE ueid LIKE '00101%';
DELETE FROM AuthenticationStatus;" 2>/dev/null
echo "  Auth reset done"

echo "=== Flushing AUSF/UDR/UDM ==="
docker compose -f docker-compose.yml restart oai-ausf oai-udr oai-udm
sleep 25

echo "=== Starting iperf3 servers ==="
docker exec oai-ext-dn pkill iperf3 2>/dev/null || true
sleep 2
for port in 5201 5202 5203 5204; do
    docker exec -d oai-ext-dn iperf3 -s -p $port
done

echo "=== Starting gNB ==="
docker compose -f docker-compose-ran.yml up -d oai-gnb
sleep 35
docker logs oai-amf 2>&1 | grep -A3 "gNBs' Information" | tail -4

echo "=== Starting UEs ==="
for SVC in oai-nr-ue oai-nr-ue2 oai-nr-ue3 oai-nr-ue4; do
    docker compose -f docker-compose-ran.yml up -d $SVC
    echo -n "  $SVC... "; sleep 90
    IP=$(docker exec $SVC ip addr show oaitun_ue1 2>/dev/null \
        | grep "inet " | awk '{print $2}' | cut -d/ -f1 || echo "")
    [[ -n "$IP" ]] && echo "$IP ✓" || echo "no IP"
done

echo "=== Final status ==="
for i in 1 2 3 4; do
    [[ $i -eq 1 ]] && N="oai-nr-ue" || N="oai-nr-ue${i}"; [[ $i -eq 2 ]] && N="oai-nr-ue2"
    IP=$(docker exec $N ip addr show oaitun_ue1 2>/dev/null \
        | grep "inet " | awk '{print $2}' | cut -d/ -f1 || echo "NO IP")
    printf "RU%d  %-14s  %s\n" $((i-1)) "$N" "$IP"
done
