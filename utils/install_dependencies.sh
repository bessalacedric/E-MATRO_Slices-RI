#!/usr/bin/bash
# From https://docs.docker.com/engine/install/ubuntu/

# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update

# Install docker
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin docker-compose tmux

# Set up a pass-through cache
sudo bash -c 'cat << EOF > /etc/docker/daemon.json
{
    "registry-mirrors" : [
        "http://docker-cache.slices-be.eu"
    ]
}
EOF'

# Enable and start docker
sudo systemctl enable --now docker
sudo systemctl restart docker         # fixes a weird bug with the cache...

sudo usermod -aG docker $USER

# install wireshark/tshark
sudo add-apt-repository -y ppa:wireshark-dev/stable
sudo apt update
sudo debconf-set-selections <<< "wireshark-common wireshark-common/install-setuid boolean true"
sudo DEBIAN_FRONTEND=noninteractive apt install -y wireshark tshark

# install iperf3
sudo debconf-set-selections <<< "iperf3 iperf3/start_daemon boolean false"
sudo DEBIAN_FRONTEND=noninteractive apt install -y iperf3

# Enable IP forwarding
sudo sed -i 's/^#\s*net\.ipv4\.ip_forward\s*=.*/net.ipv4.ip_forward=1/' /etc/sysctl.conf
sudo sysctl -p

newgrp docker