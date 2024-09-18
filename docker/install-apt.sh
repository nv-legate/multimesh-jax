#! /usr/bin/env bash

apt-get update

apt-get install -y --no-install-recommends \
  gcc-12 g++-12 apt-transport-https autoconf automake curl \
  gdb git gnupg libnl-3-200 libnl-3-dev libnl-route-3-200 libnl-route-3-dev libtool numactl python3 \
  vim wget zlib1g-dev build-essential devscripts debhelper fakeroot

update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 12
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 12

pushd opt
curl -fsSL https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/nvidia.pub | apt-key add -
echo "deb https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/ /" >> /etc/apt/sources.list.d/nsys.list

wget https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2024_1/nsight-systems-2024.1.1_2024.1.1.59-1_amd64.deb
apt-get update
dpkg --configure -a
apt-get -y -f install
apt-get -y install nsight-systems-2024.1.1 --reinstall

mkdir -p /root/.config/NVIDIA\ Corporation
echo "CuptiUsePerThreadBuffer=false" > /root/.config/NVIDIA\ Corporation/nsys-config.ini
