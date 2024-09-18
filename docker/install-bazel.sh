#! /usr/bin/env bash


curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor > bazel.gpg
mv bazel.gpg /etc/apt/trusted.gpg.d/
echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list

apt-get update
apt-get install -y bazel-6.1.2
ln -s /usr/bin/bazel-6.1.2 /usr/bin/bazel
