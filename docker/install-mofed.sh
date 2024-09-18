#! /usr/bin/env bash

pushd /opt/MLNX_OFED_LINUX-5.4-3.5.8.0-ubuntu22.04-x86_64

dpkg -i $(echo $(find . -false -or -name 'ibverbs-providers*.deb' -or -name 'libibverbs*.deb' -or -name 'librdmacm*.deb' -or -name 'openmpi_*.deb' ))
ln -s /usr/mpi/gcc/openmpi-*/bin/mpicc /usr/bin/mpicc
ln -s /usr/mpi/gcc/openmpi-*/bin/mpicxx /usr/bin/mpicxx
ln -s /usr/mpi/gcc/openmpi-*/bin/mpif90 /usr/bin/mpif90
ln -s /usr/mpi/gcc/openmpi-*/bin/mpirun /usr/bin/mpirun
