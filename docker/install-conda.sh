#! /usr/bin/env bash

wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh -O /opt/minforge.sh
/bin/bash /opt/minforge.sh -b -p /opt/install/miniconda
conda update --all --yes
conda clean -tipy
rm -f /opt/minforge.sh
echo "auto_update_conda: False" >> /opt/install/miniconda/.condarc
echo "ssl_verify: False" >> /opt/install/miniconda/.condarc
echo ". /opt/install/miniconda/etc/profile.d/conda.sh" >> /etc/skel/.bashrc
ln -s /opt/install/miniconda/etc/profile.d/conda.sh /etc/profile.d/conda.sh
echo ". /opt/install/miniconda/etc/profile.d/conda.sh" >> /root/.bashrc
echo "conda activate base" >> /root/.bashrc

export PATH=$PATH:/opt/install/miniconda/bin

conda install -k -y tini
chmod -R ugo+w /opt/install/miniconda
conda clean -tipy

conda create -n legere

echo "conda activate legere" >> /root/.bashrc

conda run --no-capture-out -n legere \
  conda install -c conda-forge -y cmake cython ninja openblas \
    python==3.10.6 rust openssh gdb==14.2

conda run --no-capture-out -n legere \
  python -m pip install build cffi gin-config numpy pyarrow \
    pybind11 pybind11-global scikit-build tensorflow typing_extensions
