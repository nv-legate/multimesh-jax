FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

SHELL [ "/bin/bash", "-eo", "pipefail", "-c" ]

RUN apt-get update && \
  apt-get install -y --no-install-recommends gcc-12 g++-12 apt-transport-https autoconf automake curl \
gdb git gnupg libnl-3-200 libnl-3-dev libnl-route-3-200 libnl-route-3-dev libtool numactl python3 \
vim wget zlib1g-dev && \
  update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 12 && \
  update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 12

ENV CUDA_PATH="/usr/local/cuda/lib64/stubs"

RUN cd /opt && \
  curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor > bazel.gpg && \
  mv bazel.gpg /etc/apt/trusted.gpg.d/ && \
  echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list && \
  apt-get update && \
  apt-get install -y bazel-6.1.2 && \
  ln -s /usr/bin/bazel-6.1.2 /usr/bin/bazel

RUN cd /opt && \
  curl -fsSL https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/nvidia.pub | apt-key add - && \
  echo "deb https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/ /" >> /etc/apt/sources.list.d/nsys.list

ENV PATH="/opt/install/miniconda/bin:${PATH}"

RUN cd /opt && \
  wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh -O \
/opt/minforge.sh && \
  /bin/bash /opt/minforge.sh -b -p /opt/install/miniconda && \
  conda update --all --yes && \
  conda clean -tipy && \
  rm -f /opt/minforge.sh && \
  echo "auto_update_conda: False" >> /opt/install/miniconda/.condarc && \
  echo "ssl_verify: False" >> /opt/install/miniconda/.condarc && \
  echo ". /opt/install/miniconda/etc/profile.d/conda.sh" >> /etc/skel/.bashrc && \
  ln -s /opt/install/miniconda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
  echo ". /opt/install/miniconda/etc/profile.d/conda.sh" >> /root/.bashrc && \
  echo "conda activate base" >> /root/.bashrc && \
  conda install -k -y tini && \
  chmod -R ugo+w /opt/install/miniconda && \
  /opt/install/miniconda/bin/conda clean -tipy

RUN cd /opt && \
  conda create -n legere

RUN cd /opt && \
  echo "conda activate legere" >> /root/.bashrc

ENV LD_LIBRARY_PATH="/opt/install/miniconda/envs/legere/lib:${LD_LIBRARY_PATH}"

COPY ibdev2netdev /usr/bin

RUN cd /opt && \
  conda run --no-capture-out -n legere curl -fsSL http://content.mellanox.com/ofed/MLNX_OFED-5.4-3.5.8.0/MLNX_OFED_LINUX-5.4-3.5.8.0-ubuntu22.04-x86_64.tgz | tar -xz

RUN cd /opt/MLNX_OFED_LINUX-5.4-3.5.8.0-ubuntu22.04-x86_64 && \
  conda run --no-capture-out -n legere dpkg -i $(echo $(find . -false -or -name 'ibverbs-providers*.deb' -or -name 'libibverbs*.deb' -or -name 'librdmacm*.deb' -or -name 'openmpi_*.deb' )) && \
  conda run --no-capture-out -n legere ln -s /usr/mpi/gcc/openmpi-*/bin/mpicc /usr/bin/mpicc && \
  conda run --no-capture-out -n legere ln -s /usr/mpi/gcc/openmpi-*/bin/mpicxx /usr/bin/mpicxx && \
  conda run --no-capture-out -n legere ln -s /usr/mpi/gcc/openmpi-*/bin/mpif90 /usr/bin/mpif90 && \
  conda run --no-capture-out -n legere ln -s /usr/mpi/gcc/openmpi-*/bin/mpirun /usr/bin/mpirun

RUN conda run --no-capture-out -n legere conda install -c conda-forge -y cmake cython ninja openblas \
python==3.11 rust

RUN cd /opt && \
  conda run --no-capture-out -n legere git clone -b master https://github.com/openucx/ucx.git ucx

RUN cd /opt/ucx && \
  conda run --no-capture-out -n legere git remote add legere https://github.com/openucx/ucx.git && \
  conda run --no-capture-out -n legere git fetch legere && \
  conda run --no-capture-out -n legere git checkout master && \
  conda run --no-capture-out -n legere ./autogen.sh && \
  conda run --no-capture-out -n legere ./contrib/configure-release --enable-mt \
--with-cuda=/usr/local/cuda --with-java=no && \
  conda run --no-capture-out -n legere make -j && \
  conda run --no-capture-out -n legere make install

RUN conda run --no-capture-out -n legere python -m pip install build cffi gin-config numpy pyarrow \
pybind11 pybind11-global scikit-build tensorflow typing_extensions

COPY legion /opt/legion

RUN cd /opt/legion && \
  conda run --no-capture-out -n legere cmake -S . -B build -DCMAKE_GENERATOR:STRING=Ninja \
-DCMAKE_CXX_COMPILER:PATH=/usr/bin/g++ -DCMAKE_C_COMPILER:PATH=/usr/bin/gcc \
-DCMAKE_LIBRARY_PATH:STRING=/usr/lib/x86_64-linux-gnu -DCMAKE_CXX_STANDARD:STRING=17 \
-DBUILD_SHARED_LIBS:BOOL=ON -DCMAKE_BUILD_TYPE:STRING=Release -DLegion_USE_CUDA:BOOL=ON \
-DLegion_MAX_DIM:STRING=4 -DLegion_USE_OpenMP:BOOL=ON -DLegion_USE_GASNet:BOOL=OFF \
-DLegion_USE_Python:BOOL=ON -DLegion_VERSION:STRING=23.11.0 -DLegion_BUILD_BINDINGS:BOOL=ON \
-DLegion_REDOP_COMPLEX:BOOL=ON -DLegion_HIJACK_CUDART:BOOL=OFF -DLegion_GPU_REDUCTIONS:BOOL=OFF \
-DLegion_REDOP_HALF:BOOL=ON -DCUDA_NVCC_FLAGS:STRING=-std=c++17 \
-DLegion_DEFAULT_LOCAL_FIELDS:STRING=0 -DLegion_NETWORKS:STRING=ucx \
-DCMAKE_INSTALL_RPATH:PATH=/opt/install/miniconda/envs/legere/lib \
-DCMAKE_INSTALL_PREFIX:PATH=/opt/install/miniconda/envs/legere && \
  conda run --no-capture-out -n legere cmake --build build && \
  conda run --no-capture-out -n legere cmake --install build --prefix \
/opt/install/miniconda/envs/legere && \
  conda run --no-capture-out -n legere cmake --install build --prefix \
/opt/install/miniconda/envs/legere

COPY xla_extension /opt/xla_extension

RUN cd /opt/xla_extension && \
  export PYTHON_BIN_PATH=/opt/install/miniconda/envs/legere/bin/python && \
  export USE_DEFAULT_PYTHON_LIB_PATH=1 && \
  export TF_NEED_ROCM=0 && \
  export TF_NEED_CUDA=1 && \
  export TF_NEED_TENSORRT=0 && \
  export TF_CUDA_CLANG=0 && \
  export TF_CUDA_COMPUTE_CAPABILITIES=8.0 && \
  export GCC_HOST_COMPILER_PATH=/usr/bin/gcc && \
  export CC_OPT_FLAGS=--Wno-sign-compare && \
  export TF_SET_ANDROID_WORKSPACE=0 && \
  export TF_NCCL_VERSION=2.17.1 && \
  conda run --no-capture-out -n legere python configure.py

COPY jaxlib /opt/jaxlib

RUN cd /opt/jaxlib && \
  export PYTHON_BIN_PATH=/opt/install/miniconda/envs/legere/bin/python && \
  export USE_DEFAULT_PYTHON_LIB_PATH=1 && \
  export TF_NEED_ROCM=0 && \
  export TF_NEED_CUDA=1 && \
  export TF_NEED_TENSORRT=0 && \
  export TF_CUDA_CLANG=0 && \
  export TF_CUDA_COMPUTE_CAPABILITIES=8.0 && \
  export GCC_HOST_COMPILER_PATH=/usr/bin/gcc && \
  export CC_OPT_FLAGS=--Wno-sign-compare && \
  export TF_SET_ANDROID_WORKSPACE=0 && \
  export TF_NCCL_VERSION=2.17.1 && \
  conda run --no-capture-out -n legere python build/build.py \
--bazel_options=--override_repository=xla=/opt/xla_extension --bazel_startup_options=--batch \
--enable_cuda && \
  conda run --no-capture-out -n legere python -m pip install dist/*.whl

COPY legate /opt/legate

RUN cd /opt/legate && \
  conda run --no-capture-out -n legere cmake -S . -B build -DCMAKE_GENERATOR:STRING=Ninja \
-DCMAKE_CXX_COMPILER:PATH=/usr/bin/g++ -DCMAKE_C_COMPILER:PATH=/usr/bin/gcc \
-DCMAKE_LIBRARY_PATH:STRING=/usr/lib/x86_64-linux-gnu -DCMAKE_CXX_STANDARD:STRING=17 \
-DBUILD_SHARED_LIBS:BOOL=ON -DCMAKE_BUILD_TYPE:STRING=Release \
-DLegion_ROOT:PATH=/opt/install/miniconda/envs/legere \
-DCMAKE_INSTALL_RPATH:PATH=/opt/install/miniconda/envs/legere/lib \
-DCMAKE_INSTALL_PREFIX:PATH=/opt/install/miniconda/envs/legere && \
  conda run --no-capture-out -n legere cmake --build build

RUN cd /opt/legate && \
  export CMAKE_ARGS="-DFIND_LEGATE_CORE_CPP:BOOL=ON -Dlegate_core_ROOT:PATH=/opt/legate/build" && \
  conda run --no-capture-out -n legere cmake --install build --prefix \
/opt/install/miniconda/envs/legere && \
  conda run --no-capture-out -n legere python -m pip install .

COPY jax /opt/jax

RUN cd /opt/jax && \
  conda run --no-capture-out -n legere python -m pip install -e .

COPY legate_xla /opt/legate_xla

RUN cd /opt/legate_xla && \
  conda run --no-capture-out -n legere cmake -S . -B build -DCMAKE_GENERATOR:STRING="Unix Makefiles" \
-DCMAKE_CXX_COMPILER:PATH=/usr/bin/g++ -DCMAKE_C_COMPILER:PATH=/usr/bin/gcc \
-DCMAKE_LIBRARY_PATH:STRING=/usr/lib/x86_64-linux-gnu -DCMAKE_CXX_STANDARD:STRING=17 \
-DBUILD_SHARED_LIBS:BOOL=ON -DCMAKE_BUILD_TYPE:STRING=Release \
-DCPM_xla_SOURCE:PATH=/opt/xla_extension -Dlegate_core_ROOT:PATH=/opt/install/miniconda/envs/legere \
-DLegateXLA_HLO_RUNNER:BOOL=OFF -DLegateXLA_ASAN:BOOL=OFF \
-DCMAKE_INSTALL_RPATH:PATH=/opt/install/miniconda/envs/legere/lib \
-DCMAKE_INSTALL_PREFIX:PATH=/opt/install/miniconda/envs/legere && \
  conda run --no-capture-out -n legere cmake --build build

RUN cd /opt/legate_xla && \
  export CMAKE_ARGS="-DCPM_xla_SOURCE:PATH=/opt/xla_extension -DLegateXLA_ROOT:PATH=/opt/legate_xla/build" && \
  export SETUPTOOLS_ENABLE_FEATURES=legacy-editable && \
  conda run --no-capture-out -n legere cmake --install build --prefix \
/opt/install/miniconda/envs/legere && \
  conda run --no-capture-out -n legere python -m pip install -e .

COPY entrypoint.sh /opt/entrypoint.sh

ENTRYPOINT [ "/opt/entrypoint.sh" ]

CMD [ "/bin/bash" ]
