ARG CUDA_VERSION
ARG CUDNN_VERSION
ARG FRAMEWORK="paxml"
FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-ubuntu22.04 AS conda_base

SHELL [ "/bin/bash", "-eo", "pipefail", "-c" ]

COPY install-apt.sh /opt
RUN /opt/install-apt.sh

ENV CUDA_PATH="/usr/local/cuda/lib64/stubs"

ARG CUSTOM_NCCL
COPY nccl /opt/nccl
COPY build-nccl.sh /opt/nccl
RUN echo ${CUSTOM_NCCL} && cd /opt/nccl && CUSTOM_NCCL="${CUSTOM_NCCL}" && ./build-nccl.sh

COPY install-bazel.sh /opt
RUN /opt/install-bazel.sh

COPY install-conda.sh /opt
RUN /opt/install-conda.sh

ENV LD_LIBRARY_PATH="/opt/install/miniconda/envs/legere/lib:${LD_LIBRARY_PATH}"
ENV PATH="/opt/install/miniconda/bin:${PATH}"

FROM conda_base AS mellanox_download

RUN cd /opt && \
  curl -fsSL http://content.mellanox.com/ofed/MLNX_OFED-5.4-3.5.8.0/MLNX_OFED_LINUX-5.4-3.5.8.0-ubuntu22.04-x86_64.tgz | tar -xz

FROM conda_base AS mellanox_install
COPY ibdev2netdev /usr/bin
COPY install-mofed.sh /opt
RUN --mount=type=cache,from=mellanox_download,source=/opt/MLNX_OFED_LINUX-5.4-3.5.8.0-ubuntu22.04-x86_64,target=/opt/MLNX_OFED_LINUX-5.4-3.5.8.0-ubuntu22.04-x86_64 \
  /opt/install-mofed.sh

FROM mellanox_install AS ucx_build
COPY download-build-ucx.sh /opt
RUN /opt/download-build-ucx.sh

FROM mellanox_install AS ucx_install
RUN --mount=type=cache,from=ucx_build,source=/opt/ucx,target=/opt/ucx \
  cd /opt/ucx && make -j install

FROM ucx_install AS te_install
COPY te /opt/te
COPY install-te.sh /opt
RUN conda run -n legere python -m pip install cmake
RUN /opt/install-te.sh -e

FROM te_install AS build_jaxlib

ARG BAZEL_CACHE=grpc://host.docker.internal:9092
ENV PATH=$PATH:/opt/bazel
COPY xla /opt/xla
COPY jax /opt/jaxlib
COPY build-jaxlib.sh /opt
RUN /opt/build-jaxlib.sh ${BAZEL_CACHE}

FROM build_jaxlib AS build_jax_plugin
ARG LEGATE_BUILD_TYPE=Release
COPY realm /opt/realm
COPY build-realm.sh /opt
RUN /opt/build-realm.sh ${LEGATE_BUILD_TYPE}
RUN cd /opt/realm && conda run --no-capture-out -n legere cmake --install build --prefix \
    /opt/install/miniconda/envs/legere

COPY zuku /opt/zuku
COPY build-zuku.sh /opt
RUN /opt/build-zuku.sh ${LEGATE_BUILD_TYPE}

COPY legate-jax /opt/legate-jax
COPY build-legate-jax.sh /opt
RUN /opt/build-legate-jax.sh

FROM build_jax_plugin AS copy_common_frameworks

COPY chex /opt/chex
COPY flax /opt/flax
COPY clu /opt/clu
COPY optax opt/optax
COPY seqio /opt/seqio
COPY orbax /opt/orbax
COPY praxis /opt/praxis

FROM copy_common_frameworks AS copy_requirements_maxtext
COPY maxtext-requirements.txt /opt/requirements.txt

FROM copy_common_frameworks AS copy_requirements_paxml
COPY paxml-requirements.txt /opt/requirements.txt

FROM copy_requirements_${FRAMEWORK} AS install_python_libs
COPY install-pip-dependencies.sh /opt
RUN /opt/install-pip-dependencies.sh -e

FROM install_python_libs AS reinstall_custom_jax
RUN conda run -n legere python -m pip uninstall -y jaxlib
RUN cd /opt/jaxlib && conda run --no-capture-out -n legere python -m pip install dist/*.whl --force-reinstall --no-deps
COPY jax /opt/jax
RUN conda run -n legere python -m pip uninstall -y jax
RUN cd /opt/jax && \
  conda run --no-capture-out -n legere python -m pip install -e . --force-reinstall --no-deps

COPY install-legate-jax.sh /opt
RUN /opt/install-legate-jax.sh -e

RUN conda run --no-capture-out -n legere python -m pip install jupyter

COPY tags /opt/tags

COPY entrypoint.sh /opt/entrypoint.sh

FROM reinstall_custom_jax AS paxml_install
COPY paxml /opt/paxml
COPY legate-jax/examples/paxml/run.py /opt
RUN cd /opt/paxml && conda run -n legere python -m pip install -e . --no-deps --no-build-isolation --force-reinstall
ENTRYPOINT [ "/opt/entrypoint.sh" ]
CMD [ "/bin/bash" ]

FROM reinstall_custom_jax AS maxtext_install
COPY maxtext /opt/maxtext
COPY legate-jax/examples/maxtext/run.py /opt
RUN cd /opt/maxtext && conda run -n legere python -m pip install -e . --no-deps --no-build-isolation --force-reinstall
COPY maxtext-fixup-pip.sh /opt
RUN /opt/maxtext-fixup-pip.sh
ENTRYPOINT [ "/opt/entrypoint.sh" ]
CMD [ "/bin/bash" ]
