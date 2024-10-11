ARG CUDA_VERSION
ARG CUDNN_VERSION
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

FROM ucx_install AS zuku_build
ARG LEGATE_BUILD_TYPE=Release
COPY realm /opt/realm
COPY build-realm.sh /opt
RUN /opt/build-realm.sh ${LEGATE_BUILD_TYPE}
RUN cd /opt/realm && conda run --no-capture-out -n legere cmake --install build --prefix \
    /opt/install/miniconda/envs/legere

COPY zuku /opt/zuku
COPY build-zuku.sh /opt
RUN /opt/build-zuku.sh ${LEGATE_BUILD_TYPE}

FROM zuku_build AS jax_build
COPY xla /opt/xla

COPY legate-jax /opt/legate-jax
COPY build-legate-jax.sh /opt
COPY install-legate-jax.sh /opt

RUN /opt/build-legate-jax.sh
RUN /opt/install-legate-jax.sh -e

ARG BAZEL_CACHE=grpc://host.docker.internal:9092
ENV PATH=$PATH:/opt/bazel
COPY jax /opt/jaxlib
COPY build-jaxlib.sh /opt
RUN /opt/build-jaxlib.sh ${BAZEL_CACHE}
RUN cd /opt/jaxlib && /opt/build-jaxlib.sh  ${BAZEL_CACHE} && \
  conda run --no-capture-out -n legere python -m pip install dist/*.whl

COPY jax /opt/jax
RUN cd /opt/jax && \
  conda run --no-capture-out -n legere python -m pip install -e .

FROM jax_build AS te_install
COPY te /opt/te
COPY install-te.sh /opt
RUN conda run -n legere python -m pip install cmake
RUN /opt/install-te.sh -e

FROM te_install AS pip_install
COPY chex /opt/chex
COPY flax /opt/flax
COPY clu /opt/clu
COPY optax opt/optax
COPY seqio /opt/seqio
COPY orbax /opt/orbax
COPY requirements.txt /opt
COPY praxis /opt/praxis
COPY paxml /opt/paxml
COPY install-pip-dependencies.sh /opt
RUN /opt/install-pip-dependencies.sh -e

# The dependencies above might thrash jax/jaxlib
# Force re-install them now
RUN cd /opt/jaxlib && \
  python -m pip uninstall jaxlib && \
  conda run --no-capture-out -n legere \
    python -m pip install dist/*.whl --force-reinstall --no-deps
RUN cd /opt/jax && \
  python -m pip uninstall jax && \
  conda run --no-capture-out -n legere \
    python -m pip install -e . --force-reinstall --no-deps

COPY tags /opt/tags

COPY entrypoint.sh /opt/entrypoint.sh
COPY legate-jax/examples/paxml/run.py /opt

ENTRYPOINT [ "/opt/entrypoint.sh" ]

CMD [ "/bin/bash" ]
