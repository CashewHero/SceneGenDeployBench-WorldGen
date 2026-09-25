# syntax=docker/dockerfile:1.7

ARG CUDA_VERSION=12.8.1
ARG UBUNTU_VERSION=24.04

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      git \
      ninja-build \
      python3 \
      python3-pip && \
    rm -rf /var/lib/apt/lists/*

ARG UV_VERSION=0.12.5
ARG TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9+PTX"
ARG MAX_JOBS=8
ARG PYTORCH3D_COMMIT=978cd99221b9e0a6a568f1d427854d73363265cf

ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    VIRTUAL_ENV=/opt/runner \
    PATH=/opt/runner/bin:${PATH} \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    MAX_JOBS=${MAX_JOBS} \
    UV_LINK_MODE=copy

RUN python3 -m pip install --break-system-packages --no-cache-dir "uv==${UV_VERSION}" && \
    uv python install 3.11 && \
    uv venv --python 3.11 ${VIRTUAL_ENV}

WORKDIR /build

COPY --link runner_wrapper/requirements.txt /build/requirements.txt

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install \
      torch==2.7.1 \
      torchvision==0.22.1 \
      --index-url https://download.pytorch.org/whl/cu128 && \
    uv pip install -r /build/requirements.txt

# Build the repository-pinned DA-2 and Sharp revisions. PyTorch3D was listed in
# the upstream .gitmodules without a matching gitlink, so pin its source here.
COPY --link submodules /build/submodules
RUN test -f /build/submodules/DA-2/src/pyproject.toml && \
    test -f /build/submodules/ml-sharp/pyproject.toml

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --no-deps /build/submodules/DA-2/src && \
    uv pip install --no-deps /build/submodules/ml-sharp && \
    FORCE_CUDA=1 CUDA_HOME=/usr/local/cuda uv pip install \
      --reinstall --no-cache --no-build-isolation \
      "git+https://github.com/facebookresearch/pytorch3d.git@${PYTORCH3D_COMMIT}"

COPY --link pyproject.toml README.md /build/model_repo/
COPY --link src /build/model_repo/src
RUN uv pip install --no-deps /build/model_repo

# Confirm that the compiled extension covers the supported GPU families and
# retains PTX for newer GPUs.
RUN extension_path="$(python -c 'import torch; import pytorch3d._C; print(pytorch3d._C.__file__)')" && \
    elf_targets="$(cuobjdump --list-elf "${extension_path}")" && \
    ptx_targets="$(cuobjdump --list-ptx "${extension_path}")" && \
    printf '%s\n%s\n' "${elf_targets}" "${ptx_targets}" && \
    for architecture in 75 80 86 89; do \
      echo "${elf_targets}" | grep -q "sm_${architecture}"; \
    done && \
    echo "${ptx_targets}" | grep -q "sm_89"

FROM nvidia/cuda:${CUDA_VERSION}-base-ubuntu${UBUNTU_VERSION} AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      libegl1 \
      libgl1 \
      libglib2.0-0 \
      libgomp1 && \
    rm -rf /var/lib/apt/lists/*

ARG TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9+PTX"

ENV VIRTUAL_ENV=/opt/runner \
    PATH=/opt/runner/bin:${PATH} \
    PYTHONPATH=/app/model_repo/src:/app/model_repo:/app \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RUNNER_PORT=58090 \
    RUNNER_NAME=worldgen-panorama-sharp \
    RUNNER_TYPE=generator \
    RUNNER_VERSION=0.1.1 \
    RUNNER_CONTRACT_VERSION=1 \
    RUNNER_IDLE_TIMEOUT_SECONDS=900 \
    RUNNER_ADAPTER=runner_wrapper.sharp_adapter:run_job \
    HF_HOME=/data/model_cache/worldgen/huggingface \
    TORCH_HOME=/data/model_cache/worldgen/torch \
    XDG_CACHE_HOME=/data/model_cache/worldgen/xdg \
    HF_HUB_DISABLE_XET=1 \
    PYTORCH_ALLOC_CONF=expandable_segments:True

COPY --link --from=builder /opt/python /opt/python
COPY --link --from=builder /opt/runner /opt/runner

WORKDIR /app/model_repo
COPY --link src /app/model_repo/src
COPY --link runner_wrapper /app/model_repo/runner_wrapper

RUN --network=none python -B -c "from worldgen.pano_depth import build_depth_model; from worldgen.pano_sharp import predict_equirectangular; from worldgen.utils.splat_utils import convert_rgbd_to_gs; from runner_wrapper.adapter import run_job as rgbd_job; from runner_wrapper.sharp_adapter import run_job as sharp_job; print(build_depth_model.__name__, predict_equirectangular.__name__, convert_rgbd_to_gs.__name__, rgbd_job.__name__, sharp_job.__name__)"

EXPOSE 58090

CMD ["python", "-m", "runner_wrapper.server"]
