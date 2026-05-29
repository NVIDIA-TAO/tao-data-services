FROM nvcr.io/nvidia/vllm:26.03-py3

ARG TARGETARCH
RUN echo "Building for TARGETARCH: $TARGETARCH"

ENV TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 9.0 9.0a 10.0 11.0 12.0+PTX"

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y ffmpeg libsm6 libxext6 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Snapshot base-image package versions into a pip constraints file. Every subsequent
# `pip install` is invoked with `-c /tmp/base-constraints.txt` so transitive resolution
# can never silently downgrade or upgrade a package shipped by the curated vllm:26.03 base.
# A conflict surfaces as ResolutionImpossible at build time instead of a silent override.
# Keep only `name==version` lines: drop `pkg @ file:///opt/transfer/...whl` URL refs that
# point to NGC build-time wheel paths which are cleaned up at the end of the base image
# build. (vllm:26.03 also enforces a small global pin set via PIP_CONSTRAINT=
# /etc/pip/constraint.txt — that is layered additively on top of this file.)
RUN pip freeze --exclude-editable \
    | grep -E "^[a-zA-Z0-9_.-]+==" \
    > /tmp/base-constraints.txt

# vllm:26.03 ships setuptools-scm 9.2.2 already; this is a no-op when the constraint
# matches but keeps the Dockerfile working if the base ever drops setuptools-scm.
RUN pip install -c /tmp/base-constraints.txt "setuptools-scm<10"

# Installing requirements
# vllm:26.03 already ships opencv-python-headless==4.13.0.92; the requirements pin
# matches exactly, so this is a no-op for opencv but lets the constraints file catch
# any future base drift.
COPY docker/requirements-pip.txt requirements-pip.txt
RUN pip install -c /tmp/base-constraints.txt -r requirements-pip.txt && \
    rm -rf requirements-pip.txt

# spatialai_data_utils 1.0.0 pins tqdm==4.67.1 exactly which would downgrade base tqdm 4.67.3.
# leptonai pins anyio<=4.9.0 across all published versions, conflicting with base anyio 4.13.0.
# nvidia-eff is imported by tao-pytorch's checkpoint encryption code (eff.core/eff.codec/
# eff.validator). Its wheel has cp312 binaries but the PyPI metadata mis-declares
# requires_python==3.8.*, hence --ignore-requires-python. Its declared deps include
# legacy pins (pyarmor==8.5.8, isort<5, black==22.3.0) we don't want to honor.
# All three are installed --no-deps; usage in tao-core/tao-pytorch only touches public
# APIs that are tolerant of newer transitives.
RUN pip install --no-deps \
        --extra-index-url https://edge.urm.nvidia.com/artifactory/api/pypi/sw-metropolis-pypi/simple \
        spatialai_data_utils==1.0.0 \
    && pip install --no-deps "leptonai>=0.26.0" \
    && pip install --no-deps --ignore-requires-python "nvidia-eff==0.6.6"

# Install OpenAI CLIP and NVIDIA apex separately (require --no-build-isolation:
# CLIP for pkg_resources in Python 3.12+, apex builds against the base torch/CUDA).
RUN pip install --no-build-isolation -c /tmp/base-constraints.txt git+https://github.com/openai/CLIP.git && \
    pip install --no-cache-dir --no-build-isolation "apex @ git+https://github.com/NVIDIA/apex.git"

# NGC Security scan related fixes (Python packages).
# These packages are pinned by vllm:26.03 base-constraints.txt at older versions; we
# upgrade them in a separate layer without `-c /tmp/base-constraints.txt` and with
# `PIP_CONSTRAINT=""` to bypass both the snapshot constraints file and the base image's
# global PIP_CONSTRAINT=/etc/pip/constraint.txt. The new versions are validated to be
# backward-compatible with all reachable usages (no deprecated Pillow constants in code;
# ray is a transitive vllm dep that is not directly imported in TAO sources).
#   - Pillow 12.1.1 -> 12.2.0 (GHSA-pwv6-vv43-88gr, GHSA-whj4-6x5x-4v2j — High)
#   - ray    2.54.0 -> 2.55.0 (GHSA-mw35-8rx3-xf9r — High)
RUN PIP_CONSTRAINT="" python -m pip install --upgrade "pillow>=12.2.0" "ray>=2.55.0"

# NGC Security scan related fixes — upgrade AWS CLI v2 in-place. The vllm:26.03
# base ships v2.34.6 whose bundled libpython3.13.so.1.0 carries CVE-2026-6100
# (Critical), CVE-2026-3298, CVE-2026-4786. The official `aws/install --update`
# downloads the current AWS CLI v2 binary with a fresh bundled Python.
#
# Note: the bundled ffmpeg CVEs (libav*.so inside opencv_python*.libs/) are
# tracked as Skipped — opencv-python-headless 4.13.0.92 hard-links its main
# .so against the bundled libav*/libsw*/libpostproc shared objects via
# DT_NEEDED + versioned-symbol references, so deleting (or patchelf-stripping)
# them breaks `import cv2` at load time. No newer opencv-python-headless wheel
# is currently published on PyPI. Real fix path is a from-source rebuild with
# WITH_FFMPEG=0 or a SOSS opencv variant that links against system ffmpeg
# (follow-up).
RUN ARCH=$(uname -m) && \
    curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-${ARCH}.zip" -o /tmp/awscliv2.zip && \
    cd /tmp && unzip -q awscliv2.zip && \
    ./aws/install --update --bin-dir /usr/local/bin --install-dir /usr/local/aws-cli && \
    rm -rf /tmp/awscliv2.zip /tmp/aws

# Create user that will run commands
ARG user_id=1000
ARG user_name=developer
ARG groups=developer:1000
ARG home=/home/developer
RUN echo "ALL   ALL = (ALL) NOPASSWD: ALL" > /etc/sudoers \
    && mkdir -p "$(dirname $home)"
WORKDIR /workspace/