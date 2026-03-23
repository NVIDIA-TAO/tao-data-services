FROM nvcr.io/nvidia/pytorch:25.09-py3

ARG TARGETARCH
RUN echo "Building for TARGETARCH: $TARGETARCH"

ENV TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 9.0 9.0a 10.0 11.0 12.0+PTX"

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y ffmpeg libsm6 libxext6 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
# upgrade pip and ensure setuptools is available (needed for pkg_resources in Python 3.12+)
RUN pip install --upgrade pip "setuptools<81"

# Remove any existing opencv installations to avoid conflicts
RUN pip uninstall -y opencv || true && \
    rm -rf /usr/local/lib/python3.12/dist-packages/cv2

# Installing requirements
COPY docker/requirements-pip.txt requirements-pip.txt
RUN pip install -r requirements-pip.txt && \
    rm -rf requirements-pip.txt

# Build vLLM from source so it uses the container's PyTorch (CUDA 13). Only on x86 (amd64).
RUN if [ "$TARGETARCH" = "amd64" ]; then \
    git clone https://github.com/vllm-project/vllm.git /tmp/vllm && \
    cd /tmp/vllm && \
    git checkout v0.14.1 && \
    python use_existing_torch.py && \
    pip install -r requirements/build.txt && \
    MAX_JOBS=4 pip install --no-build-isolation . && \
    cd / && \
    rm -rf /tmp/vllm; \
    fi

# Install OpenAI CLIP separately (requires --no-build-isolation for pkg_resources in Python 3.12+)
RUN pip install --no-build-isolation git+https://github.com/openai/CLIP.git

# Create user that will run commands
ARG user_id=1000
ARG user_name=developer
ARG groups=developer:1000
ARG home=/home/developer
RUN echo "ALL   ALL = (ALL) NOPASSWD: ALL" > /etc/sudoers \
    && mkdir -p "$(dirname $home)"
WORKDIR /workspace/