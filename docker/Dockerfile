FROM nvcr.io/nvidia/vllm:26.03-py3

ARG TARGETARCH
RUN echo "Building for TARGETARCH: $TARGETARCH"

ENV TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 9.0 9.0a 10.0 11.0 12.0+PTX"

# Codec-royalty mitigation (TAO-2183 / FF-4): the system `ffmpeg` apt package is the
# Ubuntu GPL build, which bundles libx264/libx265 (royalty-bearing H.264/H.265 encoders).
# It is intentionally NOT installed here; a codec-disabled, LGPL-only ffmpeg is built from
# source below (see the FFMPEG_VERSION block). Keep libsm6/libxext6 — they are OpenGL/X11
# runtime deps for OpenCV, not codecs.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y libsm6 libxext6 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Codec-disabled, LGPL-only FFmpeg (TAO-2183 / FF-4)
# ---------------------------------------------------------------------------
# Built --disable-gpl --disable-nonfree --disable-everything, re-enabling only an
# allow-list of royalty-free / hardware components. NO software H.264/H.265/AAC codec
# ships: H.264 *input* decode is hardware-only via h264_cuvid (NVDEC); the FF-1 chunk /
# highlight path encodes royalty-free VP9 (libvpx_vp9 -> .webm) and samples frames to
# JPEG (mjpeg). libavformat/libavutil (containers, muxing, file I/O) are unencumbered and
# retained per Legal guidance (2026-06-12). Build-time assertions below abort the image if
# a forbidden codec leaks back in.
#
# NVIDIA_DRIVER_CAPABILITIES must include `video` so libnvcuvid.so / libnvidia-encode.so
# are exposed at runtime; without it the h264_cuvid path fails with "Cannot load libnvcuvid.so.1".
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

ARG FFMPEG_VERSION=8.1
ARG FFMPEG_SHA256=b072aed6871998cce9b36e7774033105ca29e33632be5b6347f3206898e0756a
ARG NV_CODEC_HEADERS_VERSION=n12.1.14.0
# Mirror the build ARGs into ENV so the quoted (<<'EOF') heredoc below — which bash receives
# verbatim, with no BuildKit variable substitution — can read them from the shell environment.
ENV FFMPEG_VERSION=${FFMPEG_VERSION} \
    FFMPEG_SHA256=${FFMPEG_SHA256} \
    NV_CODEC_HEADERS_VERSION=${NV_CODEC_HEADERS_VERSION}
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential yasm nasm pkg-config git curl ca-certificates xz-utils libvpx-dev libgnutls28-dev && \
    rm -rf /var/lib/apt/lists/*
RUN <<'EOF'
#!/usr/bin/env bash
set -euxo pipefail
mkdir -p /tmp/ffmpeg-build /usr/share/oss-sources/source-builds && cd /tmp/ffmpeg-build
git clone --depth 1 --branch "${NV_CODEC_HEADERS_VERSION}" https://github.com/FFmpeg/nv-codec-headers.git
tar -C /tmp/ffmpeg-build --exclude='.git' \
    -czf "/usr/share/oss-sources/source-builds/nv-codec-headers-${NV_CODEC_HEADERS_VERSION}.tar.gz" nv-codec-headers
make -C nv-codec-headers -j"$(nproc)" && make -C nv-codec-headers install
curl -fsSL "https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.xz" -o ffmpeg.tar.xz
echo "${FFMPEG_SHA256}  ffmpeg.tar.xz" | sha256sum -c -
cp ffmpeg.tar.xz "/usr/share/oss-sources/source-builds/ffmpeg-${FFMPEG_VERSION}.tar.xz"
tar -xf ffmpeg.tar.xz
cd "ffmpeg-${FFMPEG_VERSION}"
./configure \
  --prefix=/usr/local \
  --disable-debug --disable-doc --disable-static --enable-shared --enable-pic \
  --enable-gnutls \
  --disable-nonfree --disable-gpl \
  --disable-encoders --disable-decoders --disable-everything \
  --enable-libvpx \
  --enable-encoder=libvpx_vp9 --enable-encoder=mjpeg \
  --enable-decoder=h264_cuvid --enable-decoder=vp8 --enable-decoder=vp9 --enable-decoder=mjpeg \
  --enable-nvenc --enable-nvdec \
  --enable-protocol=file \
  --enable-muxer=webm --enable-muxer=image2 \
  --enable-demuxer=mov --enable-demuxer=matroska --enable-demuxer=image2 \
  --enable-parser=h264 --enable-parser=vp8 --enable-parser=vp9 \
  --enable-bsf=h264_mp4toannexb --enable-bsf=vp9_superframe \
  --enable-filter=fps --enable-filter=scale --enable-filter=format \
  --extra-cflags="-I/usr/local/include" --extra-ldflags="-L/usr/local/lib"
make -j"$(nproc)"
make install
ldconfig
rm -rf /tmp/ffmpeg-build
# ---- build-time codec assertions (fail the build, not the OSRB audit) ----
ffmpeg -hide_banner -version | head -n 3
! ffmpeg -buildconf | grep -E "enable-gpl|enable-nonfree"
# No software / GPL H.264/H.265 encoder may exist (this image enables NO h264 encoder at all).
for forbidden in libx264 libopenh264 libx265 h264 hevc aac; do
  if ffmpeg -hide_banner -encoders 2>/dev/null | awk '{print $2}' | grep -Fx "$forbidden" >/dev/null; then
    echo "ERROR: forbidden encoder '$forbidden' leaked into the image (GPL/patent-encumbered)" >&2; exit 1; fi
done
# Required royalty-free encoders. NOTE: the configure component is `libvpx_vp9` (underscore)
# but `ffmpeg -encoders` lists the runtime encoder name as `libvpx-vp9` (hyphen) — which is
# the name FF-1's video_utils.py passes to `-c:v`.
for enc in libvpx-vp9 mjpeg; do
  ffmpeg -hide_banner -encoders 2>/dev/null | awk '{print $2}' | grep -Fx "$enc" >/dev/null \
    || { echo "ERROR: expected encoder '$enc' to be enabled" >&2; exit 1; }
done
# Required decoders (H.264 input decode is hardware-only via h264_cuvid).
for dec in h264_cuvid vp8 vp9 mjpeg; do
  ffmpeg -hide_banner -decoders 2>/dev/null | awk '{print $2}' | grep -Fx "$dec" >/dev/null \
    || { echo "ERROR: expected decoder '$dec' to be enabled" >&2; exit 1; }
done
# Only h264_cuvid may appear in the H.264 decoder family — no SW h264 decoder.
h264_decoders="$(ffmpeg -hide_banner -decoders 2>/dev/null | awk '$2 ~ /^h264/ {print $2}' | sort -u)"
forbidden_h264_decoders="$(printf '%s\n' "${h264_decoders}" | awk 'NF && $0 != "h264_cuvid"')"
if [[ -n "${forbidden_h264_decoders}" ]]; then
  echo "ERROR: forbidden H.264 decoder(s) leaked into the image: ${forbidden_h264_decoders}" >&2; exit 1; fi
# No HEVC/AAC software decoder either.
for forbidden in hevc aac; do
  if ffmpeg -hide_banner -decoders 2>/dev/null | awk '{print $2}' | grep -Fx "$forbidden" >/dev/null; then
    echo "ERROR: forbidden decoder '$forbidden' leaked into the image" >&2; exit 1; fi
done
# IO scaffolding required by the FF-1 VP9/webm + sample-frames paths.
ffmpeg -hide_banner -muxers   2>/dev/null | awk '{print $2}' | grep -Fx webm   >/dev/null || { echo "ERROR: muxer webm missing"   >&2; exit 1; }
ffmpeg -hide_banner -muxers   2>/dev/null | awk '{print $2}' | grep -Fx image2 >/dev/null || { echo "ERROR: muxer image2 missing" >&2; exit 1; }
ffmpeg -hide_banner -demuxers 2>/dev/null | awk '{print $2}' | grep -E '(^|,)matroska(,|$)' >/dev/null || { echo "ERROR: demuxer matroska,webm missing" >&2; exit 1; }
ffmpeg -hide_banner -demuxers 2>/dev/null | awk '{print $2}' | grep -E '(^|,)mov(,|$)'      >/dev/null || { echo "ERROR: demuxer mov missing" >&2; exit 1; }
ffmpeg -hide_banner -protocols 2>/dev/null | awk '{print $1}' | grep -Fx file             >/dev/null || { echo "ERROR: protocol file missing" >&2; exit 1; }
ffmpeg -hide_banner -bsfs      2>/dev/null | awk '{print $1}' | grep -Fx h264_mp4toannexb  >/dev/null || { echo "ERROR: bsf h264_mp4toannexb missing" >&2; exit 1; }
# vp9_superframe is required to mux libvpx-vp9 output into WebM/Matroska (the FF-1 path).
ffmpeg -hide_banner -bsfs      2>/dev/null | awk '{print $1}' | grep -Fx vp9_superframe    >/dev/null || { echo "ERROR: bsf vp9_superframe missing (needed to mux VP9 into WebM)" >&2; exit 1; }
for filt in fps scale format; do
  ffmpeg -hide_banner -filters 2>/dev/null | awk '{print $2}' | grep -Fx "$filt" >/dev/null \
    || { echo "ERROR: filter '$filt' missing" >&2; exit 1; }
done
echo "FFmpeg codec allow-list verified: no libx264/libx265/openh264/SW-h264/hevc/aac; VP9+mjpeg encode, h264_cuvid+vp8/vp9/mjpeg decode."
EOF
RUN apt-get purge -y --auto-remove yasm nasm && rm -rf /var/lib/apt/lists/*

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
# Note: the bundled ffmpeg libs (libav*.so inside opencv_python*.libs/) carry both
# CVEs and the royalty-bearing H.264/H.265/AAC codecs. opencv-python-headless 4.13.0.92
# hard-links its main .so against those bundled libav*/libsw*/libpostproc objects via
# DT_NEEDED + versioned-symbol references, so they cannot be deleted/patchelf-stripped
# post-hoc without breaking `import cv2`. The from-source rebuild (WITH_FFMPEG=OFF)
# implemented below is the real fix — it produces a cv2 with NO libav* linkage (TAO-2183 / FF-3).
RUN ARCH=$(uname -m) && \
    curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-${ARCH}.zip" -o /tmp/awscliv2.zip && \
    cd /tmp && unzip -q awscliv2.zip && \
    ./aws/install --update --bin-dir /usr/local/bin --install-dir /usr/local/aws-cli && \
    rm -rf /tmp/awscliv2.zip /tmp/aws

# ---------------------------------------------------------------------------
# OpenCV rebuilt WITH_FFMPEG=OFF (TAO-2183 / FF-3)
# ---------------------------------------------------------------------------
# Runs LAST among the pip layers so it overrides the opencv-python-headless 4.13.0.92
# shipped by the vllm:26.03 base AND the requirements pin. Uninstalls every opencv flavor,
# scrubs the leftover cv2/ + opencv_python*.libs/ (which is where the bundled libav* lives),
# then builds from sdist with the codec backends disabled. The embedded python check fails
# the build if cv2 still reports FFMPEG: YES or links any libav*. Image codecs
# (libjpeg/png/tiff/webp) are royalty-free and kept.
#
# VERSION NOTE: the base wheel 4.13.0.92 is an NVIDIA DL-frameworks build published as a
# WHEEL ONLY (no public sdist), so it cannot be rebuilt from source. We build the latest
# source-available release, 4.12.0.88, mirroring paidf-auto-labeling (which pins exactly
# 4.12.0.88). The 4.13->4.12 delta is immaterial for TAO's cv2 image ops; revisit if
# DL-frameworks publishes a libav-free 4.13 wheel (FF-3).
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential cmake libjpeg-dev libpng-dev libtiff-dev libwebp-dev zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*
RUN <<'EOF'
#!/usr/bin/env bash
set -euxo pipefail
# Bypass the vllm:26.03 global PIP_CONSTRAINT=/etc/pip/constraint.txt for this layer; every
# install below is exact-pinned (opencv) or an explicit build-tool floor, so it is safe.
export PIP_CONSTRAINT=""
python -m pip uninstall -y \
    opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless || true
SP="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
rm -rf "${SP}"/cv2 \
       "${SP}"/opencv_python*.libs \
       "${SP}"/opencv_python*.dist-info \
       "${SP}"/opencv_contrib_python*.dist-info \
       "${SP}"/opencv_contrib_python*.libs
python -m pip install --no-cache-dir \
    "setuptools>=78.1.1" "wheel>=0.46.2" "scikit-build>=0.18,<1" "cmake>=3.27,<4"
export ENABLE_CONTRIB=0
export ENABLE_HEADLESS=1
export CMAKE_ARGS="\
    -DWITH_FFMPEG=OFF \
    -DWITH_GSTREAMER=OFF \
    -DWITH_V4L=OFF \
    -DWITH_LIBV4L=OFF \
    -DWITH_QT=OFF \
    -DWITH_GTK=OFF \
    -DOPENCV_ENABLE_NONFREE=OFF \
    -DBUILD_TESTS=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DBUILD_EXAMPLES=OFF"
# Build the headless flavor from sdist. 4.12.0.88 is the latest version with a public
# sdist (4.13.0.92 is wheel-only). PIP_CONSTRAINT="" (set above) lets this differ from the
# base-constraints snapshot pin of 4.13.0.92.
python -m pip install --no-cache-dir --no-build-isolation \
    --no-binary=opencv-python-headless \
    "opencv-python-headless==4.12.0.88"
python - <<'PY'
import re, subprocess
from pathlib import Path
import cv2
info = cv2.getBuildInformation()
if re.search(r"FFMPEG:\s+YES", info):
    raise SystemExit("ERROR: OpenCV unexpectedly built WITH FFmpeg")
cv2_pkg_dir = Path(cv2.__file__).resolve().parent
cv2_sos = sorted(cv2_pkg_dir.rglob("cv2*.so"))
if not cv2_sos:
    raise SystemExit(f"ERROR: could not locate cv2 compiled extension under {cv2_pkg_dir}")
forbidden = ("libavcodec", "libavdevice", "libavfilter",
             "libavformat", "libavutil", "libswresample", "libswscale")
for cv2_so in cv2_sos:
    ldd = subprocess.check_output(["ldd", str(cv2_so)], text=True)
    for lib in forbidden:
        if lib in ldd:
            raise SystemExit(f"ERROR: {cv2_so.name} still links {lib}")
print("opencv", cv2.__version__, "FFmpeg disabled, no libav* linkage in", [p.name for p in cv2_sos])
PY
# Defensive: fail if any bundled libav* slipped back into an opencv .libs dir.
leftovers="$(find "${SP}" \( \
    -path '*opencv_python*.libs/libavcodec*' -o \
    -path '*opencv_python*.libs/libavdevice*' -o \
    -path '*opencv_python*.libs/libavfilter*' -o \
    -path '*opencv_python*.libs/libavformat*' -o \
    -path '*opencv_python*.libs/libavutil*' -o \
    -path '*opencv_python*.libs/libswresample*' -o \
    -path '*opencv_python*.libs/libswscale*' \
\) -print 2>/dev/null || true)"
if [ -n "${leftovers}" ]; then echo "ERROR: bundled libav* leaked back in:"; echo "${leftovers}"; exit 1; fi
EOF
# NOTE: build-essential/cmake are intentionally NOT purged — the vllm:26.03 base relies on a
# C/C++ toolchain at runtime (torch.compile / Triton JIT). Only the truly build-only codec
# assemblers (yasm/nasm) were purged after the ffmpeg layer above.

# Create user that will run commands
ARG user_id=1000
ARG user_name=developer
ARG groups=developer:1000
ARG home=/home/developer
RUN echo "ALL   ALL = (ALL) NOPASSWD: ALL" > /etc/sudoers \
    && mkdir -p "$(dirname $home)"
WORKDIR /workspace/