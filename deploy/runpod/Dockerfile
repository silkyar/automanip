# Preserve the project's torch 2.6 pin rather than replacing it with Runpod's
# newer torch 2.8 stack. Docker Hub amd64 digest verified 2026-10-23.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime@sha256:77f17f843507062875ce8be2a6f76aa6aa3df7f9ef1e31d9d7432f4b0f563dee

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/automanip/src \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    MS_ASSET_DIR=/workspace/mani_skill_assets

# SAPIEN's headless graphics libraries and SSH for an interactive training Pod.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates openssh-server libegl1 libgl1 libxext6 libvulkan1 vulkan-tools \
    && rm -rf /var/lib/apt/lists/* /etc/ssh/ssh_host_* \
    && mkdir -p /run/sshd \
    && printf '%s\n' 'PermitRootLogin prohibit-password' 'PasswordAuthentication no' \
        'KbdInteractiveAuthentication no' > /etc/ssh/sshd_config.d/10-automanip.conf

# Fail if upstream changes the interpreter. Keep runtime outside /workspace.
RUN python -c 'import sys; assert sys.version_info[:2] == (3, 11)'
COPY requirements/linux-cu124.txt /opt/automanip-requirements.txt
RUN python -m pip install --no-cache-dir --only-binary=:all: --require-hashes -r /opt/automanip-requirements.txt \
    && python -m pip check \
    && python -m pip freeze --all > /opt/automanip-installed.txt

WORKDIR /opt/automanip
COPY pyproject.toml train.py THIRD_PARTY_NOTICES.md ./
COPY third_party/ ./third_party/
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY tests/ ./tests/
# Supply source identity explicitly; the build context deliberately excludes .git.
ARG SOURCE_REVISION=unrecorded
LABEL org.opencontainers.image.revision=$SOURCE_REVISION
RUN printf '%s\n' "$SOURCE_REVISION" > /opt/automanip-source-revision.txt \
    && sha256sum train.py pyproject.toml /opt/automanip-requirements.txt > /opt/automanip-source-sha256.txt \
    && find src scripts configs tests third_party -type f -exec sha256sum {} + \
        >> /opt/automanip-source-sha256.txt \
    && python -m pip install --no-deps --no-build-isolation . \
    && python -m unittest discover -s tests -v \
    && python train.py --config configs/gpu-smoke.json --dry-run

COPY deploy/runpod/start.sh /usr/local/bin/automanip-start
RUN chmod 755 /usr/local/bin/automanip-start
EXPOSE 22
ENTRYPOINT ["/usr/local/bin/automanip-start"]
