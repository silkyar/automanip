FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime@sha256:77f17f843507062875ce8be2a6f76aa6aa3df7f9ef1e31d9d7432f4b0f563dee
RUN apt-get update && apt-get install -y --no-install-recommends \
    libegl1 libgl1 libglib2.0-0 libxext6 libvulkan1 vulkan-tools ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY requirements/ /opt/requirements/
RUN python -m pip install --no-cache-dir --only-binary=:all: --require-hashes \
    -r /opt/requirements/linux-cu124.txt -r /opt/requirements/stackcube-extra.txt \
    && python -m pip check
COPY ppo.py run.py recipe.json official.json ./
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MS_SKIP_ASSET_DOWNLOAD_PROMPT=1
CMD ["python", "run.py"]
