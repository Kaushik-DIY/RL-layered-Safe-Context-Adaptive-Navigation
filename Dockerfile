# Training-environment image (plan Week 0 / sec. 8): guarantees Kaggle<->laptop parity.
# This image is for the sim-agnostic `core/` stack + PPO training ONLY.
# Gazebo/ROS 2 evaluation runs on the host (or a separate ros:humble image).
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch (plan D6: CPU training is sufficient; tiny 2x256 MLP).
RUN pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.lock.txt .
# torch already installed above from the CPU index; install the rest.
RUN grep -v '^torch==' requirements.lock.txt > /tmp/req.txt \
    && pip install -r /tmp/req.txt

COPY . .
RUN pip install -e .

CMD ["python", "-m", "pytest", "-q"]
