# Use NVIDIA’s TensorRT container as a base
FROM nvcr.io/nvidia/tensorrt:22.12-py3

# In case of strict GPU checks (e.g., 920MX), default to bypass
ENV NVIDIA_DISABLE_REQUIRE=1
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Build deps + socat (for optional TCP proxy)
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential cmake socat && \
    rm -rf /var/lib/apt/lists/*

# App dir
WORKDIR /app

# Copy and build your daemon
COPY build/trt-daemon /app/trtdaemon/build/

# Models folder (mounted or baked)
RUN mkdir -p /opt/models
COPY models/ /opt/models/

# Entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Healthcheck: socket should exist once daemon is up
HEALTHCHECK --interval=20s --timeout=3s --retries=5 CMD test -S /run/trt.sock || exit 1

# Optional TCP port
EXPOSE 8008

ENTRYPOINT ["/entrypoint.sh"]
