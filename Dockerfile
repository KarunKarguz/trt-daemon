# Builder stage compiles the daemon and runs smoke tests
FROM nvcr.io/nvidia/tensorrt:22.12-py3 AS build

ENV NVIDIA_DISABLE_REQUIRE=1
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential cmake socat && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . /workspace

RUN cmake -B build -S . -DCUDAToolkit_ROOT=/usr/local/cuda
RUN cmake --build build -j
RUN chmod +x scripts/smoke_test.sh

FROM build AS tester
RUN scripts/smoke_test.sh

# Runtime image keeps only binaries and models
FROM nvcr.io/nvidia/tensorrt:22.12-py3 AS runtime

ENV NVIDIA_DISABLE_REQUIRE=1
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update && \
    apt-get install -y --no-install-recommends socat && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app/trtdaemon

COPY --from=build /workspace/build/trt-daemon /app/trtdaemon/bin/trt-daemon
COPY --from=build /workspace/build/trt-client /app/trtdaemon/bin/trt-client

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

RUN mkdir -p /opt/models
COPY models/ /opt/models/

HEALTHCHECK --interval=20s --timeout=3s --retries=5 CMD test -S /run/trt.sock || exit 1

EXPOSE 8008

ENTRYPOINT ["/entrypoint.sh"]
