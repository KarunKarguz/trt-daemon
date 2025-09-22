# Builder stage compiles the daemon and runs smoke tests
# using TensorRT 21.06 as base image (which supports CUDA 11.3) on ubuntu 18.04
FROM nvcr.io/nvidia/tensorrt:21.06-py3 AS build 

ENV NVIDIA_DISABLE_REQUIRE=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        tzdata ca-certificates curl gnupg software-properties-common && \
    ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata && \
    apt-get install -y --no-install-recommends \
        build-essential ninja-build socat && \
    rm -rf /var/lib/apt/lists/*

ENV CMAKE_VERSION=3.27.9
RUN curl -fsSL "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz" -o /tmp/cmake.tgz && \
    tar -xzf /tmp/cmake.tgz -C /opt && \
    mv "/opt/cmake-${CMAKE_VERSION}-linux-x86_64" /opt/cmake && \
    rm /tmp/cmake.tgz

ENV PATH="/opt/cmake/bin:${PATH}"

WORKDIR /workspace
COPY . /workspace

RUN cmake -B build -S . -DCUDAToolkit_ROOT=/usr/local/cuda
RUN cmake --build build -j
RUN chmod +x scripts/smoke_test.sh

FROM build AS tester
RUN scripts/smoke_test.sh

# Runtime image keeps only binaries and models
FROM nvcr.io/nvidia/tensorrt:21.06-py3 AS runtime

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
