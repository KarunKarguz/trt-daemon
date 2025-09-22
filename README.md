# TensorRT Daemon Playground

This workspace captures a small TensorRT inference daemon I bring up inside NVIDIA's official containers. The daemon loads a serialized `.plan` file, serves inference over a Unix domain socket, and can optionally proxy TCP traffic for remote clients. C++ sources live under `workspace/trtdaemon` alongside helper scripts for packaging and testing.

## Why this exists
- Keep a reproducible sandbox for Maxwell-class GPUs (GeForce 920MX) that must stick to TensorRT 8.5.x + CUDA 11.8
- Validate serialized engines quickly without pulling in Triton or larger microservice frameworks
- Provide a thin reference for both local socket clients and simple TCP proxies
- Encode model binding metadata in a tiny handshake so clients can adapt to different engines (multi-input/output ready)

Everything runs inside the TensorRT 22.12 container, launched the same way I used during development:

```bash
docker run --gpus all -e NVIDIA_DISABLE_REQUIRE=1 -it --rm \
  -v "$PWD:/workspace" nvcr.io/nvidia/tensorrt:22.12-py3 bash
```

Once inside the container, the project root lives at `/workspace`.

## Project layout

- `workspace/trtdaemon/` – daemon sources, CMake build, Docker packaging, and helper scripts
- `install_trt85.sh` – host-side convenience script that provisions CUDA 11.8 + TensorRT 8.5.3 on Ubuntu 24.04
- `TensorRT-8.5.3.1/` – extracted TensorRT SDK dropped here during setup (ignored in Git)
- `resnet50_perf.json` – quick benchmark record from the daemon client

The remainder of this README focuses on the daemon module because that is what will become the public GitLab repository.

---

# `workspace/trtdaemon`

## Documentation

Inline Doxygen-style comments live in `include/trt_server.hpp` and the daemon/client entrypoints to explain the socket protocol, binding metadata, and lifecycle expectations. Run `doxygen` with the provided headers to generate reference pages, or browse the comments directly when extending the service.

## Build system at a glance

- **Generator:** CMake ≥ 3.18
- **Language:** C++17
- **Dependencies:** CUDA Toolkit (11.8), TensorRT 8.5 runtime libraries, POSIX sockets
- **Outputs:**
  - `trt-daemon` – long-running daemon exposing a Unix socket
  - `trt-client` – benchmarking client speaking the daemon protocol

I keep a small wrapper (`ForgeX.sh`) that configures CMake against `/usr/local/cuda` and performs a parallel `make` build. Inside the TensorRT container the CUDA toolkit already lives at that location so the defaults just work.

### Manual build
```bash
cd workspace/trtdaemon
mkdir -p build
cmake -B build -S . -DCUDAToolkit_ROOT=/usr/local/cuda
cmake --build build -j$(nproc)
```

The resulting binaries land in `build/`.

### One-liner helper
```bash
cd workspace/trtdaemon
./ForgeX.sh
```

`ForgeX.sh` simply nukes the local `build/` directory and repeats the commands above.

## Running the daemon

1. Ensure a serialized engine exists (for example `models/mobilenetv2_fp32.plan`).
2. Inside the container:
 ```bash
  cd /workspace/workspace/trtdaemon
  ./build/trt-daemon -e models/mobilenetv2_fp32.plan -s /run/trt.sock
  ```
3. By default the daemon listens on `/run/trt.sock`. It emits a text handshake (`TRTD 1` header) listing each binding with byte length, dtype, and dimensions before reading tensors. Pinned host buffers are sized to these bindings and latency stats print every 100 requests. If `/opt/models` already contains a `.plan`, the entrypoint auto-selects it; otherwise provide `ENGINE_PATH` or `ONNX_PATH`.

Dynamic engines that rely on runtime dimensions can be pinned to a concrete shape by providing overrides on the command line or through the `TRT_SHAPES` environment variable. Shapes use `name=dim,dim,...` syntax and multiple entries are separated by semicolons:

```bash
./build/trt-daemon -e dynamic.plan --shape input=1,3,640,640 --shape boxes=1,100,4
# or
TRT_SHAPES="input=1,3,640,640;boxes=1,100,4" ./build/trt-daemon -e dynamic.plan
```

The daemon applies the overrides before it advertises the handshake, ensuring clients see the concrete sizes they need to stream.

Stop it with `Ctrl+C`; the socket file is removed on shutdown.

## Talking to the daemon

### Local Unix socket
```bash
cd workspace/trtdaemon
./build/trt-client 500 50   # 500 iterations, 50 warmup (handshake parsed automatically)
```

### Python helpers
- `scripts/infer_image.py` – sends a preprocessed image over the Unix socket and prints Top-5 predictions
- `scripts/infer_image_tcp.py` – same flow but talks to the TCP proxy (see next section)
- `scripts/inspect_bindings.py` – dumps the negotiated bindings from the daemon handshake for troubleshooting
- `scripts/smoke_test.sh` – internal smoke runner invoked during Docker builds

Example:
```bash
python3 scripts/infer_image.py --image images/tabby_tiger_cat.jpg \
  --labels images/imagenet_classes.txt
```

### Docker Compose workflow
- `docker compose build test` compiles the daemon inside the TensorRT base image and runs the smoke handshake (skips automatically if the bundled model is absent).
- `docker compose up trt` reuses the cached build artifacts, starts the daemon, and exposes TCP on port 8008.
- Both services assume a GPU-enabled host (the smoke test loads the bundled Mobilenet plan).

### Optional TCP proxy
If `EXPOSE_TCP=1` is set (the default in `entrypoint.sh`), the Docker build bundles `socat` and exposes port `8008`. External clients can then connect over TCP while the daemon still reads from `/run/trt.sock`.

```bash
docker compose up --build
# or
docker run --rm --gpus all -p 8008:8008 trt-daemon:local
```

Use the TCP Python script to test remotely:
```bash
python3 scripts/infer_image_tcp.py --host 127.0.0.1 --port 8008 \
  --image images/reflex_camera.jpeg --labels images/imagenet_classes.txt
```

## Packaging choices

- `Dockerfile` builds on `nvcr.io/nvidia/tensorrt:22.12-py3`, compiles the daemon, runs the smoke handshake, and wires up `entrypoint.sh` for runtime configuration.
- `docker-compose.yml` separates the smoke stage (`docker compose build test`) from the runtime deployment (`docker compose up trt`).
- `deploy/trt-daemon.service` + `install.sh` provide a bare systemd deployment path for non-container hosts.
