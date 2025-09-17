# TensorRT Daemon Playground

This workspace captures a small TensorRT inference daemon I bring up inside NVIDIA's official containers. The daemon loads a serialized `.plan` file, serves inference over a Unix domain socket, and can optionally proxy TCP traffic for remote clients. C++ sources live under `workspace/trtdaemon` alongside helper scripts for packaging and testing.

## Why this exists
- Keep a reproducible sandbox for Maxwell-class GPUs (GeForce 920MX) that must stick to TensorRT 8.5.x + CUDA 11.8
- Validate serialized engines quickly without pulling in Triton or larger microservice frameworks
- Provide a thin reference for both local socket clients and simple TCP proxies

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
3. By default the daemon listens on `/run/trt.sock`. It allocates pinned host buffers sized to the engine bindings and prints latency stats every 100 requests.

Stop it with `Ctrl+C`; the socket file is removed on shutdown.

## Talking to the daemon

### Local Unix socket
```bash
cd workspace/trtdaemon
./build/trt-client 500 50   # 500 iterations, 50 warmup
```

### Python helpers
- `scripts/infer_image.py` – sends a preprocessed image over the Unix socket and prints Top-5 predictions
- `scripts/infer_image_tcp.py` – same flow but talks to the TCP proxy (see next section)

Example:
```bash
python3 scripts/infer_image.py --image images/tabby_tiger_cat.jpg \
  --labels images/imagenet_classes.txt
```

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

- `Dockerfile` builds on `nvcr.io/nvidia/tensorrt:22.12-py3`, copies the freshly compiled binaries and sample models, and wires up `entrypoint.sh` for runtime configuration.
- `docker-compose.yml` drives the same image in a simple single-service stack and publishes port `8008`.
- `deploy/trt-daemon.service` + `install.sh` provide a bare systemd deployment path for non-container hosts.

