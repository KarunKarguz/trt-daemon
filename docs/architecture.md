# Runtime Architecture

The daemon sits between TensorRT's C++ runtime and simple client programs. Everything runs inside a single process and stays close to the metal:

```
   client (unix/tcp)                daemon                    TensorRT
┌─────────────────────┐   ┌────────────────────────────┐   ┌───────────────┐
│ float32 tensor data │ → │ copy H→D → enqueue V2 → D→H │ → │ serialized engine │
└─────────────────────┘   └────────────────────────────┘   └───────────────┘
```

## Components

- **`TRTServer` (`src/trt_server.cpp`)**
  - Deserializes an engine from disk using TensorRT 8.5 APIs
  - Owns the CUDA execution context and device buffers
  - Provides a synchronous `infer()` entry point for the daemon loop

- **`daemon.cpp`**
  - Exposes a Unix domain socket (`/run/trt.sock` by default)
  - Uses `epoll` to multiplex multiple clients
  - Allocates pinned host buffers sized to the engine bindings
  - Collects lightweight latency stats (mean, min/max, EMA-based p95)

- **`client.cpp`**
  - Simple throughput benchmark used during bring-up
  - Helps smoke test the serialization compatibility of engines

- **Python scripts**
  - Mirror the binary protocol and provide quick end-to-end validation with image preprocessing

## Protocol

1. Client connects and sends `N` bytes where `N` is the size of the first binding (float32 input tensor).
2. Daemon copies host memory into device memory, enqueues inference, and waits for completion.
3. Daemon copies the output binding back to host memory and writes it to the socket.
4. Connection stays open for subsequent requests until the client closes it.

There is no header, framing, or metadata. The contract is implicit: both sides agree on the binding shapes beforehand (e.g., 1×3×224×224 input, 1000 float outputs). For public releases, document the target model alongside the serialization recipe to avoid confusion.

## Extending safely

- **Multiple Models:** Instantiate additional `TRTServer` objects and route sockets based on paths (e.g., `/run/resnet50.sock`).
- **Dynamic Shapes:** Use `setInputShape()` before the first `infer()` call and expose a richer protocol so clients can specify dimensions.
- **Telemetry:** Emit stats to stdout logs or push to Prometheus via a sidecar; the current implementation prints every 100 requests to keep footprints small.
