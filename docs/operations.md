# Operations Notes

I keep these notes around for deploying the daemon on developer workstations and lightweight edge boxes.

## Health checks
- Socket presence: `test -S /run/trt.sock`
- TCP proxy (if enabled): `nc -z localhost 8008`
- Engine sanity: run `./build/trt-client 20 5` and look for mean latency under target (e.g., < 20 ms for MobileNetv2 on 920MX)

## Log locations
- Docker: `docker logs <container>` shows daemon stdout plus the entrypoint banner
- Systemd: `journalctl -u trt-daemon -f`

## Updating engines
1. Drop the new `.plan` file under `/opt/models` (Docker) or `/opt/model` (systemd path)
2. Restart the service (`docker restart` or `systemctl restart trt-daemon`)
3. Validate with the Python client on a known image

## Building a fresh engine
When the serialized engine is missing, `entrypoint.sh` can fall back to `trtexec` if an ONNX path and options are supplied:

```bash
ONNX_PATH=/opt/models/mobilenetv2.onnx \
TRT_OPTS="--fp16 --workspace=512" \
./entrypoint.sh
```

The script mirrors how I usually rebuild engines during experimentation.

## Troubleshooting
- **`cudaHostAlloc failed`**: reduce batch size or verify the host has enough pinned memory allowances.
- **`enqueueV2 failed`**: double-check TensorRT version compatibility between the engine and runtime.
- **Socket timeouts**: if using the TCP proxy, confirm `socat` is running (displayed by entrypoint logs).

## Scheduled restarts
For long-running deployments I schedule a nightly restart to release pinned memory back to the system:
```
0 4 * * * systemctl restart trt-daemon
```
Adjust to your environment as needed.
