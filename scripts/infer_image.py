#!/usr/bin/env python3
import argparse, socket, json
from pathlib import Path
from typing import Optional
import numpy as np
from PIL import Image

try:
    RESIZE_BILINEAR = Image.Resampling.BILINEAR  # Pillow >= 9.1
except AttributeError:
    RESIZE_BILINEAR = Image.BILINEAR    

DEF_SOCK = "/run/trt.sock"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def load_labels(path: Optional[Path]):
    if path and path.exists():
        try:
            if path.suffix == ".json":
                j = json.loads(path.read_text())
                return [j[str(i)][1] for i in range(1000)]
            else:
                return [l.strip() for l in path.read_text().splitlines() if l.strip()]
        except Exception:
            pass
    return None

def preprocess(image_path, size=224, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((size, size), RESIZE_BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0   # HWC
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2,0,1)).copy()           # CHW
    arr = np.expand_dims(arr, 0)                      # NCHW
    return arr

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)

def send_all(sock, data: memoryview):
    while data:
        n = sock.send(data)
        data = data[n:]

def recv_all(sock, nbytes: int) -> bytes:
    buf = bytearray(nbytes)
    mv = memoryview(buf)
    while mv:
        k = sock.recv_into(mv)
        if k == 0:
            raise RuntimeError("server closed before full read")
        mv = mv[k:]
    return bytes(buf)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sock", default=DEF_SOCK, help="Unix socket path (daemon)")
    ap.add_argument("--image", required=True, help="Path to input image")
    ap.add_argument("--labels", help="imagenet labels file (.txt or torchvision JSON)")
    ap.add_argument("--size", type=int, default=224)
    args = ap.parse_args()

    x = preprocess(args.image, size=args.size)        # (1,3,224,224) float32
    in_bytes = x.tobytes(order="C")
    out_elems = 1000
    out_bytes = out_elems * 4

    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(args.sock)
    send_all(s, memoryview(in_bytes))
    out_raw = recv_all(s, out_bytes)
    s.close()

    logits = np.frombuffer(out_raw, dtype=np.float32)
    probs = softmax(logits[None, :])[0]
    top5 = np.argsort(probs)[-5:][::-1]

    labels = load_labels(Path(args.labels)) if args.labels else None
    print("\nTop-5:")
    for i in top5:
        p = float(probs[i])
        if labels and i < len(labels):
            print(f"{i:4d}: {p:7.4f}  {labels[i]}")
        else:
            print(f"{i:4d}: {p:7.4f}")

if __name__ == "__main__":
    main()
