#!/usr/bin/env python3
import argparse, socket, json
from pathlib import Path
import numpy as np
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

try:
    RESIZE_BILINEAR = Image.Resampling.BILINEAR  # Pillow >=9.1
except AttributeError:
    RESIZE_BILINEAR = Image.BILINEAR

def preprocess(img_path, size=224):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((size, size), RESIZE_BILINEAR)
    x = np.asarray(img).astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = np.transpose(x, (2,0,1)).copy()
    x = np.expand_dims(x, 0)
    return x

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x); return e / np.sum(e, axis=-1, keepdims=True)

def load_labels(path):
    if not path: return None
    p = Path(path)
    if not p.exists(): return None
    if p.suffix == ".json":
        j = json.loads(p.read_text())
        return [j[str(i)][1] for i in range(1000)]
    return [l.strip() for l in p.read_text().splitlines() if l.strip()]

def send_all(sock, mv):
    while mv:
        n = sock.send(mv); mv = mv[n:]

def recv_all(sock, n):
    buf = bytearray(n); mv = memoryview(buf)
    while mv:
        k = sock.recv_into(mv)
        if k == 0: raise RuntimeError("server closed")
        mv = mv[k:]
    return bytes(buf)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8008)
    ap.add_argument("--image", required=True)
    ap.add_argument("--labels")
    ap.add_argument("--size", type=int, default=224)
    args = ap.parse_args()

    x = preprocess(args.image, args.size)
    in_bytes = x.tobytes(order="C")
    out_bytes = 1000 * 4

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((args.host, args.port))
    send_all(s, memoryview(in_bytes))
    out_raw = recv_all(s, out_bytes)
    s.close()

    logits = np.frombuffer(out_raw, dtype=np.float32)
    probs = softmax(logits[None,:])[0]
    top5 = np.argsort(probs)[-5:][::-1]
    labels = load_labels(args.labels)

    print("\nTop-5:")
    for i in top5:
        if labels and i < len(labels):
            print(f"{i:4d}: {probs[i]:7.4f}  {labels[i]}")
        else:
            print(f"{i:4d}: {probs[i]:7.4f}")
