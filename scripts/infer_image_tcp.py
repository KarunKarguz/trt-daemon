#!/usr/bin/env python3
import argparse, socket, json
from pathlib import Path
from typing import List, Tuple
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


def _readline(sock: socket.socket) -> str:
    buf = bytearray()
    while True:
        chunk = sock.recv(1)
        if not chunk:
            raise RuntimeError("socket closed during handshake")
        if chunk == b"\n":
            return buf.decode()
        buf += chunk


def read_handshake(sock: socket.socket) -> Tuple[List[dict], List[dict]]:
    header = _readline(sock)
    if header.strip() != "TRTD 1":
        raise RuntimeError("unexpected protocol preamble")
    inputs, outputs = [], []
    while True:
        line = _readline(sock)
        if line == "END":
            break
        parts = line.strip().split()
        if not parts:
            continue
        kind = parts[0]
        name = parts[1]
        bytes_len = int(parts[2])
        dtype = int(parts[3])
        ndims = int(parts[4])
        dims = list(map(int, parts[5:5 + ndims]))
        entry = {"name": name, "bytes": bytes_len, "dtype": dtype, "dims": dims}
        if kind == "I":
            inputs.append(entry)
        elif kind == "O":
            outputs.append(entry)
    if not inputs or not outputs:
        raise RuntimeError("protocol header missing bindings")
    return inputs, outputs

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8008)
    ap.add_argument("--image", required=True)
    ap.add_argument("--labels")
    ap.add_argument("--size", type=int, default=224)
    args = ap.parse_args()

    x = preprocess(args.image, args.size)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((args.host, args.port))
    inputs, outputs = read_handshake(s)

    if len(inputs) != 1:
        raise RuntimeError(f"expected single input binding, got {len(inputs)}")
    if inputs[0]["bytes"] != x.nbytes:
        raise RuntimeError("binding byte-size mismatch; adjust preprocessing or use --size")

    send_all(s, memoryview(x.tobytes(order="C")))
    out_chunks = [recv_all(s, out_info["bytes"]) for out_info in outputs]
    s.close()

    logits = np.frombuffer(out_chunks[0], dtype=np.float32)
    probs = softmax(logits[None,:])[0]
    top5 = np.argsort(probs)[-5:][::-1]
    labels = load_labels(args.labels)

    print("\nTop-5:")
    for i in top5:
        if labels and i < len(labels):
            print(f"{i:4d}: {probs[i]:7.4f}  {labels[i]}")
        else:
            print(f"{i:4d}: {probs[i]:7.4f}")
