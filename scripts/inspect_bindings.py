#!/usr/bin/env python3
"""Print the TensorRT binding metadata exposed by the daemon handshake."""
import argparse
import socket
from typing import List, Tuple


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
    return inputs, outputs


def main() -> None:
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--sock", help="Unix domain socket path")
    group.add_argument("--tcp", nargs=2, metavar=("HOST", "PORT"), help="TCP host/port")
    args = ap.parse_args()

    if args.sock:
        conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        conn.connect(args.sock)
    else:
        host, port = args.tcp[0], int(args.tcp[1])
        conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        conn.connect((host, port))

    inputs, outputs = read_handshake(conn)
    conn.close()

    print("Inputs:")
    for idx, binding in enumerate(inputs):
        print(f"  [{idx}] {binding['name']} bytes={binding['bytes']} dtype={binding['dtype']} dims={binding['dims']}")

    print("Outputs:")
    for idx, binding in enumerate(outputs):
        print(f"  [{idx}] {binding['name']} bytes={binding['bytes']} dtype={binding['dtype']} dims={binding['dims']}")


if __name__ == "__main__":
    main()
