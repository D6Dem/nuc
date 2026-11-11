"""
Блочный контейнер .nblk:
  [0:4]   MAGIC = b"NBLK"
  [4:6]   version (uint16 LE)
  [6:8]   flags   (uint16 LE)
  [8:16]  header_len (uint64 LE)
  [16:..] header (JSON) + padding до ALIGN
  [...]   body: сырые байты тензоров/скейлов (offset в header)
"""
import struct, json, numpy as np

MAGIC = b"NBLK"; VERSION = 1; ALIGN = 64
DT = {"int8": np.int8, "fp16": np.float16, "fp32": np.float32, "int32": np.int32}

def _dt(a):
    if a.dtype == np.int8: return "int8"
    if a.dtype == np.float16: return "fp16"
    return "fp32"

def _align(n, a=ALIGN): return (n + a - 1) // a * a

def write_block(path, *, name, kind, arch, meta, tensors, scales=None):
    header = {"name": name, "kind": kind, "version": VERSION,
              "arch": arch, "meta": meta, "tensors": [], "scales": []}
    parts, pos = [], 0
    def put(arr, bucket, tag):
        nonlocal pos
        arr = np.ascontiguousarray(arr)
        b = arr.tobytes()
        pad = _align(pos) - pos
        if pad: parts.append(b"\x00" * pad); pos += pad
        bucket.append({"name": tag, "dtype": _dt(arr),
                       "shape": list(arr.shape), "offset": pos, "size": len(b)})
        parts.append(b); pos += len(b)
    for n, t in tensors.items(): put(t, header["tensors"], n)
    for n, s in (scales or {}).items(): put(s, header["scales"], n)
    body = b"".join(parts)
    hb = json.dumps(header, ensure_ascii=False).encode("utf-8")
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<HH", VERSION, 1 if scales else 0))
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        pad = _align(16 + len(hb)) - (16 + len(hb))
        if pad: f.write(b"\x00" * pad)
        f.write(body)

def read_block(path):
    raw = open(path, "rb").read()
    assert raw[:4] == MAGIC, f"{path}: not .nblk"
    hlen, = struct.unpack("<Q", raw[8:16])
    header = json.loads(raw[16:16+hlen].decode("utf-8"))
    data = raw[_align(16 + hlen):]
    def mat(bucket):
        out = {}
        for e in bucket:
            n = int(np.prod(e["shape"]))
            out[e["name"]] = np.frombuffer(data, dtype=DT[e["dtype"]],
                                           count=n, offset=e["offset"]
                                           ).reshape(e["shape"]).copy()
        return out
    return header, mat(header["tensors"]), mat(header.get("scales", []))
