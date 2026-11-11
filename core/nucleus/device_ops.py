"""
ADVANCED: split-исполнение CPU+GPU. Opt-in, default OFF.
Используется только при config.core.use_advanced_gpu: true.
"""
import numpy as np
from enum import Enum
from concurrent.futures import ThreadPoolExecutor

try:
    import cupy as cp
    _HAS_GPU = True
except Exception:
    cp = None; _HAS_GPU = False

try:
    from numba import njit, prange, set_num_threads
    set_num_threads(min(8, __import__("os").cpu_count() or 4))
    _HAS_NUMBA = True
    @njit(parallel=True, fastmath=True, cache=True)
    def _matmul_np(x, w):
        out = np.empty((x.shape[0], w.shape[0]), dtype=np.float32)
        for i in prange(x.shape[0]):
            for j in range(w.shape[0]):
                acc = 0.0
                for k in range(x.shape[1]): acc += x[i,k]*w[j,k]
                out[i,j] = acc
        return out
    @njit(parallel=True, fastmath=True, cache=True)
    def _silu_np(x):
        out = np.empty_like(x)
        for i in prange(x.shape[0]):
            for j in range(x.shape[1]):
                v = x[i,j]; out[i,j] = v/(1.0+np.exp(-v))
        return out
except Exception:
    _HAS_NUMBA = False
    def _matmul_np(x, w): return x @ w.T.astype(np.float32)
    def _silu_np(x): return x * (1.0/(1.0+np.exp(-x)))


class Device(str, Enum):
    CPU = "cpu"; GPU = "gpu"


class Placement:
    def __init__(self, shards: dict, split_dim: int = None):
        self.shards = shards; self.split_dim = split_dim
    @property
    def devices(self): return list(self.shards.keys())
    @property
    def is_split(self): return len(self.shards) > 1
    def size_mb(self): return sum(a.nbytes for a in self.shards.values())/1e6


class DT:
    __slots__ = ("data", "device")
    def __init__(self, data, device: Device):
        self.data, self.device = data, device
    def to(self, device):
        if device == self.device: return self
        if device == Device.GPU and _HAS_GPU:
            return DT(cp.asarray(self.data), device)
        return DT(cp.asnumpy(self.data) if (_HAS_GPU and isinstance(self.data, cp.ndarray))
                  else np.asarray(self.data), device)


def linear(x: DT, w: Placement) -> DT:
    if not w.is_split:
        dev = w.devices[0]
        xd = x.to(dev)
        if dev == Device.GPU and _HAS_GPU:
            y = xd.data @ cp.asarray(w.shards[dev]).T
        else:
            y = _matmul_np(xd.data.astype(np.float32), w.shards[dev].astype(np.float32))
        return DT(y, dev).to(x.device)
    acc = None
    for dev, shard in w.shards.items():
        xd = x.to(dev)
        if dev == Device.GPU and _HAS_GPU:
            y = xd.data @ cp.asarray(shard).T
        else:
            y = _matmul_np(xd.data.astype(np.float32), shard.astype(np.float32))
        y_dt = DT(y, dev).to(x.device)
        acc = y_dt.data if acc is None else acc + y_dt.data
    return DT(acc, x.device)


def ffn(x: DT, w1: Placement, w2: Placement) -> DT:
    if not w1.is_split:
        dev = w1.devices[0]
        xd = x.to(dev)
        if dev == Device.GPU and _HAS_GPU:
            h = xd.data @ cp.asarray(w1.shards[dev]).T
            h = h * (1.0 / (1.0 + cp.exp(-h)))
            y = h @ cp.asarray(w2.shards[dev]).T
        else:
            h = _silu_np(_matmul_np(xd.data.astype(np.float32), w1.shards[dev].astype(np.float32)))
            y = _matmul_np(h.astype(np.float32), w2.shards[dev].astype(np.float32))
        return DT(y, dev).to(x.device)
    # split: параллельно GPU + CPU
    import threading
    results = {}
    def _gpu():
        xg = x.to(Device.GPU).data
        h = xg @ cp.asarray(w1.shards[Device.GPU]).T
        h = h * (1.0 / (1.0 + cp.exp(-h)))
        results["gpu"] = h @ cp.asarray(w2.shards[Device.GPU]).T
    def _cpu():
        xc = x.to(Device.CPU).data.astype(np.float32)
        h = _silu_np(_matmul_np(xc, w1.shards[Device.CPU].astype(np.float32)))
        results["cpu"] = _matmul_np(h.astype(np.float32), w2.shards[Device.CPU].astype(np.float32))
    tg = threading.Thread(target=_gpu)
    if Device.GPU in w1.devices and _HAS_GPU: tg.start()
    _cpu()
    if tg.is_alive(): tg.join()
    gpu_y = results.get("gpu"); cpu_y = results.get("cpu")
    if gpu_y is not None and cpu_y is not None:
        return DT(cp.asnumpy(gpu_y) + cpu_y, Device.CPU)
    return DT(cpu_y if cpu_y is not None else cp.asnumpy(gpu_y), Device.CPU)
