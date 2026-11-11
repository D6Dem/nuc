"""VRAM-бюджет и pinned-память. Только при use_advanced_gpu."""
try:
    import cupy as cp
    from cupy.cuda import memory
    _GPU = True
except Exception:
    cp = None; _GPU = False
import numpy as np


class VRAMBudget:
    def __init__(self, total_mb=0, reserve_mb=384):
        self.total_mb, self.reserve, self.used_mb = total_mb, reserve_mb, 0.0

    @property
    def free_mb(self): return max(0.0, self.total_mb - self.reserve - self.used_mb)

    def try_claim(self, tag, mb):
        if self.free_mb >= mb:
            self.used_mb += mb; return True
        return False

    def release(self, mb):
        self.used_mb = max(0.0, self.used_mb - mb)


def detect_vram_mb():
    if not _GPU: return 0
    free, total = cp.cuda.runtime.memGetInfo()
    return total // (1024*1024)
