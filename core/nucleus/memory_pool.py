"""Динамическое распределение RAM под задачу."""
import gc, psutil


class MemoryPool:
    def __init__(self, max_fraction=0.9, safety_mb=512):
        self.max_fraction, self.safety_mb = max_fraction, safety_mb

    def begin_task(self, n_layers, n_heads, d_head, max_seq_guess=4096):
        vm = psutil.virtual_memory()
        usable_mb = int(vm.available / 1e6 * self.max_fraction) - self.safety_mb
        usable_mb = max(usable_mb, 256)
        per_kv_mb = (n_layers * n_heads * d_head * 4 * 2 * max_seq_guess) / 1e6
        max_seq = min(max_seq_guess,
                      int(usable_mb * 0.3 / max(1e-3, per_kv_mb / max_seq_guess)))
        return {"available_mb": usable_mb, "kv_max_seq": max(64, max_seq)}

    def end_task(self):
        gc.collect()
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass
