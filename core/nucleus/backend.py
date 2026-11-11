"""numpy / cupy переключатель. GPU опционален."""
import os, numpy as np

_MODE = os.environ.get("NUCLEUS_GPU", "0") == "1"
if _MODE:
    try:
        import cupy as xp
        from cupy.cuda import MemoryPool
        xp.cuda.set_allocator(MemoryPool().malloc)
        GPU_AVAILABLE = True
    except Exception:
        import numpy as xp
        GPU_AVAILABLE = False
else:
    import numpy as xp
    GPU_AVAILABLE = False

def to_xp(arr, dtype=None):
    if GPU_AVAILABLE:
        import cupy
        return cupy.asarray(arr, dtype=dtype) if dtype else cupy.asarray(arr)
    return np.asarray(arr, dtype=dtype) if dtype else np.asarray(arr)

def to_np(arr):
    if GPU_AVAILABLE:
        import cupy
        return cupy.asnumpy(arr)
    return np.asarray(arr)
