"""CoreBlock и ExpertBlock. Numba JIT с fallback на numpy."""
import numpy as np

# ---------- numba (optional) ----------
try:
    from numba import njit, prange
    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False


if _HAS_NUMBA:
    @njit(parallel=True, fastmath=True, cache=True)
    def _matmul_jit(x, w):
        """x: [seq, d_in], w: [d_out, d_in] → out: [seq, d_out]."""
        out = np.empty((x.shape[0], w.shape[0]), dtype=np.float32)
        for i in prange(x.shape[0]):
            for j in range(w.shape[0]):
                acc = 0.0
                for k in range(x.shape[1]):
                    acc += x[i, k] * w[j, k]
                out[i, j] = acc
        return out

    @njit(parallel=True, fastmath=True, cache=True)
    def _silu_jit(x):
        out = np.empty_like(x)
        for i in prange(x.shape[0]):
            for j in range(x.shape[1]):
                v = x[i, j]
                out[i, j] = v / (1.0 + np.exp(-v))
        return out

    @njit(parallel=True, fastmath=True, cache=True)
    def _rmsnorm_jit(x, w):
        out = np.empty_like(x)
        eps = 1e-6
        for i in prange(x.shape[0]):
            s = 0.0
            for j in range(x.shape[1]):
                s += x[i, j] * x[i, j]
            s = s / x.shape[1]
            inv = 1.0 / ((s + eps) ** 0.5)
            for j in range(x.shape[1]):
                out[i, j] = x[i, j] * inv * w[j]
        return out


def _mm(x, w):
    """x @ w.T, где w: [d_out, d_in]."""
    if _HAS_NUMBA and x.dtype == np.float32 and w.dtype == np.float32 \
       and x.ndim == 2:
        return _matmul_jit(np.ascontiguousarray(x), np.ascontiguousarray(w))
    return x @ w.T.astype(np.float32)


def _silu(x):
    if _HAS_NUMBA and x.dtype == np.float32 and x.ndim == 2:
        return _silu_jit(np.ascontiguousarray(x))
    return x * (1.0 / (1.0 + np.exp(-x)))


def _rmsnorm(x, w, eps=1e-6):
    if _HAS_NUMBA and x.dtype == np.float32 and w.dtype == np.float32 \
       and x.ndim == 2:
        return _rmsnorm_jit(np.ascontiguousarray(x),
                            np.ascontiguousarray(w.astype(np.float32)))
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w


def _rope(x, pos, base=10000.0):
    d = x.shape[-1]; half = d // 2
    freq = base ** (-np.arange(half, dtype=np.float32) / half)
    ang = np.outer(pos.astype(np.float32), freq)
    c, s = np.cos(ang), np.sin(ang)
    c = c[:, None, :]; s = s[:, None, :]
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], axis=-1)


class KVCache:
    def __init__(self, n_layers, n_heads, d_head, max_seq=2048):
        self.n_layers, self.n_heads, self.d_head = n_layers, n_heads, d_head
        self.max_seq = max_seq; self.len = 0
        self.k = [None] * n_layers
        self.v = [None] * n_layers

    def alloc(self, L, seq):
        if self.k[L] is None:
            self.k[L] = np.zeros((self.max_seq, self.n_heads, self.d_head), np.float32)
            self.v[L] = np.zeros_like(self.k[L])
        return self.k[L][:seq], self.v[L][:seq]

    def free(self):
        self.k = [None] * self.n_layers
        self.v = [None] * self.n_layers
        self.len = 0


class CoreBlock:
    def __init__(self, header, tensors, scales):
        self.header, self.arch = header, header["arch"]
        self.n_layers = self.arch["n_layers"]
        self.n_heads = self.arch["n_heads"]
        self.d_model = self.arch["d_model"]
        self.d_head = self.d_model // self.n_heads
        self.vocab = self.arch.get("vocab_size", 512)

        self.t = {}
        for name, arr in tensors.items():
            if arr.dtype == np.int8 and name in scales:
                arr_f = arr.astype(np.float32) * scales[name].astype(np.float32)[:, None]
            else:
                arr_f = arr.astype(np.float32)
            self.t[name] = arr_f

    def _attn(self, x, L, pos, kv):
        seq = x.shape[0]
        q = _mm(x, self.t[f"L{L}.wq"]).reshape(-1, self.n_heads, self.d_head)
        k = _mm(x, self.t[f"L{L}.wk"]).reshape(-1, self.n_heads, self.d_head)
        v = _mm(x, self.t[f"L{L}.wv"]).reshape(-1, self.n_heads, self.d_head)
        q, k = _rope(q, pos), _rope(k, pos)

        if kv is not None:
            ck, cv = kv.alloc(L, kv.len + seq)
            ck[kv.len:kv.len + seq] = k
            cv[kv.len:kv.len + seq] = v
            k_all, v_all = ck, cv
            past_len = kv.len
        else:
            k_all, v_all = k, v
            past_len = 0

        total = k_all.shape[0]
        sc = np.einsum("thd,shd->hts", q, k_all) / np.sqrt(self.d_head)
        if seq > 1 or past_len > 0:
            mask = np.zeros((seq, total), dtype=bool)
            for i in range(seq):
                mask[i, past_len + i + 1:] = True
            sc = np.where(mask, -np.inf, sc)
        p = np.exp(sc - sc.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        ctx = np.einsum("hts,shd->thd", p, v_all).reshape(seq, self.d_model)
        return _mm(ctx, self.t[f"L{L}.wo"])

    def _ffn(self, x, L):
        h = _silu(_mm(x, self.t[f"L{L}.ff1"]))
        return _mm(h, self.t[f"L{L}.ff2"])

    def forward(self, input_ids, experts_by_layer=None, kv=None, pos_offset=0):
        experts_by_layer = experts_by_layer or {}
        ids = np.asarray(input_ids, dtype=np.int64)
        if ids.ndim == 0:
            ids = ids.reshape(1)
        pos = np.arange(pos_offset, pos_offset + len(ids))
        x = self.t["embed"][ids].astype(np.float32)

        for L in range(self.n_layers):
            x = x + self._attn(_rmsnorm(x, self.t[f"L{L}.norm1"]), L, pos, kv)
            h = _rmsnorm(x, self.t[f"L{L}.norm2"])
            y = self._ffn(h, L)
            for ex in experts_by_layer.get(L, []):
                y = y + ex.apply(h)
            x = x + y

        x = _rmsnorm(x, self.t["norm_out"])
        return _mm(x, self.t["lm_head"])


class ExpertBlock:
    def __init__(self, header, tensors, scales):
        self.header, self.meta = header, header["meta"]
        self.kind = header["kind"]
        self.target_layer = self.meta.get("target_layer", -1)
        self.alpha = float(self.meta.get("alpha", 1.0))
        self.t = {}
        for name, arr in tensors.items():
            if arr.dtype == np.int8 and name in scales:
                arr_f = arr.astype(np.float32) * scales[name].astype(np.float32)[:, None]
            else:
                arr_f = arr.astype(np.float32)
            self.t[name] = arr_f

    def applies_to(self, layer):
        return self.target_layer < 0 or self.target_layer == layer

    def apply(self, x):
        if self.kind == "lora":
            return (x @ self.t["A"].T) @ self.t["B"].T * self.alpha
        h = _silu(x @ self.t["W1"].T)
        return (h @ self.t["W2"].T) * self.alpha

    @property
    def size_bytes(self):
        return sum(a.nbytes for a in self.t.values())
