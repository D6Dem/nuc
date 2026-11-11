"""BlockStore: core всегда в RAM, эксперты — LRU."""
import threading
from pathlib import Path
from .nblk import read_block
from .blocks import CoreBlock, ExpertBlock


class BlockStore:
    def __init__(self, blocks_dir="./blocks", max_experts=2):
        self.dir = Path(blocks_dir)
        self.max = max_experts
        self.core = None
        self._experts = {}
        self._order = []
        self._lock = threading.Lock()

    def load_core(self, name="core.nblk"):
        path = self.dir / name
        if not path.exists():
            raise FileNotFoundError(str(path))
        h, t, s = read_block(path)
        assert h["kind"] == "core", f"{name}: not core"
        self.core = CoreBlock(h, t, s)
        return self.core

    def available(self):
        return sorted(p.name for p in self.dir.glob("*.nblk")
                      if p.name != "core.nblk")

    def get_expert(self, name):
        with self._lock:
            if name in self._experts: return self._experts[name]
            h, t, s = read_block(self.dir / name)
            ex = ExpertBlock(h, t, s)
            self._experts[name] = ex
            self._order.append(name)
            while len(self._order) > self.max:
                old = self._order.pop(0)
                self._experts.pop(old, None)
            return ex

    def drop_all_experts(self):
        with self._lock:
            self._experts.clear()
            self._order.clear()

    @property
    def footprint(self):
        core_mb = 0
        if self.core:
            core_mb = sum(a.nbytes for a in self.core.t.values()) / 1e6
        exp_mb = sum(e.size_bytes for e in self._experts.values()) / 1e6
        return {"core_MB": round(core_mb, 2), "experts_MB": round(exp_mb, 2)}
