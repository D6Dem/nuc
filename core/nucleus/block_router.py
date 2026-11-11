"""MoE-gate по тегам экспертов."""
import re
from .nblk import read_block

STOP = {"что","как","где","когда","это","для","the","a","и","на","в","по","с","к"}

class BlockRouter:
    def __init__(self, store, logger, top_k=2):
        self.store, self.logger, self.top_k = store, logger, top_k
        self._index = {}

    def refresh(self):
        self._index = {}
        for name in self.store.available():
            h, _, _ = read_block(self.store.dir / name)
            self._index[name] = set(x.lower() for x in h["meta"].get("tags", []))

    def select(self, text):
        if not self._index: self.refresh()
        words = {w.lower() for w in re.findall(r"[A-Za-zА-Яа-я0-9]{3,}", text.lower())
                 if w.lower() not in STOP}
        scored = []
        for name, tags in self._index.items():
            ov = len(words & tags)
            if ov: scored.append((ov, name))
        scored.sort(reverse=True)
        chosen = [n for _, n in scored[:self.top_k]]
        if chosen:
            self.logger.record("block_route",
                               {"query": text[:200], "chosen": chosen},
                               tags=["route"] + chosen)
        return chosen

    def build_expert_map(self, names):
        by_layer = {}
        if not self.store.core: return by_layer
        for name in names:
            ex = self.store.get_expert(name)
            if ex.target_layer >= 0:
                by_layer.setdefault(ex.target_layer, []).append(ex)
            else:
                for L in range(self.store.core.n_layers):
                    by_layer.setdefault(L, []).append(ex)
        return by_layer
