"""
SKELETON: SFT / LoRA-обучение core на готовом датасете.
"""
import json
from pathlib import Path


def load_dataset(path):
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                pairs.append(row)
            except Exception:
                continue
    return pairs


def train(dataset_path, core_init="./blocks/core.nblk",
          out_path="./blocks/core_sft.nblk", epochs=3, lr=1e-4):
    raise NotImplementedError(
        "TODO: реализуется после smoke-тестов. "
        "Пайплайн: numpy → numba/cupy → autograd (написать или использовать micrograd). "
        "До этого — использовать встроенный чекпойнт teacher.")
