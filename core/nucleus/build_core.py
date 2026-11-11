"""
SKELETON: сборка core.nblk из teacher-модели (дистилляция).
Полная реализация — отдельный шаг (требует GPU/времени).

Идея:
  1) Teacher (Qwen2.5-0.5B или similar) генерирует пары (context, query) → answer
  2) Student = CoreBlock с урезанным FFN
  3) Loss = KL(student_logits/τ || teacher_logits/τ) + CE
  4) Квантование int8 → core.nblk
"""
from .nblk import write_block
import numpy as np


def distill_to_core(teacher, dataset, out_path,
                    n_layers=12, n_heads=8, d_model=512,
                    vocab_size=32000, ffn_factor=0.25):
    raise NotImplementedError(
        "TODO: реализуется после smoke-тестов. "
        "Нужен teacher + подготовленный датасет (см. gen_training_data).")
