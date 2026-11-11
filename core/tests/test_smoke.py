import os, json, tempfile
from pathlib import Path
import pytest

os.environ.setdefault("NUCLEUS_WORKSPACE", "./workspace_test")

from nucleus.build_demo_core import make_core
from nucleus.nblk import read_block, write_block
from nucleus.logger import Logger
from nucleus.file_index import FileIndex


@pytest.fixture
def workspace():
    with tempfile.TemporaryDirectory() as d:
        os.environ["NUCLEUS_WORKSPACE"] = d
        yield Path(d)


def test_nblk_roundtrip(tmp_path):
    import numpy as np
    p = tmp_path / "x.nblk"
    t = {"a": np.ones((4, 4), np.float32)}
    write_block(p, name="t", kind="core",
                arch={"n_layers": 1, "n_heads": 1, "d_model": 4, "vocab_size": 8},
                meta={}, tensors=t)
    h, tt, _ = read_block(p)
    assert h["name"] == "t"
    assert (tt["a"] == 1).all()


def test_demo_core_forward(tmp_path):
    from nucleus.store import BlockStore
    make_core(str(tmp_path / "core.nblk"))
    s = BlockStore(str(tmp_path)); s.load_core("core.nblk")
    out = s.core.forward([1, 2, 3])
    assert out.shape == (3, s.core.arch["vocab_size"])


def test_logger_record_query(tmp_path):
    lg = Logger(str(tmp_path))
    lg.record("task", {"input": "hello"}, tags=["greet"])
    lg.record("response", {"answer": "hi"}, tags=["greet"])
    hits = lg.query(tags=["greet"])
    assert len(hits) == 2
    lg.close()


def test_file_io(workspace):
    from nucleus.modules import file_io
    path = file_io.create("a.txt", "hello")
    assert Path(path).read_text() == "hello"
    jpath = file_io.create("b.json", {"x": 1})
    assert json.loads(Path(jpath).read_text())["x"] == 1


def test_file_io_sandbox(workspace):
    from nucleus.modules import file_io
    with pytest.raises(ValueError):
        file_io.create("../escape.txt", "bad")


def test_file_io_py_syntax(workspace):
    from nucleus.modules import file_io
    with pytest.raises(ValueError):
        file_io.create("bad.py", "def x(:\n pass")


def test_file_index_search(tmp_path, workspace):
    (workspace / "договор_аренды.txt").write_text("Договор аренды квартиры")
    (workspace / "invoice.txt").write_text("invoice 2025")
    idx = FileIndex(str(tmp_path / "idx.sqlite"), roots=[str(workspace)])
    st = idx.scan()
    assert st["added"] >= 2
    hits = idx.search("договор")
    assert any("договор" in h["name"].lower() for h in hits)
    assert idx.search("invoice")
    idx.close()


def test_kernel_plan_file_intent(tmp_path):
    from nucleus.kernel import Kernel
    make_core(str(tmp_path / "blocks" / "core.nblk"))
    k = Kernel(blocks_dir=str(tmp_path / "blocks"),
               memory_root=str(tmp_path / "mem"),
               scan_roots=[str(tmp_path)])
    plan = k._plan("найди файл договор")
    assert "file_search" in plan["tools"]
    assert plan.get("maybe_research") is True
    k.close()


def test_kernel_end_to_end(tmp_path, workspace):
    from nucleus.kernel import Kernel
    (workspace / "заметка.txt").write_text("купить молоко и хлеб")
    make_core(str(tmp_path / "blocks" / "core.nblk"))
    k = Kernel(blocks_dir=str(tmp_path / "blocks"),
               memory_root=str(tmp_path / "mem"),
               scan_roots=[str(workspace)])
    k.scan()
    out = k.think("найди файл заметка")
    assert out  # non-empty
    calls = k.logger.query(entry_type="module_call", limit=20)
    modules = [c["payload"].get("module") for c in calls]
    assert "file_search" in modules
    k.close()


def test_programming_runs():
    from nucleus.modules import programming
    with tempfile.TemporaryDirectory() as d:
        lg = Logger(d)
        out = programming.run("```python\nprint(2+2)\n```",
                              {"logger": lg})
        assert "4" in out
        lg.close()


def test_file_useful():
    from nucleus.kernel import Kernel
    assert Kernel._file_useful("Файлов по 'x' не найдено.") is False
    assert Kernel._file_useful(
        "Найдено 1:\n  Прочитал через text:\nhello world " * 10) is True


@pytest.mark.skipif(
    os.environ.get("NUCLEUS_CORE_TRAINED") != "1",
    reason="requires trained core (after SFT)")
def test_answer_is_synthesis_not_log():
    """После SFT: ответ НЕ должен быть дословным логом модуля."""
    # ... реализуется после обучения
    pass
