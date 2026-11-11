import os, re
from pathlib import Path

TEXT = {".txt",".md",".log",".ini",".cfg",".rtf",".py",".js",".ts",".c",".cpp",
        ".h",".rs",".go",".java",".sh",".sql",".html",".css",".json",".jsonl",
        ".yaml",".yml",".toml",".csv",".tsv",".xml"}
DOC = {".docx",".pdf"}
AUDIO = {".wav",".mp3",".ogg",".flac",".m4a"}
VIDEO = {".mp4",".avi",".mkv",".mov",".webm"}
BIG = 100 * 1024


def _kind(p):
    e = p.suffix.lower()
    if e in TEXT: return "text"
    if e in DOC: return "doc"
    if e in AUDIO: return "audio"
    if e in VIDEO: return "video"
    return "other"


def _read_text(p, maxc=8000):
    e = p.suffix.lower()
    if e == ".docx":
        try:
            from docx import Document
            return "\n".join(x.text for x in Document(str(p)).paragraphs)[:maxc]
        except Exception as ex: return f"[docx: {ex}]"
    if e == ".pdf":
        try:
            from pypdf import PdfReader
            return "\n".join((pg.extract_text() or "") for pg in
                             PdfReader(str(p)).pages)[:maxc]
        except Exception as ex: return f"[pdf: {ex}]"
    try:
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            return f.read(maxc)
    except Exception as ex: return f"[read: {ex}]"


def open_and_read(path, ctx):
    p = Path(os.path.expanduser(path))
    if not p.exists(): return {"error": f"не найден: {path}"}
    kind = _kind(p); size = p.stat().st_size
    kernel = ctx["kernel"]; logger = ctx["logger"]
    out = {"path": str(p), "size": size, "kind": kind, "module": None}
    try:
        if kind == "text":
            if size > BIG:
                out["module"] = "recursive"
                out["summary"] = kernel.tools["recursive"].run(f"суммируй {p}", ctx=ctx)
                out["content"] = _read_text(p, 4000)
            else:
                out["module"] = "text"; out["content"] = _read_text(p)
        elif kind == "doc":
            if size > BIG:
                out["module"] = "recursive"
                raw = _read_text(p, 200_000)
                tmp = Path("/tmp") / f"nucleus_{p.stem}.txt"
                tmp.write_text(raw, encoding="utf-8")
                out["summary"] = kernel.tools["recursive"].run(f"суммируй {tmp}", ctx=ctx)
                out["content"] = raw[:4000]
            else:
                out["module"] = "text"; out["content"] = _read_text(p)
        elif kind == "audio":
            out["module"] = "audio"
            out["transcript"] = kernel.tools["audio"].run(str(p), ctx=ctx)
        elif kind == "video":
            out["module"] = "video"
            out["analysis"] = kernel.tools["video"].run(str(p), ctx=ctx)
        else:
            out["error"] = f"тип не поддержан: {p.suffix}"
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    logger.record("file_read", {"path": str(p), "kind": kind,
                                "module": out["module"],
                                "ok": "error" not in out},
                  tags=["file_search", "read", out.get("module") or kind])
    return out


def find_and_read(query, ctx, kind=None, limit=3):
    idx = ctx["kernel"].file_index
    hits = idx.search(query, kind=kind, limit=limit)
    if not hits:
        return {"query": query, "hits": [], "opened": None}
    opened = open_and_read(hits[0]["path"], ctx)
    return {"query": query, "hits": hits, "opened": opened}


def run(user_input, ctx):
    m = re.search(r"((?:~|/|\.{1,2}/)\S+\.\w+)", user_input)
    if m:
        p = os.path.expanduser(m.group(1))
        r = open_and_read(p, ctx)
        body = (r.get("content") or r.get("transcript") or r.get("analysis")
                or r.get("summary") or r.get("error") or "")
        return f"Прочитал {p} через {r.get('module') or '—'}:\n{body[:4000]}"

    low = user_input.lower()
    kind = None
    for k, ws in {"video": ("видео","video","ролик"),
                  "audio": ("аудио","audio","голос"),
                  "doc": ("pdf","docx","документ"),
                  "code": ("код","скрипт","code","python"),
                  "text": ("текст","txt","md")}.items():
        if any(w in low for w in ws): kind = k; break
    stop = {"найди","файл","файлы","открой","покажи","поищи","find","open"}
    q = " ".join(w for w in user_input.split() if w.lower() not in stop)

    res = find_and_read(q, ctx, kind=kind, limit=3)
    if not res["hits"]:
        return f"Файлов по '{q}' (kind={kind}) не найдено."
    lines = [f"Найдено {len(res['hits'])}:"]
    for h in res["hits"]:
        lines.append(f"  [{h['score']:.2f}] {h['path']}")
    o = res["opened"]
    if o:
        lines.append(f"\nПрочитал через {o.get('module')}:")
        body = (o.get("content") or o.get("transcript") or o.get("analysis")
                or o.get("summary") or o.get("error") or "(пусто)")
        lines.append(str(body)[:4000])
    return "\n".join(lines)
