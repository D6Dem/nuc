import io, os, json, csv, tarfile, zipfile, gzip, ast
from pathlib import Path

WORKSPACE = Path(os.environ.get("NUCLEUS_WORKSPACE", "./workspace")).resolve()
WORKSPACE.mkdir(parents=True, exist_ok=True)

WRITE_FORMATS = {
    ".txt":"text",".md":"text",".html":"text",".css":"text",".log":"text",
    ".json":"json",".jsonl":"jsonl",".yaml":"yaml",".yml":"yaml",
    ".toml":"toml",".csv":"csv",".tsv":"tsv",".xml":"text",
    ".py":"code",".js":"code",".ts":"code",".c":"code",".cpp":"code",
    ".h":"code",".rs":"code",".go":"code",".java":"code",".sh":"code",
    ".sql":"code",
    ".docx":"docx",".pdf":"pdf",
    ".png":"image",".jpg":"image",".jpeg":"image",
    ".zip":"zip",".gz":"gzip",".tar.gz":"tar.gz",
}

def _safe(name):
    if not name or name.startswith("/") or ".." in Path(name).parts:
        raise ValueError(f"небезопасный путь: {name}")
    p = (WORKSPACE / name).resolve()
    if WORKSPACE not in p.parents and p != WORKSPACE:
        raise ValueError(f"выход за workspace: {name}")
    return p

def _ext(name):
    n = name.lower()
    return ".tar.gz" if n.endswith(".tar.gz") else Path(n).suffix

def _w_text(p, c, **kw): p.write_text(str(c), encoding="utf-8")
def _w_json(p, c, **kw):
    if isinstance(c, str): c = json.loads(c)
    p.write_text(json.dumps(c, ensure_ascii=False, indent=2), encoding="utf-8")
def _w_jsonl(p, c, **kw):
    rows = c if isinstance(c, list) else [json.loads(l) for l in str(c).splitlines() if l.strip()]
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                 encoding="utf-8")
def _w_yaml(p, c, **kw):
    import yaml
    if isinstance(c, str): c = yaml.safe_load(c)
    p.write_text(yaml.safe_dump(c, allow_unicode=True, sort_keys=False), encoding="utf-8")
def _w_toml(p, c, **kw):
    import toml
    if isinstance(c, str): c = toml.loads(c)
    p.write_text(toml.dumps(c), encoding="utf-8")
def _w_csv(p, c, delim=",", **kw):
    if isinstance(c, str): c = [r.split(delim) for r in c.splitlines()]
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=delim)
        for row in c: w.writerow(row)
def _w_code(p, c, **kw):
    if _ext(p.name) == ".py":
        try: ast.parse(str(c))
        except SyntaxError as e: raise ValueError(f"SyntaxError line {e.lineno}: {e.msg}")
    p.write_text(str(c), encoding="utf-8")
def _w_docx(p, c, **kw):
    from docx import Document
    doc = Document()
    for para in str(c).split("\n\n"): doc.add_paragraph(para)
    doc.save(str(p))
def _w_pdf(p, c, **kw):
    from fpdf import FPDF
    pdf = FPDF(); pdf.add_page(); pdf.set_font("Helvetica", size=11)
    for para in str(c).split("\n\n"):
        pdf.multi_cell(0, 6, para); pdf.ln(2)
    pdf.output(str(p))
def _w_image(p, c, **kw):
    from PIL import Image
    spec = c if isinstance(c, dict) else {}
    img = Image.new("RGB", (spec.get("width", 256), spec.get("height", 256)),
                    tuple(spec.get("color", (30, 30, 30))))
    img.save(str(p))
def _w_zip(p, c, **kw):
    with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as z:
        for n, d in c.items(): z.writestr(n, d)
def _w_gzip(p, c, **kw):
    data = c.encode() if isinstance(c, str) else c
    with gzip.open(p, "wb") as f: f.write(data)
def _w_targz(p, c, **kw):
    with tarfile.open(p, "w:gz") as t:
        for n, d in c.items():
            b = d.encode() if isinstance(d, str) else d
            info = tarfile.TarInfo(n); info.size = len(b)
            t.addfile(info, io.BytesIO(b))

WRITERS = {"text":_w_text,"json":_w_json,"jsonl":_w_jsonl,"yaml":_w_yaml,
           "toml":_w_toml,"csv":lambda p,c,**k:_w_csv(p,c,",",**k),
           "tsv":lambda p,c,**k:_w_csv(p,c,"\t",**k),
           "code":_w_code,"docx":_w_docx,"pdf":_w_pdf,"image":_w_image,
           "zip":_w_zip,"gzip":_w_gzip,"tar.gz":_w_targz}


def create(path, content, **kw):
    p = _safe(path); ext = _ext(path)
    if ext not in WRITE_FORMATS:
        raise ValueError(f"формат {ext} не поддерживается. "
                         f"Доступно: {sorted(WRITE_FORMATS)}")
    kind = WRITE_FORMATS[ext]
    p.parent.mkdir(parents=True, exist_ok=True)
    WRITERS[kind](p, content, **kw)
    return str(p)


def list_capabilities():
    return {"write": sorted(WRITE_FORMATS)}


def run(user_input, ctx):
    import re
    logger = ctx["logger"]
    m = re.search(r'<<file:create\|(\{.*?\})>>', user_input, re.S)
    if m:
        try:
            spec = json.loads(m.group(1))
            out = create(spec["path"], spec.get("content", ""),
                         **(spec.get("opts") or {}))
            logger.record("file_write", {"path": out}, tags=["file_io", "write"])
            return f"OK: {out}"
        except Exception as e:
            return f"Ошибка записи: {e}"
    pm = re.search(r"(\S+\.\w+(?:\.gz)?)", user_input)
    if not pm:
        return ("Форматы записи: " + ", ".join(sorted(WRITE_FORMATS))
                + '. Формат: <<file:create|{"path":"a.txt","content":"..."}>>')
    return f'Используй: <<file:create|{{"path":"{pm.group(1)}","content":"..."}}>>'
