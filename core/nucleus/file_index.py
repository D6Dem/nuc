import os, re, sqlite3, threading
from pathlib import Path
from difflib import SequenceMatcher

_TR = {'а':'a','б':'b','в':'v','г':'g','д':'d','е':'e','ё':'e','ж':'zh','з':'z',
       'и':'i','й':'y','к':'k','л':'l','м':'m','н':'n','о':'o','п':'p','р':'r',
       'с':'s','т':'t','у':'u','ф':'f','х':'kh','ц':'ts','ч':'ch','ш':'sh',
       'щ':'shch','ъ':'','ы':'y','ь':'','э':'e','ю':'yu','я':'ya'}

def _norm(s):
    s = "".join(_TR.get(c, c) for c in s.lower())
    return re.sub(r"[^a-z0-9]+", " ", s).strip()
def _tok(s): return {t for t in _norm(s).split() if len(t) >= 2}
def _tri(s):
    s = f"  {_norm(s)}  "
    return {s[i:i+3] for i in range(len(s)-2)}
def _jac(a, b): return len(a & b) / len(a | b) if a and b else 0.0
def _seq(a, b): return SequenceMatcher(None, _norm(a), _norm(b)).ratio()

KINDS = {
    "text": {".txt",".md",".log",".ini",".cfg",".rtf"},
    "code": {".py",".js",".ts",".c",".cpp",".h",".rs",".go",".java",
             ".sh",".sql",".html",".css"},
    "data": {".json",".jsonl",".yaml",".yml",".toml",".csv",".tsv",".xml"},
    "doc":  {".docx",".pdf"},
    "audio":{".wav",".mp3",".ogg",".flac",".m4a"},
    "video":{".mp4",".avi",".mkv",".mov",".webm"},
    "image":{".png",".jpg",".jpeg",".gif",".bmp",".webp"},
    "archive":{".zip",".gz",".tar"},
}
EXT2KIND = {e: k for k, es in KINDS.items() for e in es}
EXCL = {".git",".hg",".svn","__pycache__","node_modules",".cache",".venv",
        "venv",".tox","build","dist"}
TEXT_KINDS = {"text","code","data"}
SNIP = 600
MAX_SIZE = 5 * 1024 * 1024


class FileIndex:
    def __init__(self, db_path, roots, max_depth=8):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.roots = [Path(os.path.expanduser(r)).resolve() for r in roots]
        self.max_depth = max_depth
        self._db = sqlite3.connect(str(self.db_path),
                                   check_same_thread=False, timeout=15.0)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._init()
        self._lock = threading.Lock()

    def _init(self):
        self._db.executescript("""
        CREATE TABLE IF NOT EXISTS files (
            path TEXT PRIMARY KEY, dir TEXT, name TEXT, ext TEXT, kind TEXT,
            size INTEGER, mtime REAL, norm_name TEXT, snippet TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_kind ON files(kind);
        CREATE INDEX IF NOT EXISTS idx_norm ON files(norm_name);
        CREATE VIRTUAL TABLE IF NOT EXISTS files_fts USING fts5(
            path UNINDEXED, name, snippet, tokenize='unicode61'
        );
        """)
        self._db.commit()

    def _iter(self, root, depth=0):
        if depth > self.max_depth: return
        try:
            with os.scandir(root) as it:
                for e in it:
                    try:
                        if e.is_symlink(): continue
                        if e.is_dir(follow_symlinks=False):
                            if e.name in EXCL or e.name.startswith("."): continue
                            yield from self._iter(Path(e.path), depth+1)
                        elif e.is_file(follow_symlinks=False):
                            yield Path(e.path)
                    except (PermissionError, OSError): continue
        except (PermissionError, OSError):
            return

    def _snip(self, p, kind):
        if kind not in TEXT_KINDS: return ""
        try:
            if p.stat().st_size > MAX_SIZE: return ""
            with p.open("r", encoding="utf-8", errors="ignore") as f:
                return f.read(SNIP)
        except Exception:
            return ""

    def scan(self, roots=None):
        roots = [Path(os.path.expanduser(r)).resolve() for r in (roots or self.roots)]
        added = updated = removed = 0
        seen = set()
        cur = self._db.cursor()
        for root in roots:
            if not root.exists(): continue
            for p in self._iter(root):
                try: st = p.stat()
                except OSError: continue
                ext = p.suffix.lower()
                kind = EXT2KIND.get(ext, "other")
                if kind == "other" and not p.suffix: continue
                norm = _norm(p.name)
                snip = self._snip(p, kind)
                seen.add(str(p))
                row = cur.execute("SELECT mtime, size FROM files WHERE path=?",
                                  (str(p),)).fetchone()
                if row is None:
                    cur.execute("INSERT INTO files VALUES (?,?,?,?,?,?,?,?,?)",
                                (str(p), str(p.parent), p.name, ext, kind,
                                 st.st_size, st.st_mtime, norm, snip))
                    cur.execute("INSERT INTO files_fts VALUES (?,?,?)",
                                (str(p), p.name, snip))
                    added += 1
                elif row[0] != st.st_mtime or row[1] != st.st_size:
                    cur.execute("""UPDATE files SET size=?, mtime=?, kind=?,
                                   norm_name=?, snippet=? WHERE path=?""",
                                (st.st_size, st.st_mtime, kind, norm, snip, str(p)))
                    cur.execute("DELETE FROM files_fts WHERE path=?", (str(p),))
                    cur.execute("INSERT INTO files_fts VALUES (?,?,?)",
                                (str(p), p.name, snip))
                    updated += 1
        self._db.commit()
        if roots:
            pref = tuple(str(r) + os.sep for r in roots)
            for (path,) in cur.execute("SELECT path FROM files").fetchall():
                if path.startswith(pref) and path not in seen:
                    cur.execute("DELETE FROM files WHERE path=?", (path,))
                    cur.execute("DELETE FROM files_fts WHERE path=?", (path,))
                    removed += 1
            self._db.commit()
        return {"added": added, "updated": updated, "removed": removed,
                "roots": [str(r) for r in roots]}

    def search(self, query, kind=None, limit=10, min_score=0.10):
        cur = self._db.cursor()
        qn = _norm(query); qtok = _tok(query); qtri = _tri(query)
        cands = {}
        try:
            cur.execute("SELECT path FROM files_fts WHERE files_fts MATCH ? LIMIT 200",
                        (query.replace('"', ''),))
            for (p,) in cur.fetchall(): cands[p] = None
        except sqlite3.Error:
            pass
        if qn:
            cur.execute("SELECT path, name, kind, size, mtime, norm_name, snippet "
                        "FROM files WHERE norm_name LIKE ? LIMIT 200",
                        (f"%{qn.split()[0]}%",))
            for r in cur.fetchall(): cands[r[0]] = r
        scored = []
        for p in cands:
            r = cands[p]
            if r is None:
                r = cur.execute("SELECT path, name, kind, size, mtime, norm_name, snippet "
                                "FROM files WHERE path=?", (p,)).fetchone()
                if not r: continue
            path, name, k, size, mtime, norm, snip = r
            if kind and k != kind: continue
            score = (0.40 * _jac(qtok, _tok(name))
                     + 0.25 * _jac(qtri, _tri(name))
                     + 0.20 * _seq(query, name)
                     + 0.15 * _jac(qtok, _tok(snip or "")))
            if score >= min_score:
                scored.append({"path": path, "name": name, "kind": k,
                               "size": size, "mtime": mtime,
                               "score": round(score, 4),
                               "snippet": (snip or "")[:200]})
        scored.sort(key=lambda x: -x["score"])
        return scored[:limit]

    def stats(self):
        cur = self._db.cursor()
        total = cur.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        by_kind = dict(cur.execute(
            "SELECT kind, COUNT(*) FROM files GROUP BY kind").fetchall())
        return {"total": total, "by_kind": by_kind,
                "roots": [str(r) for r in self.roots]}

    def close(self): self._db.close()
