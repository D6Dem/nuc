"""
MemoryStore: JSONL + SQLite FTS5 + algorithms.jsonl.
Ротация: context_end / idle_30m / size.
"""
import json, os, sqlite3, threading, time, gzip, hashlib
from datetime import datetime, timezone
from pathlib import Path


class Logger:
    def __init__(self, root="./memory", idle_rotate_sec=1800,
                 max_hot_bytes=256*1024*1024, archive_days=7):
        self.root = Path(root)
        for sub in ("session", "daily", "archive"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        self.hot_path = self.root / "session" / "current.jsonl"
        self.db_path = self.root / "index.sqlite"
        self.algo_path = self.root / "algorithms.jsonl"
        self.idle_rotate_sec = idle_rotate_sec
        self.max_hot_bytes = max_hot_bytes
        self.archive_days = archive_days

        self._lock = threading.RLock()
        self._seq = self._count_lines(self.hot_path)
        self._bytes = self.hot_path.stat().st_size if self.hot_path.exists() else 0
        self._session = self._new_session_id()
        self._last_activity = time.time()
        self._buffer = []

        self._hot = open(self.hot_path, "a", encoding="utf-8")
        self._db = sqlite3.connect(str(self.db_path),
                                   check_same_thread=False, timeout=15.0)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()

    def _new_session_id(self):
        return f"sess_{datetime.now(timezone.utc):%Y%m%d}_{os.urandom(2).hex()}"

    @staticmethod
    def _count_lines(p):
        if not p.exists(): return 0
        with p.open("rb") as f: return sum(1 for _ in f)

    def _init_schema(self):
        self._db.executescript("""
        CREATE TABLE IF NOT EXISTS entries (
            id TEXT PRIMARY KEY, ts REAL NOT NULL, iso TEXT NOT NULL,
            session TEXT, seq INTEGER, type TEXT NOT NULL,
            actor TEXT, sig TEXT, size INTEGER
        );
        CREATE TABLE IF NOT EXISTS tags (entry_id TEXT, tag TEXT);
        CREATE TABLE IF NOT EXISTS refs (entry_id TEXT, ref_id TEXT);
        CREATE INDEX IF NOT EXISTS idx_ts ON entries(ts);
        CREATE INDEX IF NOT EXISTS idx_type ON entries(type);
        CREATE INDEX IF NOT EXISTS idx_tag ON tags(tag);
        CREATE INDEX IF NOT EXISTS idx_ref ON refs(ref_id);
        CREATE VIRTUAL TABLE IF NOT EXISTS entries_fts
            USING fts5(id UNINDEXED, text, tokenize='unicode61');
        """)
        self._db.commit()

    def record(self, type_, payload, tags=None, refs=None,
               actor="kernel", sig=None):
        with self._lock:
            self._seq += 1
            ts = time.time()
            body = json.dumps(payload, ensure_ascii=False, sort_keys=True)
            eid = hashlib.sha1(f"{ts}{body}{self._seq}".encode()).hexdigest()[:16]
            entry = {"id": eid, "ts": ts,
                     "iso": datetime.now(timezone.utc).isoformat(),
                     "session": self._session, "seq": self._seq,
                     "type": type_, "actor": actor,
                     "tags": tags or [], "refs": refs or [],
                     "sig": sig or hashlib.sha1(body.encode()).hexdigest()[:12],
                     "payload": payload, "v": 1}
            line = json.dumps(entry, ensure_ascii=False) + "\n"
            self._hot.write(line); self._hot.flush()
            try: os.fsync(self._hot.fileno())
            except OSError: pass
            self._bytes += len(line.encode("utf-8"))
            self._last_activity = time.time()
            self._buffer.append(entry)
            if len(self._buffer) >= 32: self._flush_index()
            if self._bytes > self.max_hot_bytes:
                self.rotate(reason="size")
            return eid

    def _flush_index(self):
        if not self._buffer: return
        cur = self._db.cursor()
        for e in self._buffer:
            try:
                cur.execute("INSERT OR IGNORE INTO entries VALUES (?,?,?,?,?,?,?,?,?)",
                            (e["id"], e["ts"], e["iso"], e["session"], e["seq"],
                             e["type"], e["actor"], e["sig"], len(str(e["payload"]))))
                for t in e["tags"]:
                    cur.execute("INSERT INTO tags VALUES (?,?)", (e["id"], t.lower()))
                for r in e["refs"]:
                    cur.execute("INSERT INTO refs VALUES (?,?)", (e["id"], r))
                text = " ".join(str(v) for v in e["payload"].values()) \
                       if isinstance(e["payload"], dict) else str(e["payload"])
                cur.execute("INSERT INTO entries_fts (id, text) VALUES (?,?)",
                            (e["id"], text))
            except sqlite3.Error:
                pass
        self._db.commit()
        self._buffer.clear()

    def should_rotate_idle(self):
        return (time.time() - self._last_activity) >= self.idle_rotate_sec

    def rotate(self, reason="context_end"):
        with self._lock:
            self._flush_index()
            self._hot.close()
            daily = self.root / "daily" / \
                    f"{datetime.now(timezone.utc):%Y-%m-%d}_{os.urandom(2).hex()}.jsonl"
            try:
                self.hot_path.rename(daily)
            except OSError:
                daily = self.hot_path
            meta = {"reason": reason, "file": str(daily),
                    "entries": self._seq, "bytes": self._bytes}
            self._session = self._new_session_id()
            self._seq = 0; self._bytes = 0
            self._hot = open(self.hot_path, "a", encoding="utf-8")
            self._compress_old()
            self.record("idle", {"rotate": meta}, tags=["memory", "rotate"])
            return meta

    def _compress_old(self):
        cutoff = time.time() - self.archive_days * 86400
        for f in (self.root / "daily").glob("*.jsonl"):
            if f.stat().st_mtime >= cutoff: continue
            month = datetime.fromtimestamp(f.stat().st_mtime, timezone.utc).strftime("%Y-%m")
            gz = self.root / "archive" / f"{month}_{f.stem}.jsonl.gz"
            if gz.exists(): continue
            with f.open("rb") as src, gzip.open(gz, "wb") as dst:
                dst.write(src.read())
            f.unlink()

    # ---------- read ----------
    def query(self, tags=None, entry_type=None, limit=10):
        cur = self._db.cursor()
        sql = "SELECT id,ts,iso,type,tags,payload FROM entries WHERE 1=1"
        params = []
        if entry_type:
            sql += " AND type=?"; params.append(entry_type)
        if tags:
            for t in tags:
                sql += " AND tags LIKE ?"; params.append(f'%"{t}"%')
        sql += " ORDER BY ts DESC LIMIT ?"; params.append(limit)
        rows = cur.execute(sql, params).fetchall()
        # Fallback: если индекс пуст, читаем hot-файл
        if not rows and self.hot_path.exists():
            entries = []
            with self.hot_path.open("r", encoding="utf-8") as f:
                for line in f:
                    try: entries.append(json.loads(line))
                    except: continue
            entries.sort(key=lambda e: e["ts"], reverse=True)
            if entry_type: entries = [e for e in entries if e["type"] == entry_type]
            if tags:
                entries = [e for e in entries
                           if any(t in e.get("tags", []) for t in tags)]
            return entries[:limit]
        out = []
        for r in rows:
            try: tg = json.loads(r[4])
            except: tg = []
            try: pl = json.loads(r[5])
            except: pl = {}
            out.append({"id": r[0], "ts": r[1], "iso": r[2], "type": r[3],
                        "tags": tg, "payload": pl})
        return out

    def recall_digest(self, query, limit=5):
        cur = self._db.cursor()
        rows = []
        try:
            rows = cur.execute(
                "SELECT id, text FROM entries_fts WHERE entries_fts MATCH ? LIMIT ?",
                (query.replace('"', ''), limit)).fetchall()
        except sqlite3.Error:
            pass
        if not rows:
            rows = [(e["id"], str(e["payload"])[:200])
                    for e in self.query(limit=limit)]
        return "\n".join(f"[{rid}] {txt[:200]}" for rid, txt in rows)

    # ---------- algorithms ----------
    def append_algorithm(self, algo):
        with self._lock, self.algo_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(algo, ensure_ascii=False) + "\n")

    def load_algorithms(self):
        if not self.algo_path.exists(): return []
        out = []
        with self.algo_path.open("r", encoding="utf-8") as f:
            for line in f:
                try: out.append(json.loads(line))
                except: pass
        return out

    def close(self):
        with self._lock:
            self._flush_index()
            try: self._hot.close()
            except: pass
            try: self._db.close()
            except: pass
