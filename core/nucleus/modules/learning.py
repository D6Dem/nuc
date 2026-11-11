import hashlib, time


def _sig(e):
    text = e.get("payload", {}).get("input", "") or str(e.get("payload", ""))
    words = sorted({w for w in text.lower().split() if len(w) > 3})[:10]
    return hashlib.sha1("|".join(words).encode()).hexdigest()[:12]


def consolidate(logger):
    created = 0
    last_ts = getattr(logger, "_last_consolidate_ts", 0.0)
    existing = {a.get("signature") for a in logger.load_algorithms()}
    for e in logger.query(entry_type="response", limit=1000):
        if e["ts"] <= last_ts: continue
        sig = _sig(e)
        if sig in existing: continue
        logger.append_algorithm({
            "signature": sig,
            "ts": time.time(),
            "module_chain": e["payload"].get("modules", []),
            "source_ref": f"mem://{e['iso'][:10]}/{e['id']}",
            "success": 1.0,
        })
        created += 1
    logger._last_consolidate_ts = time.time()
    logger.record("consolidation", {"new": created}, tags=["learning"])
    return created
