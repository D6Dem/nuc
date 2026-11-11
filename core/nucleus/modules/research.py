def _ddg(q, n=5):
    try:
        from duckduckgo_search import DDGS
        with DDGS() as d:
            return list(d.text(q, max_results=n))
    except Exception:
        return []


def run(user_input, ctx):
    hits = _ddg(user_input)
    if not hits:
        return "(research недоступен: нет сети или ddg-search не установлен)"
    parts = [f"- {h.get('title','?')}: {(h.get('body') or '')[:300]}"
             for h in hits[:3]]
    if "logger" in ctx:
        ctx["logger"].record("research", {"query": user_input, "hits": len(hits)},
                             tags=["research"])
    return "Research:\n" + "\n".join(parts)
