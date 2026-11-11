import re, collections


def _chunks(t, size=2000, overlap=200):
    out, i = [], 0
    while i < len(t):
        out.append(t[i:i+size]); i += size - overlap
    return out


def run(user_input, ctx):
    m = re.search(r"(\S+\.(txt|md|log|jsonl|py))", user_input, re.I)
    if not m: return "Укажи путь к файлу для рекурсивного анализа."
    try:
        with open(m.group(1), encoding="utf-8", errors="ignore") as f: t = f.read()
    except Exception as e:
        return f"Ошибка чтения: {e}"
    words = re.findall(r"[A-Za-zА-Яа-я]{4,}", t.lower())
    top = collections.Counter(words).most_common(15)
    if "logger" in ctx:
        ctx["logger"].record("recursive",
                             {"path": m.group(1), "chunks": len(_chunks(t))},
                             tags=["recursive"])
    return (f"Рекурсивный анализ {m.group(1)} "
            f"({len(t)} байт, {len(_chunks(t))} чанков):\nТоп-термины: {top}")
