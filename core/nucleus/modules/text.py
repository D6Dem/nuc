import re


def _profile(text):
    ws = re.findall(r"\w+", text.lower())
    if not ws: return {}
    ss = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    return {"ttr": round(len(set(ws)) / len(ws), 3),
            "avg_word": round(sum(map(len, ws)) / len(ws), 2),
            "avg_sent": round(sum(len(s.split()) for s in ss) / max(1, len(ss)), 2)}


def run(user_input, ctx):
    m = re.search(r"(\S+\.(txt|md|log))", user_input, re.I)
    if m:
        try:
            with open(m.group(1), encoding="utf-8", errors="ignore") as f:
                c = f.read(50000)
            return f"Профиль текста {m.group(1)}: {_profile(c)}"
        except Exception as e:
            return f"Ошибка чтения: {e}"
    return "(text-модуль активен; для анализа укажи файл .txt/.md/.log)"
