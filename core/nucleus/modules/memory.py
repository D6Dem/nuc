import re


def run(user_input, ctx):
    logger = ctx["logger"]
    stop = {"что","было","вспомни","память","лог","найди","покажи"}
    terms = [w for w in re.findall(r"[A-Za-zА-Яа-я0-9]{3,}", user_input)
             if w.lower() not in stop]
    results = []
    for t in terms:
        results.extend(logger.query(tags=[t.lower()], limit=3))
    if not results:
        results = logger.query(limit=5)
    if not results: return "Память пуста."
    return "Из памяти:\n" + "\n".join(
        f"[{e['iso']}] {e['type']}: {str(e['payload'])[:180]}"
        for e in results[:5])
