import ast, subprocess, tempfile, os, sys, re


def _check(code):
    try: ast.parse(code); return None
    except SyntaxError as e: return f"line {e.lineno}: {e.msg}"


def _run(code, timeout=15):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code); p = f.name
    try:
        r = subprocess.run([sys.executable, p], capture_output=True,
                           text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "timeout"
    finally:
        try: os.unlink(p)
        except OSError: pass


def run(user_input, ctx):
    m = re.search(r"```(?:python)?\n(.*?)```", user_input, re.S)
    if not m:
        return "Оберни код в ```python ... ``` для запуска."
    code = m.group(1)
    err = _check(code)
    if err: return f"Синтаксическая ошибка: {err}"
    rc, out, e = _run(code)
    if "logger" in ctx:
        ctx["logger"].record("code_exec", {"code": code[:500], "rc": rc},
                             tags=["code", "exec"])
    return f"rc={rc}\nstdout:\n{out}\nstderr:\n{e}"
