import subprocess, tempfile, os, re


def _audio(vp):
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    try:
        subprocess.run(["ffmpeg", "-y", "-i", vp, "-ac", "1", "-ar", "16000", tmp],
                       check=True, capture_output=True, timeout=300)
        return tmp
    except Exception:
        return None


def run(user_input, ctx):
    m = re.search(r"(\S+\.(mp4|avi|mkv|mov|webm))", user_input, re.I)
    if not m: return "Укажи путь к видеофайлу."
    p = m.group(1)
    if not os.path.exists(p): return f"Не найден: {p}"
    aud = _audio(p)
    transcript = ""
    if aud:
        try:
            from . import audio as _a
            transcript = _a._asr(aud)
        finally:
            try: os.unlink(aud)
            except OSError: pass
    if "logger" in ctx:
        ctx["logger"].record("video_analyze", {"path": p}, tags=["video"])
    return f"Видео {p}\nТранскрипт:\n{transcript[:1500]}"
