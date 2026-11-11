import os, re


def _asr(path):
    try:
        import speech_recognition as sr
    except ImportError:
        return "[speechrecognition не установлен — модуль неактивен]"
    rec = sr.Recognizer()
    try:
        with sr.AudioFile(path) as src:
            audio = rec.record(src)
        try:
            return rec.recognize_sphinx(audio)
        except Exception:
            try:
                return rec.recognize_google(audio, language="ru-RU")
            except Exception as e:
                return f"[распознавание: {e}]"
    except Exception as e:
        return f"[чтение: {e}]"


def run(user_input, ctx):
    m = re.search(r"(\S+\.(wav|mp3|ogg|flac|m4a))", user_input, re.I)
    if not m: return "Укажи путь к аудиофайлу."
    p = m.group(1)
    if not os.path.exists(p): return f"Не найден: {p}"
    txt = _asr(p)
    if "logger" in ctx:
        ctx["logger"].record("audio_transcribe", {"path": p, "text": txt[:500]},
                             tags=["audio"])
    return f"Транскрипт {p}:\n{txt}"
