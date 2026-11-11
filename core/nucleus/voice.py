"""Wake-word (опционально, требует speechrecognition)."""
import threading


class VoiceTrigger:
    def __init__(self, wake_word, on_command, logger):
        self.wake_word = wake_word.lower()
        self.on_command = on_command
        self.logger = logger
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        try:
            import speech_recognition  # noqa
        except ImportError:
            self.logger.record("voice_disabled",
                               {"reason": "speechrecognition missing"},
                               tags=["voice"])
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        import speech_recognition as sr
        rec = sr.Recognizer()
        mic = sr.Microphone()
        with mic as src:
            rec.adjust_for_ambient_noise(src, duration=1)
        while not self._stop.is_set():
            try:
                with mic as src:
                    audio = rec.listen(src, timeout=5, phrase_time_limit=6)
                try:
                    text = rec.recognize_sphinx(audio).lower()
                except Exception:
                    continue
                if self.wake_word in text:
                    self.logger.record("voice_wake", {"text": text},
                                       tags=["voice", "wake"])
                    self.on_command(text)
            except Exception:
                continue

    def stop(self):
        self._stop.set()
