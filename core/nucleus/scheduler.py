import time, threading


class Scheduler:
    def __init__(self, kernel, idle_seconds=1800, poll=30, scan_every=6*3600):
        self.kernel = kernel
        self.idle_seconds = idle_seconds
        self.poll = poll
        self.scan_every = scan_every
        self.last_activity = time.time()
        self._last_scan = time.time()
        self._stop = threading.Event()
        self._thread = None

    def touch(self):
        self.last_activity = time.time()

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            time.sleep(self.poll)
            now = time.time()
            if now - self.last_activity > self.idle_seconds:
                try:
                    self.kernel.rotate_memory(reason="idle_30m")
                    self.kernel.run_idle_learning()
                except Exception:
                    pass
                self.last_activity = now
            if now - self._last_scan > self.scan_every and \
               now - self.last_activity > 60:
                try:
                    self.kernel.file_index.scan()
                    self._last_scan = now
                except Exception:
                    pass

    def stop(self):
        self._stop.set()
