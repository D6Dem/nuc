"""Резидентный демон. Unix-socket. systemd-friendly."""
import os, socket, json, signal, sys, argparse
from .kernel import Kernel


class Resident:
    def __init__(self, args):
        self.kernel = Kernel(
            blocks_dir=args.blocks, memory_root=args.memory,
            tokenizer_dir=args.tokenizer,
            scan_roots=args.scan_roots.split(",") if args.scan_roots else None)
        self.kernel.start()
        self.sock_path = args.socket

    def handle(self, task):
        try:
            answer = self.kernel.think(task["input"])
            return {"answer": answer}
        except Exception as e:
            return {"error": str(e)}

    def serve(self):
        if os.path.exists(self.sock_path):
            os.unlink(self.sock_path)
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(self.sock_path); srv.listen(8)
        os.chmod(self.sock_path, 0o660)
        print(f"[nucleus] listening on {self.sock_path}", flush=True)
        while True:
            conn, _ = srv.accept()
            with conn:
                data = b""
                while not data.endswith(b"\n"):
                    chunk = conn.recv(4096)
                    if not chunk: break
                    data += chunk
                try:
                    task = json.loads(data.decode("utf-8"))
                    resp = self.handle(task)
                except Exception as e:
                    resp = {"error": str(e)}
                conn.sendall((json.dumps(resp, ensure_ascii=False) + "\n"
                              ).encode("utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", default="./blocks")
    ap.add_argument("--memory", default="./memory")
    ap.add_argument("--tokenizer", default="./blocks/qwen_tokenizer")
    ap.add_argument("--socket", default="/run/nucleus.sock")
    ap.add_argument("--scan-roots", default="")
    ap.add_argument("--nice", type=int, default=-5)
    a = ap.parse_args()

    try: os.nice(a.nice)
    except Exception: pass

    r = Resident(a)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    try: r.serve()
    finally: r.kernel.close()


if __name__ == "__main__":
    main()
