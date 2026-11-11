"""Тонкий клиент к демону."""
import socket, json, argparse


def send(sock_path, text):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(sock_path)
    s.sendall((json.dumps({"input": text}, ensure_ascii=False) + "\n").encode("utf-8"))
    buf = b""
    while not buf.endswith(b"\n"):
        chunk = s.recv(8192)
        if not chunk: break
        buf += chunk
    s.close()
    return json.loads(buf.decode("utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--socket", default="/run/nucleus.sock")
    ap.add_argument("query", nargs="*")
    a = ap.parse_args()
    if a.query:
        print(send(a.socket, " ".join(a.query))["answer"])
    else:
        print("Nucleus REPL (/quit)")
        while True:
            try: line = input("> ").strip()
            except EOFError: break
            if line in ("/quit", "/exit"): break
            if not line: continue
            r = send(a.socket, line)
            print(r.get("answer", r.get("error", "?")))


if __name__ == "__main__":
    main()
