import argparse, sys
from .kernel import Kernel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", default="./blocks")
    ap.add_argument("--memory", default="./memory")
    ap.add_argument("--tokenizer", default="./blocks/qwen_tokenizer")
    ap.add_argument("--scan-roots", default="")
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--voice", action="store_true")
    a = ap.parse_args()

    roots = a.scan_roots.split(",") if a.scan_roots else None
    k = Kernel(blocks_dir=a.blocks, memory_root=a.memory,
               tokenizer_dir=a.tokenizer, scan_roots=roots)
    k.start()
    print(f"[{k.name}] core_ok={k.core_ok} "
          f"experts={len(k.store.available())} "
          f"tokenizer={k.tokenizer is not None}")

    if a.scan:
        print("scanning filesystem…")
        print(k.scan())

    if a.voice:
        from .voice import VoiceTrigger
        v = VoiceTrigger(k.name, lambda t: k.think(t), k.logger)
        v.start()

    print("Команды: /scan /stats /learn /rotate /quit")
    try:
        while True:
            try: line = input("> ").strip()
            except EOFError: break
            if not line: continue
            if line in ("/quit", "/exit"): break
            if line == "/scan": print(k.scan()); continue
            if line == "/stats": print(k.file_index.stats()); continue
            if line == "/learn": print("created:", k.run_idle_learning()); continue
            if line == "/rotate": print("rotated:", k.rotate_memory()); continue
            print(k.think(line))
    finally:
        k.close()


if __name__ == "__main__":
    main()
