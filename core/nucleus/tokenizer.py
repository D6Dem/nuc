"""
BPE-токенизатор + ChatML.
Работает с vocab.json + merges.txt от Qwen2.5 или любой GPT-2-style BPE.
Fallback: byte-level, если файлов нет.
"""
import json, re
from pathlib import Path


def bytes_to_unicode():
    """Стандартный GPT-2 bytes_to_unicode."""
    bs = list(range(ord("!"), ord("~")+1)) + \
         list(range(ord("¡"), ord("¬")+1)) + \
         list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    cs = [chr(c) for c in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    return {(word[i], word[i+1]) for i in range(len(word)-1)}


class BPETokenizer:
    def __init__(self, vocab_path=None, merges_path=None,
                 special_tokens=None):
        self.special = special_tokens or {
            "bos": 151643, "eos": 151645,
            "im_start": 151644, "im_end": 151645,
        }
        self.special_ids = {"bos": [self.special["bos"]],
                            "eos": [self.special["eos"]],
                            "im_end": [self.special["im_end"]]}
        if vocab_path and Path(vocab_path).exists() and \
           merges_path and Path(merges_path).exists():
            self.vocab = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
            merges_raw = Path(merges_path).read_text(encoding="utf-8").splitlines()
            if merges_raw and merges_raw[0].startswith("#"):
                merges_raw = merges_raw[1:]
            self.merges = {}
            for i, line in enumerate(merges_raw):
                parts = line.split()
                if len(parts) == 2:
                    self.merges[(parts[0], parts[1])] = i
            self.inv = {v: k for k, v in self.vocab.items()}
            self.byte_encoder = bytes_to_unicode()
            self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
            self.pat = re.compile(
                r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+""",
                re.UNICODE)
            self.available = True
        else:
            self.available = False

    def _bpe(self, token):
        """token: str (bytes already mapped via byte_encoder)."""
        word = tuple(token)
        while True:
            pairs = get_pairs(word)
            if not pairs: break
            bigram = min(pairs, key=lambda p: self.merges.get(p, float('inf')))
            if bigram not in self.merges: break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j]); i = j
                except ValueError:
                    new_word.extend(word[i:]); break
                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first + second); i += 2
                else:
                    new_word.append(word[i]); i += 1
            word = tuple(new_word)
            if len(word) == 1: break
        return " ".join(word)

    def encode(self, text: str) -> list[int]:
        if not self.available:
            return list(text.encode("utf-8"))
        ids = []
        for token in re.findall(self.pat, text):
            token_bytes = token.encode("utf-8")
            token_str = "".join(self.byte_encoder[b] for b in token_bytes)
            bpe_result = self._bpe(token_str).split(" ")
            for piece in bpe_result:
                if piece in self.vocab:
                    ids.append(self.vocab[piece])
                else:
                    for ch in piece:
                        ids.append(self.vocab.get(ch, 0))
        return ids

    def decode(self, ids: list[int]) -> str:
        if not self.available:
            return bytes(i & 0xFF for i in ids).decode("utf-8", errors="ignore")
        parts = [self.inv.get(i, "") for i in ids if i not in self.special.values()]
        text = "".join(parts)
        return bytearray([self.byte_decoder[c] for c in text
                          if c in self.byte_decoder]).decode("utf-8", errors="ignore")


def format_chatml(system: str, history: list, user: str) -> str:
    """ChatML — стандарт Qwen/GPT-OpenAI."""
    parts = [f"<|im_start|>system\n{system}<|im_end|>"]
    for turn in history:
        parts.append(f"<|im_start|>{turn['role']}\n{turn['text']}<|im_end|>")
    parts.append(f"<|im_start|>user\n{user}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)
