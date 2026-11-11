import re
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from .logger import Logger
from .store import BlockStore
from .block_router import BlockRouter
from .file_index import FileIndex
from .blocks import KVCache
from .tokenizer import BPETokenizer, format_chatml
from .memory_pool import MemoryPool
from .scheduler import Scheduler
from .modules import (file_io, file_search, text, programming, research,
                      memory as mem_mod, recursive, audio, video, learning)

FILE_TRIGGERS = ("файл", "файлы", "открой", "прочитай", "найди файл",
                 "документ", "видео", "аудио", ".pdf", ".docx", ".txt",
                 ".md", ".mp4", ".wav", "file", "open", "read")


class ConversationBuffer:
    def __init__(self, max_turns=6):
        self.max_turns = max_turns
        self.turns = []

    def add(self, role, text):
        self.turns.append({"role": role, "text": text})
        if len(self.turns) > self.max_turns * 2:
            self.turns = self.turns[-self.max_turns * 2:]

    def format(self):
        return "\n".join(f"{t['role']}: {t['text']}" for t in self.turns)


class Kernel:
    SYNTH_SYS = (
        "Ты — Nucleus, живое ядро блочного ИИ. "
        "Отвечай на вопрос пользователя СВОИМИ СЛОВАМИ, "
        "опираясь на данные, которые вернули твои модули. "
        "НЕ переписывай их как есть, НЕ выводи JSON, "
        "НЕ пиши '[file_search]' или 'модуль вернул'. "
        "Перескажи суть, ответь на вопрос, добавь связки. "
        "Если данных не хватает — скажи, что уточнить. "
        "Стиль живой, дружелюбный, короткий."
    )

    def __init__(self, name="Nucleus", blocks_dir="./blocks",
                 memory_root="./memory", tokenizer_dir=None,
                 scan_roots=None, n_ctx=2048):
        self.name = name
        self.n_ctx = n_ctx
        self.logger = Logger(memory_root)

        self.store = BlockStore(blocks_dir, max_experts=2)
        try:
            self.store.load_core("core.nblk")
            self.core_ok = True
        except FileNotFoundError:
            self.core_ok = False
            self.logger.record("no_core", {"blocks_dir": str(blocks_dir)},
                               tags=["boot", "warn"])

        self.router = BlockRouter(self.store, self.logger)

        self.tokenizer = None
        if tokenizer_dir:
            vp = Path(tokenizer_dir) / "vocab.json"
            mp = Path(tokenizer_dir) / "merges.txt"
            if vp.exists() and mp.exists():
                try:
                    self.tokenizer = BPETokenizer(str(vp), str(mp))
                except Exception as e:
                    self.logger.record("tokenizer_error", {"error": str(e)},
                                       tags=["boot", "warn"])

        self.file_index = FileIndex(
            str(Path(memory_root) / "file_index.sqlite"),
            roots=scan_roots or ["~/Documents", "~/Downloads", "~/Desktop",
                                 "./workspace"])

        self.tools = {"file_io": file_io, "file_search": file_search,
                      "text": text, "programming": programming,
                      "research": research, "memory": mem_mod,
                      "recursive": recursive, "audio": audio,
                      "video": video, "learning": learning}

        self.history = ConversationBuffer(max_turns=6)
        self.memory_pool = MemoryPool()
        self.scheduler = Scheduler(self)
        self._started = False

        self.logger.record("boot", {
            "name": name, "core_ok": self.core_ok,
            "experts": self.store.available() if self.core_ok else [],
            "tokenizer": self.tokenizer is not None,
        }, tags=["boot"])

    def start(self):
        if self._started: return
        self._started = True
        self.scheduler.start()

    def close(self):
        try: self.scheduler.stop()
        except: pass
        try: self.file_index.close()
        except: pass
        try: self.logger.close()
        except: pass

    # ---------- plan ----------
    def _plan(self, q):
        low = q.lower()
        plan = {"tools": [], "experts": []}
        if self.core_ok:
            plan["experts"] = self.router.select(q)

        file_intent = (any(t in low for t in FILE_TRIGGERS)
                       or bool(re.search(
                           r"\S+\.(pdf|docx|txt|md|mp4|wav|json|csv|py)\b", low)))
        if file_intent:
            plan["tools"].append("file_search")
            plan["maybe_research"] = True
            return plan

        if any(k in low for k in ("найди","поиск","search","research",
                                  "кто такой","что такое")):
            plan["tools"].append("research")
        if any(k in low for k in ("код","code","python","скрипт","```")):
            plan["tools"].append("programming")
        if "<<file:create" in low:
            plan["tools"].append("file_io")
        if any(k in low for k in ("вспомни","память","memory","что было")):
            plan["tools"].append("memory")
        if not plan["tools"] and not plan["experts"]:
            plan["tools"].append("memory")
        return plan

    @staticmethod
    def _file_useful(s):
        if not s: return False
        low = s.lower()
        return ("не найдено" not in low
                and "не найден" not in low
                and ("прочитал" in low or len(s) > 300))

    # ---------- tools ----------
    def _run_tools(self, plan, user_input, task_id):
        tools = plan.get("tools", [])
        if not tools: return {}
        ctx = {"kernel": self, "logger": self.logger, "task_id": task_id}
        results = {}
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {pool.submit(self.tools[t].run, user_input, ctx): t
                       for t in tools if t in self.tools}
            for fut in as_completed(futures):
                name = futures[fut]
                try:
                    results[name] = fut.result()
                    self.logger.record("module_call",
                                       {"module": name, "query": user_input[:300]},
                                       tags=["call", name], refs=[task_id])
                except Exception as e:
                    results[name] = f"error: {e}"
                    self.logger.record("module_error",
                                       {"module": name, "error": str(e)},
                                       tags=["error", name], refs=[task_id])

        # file_search пуст + maybe_research → досылаем research
        if plan.get("maybe_research") and "file_search" in results \
           and not self._file_useful(results["file_search"]) \
           and "research" not in results:
            try:
                results["research"] = self.tools["research"].run(user_input, ctx)
            except Exception as e:
                results["research"] = f"error: {e}"
        return results

    # ---------- synthesis ----------
    def _build_context(self, results, max_chars_each=3000):
        parts = []
        for name, val in results.items():
            s = str(val)
            if len(s) > max_chars_each:
                s = s[:max_chars_each] + "…"
            parts.append(f"<{name}>\n{s}\n</{name}>")
        return "\n".join(parts) if parts else ""

    def _build_prompt(self, user_input, context, memory_digest):
        system = self.SYNTH_SYS
        if context:
            system += f"\n\n<context>\n{context}\n</context>"
        if memory_digest:
            system += f"\n\n<memory>\n{memory_digest}\n</memory>"

        if self.tokenizer:
            return format_chatml(system, self.history.turns, user_input)
        # plain-text fallback
        parts = [f"SYSTEM: {system}", ""]
        if self.history.turns:
            parts.append("HISTORY:")
            for turn in self.history.turns:
                parts.append(f"  {turn['role']}: {turn['text']}")
            parts.append("")
        parts.append(f"USER: {user_input}")
        parts.append("ASSISTANT:")
        return "\n".join(parts)

    def _tokenize(self, text):
        if self.tokenizer: return self.tokenizer.encode(text)
        return list(text.encode("utf-8"))

    def _detokenize(self, ids):
        if self.tokenizer: return self.tokenizer.decode(ids)
        return bytes(i & 0xFF for i in ids).decode("utf-8", errors="ignore")

    def _generate_str(self, prompt, max_new=200, temperature=0.7):
        if not self.core_ok: return ""
        vocab_size = self.store.core.vocab

        ids = self._tokenize(prompt)
        # защита: id вне словаря core (рассинхрон токенизатора и core)
        ids = [i for i in ids if 0 <= i < vocab_size]
        if not ids:
            return ""  # ни одного валидного токена — нечего генерить
        if len(ids) > self.n_ctx - max_new:
            ids = ids[-(self.n_ctx - max_new):]

        eos_ids = set()
        if self.tokenizer and self.tokenizer.available:
            eos_ids = set(self.tokenizer.special_ids.get("eos", [])) | \
                      set(self.tokenizer.special_ids.get("im_end", []))
        eos_ids = {i for i in eos_ids if i < vocab_size}

        kv = KVCache(self.store.core.n_layers, self.store.core.n_heads,
                     self.store.core.d_head, max_seq=len(ids) + max_new + 8)

        logits = self.store.core.forward(ids, kv=kv, pos_offset=0)
        kv.len += len(ids)

        out_ids = []
        cur = logits[-1]
        for _ in range(max_new):
            last = cur.astype(np.float64)
            last -= last.max()
            p = np.exp(last / temperature); p /= p.sum()
            nid = int(np.random.choice(len(p), p=p))
            if nid >= vocab_size:
                break
            out_ids.append(nid)
            if nid in eos_ids:
                break
            cur = self.store.core.forward([nid], kv=kv, pos_offset=kv.len)[-1]
            kv.len += 1

        return self._detokenize(out_ids)

    @staticmethod
    def _is_meaningful(text):
        if not text: return False
        text = text.strip()
        if len(text) < 5: return False
        bad = sum(1 for c in text if not (c.isprintable() or c in "\n\t"))
        if bad / len(text) > 0.1: return False
        import re as _re
        if _re.search(r"(.)\1{6,}", text): return False
        if len(text) > 50 and len(set(text)) / len(text) > 0.75: return False
        return True

    def _fallback_log(self, results):
        parts = ["_(ядро не обучено, показываю сырые результаты модулей)_"]
        for k, v in results.items():
            parts.append(f"[{k}]\n{str(v)[:2000]}")
        return "\n\n".join(parts)

    def _synthesize(self, user_input, results, memory_digest):
        context = self._build_context(results)
        prompt = self._build_prompt(user_input, context, memory_digest)
        out = self._generate_str(prompt, max_new=300)
        if self._is_meaningful(out):
            return out.strip()
        if results:
            return self._fallback_log(results)
        return "(ядро не обучено, ответа нет)"

    # ---------- main ----------
    def think(self, user_input):
        self.scheduler.touch()
        task_id = self.logger.record("task", {"input": user_input},
                                     tags=["task"])
        digest = self.logger.recall_digest(user_input, limit=3)
        plan = self._plan(user_input)
        self.logger.record("plan", plan, tags=["plan"], refs=[task_id])

        results = self._run_tools(plan, user_input, task_id)

        if not results and not self.core_ok:
            answer = ("Нет core.nblk и нет tool-результатов. "
                      "Соберите core: python -m nucleus.build_demo_core")
        else:
            answer = self._synthesize(user_input, results, digest)

        self.history.add("user", user_input)
        self.history.add("assistant", answer)

        self.logger.record("response",
                           {"input": user_input, "modules": list(results),
                            "plan": plan, "answer": answer[:1500]},
                           tags=["response"] + list(results), refs=[task_id])
        return answer

    # ---------- idle / memory ----------
    def rotate_memory(self, reason="context_end"):
        meta = self.logger.rotate(reason=reason)
        self.logger.record("rotate_memory", {"reason": reason, "meta": meta},
                           tags=["memory"])
        return meta

    def run_idle_learning(self):
        try:
            created = learning.consolidate(self.logger)
            self.logger.record("idle_learning", {"created": created},
                               tags=["learning"])
            return created
        except Exception as e:
            self.logger.record("idle_learning_error", {"error": str(e)},
                               tags=["learning", "error"])
            return 0

    def scan(self):
        st = self.file_index.scan()
        self.logger.record("file_index_scan", st, tags=["file_index"])
        return st
