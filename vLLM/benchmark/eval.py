"""R-KV accuracy + throughput driver for vLLM (offline batched).

Mirrors the R-KV repo's SGLang benchmark driver, but because vLLM's R-KV path is
a drop-in serving change gated by env vars, this driver runs the model **offline**
(``LLM.generate`` over all prompts at once) instead of over HTTP. That keeps the
numeric judging identical to the SGLang harness while measuring vLLM's own
batched decode throughput. R-KV on/off is selected by the
``VLLM_V1_R_KV_BUDGET`` / ``VLLM_V1_R_KV_BUFFER`` env vars (0/unset => Full-KV).

Reported throughput is offline batched (all ``--n`` prompts submitted at once),
not a served-request rate.

Usage:
    # R-KV budget=256 buffer=128, 200 questions, one GPU:
    VLLM_V1_R_KV_BUDGET=256 VLLM_V1_R_KV_BUFFER=128 \
      python eval.py --n 200 --label rkv_b256_buf128 --out /tmp/r.json

    # Full-KV production baseline (prefix caching on, upstream defaults):
    python eval.py --n 200 --label fullkv_production

    # Full-KV constrained (fair A/B) baseline (prefix caching off, like R-KV):
    python eval.py --n 200 --no-prefix --label fullkv_constrained

    # SkipKV sentence eviction on top of R-KV (threshold > 0 enables it):
    VLLM_V1_R_KV_BUDGET=256 VLLM_V1_R_KV_BUFFER=128 VLLM_V1_SKIPKV_SENTENCE_THRESHOLD=0.95 \
      python eval.py --n 200

    # Reasoning model, zero-shot chat prompts (HF GSM8K data):
    RKV_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
      python eval.py --chat --max-tokens 4096 --n 200

    # 4-way tensor parallel:
    VLLM_V1_R_KV_BUDGET=256 VLLM_V1_R_KV_BUFFER=128 \
      python eval.py --n 500 --tp 4 --label rkv_tp4

    # MATH-500 (chat prompts, graded with the SkipKV repo's evaluation/ math_equal):
    RKV_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
      python eval.py --task math --n 500 --max-tokens 32768 --max-model-len 34816
"""

import argparse
import json
import os
import re
import time

# The compaction counter is read from each worker via collective_rpc with a
# small closure; vLLM gates callable RPCs behind this flag. It is safe here --
# a local, offline benchmark run by the same user. Set before importing vllm.
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

from vllm import LLM, SamplingParams

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DATA = os.path.join(_HERE, "data", "gsm8k_fewshot.jsonl")
# Portable default (resolved from the HF hub / local cache). Point RKV_MODEL at a
# local checkout to avoid a download, e.g. RKV_MODEL=/path/to/Qwen2.5-Math-7B-Instruct.
_DEFAULT_MODEL = os.environ.get(
    "RKV_MODEL", "Qwen/Qwen2.5-Math-7B-Instruct"
)


# Zero-shot prompt for --chat (the HuggingFace R-KV/SkipKV eval prompt).
_CHAT_DATA = os.path.join(_HERE, "..", "..", "data", "gsm8k.jsonl")
_CHAT_TEMPLATE = (
    "You are given a math problem.\n\nProblem: {question}\n\n You need to solve "
    "the problem step by step. First, you need to provide the chain-of-thought, "
    "then provide the final answer.\n\n Provide the final answer in the format: "
    "Final answer:  \\boxed{{}}"
)


# The SkipKV repo root (HuggingFace reference): data/ and evaluation/.
_HF_DIR = os.path.join(_HERE, "..", "..")
_TASK_DATA = {
    "math": os.path.join(_HF_DIR, "data", "math.jsonl"),
}


def math_grader():
    """(extract, equal) from the SkipKV evaluation/ (needs latex2sympy2,
    word2number, regex)."""
    import sys

    sys.path.insert(0, os.path.join(_HF_DIR, "evaluation"))
    from grader import math_equal
    from parser import extract_answer, strip_string

    def extract(text):
        return extract_answer(text, "math")

    def equal(pred, gold):
        try:
            return bool(pred) and math_equal(pred, strip_string(gold), timeout=True)
        except Exception:
            return False

    return extract, equal


def extract_gold(answer: str):
    m = re.search(r"####\s*([\-0-9\.,/]+)", answer)
    return m.group(1).replace(",", "").rstrip(".") if m else None


_PATS = [
    r"[Tt]he final answer is[:\s]*\$?\\?\(?\$?([\-0-9\.,/]+)",
    r"[Tt]he answer is[:\s]*\$?([\-0-9\.,/]+)",
]


def _last_boxed(text: str):
    """Content of the last ``\\boxed{...}``, with nested braces (``\\frac{a}{b}``)."""
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    j = start = i + len("\\boxed{")
    depth = 1
    while j < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        j += 1
    return text[start : j - 1] if depth == 0 else None


def _clean_answer(ans: str):
    """LaTeX answer -> plain number string: ``\\$1,\\!596`` -> 1596, ``60\\%`` -> 60,
    ``\\frac{3}{4}`` -> 3/4, ``\\text{...}`` units dropped."""
    a = re.sub(r"\\[dt]?frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", ans)
    a = re.sub(r"\\(text|mathrm|textbf|mbox)\{[^{}]*\}", " ", a)
    a = re.sub(r"\\[!,;: ]|\\?[$%]|\\(left|right)|\^\\circ|\^\{\\circ\}", "", a)
    a = a.replace(",", "").replace("{", "").replace("}", "")
    a = re.sub(r"^\s*[a-zA-Z]\s*=", "", a)  # "x = 5"
    m = re.fullmatch(r"\s*\((-?[0-9.]+)\)/\((-?[0-9.]+)\)\s*", a)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    nums = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?", a)
    return nums[-1] if nums else None


def extract_pred(text: str):
    boxed = _last_boxed(text)
    if boxed is not None:
        return _clean_answer(boxed)
    for pat in _PATS:
        found = re.findall(pat, text)
        if found:
            return found[-1].replace(",", "").replace("$", "").rstrip(".")
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return nums[-1].replace(",", "") if nums else None


def to_num(x):
    if x is None:
        return None
    x = x.strip()
    try:
        if "/" in x:
            a, b = x.split("/")
            return float(a) / float(b)
        return float(x)
    except Exception:
        return None


def read_compactions(llm: LLM, attr: str = "_n_compactions"):
    """Total physical R-KV compactions during the run (max across TP ranks).

    Read from every worker's compactor via ``collective_rpc``; returns ``None``
    when the count is unavailable (e.g. R-KV off). ``attr`` selects another
    compactor counter (``_n_sentence_penalized`` for SkipKV).
    """

    def _get(self):
        mr = getattr(self, "model_runner", None)
        comp = getattr(mr, "rkv_compactor", None)
        return int(getattr(comp, attr, 0)) if comp is not None else 0

    try:
        counts = llm.collective_rpc(_get)
        return max(counts) if counts else 0
    except Exception:
        return None


def read_steering(llm: LLM):
    """SkipKV steering stats from the workers (``None`` when steering is off)."""

    def _get(self):
        ctl = getattr(getattr(self, "model_runner", None), "rkv_steering", None)
        return dict(ctl.stats) if ctl is not None else None

    try:
        stats = llm.collective_rpc(_get)
        return stats[0] if stats else None
    except Exception:  # observability only
        return None


def read_preemptions(llm: LLM):
    """Total scheduler preemptions during the run, or ``None`` without --stats.

    Read from the engine's logging stat logger, which counts PREEMPTED events.
    Each event reaches the front end with the request's next output, so the
    total is only complete once every request has finished (as here).
    """
    mgr = getattr(llm.llm_engine, "logger_manager", None)
    if mgr is None:
        return None
    total = None
    for sl in mgr.stat_loggers:
        inner = getattr(sl, "per_engine_stat_loggers", {sl: sl}).values()
        for lg in inner:
            if hasattr(lg, "total_num_preemptions"):
                total = (total or 0) + lg.total_num_preemptions
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=_DEFAULT_DATA)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--label", default="")
    ap.add_argument("--tp", type=int, default=1, help="tensor_parallel_size")
    ap.add_argument("--mem-frac", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument(
        "--no-prefix",
        action="store_true",
        help="disable prefix caching (fair Full-KV A/B; R-KV forces this itself)",
    )
    ap.add_argument(
        "--eager",
        action="store_true",
        help="force eager (default: cudagraph -- PIECEWISE when R-KV is on)",
    )
    ap.add_argument(
        "--offset",
        type=int,
        default=0,
        help="skip the first OFFSET questions (used to shard across DP replicas)",
    )
    ap.add_argument(
        "--ignore-eos",
        action="store_true",
        help="generate exactly --max-tokens per request (stress test: fixed-length "
        "sequences to load the KV cache regardless of the model's natural stop)",
    )
    ap.add_argument(
        "--stats",
        action="store_true",
        help="enable vLLM engine stat logging (so peak 'Running: N reqs' -- the "
        "achieved concurrency -- is printed) and report the run's total preemptions",
    )
    ap.add_argument(
        "--chat",
        action="store_true",
        help="zero-shot chat-template prompts from --data rows with a 'question' "
        "field (default data: ../../data/gsm8k.jsonl); use for "
        "reasoning models such as DeepSeek-R1-Distill",
    )
    ap.add_argument(
        "--task",
        choices=["gsm8k", "math"],
        default="gsm8k",
        help="math: chat prompts from the SkipKV data/, answers graded "
        "with the SkipKV evaluation/'s math_equal (implies --chat)",
    )
    ap.add_argument(
        "--samples", type=int, default=1, help="samples per question; accuracy is avg@K"
    )
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument(
        "--save-outputs", default="", help="write per-question outputs (jsonl)"
    )
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    budget = int(os.environ.get("VLLM_V1_R_KV_BUDGET", "0"))
    buffer = int(os.environ.get("VLLM_V1_R_KV_BUFFER", "0"))
    rkv_on = budget > 0 and buffer > 0
    sent_thr = float(os.environ.get("VLLM_V1_SKIPKV_SENTENCE_THRESHOLD", "0"))
    skipkv_on = rkv_on and sent_thr > 0
    method = f"skipkv_s{sent_thr:g}" if skipkv_on else "rkv"
    label = args.label or (
        f"{method}_b{budget}_buf{buffer}" if rkv_on else "fullkv"
    )
    if args.task != "gsm8k":
        args.chat = True
        if args.data == _DEFAULT_DATA:
            args.data = _TASK_DATA[args.task]
    if args.chat and args.data == _DEFAULT_DATA:
        args.data = _CHAT_DATA

    data = []
    with open(args.data) as f:
        for line in f:
            data.append(json.loads(line))
    data = data[args.offset : args.offset + args.n]
    n = len(data)
    if not args.chat:
        prompts = [d["request"]["messages"][0]["content"] for d in data]

    kw = {}
    if args.no_prefix:
        kw["enable_prefix_caching"] = False
    llm = LLM(
        model=_DEFAULT_MODEL,
        tensor_parallel_size=args.tp,
        enforce_eager=args.eager,
        gpu_memory_utilization=args.mem_frac,
        max_model_len=args.max_model_len,
        disable_log_stats=not args.stats,
        seed=0,
        block_size=16,
        **kw,
    )
    tok = llm.get_tokenizer()
    if args.chat:
        prompts = [
            tok.apply_chat_template(
                [{"role": "user", "content": _CHAT_TEMPLATE.format(
                    question=d.get("question", d.get("problem")))}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for d in data
        ]
    # --samples K: each prompt is submitted K times (separate requests), and
    # accuracy is the mean over all K * n generations (avg@K).
    k = args.samples
    rows = [(d, p) for d, p in zip(data, prompts) for _ in range(k)]
    in_toks = k * sum(len(tok(p).input_ids) for p in prompts)
    sp = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        # ignore_eos forces fixed-length generation (stress test); otherwise stop
        # at the next few-shot problem boundary for accurate GSM8K judging.
        ignore_eos=args.ignore_eos,
        stop=None if args.ignore_eos or args.chat else ["\nProblem"],
    )

    t0 = time.time()
    outs = llm.generate([p for _, p in rows], sp)
    dt = time.time() - t0

    if args.task != "gsm8k":
        m_extract, m_equal = math_grader()
    correct = 0
    out_toks = 0
    records = []
    for i, ((d, _), o) in enumerate(zip(rows, outs)):
        text = o.outputs[0].text
        ntok = len(o.outputs[0].token_ids)
        out_toks += ntok
        if args.task == "gsm8k":
            g = to_num(extract_gold(d["answer"]))
            p = to_num(extract_pred(text))
            ok = g is not None and p is not None and abs(g - p) < 1e-4
        else:
            g = str(d["answer"])
            p = m_extract(text)
            ok = m_equal(p, g)
        correct += int(ok)
        records.append(
            {"idx": args.offset + i // k, "sample": i % k, "gold": g, "pred": p,
             "ok": ok, "ntok": ntok, "finish": o.outputs[0].finish_reason, "text": text}
        )
        if i < 5:
            print(f"[{i}] gold={g} pred={p} ok={ok} ntok={ntok}")
    n_gen = len(records)

    if args.stats:
        llm.llm_engine.do_log_stats()  # final stats line, incl. total preemptions
    preemptions = read_preemptions(llm)
    steer = read_steering(llm)
    compactions = read_compactions(llm) if rkv_on else 0
    penalized = read_compactions(llm, "_n_sentence_penalized") if skipkv_on else 0
    if skipkv_on:
        def _stats(self):
            sent = getattr(getattr(self.model_runner, "rkv_compactor", None), "sentences", None)
            if sent is None:
                return None
            return {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in sent.stats.items()}

        try:
            print("sentence stats:", llm.collective_rpc(_stats)[0])
        except Exception as e:  # observability only
            print("sentence stats unavailable:", e)
    if args.save_outputs:
        with open(args.save_outputs, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
    res = {
        "label": label,
        "budget": budget,
        "buffer": buffer,
        "n": n,
        "task": args.task,
        "samples": k,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "granularity": os.environ.get("VLLM_V1_SKIPKV_GRANULARITY", "global"),
        "truncated": sum(r["finish"] == "length" for r in records),
        "tp": args.tp,
        "max_tokens": args.max_tokens,
        "accuracy": round(correct / n_gen, 4),
        "correct": correct,
        "wall_s": round(dt, 1),
        "out_tokens": out_toks,
        "in_tokens": in_toks,
        "decode_tok_s": round(out_toks / dt, 1),
        "total_tok_s": round((out_toks + in_toks) / dt, 1),
        "avg_gen_len": round(out_toks / n_gen, 1),
        "compactions": compactions,
        "preemptions": preemptions,
        "steering": {
            "vector": os.environ.get("VLLM_V1_SKIPKV_STEER_VECTOR", ""),
            "layer": int(os.environ.get("VLLM_V1_SKIPKV_STEER_LAYER", "20")),
            "coef": float(os.environ.get("VLLM_V1_SKIPKV_STEER_COEF", "-1.25")),
            "gamma": float(os.environ.get("VLLM_V1_SKIPKV_STEER_GAMMA", "0.02")),
            "refresh": int(os.environ.get("VLLM_V1_SKIPKV_STEER_REFRESH", "128")),
            **steer,
        } if steer else None,
        "sentence_threshold": sent_thr if skipkv_on else 0.0,
        "sentence_penalized_compactions": penalized,
        "model": _DEFAULT_MODEL,
        "chat": args.chat,
    }
    print(
        f"\n=== {label} ===\n"
        f"accuracy    : {correct}/{n_gen} = {correct / n_gen:.3f}\n"
        f"avg_tokens  : {out_toks / n_gen:.0f}\n"
        f"wall_time   : {dt:.1f}s\n"
        f"decode_tput : {out_toks / dt:.1f} tok/s (offline batched, {n_gen} prompts in flight)\n"
        f"preemptions : {preemptions if preemptions is not None else 'n/a (run with --stats)'}\n"
        + (f"steering    : {steer['steered_tokens']} tokens steered, mean coef "
           f"{steer['coef_sum'] / max(steer['steered_tokens'], 1):.3f}\n" if steer else "")
        + f"compactions : {compactions}"
        + (f" ({penalized} with a redundant-sentence penalty)" if skipkv_on else "")
    )
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f)
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
