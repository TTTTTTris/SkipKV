"""Logged compactions vs. the count simulated from output lengths (docs/skipkv/IMPLEMENTATION.md §9 validity check).

Run in a sweep's output directory: it needs each run's ``--save-outputs`` JSONL, which is not
committed (only the summary JSONs are).
"""
import glob, json, os, sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
T = ("You are given a math problem.\n\nProblem: {question}\n\n You need to solve the problem step by step. "
     "First, you need to provide the chain-of-thought, then provide the final answer.\n\n "
     "Provide the final answer in the format: Final answer:  \\boxed{{}}")
for task in ["math"]:
    data = [json.loads(l) for l in open(f"../../../../data/{task}.jsonl")]
    # tokenize the templated text, as eval.py / vLLM do (apply_chat_template's
    # default return is a BatchEncoding dict in this transformers version)
    plen = [len(tok(tok.apply_chat_template([{"role": "user", "content": T.format(question=d["problem"])}],
                                            add_generation_prompt=True, tokenize=False)).input_ids)
            for d in data]
    print(f"{task}: prompt tokens min {min(plen)} median {sorted(plen)[len(plen) // 2]} max {max(plen)}")
    for f in sorted(glob.glob(f"{task}_*_b*.json")):
        r = json.load(open(f)); B, buf = r["budget"], r["buffer"]
        sim = 0
        for l in open(f[:-5] + ".jsonl"):
            x = json.loads(l); phys = plen[x["idx"]]
            for d in range(1, x["ntok"] + 1):
                phys += 1
                if d % buf == 0 and phys >= B + buf:
                    phys = B; sim += 1
        print(f"{f:32s} logged={r['compactions']:6d} simulated={sim:6d} ratio={r['compactions'] / max(sim, 1):.3f}")
