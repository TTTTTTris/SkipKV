"""Tables in docs/skipkv/IMPLEMENTATION.md §10 format: accuracy / wall-time speedup vs Full-KV (+ avg tokens, truncated)."""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
R = {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob(os.path.join(HERE, "*.json"))}
NAMES = [("rkv", "R-KV global"), ("rkv_head", "R-KV per-head"), ("skipkv", "SkipKV global"),
         ("skipkv_head", "SkipKV per-head")]

for task, title, budgets in [("math", "MATH-500, greedy", [512, 1024, 2048])]:
    full = R.get(f"{task}_fullkv")
    head = f"**{title}, buffer 128**"
    if full:
        head += (f" (Full-KV {100 * full['accuracy']:.1f}% / 1.00×, {full['wall_s']:.0f} s, "
                 f"avg {full['avg_gen_len']:.0f} tok, {full['truncated']} truncated)")
    print(head + "\n")
    print("| Method | " + " | ".join(f"budget {b}" for b in budgets) + " |")
    print("| --- |" + " --- |" * len(budgets))
    for key, name in NAMES:
        cells = []
        for b in budgets:
            r = R.get(f"{task}_{key}_b{b}")
            if not r:
                cells.append("—")
                continue
            sp = f" / {full['wall_s'] / r['wall_s']:.2f}×" if full else ""
            cells.append(f"{100 * r['accuracy']:.1f}%{sp} ({r['avg_gen_len'] / 1000:.1f}K tok)")
        if any(c != "—" for c in cells):
            print(f"| {name} | " + " | ".join(cells) + " |")
    print()
