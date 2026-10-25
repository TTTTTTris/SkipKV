"""Steering strength: no steering vs coef -1.0 vs -1.25, per method -> steer_strength.png

Accuracy vs throughput (relative to Full-KV), one panel per (task, method). Unsteered runs come
from the first sweep (../math), steered ones from this directory. Missing runs are skipped.
"""
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = {}
for d in (os.path.join(HERE, "..", "math"), HERE):
    for f in glob.glob(os.path.join(d, "*.json")):
        R[os.path.basename(f)[:-5]] = json.load(open(f))

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
GRAY, ORANGE = "#8f8e89", "#eb6834"
METHODS = [("skipkv", "SkipKV", "s"), ("skipkv_head", "SkipKV per-head", "s")]
STRENGTHS = [  # suffix, label, color, filled, linestyle
    ("", "no steering", GRAY, True, "-"),
    ("_steer_c1", "steering, coef −1.0", ORANGE, False, (0, (5, 2.5))),
    ("_steer", "steering, coef −1.25", ORANGE, True, "-"),
]
TASKS = [("math", "MATH-500", [512, 1024, 2048])]
LABEL_BOX = dict(boxstyle="round,pad=0.12", fc=SURFACE, ec="none", alpha=0.85)

plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2})
fig, grid = plt.subplots(len(TASKS), len(METHODS), figsize=(4.5 * len(METHODS) + 1, 4 * len(TASKS) + 1),
                        facecolor=SURFACE, sharey="row", squeeze=False)
for r, (task, tname, budgets) in enumerate(TASKS):
    full = R[f"{task}_fullkv"]
    for c, (mkey, mname, marker) in enumerate(METHODS):
        ax = grid[r][c]
        ax.set_facecolor(SURFACE)
        for suffix, slabel, color, filled, ls in STRENGTHS:
            pts = [(full["wall_s"] / R[k]["wall_s"], 100 * R[k]["accuracy"], b)
                   for b in budgets if (k := f"{task}_{mkey}{suffix}_b{b}") in R]
            if not pts:
                continue
            xs, ys, _ = zip(*pts)
            ax.plot(xs, ys, color=color, lw=2, ls=ls, zorder=2)
            ax.plot(xs, ys, ls="none", marker=marker, ms=8, zorder=3, label=slabel,
                    mfc=color if filled else SURFACE, mec=color, mew=1.5 if filled else 2)
            for x, y, b in pts:
                ax.annotate(str(b), (x, y), xytext=(6, -3), textcoords="offset points",
                            fontsize=7.5, color=INK2, bbox=LABEL_BOX, zorder=5)
        fa = 100 * full["accuracy"]
        ax.axhline(fa, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.annotate(f"Full-KV {fa:.1f}%", (0, fa), xycoords=("axes fraction", "data"),
                    xytext=(4, -5), textcoords="offset points", fontsize=8, color=INK, va="top")
        ax.set_title(f"{tname} — {mname}", color=INK, fontsize=10.5, loc="left")
        ax.set_title("better ↗", loc="right", fontsize=9, color=INK2)
        if r == len(TASKS) - 1:
            ax.set_xlabel("Throughput vs Full-KV (×)")
        if c == 0:
            ax.set_ylabel("Accuracy (%)")
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.margins(x=0.15, y=0.08)
h, l = grid[0][0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, loc="upper left", bbox_to_anchor=(0.01, 0.965), ncol=3, fontsize=9)
fig.suptitle("Steering strength — DeepSeek-R1-Distill-Qwen-7B on vLLM, buffer 128, max 32K tokens; "
             "layer 20, gamma 0.02; point labels = KV budget", fontsize=9, color=INK2, x=0.01, y=0.995,
             ha="left")
fig.text(0.01, 0.008, "Full-KV baseline ran on vLLM's V2 runner with async scheduling; all compressed runs "
         "used V1 with synchronous scheduling,\nso speedups vs Full-KV are not like-for-like. One run per point.",
         fontsize=8, color=INK2, ha="left", va="bottom")
fig.tight_layout(rect=(0, 0.045, 1, 0.94))
out = os.path.join(HERE, "steer_strength.png")
fig.savefig(out, dpi=160, facecolor=SURFACE)
print(out)
