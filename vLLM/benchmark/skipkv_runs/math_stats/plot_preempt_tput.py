"""Throughput (relative to Full-KV) vs. preemptions per request (left) and vs. average
running requests (right), from the --stats runs -> preempt_vs_tput.png

``avg_running`` in each result JSON is the mean "Running: N reqs" over the run's vLLM stat-log
intervals with any request queued or running. Missing points are skipped.
"""
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
# --stats runs: this sweep, plus SkipKV + steering (also run with --stats)
R = {}
for d in (HERE, os.path.join(HERE, "..", "math_steer")):
    for f in glob.glob(os.path.join(d, "*.json")):
        R[os.path.basename(f)[:-5]] = json.load(open(f))

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
GRAY, BLUE, ORANGE = "#8f8e89", "#2a78d6", "#eb6834"
METHODS = [  # key, label, color, marker, per_head  (same encoding as acc_vs_tput.png)
    ("rkv", "R-KV", GRAY, "o", False),
    ("rkv_head", "R-KV per-head", GRAY, "o", True),
    ("skipkv", "SkipKV", BLUE, "s", False),
    ("skipkv_head", "SkipKV per-head", BLUE, "s", True),
    ("skipkv_steer", "SkipKV + steering", ORANGE, "s", False),
    ("skipkv_head_steer", "SkipKV per-head + steering", ORANGE, "s", True),
]
TASKS = [
    ("math", "MATH-500 (greedy)", [1024, 2048]),
]
LABEL_BOX = dict(boxstyle="round,pad=0.12", fc=SURFACE, ec="none", alpha=0.85)


def n_req(r):
    return r["n"] * r.get("samples", 1)


plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2})
ROWS = [  # x metric, axis label, which direction is better
    (lambda r: r["preemptions"] / n_req(r), "Preemptions per request (total preemptions ÷ requests)",
     "better ↖  (faster, fewer preemptions)"),
    (lambda r: r["avg_running"], "Average running requests (from stats log)",
     "better ↗  (faster, more requests in flight)"),
]
fig, grid = plt.subplots(1, 2, figsize=(11, 5.4), facecolor=SURFACE, squeeze=False)


def spread_labels(ax, labels, min_gap=11):
    """Annotate budget labels right of their points, nudging stacked ones apart (display px)."""
    disp = [(ax.transData.transform((x, y)), t) for x, y, t in labels]
    disp.sort(key=lambda d: (round(d[0][0] / 40), d[0][1]))
    placed = []
    for (px, py), t in disp:
        ly = py
        for qx, qy in placed:
            if abs(qx - px) < 40 and abs(qy - ly) < min_gap:
                ly = qy + min_gap
        placed.append((px, ly))
        x, y = ax.transData.inverted().transform((px, py))
        k = 72 / ax.figure.dpi  # display px -> points, so offsets survive savefig's dpi
        ax.annotate(t, (x, y), xytext=(7, (ly - py) * k), textcoords="offset points",
                    fontsize=7.5, color=INK2, bbox=LABEL_BOX, zorder=5, va="center")


for row, (xfun, xlabel, better) in enumerate(ROWS):
    for col, (task, title, budgets) in enumerate(TASKS):
        ax = grid[0][row]  # one panel per metric (left: preemptions, right: running requests)
        ax.set_facecolor(SURFACE)
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        ax.set_title(better, loc="right", fontsize=9, color=INK2)
        full = R.get(f"{task}_fullkv")
        all_x, labels = [0.0], []
        for key, name, color, marker, per_head in METHODS:
            pts = []
            for b in budgets:
                r = R.get(f"{task}_{key}_b{b}")
                if r and full and r.get("preemptions") is not None and r.get("avg_running"):
                    pts.append((xfun(r), full["wall_s"] / r["wall_s"], b))
            if not pts:
                continue
            xs, ys, _ = zip(*pts)
            all_x += xs
            ax.plot(xs, ys, color=color, lw=2, zorder=2, ls=(0, (5, 2.5)) if per_head else "-")
            ax.plot(xs, ys, ls="none", marker=marker, ms=8, zorder=3, label=name,
                    mfc=SURFACE if per_head else color, mec=color, mew=2 if per_head else 1.5)
            labels += [(x, y, str(b)) for x, y, b in pts]
        if full and full.get("preemptions") is not None and full.get("avg_running"):
            fx = xfun(full)
            all_x.append(fx)
            ax.scatter([fx], [1.0], s=150, marker="*", color=INK, edgecolors=SURFACE,
                       linewidths=1.5, zorder=4, label="Full-KV")
            ax.annotate("Full-KV", (fx, 1.0), xytext=(0, 9), textcoords="offset points",
                        fontsize=8.5, color=INK, ha="center")
        ax.axhline(1.0, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Throughput vs Full-KV (×)")
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        xmax = max(max(all_x), 0.01)
        ax.set_xlim(-0.04 * xmax, 1.15 * xmax)
        ax.margins(y=0.12)
        fig.canvas.draw()
        spread_labels(ax, labels)
axes = [a for r in grid for a in r]
seen = {}
for ax in axes:
    for h, l in zip(*ax.get_legend_handles_labels()):
        seen.setdefault(l, h)
fig.legend(list(seen.values()), list(seen.keys()), frameon=False, loc="upper left",
           bbox_to_anchor=(0.01, 0.968), ncol=5, fontsize=8.5)
fig.suptitle("DeepSeek-R1-Distill-Qwen-7B on vLLM with --stats, buffer 128, max 32K tokens, 1× RTX 6000 Ada (48 GB) "
             "per run  -  point labels = KV budget", fontsize=9, color=INK2, x=0.01, y=0.995, ha="left")
fig.text(0.01, 0.008, "Full-KV baseline ran on vLLM's V2 runner with async scheduling; all compressed runs used the "
         "V1 runner with synchronous scheduling (R-KV and steering require it),\nso speedups vs Full-KV are not "
         "like-for-like. Steering (orange): layer 20, coef -1.25, gamma 0.02. MATH budget 512 is not in "
         "this sweep.", fontsize=8, color=INK2, ha="left", va="bottom")
fig.tight_layout(rect=(0, 0.06, 1, 0.87))
out = os.path.join(HERE, "preempt_vs_tput.png")
fig.savefig(out, dpi=160, facecolor=SURFACE)
print(out, f"({sum(1 for k in R if k.startswith('math'))} runs plotted from)")
