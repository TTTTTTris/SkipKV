"""Accuracy vs. throughput (relative to Full-KV) for MATH-500 -> acc_vs_tput.png

    python plot_acc_tput.py                                   # all series
    python plot_acc_tput.py --exclude rkv_head --out NAME.png # drop series by METHODS key
"""
import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob(os.path.join(HERE, "*.json"))}
# SkipKV + activation steering runs live in a sibling directory
R.update({os.path.basename(f)[:-5]: json.load(open(f))
          for f in glob.glob(os.path.join(HERE, "..", "math_steer", "*.json"))})

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
# Shape = method family (circle R-KV, square SkipKV); dashed + hollow = per-head;
# color = R-KV gray, SkipKV blue, and orange for any method with activation steering
# (categorical slots 1-3, validated all-pairs).
# Line style = eviction granularity: solid global, dashed + hollow per-head.
GRAY, BLUE, ORANGE, AQUA = "#8f8e89", "#2a78d6", "#eb6834", "#1baf7a"
VIOLET = "#6250d6"  # categorical slot 7 (for --color overrides)
METHODS = [  # key, label, color, marker, per_head
    ("rkv", "R-KV", GRAY, "o", False),
    ("rkv_head", "R-KV per-head", GRAY, "o", True),
    ("skipkv", "SkipKV", BLUE, "s", False),
    ("skipkv_head", "SkipKV per-head", BLUE, "s", True),
    ("skipkv_steer", "SkipKV + steering", ORANGE, "s", False),
    ("skipkv_head_steer", "SkipKV per-head + steering", ORANGE, "s", True),
]
# Five series: identity comes from the legend (color = method, dashed/hollow =
# per-head); points carry only their KV budget.
# Budget-label offsets (dx, dy, ha); default is right of the point.
BUDGET_OFF = {("math", "skipkv_head_steer", 1024): (-7, -3, "right"),
              ("math", "skipkv", 2048): (4, 7, "left"), ("math", "rkv_head", 2048): (-4, 7, "right"),
              ("math", "skipkv", 1024): (-6, -3, "right"),
              ("math", "skipkv_steer", 2048): (-7, -3, "right")}
LABEL_BOX = dict(boxstyle="round,pad=0.12", fc=SURFACE, ec="none", alpha=0.85)
ap = argparse.ArgumentParser()
ap.add_argument("--exclude", nargs="*", default=[], help="METHODS keys to leave out")
ap.add_argument("--only", nargs="*", default=[], help="plot only these METHODS keys (plus Full-KV)")
ap.add_argument("--out", default="acc_vs_tput.png")
ap.add_argument("--steer-color", choices=["orange", "base"], default="orange",
                help="steering series in orange, or in their base method's color (for steering-only plots)")
ap.add_argument("--color", nargs="*", default=[], metavar="KEY=NAME",
                help="per-series color override, NAME one of gray/blue/orange/aqua/violet")
ap.add_argument("--legend", choices=["top", "lower-left"], default="top",
                help="legend above the panels, or inside the first panel's lower-left corner")
ap.add_argument("--steer-coef", choices=["-1.25", "-1.0", "best"], default="-1.25",
                help="steering runs to plot: -1.25 (*_steer), -1.0 (*_steer_c1), or the "
                "more accurate of the two per point (ties: faster)")
args = ap.parse_args()
ALL_METHODS = list(METHODS)
METHODS = [m for m in METHODS if m[0] not in args.exclude and (not args.only or m[0] in args.only)]
if args.steer_coef == "-1.0":  # plot the coefficient -1.0 runs under the same series
    METHODS = [(k + "_c1" if k.endswith("_steer") else k, *rest) for k, *rest in METHODS]
TASKS = [
    ("math", "MATH-500 (greedy)", [512, 1024, 2048]),
]
single = len(TASKS) == 1

plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2})
fig, axes = plt.subplots(1, len(TASKS), figsize=(7.2, 5.6) if single else (11, 4.8), facecolor=SURFACE,
                         squeeze=False)
axes = axes[0]
for ax, (task, title, budgets) in zip(axes, TASKS):
    ax.set_facecolor(SURFACE)
    full = R[f"{task}_fullkv"]
    def run(key, b):
        """Result for a series point; with --steer-coef best, the better of the
        two steering coefficients (accuracy, then speed)."""
        cands = [f"{task}_{key}_b{b}"]
        if args.steer_coef == "best" and key.endswith("_steer"):
            cands.append(f"{task}_{key}_c1_b{b}")
        cands = [R[c] for c in cands if c in R]
        return max(cands, key=lambda r: (r["accuracy"], -r["wall_s"])) if cands else None

    for key, name, color, marker, per_head in METHODS:
        color = {k: dict(gray=GRAY, blue=BLUE, orange=ORANGE, aqua=AQUA, violet=VIOLET)[v]
                 for k, v in (o.split("=") for o in args.color)}.get(key.removesuffix("_c1"), color)
        if args.steer_color == "base" and key.removesuffix("_c1").endswith("_steer"):
            color = next(c for k, _, c, _, _ in ALL_METHODS if k == key.removesuffix("_c1").removesuffix("_steer"))
        pts = [(full["wall_s"] / r["wall_s"], 100 * r["accuracy"], b)
               for b in budgets if (r := run(key, b)) is not None]  # finished runs only
        if not pts:
            continue
        xs, ys, _ = zip(*pts)
        ax.plot(xs, ys, color=color, lw=2, zorder=2, ls=(0, (5, 2.5)) if per_head else "-")
        ax.plot(xs, ys, ls="none", marker=marker, ms=8, zorder=3, label=name,
                mfc=SURFACE if per_head else color, mec=color, mew=2 if per_head else 1.5)
        for x, y, b in pts:  # budget labels, in text ink
            dx, dy, ha = BUDGET_OFF.get((task, key.removesuffix("_c1"), b), (6, -3, "left"))
            ax.annotate(str(b), (x, y), xytext=(dx, dy), textcoords="offset points",
                        fontsize=7.5, color=INK2, ha=ha, bbox=LABEL_BOX, zorder=5)
    fa = 100 * full["accuracy"]
    ax.axhline(fa, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.scatter([1.0], [fa], s=150, marker="*", color=INK, edgecolors=SURFACE,
               linewidths=1.5, zorder=4, label="Full-KV")
    ax.annotate(f"Full-KV {fa:.1f}%", (0.78, fa), xytext=(0, -5), textcoords="offset points",
                fontsize=8.5, color=INK, ha="left", va="top")
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_title("better ↗  (more accurate, faster)", loc="right", fontsize=9, color=INK2)
    ax.set_xlabel("Throughput vs Full-KV (×, same requests; Full-KV wall ÷ run wall)")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    xs_all = [x for ln in ax.lines for x in ln.get_xdata()
              if isinstance(x, (int, float)) or hasattr(x, "real")]
    ax.set_xlim(0.75, max(2.95, 1.06 * max(xs_all)))
    ax.margins(y=0.08)

def clear_legend(ax, leg):
    """Lower the y-axis until no series passes under the in-axes legend box."""
    import numpy as np

    fig.canvas.draw()
    ab = leg.get_window_extent().transformed(ax.transAxes.inverted())  # legend box, axes fraction
    x0, x1 = ax.transData.inverted().transform(ax.transAxes.transform([[ab.x0, 0], [ab.x1, 0]]))[:, 0]
    xs = np.linspace(x0, x1, 200)
    series = [ln for ln in ax.lines if ln.get_transform().contains_branch(ax.transData)]  # not the axhline
    ys = [np.interp(xs, *zip(*sorted(zip(ln.get_xdata(), ln.get_ydata()))), left=np.nan, right=np.nan)
          for ln in series if len(ln.get_xdata()) > 1]
    ys += [np.array([y]) for ln in series if len(ln.get_xdata()) == 1
           for x, y in zip(ln.get_xdata(), ln.get_ydata()) if x0 <= x <= x1]
    ys = np.concatenate(ys) if ys else np.array([])
    ymin = ys[np.isfinite(ys)].min() if np.isfinite(ys).any() else np.inf
    y0, y1 = ax.get_ylim()
    top = ab.y1 + 0.06  # legend top + a gap, in axes fraction
    if y0 + top * (y1 - y0) > ymin:
        ax.set_ylim((ymin - top * y1) / (1 - top), y1)


h, l = [], []
for a in axes:  # union of series
    for x, y in zip(*a.get_legend_handles_labels()):
        if y not in l:
            h.append(x), l.append(y)
if args.legend == "lower-left":
    leg = axes[0].legend(h, l, loc="lower left", fontsize=8, frameon=True, framealpha=0.92,
                         borderpad=0.5, handletextpad=0.5, ncol=2 if len(l) > 7 else 1,
                         columnspacing=1.0)
    leg.get_frame().set_facecolor(SURFACE)
    leg.get_frame().set_edgecolor(GRID)
else:
    fig.legend(h, l, frameon=False, loc="upper left", bbox_to_anchor=(0.01, 0.95 if not single else 0.935),
               ncol=3 if single else 5, fontsize=8.5)
title = ("DeepSeek-R1-Distill-Qwen-7B on vLLM, buffer 128, max 32K tokens, 1× RTX 6000 Ada (48 GB) per run"
         "  -  point labels = KV budget")
fig.suptitle(title.replace(", 1×", ",\n1×") if single else title, fontsize=9, color=INK2, x=0.01, ha="left")
steered = any(m[0].removesuffix("_c1").endswith("_steer") for m in METHODS)
note = ("Full-KV baseline ran on vLLM's V2 runner with async scheduling; all compressed runs used the "
        "V1 runner with synchronous scheduling (R-KV and steering require it), so speedups vs Full-KV are "
        "not like-for-like."
        + ((" Steering" + (" (orange)" if args.steer_color == "orange" and not args.color else "") + ": layer 20, gamma 0.02, coef "
            + ("per point = better of -1.25 / -1.0 (chosen on the test set)."
               if args.steer_coef == "best" else f"{args.steer_coef}.")) if steered else ""))
import textwrap  # noqa: E402

fig.text(0.01, 0.012, textwrap.fill(note, 118 if single else 175), fontsize=7.5 if single else 8,
         color=INK2, ha="left", va="bottom")
rows = -(-len(l) // (3 if single else 5)) if args.legend == "top" else 0  # legend rows above the panels
top = (0.875 + 0.04 * (4 - rows) if single else 0.86) if rows else (0.95 if single else 0.94)
fig.tight_layout(rect=(0, 0.075 if single else 0.05, 1, top))
if args.legend == "lower-left":
    for _ in range(2):  # after layout: the legend's size relative to the panel is final
        clear_legend(axes[0], leg)
        fig.tight_layout(rect=(0, 0.075 if single else 0.05, 1, top))
out = os.path.join(HERE, args.out)
fig.savefig(out, dpi=160, facecolor=SURFACE)
print(out)
