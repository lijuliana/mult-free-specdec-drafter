"""Plot experiment 1: KL to the exact model (and top-1 agreement) against subvector length L,
one line per encoder, with the INT8 and INT4 weight baselines as horizontal references.

Usage: python experiments/plot_exp1.py results/exp1/results.csv figures/exp1_kl.pdf
"""

import csv
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.2,
})

# Okabe-Ito, muted
COLORS = {"nearest": "#0072B2", "tree": "#E69F00", "hash": "#009E73"}
MARKERS = {"nearest": "o", "tree": "s", "hash": "^"}
LABELS = {"nearest": "Nearest prototype (ceiling)", "tree": "MADDNESS tree", "hash": "Random-hyperplane hash"}


def main(csv_path, out_path, metric="kl"):
    rows = list(csv.DictReader(open(csv_path)))
    targets = sorted({r["target"] for r in rows})
    fig, axes = plt.subplots(1, len(targets), figsize=(3.25 * len(targets), 2.6), constrained_layout=True, sharey=False)
    if len(targets) == 1:
        axes = [axes]
    for ax, target in zip(axes, targets):
        sub = [r for r in rows if r["target"] == target]
        for enc in ("nearest", "tree", "hash"):
            labeled = False
            for smooth, ls in ((1, "-"), (0, ":")):
                pts = sorted(
                    (int(r["L"]), float(r[metric]))
                    for r in sub
                    if r["method"] == "lut" and r["encoder"] == enc and int(r["smooth"]) == smooth and int(r["ridge"]) == 1
                )
                if not pts:
                    continue
                ax.plot([p[0] for p in pts], [p[1] for p in pts], ls, marker=MARKERS[enc], color=COLORS[enc],
                        markersize=4, label=None if labeled else LABELS[enc])
                labeled = True
        for method, color, name in (("int4", "#555555", "INT4 weights"), ("int8", "#AAAAAA", "INT8 weights")):
            v = [float(r[metric]) for r in sub if r["method"] == method]
            if v:
                ax.axhline(v[0], color=color, linewidth=0.8, linestyle="--")
                ax.text(ax.get_xlim()[1] if False else 33, v[0], name, va="bottom", ha="right", fontsize=7, color=color)
        ax.set_xscale("log", base=2)
        ax.set_xticks([8, 16, 32])
        ax.set_xticklabels(["8", "16", "32"])
        ax.set_xlabel("Subvector length $L$ (bytes read per token $\\propto 1/L$)")
        ax.set_ylabel("KL to exact model (nats)" if metric == "kl" else "Top-1 agreement with exact model")
        if metric == "kl":
            ax.set_yscale("log")
        ax.set_title(target.replace("mlp.down_proj", "MLP down-projection").replace("self_attn.o_proj", "Attention output projection"),
                     fontsize=9, loc="left")
    axes[0].legend(frameon=False, loc="best")
    fig.text(0.01, -0.04, "Solid: per-channel outlier scaling on. Dotted: off.", fontsize=7, ha="left")
    fig.savefig(out_path, bbox_inches="tight")
    print("wrote", out_path)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "kl")
