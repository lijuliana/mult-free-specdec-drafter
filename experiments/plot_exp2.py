"""Plot experiment 2: acceptance against the target as a function of what was replaced.

Left: acceptance (teacher-forced E[min(p, q)], and the gamma-round estimate) for each configuration,
as a horizontal bar chart grouped by replaced set, with the exact and INT4 drafters as reference
lines. Right: accepted tokens per 5-token draft round.

Usage: python experiments/plot_exp2.py out.png results_a.csv [results_b.csv ...]
"""

import csv
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm", "font.size": 9, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
})
COLORS = {"nearest": "#0072B2", "tree": "#E69F00", "hash": "#009E73", "int4": "#555555", "fp16": "#000000"}


def label(r):
    if r["method"] == "fp16":
        return "exact 16-bit drafter"
    s = r["set"] or "all"
    s = s.replace("@", " layers ").replace("-", "–")
    if r["method"] == "int4":
        return f"INT4, {s}"
    seq = " (sequential cal.)" if r["config"].endswith("+seq") else ""
    return f"{r['encoder']} L={r['L']}, {s}{seq}"


def main(out_path, csv_paths):
    rows = []
    for p in csv_paths:
        rows += list(csv.DictReader(open(p)))
    # one row per config name; later files win
    uniq = {}
    for r in rows:
        uniq[r["config"]] = r
    rows = sorted(uniq.values(), key=lambda r: (-float(r["alpha_sample"])))
    n = len(rows)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 0.28 * n + 1.2), constrained_layout=True, sharey=True)
    y = list(range(n))
    for ax, key, xlabel in ((axes[0], "alpha_sample", "acceptance per token (teacher-forced)"),
                            (axes[1], "accepted_per_round", "accepted tokens per 5-token round")):
        vals = [float(r[key]) for r in rows]
        cols = [COLORS.get(r["encoder"] or r["method"], "#888888") for r in rows]
        ax.barh(y, vals, color=cols, height=0.7)
        for r, yy, v in zip(rows, y, vals):
            ax.text(v + (0.01 if key == "alpha_sample" else 0.03), yy, f"{v:.2f}", va="center", fontsize=7)
        for r in rows:
            if r["method"] == "fp16":
                ax.axvline(float(r[key]), color="#000000", linewidth=0.8, linestyle="--")
            if r["config"] == "int4:all":
                ax.axvline(float(r[key]), color="#555555", linewidth=0.8, linestyle=":")
        ax.set_xlabel(xlabel)
        ax.set_xlim(0, (1.0 if key == "alpha_sample" else 2.6))
    axes[0].invert_yaxis()  # shared y: invert once so the best configuration is at the top
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([label(r) for r in rows])
    fig.text(0.01, -0.02, "Dashed: exact drafter. Dotted: INT4 drafter, all layers.", fontsize=7, ha="left")
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    print("wrote", out_path)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
