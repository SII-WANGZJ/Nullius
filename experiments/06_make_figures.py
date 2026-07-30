#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_figures.py -- render the audit figures for paper/audit.tex.

Reads results/audit_results.json and results/robustness_results.json;
writes vector PDFs into ../paper/figs/.  No numbers are hard-coded here:
every value plotted is read back from the result files, so the figures
cannot drift from the computation.
"""

from __future__ import annotations

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import _path  # noqa: F401  (puts src/ on sys.path)
import nullius as N
# --- palette (validated: all-pairs CVD dE 24.7, normal-vision 33.6) ----------
BLUE = "#2a78d6"      # categorical slot 1 -- "as released"
ORANGE = "#eb6834"    # categorical slot 2 -- audit modification
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

FIG_DIR = N.FIG_DIR

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 8,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
})


def load(name):
    with open(os.path.join(N.RESULTS_DIR, name), encoding="utf-8") as fh:
        return json.load(fh)


def style_axes(ax, xgrid=True):
    ax.set_axisbelow(True)
    ax.grid(axis="x" if xgrid else "y", color=GRID, lw=0.6)
    ax.tick_params(length=0)


# ---------------------------------------------------------------------------
def fig1_taskD(audit):
    """Task D: as released vs self-pairs removed, against the identity rule."""
    b25 = audit["B2_selfpair_bin25"]
    b10 = audit["B2_selfpair_bin10"]
    ident = b25["trivial_rule_with_self_pairs"]

    rows = [
        ("Complex-B  (optical, bin 25)",      b25["rows"]["optical:complex_B"]),
        ("Complex-B  (optical, bin 10)",      b10["rows"]["optical:complex_B"]),
        ("Digital bilinear  (intensity)",     b25["rows"]["optical:digital_bilinear"]),
        (r"Raw $\mathrm{conj}\cdot\mathrm{prod}$  (no optics)",
                                              b25["rows"]["raw:conj_prod_raw"]),
        (r"Raw $|\mathrm{diff}|$  (no optics)", b25["rows"]["raw:abs_diff_raw"]),
    ]
    labels = [r[0] for r in rows]
    rel = [r[1]["D_released"] for r in rows]
    abl = [r[1]["D_no_self_pairs"] for r in rows]

    y = np.arange(len(rows))[::-1]
    h = 0.34
    COL_R, COL_A = 0.90, 1.03          # value columns, clear of every bar
    fig, ax = plt.subplots(figsize=(6.9, 3.0))

    ax.barh(y + h / 2 + 0.01, rel, height=h, color=BLUE,
            label="as released", zorder=3)
    ax.barh(y - h / 2 - 0.01, abl, height=h, color=ORANGE,
            label="self-pairs removed", zorder=3)

    ax.axvline(0.5, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax.axvline(ident, color=INK2, lw=1.2, ls=(0, (5, 2)), zorder=2)
    top = len(rows) - 0.28                 # line annotations
    hdr = len(rows) + 0.36                 # column headers, one line higher
    ax.text(0.49, top, "chance", color=MUTED, ha="right", va="bottom", fontsize=7)
    ax.text(ident + 0.012, top,
            "identity rule $(x{=}y)$ $=$ " + f"{ident:.3f}",
            color=INK2, ha="left", va="bottom", fontsize=7)

    # values live in their own columns, so nothing collides with the rules
    ax.text(COL_R, hdr, "released", color=MUTED, ha="right", va="bottom", fontsize=6.5)
    ax.text(COL_A, hdr, "ablated", color=MUTED, ha="right", va="bottom", fontsize=6.5)
    for yy, v in zip(y, rel):
        ax.text(COL_R, yy, f"{v:.3f}", va="center", ha="right",
                fontsize=7.5, color=INK2)
    for yy, v in zip(y, abl):
        ax.text(COL_A, yy, f"{v:.3f}", va="center", ha="right",
                fontsize=7.5, color=INK2)

    ax.set_yticks(y, labels, color=INK2, fontsize=8)
    ax.set_xticks(np.arange(0, 0.9, 0.2))
    ax.set_xlim(0, 1.05)
    ax.set_ylim(-0.60, len(rows) + 0.85)
    ax.set_xlabel("balanced accuracy, Task D (cross-split same category)")
    style_axes(ax)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.38, -0.20),
              ncol=2, fontsize=7.5, labelcolor=INK2, handlelength=1.2)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig1_taskD_selfpair.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# ---------------------------------------------------------------------------
def fig2_leakage(audit, robust):
    """(a) Tasks B/C released vs pair-grouped;  (b) Task C fold-count sweep."""
    lk = audit["B3_leakage_bin25"]
    feats = [("complex_B", "Complex-B"),
             ("digital_bilinear", "Digital bilinear"),
             ("concat", "Concat")]

    fig, (axB, axC, axS) = plt.subplots(
        1, 3, figsize=(6.9, 2.5),
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.15], "wspace": 0.45})

    for ax, task, chance, title in (
            (axB, "B", 0.5, "Task B  (binary)"),
            (axC, "C", 1 / 16, "Task C  (16-way)")):
        x = np.arange(len(feats))
        rel = [lk[f"{k}:{task}"]["released_skf"] for k, _ in feats]
        grp = [lk[f"{k}:{task}"]["grouped_sgkf"] for k, _ in feats]
        w = 0.34
        ax.bar(x - w / 2 - 0.01, rel, width=w, color=BLUE, zorder=3)
        ax.bar(x + w / 2 + 0.01, grp, width=w, color=ORANGE, zorder=3)
        ax.axhline(chance, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=2)
        for xx, v in zip(x - w / 2 - 0.01, rel):
            ax.text(xx, v + 0.03, f"{v:.2f}", ha="center", fontsize=6.5, color=INK2)
        for xx, v in zip(x + w / 2 + 0.01, grp):
            ax.text(xx, v + 0.03, f"{v:.2f}", ha="center", fontsize=6.5, color=INK2)
        ax.set_xticks(x, [n for _, n in feats], fontsize=7, color=INK2,
                      rotation=18, ha="right")
        ax.set_xlim(-0.55, len(feats) - 0.45)
        ax.set_ylim(0, 1.28)
        ax.set_title(title, fontsize=8, color=INK, pad=6)
        style_axes(ax, xgrid=False)
    axB.set_ylabel("balanced accuracy")

    # (b) fold-count sweep for Task C
    for bin_size, colour, marker in ((25, BLUE, "o"), (10, ORANGE, "s")):
        sweep = robust[f"Q1_taskC_sweep_bin{bin_size}"]
        ns = sorted(int(k) for k in sweep)
        vals = [sweep[str(n)]["balanced_acc"] for n in ns]
        axS.plot(ns, vals, color=colour, lw=2, marker=marker, ms=5,
                 mec=SURFACE, mew=1.0, zorder=3, label=f"bin {bin_size}")
    axS.axhline(1 / 16, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=2)
    axS.set_xticks([2, 4, 5, 8], ["2", "4", "5", "8"], fontsize=7, color=INK2)
    axS.set_xlabel("number of pair-grouped folds", fontsize=7.5)
    axS.set_ylim(0, 0.30)
    axS.set_title("Task C, fold-count sweep", fontsize=8, color=INK, pad=6)
    axS.legend(frameon=False, fontsize=7, labelcolor=INK2,
               handlelength=1.4, loc="upper right")
    style_axes(axS, xgrid=False)

    handles = [Patch(facecolor=BLUE, label="released scoring (repeats split)"),
               Patch(facecolor=ORANGE, label="pair-grouped scoring"),
               Line2D([0], [0], color=MUTED, lw=1.0, ls=(0, (4, 3)),
                      label="chance level (0.500 / 0.063)")]
    fig.legend(handles=handles, frameon=False, fontsize=7.5, labelcolor=INK2,
               ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.16),
               handlelength=1.6)

    out = os.path.join(FIG_DIR, "fig2_leakage.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# ---------------------------------------------------------------------------
def fig3_repeat_correlation():
    """Distribution of pairwise repeat correlations of the Complex-B feature."""
    single, blank, pair = N.load_all_data_binned(25)
    B_per_rep = N.compute_B_per_repeat(blank, pair)
    cors = []
    for arr in B_per_rep.values():
        v = np.concatenate([arr.real, arr.imag], axis=1)
        for i in range(v.shape[0]):
            for j in range(i + 1, v.shape[0]):
                cors.append(np.corrcoef(v[i], v[j])[0, 1])
    cors = np.array(cors)

    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    ax.hist(cors, bins=48, color=BLUE, zorder=3)
    ax.axvline(0.97, color=INK2, lw=1.2, ls=(0, (5, 2)), zorder=4)
    ax.text(0.968, ax.get_ylim()[1] * 0.94, "0.97 ", color=INK2,
            fontsize=7, ha="right", va="top")
    ax.set_xlabel("pairwise correlation between physical repeats", fontsize=7.5)
    ax.set_ylabel("count", fontsize=7.5)
    ax.set_title(f"mean {cors.mean():.4f} · {100*(cors>0.97).mean():.1f}% above 0.97",
                 fontsize=7.5, color=INK, pad=5)
    style_axes(ax, xgrid=False)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig3_repeat_correlation.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def fig4_permutation_null(struct):
    """Exact null over all 105 admissible category assignments."""
    s = struct["S2"]
    scores = np.array(s["all_scores"])
    obs = s["observed"]

    fig, ax = plt.subplots(figsize=(5.4, 2.6))
    xs = np.sort(scores)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    ax.step(xs, ys, where="post", color=BLUE, lw=2, zorder=3,
            label=f"all {s['n_partitions']} admissible assignments")
    ax.plot(scores, np.full_like(scores, -0.045), "|", color=BLUE,
            ms=5, mew=0.9, alpha=0.55, zorder=3, clip_on=False)

    ax.axvline(obs, color=ORANGE, lw=2, zorder=4)
    ax.text(obs + 0.004, 0.60, f"designated semantic\nassignment  {obs:.4f}",
            color=ORANGE, fontsize=7, ha="left", va="center", linespacing=1.3)
    ax.axvline(s["null_mean"], color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax.text(s["null_mean"] + 0.003, 0.06, f"null mean {s['null_mean']:.4f}",
            color=MUTED, fontsize=6.5, ha="left")

    ax.set_xlabel("pair-grouped balanced accuracy, Task B", fontsize=7.5)
    ax.set_ylabel("cumulative fraction", fontsize=7.5)
    ax.set_ylim(-0.09, 1.04)
    ax.set_title(f"exact one-sided $p$ = {s['p_exact_fraction']} = {s['p_exact']:.4f}",
                 fontsize=8, color=INK, pad=6)
    style_axes(ax, xgrid=False)
    ax.legend(frameon=False, fontsize=7, labelcolor=INK2, loc="upper left",
              handlelength=1.4)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig4_permutation_null.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def fig5_xor(x):
    """XOR Task B across splitting regimes with different generalisation units."""
    AQUA = "#1baf7a"                     # categorical slot 3
    regimes = [
        ("sample\nlevel", "sample_level", None),
        ("ordered pair\n(K-fold)", "ordered_pair_grouped", None),
        ("unordered pair\n(K-fold)", "unordered_pair_grouped", None),
        ("ordered pair\n(exhaustive)", None, "leave_one_ordered_pair_out"),
        ("unordered group\n(exhaustive)", None, "leave_one_unordered_group_out"),
        ("leave one\ntoken out", "leave_one_token_out", None),
    ]
    fams = [("complex_B", "Complex-B", BLUE, "o"),
            ("digital_bilinear", "Digital bilinear", ORANGE, "s"),
            ("concat", "Concat", AQUA, "^")]

    tb, ex = x["X1_X3_X8_taskB"], x["X9_X10_exhaustive"]
    xs = np.arange(len(regimes))
    fig, ax = plt.subplots(figsize=(6.9, 3.0))
    ax.axhline(0.5, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax.text(len(regimes) - 0.52, 0.508, "chance", color=MUTED, fontsize=6.5,
            ha="right", va="bottom")

    for key, label, col, mk in fams:
        ys = []
        for _, k1, k2 in regimes:
            v = tb[key][k1] if k1 else ex[key][k2]
            ys.append(np.nan if v is None else v)
        ys = np.array(ys, dtype=float)
        ax.plot(xs, ys, color=col, lw=1.2, alpha=0.45, zorder=3)
        ax.plot(xs, ys, marker=mk, ms=7, mec=SURFACE, mew=1.2, ls="none",
                color=col, zorder=4, label=label)

    ax.set_xticks(xs, [r[0] for r in regimes], fontsize=6.8, color=INK2)
    ax.set_ylim(-0.04, 1.06)
    ax.set_ylabel("balanced accuracy, XOR relation (Task B)", fontsize=7.5)
    ax.set_xlim(-0.5, len(regimes) - 0.4)
    style_axes(ax, xgrid=False)
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK2, ncol=3,
              loc="lower center", bbox_to_anchor=(0.45, -0.36), handlelength=1.0)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig5_xor_splits.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    audit = load("audit_results.json")
    robust = load("robustness_results.json")
    struct = load("structure_tests.json")
    fig1_taskD(audit)
    fig2_leakage(audit, robust)
    fig3_repeat_correlation()
    fig4_permutation_null(struct)
    fig5_xor(load("xor_audit_results.json"))
