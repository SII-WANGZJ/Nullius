#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_audit.py -- benchmark-integrity audit of the eight-token pair benchmark
released with arXiv:2604.27092 (data DOI 10.5281/zenodo.19890402).

Three stages:

  R   Replication.  Re-run the released pipeline unmodified and check that we
      reproduce the published accuracies.  Nothing below means anything if
      this stage fails.

  B2  Self-pair ablation (Task D).  In cross_split_eval() each held-out
      category contributes 4 positives, of which 2 are self-pairs (x, x).
      A featureless rule "same_category := (x == y)" therefore attains
      balanced accuracy 0.5 * (2/4) + 0.5 * (24/24) = 0.75.  Every published
      Task-D number lies at or below that value.  We re-run Task D with all
      (x, x) samples removed and compare against the featureless rule.

  B3  Repeat-leakage ablation (Tasks A/B/C).  The released scoring splits the
      five physical repeats of each pair across train and test folds, while
      the authors report a five-repeat correlation above 0.97.  We re-run with
      StratifiedGroupKFold grouped by pair, so that no pair appears in both
      folds.

Also computes the participation-ratio effective rank of the Complex-B feature
ensemble at each bin, to check the "~3 at bin = 25" statement in Report 5.

Usage:
    python run_audit.py                # bins 25 and 10
    python run_audit.py --bins 25      # single bin
    python run_audit.py --no-cache     # re-read all .npy frames
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score

import _path  # noqa: F401  (puts src/ on sys.path)
import nullius as N
# Values printed in the released reports, for the replication check.
# Report 5, advantage-boundary raw-input baselines (no optical propagation).
PUBLISHED_RAW = {
    "concat_raw":    {"A": 1.000, "B": 0.500, "C": 1.000, "D": 0.500},
    "diff_raw":      {"A": 0.875, "B": 0.500, "C": 0.769, "D": 0.500},
    "abs_diff_raw":  {"A": 0.453, "B": 1.000, "C": 0.334, "D": 0.719},
    "prod_raw":      {"A": 0.562, "B": 1.000, "C": 0.444, "D": 0.510},
    "conj_prod_raw": {"A": 0.891, "B": 1.000, "C": 0.891, "D": 0.698},
}

TASKS = ("A", "B", "C", "D")
TASK_NAMES = {
    "A": "pair identity (64-way)",
    "B": "same category (binary)",
    "C": "category pair (16-way)",
    "D": "cross-split same category",
}


# ---------------------------------------------------------------------------
def grouped_balanced_accuracy(X, y, groups, n_splits: int = 5):
    """StratifiedGroupKFold scoring: no group (= no pair) spans train/test.

    Returns (mean_balanced_accuracy, status).  Status is 'degenerate' when
    every class lives in exactly one group, which makes the task unlearnable
    by construction rather than merely hard.
    """
    n_classes = len(np.unique(y))
    n_groups = len(np.unique(groups))
    if n_classes >= n_groups:
        return float("nan"), "degenerate: one class per group"

    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs = []
    for tr, te in skf.split(X, y, groups=groups):
        if len(np.unique(y[tr])) < 2:
            continue
        clf = RidgeClassifier(alpha=1.0)
        clf.fit(X[tr], y[tr])
        accs.append(balanced_accuracy_score(y[te], clf.predict(X[te])))
    if not accs:
        return float("nan"), "degenerate: no usable fold"
    return float(np.mean(accs)), "ok"


def fmt(v) -> str:
    return "  n/a " if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.4f}"


# ---------------------------------------------------------------------------
def stage_R_raw_inputs(report: dict) -> None:
    """Replication of the raw-input baselines (no optics, bin-independent)."""
    print("\n" + "=" * 74)
    print("STAGE R1  Replication -- raw-input baselines (no optical propagation)")
    print("=" * 74)

    feats, pid, sc, cp, pinfo = N.build_raw_input_features()
    rows, max_dev = {}, 0.0
    print(f"  {'feature':16s} {'dim':>5s}  " + "  ".join(f"{t:>6s}" for t in TASKS)
          + "     (published in parentheses)")
    for name, X in feats.items():
        r = N.run_benchmark(X, pid, sc, cp, pinfo, name)
        rows[name] = r
        pub = PUBLISHED_RAW.get(name, {})
        cells = []
        for t in TASKS:
            got, exp = r[t], pub.get(t)
            if exp is not None:
                max_dev = max(max_dev, abs(got - exp))
                cells.append(f"{got:.3f}({exp:.3f})")
            else:
                cells.append(f"{got:.3f}")
        print(f"  {name:16s} {r['dim']:5d}  " + "  ".join(f"{c:>13s}" for c in cells))

    print(f"\n  max |ours - published| = {max_dev:.4f}"
          f"   -> {'REPLICATED' if max_dev <= 0.01 else 'MISMATCH, investigate'}")
    report["R1_raw_inputs"] = {"rows": rows, "max_deviation": round(max_dev, 4)}


def stage_R_optical(bin_size: int, report: dict) -> dict:
    """Replication of the optical feature families at one bin size."""
    print("\n" + "=" * 74)
    print(f"STAGE R2  Replication -- optical features, bin = {bin_size}")
    print("=" * 74)

    t0 = time.time()
    single, blank, pair = N.load_all_data_binned(bin_size)
    B_per_rep = N.compute_B_per_repeat(blank, pair)
    feats, pid, sc, cp, pinfo = N.build_features(single, B_per_rep)
    print(f"  loaded and demodulated in {time.time() - t0:.1f}s"
          f"   ({len(pid)} samples = {len(pid)//N.N_REPS} pairs x {N.N_REPS} repeats)")

    rows = {}
    print(f"\n  {'feature':18s} {'dim':>6s}  " + "  ".join(f"{t:>7s}" for t in TASKS))
    for name, X in feats.items():
        r = N.run_benchmark(X, pid, sc, cp, pinfo, name)
        rows[name] = r
        print(f"  {name:18s} {r['dim']:6d}  "
              + "  ".join(f"{r[t]:7.4f}" for t in TASKS))

    pr = N.participation_ratio_effective_rank(feats["complex_B"])
    print(f"\n  participation-ratio effective rank of Complex-B: {pr:.2f}")
    report[f"R2_optical_bin{bin_size}"] = {
        "rows": rows, "complex_B_effective_rank": round(pr, 3)}
    return {"feats": feats, "pid": pid, "sc": sc, "cp": cp, "pinfo": pinfo}


def stage_B2_selfpair(bundle_optical, bin_size: int, report: dict) -> None:
    """Task D with and without self-pairs, against the featureless rule."""
    print("\n" + "=" * 74)
    print(f"STAGE B2  Self-pair ablation on Task D, bin = {bin_size}")
    print("=" * 74)

    raw_feats, r_pid, r_sc, r_cp, r_pinfo = N.build_raw_input_features()
    families = {f"optical:{k}": (v, bundle_optical["sc"], bundle_optical["pinfo"])
                for k, v in bundle_optical["feats"].items()}
    families.update({f"raw:{k}": (v, r_sc, r_pinfo) for k, v in raw_feats.items()})

    triv_with = N.identity_rule_taskD(
        bundle_optical["pinfo"], bundle_optical["sc"], exclude_self_pairs=False)
    triv_without = N.identity_rule_taskD(
        bundle_optical["pinfo"], bundle_optical["sc"], exclude_self_pairs=True)

    print(f"  featureless rule  'same_category := (x == y)'")
    print(f"      with self-pairs   : {fmt(triv_with)}")
    print(f"      without self-pairs: {fmt(triv_without)}")
    print(f"\n  {'feature':26s} {'D (as released)':>16s} {'D (no self-pairs)':>18s} {'delta':>9s}")

    rows = {}
    for name, (X, sc, pinfo) in families.items():
        d_with = N.cross_split_eval(X, sc, pinfo, exclude_self_pairs=False)
        d_without = N.cross_split_eval(X, sc, pinfo, exclude_self_pairs=True)
        rows[name] = {"D_released": round(d_with, 4),
                      "D_no_self_pairs": round(d_without, 4),
                      "delta": round(d_without - d_with, 4)}
        print(f"  {name:26s} {fmt(d_with):>16s} {fmt(d_without):>18s} "
              f"{d_without - d_with:>+9.4f}")

    report[f"B2_selfpair_bin{bin_size}"] = {
        "trivial_rule_with_self_pairs": round(triv_with, 4),
        "trivial_rule_without_self_pairs": round(triv_without, 4),
        "rows": rows,
    }


def stage_B3_leakage(bundle_optical, bin_size: int, report: dict) -> None:
    """Tasks A/B/C under pair-grouped folds."""
    print("\n" + "=" * 74)
    print(f"STAGE B3  Repeat-leakage ablation on Tasks A/B/C, bin = {bin_size}")
    print("=" * 74)

    pid = bundle_optical["pid"]
    sc = bundle_optical["sc"]
    cp = bundle_optical["cp"]
    groups = pid  # one group per ordered pair

    print(f"  {'feature':18s} {'task':6s} {'released (SKF)':>15s} "
          f"{'grouped (SGKF)':>15s} {'delta':>9s}   status")
    rows = {}
    for name, X in bundle_optical["feats"].items():
        for task, y in (("A", pid), ("B", sc), ("C", cp)):
            released = N.skf_balanced_accuracy(X, y)
            grouped, status = grouped_balanced_accuracy(X, y, groups)
            rows[f"{name}:{task}"] = {
                "released_skf": round(released, 4),
                "grouped_sgkf": None if np.isnan(grouped) else round(grouped, 4),
                "status": status,
            }
            delta = "" if np.isnan(grouped) else f"{grouped - released:+9.4f}"
            print(f"  {name:18s} {task:6s} {released:15.4f} {fmt(grouped):>15s} "
                  f"{delta:>9s}   {status}")

    report[f"B3_leakage_bin{bin_size}"] = rows


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bins", type=int, nargs="+", default=[25, 10],
                    help="detector bin sizes to audit (main manuscript uses 25)")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    if not os.path.isdir(N.SEMANTIC_DATA_DIR):
        print(f"ERROR: frame data not found at\n  {N.SEMANTIC_DATA_DIR}", file=sys.stderr)
        return 2
    if args.no_cache:
        for f in os.listdir(N.CACHE_DIR):
            os.remove(os.path.join(N.CACHE_DIR, f))

    report: dict = {
        "source": "arXiv:2604.27092 / zenodo.19890402",
        "pipeline_mirrored_from":
            "05_result4_refine_paper/scripts/analyze_advantage_boundary.py",
        "bins": args.bins,
    }

    stage_R_raw_inputs(report)
    for b in args.bins:
        bundle = stage_R_optical(b, report)
        stage_B2_selfpair(bundle, b, report)
        stage_B3_leakage(bundle, b, report)

    out = os.path.join(N.RESULTS_DIR, "audit_results.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
