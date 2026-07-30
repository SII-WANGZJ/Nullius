#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
robustness.py -- adversarial self-checks on the audit findings.

Every finding in run_audit.py is attacked here with the objection a referee
would raise.  A finding that does not survive its own objection is downgraded
before it reaches the manuscript.

  Q1  Is Task-C's collapse under grouping an artefact of n_splits?
      With 4 ordered pairs per category-pair class, StratifiedGroupKFold(5)
      can place all 4 in one fold, leaving the class unseen in training.
      Sweep n_splits and report how many classes are actually trainable.

  Q2  Is the Task-D self-pair ablation statistically meaningful at all?
      Report per-fold accuracies and the number of DISTINCT positive pairs
      per fold (repeats are near-duplicates, so distinct pairs are the real n).

  Q3  Does the picture hold at the speckle-matched bin = 10 readout, or is it
      specific to the coarse bin = 25 operating point of the main manuscript?

  Q4  Repeat correlation: verify the authors' ">0.97" figure ourselves, since
      the leakage argument rests on it.
"""

from __future__ import annotations

import json
import os
import warnings

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score

import _path  # noqa: F401  (puts src/ on sys.path)
import nullius as N
warnings.filterwarnings("ignore", category=UserWarning)

TASK_D_CATS = [0, 1, 2, 3]


# ---------------------------------------------------------------------------
def taskC_nsplits_sweep(X, cp, groups):
    out = {}
    for n in (2, 4, 5, 8):
        skf = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=42)
        accs, trainable = [], []
        for tr, te in skf.split(X, cp, groups=groups):
            classes_tr = set(np.unique(cp[tr]))
            classes_te = set(np.unique(cp[te]))
            trainable.append(len(classes_te & classes_tr) / max(1, len(classes_te)))
            clf = RidgeClassifier(alpha=1.0)
            clf.fit(X[tr], cp[tr])
            accs.append(balanced_accuracy_score(cp[te], clf.predict(X[te])))
        out[n] = {"balanced_acc": round(float(np.mean(accs)), 4),
                  "frac_test_classes_seen_in_train": round(float(np.mean(trainable)), 3)}
    return out


def taskD_per_fold(X, sc, pinfo, exclude_self_pairs: bool):
    rows = []
    for held in TASK_D_CATS:
        tr, te = [], []
        for i, (x, y) in enumerate(pinfo):
            if exclude_self_pairs and x == y:
                continue
            cx = N.SEMANTIC_TOKENS[x]["category"]
            cy = N.SEMANTIC_TOKENS[y]["category"]
            (te if (cx == held or cy == held) else tr).append(i)
        tr, te = np.array(tr), np.array(te)
        if len(tr) < 2 or len(te) < 2 or len(np.unique(sc[tr])) < 2:
            rows.append({"held_category": held, "status": "skipped"})
            continue
        clf = RidgeClassifier(alpha=1.0)
        clf.fit(X[tr], sc[tr])
        acc = balanced_accuracy_score(sc[te], clf.predict(X[te]))
        pos_pairs = {pinfo[i] for i in te if sc[i] == 1}
        neg_pairs = {pinfo[i] for i in te if sc[i] == 0}
        rows.append({
            "held_category": held,
            "balanced_acc": round(float(acc), 4),
            "n_distinct_positive_pairs": len(pos_pairs),
            "n_distinct_negative_pairs": len(neg_pairs),
        })
    return rows


def repeat_correlation(B_per_rep):
    cors = []
    for key, arr in B_per_rep.items():
        v = np.concatenate([arr.real, arr.imag], axis=1)
        n = v.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                cors.append(np.corrcoef(v[i], v[j])[0, 1])
    cors = np.array(cors)
    return {"mean": round(float(cors.mean()), 4),
            "min": round(float(cors.min()), 4),
            "frac_above_0.97": round(float((cors > 0.97).mean()), 4)}


# ---------------------------------------------------------------------------
def main() -> int:
    report = {}
    for bin_size in (25, 10):
        print("\n" + "=" * 74)
        print(f"BIN = {bin_size}")
        print("=" * 74)

        single, blank, pair = N.load_all_data_binned(bin_size)
        B_per_rep = N.compute_B_per_repeat(blank, pair)
        feats, pid, sc, cp, pinfo = N.build_features(single, B_per_rep)
        Xb = feats["complex_B"]

        if bin_size == 25:
            rc = repeat_correlation(B_per_rep)
            print(f"\nQ4  Complex-B repeat correlation: mean={rc['mean']}, "
                  f"min={rc['min']}, frac>0.97={rc['frac_above_0.97']}")
            report["Q4_repeat_correlation_bin25"] = rc

        print("\nQ1  Task C under pair-grouped folds, n_splits sweep (Complex-B)")
        sweep = taskC_nsplits_sweep(Xb, cp, groups=pid)
        for n, v in sweep.items():
            print(f"      n_splits={n}: balanced_acc={v['balanced_acc']:.4f}   "
                  f"test classes seen in train = {v['frac_test_classes_seen_in_train']:.1%}")
        report[f"Q1_taskC_sweep_bin{bin_size}"] = sweep

        print("\nQ2/Q3  Task D per held-out category (Complex-B)")
        for excl in (False, True):
            rows = taskD_per_fold(Xb, sc, pinfo, exclude_self_pairs=excl)
            tag = "no self-pairs" if excl else "as released"
            vals = [r["balanced_acc"] for r in rows if "balanced_acc" in r]
            print(f"      [{tag:14s}] per-fold = "
                  + ", ".join(f"{v:.4f}" for v in vals)
                  + f"   mean={np.mean(vals):.4f}")
            for r in rows:
                if "n_distinct_positive_pairs" in r:
                    print(f"          held cat {r['held_category']}: "
                          f"{r['n_distinct_positive_pairs']} distinct positive pairs, "
                          f"{r['n_distinct_negative_pairs']} negative")
            report[f"Q2_taskD_bin{bin_size}_{'excl' if excl else 'incl'}"] = rows

        pr = N.participation_ratio_effective_rank(Xb)
        print(f"\n      Complex-B participation-ratio effective rank = {pr:.2f}")
        report[f"Q3_eff_rank_bin{bin_size}"] = round(pr, 3)

    out = os.path.join(N.RESULTS_DIR, "robustness_results.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
