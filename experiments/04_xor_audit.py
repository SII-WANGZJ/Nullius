#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xor_audit.py -- audit of the four-token XOR showcase (exp_bilinear_001).

The plan (X1-X8) was frozen in README.md before this script was run.

The XOR task differs in kind from the semantic benchmark: linear
concatenation genuinely cannot express (x+y) mod 2, while a product feature
can.  That identity is not in question.  What we test is whether the
evaluation demonstrates a generalisable interaction feature or memorisation
of a 16-entry pair table.

Label structure, established before running:
  label(x, y) = (x + y) % 2, i.e. "tokens fall in different groups" for
  groups {0,2} and {1,3}.  It is symmetric, and all four self-pairs (x,x)
  carry label 0.
"""

from __future__ import annotations

import json
import os
import warnings
from itertools import combinations

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import (StratifiedKFold, StratifiedGroupKFold,
                                     cross_val_score)
from sklearn.metrics import balanced_accuracy_score

import _path  # noqa: F401  (puts src/ on sys.path)
import nullius as N
warnings.filterwarnings("ignore", category=UserWarning)

N_TOKENS = 4
BIN_SIZE = 25                      # authors' setting: 500/25 -> 400 channels
N_REPS = 5
PHASE = ["0", "pi2", "pi", "3pi2"]
DATA = os.path.join(N.DEPOSIT_DIR, "shared_raw_data", "result4_frame_data",
                    "exp_bilinear_001", "exp_bilinear_001")

PAIRS = [(x, y) for x in range(N_TOKENS) for y in range(N_TOKENS)]
XOR = {(x, y): (x + y) % 2 for (x, y) in PAIRS}


# ---------------------------------------------------------------------------
def load():
    def rd(p):
        return N.bin_roi(np.load(p), BIN_SIZE)

    single = {t: np.array([rd(os.path.join(DATA, "singles",
              f"single_t{t}_rep{r:02d}.npy")) for r in range(N_REPS)])
              for t in range(N_TOKENS)}
    blank = {(y, p): np.array([rd(os.path.join(DATA, "blank_controls",
             f"blank_y{y}_phi{PHASE[p]}_rep{r:02d}.npy")) for r in range(N_REPS)])
             for y in range(N_TOKENS) for p in range(4)}
    pair = {(x, y, p): np.array([rd(os.path.join(DATA, "pair_interactions",
            f"pair_x{x}_y{y}_phi{PHASE[p]}_rep{r:02d}.npy")) for r in range(N_REPS)])
            for (x, y) in PAIRS for p in range(4)}
    return single, blank, pair


def build(single, blank, pair):
    Q0 = {y: np.array([(blank[(y, 0)][r] - blank[(y, 2)][r]) / 4.0
                       + 1j * (blank[(y, 3)][r] - blank[(y, 1)][r]) / 4.0
                       for r in range(N_REPS)]) for y in range(N_TOKENS)}
    z = {t: single[t].mean(axis=0) for t in range(N_TOKENS)}

    feats = {"concat": [], "digital_bilinear": [], "complex_B": []}
    pid, xor, pinfo = [], [], []
    for (x, y) in PAIRS:
        for r in range(N_REPS):
            Q = ((pair[(x, y, 0)][r] - pair[(x, y, 2)][r]) / 4.0
                 + 1j * (pair[(x, y, 3)][r] - pair[(x, y, 1)][r]) / 4.0)
            B = Q - Q0[y][r]
            feats["concat"].append(np.concatenate([z[x], z[y]]))
            feats["digital_bilinear"].append(z[x] * z[y])
            feats["complex_B"].append(np.concatenate([B.real, B.imag]))
            pid.append(x * N_TOKENS + y)
            xor.append(XOR[(x, y)])
            pinfo.append((x, y))
    return ({k: np.array(v) for k, v in feats.items()},
            np.array(pid), np.array(xor), pinfo)


# ---------------------------------------------------------------------------
def skf(X, y, n_splits=5):
    clf = RidgeClassifier(alpha=1.0)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    return float(cross_val_score(clf, X, y, cv=cv,
                                 scoring="balanced_accuracy").mean())


def grouped(X, y, groups, n_splits=4):
    n_g = len(np.unique(groups))
    if len(np.unique(y)) >= n_g:
        return float("nan"), "degenerate: one class per group"
    cv = StratifiedGroupKFold(n_splits=min(n_splits, n_g), shuffle=True,
                              random_state=42)
    accs = []
    for tr, te in cv.split(X, y, groups=groups):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        clf = RidgeClassifier(alpha=1.0).fit(X[tr], y[tr])
        accs.append(balanced_accuracy_score(y[te], clf.predict(X[te])))
    return (float(np.mean(accs)), "ok") if accs else (float("nan"), "no usable fold")


def leave_one_token_out(X, y, pinfo):
    accs, detail = [], []
    for t in range(N_TOKENS):
        te = [i for i, (a, b) in enumerate(pinfo) if a == t or b == t]
        tr = [i for i, (a, b) in enumerate(pinfo) if a != t and b != t]
        tr, te = np.array(tr), np.array(te)
        if len(tr) < 2 or len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            detail.append({"held_token": t, "status": "skipped"})
            continue
        clf = RidgeClassifier(alpha=1.0).fit(X[tr], y[tr])
        a = balanced_accuracy_score(y[te], clf.predict(X[te]))
        accs.append(a)
        detail.append({"held_token": t, "balanced_acc": round(float(a), 4),
                       "n_train": int(len(tr)), "n_test": int(len(te))})
    return (float(np.mean(accs)) if accs else float("nan")), detail


def selfpair_2x2(X, y, pinfo):
    """Train x test inclusion of self-pairs, under unordered-pair grouping."""
    out = {}
    upid = np.array([sorted(set([a, b])).__len__() * 0 +
                     list(combinations(range(N_TOKENS), 2)).index((min(a, b), max(a, b)))
                     if a != b else 100 + a for (a, b) in pinfo])
    for tr_s in (True, False):
        for te_s in (True, False):
            keep_tr = np.array([tr_s or a != b for (a, b) in pinfo])
            keep_te = np.array([te_s or a != b for (a, b) in pinfo])
            accs = []
            for g in np.unique(upid):
                te = np.where((upid == g) & keep_te)[0]
                tr = np.where((upid != g) & keep_tr)[0]
                if (len(tr) < 2 or len(te) < 1
                        or len(np.unique(y[tr])) < 2):
                    continue
                clf = RidgeClassifier(alpha=1.0).fit(X[tr], y[tr])
                pred = clf.predict(X[te])
                accs.append(float((pred == y[te]).mean()))
            k = f"train_{'with' if tr_s else 'without'}__test_{'with' if te_s else 'without'}"
            out[k] = round(float(np.mean(accs)), 4) if accs else None
    return out


def exhaustive_leave_group_out(X, y, groups):
    """Leave each group out in turn; pool ALL out-of-fold predictions, then
    score once.  Per-fold scoring is meaningless here because a held-out
    ordered pair contains a single class."""
    pred = np.full(len(y), -1, dtype=int)
    per_group = []
    for g in np.unique(groups):
        te = np.where(groups == g)[0]
        tr = np.where(groups != g)[0]
        if len(np.unique(y[tr])) < 2:
            continue
        clf = RidgeClassifier(alpha=1.0).fit(X[tr], y[tr])
        p = clf.predict(X[te])
        pred[te] = p
        per_group.append({"group": int(g), "n_test": int(len(te)),
                          "true_class": int(y[te][0]),
                          "frac_correct": round(float((p == y[te]).mean()), 4)})
    ok = pred >= 0
    return (round(float(balanced_accuracy_score(y[ok], pred[ok])), 4),
            per_group, int(ok.sum()))


def measurement_free(y, pinfo):
    """Rules that use only the pair table, never the optics."""
    rules = {
        "x == y  -> label 0, else 1": [0 if a == b else 1 for (a, b) in pinfo],
        "always majority class": [int(np.bincount(y).argmax())] * len(y),
        "group identity (oracle {0,2}/{1,3})":
            [int((a % 2) != (b % 2)) for (a, b) in pinfo],
    }
    return {k: round(float(balanced_accuracy_score(y, np.array(v))), 4)
            for k, v in rules.items()}


def exact_null_partitions(X, pinfo, groups_unordered):
    """All 3 partitions of 4 tokens into 2 unlabelled groups of 2."""
    parts = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]
    rows = []
    for p in parts:
        g = {}
        for gi, grp in enumerate(p):
            for t in grp:
                g[t] = gi
        lab = np.array([int(g[a] != g[b]) for (a, b) in pinfo])
        acc, status = grouped(X, lab, groups_unordered)
        rows.append({"partition": f"{p[0]}|{p[1]}",
                     "is_designated": p == ((0, 2), (1, 3)),
                     "balanced_acc": None if np.isnan(acc) else round(acc, 4),
                     "status": status})
    return rows


# ---------------------------------------------------------------------------
def main() -> int:
    if not os.path.isdir(DATA):
        print(f"ERROR: {DATA} not found")
        return 2
    rep = {"experiment": "exp_bilinear_001", "bin": BIN_SIZE}

    print("=" * 78)
    print("X2  Label structure")
    print("=" * 78)
    M = np.array([[XOR[(x, y)] for y in range(N_TOKENS)] for x in range(N_TOKENS)])
    print("  label matrix (row = x, col = y), label = (x+y) % 2:")
    for r in M:
        print("      " + "  ".join(str(v) for v in r))
    n_self_pos = sum(XOR[(t, t)] for t in range(N_TOKENS))
    print(f"  positives {int(M.sum())}/16, negatives {int(16-M.sum())}/16")
    print(f"  self-pairs carrying label 1: {n_self_pos}/4  "
          f"(all self-pairs are negatives)")
    print(f"  label is symmetric: {bool(np.all(M == M.T))}")
    rep["X2_label_structure"] = {
        "n_positive_pairs": int(M.sum()), "n_negative_pairs": int(16 - M.sum()),
        "self_pairs_positive": int(n_self_pos), "symmetric": bool(np.all(M == M.T))}

    single, blank, pair = load()
    feats, pid, xor, pinfo = build(single, blank, pair)
    print(f"\n  loaded {len(xor)} samples = 16 ordered pairs x {N_REPS} repeats, "
          f"{feats['complex_B'].shape[1]} Complex-B dims")

    ordered_g = pid
    unordered_g = np.array([min(a, b) * N_TOKENS + max(a, b) for (a, b) in pinfo])

    print("\n" + "=" * 78)
    print("X5  Measurement-free structural baselines (Task B)")
    print("=" * 78)
    mf = measurement_free(xor, pinfo)
    for k, v in mf.items():
        print(f"  {k:40s} {v:.4f}")
    rep["X5_measurement_free"] = mf

    print("\n" + "=" * 78)
    print("X1/X3/X8  Task B under four splitting regimes")
    print("=" * 78)
    print(f"  {'feature':20s} {'sample-level':>13s} {'ordered-pair':>13s} "
          f"{'unordered-pair':>15s} {'leave-1-token':>14s}")
    rows = {}
    for name, X in feats.items():
        s = skf(X, xor)
        o, _ = grouped(X, xor, ordered_g)
        u, _ = grouped(X, xor, unordered_g)
        l, ldet = leave_one_token_out(X, xor, pinfo)
        rows[name] = {"sample_level": round(s, 4),
                      "ordered_pair_grouped": None if np.isnan(o) else round(o, 4),
                      "unordered_pair_grouped": None if np.isnan(u) else round(u, 4),
                      "leave_one_token_out": None if np.isnan(l) else round(l, 4),
                      "leave_one_token_detail": ldet}
        f = lambda v: "   n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.4f}"
        print(f"  {name:20s} {f(s):>13s} {f(o):>13s} {f(u):>15s} {f(l):>14s}")
    rep["X1_X3_X8_taskB"] = rows

    print("\n  Task A (16-class ordered pair identity):")
    for name, X in feats.items():
        s = skf(X, pid)
        o, st = grouped(X, pid, ordered_g)
        print(f"    {name:20s} sample-level {s:.4f}   ordered-pair-grouped: {st}")
        rows[name]["taskA_sample_level"] = round(s, 4)
        rows[name]["taskA_grouped_status"] = st

    print("\n" + "=" * 78)
    print("X9/X10  Exhaustive leave-one-group-out (pooled out-of-fold predictions)")
    print("=" * 78)
    print(f"  {'feature':20s} {'leave-1-ordered-pair':>22s} {'leave-1-unordered-group':>25s}")
    ex = {}
    for name, X in feats.items():
        a_o, pg_o, n_o = exhaustive_leave_group_out(X, xor, ordered_g)
        a_u, pg_u, n_u = exhaustive_leave_group_out(X, xor, unordered_g)
        ex[name] = {"leave_one_ordered_pair_out": a_o,
                    "leave_one_unordered_group_out": a_u,
                    "n_pooled_ordered": n_o, "n_pooled_unordered": n_u,
                    "per_group_ordered": pg_o, "per_group_unordered": pg_u}
        print(f"  {name:20s} {a_o:22.4f} {a_u:25.4f}")
    print(f"  (16 ordered-pair folds; 10 unordered groups = 4 self + 6 cross)")
    rep["X9_X10_exhaustive"] = ex

    print("\n" + "=" * 78)
    print("X4  Self-pair 2x2 (unordered-pair held-out unit), Complex-B")
    print("=" * 78)
    s22 = selfpair_2x2(feats["complex_B"], xor, pinfo)
    for k, v in s22.items():
        print(f"  {k:44s} {v}")
    rep["X4_selfpair_2x2_complexB"] = s22

    print("\n" + "=" * 78)
    print("X6  Exact enumeration of all 3 token partitions, Complex-B")
    print("=" * 78)
    en = exact_null_partitions(feats["complex_B"], pinfo, unordered_g)
    for r in en:
        tag = "  <- designated" if r["is_designated"] else ""
        print(f"  {r['partition']:14s} {str(r['balanced_acc']):>8s}   {r['status']}{tag}")
    rep["X6_partition_enumeration"] = en

    out = os.path.join(N.RESULTS_DIR, "xor_audit_results.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
