#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Splitting regimes, metrics, and measurement-free reference rules.

The distinction this module exists to make explicit: the *unit of
generalisation* a split respects. ``skf_balanced_accuracy`` splits over
samples, so repeats of one physical pair fall on both sides; the grouped
variants do not.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import (StratifiedGroupKFold, StratifiedKFold,
                                     cross_val_score)

from .config import N_TOKENS, SEMANTIC_TOKENS


def skf_balanced_accuracy(X, y, n_splits: int = 5) -> float:
    """As released: StratifiedKFold over samples (repeats span folds)."""
    clf = RidgeClassifier(alpha=1.0)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    return float(cross_val_score(clf, X, y, cv=cv,
                                 scoring="balanced_accuracy").mean())


def grouped_balanced_accuracy(X, y, groups, n_splits: int = 5):
    """Group-respecting scoring. Returns ``(accuracy, status)``.

    Status is ``degenerate`` when every class occupies exactly one group, which
    makes the task unlearnable by construction rather than merely hard.
    """
    n_classes, n_groups = len(np.unique(y)), len(np.unique(groups))
    if n_classes >= n_groups:
        return float("nan"), "degenerate: one class per group"
    cv = StratifiedGroupKFold(n_splits=min(n_splits, n_groups), shuffle=True,
                              random_state=42)
    accs = []
    for tr, te in cv.split(X, y, groups=groups):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        clf = RidgeClassifier(alpha=1.0).fit(X[tr], y[tr])
        accs.append(balanced_accuracy_score(y[te], clf.predict(X[te])))
    if not accs:
        return float("nan"), "degenerate: no usable fold"
    return float(np.mean(accs)), "ok"


def exhaustive_leave_group_out(X, y, groups):
    """Leave each group out once; pool all out-of-fold predictions, score once.

    Per-fold scoring is undefined when a held-out group carries a single class,
    which is why the predictions are pooled before scoring.
    """
    pred = np.full(len(y), -1, dtype=int)
    per_group = []
    for g in np.unique(groups):
        te, tr = np.where(groups == g)[0], np.where(groups != g)[0]
        if len(np.unique(y[tr])) < 2:
            continue
        p = RidgeClassifier(alpha=1.0).fit(X[tr], y[tr]).predict(X[te])
        pred[te] = p
        per_group.append({"group": int(g), "n_test": int(len(te)),
                          "true_class": int(y[te][0]),
                          "frac_correct": round(float((p == y[te]).mean()), 4)})
    ok = pred >= 0
    return (round(float(balanced_accuracy_score(y[ok], pred[ok])), 4),
            per_group, int(ok.sum()))


def cross_split_eval(X, y_same_cat, pinfo, exclude_self_pairs: bool = False):
    """Task D: hold out one category at a time.

    ``exclude_self_pairs`` is the audit modification. Per held-out category the
    test set has 4 distinct positives, 2 of them self-pairs, against 24
    negatives.
    """
    cats = sorted({SEMANTIC_TOKENS[i]["category"] for i in range(N_TOKENS)})
    accs = []
    for held in cats:
        tr, te = [], []
        for i, (x, y) in enumerate(pinfo):
            if exclude_self_pairs and x == y:
                continue
            cx = SEMANTIC_TOKENS[x]["category"]
            cy = SEMANTIC_TOKENS[y]["category"]
            (te if (cx == held or cy == held) else tr).append(i)
        tr, te = np.array(tr), np.array(te)
        if len(tr) < 2 or len(te) < 2 or len(np.unique(y_same_cat[tr])) < 2:
            continue
        clf = RidgeClassifier(alpha=1.0).fit(X[tr], y_same_cat[tr])
        accs.append(balanced_accuracy_score(y_same_cat[te], clf.predict(X[te])))
    return float(np.mean(accs)) if accs else float("nan")


def run_benchmark(X, pid, sc, cp, pinfo, name: str) -> dict:
    """All four released tasks, scored exactly as released."""
    return {"name": name, "dim": int(X.shape[1]),
            "A": round(skf_balanced_accuracy(X, pid), 4),
            "B": round(skf_balanced_accuracy(X, sc), 4),
            "C": round(skf_balanced_accuracy(X, cp), 4),
            "D": round(cross_split_eval(X, sc, pinfo), 4)}


def identity_rule_taskD(pinfo, y_same_cat, exclude_self_pairs: bool = False):
    """Balanced accuracy of ``same_category := (x == y)``.

    Uses the pair labels and no measurement, on exactly the test masks that
    ``cross_split_eval`` builds, so it is directly comparable.
    """
    cats = sorted({SEMANTIC_TOKENS[i]["category"] for i in range(N_TOKENS)})
    accs = []
    for held in cats:
        te = []
        for i, (x, y) in enumerate(pinfo):
            if exclude_self_pairs and x == y:
                continue
            cx = SEMANTIC_TOKENS[x]["category"]
            cy = SEMANTIC_TOKENS[y]["category"]
            if cx == held or cy == held:
                te.append(i)
        if len(te) < 2:
            continue
        te = np.array(te)
        if len(np.unique(y_same_cat[te])) < 2:
            continue
        pred = np.array([int(pinfo[i][0] == pinfo[i][1]) for i in te])
        accs.append(balanced_accuracy_score(y_same_cat[te], pred))
    return float(np.mean(accs)) if accs else float("nan")


def participation_ratio_effective_rank(X: np.ndarray) -> float:
    """(sum s_i)^2 / sum s_i^2 on the centred feature ensemble.

    A property of the observed feature ensemble on this dataset, not a rank
    bound on the apparatus.
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    ev = np.linalg.svd(Xc, compute_uv=False) ** 2
    return float(ev.sum() ** 2 / (ev ** 2).sum())
