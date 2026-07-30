#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
common.py -- faithful re-implementation of the released analysis pipeline.

Mirrors, line for line where it matters:
    SM_Source_Materials_CORE_5REPORTS/05_result4_refine_paper/scripts/
        analyze_advantage_boundary.py

The point of mirroring rather than rewriting is attribution: any difference
between our numbers and the published ones must be traceable to an explicit,
documented audit modification, not to an incidental difference in loading,
binning, classifier or scoring.

Reference: Yang et al., arXiv:2604.27092; data DOI 10.5281/zenodo.19890402.
"""

from __future__ import annotations

import os
import numpy as np

from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import balanced_accuracy_score

# ---------------------------------------------------------------------------
# Configuration -- identical to the released script
# ---------------------------------------------------------------------------
N_TOKENS = 8
ROI_H, ROI_W = 500, 500
PHASE_FILE_LABELS = ["0", "pi2", "pi", "3pi2"]
N_REPS = 5

SEMANTIC_TOKENS = [
    {"id": 0, "word": "cat",    "category": 0, "category_name": "animals"},
    {"id": 1, "word": "dog",    "category": 0, "category_name": "animals"},
    {"id": 2, "word": "hammer", "category": 1, "category_name": "tools"},
    {"id": 3, "word": "saw",    "category": 1, "category_name": "tools"},
    {"id": 4, "word": "apple",  "category": 2, "category_name": "fruits"},
    {"id": 5, "word": "grape",  "category": 2, "category_name": "fruits"},
    {"id": 6, "word": "car",    "category": 3, "category_name": "vehicles"},
    {"id": 7, "word": "train",  "category": 3, "category_name": "vehicles"},
]

AUDIT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(AUDIT_DIR)
PACKAGE_DIR = os.path.join(PROJECT_DIR, "SM_Source_Materials_CORE_5REPORTS")
DATA_DIR = os.path.join(PACKAGE_DIR, "shared_raw_data", "result4_frame_data",
                        "exp_complex_B_semantic")
CACHE_DIR = os.path.join(AUDIT_DIR, "cache")
RESULTS_DIR = os.path.join(AUDIT_DIR, "results")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Token input vectors E(x)
#
# NOTE (audit finding B1): these are seeded uniform random phases on the unit
# circle.  The words and categories in SEMANTIC_TOKENS are labels attached to
# random vectors; no semantic embedding enters the pipeline at any point.
# ---------------------------------------------------------------------------
def make_token_input(tid: int) -> np.ndarray:
    """Reconstruct the 36-dim complex input vector E(x) from the same seed."""
    seed = 10000 + tid * 7919
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0, 2 * np.pi, (6, 6))
    return np.exp(1j * phases).flatten()


TOKEN_INPUTS = [make_token_input(t) for t in range(N_TOKENS)]
INPUT_DIM = 36


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def bin_roi(roi: np.ndarray, bin_size: int) -> np.ndarray:
    h, w = roi.shape
    nh, nw = h // bin_size, w // bin_size
    cropped = roi[:nh * bin_size, :nw * bin_size]
    return cropped.reshape(nh, bin_size, nw, bin_size).mean(axis=(1, 3)).flatten()


def load_all_data_binned(bin_size: int, use_cache: bool = True):
    """Return (single_data, blank_data, pair_data), binned exactly as released."""
    cache_path = os.path.join(CACHE_DIR, f"binned_{bin_size}.npz")
    if use_cache and os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=False)
        single_data = {t: z[f"s{t}"] for t in range(N_TOKENS)}
        blank_data = {(y, p): z[f"b{y}_{p}"]
                      for y in range(N_TOKENS) for p in range(4)}
        pair_data = {(x, y, p): z[f"p{x}_{y}_{p}"]
                     for x in range(N_TOKENS) for y in range(N_TOKENS)
                     for p in range(4)}
        return single_data, blank_data, pair_data

    singles_dir = os.path.join(DATA_DIR, "singles")
    blanks_dir = os.path.join(DATA_DIR, "blank_controls")
    pairs_dir = os.path.join(DATA_DIR, "pair_interactions")

    single_data = {}
    for tid in range(N_TOKENS):
        vecs = [bin_roi(np.load(os.path.join(
            singles_dir, f"single_t{tid}_rep{rep:02d}.npy")), bin_size)
            for rep in range(N_REPS)]
        single_data[tid] = np.array(vecs)

    blank_data = {}
    for y_tid in range(N_TOKENS):
        for phi_idx in range(4):
            plabel = PHASE_FILE_LABELS[phi_idx]
            vecs = [bin_roi(np.load(os.path.join(
                blanks_dir, f"blank_y{y_tid}_phi{plabel}_rep{rep:02d}.npy")), bin_size)
                for rep in range(N_REPS)]
            blank_data[(y_tid, phi_idx)] = np.array(vecs)

    pair_data = {}
    for x in range(N_TOKENS):
        for y in range(N_TOKENS):
            for phi_idx in range(4):
                plabel = PHASE_FILE_LABELS[phi_idx]
                vecs = [bin_roi(np.load(os.path.join(
                    pairs_dir, f"pair_x{x}_y{y}_phi{plabel}_rep{rep:02d}.npy")), bin_size)
                    for rep in range(N_REPS)]
                pair_data[(x, y, phi_idx)] = np.array(vecs)

    if use_cache:
        payload = {f"s{t}": single_data[t] for t in range(N_TOKENS)}
        payload.update({f"b{y}_{p}": blank_data[(y, p)]
                        for y in range(N_TOKENS) for p in range(4)})
        payload.update({f"p{x}_{y}_{p}": pair_data[(x, y, p)]
                        for x in range(N_TOKENS) for y in range(N_TOKENS)
                        for p in range(4)})
        np.savez_compressed(cache_path, **payload)

    return single_data, blank_data, pair_data


def compute_B_per_repeat(blank_data, pair_data):
    """Four-step phase-shifting demodulation with blank subtraction."""
    Q_blank_per_rep = {}
    for y_tid in range(N_TOKENS):
        Q_list = []
        for rep in range(N_REPS):
            I0 = blank_data[(y_tid, 0)][rep]
            I1 = blank_data[(y_tid, 1)][rep]
            I2 = blank_data[(y_tid, 2)][rep]
            I3 = blank_data[(y_tid, 3)][rep]
            Q_list.append((I0 - I2) / 4.0 + 1j * (I3 - I1) / 4.0)
        Q_blank_per_rep[y_tid] = np.array(Q_list)

    B_per_rep = {}
    for x in range(N_TOKENS):
        for y in range(N_TOKENS):
            B_list = []
            for rep in range(N_REPS):
                I0 = pair_data[(x, y, 0)][rep]
                I1 = pair_data[(x, y, 1)][rep]
                I2 = pair_data[(x, y, 2)][rep]
                I3 = pair_data[(x, y, 3)][rep]
                Q_xy = (I0 - I2) / 4.0 + 1j * (I3 - I1) / 4.0
                B_list.append(Q_xy - Q_blank_per_rep[y][rep])
            B_per_rep[(x, y)] = np.array(B_list)
    return B_per_rep


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------
def build_features(single_data, B_per_rep):
    """Optical feature families, as released.

    NOTE (audit finding O1): `single_data` holds binned camera *intensity*,
    so `digital_bilinear = zx * zy` is a real intensity product.  It is not
    the conjugate product of the propagated complex fields, and therefore is
    not the operator-matched digital reconstruction of Complex-B.
    """
    z_mean = {tid: single_data[tid].mean(axis=0) for tid in range(N_TOKENS)}
    features = {"concat": [], "digital_bilinear": [], "complex_B": []}
    pid_labels, sc_labels, cp_labels, pair_info = [], [], [], []

    for x in range(N_TOKENS):
        for y in range(N_TOKENS):
            cx = SEMANTIC_TOKENS[x]["category"]
            cy = SEMANTIC_TOKENS[y]["category"]
            B_reps = B_per_rep[(x, y)]
            for rep in range(min(N_REPS, B_reps.shape[0])):
                zx, zy = z_mean[x], z_mean[y]
                features["concat"].append(np.concatenate([zx, zy]))
                features["digital_bilinear"].append(zx * zy)
                features["complex_B"].append(
                    np.concatenate([np.real(B_reps[rep]), np.imag(B_reps[rep])]))
                pid_labels.append(x * N_TOKENS + y)
                sc_labels.append(int(cx == cy))
                cp_labels.append(cx * 4 + cy)
                pair_info.append((x, y))

    return ({k: np.array(v) for k, v in features.items()},
            np.array(pid_labels), np.array(sc_labels),
            np.array(cp_labels), pair_info)


def build_raw_input_features(noise_scale: float = 1e-8, seed: int = 12345):
    """Raw-input baselines computed directly from E(x), E(y): no optics at all.

    Mirrors experiment_raw_input_baselines() in the released script, including
    the five *pseudo*-repeats (identical vectors plus 1e-8 noise).
    """
    rng = np.random.default_rng(seed)
    X_raw = {"concat_raw": [], "diff_raw": [], "abs_diff_raw": [],
             "prod_raw": [], "conj_prod_raw": []}
    pid_labels, sc_labels, cp_labels, pair_info = [], [], [], []

    for x in range(N_TOKENS):
        for y in range(N_TOKENS):
            cx = SEMANTIC_TOKENS[x]["category"]
            cy = SEMANTIC_TOKENS[y]["category"]
            Ex, Ey = TOKEN_INPUTS[x], TOKEN_INPUTS[y]

            concat_v = np.concatenate([np.real(Ex), np.imag(Ex),
                                       np.real(Ey), np.imag(Ey)])
            diff_v = Ex - Ey
            abs_diff_v = np.abs(Ex - Ey)
            prod_v = Ex * Ey
            conj_prod_v = np.conj(Ex) * Ey

            for _ in range(N_REPS):
                X_raw["concat_raw"].append(
                    concat_v + rng.normal(0, noise_scale, concat_v.shape))
                X_raw["diff_raw"].append(np.concatenate(
                    [np.real(diff_v), np.imag(diff_v)]) + rng.normal(0, noise_scale, 72))
                X_raw["abs_diff_raw"].append(
                    abs_diff_v + rng.normal(0, noise_scale, 36))
                X_raw["prod_raw"].append(np.concatenate(
                    [np.real(prod_v), np.imag(prod_v)]) + rng.normal(0, noise_scale, 72))
                X_raw["conj_prod_raw"].append(np.concatenate(
                    [np.real(conj_prod_v), np.imag(conj_prod_v)]) + rng.normal(0, noise_scale, 72))

                pid_labels.append(x * N_TOKENS + y)
                sc_labels.append(int(cx == cy))
                cp_labels.append(cx * 4 + cy)
                pair_info.append((x, y))

    return ({k: np.array(v) for k, v in X_raw.items()},
            np.array(pid_labels), np.array(sc_labels),
            np.array(cp_labels), pair_info)


# ---------------------------------------------------------------------------
# Scoring -- released versions
# ---------------------------------------------------------------------------
def skf_balanced_accuracy(X, y, n_splits: int = 5) -> float:
    """As released: StratifiedKFold over samples, i.e. repeats of the same
    physical pair are split across train and test (audit finding B3)."""
    clf = RidgeClassifier(alpha=1.0)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=skf, scoring="balanced_accuracy")
    return float(scores.mean())


def cross_split_eval(X, y_same_cat, pair_info, exclude_self_pairs: bool = False) -> float:
    """Task D, as released; `exclude_self_pairs` is the audit modification.

    For each held-out category the test set contains 4 positives, of which 2
    are self-pairs (x, x).  Setting exclude_self_pairs=True removes every
    (x, x) sample from both train and test.
    """
    unique_cats = sorted(set(SEMANTIC_TOKENS[i]["category"] for i in range(N_TOKENS)))
    accs = []
    for held_cat in unique_cats:
        train_mask, test_mask = [], []
        for i, (x, y) in enumerate(pair_info):
            if exclude_self_pairs and x == y:
                continue
            cx = SEMANTIC_TOKENS[x]["category"]
            cy = SEMANTIC_TOKENS[y]["category"]
            (test_mask if (cx == held_cat or cy == held_cat) else train_mask).append(i)
        if len(train_mask) < 2 or len(test_mask) < 2:
            continue
        train_mask, test_mask = np.array(train_mask), np.array(test_mask)
        if len(np.unique(y_same_cat[train_mask])) < 2:
            continue
        clf = RidgeClassifier(alpha=1.0)
        clf.fit(X[train_mask], y_same_cat[train_mask])
        accs.append(balanced_accuracy_score(
            y_same_cat[test_mask], clf.predict(X[test_mask])))
    return float(np.mean(accs)) if accs else float("nan")


def run_benchmark(X, pid, sc, cp, pinfo, name: str) -> dict:
    """All four tasks, exactly as released."""
    return {
        "name": name,
        "dim": int(X.shape[1]),
        "A": round(skf_balanced_accuracy(X, pid), 4),
        "B": round(skf_balanced_accuracy(X, sc), 4),
        "C": round(skf_balanced_accuracy(X, cp), 4),
        "D": round(cross_split_eval(X, sc, pinfo), 4),
    }


# ---------------------------------------------------------------------------
# Audit-only helpers
# ---------------------------------------------------------------------------
def trivial_self_pair_detector_taskD(pair_info, y_same_cat,
                                     exclude_self_pairs: bool = False) -> float:
    """Balanced accuracy of the rule `same_category := (x == y)`.

    Uses no features whatsoever.  Evaluated on exactly the test masks that
    cross_split_eval() builds, so it is directly comparable to every reported
    Task-D number.
    """
    unique_cats = sorted(set(SEMANTIC_TOKENS[i]["category"] for i in range(N_TOKENS)))
    accs = []
    for held_cat in unique_cats:
        test_idx = []
        for i, (x, y) in enumerate(pair_info):
            if exclude_self_pairs and x == y:
                continue
            cx = SEMANTIC_TOKENS[x]["category"]
            cy = SEMANTIC_TOKENS[y]["category"]
            if cx == held_cat or cy == held_cat:
                test_idx.append(i)
        if len(test_idx) < 2:
            continue
        test_idx = np.array(test_idx)
        pred = np.array([int(pair_info[i][0] == pair_info[i][1]) for i in test_idx])
        if len(np.unique(y_same_cat[test_idx])) < 2:
            continue
        accs.append(balanced_accuracy_score(y_same_cat[test_idx], pred))
    return float(np.mean(accs)) if accs else float("nan")


def participation_ratio_effective_rank(X: np.ndarray) -> float:
    """PR effective rank of the feature ensemble: (sum s_i)^2 / sum s_i^2,
    computed on eigenvalues of the centred covariance."""
    Xc = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    ev = s ** 2
    return float(ev.sum() ** 2 / (ev ** 2).sum())
