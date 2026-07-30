#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
structure_tests.py -- three checks that must settle before the corresponding
claims may be written down.

  S1  Ideal-model identities versus the empirical estimator.
      Under the stated linear-optical model the unbinned channel kernel
      K_p = T_p^dag T_p is *algebraically* rank-one, Hermitian and positive
      semidefinite; this is a theorem, not an assumption.  What the release
      contains is an estimator Bhat = B + eta carrying a reference arm, phase
      stepping, blank subtraction, drift and calibration error.  We quantify
      how far Bhat departs from the ideal identities, and then test whether a
      single phase-gauge correction accounts for the departure -- because a
      global or channel-wise phase offset produces exactly this signature
      without implying that the cross term was not isolated.

  S2  Exact permutation null for the residual Task-B accuracy.
      The same-category label is fixed by a partition of 8 tokens into 4
      unlabelled two-token categories.  There are 8!/((2!)^4 4!) = 105 such
      partitions, so the null is enumerated EXACTLY.  The observed partition
      is a member of the null, so the exact one-sided p is
          #{pi : score(pi) >= score(observed)} / 105,
      with the observed partition counted in the numerator.  No Monte Carlo,
      no +1 correction, no denominator of 104.

  S3  Full 2x2 self-pair design for Task D, with class counts per cell.
"""

from __future__ import annotations

import json
import os
import warnings

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score

import common as C

warnings.filterwarnings("ignore", category=UserWarning)
BIN = 25


# ---------------------------------------------------------------------------
# S1
# ---------------------------------------------------------------------------
def _sym_diagnostics(Bm):
    """Swap-conjugate vs plain-swap correlation, and self-pair imaginary ratio."""
    conj_c, plain_c = [], []
    for x in range(C.N_TOKENS):
        for y in range(x + 1, C.N_TOKENS):
            a, b = Bm[(x, y)], Bm[(y, x)]
            av = np.concatenate([a.real, a.imag])
            cj = np.concatenate([a.real, -a.imag])
            bv = np.concatenate([b.real, b.imag])
            conj_c.append(np.corrcoef(bv, cj)[0, 1])
            plain_c.append(np.corrcoef(bv, av)[0, 1])
    im_ratio, neg_frac = [], []
    for x in range(C.N_TOKENS):
        d = Bm[(x, x)]
        im_ratio.append(np.abs(d.imag).mean() / np.abs(d).mean())
        neg_frac.append(float((d.real < 0).mean()))
    return {
        "swap_conjugate_corr_mean": round(float(np.mean(conj_c)), 4),
        "swap_conjugate_corr_min": round(float(np.min(conj_c)), 4),
        "swap_conjugate_corr_max": round(float(np.max(conj_c)), 4),
        "plain_swap_corr_mean": round(float(np.mean(plain_c)), 4),
        "self_pair_imag_ratio": round(float(np.mean(im_ratio)), 4),
        "self_pair_frac_negative_real": round(float(np.mean(neg_frac)), 4),
    }


def s1_ideal_vs_empirical(B_per_rep):
    """Raw estimator, then global and channel-wise phase-gauge corrections.

    The phase gauge is estimated ONLY from self-pairs, where the ideal model
    predicts a positive real value; it is then applied to every pair.
    """
    Bm = {k: v.mean(axis=0) for k, v in B_per_rep.items()}
    out = {"raw": _sym_diagnostics(Bm)}

    # global gauge: one theta for the whole detector
    acc = sum(Bm[(x, x)].sum() for x in range(C.N_TOKENS))
    theta_g = np.angle(acc)
    Bg = {k: v * np.exp(-1j * theta_g) for k, v in Bm.items()}
    out["global_phase_corrected"] = _sym_diagnostics(Bg)
    out["global_theta_rad"] = round(float(theta_g), 4)

    # channel-wise gauge: one theta per detector channel
    stack = np.stack([Bm[(x, x)] for x in range(C.N_TOKENS)], axis=0).sum(axis=0)
    theta_k = np.angle(stack)
    Bk = {k: v * np.exp(-1j * theta_k) for k, v in Bm.items()}
    out["channelwise_phase_corrected"] = _sym_diagnostics(Bk)
    out["channelwise_theta_circular_spread"] = round(
        float(1 - np.abs(np.exp(1j * theta_k).mean())), 4)
    return out


# ---------------------------------------------------------------------------
# S2
# ---------------------------------------------------------------------------
def all_pairings(tokens):
    """Every partition into unordered pairs; 8 tokens -> 105 partitions."""
    if not tokens:
        yield []
        return
    a, rest = tokens[0], tokens[1:]
    for i, b in enumerate(rest):
        for tail in all_pairings(rest[:i] + rest[i + 1:]):
            yield [(a, b)] + tail


def grouped_taskB(X, sc, groups, n_splits=5):
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs = []
    for tr, te in skf.split(X, sc, groups=groups):
        if len(np.unique(sc[tr])) < 2 or len(np.unique(sc[te])) < 2:
            continue
        clf = RidgeClassifier(alpha=1.0)
        clf.fit(X[tr], sc[tr])
        accs.append(balanced_accuracy_score(sc[te], clf.predict(X[te])))
    return float(np.mean(accs)) if accs else float("nan")


def s2_exact_permutation(X, pinfo, pid):
    true_cat = {i: C.SEMANTIC_TOKENS[i]["category"] for i in range(C.N_TOKENS)}
    scores, obs_idx = [], None
    for j, pairing in enumerate(all_pairings(list(range(C.N_TOKENS)))):
        cat = {}
        for ci, (a, b) in enumerate(pairing):
            cat[a] = cat[b] = ci
        sc = np.array([int(cat[x] == cat[y]) for (x, y) in pinfo])
        scores.append(grouped_taskB(X, sc, pid))
        if all((cat[a] == cat[b]) == (true_cat[a] == true_cat[b])
               for a in range(C.N_TOKENS) for b in range(C.N_TOKENS)):
            obs_idx = j
    scores = np.array(scores)
    obs = float(scores[obs_idx])
    n_ge = int((scores >= obs).sum())          # includes the observed partition
    n_tot = int(len(scores))
    return {
        "n_partitions": n_tot,
        "observed": round(obs, 4),
        "n_at_least_observed": n_ge,
        "p_exact": round(n_ge / n_tot, 4),
        "p_exact_fraction": f"{n_ge}/{n_tot}",
        "null_mean": round(float(scores.mean()), 4),
        "null_sd": round(float(scores.std(ddof=1)), 4),
        "null_q95": round(float(np.quantile(scores, 0.95)), 4),
        "null_max": round(float(scores.max()), 4),
        "null_min": round(float(scores.min()), 4),
        "all_scores": [round(float(s), 4) for s in scores],
    }


# ---------------------------------------------------------------------------
# S3
# ---------------------------------------------------------------------------
def s3_selfpair_2x2(X, sc, pinfo):
    cats = sorted(set(C.SEMANTIC_TOKENS[i]["category"] for i in range(C.N_TOKENS)))
    out = {}
    for tr_self in (True, False):
        for te_self in (True, False):
            accs, folds = [], []
            for held in cats:
                tr, te = [], []
                for i, (x, y) in enumerate(pinfo):
                    selfp = (x == y)
                    cx = C.SEMANTIC_TOKENS[x]["category"]
                    cy = C.SEMANTIC_TOKENS[y]["category"]
                    if cx == held or cy == held:
                        if selfp and not te_self:
                            continue
                        te.append(i)
                    else:
                        if selfp and not tr_self:
                            continue
                        tr.append(i)
                tr, te = np.array(tr), np.array(te)
                if (len(tr) < 2 or len(te) < 2
                        or len(np.unique(sc[tr])) < 2 or len(np.unique(sc[te])) < 2):
                    continue
                clf = RidgeClassifier(alpha=1.0)          # identical protocol
                clf.fit(X[tr], sc[tr])
                a = balanced_accuracy_score(sc[te], clf.predict(X[te]))
                accs.append(a)
                folds.append({
                    "held_category": held,
                    "balanced_acc": round(float(a), 4),
                    "n_train_pos": int((sc[tr] == 1).sum()),
                    "n_train_neg": int((sc[tr] == 0).sum()),
                    "n_test_pos": int((sc[te] == 1).sum()),
                    "n_test_neg": int((sc[te] == 0).sum()),
                    "n_distinct_test_pos_pairs": len({pinfo[i] for i in te if sc[i] == 1}),
                })
            key = (f"train_{'with' if tr_self else 'without'}"
                   f"__test_{'with' if te_self else 'without'}")
            out[key] = {"mean_balanced_acc": round(float(np.mean(accs)), 4) if accs else None,
                        "folds": folds}
    return out


# ---------------------------------------------------------------------------
def main() -> int:
    single, blank, pair = C.load_all_data_binned(BIN)
    B_per_rep = C.compute_B_per_repeat(blank, pair)
    feats, pid, sc, cp, pinfo = C.build_features(single, B_per_rep)
    X = feats["complex_B"]
    report = {"bin": BIN, "scoring": "RidgeClassifier(alpha=1.0), balanced accuracy"}

    print("=" * 76)
    print("S1  Ideal-model identities vs the empirical estimator")
    print("=" * 76)
    r1 = s1_ideal_vs_empirical(B_per_rep)
    hdr = f"  {'gauge':28s} {'swap-conj corr':>15s} {'plain-swap':>11s} {'self |Im|/|B|':>14s}"
    print(hdr)
    for k in ("raw", "global_phase_corrected", "channelwise_phase_corrected"):
        d = r1[k]
        print(f"  {k:28s} {d['swap_conjugate_corr_mean']:15.4f} "
              f"{d['plain_swap_corr_mean']:11.4f} {d['self_pair_imag_ratio']:14.4f}")
    print(f"  self-pair fraction Re<0 (raw): {r1['raw']['self_pair_frac_negative_real']:.4f}")
    print(f"  channel-wise theta circular spread: "
          f"{r1['channelwise_theta_circular_spread']:.4f}  (0 = one common phase)")
    report["S1"] = r1

    print("\n" + "=" * 76)
    print("S2  Exact permutation null over all partitions of 8 tokens into 4 pairs")
    print("=" * 76)
    r2 = s2_exact_permutation(X, pinfo, pid)
    print(f"  partitions enumerated          : {r2['n_partitions']}  (exact, no sampling)")
    print(f"  observed semantic partition    : {r2['observed']:.4f}")
    print(f"  null mean / sd                 : {r2['null_mean']:.4f} / {r2['null_sd']:.4f}")
    print(f"  null min / q95 / max           : {r2['null_min']:.4f} / "
          f"{r2['null_q95']:.4f} / {r2['null_max']:.4f}")
    print(f"  # partitions >= observed       : {r2['p_exact_fraction']}")
    print(f"  exact one-sided p              : {r2['p_exact']:.4f}")
    report["S2"] = r2

    print("\n" + "=" * 76)
    print("S3  Task D, full 2x2 self-pair design")
    print("=" * 76)
    r3 = s3_selfpair_2x2(X, sc, pinfo)
    for k, v in r3.items():
        f0 = v["folds"][0] if v["folds"] else {}
        print(f"  {k:42s} {v['mean_balanced_acc']}"
              f"   (per fold: {f0.get('n_test_pos','?')}+/{f0.get('n_test_neg','?')}-,"
              f" {f0.get('n_distinct_test_pos_pairs','?')} distinct pos pairs)")
    report["S3"] = r3

    out = os.path.join(C.RESULTS_DIR, "structure_tests.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
