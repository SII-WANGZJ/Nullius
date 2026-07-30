#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Feature families.

Audit note (finding B1): ``make_token_input`` shows that the tokens are seeded
uniform random phase vectors. The words and categories in ``SEMANTIC_TOKENS``
are labels attached to those vectors; no semantic embedding enters anywhere.

Audit note (finding O1): ``single`` holds binned camera *intensity*, so
``digital_bilinear = z(x) * z(y)`` is a real intensity product. It is not the
conjugate product of the propagated complex fields and therefore not an
operator-matched reconstruction of Complex-B.
"""

from __future__ import annotations

import numpy as np

from .config import N_REPS, N_TOKENS, SEMANTIC_TOKENS


def make_token_input(tid: int) -> np.ndarray:
    """The 36-dim complex input vector E(x), from the released seed rule."""
    seed = 10000 + tid * 7919
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0, 2 * np.pi, (6, 6))
    return np.exp(1j * phases).flatten()


TOKEN_INPUTS = [make_token_input(t) for t in range(N_TOKENS)]


def build_features(single, B_per_rep, n_tokens: int = N_TOKENS):
    """Optical feature families and the four task label vectors."""
    z_mean = {t: single[t].mean(axis=0) for t in range(n_tokens)}
    feats = {"concat": [], "digital_bilinear": [], "complex_B": []}
    pid, sc, cp, pinfo = [], [], [], []

    for x in range(n_tokens):
        for y in range(n_tokens):
            cx = SEMANTIC_TOKENS[x]["category"]
            cy = SEMANTIC_TOKENS[y]["category"]
            B = B_per_rep[(x, y)]
            for rep in range(min(N_REPS, B.shape[0])):
                feats["concat"].append(np.concatenate([z_mean[x], z_mean[y]]))
                feats["digital_bilinear"].append(z_mean[x] * z_mean[y])
                feats["complex_B"].append(
                    np.concatenate([np.real(B[rep]), np.imag(B[rep])]))
                pid.append(x * n_tokens + y)
                sc.append(int(cx == cy))
                cp.append(cx * 4 + cy)
                pinfo.append((x, y))

    return ({k: np.array(v) for k, v in feats.items()},
            np.array(pid), np.array(sc), np.array(cp), pinfo)


def build_raw_input_features(noise_scale: float = 1e-8, seed: int = 12345):
    """Baselines computed directly from E(x), E(y) with no optical propagation.

    Mirrors ``experiment_raw_input_baselines`` in the deposit, including the
    five *pseudo*-repeats: identical vectors plus 1e-8 noise. That asymmetry
    against the optical families, whose repeats carry real measurement noise,
    is documented in the manuscript.
    """
    rng = np.random.default_rng(seed)
    X = {"concat_raw": [], "diff_raw": [], "abs_diff_raw": [],
         "prod_raw": [], "conj_prod_raw": []}
    pid, sc, cp, pinfo = [], [], [], []

    for x in range(N_TOKENS):
        for y in range(N_TOKENS):
            cx = SEMANTIC_TOKENS[x]["category"]
            cy = SEMANTIC_TOKENS[y]["category"]
            Ex, Ey = TOKEN_INPUTS[x], TOKEN_INPUTS[y]

            concat_v = np.concatenate([np.real(Ex), np.imag(Ex),
                                       np.real(Ey), np.imag(Ey)])
            diff_v, prod_v = Ex - Ey, Ex * Ey
            abs_diff_v, conj_prod_v = np.abs(Ex - Ey), np.conj(Ex) * Ey

            for _ in range(N_REPS):
                n = lambda k: rng.normal(0, noise_scale, k)
                X["concat_raw"].append(concat_v + n(concat_v.shape[0]))
                X["diff_raw"].append(
                    np.concatenate([np.real(diff_v), np.imag(diff_v)]) + n(72))
                X["abs_diff_raw"].append(abs_diff_v + n(36))
                X["prod_raw"].append(
                    np.concatenate([np.real(prod_v), np.imag(prod_v)]) + n(72))
                X["conj_prod_raw"].append(
                    np.concatenate([np.real(conj_prod_v),
                                    np.imag(conj_prod_v)]) + n(72))
                pid.append(x * N_TOKENS + y)
                sc.append(int(cx == cy))
                cp.append(cx * 4 + cy)
                pinfo.append((x, y))

    return ({k: np.array(v) for k, v in X.items()},
            np.array(pid), np.array(sc), np.array(cp), pinfo)
