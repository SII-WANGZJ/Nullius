#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Frame loading, detector binning, and four-step phase demodulation.

Mirrors ``analyze_advantage_boundary.py`` in the authors' deposit.
"""

from __future__ import annotations

import os

import numpy as np

from .config import (CACHE_DIR, N_REPS, N_TOKENS, PHASE_FILE_LABELS,
                     SEMANTIC_DATA_DIR)


def bin_roi(roi: np.ndarray, bin_size: int) -> np.ndarray:
    """Mean-pool a square ROI into ``bin_size`` x ``bin_size`` cells."""
    h, w = roi.shape
    nh, nw = h // bin_size, w // bin_size
    cropped = roi[:nh * bin_size, :nw * bin_size]
    return cropped.reshape(nh, bin_size, nw, bin_size).mean(axis=(1, 3)).flatten()


def load_all_data_binned(bin_size: int, data_dir: str | None = None,
                         n_tokens: int = N_TOKENS, use_cache: bool = True):
    """Return ``(single, blank, pair)`` dictionaries of binned intensity frames.

    ``single[t]``          -> (N_REPS, n_channels)
    ``blank[(y, phi)]``    -> (N_REPS, n_channels)
    ``pair[(x, y, phi)]``  -> (N_REPS, n_channels)
    """
    data_dir = data_dir or SEMANTIC_DATA_DIR
    tag = f"{os.path.basename(os.path.normpath(data_dir))}_{n_tokens}t_bin{bin_size}"
    cache_path = os.path.join(CACHE_DIR, f"{tag}.npz")

    if use_cache and os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=False)
        single = {t: z[f"s{t}"] for t in range(n_tokens)}
        blank = {(y, p): z[f"b{y}_{p}"]
                 for y in range(n_tokens) for p in range(4)}
        pair = {(x, y, p): z[f"p{x}_{y}_{p}"]
                for x in range(n_tokens) for y in range(n_tokens)
                for p in range(4)}
        return single, blank, pair

    def rd(sub, name):
        return bin_roi(np.load(os.path.join(data_dir, sub, name)), bin_size)

    single = {t: np.array([rd("singles", f"single_t{t}_rep{r:02d}.npy")
                           for r in range(N_REPS)])
              for t in range(n_tokens)}

    blank = {(y, p): np.array(
        [rd("blank_controls",
            f"blank_y{y}_phi{PHASE_FILE_LABELS[p]}_rep{r:02d}.npy")
         for r in range(N_REPS)])
        for y in range(n_tokens) for p in range(4)}

    pair = {(x, y, p): np.array(
        [rd("pair_interactions",
            f"pair_x{x}_y{y}_phi{PHASE_FILE_LABELS[p]}_rep{r:02d}.npy")
         for r in range(N_REPS)])
        for x in range(n_tokens) for y in range(n_tokens) for p in range(4)}

    if use_cache:
        payload = {f"s{t}": single[t] for t in range(n_tokens)}
        payload.update({f"b{y}_{p}": blank[(y, p)]
                        for y in range(n_tokens) for p in range(4)})
        payload.update({f"p{x}_{y}_{p}": pair[(x, y, p)]
                        for x in range(n_tokens) for y in range(n_tokens)
                        for p in range(4)})
        np.savez_compressed(cache_path, **payload)

    return single, blank, pair


def _quadrature(frames, rep):
    """(I0 - I2)/4 + i (I3 - I1)/4 -- the four-step demodulation."""
    I0, I1, I2, I3 = (frames[p][rep] for p in range(4))
    return (I0 - I2) / 4.0 + 1j * (I3 - I1) / 4.0


def compute_B_per_repeat(blank, pair, n_tokens: int = N_TOKENS):
    """Complex interaction field per ordered pair and repeat, blank-subtracted."""
    Q_blank = {}
    for y in range(n_tokens):
        frames = {p: blank[(y, p)] for p in range(4)}
        Q_blank[y] = np.array([_quadrature(frames, r) for r in range(N_REPS)])

    B = {}
    for x in range(n_tokens):
        for y in range(n_tokens):
            frames = {p: pair[(x, y, p)] for p in range(4)}
            B[(x, y)] = np.array(
                [_quadrature(frames, r) - Q_blank[y][r] for r in range(N_REPS)])
    return B
