#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Constants and paths.

All values here mirror the released analysis scripts of arXiv:2604.27092 so
that any difference in results is attributable to a documented modification
rather than to configuration drift.
"""

from __future__ import annotations

import os

# --- experiment constants, as released --------------------------------------
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

INPUT_DIM = 36

# --- paths ------------------------------------------------------------------
PKG_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(PKG_DIR)

#: The authors' deposit (DOI 10.5281/zenodo.19890402), unpacked beside the repo.
#: Not redistributed here; see results/input_manifest.json for per-file digests.
DEPOSIT_DIR = os.environ.get(
    "NULLIUS_DEPOSIT",
    os.path.join(REPO_DIR, "SM_Source_Materials_CORE_5REPORTS"))

FRAME_DIR = os.path.join(DEPOSIT_DIR, "shared_raw_data", "result4_frame_data")
SEMANTIC_DATA_DIR = os.path.join(FRAME_DIR, "exp_complex_B_semantic")
XOR_DATA_DIR = os.path.join(FRAME_DIR, "exp_bilinear_001", "exp_bilinear_001")

RESULTS_DIR = os.path.join(REPO_DIR, "results")
CACHE_DIR = os.path.join(REPO_DIR, ".cache")
FIG_DIR = os.path.join(REPO_DIR, "paper", "figs")

for _d in (RESULTS_DIR, CACHE_DIR, FIG_DIR):
    os.makedirs(_d, exist_ok=True)


def require_deposit() -> None:
    """Fail early and legibly if the deposit is not where we expect it."""
    if not os.path.isdir(SEMANTIC_DATA_DIR):
        raise SystemExit(
            f"Deposit not found at:\n  {DEPOSIT_DIR}\n\n"
            "Download it from https://doi.org/10.5281/zenodo.19890402, unpack\n"
            "it beside this repository, or set NULLIUS_DEPOSIT to its path.")
