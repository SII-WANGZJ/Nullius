#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nullius -- a reproducible audit of a claim of autonomous scientific discovery.

*Nullius in verba* -- on the word of no one. The package re-runs the analysis
released with arXiv:2604.27092 unmodified, then changes one documented thing
at a time and reports what happens.

Design rule: :mod:`nullius.data`, :mod:`nullius.features` and the released
paths in :mod:`nullius.config` mirror the deposit exactly. Anything that
differs from the released pipeline lives in :mod:`nullius.scoring` and is
named for the assumption it changes.
"""

from __future__ import annotations

__version__ = "0.2.0"

from .config import (CACHE_DIR, DEPOSIT_DIR, FIG_DIR, INPUT_DIM, N_REPS,
                     N_TOKENS, PHASE_FILE_LABELS, REPO_DIR, RESULTS_DIR,
                     ROI_H, ROI_W, SEMANTIC_DATA_DIR, SEMANTIC_TOKENS,
                     XOR_DATA_DIR, require_deposit)
from .data import bin_roi, compute_B_per_repeat, load_all_data_binned
from .features import (TOKEN_INPUTS, build_features, build_raw_input_features,
                       make_token_input)
from .scoring import (cross_split_eval, exhaustive_leave_group_out,
                      grouped_balanced_accuracy, identity_rule_taskD,
                      participation_ratio_effective_rank, run_benchmark,
                      skf_balanced_accuracy)

__all__ = [
    "__version__",
    # config
    "N_TOKENS", "N_REPS", "ROI_H", "ROI_W", "PHASE_FILE_LABELS",
    "SEMANTIC_TOKENS", "INPUT_DIM", "REPO_DIR", "DEPOSIT_DIR",
    "SEMANTIC_DATA_DIR", "XOR_DATA_DIR", "RESULTS_DIR", "CACHE_DIR",
    "FIG_DIR", "require_deposit",
    # data
    "bin_roi", "load_all_data_binned", "compute_B_per_repeat",
    # features
    "make_token_input", "TOKEN_INPUTS", "build_features",
    "build_raw_input_features",
    # scoring
    "skf_balanced_accuracy", "grouped_balanced_accuracy",
    "exhaustive_leave_group_out", "cross_split_eval", "run_benchmark",
    "identity_rule_taskD", "participation_ratio_effective_rank",
]
