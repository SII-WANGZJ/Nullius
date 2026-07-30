# discovery-audit

Benchmark-integrity and provenance audit of the autonomous-discovery claim in
**arXiv:2604.27092**, *End-to-end autonomous scientific discovery on a real
optical platform* (Yang et al., 2026).

The audit consumes **only the authors' own public deposit**
(DOI [10.5281/zenodo.19890402](https://doi.org/10.5281/zenodo.19890402)).
No new measurements were made.

## Design principle

`common.py` is a faithful mirror of the released pipeline
(`05_result4_refine_paper/scripts/analyze_advantage_boundary.py`) — same
loading, same binning, same four-step demodulation, same `RidgeClassifier`,
same folds. Every audit result is produced by changing **exactly one**
documented thing. If the replication stage does not reproduce the published
numbers, nothing downstream is meaningful.

## Environment

```bash
conda create -n discovery-audit -y python=3.13 numpy scipy scikit-learn matplotlib
conda activate discovery-audit
```

Expects the unpacked deposit at `../SM_Source_Materials_CORE_5REPORTS/`.

## Run

```bash
python run_audit.py              # replication + B2 + B3, bins 25 and 10
python run_audit.py --bins 25    # main-manuscript operating point only
python robustness.py             # adversarial self-checks Q1-Q4
```

First run reads 1481 `.npy` frames (~8 s per bin); binned arrays are cached in
`cache/`. Results land in `results/*.json`.

## Findings

**Replication is exact.** Max deviation from published raw-input baselines:
`5e-4`. Authors' participation-ratio effective ranks independently confirmed
(3.00 at bin 25, 9.62 at bin 10, versus their stated ~3 and ~9).

| ID | Finding | Evidence |
|----|---------|----------|
| **B1** | The "semantic benchmark" has no semantic structure — the 8 tokens are seeded uniform random phase vectors; words and categories are labels attached to them. The data-generating process contains no designed semantic relation from which held-out-category generalisation could be learned, so any above-chance Task-D result must be a finite-sample accident or an evaluation cue. | `analyze_advantage_boundary.py:67-74` |
| **B2** | Task D is accounted for by self-pairs. The measurement-free identity-rule baseline `same_category := (x == y)` scores **0.750**, at or above every Task-D value obtained under the refined held-out-category protocol reproduced here. Removing self-pairs collapses Complex-B to chance: 0.7292 → **0.4979** (bin 25), 0.7396 → **0.4948** (bin 10). | `run_audit.py` stage B2 |
| **B3** | Tasks A–C are inflated by repeat leakage. Repeat correlation mean **0.9828** (97.0% above 0.97), yet repeats are split across folds. Pair-grouped scoring: Task A degenerate by construction; Task C 1.0000 → **0.0474** (below 16-class chance 0.0625); Task B 1.0000 → 0.6522 (reduced, but still above chance). | `run_audit.py` stage B3 |
| **O1** | The operator-matched control is absent. The "digital bilinear" baseline is an intensity product `zx * zy` (phase discarded); pseudo-B randomises `T`. Neither is the same-`T` digital reconstruction `(TEx)* ⊙ (TEy)`. | `analyze_advantage_boundary.py:169`, `:88-99` |
| **P1** | No backbone model or version, prompts, agent trace, intervention log or engine code anywhere in the deposit. | full-package grep, zero hits |

## Adversarial self-checks (`robustness.py`)

| | Objection | Outcome |
|---|---|---|
| Q1 | Task-C collapse is a fold-count artefact | **Refuted.** n_splits ∈ {2,4,5,8}: 100% of test classes present in training; accuracy 0.043–0.109 throughout |
| Q2 | Task-D ablation lacks power | Per-fold reported. 4 distinct positives per fold (2 self-pairs) → 2 after ablation; unablated folds constant to 4 d.p. |
| Q3 | Specific to the coarse bin = 25 readout | **Refuted.** Same picture at speckle-matched bin = 10 |
| Q4 | Leakage premise unverified | **Confirmed from data.** mean repeat correlation 0.9828 |

## What this audit does *not* claim

No evidence of fabrication. The raw frames, repeat structure, drift monitoring
and transmission-matrix data are substantial and internally consistent. The
automation achievement — sustained agent control of a real optical platform
over hundreds of steps — is untouched by these findings. The claim here is
narrow: the specific evidence offered for the specific central conclusion does
not support it.

## Layout

```
common.py        mirrored pipeline + audit-only helpers
run_audit.py     replication (R), self-pair ablation (B2), leakage ablation (B3)
robustness.py    adversarial self-checks Q1-Q4
results/         audit_results.json, robustness_results.json
cache/           binned frame cache (regenerable, not tracked)
../paper/        manuscript (audit.tex)
```
