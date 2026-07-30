# nullius

*Nullius in verba* — on the word of no one.

A reproducible audit of the computational claims in **arXiv:2604.27092**,
*End-to-end autonomous scientific discovery on a real optical platform*.

The audit consumes **only the authors' own public deposit**
([DOI 10.5281/zenodo.19890402](https://doi.org/10.5281/zenodo.19890402)).
No new measurements were made, and the deposit is **not redistributed** here —
`results/input_manifest.json` records a SHA-256 for every file consumed.

Manuscript: [`paper/audit.pdf`](paper/audit.pdf)

---

## Design rule

`src/nullius/data.py` and `src/nullius/features.py` mirror the released
pipeline (`05_result4_refine_paper/scripts/analyze_advantage_boundary.py`)
exactly — same loading, binning, four-step demodulation, classifier and folds.
Everything that departs from it lives in `src/nullius/scoring.py` and is named
for the assumption it changes.

**Replication runs first.** If stage R does not reproduce the released
accuracies, nothing downstream means anything. It currently reproduces them to
within `5e-4`.

## Quick start

```bash
conda env create -f environment.yml
conda activate discovery-audit

# unpack the Zenodo deposit beside this repo, or:
export NULLIUS_DEPOSIT=/path/to/SM_Source_Materials_CORE_5REPORTS

python experiments/01_replicate_and_ablate.py   # replication + B2 + B3
python experiments/02_robustness.py             # adversarial self-checks
python experiments/03_structure_tests.py        # S1, exact permutation, 2x2
python experiments/04_xor_audit.py              # pre-registered XOR audit
python experiments/05_verify_inputs.py          # SHA-256 of every input
python experiments/06_make_figures.py           # figures for the manuscript
```

First run reads ~1500 `.npy` frames (~8 s per bin size); binned arrays are
cached in `.cache/`. Results land in `results/`.

## Findings

Replication is exact: maximum deviation from the released raw-input baselines
is `5e-4`, and the authors' participation-ratio effective ranks are recovered
independently (3.00 at bin 25, 9.62 at bin 10, against their stated ~3 and ~9).

| ID | Finding | Where |
|----|---------|-------|
| **B1** | The eight "semantic" tokens are seeded uniform random phase vectors; the words and categories are labels attached to them. The authors' own Report 4 lists this mapping as arbitrary. | `features.make_token_input` |
| **B2** | Task D is accounted for by self-pairs. The measurement-free identity rule `same_category := (x == y)` scores **0.750**. Removing self-pairs: 0.7292 → **0.4979** (bin 25), 0.7396 → **0.4948** (bin 10). | exp. 01, 03 |
| **B3** | Tasks A–C depend on repeat-level splitting. Repeat correlation mean **0.9828**, yet repeats span folds. Pair-grouped: Task C 1.0000 → **0.0474** (below 16-class chance 0.0625); Task A degenerate by construction. | exp. 01, 02 |
| **B4** | The residual pair-grouped Task-B accuracy is not exceptional: exact enumeration of all 105 admissible category assignments gives **p = 102/105**. | exp. 03 |
| **B5** | The Task-D shortcut is learned, not merely present: above chance only with self-pairs in **both** train and test. | exp. 03 |
| **B6** | The XOR showcase reproduces at 1.0000 sample-level and is not retained under any pair-respecting split. The digital-bilinear 0.8125 is mirror-pair leakage — `z(x)⊙z(y)` is symmetric, so the held-out pair's mirror is an identical training vector; grouping mirrors together collapses it to 0.0625. | exp. 04 |
| **O1** | The operator-matched control is absent. The released "digital bilinear" is an intensity product; pseudo-B randomises `T`. Neither reconstructs `A_x* ⊙ A_y` from separately measured complex fields. | `features.build_features` |
| **P1** | No backbone model, prompts, agent trace, intervention log or engine code anywhere in the deposit. | full-package grep |

## Pre-registration status

The XOR audit (exp. 04) was **pre-specified**: the ten tests, the splitting
regimes and the commitment to publish all three possible outcomes were written
into [`docs/PRE_REGISTRATION.md`](docs/PRE_REGISTRATION.md) and committed
before the analysis ran. The semantic-benchmark analyses were **exploratory**,
developed while reading the released code. The manuscript says so in §2.

## What this audit does *not* claim

No evidence of fabrication. The raw frames, repeat structure and drift
monitoring are substantial and mutually consistent. The reported hardware
automation may be substantial and is **not evaluated here**. The two preceding
studies in the target work (transmission-matrix reproduction, majorization
order) were **not audited**, and no conclusion is drawn about them.

## Layout

```
src/nullius/       config · data · features · scoring · figures
experiments/       numbered, runnable, one concern each
docs/              pre-registration · prior-art matrix · rebuttal stress test · change log
results/           JSON output; *.reference.json are pre-refactor snapshots
paper/             manuscript source and figures
```

## Licence

Code: MIT. Manuscript text and figures: CC BY 4.0.
The authors' deposit is CC BY 4.0 and is not included here.
