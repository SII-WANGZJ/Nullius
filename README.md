<h1 align="center">nullius</h1>

<p align="center">
  <strong>A reproducible audit of the computational claims in arXiv:2604.27092</strong><br>
  <em>End-to-end autonomous scientific discovery on a real optical platform</em>
</p>

<p align="center">
  <em>Nullius in verba</em> — on the word of no one.<br>
  <sub>The motto of the Royal Society: take nobody's word for it, check the thing yourself.</sub>
</p>

<p align="center">
  Zhengjie Wang<br>
  <sub>Shanghai Innovation Institute</sub><br>
  <sub>wangzhengjie@sii.edu.cn</sub>
</p>

<p align="center">
  <a href="paper/nullius.pdf"><img alt="manuscript" src="https://img.shields.io/badge/manuscript-PDF-1b4b91"></a>
  <a href="https://arxiv.org/abs/2604.27092"><img alt="target" src="https://img.shields.io/badge/audited-arXiv%3A2604.27092-6b6b6b"></a>
  <a href="https://doi.org/10.5281/zenodo.19890402"><img alt="deposit" src="https://img.shields.io/badge/deposit-10.5281%2Fzenodo.19890402-6b6b6b"></a>
  <a href="LICENSE"><img alt="licence" src="https://img.shields.io/badge/code-MIT-2a7f4f"></a>
</p>

---

A reproducible audit of the computational claims in **arXiv:2604.27092**,
*End-to-end autonomous scientific discovery on a real optical platform* — a
paper reporting that an LLM agent autonomously proposed and experimentally
validated a previously unreported optical mechanism.

**The audit uses nothing but the authors' own published data and analysis
code.** No new measurements. The deposit is not redistributed here; every file
consumed is recorded with its SHA-256 in
[`results/input_manifest.json`](results/input_manifest.json).

## The short version

The released accuracies reproduce **exactly** — maximum deviation `5e-4`
across every feature family and task. We also independently recover the
authors' own participation-ratio effective ranks (3.00 and 9.62 against their
stated ~3 and ~9). So nothing below is a pipeline discrepancy.

Both experiments carrying the paper's computational conclusion are then scored
under splits that place the **five repeated measurements of the same physical
pair on both sides of the train/test boundary**, at a measured repeat
correlation of 0.9828. Change the unit of generalisation from the camera
repeat to the physical pair and:

| | released split | pair-respecting split |
|---|---|---|
| Task C (16-way category pair) | 1.0000 | **0.0474** — at or below the 0.0625 chance level |
| Task A (16/64-way pair identity) | 1.0000 | **degenerate by construction** — each class is one group |
| XOR showcase | 1.0000 | **not retained** under any pair-respecting split |
| Task D (held-out category, refined protocol) | 0.7292 | **0.4979** once self-pairs are removed |

These are point estimates against the stated chance reference; no significance
is claimed for any of them, and the test sets differ in size and dependence
structure, so they are not pooled.

For Task D a measurement-free rule — *same category if and only if `x == y`*,
using the pair labels and no optical data whatsoever — scores **0.750**, at or
above every value the optical feature attains under the refined
held-out-category protocol reported here. We do not extend that comparison to
the main manuscript's different fixed-holdout protocol, nor to the higher
values (0.772–0.780) the supplement obtains under searched sparse-tap and
output-budget configurations.

## Verify it yourself

That is the point of the repository, and it takes about two minutes.

```bash
pip install -r requirements.txt          # or -r requirements-lock.txt for the
                                         # exact versions the numbers were run on

# put the Zenodo deposit under data/  (see data/README.md), then:
python experiments/01_replicate_and_ablate.py
```

Stage R prints the replication check against the released numbers. Stages B2
and B3 then change one documented thing each and print what happens. The
remaining experiments run the same way:

| script | what it does |
|---|---|
| `01_replicate_and_ablate.py` | replication, self-pair ablation, repeat-leakage ablation |
| `02_robustness.py` | adversarial self-checks: fold-count sweep, per-fold detail, repeat correlation |
| `03_structure_tests.py` | exact permutation null, full 2×2 self-pair design, phase-gauge diagnostics |
| `04_xor_audit.py` | the pre-registered XOR audit (X1–X10) |
| `05_verify_inputs.py` | SHA-256 of every consumed input, plus a roll-up digest |
| `06_make_figures.py` | regenerates every figure in the manuscript from `results/` |

First run reads ~1500 `.npy` frames (~8 s per bin size); binned arrays cache to
`.cache/`.

## Design rule

`nullius/data.py` and `nullius/features.py` **mirror the released pipeline
exactly** — same loading, binning, four-step demodulation, classifier and
folds as `analyze_advantage_boundary.py` in the deposit. Everything that
departs from it lives in `nullius/scoring.py` and is named for the assumption
it changes.

Replication runs first, by construction. If it fails, nothing downstream
means anything.

## Findings

| ID | Finding |
|----|---------|
| **B1** | The eight "semantic" tokens are seeded uniform random phase vectors; the words and categories are labels attached to them. The authors' Report 4 lists this mapping as arbitrary — we take that premise from them and differ on the inference drawn from it. |
| **B2** | Task D is accounted for by self-pairs: above chance only when self-pairs are present in **both** training and test. |
| **B3** | Tasks A–C depend on repeat-level splitting, relative to a claim of out-of-pair or semantic generalisation. Repeat correlation mean 0.9828, yet repeats span folds. |
| **B4** | The residual pair-grouped Task-B accuracy is not exceptional. Enumerating **all 105** admissible category assignments exactly gives **p = 102/105**. |
| **B5** | The Task-D shortcut is learned, not merely present. |
| **B6** | The XOR showcase tests repeat recognition. Its digital-bilinear baseline scores 0.8125 through **mirror-pair leakage** — `z(x)⊙z(y)` is symmetric, so a held-out pair's mirror is a numerically identical training vector; grouping mirrors together collapses it to 0.0625. |
| **O1** | The operator-matched control is absent. The released "digital bilinear" is an intensity product; pseudo-B randomises `T`. Neither reconstructs `A_x* ⊙ A_y` from separately measured complex fields. |
| **P1** | No backbone model, prompts, agent trace, intervention log or engine code appears anywhere in the deposit. |

## Pre-registration status

We ask the audited work for pre-registration, so we state our own.

The **XOR audit was pre-specified**: the ten tests, the splitting regimes, and
the commitment to publish all three possible outcomes were written into
[`docs/PRE_REGISTRATION.md`](docs/PRE_REGISTRATION.md) and committed before it
ran, at pre-analysis commit `fcff795`. The semantic-benchmark analyses were
**exploratory**, developed while reading the released code. The manuscript
says so, in §2.

## What this audit does not claim

- **No evidence of fabrication.** The raw frames, repeat structure and drift
  monitoring are substantial and mutually consistent.
- **The hardware automation is not evaluated here.** It may well be
  substantial; the materials needed to assess it are not public and we did not
  audit it.
- **Results 2 and 3** of the target work — the transmission-matrix
  reproduction and the majorization-order study — were **not audited**, and no
  conclusion is drawn about them either way.
- **Physical novelty is *not established*, which is not the same as absent.**
  The demodulation ingredients have long precedent; the priority of the
  specific combination was not resolved by this scoped audit. See
  [`docs/PRIOR_ART_MATRIX.md`](docs/PRIOR_ART_MATRIX.md).

The findings are about evidential attribution. The inspected Result-4
measurements are internally consistent and highly correlated across repeats;
the hardware platform as a whole was not audited.

## Layout

```
nullius/        config · data · features · scoring   (mirror vs. modification, separated)
experiments/    numbered, runnable, one concern each
data/           where the authors' deposit goes; only its README is tracked
docs/           pre-registration · prior-art matrix · rebuttal stress test · change log
results/        JSON output; *.reference.json are pre-refactor snapshots
paper/          manuscript source, figures, compiled PDF
```

[`docs/REBUTTAL_STRESS_TEST.md`](docs/REBUTTAL_STRESS_TEST.md) argues the
authors' side as strongly as it can be put, and marks which objections the
manuscript already answers. It is not part of the paper.

## Citing

See [`CITATION.cff`](CITATION.cff). Please cite the target preprint and the
Zenodo deposit alongside this repository.

## Licence

Code MIT, manuscript text and figures CC BY 4.0. The audited deposit
(CC BY 4.0) is not included and must be obtained from Zenodo.
