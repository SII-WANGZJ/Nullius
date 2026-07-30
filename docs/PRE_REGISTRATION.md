# Pre-registration: XOR showcase audit

**Frozen before `experiments/04_xor_audit.py` was run** (commit `fcff795`).

**Frozen before `xor_audit.py` was run.** The four-token XOR showcase is the
second experiment supporting Complex-B, and its logic differs from the
semantic benchmark: linear concatenation genuinely cannot express XOR, while a
product feature can. The question is not whether that identity holds — it does
— but whether the evaluation demonstrates *generalisable* interaction rather
than memorisation of a 16-entry pair table.

Structural facts established before running, from
`verify_xor_structure.py` and `token_definitions.json`:

- Label is `(x + y) % 2`, a checkerboard on **token index**. Tokens are random
  phase patterns (seeds 42, 1042, 2042, 3042); parity is not an encoded
  optical attribute.
- The label is symmetric: `label(x,y) == label(y,x)`.
- All four self-pairs `(x,x)` carry label 0 — self-pairs are deterministically
  aligned with one parity class.
- Equivalently the label is "the two tokens fall in different groups", with
  groups `{0,2}` and `{1,3}`.

Primary tests, fixed in advance:

| ID | Test | Purpose |
|----|------|---------|
| X1 | Reproduce released XOR accuracies with the authors' pipeline unchanged | Nothing downstream counts otherwise |
| X2 | Enumerate the full 4×4 label matrix, class counts, repeats | Make the label-generating process explicit |
| X3 | Sample-level vs ordered-pair-grouped splits, all feature families | Repeat leakage |
| X4 | Self-pair 2×2 ablation (train × test) | Self-pairs are 4/8 of the negatives |
| X5 | Measurement-free structural baselines (`x=y`, group identity, ordered vs unordered) | Does the classifier use the optics or the pair table? |
| X6 | Exact null over all 3 partitions of 4 tokens into 2 unlabelled groups | Is the designated parity assignment special? |
| X7 | Leave-one-token-out | Parity is an index property; this should fail unless attributes leak into the field |
| X8 | Unordered-pair grouping | `(x,y)` and `(y,x)` share a label, and Complex-B has approximate swap-conjugate structure (measured 0.60), so ordered grouping may still leak |

All three possible outcomes are to be reported: XOR also collapses; XOR
survives unordered-pair grouping (in which case the designed quadratic
relation is genuinely linearly separable from the physical feature, and we say
so); or XOR survives ordered grouping but fails unordered grouping and
leave-one-token-out, indicating reverse-pair leakage or token memorisation.

