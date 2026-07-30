# data/

**Nothing in this directory is redistributed by this repository.** Everything
here belongs to the authors of the audited work and must be obtained from the
sources below. Only this file is tracked by git; the rest is `.gitignore`d.

## What to download

| | Source |
|---|---|
| **Preprint** | [arXiv:2604.27092](https://arxiv.org/abs/2604.27092) — *End-to-end autonomous scientific discovery on a real optical platform*, Yang et al. (2026) |
| **Deposit** (required to run the audit) | [DOI 10.5281/zenodo.19890402](https://doi.org/10.5281/zenodo.19890402) — `SM_Source_Materials_CORE_5REPORTS.tar.gz`, ~1.4 GB compressed, CC BY 4.0 |

## Where to put it

Unpack the deposit here, so that this path exists:

```
data/SM_Source_Materials_CORE_5REPORTS/shared_raw_data/result4_frame_data/
```

Or leave it anywhere and point the package at it:

```bash
export NULLIUS_DEPOSIT=/wherever/SM_Source_Materials_CORE_5REPORTS
```

## Verifying you have the same bytes we did

```bash
python experiments/05_verify_inputs.py
```

This writes a SHA-256 for every consumed file to
`results/input_manifest.json`, plus a single roll-up digest over all of them.
The audit was computed against:

```
rollup SHA-256  b273f7f6c8d5bd8d7bb849e1d9908ea35a8ec171b9870b7cd2cb27b496351350
                (1543 files: 1521 frames, 17 analysis scripts, 5 deposit documents)
```

If your roll-up differs, the deposit has changed since the audit was run and
the reported numbers should be re-derived rather than assumed.

## What the deposit does and does not contain

Stated by the depositors themselves: the deposit provides result-level data,
analysis scripts and five staged reports. It explicitly **excludes** the
Qiushi Engine core system, the hardware-control software, and the formal
Supplementary Information submitted to the journal. Several of this audit's
observations about absent evidence refer to that boundary, and the manuscript
distinguishes *not present in the public materials* from *not performed*
throughout.
