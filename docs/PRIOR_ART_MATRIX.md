# Prior-art audit — bounded scope

Answers exactly two questions, and nothing else:

**Q1.** Are four-step phase shifting, square-law detection, and recovery of a
complex cross term from intensity measurements established prior art?

**Q2.** Relative to optical correlators, bilinear optical processors and
photonic attention, what does a fixed optical bilinear feature add?

Status of this file: **first pass, partly unverified.** Entries marked
`[abstract-level]` were located by search and their bibliographic details
confirmed, but the device correspondence has **not** been checked against the
full text. Those must be read before anything from them enters §4.

---

## Tier 1 — Exact precedent (measurement protocol and recovered quantity essentially the same)

| Work | Year | Input structure | Optical operation | Detection / demodulation | Quantity recovered | Same as Qiushi | Key difference |
|---|---:|---|---|---|---|---|---|
| Bruning et al., *Appl. Opt.* **13**(11), 2693–2703, [doi:10.1364/AO.13.002693](https://doi.org/10.1364/AO.13.002693) | 1974 | single wavefront vs reference arm | free-space interferometer, piezo-stepped reference | intensity on a 32×32 detector array, stepped reference phase, sine–cosine fit | phase of the interference term at each detector point | **the demodulation principle**: stepped reference phase + per-pixel intensity → complex term | one signal arm, not two independently encoded inputs; goal is surface metrology, not a pair feature |
| Yamaguchi & Zhang, *Opt. Lett.* **22**(16), 1268–1270, [doi:10.1364/OL.22.001268](https://doi.org/10.1364/OL.22.001268) | 1997 | object wave vs reference wave | free-space, phase-shifted reference | phase-shifting interferometry on intensity frames | **full complex amplitude** at the detector plane | recovery of a complex field from intensity-only measurements via phase stepping | reference-vs-object, not input-vs-input; no scattering operator between |

**Q1 verdict.** Established. Recovering a complex interference term from a
small set of phase-stepped intensity frames is standard and long-predates the
target work. This matches the authors' own characterisation of phase-stepping
interferometry as "a mature technique".

**What Tier 1 does *not* settle.** In both, one arm is a *reference*. In the
target work both arms carry encoded data and the medium acts on both. Whether
that specific two-data-input configuration through a fixed scattering operator
has direct precedent is the open question, and Tier 2 is where it would live.

---

## Tier 2 — Closely related precedent (phase stepping or optical correlation, different input/output/purpose)

| Work | Year | Input structure | Optical operation | Detection / demodulation | Quantity recovered | Same as Qiushi | Key difference |
|---|---:|---|---|---|---|---|---|
| Weaver & Goodman, *Appl. Opt.* **5**(7), 1248–1249, [doi:10.1364/AO.5.001248](https://doi.org/10.1364/AO.5.001248) | 1966 | **two data inputs side by side** in one input plane | Fourier transform by lens | square-law detection of the **joint** power spectrum, then a second transform | cross-correlation of the two inputs | **two simultaneously present data inputs; the wanted term is the interference cross term of the joint intensity** | fixed Fourier operator, not a random scattering matrix; output is a correlation plane, not a per-channel feature vector; no per-channel phase demodulation |
| Phase-shifting joint-transform correlators (several; e.g. phase-encoded and Mach–Zehnder JTC variants, *Appl. Opt.* / *Opt. Commun.*, 1990s–2000s) `[abstract-level]` | 1990s– | two data inputs | Fourier / MZ interferometer | **phase-stepped** joint power spectrum to suppress DC and conjugate terms | isolated cross-correlation term | **phase stepping applied specifically to isolate the two-input cross term** | correlation-plane output; no scattering medium; not framed as a learned or computational feature |
| Popoff et al., *Phys. Rev. Lett.* **104**, 100601, [doi:10.1103/PhysRevLett.104.100601](https://doi.org/10.1103/PhysRevLett.104.100601) | 2010 | one encoded input + internal reference | **random scattering medium** | phase-stepped intensity, per-output-channel | complex transmission matrix entries | scattering operator + phase-stepped per-channel complex recovery | single input; recovers $T$ itself, not a two-input cross term |
| Hamerly et al., *Phys. Rev. X* **9**, 021032, [doi:10.1103/PhysRevX.9.021032](https://doi.org/10.1103/PhysRevX.9.021032) | 2019 | two encoded fields | free-space / fibre | **homodyne (square-law) detection** | inner product of the two fields | two data inputs, product obtained by square-law detection | balanced homodyne for a scalar inner product; no random medium; no per-channel bilinear family |
| Rafayelyan et al., *Phys. Rev. X* **10**, 041037, [doi:10.1103/PhysRevX.10.041037](https://doi.org/10.1103/PhysRevX.10.041037) | 2020 | single encoded input | random scattering medium | intensity | random features for reservoir computing | scattering medium as a fixed random feature generator | single-input; intensity only, no complex cross term |

**Reading of Tier 2.** The three ingredients of the target mechanism each have
clear precedent, and *pairs* of them are combined in prior work: two data
inputs + square-law cross term (Weaver & Goodman; and with explicit phase
stepping in the JTC variants), scattering medium + phase-stepped per-channel
complex recovery (Popoff), two inputs + product by square-law detection
(Hamerly). What we did **not** find is a single prior work combining all three
in the target's configuration — two independently encoded data inputs through
one fixed random operator with per-channel four-step demodulation yielding a
family of channel-wise bilinear forms.

That absence is a **search result, not a proof of novelty**, and this search
was neither exhaustive nor full-text for the `[abstract-level]` rows.

---

## Tier 3 — Context only (bounds the "attention-like" claim; not precedent for the mechanism)

| Work | Year | What it implements | Why it is context, not precedent |
|---|---:|---|---|
| Vaswani et al., NeurIPS 30 | 2017 | the attention operation being invoked | defines the target of the analogy |
| Optronic vision transformer (SLM + lens matrix multiplication), *Optik* `[abstract-level]` | 2023 | transformer blocks in an optical neural network | implements the architecture, not a scattering cross term |
| Lightening-Transformer, IEEE HPCA `[abstract-level]` | 2024 | photonic transformer accelerator, dynamic matmul for attention | integrated photonics, programmable; opposite of a fixed random kernel |
| Optical-frequency-comb multi-head attention chip `[abstract-level]` | 2024 | self-attention with reported energy figures | supplies the throughput/energy comparison the target work does not make |

**Q2 verdict.** Photonic attention already exists as programmable, integrated
hardware that implements the full operation with measured energy figures. A
fixed, non-programmable, random per-channel bilinear family is a different
kind of object. It is not a step toward those systems on the axis they are
measured by, and the target work reports none of the metrics they report.

---

## Consequence for the manuscript's novelty wording

Given the above, the sentence the audit may support is the **second** of the
two options considered, not the first:

> The public materials do not distinguish the reported mechanism from the
> existing phase-shifting-interferometry and optical-correlation literature,
> in which two-input cross terms are isolated by phase stepping and complex
> fields are recovered from intensity. Physical novelty is therefore not
> established. We do not assert that no aspect is new: the specific
> combination — two independently encoded inputs through one fixed random
> operator, demodulated per channel into a family of bilinear forms — was not
> located in this search, and a systematic novelty determination would require
> a full-text prior-art review we have not performed.

Two things this file explicitly does **not** license:

1. Concluding "the mechanism is not new" from the authors' own remark that
   phase-stepping interferometry is mature. A mature demodulation method and a
   novel computational use are separate questions.
2. Treating mathematical similarity as device identity. Weaver & Goodman and
   the JTC variants share the *algebraic* structure (cross term of a joint
   intensity) but differ in operator, output representation and purpose.

---

## Outstanding before any of this enters §4

- [ ] Read Weaver & Goodman full text; confirm whether the recovered quantity
      is the correlation plane only, or per-channel cross terms.
- [ ] Obtain and read at least one phase-shifting JTC paper in full; establish
      whether four-step demodulation of a two-input joint spectrum is stated
      there in the form used here.
- [ ] Confirm the Optronic ViT and comb-attention device details from full
      text before citing them as the comparison class.
- [ ] Decide whether Tier 3 belongs in §4 (attention scope) or §6 (evidence
      needed); it is not prior art for the mechanism either way.
