# Final pass — change log and scope-consistency checklist

Commit range: `fd2bd3c` → this pass. No experimental number changed.

## §4.2 evidence status

| Change | Rationale |
|---|---|
| Added: "This section is a scoped bibliographic comparison based on the principal methods and claims associated with these works; it is not an exhaustive, apparatus-level priority analysis." | None of the five works was read in full text. The propositions attached to them are their headline contributions only. |
| Added: "Each work below is cited for its headline contribution only, and we attach no finer device correspondence to any of them." | Prevents a reader inferring formula- or apparatus-level equivalence from a search-level citation. |

Per-citation check — each cited only for its headline claim:

| Work | Proposition it carries here | Finer meaning attached? |
|---|---|---|
| Bruning 1974 | stepped reference phase + per-detector intensity → interference term | no |
| Yamaguchi & Zhang 1997 | complex field recoverable from intensity-only frames | no |
| Popoff 2010 | phase-stepped per-channel complex recovery through a scattering operator | no |
| Hamerly 2019 | square-law detection as a route to optical multiplication | no |
| Rafayelyan 2020 | fixed scattering medium as a random feature map | no |

## Abstract

Rewritten to four layers plus a separated autonomy sentence.

| Before | After | Reason |
|---|---|---|
| "the reported cross-split generalisation is accounted for by the presence of self-paired samples, and the remaining task accuracies are inflated by near-duplicate repeats" | "both experiments … are scored under sample-level splits that place repeated measurements of the same physical pair in training and test" | Describes the design fact; does not assert leakage as sole cause. |
| "category-pair classification falls below chance, and same-category classification is not significant…" (enumerated point estimates) | folded into "do not distinguish a relation-specific signal from pair identity together with structural cues" | Point estimates no longer presented as significance statements in the abstract; only the exact permutation result ($102/105$) is quoted, because only it is a test. |
| "what they do not support is the … reading placed on them" | "The benchmarks remain valid demonstrations that repeated measurements from a fixed, previously observed pair library are decodable; they do not, under the released evaluation protocol, establish relation-specific structure, semantic organisation, or transfer to unseen pairs." | Removes the adversarial register; states both halves symmetrically. |
| autonomy implied by benchmark findings | separate sentence, ending "independently of the benchmark findings" | Benchmark failure must not read as disproving autonomy. |
| — | added "the two preceding studies in the target work were not audited" | Blocks extrapolation to Results 2/3 at the point of first contact. |
| S1 diagnostics, prior-art position | absent from abstract | Neither is primary evidence. |

## Findings table

| Check | Status |
|---|---|
| Claims are locatable statements of the target work | yes; none strengthened on the authors' behalf |
| Finding column separated from interpretation | column renamed "Outcome (finding)"; interpretation confined to the final column and body text |
| Chance-level statements backed by a test | Task B row cites the exact permutation ($102/105$); Task D row states the ablation condition rather than a chance claim; XOR row states "not retained", not "at chance" |
| Final column preserves a minimum standing conclusion | yes, all rows |
| Results 2/3 present and marked | **row added**, marked `not audited`, "no conclusion drawn either way" |
| Physical novelty row | **row added**: "demodulation ingredients have established precedent; priority of the specific combination unresolved by this audit" |

## §6 and closing judgment

Added a three-way separation before the evidence list:

1. **Direct findings** — each established by an ablation or exact test:
   evaluation-unit mismatch; designated assignment not exceptional; Task-D
   self-pair shortcut; XOR mirror-pair reuse.
2. **Absences in the public record** — explicitly "not findings about the
   underlying work": operator-matched control, backbone/prompt disclosure,
   agent trace, intervention log, architectural ablation, selection timing.
3. **Outside scope** — Results 2/3, hardware performance, exhaustive priority
   determination.

Closing judgment now reads: a failure of evidential attribution, not a failure
to produce repeatable optical measurements. Final clause phrased as
inability to distinguish among model prior knowledge, agent-level synthesis
and human steering — not as evidence that the interpretation came from
training data. No summary sentence calls all three categories "failures".

## Causal-language sweep

| Location | Before | After | Basis |
|---|---|---|---|
| §3.3 heading | "Tasks A–C are inflated by repeat leakage" | "Tasks A–C depend on repeat-level splitting" | The manipulation shows dependence; it does not isolate leakage as sole cause. |
| Fig. 2 caption | same | same | as above |
| §3.2 heading and Fig. 1 caption | "Task D is accounted for by self-pairs" | unchanged | The $2\times2$ ablation directly establishes the mechanism: above chance only with self-pairs on both sides. |

## Scope-consistency checklist

- [x] Abstract, Table 1 and §6 assert the same tier of conclusion
- [x] No claim about Results 2 or 3 anywhere in the paper
- [x] No point estimate presented as a significance result
- [x] Every "not established" distinguished from "absent"
- [x] Autonomy attribution decoupled from benchmark findings in all three exits
- [x] S1 confined to Appendix A and flagged diagnostic
- [x] Prior art confined to §4.2 with its scope stated in-line
- [x] Fabrication mentioned only in the negative, three times, all scoped to inspected data
- [x] Cross-references resolve; figures and tables referenced before appearing
- [x] Exploratory-versus-pre-specified status stated once, in §2, not repeated

## Status

Technically frozen. Remaining work is author confirmation, the repository URL,
formatting and submission strategy. Two items stay open by choice and are
recorded as such in the paper: full-text verification of the phase-shifting
joint-transform-correlator literature (the only source that could change the
strength of §4.2), and the bare-model prior-availability control (listed in §6
as evidence that would help, not as a gap in the present findings).
