# Adversarial rebuttal memo — NOT part of the manuscript

Written from the original authors' position, as strongly as it can honestly be
put. For each objection: whether the manuscript already answers it, and where.
An objection that survives is a defect to fix, not a point to argue harder.

Status key: **ANSWERED** (text exists) · **PARTIAL** (weak or scattered) ·
**OPEN** (no answer in the paper)

---

### R1. "We never claimed unseen-pair generalisation."

*Our tasks establish that the measured field is stable and linearly decodable
on a fully measured pair library. The staged reports say "under the tested
evaluation protocol". The audit attacks a claim we did not make.*

**ANSWERED** — §6.6 states this position, accepts it, and shows what follows:
the manuscript's own interpretation (semantic relational structure, pairwise
relational computation, attention analogy) requires transfer, which the
evaluation does not test. The summary table's final column concedes
observed-pair decodability explicitly.

*Residual risk:* if the authors retreat fully to "decodability only", the
audit's benchmark findings stand but become less damaging. The paper must
therefore keep the link to the manuscript's interpretive claims visible, not
just the benchmark mechanics. Currently that link appears in §6.6 and §11 but
not in the abstract. **Consider adding one abstract sentence.**

### R2. "Sample-level CV was a measurement-stability check, not a generalisation claim."

**ANSWERED** — same section. Note we independently confirm the stability
(repeat correlation 0.9828), so we agree with the premise.

### R3. "XOR is a didactic illustration of bilinear expressivity, not a generalisation experiment."

**ANSWERED** — §4.6 states the mathematical identity is not in dispute, and
confines the finding to what the evaluation can separate.

### R4. "The supplement already acknowledges an arbitrary token mapping."

**PARTIAL** — §4.1 shows the tokens are seeded random phase vectors and draws
the consequence for Task D. It does not quote the authors' own acknowledgment.
Quoting it would strengthen, not weaken, our position: it shows the limitation
was known and the main-text framing nevertheless describes the benchmark as
semantic. **Locate and cite the acknowledgment.**

### R5. "Complex-B is not exactly swap-symmetric, so the mirror pair is not a duplicate."

**ANSWERED** — §4.6 gives the measured swap-conjugate correlation (0.60) and
describes the leakage as partial, contrasting it with the digital bilinear,
which is algebraically exact and collapses 0.8125 → 0.0625. We do not claim
Complex-B mirrors are duplicates.

### R6. "Grouped CV on 16 pairs has enormous variance; your 0.5000 is one draw."

**ANSWERED** — this is why exhaustive leave-one-group-out was added, pooling
all out-of-fold predictions. §4.6 also declines to interpret the non-monotone
ordering and attaches no uncertainty band.

*Residual risk:* the exhaustive estimates are themselves artefact-prone at this
size (concat = 0.0000). The paper says so. Do not strengthen beyond "no
above-chance transfer observed".

### R7. "Attention analogy was stated as 'analogous to a core operation', not as a full attention implementation."

**ANSWERED** — §8 concedes the weak mathematical form explicitly before
listing what is absent. The conclusion is scoped to "not attention hardware".

### R8. "Effective rank 3.00 is a property of this configuration, not a hardware limit."

**ANSWERED** — §9 states exactly this, twice, and calls the quantity a
property of the observed feature ensemble.

### R9. "Your S1 deviation may just mean the two arms differ; that does not impugn the measurement."

**ANSWERED** — §7 draws no conclusion about cause, lists the open
explanations, and notes that $T_x \neq T_y$ would *weaken* one of our own
lines of criticism. §10 repeats the bound.

### R10. "The operator-matched control exists in the Supplementary Information you did not see."

**ANSWERED** — §10 distinguishes "not present in the public materials" from
"not performed", and names the deposit's own scope statement.

*Residual risk:* if the SI does contain it, §6 is materially weakened. This is
the single largest exposure in the paper, and it is correctly flagged rather
than hidden.

### R11. "You audited one of three studies and are generalising."

**ANSWERED** — §10 scopes this and states no conclusion is drawn about
Results 2 and 3.

### R12. "Resource counts were context, not evidence of architectural efficacy."

**PARTIAL** — §11.1 says the counts are resource counts, which concedes the
point, but then the ablation critique stands independently. Fine as is.

### R13. "You allege misconduct by implication."

**ANSWERED** — §11 and the abstract both state no fabrication was found, and
the automation is explicitly not evaluated rather than dismissed.

### R14. "Your own audit is adaptive: you searched for the split that breaks it."

**PARTIAL** — the XOR plan (X1–X10) was pre-registered in the repository
README before running, which answers it for that section. The semantic
benchmark findings were **not** pre-registered in the same way. **The paper
should say which analyses were pre-specified and which were exploratory.**
This is the objection most likely to land.

---

## Actions this memo generates

1. **Add a pre-registration statement** distinguishing the pre-specified XOR
   plan from the exploratory semantic-benchmark analysis. (R14 — highest
   priority; we apply this standard to the authors in PROBE-E.)
2. Locate and cite the authors' own acknowledgment of the arbitrary token
   mapping. (R4)
3. Consider one abstract sentence tying the benchmark findings to the
   manuscript's interpretive claims. (R1)

Nothing in this memo requires new experiments.
