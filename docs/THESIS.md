# CAR — thesis plan (measurement framing)

Supersedes the framing in `CAR_Project_Deep_Dive_Upgraded.pdf`. The original
document proposed a method; the evidence supports a measurement study instead.
This is the working structure, the claim-to-evidence map, and the remaining
work.

---

## Thesis statement

> Conformal machinery applied to multi-step reasoning certifies whether a step
> is **locally valid** — that is what a verifier reports and therefore what can
> be calibrated. In multi-step reasoning that is not the quantity of interest.
> We measure the gap: on GSM8K, 69.8% of globally-wrong steps are
> arithmetically perfect, wrong only because a premise was, and no sound
> deterministic verifier can see any of them. We show the gap is governed by
> **verifier reach** rather than step uncertainty, that the resulting optimal
> allocation contradicts the standard "verify early" intuition, and that the
> benchmark most used for this work cannot exhibit the phenomenon at all.

One measurable gap. One controlling variable. One benchmark correction.

---

## What this is and is not

**Is:** an empirical characterisation of where step-level verification helps,
where it provably cannot, and what to measure before designing a gate.

**Is not:** a new calibration method. The censored-feedback machinery is
implemented and used, but it is cited to Khosravi & Huo
([arXiv:2605.20270](https://arxiv.org/abs/2605.20270), Thm E.1), not claimed.

Stating this plainly in the introduction is a strength, not a concession. A
reviewer who finds the scoop themselves will discount everything else.

---

## Chapter structure

| ch | content | evidence status |
|---|---|---|
| 1 | Introduction — the local/global gap, thesis statement, contributions | written |
| 2 | Background — conformal prediction, risk control, PRMs, selective labels, propagation | references verified |
| 3 | Framework — step schema, local vs global correctness, the Markov model, verifier reach | implemented + tested |
| 4 | **Measuring the gap** — Math-Shepherd, 93k steps, stratification | **done** |
| 5 | **Measuring verifier reach** — retrieval vs calculator scope | **NOT STARTED — the critical gap** |
| 6 | Allocation — what follows from reach; refutation of "verify early" | simulation done, needs real-model confirmation |
| 7 | Feasibility — the Kotte floor with measured μ; what α is attainable | done |
| 8 | Benchmark analysis — StrategyQA has no propagation headroom | done |
| 9 | Limitations, negative results, conclusion | ongoing |

Chapters 4, 7 and 8 are complete on measured data. Chapter 5 is the one that
decides whether this is a good thesis or a merely adequate one.

---

## Claim → evidence map

| claim | evidence | source | status |
|---|---|---|---|
| C1 — 69.8% of wrong steps are locally valid | 93,129 steps, stratified | Math-Shepherd | measured |
| C2 — corruption is near-absorbing | 95.9% persistence; 0/25,971 recovered | Math-Shepherd | measured |
| C2b — local risk does not track final error | local pinned ~0.15, final 0.73→0.27 | simulation | simulated |
| C3 — verifier reach controls the outcome | ~2× error at fixed budget | simulation | **needs ch. 5** |
| C4 — "verify early" is false | chains, 5 DAG families, 2 real corpora | simulation + real graphs | measured |
| C4b — later steps are harder | corr(pos, local err) = +0.950 | Math-Shepherd | measured |
| C5 — α = 0.10 costs 32% of budget | μ = 0.3908 + Kotte Prop. 3 | measured + cited | measured |
| C6 — StrategyQA has no headroom | 72.9% one hop; 11.2% intermediate | 2272 annotated graphs | measured |
| local error rate ≈ 0.10 | post-stratified, robust 0.091–0.108 | Math-Shepherd | measured |

Everything marked *simulated* rests on the propagation model in
`src/car/propagation.py`, whose closed form is validated against simulation to
within 0.015 over 12 configurations. That is defensible as modelling, but it is
not measurement, and the thesis must not blur the two.

---

## The one experiment that matters most

**Measure verifier scope.**

Every allocation result treats `scope` — the probability that a verifier
detects inherited corruption — as a free parameter. It is the variable C3 says
controls everything, and no published work reports it.

Design:

1. Take GSM8K solutions containing a known local error at step *k* (available
   from Math-Shepherd, or injected deliberately).
2. Present step *k+d* — locally valid, resting on the corrupted premise — to
   each verifier in turn.
3. Measure detection rate as a function of *d*.

Verifiers to compare:

| verifier | expected scope | why it matters |
|---|---|---|
| calculator | ~0 by construction | the control; confirms the model's assumption |
| retrieval + entailment | unknown | the whole case for evidence-grounded verification |
| same-model critic | unknown, likely low | negative control (Huang et al.) |
| PRM | unknown | the baseline a reviewer will demand |

Outputs: a scope estimate per verifier, and a decay curve in *d* directly
comparable to Singh & Pawar's measured escape probabilities
(24.6% / 48.3% / 89.3%).

This is a genuinely novel measurement, it is cheap, and it converts the
allocation rule from simulation into a calibrated design rule. **It should be
the next thing built.**

---

## Remaining work, ordered

| # | task | cost | blocks |
|---|---|---|---|
| 1 | Measure verifier scope (ch. 5) | ~1 GPU-day | C3, ch. 6 |
| 2 | Re-measure error rates on Llama 3.1 8B | ~1 GPU-day | all numbers currently Mistral-7B-SFT |
| 3 | Full gate pipeline end-to-end on GSM8K | scaffold ready | ch. 6 confirmation |
| 4 | Hand-validate ~50 GSM8K dependency graphs | ~2 hours | 9.6% ambiguous links |
| 5 | Cross-domain check on StrategyQA + retrieval | needs 1 | generality |

Items 1 and 4 are the highest value per hour. Item 4 needs no GPU at all and
removes the biggest labelling caveat in the thesis.

---

## Threats to validity, stated up front

- **Generator mismatch.** All measured rates come from Mistral-7B-SFT via
  Math-Shepherd, not the model CAR runs. The method transfers; the numbers may
  not. Item 2 fixes this.
- **Label semantics.** Math-Shepherd's `+`/`-` are automatic Monte-Carlo
  estimates of "leads to a correct answer", not proofs. A lucky wrong step can
  be labelled `+`.
- **Constructed corpus.** Math-Shepherd is a PRM *training* set with a
  deliberate class mix, so no unconditional rate can be read off it. Every
  number is stratified and post-stratified to reported accuracy.
- **Derived graphs.** GSM8K dependencies are inferred from calculator operand
  matching; 9.6% of links are ambiguous.
- **Uncheckable steps.** 12.3% of steps carry no arithmetic and are reported
  separately rather than assumed correct.
- **Modelling vs measurement.** C2b, C3 and C4 rest partly on the propagation
  model. Its assumptions — full repair, i.i.d. per-step error — are known to be
  wrong; the i.i.d. one is wrong in a direction that makes the conclusions
  *conservative*.

---

## Expected outcomes

All three remain publishable, which is the property that made this project
worth doing:

- **Positive** — verifier reach is measurable, differs sharply across verifier
  types, and predicts the optimal allocation. Strongest outcome; needs ch. 5.
- **Mixed** — the gap is confirmed but reach turns out hard to estimate
  reliably. Still yields the measurement, the benchmark critique, and the
  feasibility analysis.
- **Negative** — retrieval verifiers turn out to have near-zero scope too. That
  would be the most interesting result: it means step-level verification cannot
  close the gap at all, and the intervention has to move to the reasoning
  structure rather than the checking.

---

## Venue

Realistic targets: an ACL/EMNLP short paper, or a NeurIPS/ICLR workshop on
LLM evaluation or uncertainty. The measurement plus benchmark critique is a
credible short-paper contribution. A main-conference submission would need
ch. 5 to produce a strong, surprising scope result.

For the M.Tech thesis this is comfortably sufficient: a clear question,
measured answers, refuted hypotheses recorded honestly, and a working system.
