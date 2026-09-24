# CAR — thesis plan (measurement framing)

> **The full draft is written: [docs/thesis/](thesis/README.md), nine chapters.**
> This file remains the working plan and the claim-to-evidence map — what is
> measured, what is modelled, and what is still open. The draft is the argument;
> this is the ledger.

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
| 4b | **Generator transfer** — does C1 survive a stronger model? | **DONE** — it widens |
| 5 | **Measuring verifier reach** — arithmetic vs semantic scope | **DONE** — 0.1999 vs 0.9033 |
| 6 | Allocation — what follows from reach; refutation of "verify early" | simulation done, needs real-model confirmation |
| 6b | **The assembled gate** — end to end, and why it fails | **DONE** — negative at three points |
| 7 | Feasibility — the Kotte floor with measured μ; what α is attainable | done |
| 8 | Benchmark analysis — StrategyQA has no propagation headroom | done |
| 9 | Limitations, negative results, conclusion | ongoing |

Chapters 4, 4b, 5, 7 and 8 are now complete on measured data. Chapter 5 returned
the high-scope outcome: reach is semantic rather than structural. Chapter 4b
answers the most obvious reviewer objection to chapter 4 and answers it in the
direction that helps: on a generator at 80% GSM8K rather than 45%, the fraction
of globally-wrong steps that are arithmetically perfect rises from 0.78 to 0.90.
The practical statement is that a deterministic verifier gets LESS useful as the
generator improves.

---

## Claim → evidence map

| claim | evidence | source | status |
|---|---|---|---|
| C1 — 69.8% of wrong steps are locally valid | 93,129 steps, stratified | Math-Shepherd | measured |
| C1b — C1 STRENGTHENS on a better generator | 0.7848 -> 0.9040, CIs disjoint | Qwen2.5-7B, 500 solutions | **measured** |
| C2 — corruption is near-absorbing | 95.9% persistence; 0/25,971 recovered | Math-Shepherd | measured |
| C2b — local risk does not track final error | local pinned ~0.15, final 0.73→0.27 | simulation | simulated |
| C3 — reach needs independence AND task-training | same-model 0.00, judge 0.23, PRM 0.90 | Math-Shepherd + 3 verifiers | **measured** |
| C4 — "verify early" is false | chains, 5 DAG families, 2 real corpora | simulation + real graphs | measured |
| C4b — later steps are harder | corr(pos, local err) = +0.950 | Math-Shepherd | measured |
| C5 — α = 0.10 costs 32% of budget | μ = 0.3908 + Kotte Prop. 3 | measured + cited | measured |
| C5b — the floor is generator-dependent | μ 0.3908 -> 0.1221; no floor at α=0.20 | Qwen2.5-7B | **measured** |
| C6 — StrategyQA has no headroom | 72.9% one hop; 11.2% vs GSM8K 29.9% | 2272 annotated + 6974 derived graphs | measured |
| C7 — derived GSM8K edges are 94.4% correct | 50 graphs, stratified, hand-adjudicated | FINDINGS-DEPGRAPH | measured |
| C8 — generator uncertainty does not rank global step error | AUROC 0.5589 token-level, 0.5740 semantic, 0.5742 both (0.8668 on synthetic signal, 0.4828 on noise) | 940 test steps | **measured** |
| C8b — the gate misses every binding alpha | risk 0.147 at alpha=0.05, flat across the sweep | end-to-end run | **measured** |
| C8c — the verifier result is DOWNSTREAM of the score | 0.7637 behind the real score, 0.9780 behind an oracle, same verifier | end-to-end run | **measured** |
| C8d — semantic divergence does not rescue it | resampled K=5 over 2,573 steps; r=+0.44 with the token score, combining buys 0.0002 | 12,865 generations | **measured** |
| C8e — a perfect score still misses a binding alpha | oracle risk 0.0885 vs alpha=0.05; the budget binds | end-to-end run | **measured** |
| C9 — a probe on internal states DOES rank step error | AUROC 0.6968 vs 0.5742; +0.12 | 925 test steps | **measured** |
| C9b — and it still does not close the gap | risk 0.1432 at alpha=0.05 (2.9x target); PRM still net-negative at 0.7802 | end-to-end run | **measured** |
| C9c — 0.6968 is a floor, not a ceiling | learning curve 0.7374 -> 0.8748 over 167 -> 670 steps, no plateau | probe run | **measured** |
| C10 — the verifier turns net-positive at AUROC ~0.65 | net- to 0.625, net+ from 0.700; below the probe's own 0.6968 | 13 x 64 sweep | **measured** |
| C10b — that threshold is made entirely of false alarms | at FA=0 the same PRM is net-positive down to AUROC 0.55 | ablation | **measured** |
| C10c — no score quality holds a binding alpha | best is 0.0956 at AUROC 0.99, 1.9x the alpha=0.05 target | sweep | **measured** |
| C10d — AUROC is not a sufficient figure of merit | equal-AUROC scores differ by 0.04 proj. acc; corr(position, probe)=+0.18 vs composite -0.28 | matched-budget sweep | **measured** |
| local error rate ≈ 0.10 | post-stratified, robust 0.091–0.108 | Math-Shepherd | measured |

Everything marked *simulated* rests on the propagation model in
`src/car/propagation.py`, whose closed form is validated against simulation to
within 0.015 over 12 configurations. That is defensible as modelling, but it is
not measurement, and the thesis must not blur the two.

---

## Chapter 5 result: reach is semantic, not structural

| verifier | scope | false alarm | net |
|---|---|---|---|
| arithmetic, step-local (k=0) | 0.0000 | — | — |
| arithmetic, unbounded lookback | 0.1999 | — | — |
| same-model critic (the generator) | 0.0000 | 0.0000 | 0.0000 |
| independent judge (Qwen2.5-7B) | 0.2283 | 0.0200 | 0.2083 |
| task PRM (Math-Shepherd 7B) | **0.9033** | 0.0987 | **0.8047** |

Widening an arithmetic window buys 20% and saturates, because 80% of inherited
corruption has no upstream arithmetic error at all. Changing the verifier class
buys 80%. Data: `runs/semantic_scope_prm.json`.

**A methodological point worth a paragraph in the thesis.** The first completed
run reported the opposite — negative net scope — and it was an artifact: the
PRM flagged 94.9% of the control group, which are steps carrying its own
training labels. A model that cannot recognise its own training signal is not
measuring anything.

Two distinct tokenizer faults produced it, both from the same cause — Kaggle's
transformers tokenizes this SentencePiece model differently from how it was
trained:

1. **Wrong output logits.** The `+`/`-` candidates were derived with
   `tok.encode`, which locally yields `▁+`=648 / `▁-`=387 (the trained tokens)
   but on Kaggle yielded `+`=28806 / `-`=28733 — the same characters without
   the word-boundary marker, i.e. entirely different embeddings. The PRM was
   read at vocabulary indices it had never been trained to use, and separated
   its own labels by 0.0108. These ids are now hardcoded and verified by
   decoding.
2. **Wrong input positions.** The `ки` step-tag id, hardcoded to the reference
   12902, is an *input* id used to locate scoring positions. Kaggle encoded it
   differently, the position mask matched nothing, and every score came back
   NaN. It is now resolved in context from the live tokenizer.

A validation gate scores 400 known-label steps before the real measurement and
aborts below 0.15 separation. It caught three successive broken configurations
before the fourth passed at 0.5788. Without it the project would have published
a confident negative result produced entirely by tokenizer mismatch.

The judge arms had their own instance of the same class of bug: **right padding
for batched decoder-only generation**, which inserts pad tokens between prompt
and continuation so the model generates from padding. It surfaced only as a
warning, and moved the Qwen result from an untrustworthy 0.1933 to 0.2283.
The general lesson for the thesis: on a borrowed model, the harness must prove
it can reproduce that model's known behaviour before any novel number from it
is believed.

The same-model critic detects zero errors (validation separation 0.0000 -- it
approves its own known-bad steps), confirming Huang et al. A general
independent judge barely beats arithmetic. Only the task-specialised,
independent PRM closes the gap. Retrieval+entailment has no meaning on GSM8K
(no external corpus) and is genuinely not measurable here.

---

## Remaining work, ordered

| # | task | cost | blocks |
|---|---|---|---|
| 1 | ~~Measure verifier scope (ch. 5), all arms~~ | done | — |
| 2 | ~~Hand-validate ~50 GSM8K dependency graphs~~ | done | found a systematic extraction bug; edge error measured at 5.6% |
| 3 | ~~Full gate pipeline end-to-end on GSM8K~~ | done | negative result; see FINDINGS-PIPELINE |
| 4 | ~~Re-measure error rates on a second generator~~ | done | Qwen2.5-7B; Llama 3.1 is licence-gated on Kaggle |
| 5 | Cross-domain check on StrategyQA + retrieval | ~1 GPU-day | generality; limited by C6 |

**Item 2 is done, and it paid for itself.** It was queued to quantify the
ambiguous-link caveat; instead it exposed a systematic bug — the operand regex
was reading each subtraction operator as a minus sign, so every subtraction lost
its dependency edge. Fixing it moved mean depth 2.54 → 2.79, headroom 26.6% →
29.9%, and collapsed the unclassifiable `other` shape category from 22.9% to
5.9%. It also overturned one published conclusion (see below). The corrected
edge error rate is **5.6%**, measured rather than proxied.

The general lesson, worth a line in the thesis: the caveat that gets quantified
is rarely the one that matters. Auditing the derivation found a bug an order of
magnitude more consequential than the ambiguity it was meant to measure.

Item 5 is worth doing but is capped by C6 — StrategyQA has almost no propagation
headroom, so a retrieval arm there tests generality of the *verifier* finding on
a benchmark that cannot exhibit the *propagation* finding.

---

## Threats to validity, stated up front

- **Generator mismatch, now partly addressed.** The headline rates come from
  Mistral-7B-SFT via Math-Shepherd. C1 has been reproduced on Qwen2.5-7B-Instruct
  (80% GSM8K) and comes out *higher*, 0.7848 -> 0.9040 with disjoint Wilson
  intervals, so the gap is not an artifact of a weak generator. Two numbers do
  NOT transfer and are now labelled per-generator: μ (0.3908 -> 0.1221, which
  removes the α=0.20 floor entirely) and corruption persistence (95.9% ->
  66.4%). See docs/FINDINGS-GENERATOR.md.

- **A "step" is not model-invariant.** Math-Shepherd's step is one calculator
  operation; Qwen writes 5.15 steps per solution of which 59% are narration
  carrying no arithmetic at all. Per-step rates across generators are rates over
  different units. This is a limitation of the unit of analysis, not of the
  measurement, and the thesis should say so explicitly.
- **Label semantics.** Math-Shepherd's `+`/`-` are automatic Monte-Carlo
  estimates of "leads to a correct answer", not proofs. A lucky wrong step can
  be labelled `+`.
- **Constructed corpus.** Math-Shepherd is a PRM *training* set with a
  deliberate class mix, so no unconditional rate can be read off it. Every
  number is stratified and post-stratified to reported accuracy.
- **Derived graphs.** GSM8K dependencies are inferred from calculator operand
  matching. Hand-validated on 50 stratified graphs: **5.6% of edges are wrong**
  (16.5% in graphs containing an ambiguous link, 4.2% elsewhere). That is a
  lower bound — dependencies routed through unannotated solution lines are
  invisible to any operand-matching scheme. Adjudication was by LLM, not blind
  human annotation. See docs/FINDINGS-DEPGRAPH.md.
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

These were written before the end-to-end run. **The outcome was the negative
one**, which this section had already called the most interesting, and it
arrived by a different route than expected: not because reach is universally
low, but because the signal driving the gate does not rank the risk, the
calibration certifies coverage rather than risk, and the one verifier with
reach loses more to false alarms than it recovers.

The resulting claim is about the design space rather than one system:

> Selective verification of LLM reasoning fails at three independent points.
> The signal does not rank the risk -- neither token-level uncertainty (AUROC
> 0.5589) nor sampling-based semantic divergence (0.5740), and combining them
> buys 0.0002; the calibration certifies a quantity that is not the risk
> (coverage holds, selective risk misses alpha by 3x); and the only verifier
> with reach costs more in false alarms than it recovers on a realistic base
> rate. Fixing any one of them is not sufficient.

That is a stronger contribution than a working gate, and it is falsifiable:
each of the three has a measured number and a regression test.

---

## Venue

Realistic targets: an ACL/EMNLP short paper, or a NeurIPS/ICLR workshop on
LLM evaluation or uncertainty. The measurement plus benchmark critique is a
credible short-paper contribution. A main-conference submission would need
ch. 5 to produce a strong, surprising scope result.

For the M.Tech thesis this is comfortably sufficient: a clear question,
measured answers, refuted hypotheses recorded honestly, and a working system.
