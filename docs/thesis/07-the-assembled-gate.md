# 7. The assembled gate

> Reproduce: `python scripts/exp_gate_pipeline.py`
> Null control: `python scripts/exp_gate_pipeline.py --synthetic-signal 0.0`

Chapters 4, 5 and 6 measured one component each. This chapter runs the whole
control loop on real model output — real generator uncertainty, real step
labels, real conformal calibration, real budget — and asks the only question
the assembled system can answer:

> Does calibrating a threshold on an uncertainty score actually bound the risk
> of the steps the gate lets through?

It does not. The result is negative and it is the most useful thing this
project has measured.

## 7.1 Setup

500 Qwen2.5-7B-Instruct solutions on GSM8K test, 2,573 steps, split
175 dev / 143 calibration / 182 test by hash of example id. Uncertainty
recovered by teacher-forcing (`scripts/gpu_score_uncertainty.py`, ~10 min on
two T4s) plus resampled semantic divergence
(`scripts/gpu_semantic_divergence.py`, ~4h, 12,865 generations). Corpus,
features and raw samples are all committed.

Conditions, all through the same `CARAgent` loop with one component swapped:
plain chain-of-thought, always-verify, random gate at matched budget, quantile
gate, split conformal, adaptive conformal with IPW, adaptive with naive
updates, and an **oracle score** — the same split-conformal calibrator driven by
a score that reads the label. The oracle is not deployable; it is the ceiling,
and it is what separates "the gate is bad" from "the task is hard at this
budget".

Verifiers, parameterised by the reach measured in Chapter 5: step-local
arithmetic (scope 0.0000), independent judge (0.2283 at 2.0% false alarm), task
PRM (0.9033 at 9.87%), and an **ablation** of the task PRM with its false-alarm
rate forced to zero.

## 7.2 The signal does not rank the risk

```
score AUROC for detecting a globally-wrong step   0.5589   token-level only
                                                  0.5742   with semantic divergence
```

Token entropy, max surprisal and mean log-probability, combined and
standardised on dev data only. `mean_logprob` on good steps is −0.2916; on bad
steps −0.3323. The direction is right and the separation is negligible.

**This is not a harness failure.** The same pipeline on synthetic features with
a one-standard-deviation separation gives AUROC **0.8668**, and on pure noise
**0.4828**. The machinery detects signal when there is signal.

### A probe on internal states does rank it

The obvious objection — that the signal might exist somewhere the measured
features do not reach — is testable, and §7.6 tests it. A logistic probe on the
generator's own frozen hidden states reaches **AUROC 0.6968**, +0.12 over
everything above. So the result in this section is about *these signals*, not
about step-level uncertainty in general. It does not change the conclusions
below, and why it does not is the more interesting half.


![ROC for detecting a globally-wrong step. All three real signals hug the diagonal; the dashed control shows what the same harness does when a signal exists.](figures/fig6-roc.png)

**Figure 7.1.** ROC for detecting a globally-wrong step. All three real signals hug the diagonal; the dashed control shows what the same harness does when a signal exists.

### Semantic divergence does not rescue it

Semantic divergence is the signal the original specification weighted most
heavily, and it cannot be recovered by teacher-forcing — it is defined over
independently sampled continuations. It was therefore resampled: K = 5
continuations of the next step for all 2,573 steps, clustered into
meaning classes.

| feature set | AUROC (test) | AUROC (all steps) |
|---|---|---|
| token-level only | 0.5589 | 0.5277 |
| **semantic divergence only** | **0.5740** | **0.5488** |
| both | 0.5742 | 0.5427 |

Divergence is 0.4115 on globally-correct steps and 0.4809 on wrong ones — the
right direction, worth 0.07 on a [0,1] scale. Semantic divergence is marginally
the better of the two signals, and **combining them adds nothing**
(0.5740 → 0.5742). The two scores correlate at **r = +0.44**, so they are
largely the same information.

This is the same conclusion Wen et al.
([arXiv:2602.02427](https://arxiv.org/abs/2602.02427)) reach from the other
direction: they argue embedding perturbation reflects intermediate-step
uncertainty better than sampling-based agreement, and this is the
sampling-based half measured on a second benchmark. Their alternative signal
is not tested here — see Chapter 9.

### The equivalence relation was load-bearing

Clustering samples requires deciding when two candidate next-steps mean the
same thing. The library default was string equality. Qwen writes one
computation three ways:

```
48/2 = <<48/2=24>>24 clips
\( 48 / 2 = 24 \) clips
That gives 48 / 2 = 24 clips.
```

Under string equality those are three meanings. `numeric_equivalence` — two
arithmetic steps agree if they assert the same number, whatever the phrasing —
is not a cheap substitute for bidirectional entailment so much as a closer fit
to what an arithmetic step actually asserts.

The same samples, re-clustered:

| clustering | mean divergence | unanimous steps | AUROC |
|---|---|---|---|
| numeric equivalence | 0.4239 | 1006 / 2573 | **0.5488** |
| exact string match | 0.6468 | 524 / 2573 | **0.4904** |

Exact match inflates divergence by half and drives AUROC *below chance*, because
measured "disagreement" then tracks notational variety — a property of
verbosity rather than of doubt.

**Had the default been used, this chapter would have reported that semantic
divergence is anti-predictive.** That is a different conclusion, a wrong one,
and it would have been entirely believable: a clean sub-chance AUROC reads as
"this signal is actively misleading" rather than "your clustering is broken."

### So the obvious objection is that the relation is the result

If AUROC over all steps moves from 0.5488 to 0.4904 when the relation changes
— 0.5740 to 0.5287 on the test split — then `numeric_equivalence`, a cheap
stand-in for the relation the literature actually uses, might be the thing
being measured. Kuhn et al. and Farquhar et al. cluster by **bidirectional
entailment** under an NLI model, so that is the relation the objection points
at, and the committed K = 5 samples make it answerable without regenerating
anything.

> Reproduce: `python scripts/exp_equivalence_relations.py`
> Full writeup: [docs/FINDINGS-ENTAILMENT.md](../FINDINGS-ENTAILMENT.md)

| relation | mean divergence | unanimous | AUROC all | AUROC test | composite |
|---|---|---|---|---|---|
| numeric equivalence | 0.4239 | 1,006 / 2,573 | 0.5488 | **0.5740** | 0.5742 |
| exact string match | 0.6468 | 524 / 2,573 | 0.4904 | 0.5287 | 0.5612 |
| **bidirectional entailment** | **0.1205** | **1,854 / 2,573** | 0.5174 | **0.5625** | **0.5805** |

**The reference relation does not rescue the signal.** Its test AUROC is 0.5625
against numeric equivalence's 0.5740, and that 0.0114 gap is *not* a ranking —
a solution-clustered bootstrap over the 182 test solutions puts it at 95% CI
[−0.087, +0.059]. What does survive the interval is entailment's own:
**0.5625, CI [0.512, 0.614]**. Even the optimistic end is a signal nobody would
gate on.

How much three relations corroborate each other is worth being careful about,
because the overclaim is easy. They are not three independent probes — they are
three points on one permissiveness knob, and they nest: exact match calls 0.1%
of pairs equal, numeric equivalence 31.9%, entailment 78.0%, with **100%** of
exact-equal pairs numeric-equal and **93.0%** of numeric-equal pairs mutually
entailing. They share the generations, K, temperature, the clustering algorithm
and the entropy map too. So the claim they support is narrower and still
sufficient: *across the full usable range of cluster permissiveness, from
merging 0.1% of pairs to merging 78%, step-level sampling divergence does not
rank step error on this corpus.*

**Entailment turns out to be the most permissive relation, not the strictest.**
86.6% of the 27,936 directed pairs are judged entailment and 72% of steps come
out unanimous, against 39% under numeric equivalence. Most of that gap is not
NLI error: **61.7%** of pairs have no extractable number on one side, so
`numeric_equivalence` falls back to string equality and calls them distinct.
Those are the narration and algebra-rearrangement steps, which §9.2 notes are
59% of what Qwen writes — the relations differ most exactly where neither is
well defined. On the 5,343 pairs where both sides assert a number they agree
85.5% of the time.

That pooled rate conceals the asymmetry that matters:

| comparable pairs | count | judged mutually entailing |
|---|---|---|
| numeric equivalence **agrees** | 4,453 | 4,141 — **93.0%** |
| numeric equivalence **disagrees** | 890 | 463 — **52.0%** |

**On the pairs carrying an arithmetic disagreement, the reference relation
erases half of it.** An MNLI model checks whether two sentences are about the
same thing, not whether they compute the same quantity, and on arithmetic those
come apart — it calls a step asserting an 80%-of-$40 increase and a step
asserting the resulting total mutually entailing.

Whether that leniency props the AUROC up or drags it down is measurable rather
than arguable. Re-clustering with an **arithmetic veto** that undoes exactly
those 463 merges (changing 134 of 2,573 steps) moves test AUROC 0.5625 → 0.5577,
paired difference −0.0048, CI [−0.018, +0.011]: no measurable direction either
way. The relation is a poor fit for arithmetic steps, and fixing that does not
recover a signal.

## 7.3 So the calibration certifies nothing useful

Base risk on test steps is **0.1578**. Any α above that is satisfied by
verifying nothing, so only α < 0.1578 tests anything at all — and the
configured operating point of α = 0.30 is vacuous.

| α | Kotte floor | binds? | split conformal risk | verify % | CAR risk | verify % |
|---|---|---|---|---|---|---|
| 0.05 | 11.4% | **yes** | **0.1491** | 4.6% | **0.1555** | 18.4% |
| 0.10 | 6.4% | **yes** | **0.1494** | 9.2% | **0.1567** | 19.7% |
| 0.15 | 0.9% | **yes** | **0.1516** | 14.1% | **0.1582** | 22.5% |
| 0.20 | 0.0% | no | 0.1502 | 17.0% | 0.1552 | 23.9% |
| 0.30 | 0.0% | no | 0.1538 | 21.8% | 0.1587 | 26.9% |

At α = 0.05 the measured selective risk is **three times the target**. The risk
hardly moves across the whole sweep — 0.1491 to 0.1587 — while verification
climbs from 4.6% to 26.9%. **The gate spends budget and buys nothing.**

Token-level features alone give the same picture (0.1468 at α = 0.05), which is
the point: adding the specification's favoured signal changed AUROC by 0.015
and changed risk control by nothing.

### How much of that is the score, and how much the budget

The oracle answers this, because it changes the score and nothing else:

| condition | verify % | calls/q | selective risk |
|---|---|---|---|
| no gate | 0.0% | 0.00 | 0.1554 |
| split conformal, α = 0.05 | 4.6% | 0.23 | 0.1491 |
| split conformal, α = 0.30 | 21.8% | 1.09 | 0.1538 |
| **probe score, α = 0.30** | **19.1%** | **0.96** | **0.1394** |
| **oracle score** | **7.3%** | **0.37** | **0.0885** |

A perfect score cuts selective risk by 43% *and* spends a third of the
verification the α = 0.30 setting does. The score is genuinely the binding
constraint on everything the deployable conditions achieve.

But **the oracle still misses α = 0.05 by 1.8×**. At two calls per question over
a mean of 5.15 steps, most globally-wrong steps go unverified however perfectly
they are ranked; the residual 0.0885 is the budget, not the score. (The oracle's
verification rate is also flat across α — a binary score gives the conformal
quantile nowhere to move.)

Two bottlenecks, and they are separable: the score is worth 0.154 → 0.089, and
the budget is what stands between 0.089 and 0.05.

The probe sits where its AUROC predicts. At every α it improves on the measured
score and misses the target:

| α | target | token + semantic | **probe** | oracle |
|---|---|---|---|---|
| 0.05 | 0.05 | 0.1491 | **0.1432** | 0.0885 |
| 0.10 | 0.10 | 0.1494 | **0.1363** | 0.0885 |
| 0.30 | — | 0.1538 | **0.1394** | 0.0885 |

It closes about 29% of the AUROC gap to a perfect score and about 22% of the
risk gap, which is the internal consistency one would want before believing
either number. And it misses α = 0.05 by **2.9×**.

> The earlier version of this chapter could be read as "the signal happened to
> be absent." It is not. A better signal exists, improves the gate, uses fewer
> calls — and the gate still does not hold its target.

Figure 7.7 plots risk against score quality for all three; §7.6 is where the
probe itself is measured.

![Measured selective risk against the target. The gate misses every α that binds, and the measured risk barely responds to the target at all.](figures/fig7-alpha-sweep.png)

**Figure 7.2.** Measured selective risk against the target. The gate misses every α that binds, and the measured risk barely responds to the target at all.

### Why conformal calibration does not save it

Split conformal fits the threshold so the acceptance region *covers* 1 − α of
correct steps. That is coverage. Coverage is not selective risk. When the score
is uninformative, the accepted region contains wrong steps at approximately the
base rate no matter where the threshold sits — the guarantee holds perfectly
and the quantity of interest is untouched.

> A conformal guarantee is a statement about the acceptance rule, not about the
> risk of what it accepts. With an uninformative score the two come apart
> completely, and **nothing in the calibration procedure reports that**.

This is the sentence "coverage is not accuracy," which this project's
documentation carried from the start, finally measured.


![The accuracy–cost trade-off against Kotte's impossibility floor. Verification climbs from 4.6% to 21.8% and risk moves by 0.005.](figures/fig8-pareto-with-floor.png)

**Figure 7.3.** The accuracy–cost trade-off against Kotte's impossibility floor. Verification climbs from 4.6% to 21.8% and risk moves by 0.005.

## 7.4 The best verifier available is net-negative

Projected final-answer accuracy against a no-gate baseline of **0.8022**:

| verifier | scope | false alarm | always verify | random gate | split conformal |
|---|---|---|---|---|---|
| arithmetic, step-local | 0.0000 | 0.0000 | 0.8022 | 0.8022 | 0.8022 |
| independent judge | 0.2283 | 0.0200 | 0.8022 | 0.7802 | 0.7857 |
| task PRM | 0.9033 | 0.0987 | **0.7637** | **0.7143** | 0.7802 |
| **task PRM, ablation FA = 0** | 0.9033 | **0.0000** | **0.9231** | 0.8516 | 0.8736 |

The highest-scope verifier available **loses 4 points of accuracy** at its
measured operating point, and more verification makes it worse. Zero out its
false-alarm rate and the same verifier gains **12 points**.


![Projected accuracy for the same verifier at the same 9.87% false-alarm rate under three scores. The FA = 0 ablation isolates the mechanism; the oracle column shows the score controls it. MODELLED, not measured.](figures/fig9-verifier-value.png)

**Figure 7.4.** Projected accuracy for the same verifier at the same 9.87% false-alarm rate under three scores. The FA = 0 ablation isolates the mechanism; the oracle column shows the score controls it. MODELLED, not measured.

The difference is a base-rate effect. Chapter 5 reported
`net = scope − false alarm = 0.8047`, measured on a population *conditioned on
being arithmetic-blind inherited corruption* — every item in it was wrong. In
deployment the verifier is pointed at whatever the gate selects, and behind a
near-chance score that is mostly correct steps.

**The oracle row settles which component is at fault, and it is not the
verifier:**

| score | verifier | calls/q | recall | projected accuracy |
|---|---|---|---|---|
| real, split conformal | task PRM | 1.09 | 0.1761 | 0.7912 |
| real, always verify | task PRM | 1.89 | 0.2606 | **0.7637** |
| probe | task PRM | 0.96 | 0.2465 | 0.7802 |
| **oracle** | **task PRM** | **0.37** | **0.4366** | **0.9780** |

The probe row is the informative one. It catches **40% more wrong steps** than
the measured score on *fewer* calls, and projected accuracy is still below the
0.8022 baseline. At AUROC 0.70 the gate still points the verifier at correct
steps often enough for false alarms to dominate; only the oracle escapes. So
there is a score quality at which a 9.87%-false-alarm verifier becomes worth
having, and the next subsection measures it rather than guessing.

The same verifier, at the same measured 9.87% false-alarm rate, moves projected
accuracy from 0.7637 to **0.9780** — 17.6 points above the no-gate baseline — on
*a third* of the calls. A false alarm can only fire on a step the gate chose to
verify, and a good score almost never chooses a correct one.

> The figure of merit is not `scope − FA`, and it is not
> `scope × P(wrong)` against `FA × P(correct)` over the population either. It is
> conditioned on **what the gate selects**:
> `scope × P(wrong | verified)` against `FA × P(correct | verified)`.
> The score sets that conditioning, so improving the score raises the
> verifier's net value without touching the verifier.

That is a correction to how C3 should be stated, and a sharper one than the
population-level version: the PRM is not a bad verifier being oversold, it is a
good verifier being aimed badly. It also explains the first row of the table
above — with scope 0.0000 a calculator gate cannot move the answer at all, however
well aimed, which is exactly what C1 predicts.

### How good does the score have to be?

> Reproduce: `python scripts/exp_score_quality_threshold.py --seeds 64`
> Full writeup: [docs/FINDINGS-SCORE-QUALITY.md](../FINDINGS-SCORE-QUALITY.md)

Score quality can be made a dial. Under the binormal model a globally-wrong step
draws its score from N(d, 1) and a correct step from N(0, 1), so
AUROC = Φ(d / √2) and the separation needed for a target AUROC inverts in closed
form as d = √2 · Φ⁻¹(AUROC). Everything else is held fixed — same corpus, same
splits, same calibrator, same budget, same verifier at its measured scope and
false-alarm rate — so what moves between rows is the ranking and nothing else.

| AUROC | verify % | sel. risk | 1st-bad recall | PROJ acc | 95% CI | |
|---|---|---|---|---|---|---|
| 0.550 | 22.5% | 0.1536 | 0.2853 | 0.7782 | [0.7725, 0.7840] | net − |
| 0.625 | 23.1% | 0.1463 | 0.3786 | 0.7940 | [0.7880, 0.8001] | net − |
| 0.650 | 23.3% | 0.1441 | 0.4111 | 0.8010 | [0.7953, 0.8067] | ~same |
| **0.700** | 23.6% | 0.1405 | 0.4635 | **0.8119** | [0.8057, 0.8181] | **net +** |
| 0.800 | 24.1% | 0.1335 | 0.5753 | 0.8341 | [0.8282, 0.8400] | net + |
| 0.990 | 24.6% | 0.1258 | 0.7736 | 0.8712 | [0.8650, 0.8775] | net + |

**The crossing is at AUROC ≈ 0.65** — net-negative up to 0.625, net-positive
from 0.700, with the means crossing the 0.8022 baseline at 0.657. Between those
two the sweep cannot distinguish the gated system from doing nothing, so the
result is an interval, not a point. 64 seeds per row.

An earlier draft of this section guessed "well above 0.70". That was wrong in
the direction that matters: the requirement is *lower* than assumed, and the
probe's 0.6968 is sitting on the boundary rather than far short of it.

**The threshold is made entirely of false alarms.** Rerun the sweep with the
verifier's false-alarm rate switched off and there is no crossing to find:

| AUROC | 0.550 | 0.650 | 0.750 | 0.900 | 0.990 |
|---|---|---|---|---|---|
| PROJ acc, FA = 0.0987 | 0.7782 | 0.8010 | 0.8219 | 0.8566 | 0.8712 |
| PROJ acc, FA = 0 | 0.8589 | 0.8844 | 0.9045 | 0.9421 | 0.9563 |

At FA = 0 the verifier is worth having at *every* score quality tested,
including 0.55 — a score barely better than a coin. The entire question "how
good does the score need to be" is created by the false-alarm rate, which makes
this the sharpest form of §7.4's correction and the one a practitioner can act
on: halving a verifier's false-alarm rate lowers the score quality you need
more than raising its detection rate does.

**And no score quality holds a binding α at this budget.** At α = 0.05 the best
any row manages is selective risk 0.0956, at AUROC 0.99 — still 1.9× the
target. §7.3 showed that at a single point with a binary oracle; the sweep shows
it across the whole range, which rules out reading the oracle's failure as an
artifact of its degenerate score distribution. Score quality is not the binding
constraint on risk control. The budget is.

![Left: projected accuracy against score AUROC, with and without false alarms. Right: first-bad-step recall, where the two real scores come apart from the synthetic curve.](figures/fig12-score-quality.png)

**Figure 7.5.** Where the verifier stops being a liability, and why AUROC does
not predict it. Shaded column on the left is the interval the sweep cannot
resolve.

### AUROC is not a sufficient description of a score

The sweep also answers a question it was not built for. Pinning verification at
19.1% — the probe's measured rate, so only the ranking differs — gives:

| | AUROC | calls/q | recall | 1st-bad recall | PROJ acc |
|---|---|---|---|---|---|
| synthetic score | 0.6974 | 0.79 | 0.2416 | **0.3413** | **0.8206** |
| **probe, layer 25** | 0.6968 | 0.96 | 0.2465 | **0.3077** | **0.7802** |

The probe sits at the **0th percentile of 64 synthetic draws** on the same 182
test questions. It catches slightly *more* wrong steps and rescues *fewer*
answers, and the mechanism is measurable rather than inferred:

```
corr(normalised step position, probe score)      +0.1813
corr(normalised step position, composite score)  -0.2766
```

**The probe flags late steps; the composite flags early ones.** Chapter 6
measured local error rising with position — corr(position, error) = +0.950,
doubling from 11% at step 1 to 22% at step 8 — so a score that chases positional
difficulty is rewarded on AUROC. But the projection only pays for the **first**
bad step, because everything downstream inherits corruption that repairing a
later step does not undo (C2, and 0 recoveries in 25,971 solutions). The
probe's AUROC advantage is being spent where it cannot buy an answer.

> Two scores with identical AUROC are worth different amounts. What a
> step-level score is worth depends on **which** errors it ranks highly, and
> under a propagation objective the ones that count are the earliest.

Two consequences for how the rest of this chapter should be read. The crossing
above is an estimate for a *well-behaved* score, so 0.65 is a floor on the
requirement rather than a specification — a real score with a positional bias
lands below the synthetic curve, exactly as the probe does. And the AUROCs in
§7.2 are the right measurement of the wrong quantity: comparable to each other
and to the literature, but not sufficient to predict what a score is worth in
the loop.

The honest limit on this: only **40 of 182** test trajectories contain a
globally-wrong step, so first-bad recall has a denominator near 39 and any
single pair of rows differs by two or three trajectories. The claim rests on the
monotone trend across 13 grid points × 64 seeds (0.2853 → 0.7736), not on the
probe-versus-synthetic pair alone.

## 7.5 What is measured and what is modelled

The corpus is fixed, so the gate cannot change what the model writes.
`traj.correct` is therefore identical under every condition, and reporting it
as accuracy would say the gate does nothing — an artifact of replay, not a
finding.

The projection applies the propagation model's assumption explicitly:

- a wrong answer is rescued **iff** the first globally-bad step was detected and
  repaired, since everything downstream inherits the corruption;
- a right answer is lost if a false alarm "repaired" a step that was correct;
- an answer that was already correct stays correct even when the trajectory
  contains a `-` step, because Math-Shepherd labels are optimistic.

With scope 0 and no false alarms the projection reduces **exactly** to the
observed accuracy — the first row of the table above is 0.8022 across every
condition — and a regression test pins that. A projection that moves when
nothing was repaired is fabricating.

The projection is nonetheless **pessimistic**: it treats any false revision of
a correct step as fatal to the answer, where a real system might revise a
correct step to another correct value. The *direction* of the false-alarm
result is isolated by the FA = 0 ablation; the *magnitude* is not robust.

## 7.6 A probe on internal states

> Reproduce: `python scripts/gpu_probe_states.py`, then
> `python scripts/exp_gate_pipeline.py --probe runs/probe_qwen25_7b.json`
> Full writeup: [docs/FINDINGS-PROBE.md](../FINDINGS-PROBE.md)

ReProbe (Ni et al., §2.4) trains a sub-10M-parameter probe on a frozen model's
internal states and matches PRMs up to 810× larger. That makes it the cheapest
available test of whether §7.2's result is about the *particular* signals
measured there.

One teacher-forced pass per solution over the same corpus, hidden states taken
at each step's final token for all 29 hidden-state tensors (the embedding
output plus 28 transformer layers), a logistic probe per layer, and
the same hash splits — so the AUROCs compare step for step.

**AUROC 0.6968 on test**, against 0.5742 for the best of the measured signals.

![Left: AUROC by hidden layer, every layer tried. Right: the held-out learning curve, which stays flat.](figures/fig10-probe-layers.png)

**Figure 7.6.** The probe's layer profile and learning curve. Left: a smooth rise through the network peaking at layers 24-27 — the shape of a real encoded property, not selection noise over 29 candidates. Right: the held-out curve at up to 2× the training data, flat. An earlier version plotted the selection-split curve here, which read as climbing.

![Selective risk against score AUROC, with the alpha targets and the no-gate line.](figures/fig11-score-vs-risk.png)

**Figure 7.7.** Same corpus, calibrator and budget; only the score differs — the measured signals at 0.5742, the probe at 0.6968, an oracle at 1.0. A better score helps monotonically and still does not reach either target.


### Two things that stop this being over-read

**The winner's curse is large, and visible.** The selection-split AUROC is
**0.8748** against 0.6968 on test — a **0.18 gap**, from choosing the best of 29
layers × 5 regularisation strengths on 287 selection steps. Had the layer been
chosen on test, this section would report ~0.87 and claim the probe beats the
ch. 5 PRM. `select_and_fit` does not take test indices as a parameter, and a
test asserts its signature cannot grow one; the discipline is worth 0.18 AUROC
of wrongness here.

**An earlier draft called 0.6968 a floor on evidence that could not support
it.** The claim rested on a learning curve climbing 0.7374 → 0.8748 across
167 → 670 training steps. That curve was evaluated on the *selection* split —
the same data the layer and the regularisation strength were chosen on — so it
was biased by that selection and measured on the wrong population. The claim
therefore had no evidence, which is not the same as being false.

Measuring it properly needs the configuration held fixed: re-running layer
selection on a larger training set can land somewhere else, and then the
difference is partly the layer rather than the data. It does — the dev-only
probe selects layer 25 and the pooled one layer 28.

| configuration | 670 → 1,361 training steps | change | 95% CI |
|---|---|---|---|
| layer 25, C = 0.1 (the published probe) | 0.6968 → **0.7086** | **+0.0105** | [−0.035, +0.057] |
| layer 28, C = 0.1 | 0.6817 → 0.6896 | +0.0072 | [−0.041, +0.055] |

And the held-out learning curve, over a shuffled pool so that size is the only
thing changing:

| training steps | 136 | 340 | 680 | 1020 | 1361 |
|---|---|---|---|---|---|
| test AUROC | 0.5686 | 0.6691 | 0.6590 | 0.6856 | 0.6896 |

**More data helps, and it helps very little.** The sign is positive at every
configuration, every interval spans zero, and doubling the training set buys
about a hundredth of AUROC — nothing like the trajectory the leaked curve
implied. So C9c is recorded as **unsupported**, not refuted: its evidence was
invalid, and the honest measurement neither establishes nor rules it out.

> Reproduce: `python scripts/exp_probe_variants.py`
> Full writeup: [docs/FINDINGS-PROBE2.md](../FINDINGS-PROBE2.md)

> Reproduce: `python scripts/exp_probe_variants.py`
> Full writeup: [docs/FINDINGS-PROBE2.md](../FINDINGS-PROBE2.md)

### AUROC was also the wrong thing to train it on

§7.4 showed the projection only pays for the **first** globally-wrong step in a
solution, and that this probe's score correlates +0.18 with step position — it
spends its ranking power on late steps no repair can rescue. That is a
diagnosis; it can be acted on. Retraining on the first-bad label instead, with
first-bad recall measured at a fixed 19.1% verification rate so no variant wins
by flagging more:

| trained on | AUROC | **first-bad recall** | score–position corr |
|---|---|---|---|
| the global label, dev only (published) | 0.6968 | 0.3000 | +0.233 |
| the global label, pooled | 0.6896 | 0.2250 | +0.163 |
| **the first-bad label** | 0.5735 | **0.4500** | −0.472 |
| the global label, position projected out | 0.6811 | 0.2750 | −0.066 |

Training on the right target raises the quantity that pays and **eliminates**
the probe's AUROC advantage rather than merely reducing it — 0.5735 sits below
the 0.5742 token+semantic baseline whose failure is this chapter's central
negative result. That trade is the chapter's own point made concrete: ranked by
AUROC the first-bad probe is the worst of the four, and it is the one that best
does the job the system is for.

**How large the gain is depends on which global-target probe it is measured
against**, and both are reasonable:

| comparison | gain | 95% CI | |
|---|---|---|---|
| vs the pooled global probe | +0.2216 | [+0.051, +0.390] | excludes zero |
| vs the **published** probe (0.3000) | +0.1412 | [−0.065, +0.333] | **spans zero** |

The pooled probe has the lowest first-bad recall of any global-target variant
here, so quoting only the first row picks the flattering baseline. Against the
probe a reader actually has in mind the effect is +47% relative with an
interval that includes no effect. The direction is consistent; the significance
is not robust to the comparator.

In the gate, each score chooses its own operating point, so the verification
rate is not matched and recall per call is reported with it:

| score | calls/q | selective risk | first-bad recall | per call | PROJ accuracy |
|---|---|---|---|---|---|
| no gate | 0.00 | 0.1554 | — | — | **0.8022** |
| split conformal (token+semantic) | 1.09 | 0.1538 | 0.3590 | **0.3294** | 0.7912 |
| probe, global target | 0.96 | **0.1394** | 0.3077 | 0.3205 | 0.7802 |
| probe, first-bad target | 1.18 | 0.1600 | **0.3846** | 0.3259 | 0.7912 |

The first-bad probe has the highest raw first-bad recall and spends 23% more
calls to get it; **per call the ordering reverses and the chapter's existing
token+semantic baseline is best**, reaching the same 0.7912 projected accuracy
on fewer calls. Selective risk also *worsens* 0.1394 → 0.1600, because a score
tuned for first-bad steps is near chance on the global label. Against what the
chapter already had, this is not an improvement.

**The limit is the label.** The corpus holds 108 first-bad steps, 40 in test.
The significant comparison trains on 49 positives and saw the calibration
split; the gate-safe version trains on 27 and gives +0.0805, CI [−0.119,
+0.262]. So: the target is demonstrably the right one to optimise, and this
corpus is too small to show it buys anything deployable.

### What it changes

Chapter 9 predicted, before the run, that a probe here would land near the
ch. 5 PRM's 0.9033 rather than above it — reasoning that ReProbe's margin is
largest out of domain while strong PRMs reach parity on GSM8K. The measured
0.6968 is **below** that, and doubling the training data recovers about
0.01 — so probe-scale training is not what holds the number down, though it is
worth a little. Either the prediction was wrong about this setting, or a
*linear* probe on frozen states is the wrong instrument for it. ReProbe's
probes are not linear, and that is the untested half.

## 7.7 What this chapter establishes

Three independent failures, each with a number and a regression test:

| failure point | measurement | what the oracle and the probe say about it |
|---|---|---|
| the measured signals do not rank the risk | AUROC 0.5589 / 0.5740 / 0.5742; a probe reaches 0.6968 | binding, and *not* a dead end — the signal exists, and a perfect score takes risk 0.154 → 0.089 and accuracy 0.79 → 0.98 |
| the calibration certifies the wrong quantity | coverage holds; selective risk misses α by 3× | not repaired by a perfect score — the oracle still misses α = 0.05 by 1.8×, because the budget binds |
| the verifier with reach costs more than it recovers | 0.7637 against a 0.8022 baseline | **downstream of the score**, not independent: the same verifier gains 17.6 points behind the oracle, and turns positive at AUROC ≈ 0.65 |

The oracle run changes the shape of this conclusion, and it is worth being
exact. The three failures are *not* independent in the way an earlier draft of
this chapter claimed. The verifier's net-negative result is a consequence of the
score, not a separate defect: fix the score and the same verifier becomes worth
+17.6 points. What survives as genuinely separate is the score and the budget —
a perfect score still cannot hold α = 0.05 at two calls per question.

So the honest summary is two bottlenecks, one of which subsumes the third:
**the signal, and the budget.** Better uncertainty estimation is necessary and
not sufficient; better calibration cannot help while the score does not rank;
and the verifier was never the problem.

The score-quality sweep puts numbers on both halves of that and adds a third
thing nobody asked for. The verifier turns positive at AUROC ≈ 0.65, which is
*below* the probe's 0.6968 — so the signal bottleneck is nearly cleared
already, and it is cleared entirely by removing false alarms rather than by
improving the score at all. The budget bottleneck, by contrast, survives the
whole range: no score quality up to 0.99 holds α = 0.05. And the third thing is
that **AUROC turns out to be the wrong target** — the probe converts its ranking
into answers worse than a synthetic score of identical AUROC, because it ranks
late steps and only the first bad step can be repaired.

`docs/THESIS.md` listed three possible outcomes before this run and called the
negative one the most interesting. It arrived by a different route than
predicted — not because reach is universally low, but because the failures are
distributed across the whole pipeline.

## 7.8 Limits

- **182 test questions, 925 test steps.** The α-sweep gap (0.149 against a 0.05
  target) is far too large to be sampling noise, but finer between-condition
  differences are not resolvable.
- **Uncertainty is recovered, not native.** Features come from teacher-forcing
  the sampled text rather than from the sampling pass. That measures how
  surprising the model finds the step, which is what the gate consumes, but it
  is not identical to the generation-time distribution.
- **Three equivalence relations, one family.** §7.2 now reports bidirectional
  entailment alongside numeric equivalence and exact match, but the three nest
  on a single permissiveness axis and share the generations, K, temperature and
  clustering algorithm. A relation sensitive to the *asserted quantity* on
  arithmetic steps while still handling the 59% that carry none would be a
  genuinely different probe, and does not exist here.
- **The probe is linear, and that is the open variable.** Training data is a
  small one: §7.6 doubles it for about +0.01, with the interval spanning zero.
  ReProbe's probes are not linear, so statements here about what internal
  states *cannot* encode are statements about what a logistic probe on one
  layer cannot read off them. Every statement about what AUROC 0.70 fails to
  buy is unaffected.
- **108 first-bad steps in the corpus, 40 in test.** The first-bad-target result
  trains on 49 positives pooled and 27 gate-safe, and its significance depends
  on which global-target probe it is compared against. That is the binding
  constraint on §7.6's constructive half, and more solutions would relieve it —
  the one purpose extra data would clearly serve here.
- **Chain topology.** Replay assumes each step depends on the previous one.
  Influence weighting is off by default, having lost to uniform five times, so
  this affects little — but it is an assumption.
