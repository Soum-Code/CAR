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
updates.

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

This is a direct replication, on a second benchmark, of
[arXiv:2602.02427](https://arxiv.org/html/2602.02427)'s finding that
sampling-agreement methods are weaker at *intermediate step* level than at
whole-answer level.

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


![Projected accuracy by verifier. The FA = 0 ablation isolates the false-alarm rate as the whole of the difference. MODELLED, not measured.](figures/fig9-verifier-value.png)

**Figure 7.4.** Projected accuracy by verifier. The FA = 0 ablation isolates the false-alarm rate as the whole of the difference. MODELLED, not measured.

The entire difference is a base-rate effect. Chapter 5 reported
`net = scope − false alarm = 0.8047`, measured on a population *conditioned on
being arithmetic-blind inherited corruption* — every item in it was wrong. In
deployment the verifier is pointed at all steps, and **84.2% of them are
correct**, so the 9.87% false-alarm rate acts on a population four times larger
than the one the 90.33% detection rate acts on.

> Verifier scope measured on a positive-only population overstates its
> deployment value. The figure of merit is not `scope − FA`; it is
> `scope × P(wrong)` against `FA × P(correct)`, and on a strong generator the
> second term dominates.

This is a correction to how C3 should be stated, and it is the sharpest
practical result in the thesis. It also explains the first row: with scope
0.0000 and no false alarms, a calculator gate cannot move the answer *at all* —
exactly what C1 predicts.

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

## 7.6 What this chapter establishes

Three independent failures, each with a number and a regression test:

| failure point | measurement |
|---|---|
| the signal does not rank the risk | AUROC 0.5589 / 0.5740 / 0.5742 |
| the calibration certifies the wrong quantity | coverage holds; selective risk misses α by 3× |
| the verifier with reach costs more than it recovers | 0.7637 against a 0.8022 baseline |

They compose into a claim about the design space rather than about one system,
and repairing any single one of them is not sufficient. Better uncertainty
estimation still leaves a calibration target that is not the risk. Better
calibration still leaves a score that does not rank. A better verifier still
has a false-alarm rate acting on a mostly-correct population.

`docs/THESIS.md` listed three possible outcomes before this run and called the
negative one the most interesting. It arrived by a different route than
predicted — not because reach is universally low, but because the failures are
distributed across the whole pipeline.

## 7.7 Limits

- **182 test questions, 940 test steps.** The α-sweep gap (0.149 against a 0.05
  target) is far too large to be sampling noise, but finer between-condition
  differences are not resolvable.
- **Uncertainty is recovered, not native.** Features come from teacher-forcing
  the sampled text rather than from the sampling pass. That measures how
  surprising the model finds the step, which is what the gate consumes, but it
  is not identical to the generation-time distribution.
- **One equivalence relation.** Numeric equivalence fits arithmetic steps;
  bidirectional entailment might cluster differently on the 59% of steps that
  are prose. The raw samples are committed
  (`runs/semantic_samples.jsonl`) so another relation can be tried with no GPU.
- **Chain topology.** Replay assumes each step depends on the previous one.
  Influence weighting is off by default, having lost to uniform five times, so
  this affects little — but it is an assumption.
