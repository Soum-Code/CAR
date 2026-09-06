# The assembled gate, end to end — and why it does not work

Chapters 4, 5 and 6 measured one piece each. This runs the whole control loop
on real model output: real generator uncertainty, real step labels, real
conformal calibration, real budget.

```bash
python scripts/exp_gate_pipeline.py
python scripts/exp_gate_pipeline.py --synthetic-signal 0.0   # null control
```

500 Qwen2.5-7B-Instruct solutions on GSM8K test, 2,573 steps, split
175 dev / 143 calibration / 182 test by hash of example id. Uncertainty
recovered by teacher-forcing (`scripts/gpu_score_uncertainty.py`, Kaggle T4×2,
~10 min). Corpus and features are committed.

**The result is negative, and it is the most useful thing this project has
measured.** The proposal CAR started from — calibrate a threshold on generator
uncertainty, gate verification with it — does not control the risk it claims
to, and the reason is measurable at three separate points in the chain.

---

## 1. Generator uncertainty barely predicts global step error

```
score AUROC for detecting a globally-wrong step   0.5589
```

Token entropy, max surprisal and mean log-probability, combined and
standardised on dev data. `mean_logprob` on good steps is −0.2916; on bad steps
−0.3323. The direction is right and the separation is negligible.

This is not a harness failure. The same pipeline on synthetic features with a
one-standard-deviation separation gives AUROC **0.8668**, and on pure noise
**0.4828**. The machinery detects signal when there is signal.

`semantic_divergence` is absent — it requires independently sampled
continuations and cannot be recovered by teacher-forcing, so its weight is
zeroed. The literature already calls it [contested at step
level](https://arxiv.org/html/2602.02427), but this result does not test it.

---

## 2. So the gate does not control risk, at any α that binds

Base risk on test steps is **0.1578**. Any α above that is satisfied by
verifying nothing, so only α < 0.1578 tests anything at all.

| α | Kotte floor | binds? | split conformal risk | verify % | CAR risk | verify % |
|---|---|---|---|---|---|---|
| 0.05 | 11.4% | **yes** | **0.1468** | 5.4% | **0.1557** | 18.5% |
| 0.10 | 6.4% | **yes** | **0.1511** | 9.5% | **0.1589** | 20.1% |
| 0.15 | 0.9% | **yes** | **0.1501** | 13.2% | **0.1575** | 22.2% |
| 0.20 | 0.0% | no | 0.1507 | 16.5% | 0.1597 | 24.0% |
| 0.30 | 0.0% | no | 0.1570 | 22.6% | 0.1610 | 26.6% |

At α = 0.05 the measured selective risk is **three times the target**. The risk
hardly moves across the entire sweep — 0.1468 to 0.1610 — while verification
climbs from 5.4% to 26.6%. The gate spends budget and buys nothing.

### Why conformal calibration does not save it

Split conformal fits the threshold so that the acceptance region *covers* 1−α
of correct steps. That is coverage, and coverage is not selective risk. When
the score is uninformative, the accepted region contains wrong steps at
approximately the base rate no matter where the threshold sits — the guarantee
holds and the quantity of interest is untouched.

This project's README has always said *"coverage is not accuracy."* This is
that sentence, measured.

> A conformal guarantee is a statement about the acceptance rule, not about the
> risk of what it accepts. With an uninformative score the two come apart
> completely, and nothing in the calibration procedure reports that.

---

## 3. The best verifier we measured is net-negative in deployment

Projected final-answer accuracy, baseline (no gate) **0.8022**:

| verifier | scope | false alarm | always verify | random gate | split conformal |
|---|---|---|---|---|---|
| arithmetic, step-local | 0.0000 | 0.0000 | 0.8022 | 0.8022 | 0.8022 |
| independent judge | 0.2283 | 0.0200 | 0.8022 | 0.7802 | 0.7857 |
| task PRM | 0.9033 | 0.0987 | **0.7637** | **0.7143** | 0.7802 |
| **task PRM, ablation: FA = 0** | 0.9033 | **0.0000** | **0.9231** | 0.8516 | 0.8736 |

The highest-scope verifier available **loses 4 points of accuracy** at its
measured operating point, and more verification makes it worse. Zero out its
false-alarm rate and the same verifier gains **12 points**.

The whole difference is a base-rate effect. Chapter 5 reported
`net = scope − false alarm = 0.8047`, measured on a population *conditioned on
being arithmetic-blind inherited corruption* — every item in it was wrong. In
deployment the verifier is pointed at all steps, and **84.2% of them are
correct**, so the 9.87% false-alarm rate is applied to a population four times
larger than the one the 90.33% detection rate acts on.

> Verifier scope measured on a positive-only population overstates its
> deployment value. The figure of merit is not `scope − FA`; it is
> `scope × P(wrong)` against `FA × P(correct)`, and on a strong generator the
> second term dominates.

That correction applies to how C3 is stated in the README, and it is the
sharpest practical result here.

---

## What this means for the thesis

`docs/THESIS.md` listed three possible outcomes and called the negative one the
most interesting. This is it, and it is coherent with everything upstream
rather than in tension with it:

| chapter | finding | consequence here |
|---|---|---|
| C1 | verifiers certify local validity; 90% of wrong steps are locally valid | a calculator gate cannot move the answer at all (row 1 above) |
| C3 | only an independent, task-trained verifier has reach | and its false-alarm rate makes it net-negative on a realistic base rate |
| new | generator uncertainty does not rank global step error (AUROC 0.56) | so there is nothing for the conformal layer to calibrate |

The three compose into one statement:

> Selective verification of LLM reasoning fails at three independent points:
> the signal does not rank the risk, the calibration certifies a quantity that
> is not the risk, and the only verifier with reach costs more in false alarms
> than it recovers. Fixing any one of them is not sufficient.

That is a stronger contribution than a working gate would have been, because it
is a claim about the *design space* rather than about one system.

---

## Limits

- **The projection is pessimistic.** Final-answer accuracy is MODELLED, not
  measured — replay cannot regenerate text, so a caught error is *assumed*
  repaired and a false alarm is *assumed* fatal to the answer. A real system
  might revise a correct step to another correct value. The direction of the
  false-alarm result is robust (the FA=0 ablation isolates it), the magnitude
  is not. With scope 0 and no false alarms the projection reduces exactly to
  the observed accuracy, and a test pins that.
- **Uncertainty is recovered, not native.** Features come from teacher-forcing
  the sampled text, not from the sampling pass. That measures how surprising
  the model finds the step, which is what the gate consumes, but it is not
  identical to the generation-time distribution.
- **No semantic divergence.** The one uncertainty signal this project's own
  spec weighted most heavily is absent. AUROC 0.56 is a result about
  token-level signals only, and a semantic signal could do better — that is the
  clearest thing left to test, and it costs a full resampling run.
- **182 test questions, 940 test steps.** Small. The α-sweep gap (0.147 vs a
  0.05 target) is far too large to be sampling noise, but the finer
  between-condition differences are not resolvable.
- **Chain topology.** Replay assumes each step depends on the previous one.
  Influence weighting is off by default (it lost to uniform in five separate
  tests), so this affects little, but it is an assumption.
