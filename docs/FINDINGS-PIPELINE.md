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
~10 min) plus resampled semantic divergence
(`scripts/gpu_semantic_divergence.py`, ~4h). Corpus, features and the raw
samples are committed.

**The result is negative, and it is the most useful thing this project has
measured.** The proposal CAR started from — calibrate a threshold on generator
uncertainty, gate verification with it — does not control the risk it claims
to, and the reason is measurable at three separate points in the chain.

---

## 1. Generator uncertainty barely predicts global step error

```
score AUROC for detecting a globally-wrong step   0.5589   token-level only
                                                 0.5742   with semantic divergence
```

Token entropy, max surprisal and mean log-probability, combined and
standardised on dev data. `mean_logprob` on good steps is −0.2916; on bad steps
−0.3323. The direction is right and the separation is negligible.

This is not a harness failure. The same pipeline on synthetic features with a
one-standard-deviation separation gives AUROC **0.8668**, and on pure noise
**0.4828**. The machinery detects signal when there is signal.

### Semantic divergence does not rescue it

`semantic_divergence` is the signal the original spec weighted most heavily,
and it could not be recovered by teacher-forcing — it is defined over
independently sampled continuations. So it was resampled: K=5 continuations of
the next step for all 2,573 steps, clustered by numeric equivalence
(`scripts/gpu_semantic_divergence.py`, Kaggle T4×2, ~4h, 12,865 generations).

| feature set | AUROC (test) | AUROC (all steps) |
|---|---|---|
| token-level only | 0.5589 | 0.5277 |
| **semantic divergence only** | **0.5740** | **0.5488** |
| both | 0.5742 | 0.5427 |

Divergence is 0.4115 on globally-correct steps and 0.4809 on wrong ones — the
right direction, and worth 0.07 on a [0,1] scale. Semantic divergence is
marginally the better of the two signals and **combining them adds nothing**
(0.5740 → 0.5742); the two scores correlate at **r = +0.44**, so they are
largely the same information.

Re-running the whole pipeline with it weighted in moves the α = 0.05 selective
risk from 0.1468 to 0.1491. **Nothing about risk control changes.**

This is a direct replication, on a second benchmark, of
[arXiv:2602.02427](https://arxiv.org/html/2602.02427)'s claim that
sampling-agreement methods are weaker at *intermediate step* level than at
whole-answer level.

### The equivalence relation was load-bearing

The same samples, re-clustered by string equality instead of numeric value:

| clustering | mean divergence | unanimous steps | AUROC |
|---|---|---|---|
| numeric equivalence | 0.4239 | 1006 / 2573 | **0.5488** |
| exact string match | 0.6468 | 524 / 2573 | **0.4904** |

Exact match inflates divergence by half and drives AUROC *below chance*. Qwen
writes one computation as `<<48/2=24>>`, `\( 48 / 2 = 24 \)` and `48 / 2 = 24`;
string equality calls those three meanings, so measured "disagreement" tracks
notational variety, which is a property of verbosity rather than of doubt.

Had the default equivalence been used, this experiment would have reported that
semantic divergence is anti-predictive — a different conclusion, and a wrong
one. The ablation is free because it re-clusters the stored samples rather than
resampling, which is also the sounder comparison: it isolates the relation
instead of confounding it with a fresh draw.

---

## 2. So the gate does not control risk, at any α that binds

Base risk on test steps is **0.1578**. Any α above that is satisfied by
verifying nothing, so only α < 0.1578 tests anything at all.

With the full score (token-level + semantic divergence):

| α | Kotte floor | binds? | split conformal risk | verify % | CAR risk | verify % |
|---|---|---|---|---|---|---|
| 0.05 | 11.4% | **yes** | **0.1491** | 4.6% | **0.1555** | 18.4% |
| 0.10 | 6.4% | **yes** | **0.1494** | 9.2% | **0.1567** | 19.7% |
| 0.15 | 0.9% | **yes** | **0.1516** | 14.1% | **0.1582** | 22.5% |
| 0.20 | 0.0% | no | 0.1502 | 17.0% | 0.1552 | 23.9% |
| 0.30 | 0.0% | no | 0.1538 | 21.8% | 0.1587 | 26.9% |

At α = 0.05 the measured selective risk is **three times the target**. The risk
hardly moves across the entire sweep — 0.1491 to 0.1587 — while verification
climbs from 4.6% to 26.9%. The gate spends budget and buys nothing.

Token-level features alone give the same picture (0.1468 at α = 0.05, 0.1570 at
α = 0.30), which is the point: adding the spec's favoured signal changed the
score by 0.015 AUROC and changed risk control by nothing.

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
- **Semantic divergence is measured under three equivalence relations** (see
  [FINDINGS-ENTAILMENT.md](FINDINGS-ENTAILMENT.md)) **but they nest.**
  Numeric equivalence fits arithmetic steps; bidirectional entailment might
  cluster differently on the prose steps, which are 59% of the corpus. The raw
  samples are committed (`runs/semantic_samples.jsonl`) so another relation can
  be tried with no GPU at all.
- **182 test questions, 925 test steps.** Small. The α-sweep gap (0.147 vs a
  0.05 target) is far too large to be sampling noise, but the finer
  between-condition differences are not resolvable.
- **Chain topology.** Replay assumes each step depends on the previous one.
  Influence weighting is off by default (it lost to uniform in five separate
  tests), so this affects little, but it is an assumption.
