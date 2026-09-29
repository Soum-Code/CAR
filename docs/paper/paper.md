# Coverage Is Not Risk: Locating Where Selective Verification of LLM Reasoning Fails

*Workshop submission draft — target: NeurIPS/ICLR workshop on LLM evaluation or
uncertainty. ~8 pages excluding references.*

---

## Abstract

A natural design for making multi-step LLM reasoning reliable is to score each
step's uncertainty, calibrate a threshold with conformal prediction, and spend a
verification budget on the steps the calibrated score flags. Every component has
substantial literature behind it. We built the assembled system and measured it,
and it does not work. The contribution is locating *why*, at component
granularity, using two controls that separate cause from consequence.

Three findings. First, the quantity a verifier certifies is not the quantity of
interest: on 93,129 Math-Shepherd steps, 78.5% of globally-wrong steps are
arithmetically perfect, wrong only because a premise was. Re-measured on two
further generators this fraction rises monotonically with model accuracy: 0.7848, 0.8738,
0.9040 at roughly 45%, 68.4% and 80% GSM8K. So a deterministic
verifier becomes *less* useful as generators improve. Second, the assembled gate
holds its conformal coverage guarantee while missing its selective-risk target
by 3× at α = 0.05, and reports nothing amiss. Third, an oracle score takes
selective risk 0.1554 → 0.0885 and projected accuracy 0.79 → 0.98 on a third of
the verification calls, which shows the verifier was never the defect. It was
being aimed by a near-chance score. The oracle still misses α = 0.05 by 1.8×,
isolating the budget as the one genuinely separate bottleneck.

A sweep over synthetic scores of controlled AUROC locates the score quality the
verifier needs at **AUROC ≈ 0.65**, below a probe we already trained, and shows
that threshold is made entirely of false alarms: at a 0% false-alarm rate the
same verifier pays for itself down to AUROC 0.55. The sweep also refutes the
metric it is built on. Two scores of identical AUROC differ by 0.04 projected
accuracy, because error propagation means only the *first* bad step in a
solution can be repaired, and a score that ranks late steps earns AUROC it
cannot convert.

---

## 1. Introduction

A language model answering a multi-step question emits a chain of intermediate
claims. Any claim can be wrong, and a wrong claim early is not merely a local
defect: everything after it is generated conditioned on it. The practical
question is where to spend a limited verification budget.

The obvious design has three parts, each with its own literature. **A score.**
Token entropy, surprisal, or disagreement across samples. **A calibration.**
Conformal prediction, which turns that score into a threshold carrying a
distribution-free guarantee [vovk2005alrw; angelopoulos2024crc]. **A gate.**
Verify the steps above the threshold, subject to a budget, using a process
reward model [lightman2023verify; wang2024mathshepherd] or a deterministic
checker. Assembling them is a natural system.

We assembled it. It does not control risk, and the reasons are separable and
measurable.

**CAR is the instrument, not the claim.** Conformalized Agentic Reasoning is
the harness we built to run that loop: a generator, a step-level score, a
conformal calibrator, a budgeted gate and a pluggable verifier, all behind one
interface. We make no claim that it works, and the paper is not a system
paper. Its value here is that every condition below runs through *the same*
loop with a single component swapped, rather than through parallel
implementations of each baseline. That design is what upgrades "the system
failed" into "the failure is here and not there."

Two controls do the separating work: an **oracle score** that reads the label
while holding the calibrator, budget and verifier fixed, and an **ablation** of
the verifier's false-alarm rate to zero while holding its detection rate.
Neither is deployable. Both are diagnostics.

**Contributions.**

1. A measurement of the gap between what a verifier certifies (local validity)
   and what determines the answer (global correctness), on 93,129 labelled steps
   and two further corpora generated for this work. The gap *widens*
   monotonically as the generator improves (§3).
2. A measurement of verifier reach across four verifier classes on the
   population deterministic checking provably cannot see, showing reach requires
   generator-independence **and** task specialisation, and neither alone
   suffices (§4).
3. An end-to-end measurement of the assembled gate showing conformal coverage
   holds while selective risk misses α by 3×, with an oracle establishing that
   the verifier result is *downstream of the score* rather than an independent
   defect (§5).
4. A sweep over synthetic scores of controlled AUROC locating the crossing at
   ≈ 0.65, showing it is composed entirely of false alarms, and demonstrating
   that AUROC is not a sufficient figure of merit for a step-level score (§6).

---

## 2. Setup

### 2.1 The decomposition

Every measurement rests on separating two failure modes of a reasoning step:

```
global_correct(t)  =  local_valid(t)  AND  NOT premise_corrupt(t)
```

**Local invalidity** means the step does not follow from its own premises;
`47 × 3 = 131` is wrong in any context and a calculator catches it.
**Inherited corruption** means the step follows perfectly and a premise is
false. No amount of checking the step reveals anything.

A verifier reports the first. It is a function of the step and at most a
bounded window of context, with no access to the truth of its premises unless it
can independently establish them. Conformal machinery calibrates whatever the
verifier reports, so it calibrates local validity.

The two are separately observable on GSM8K [cobbe2021gsm8k], which is what makes
this study possible. Inline `<<expr=result>>` annotations give local validity
deterministically, with no model and no judge. Math-Shepherd
[wang2024mathshepherd] supplies global labels by Monte-Carlo rollout: a step is
`+` if any completion from that prefix reaches the gold answer. So the label is
*global* correctness, and a step inherits `-` from a corrupted premise even when
it is itself impeccable. Having both signals on the same steps turns the gap
from a thought experiment into a measurement.

### 2.2 Two estimators

The fraction of globally-wrong steps that are locally valid can be computed two
ways, and the difference only becomes visible across generators:

```
C1_all        =  P(inherited) / P(global error)          over ALL steps
C1_checkable  =  P(local_valid | global error, checkable)
```

`C1_all` is biased downward by the uncheckable rate: a step with no arithmetic
never enters the numerator but always sits in the denominator. Where 91.4% of
steps are checkable the bias is small; on a generator whose steps are 59% prose
it collapses the estimate to 0.25 and says nothing about reasoning. We report
`C1_checkable` throughout.

### 2.3 Corpora and the instrument

**Math-Shepherd**: 93,129 GSM8K steps from 25,971 Mistral-7B-SFT solutions. It
is a PRM *training* set with a deliberate class mix, so no unconditional rate
can be read off it; every rate is reported per stratum and post-stratified.

**Two generated corpora**, produced for this work under identical settings (500
GSM8K test problems, K = 4 rollouts, temperature 0.7, 4-shot) and written in
Math-Shepherd's own format so one analysis path reads all three:
Qwen2.5-7B-Instruct (2,573 steps) and Llama 3.1 8B Instruct (1,938 steps).

**The instrument.** All conditions in §5 run through the same agent loop with
one component replaced: plain chain-of-thought, always-verify, a random gate at
matched budget, a quantile gate, split conformal, adaptive conformal with
inverse-propensity weighting, and the oracle. Verifiers enter parameterised by
the reach measured in §4 rather than as whatever model happens to be wired in.

**A null-hypothesis control.** The same pipeline on synthetic features with
one-standard-deviation separation scores AUROC 0.8668, and on pure noise 0.4828.
The machinery detects signal when signal exists, which is what licenses reading
the measured values as properties of the signal rather than of the harness.

---

## 3. The certified quantity is not the quantity of interest

Within wrong-answer solutions on Math-Shepherd, local error is 0.1708 and global
error 0.7106. So 78.5% of globally-wrong steps are arithmetically perfect.
Controlling local selective risk at level α bounds nothing about the answer.

Corruption is close to absorbing. Of 33,236 steps after the first globally-bad
step, 95.9% remain bad, and 0 of the 14,573 solutions that could have
recovered ever do, where recovery means every subsequent step is good.

### 3.1 The gap widens as generators improve

The obvious objection is that Mistral-7B-SFT is a weak 2023 model and a better
generator closes the gap by itself. It does the opposite.

| | Mistral-7B-SFT | Llama 3.1 8B | Qwen2.5-7B |
|---|---|---|---|
| GSM8K accuracy | ~45% | **68.4%** | **80.0%** |
| local error | 0.1708 | 0.1034 | 0.0813 |
| global error | 0.7106 | 0.7442 | 0.5772 |
| **C1 (checkable)** | **0.7848** | **0.8738** | **0.9040** |
| *n* globally-wrong checkable steps | 46,555 | 309 | 125 |
| 95% Wilson CI | [0.781, 0.789] | [0.832, 0.906] | [0.840, 0.944] |

C1 is **monotone in generator accuracy**. The Llama point was a prediction
before it was a measurement: this table had two rows, and the claim that a model
between the two would fall between them was written down first. Mistral's
interval is disjoint from both others.

Local error is monotone on the same ordering (0.1708, 0.1034, 0.0813) and that
is the mechanism. Global error is *not* monotone, so C1 does not rise because
the numerator grows. It rises because the denominator of arithmetic slips
shrinks faster than inherited corruption does.

> A deterministic verifier becomes *less* useful as the generator improves.

Two further quantities are generator-dependent and also monotone: base risk μ
(0.3908, 0.2428, 0.1221) and corruption persistence (95.9%, 81.6%, 66.4%, with
0/14,573, 12/141 and 6/83 eligible solutions recovering). "Near-absorbing" is
therefore a property of weak generators that decays smoothly, not an
idiosyncrasy of one model.

### 3.2 Feasibility is constrained before any method is chosen

[kotte2026certify, Prop. 3] shows that when base risk μ exceeds target α, any
distribution-free method must verify or abstain on at least `(μ − α)/(1 − α)` of
items. This is a closed form checkable in advance, and it is not a small
constraint: at μ = 0.3908 a target of α = 0.10 charges **32.3%** of the budget
as an entry fee. The floor falls monotonically with generator accuracy (32.3%,
15.9%, 2.5%), so any statement about attainable α must name its model. We use
the floor as a reference line rather than a result.

---

## 4. Verifier reach is the controlling variable, and it is semantic

If a verifier misses inherited corruption because it sees too little context,
widening the lookback should help. On 35,535 steps, arithmetic reach rises from
0.0000 at step-local to **0.1999** unbounded, and saturates. The ceiling is
structural: **80.0%** of those steps have no upstream arithmetic error at all.
The mistake is in the setup, not the calculation.

On the population arithmetic provably cannot see:

| verifier | independent? | task-trained? | scope | false alarm |
|---|---|---|---|---|
| arithmetic, unbounded lookback | yes | — | 0.1999 | — |
| same-model critic (the generator) | **no** | no | **0.0000** | 0.0000 |
| independent judge (Qwen2.5-7B) | yes | no | 0.2283 | 0.0200 |
| task PRM (Math-Shepherd-7B) | yes | **yes** | **0.9033** | 0.0987 |

The same-model critic detects zero errors. The model that wrote these
solutions, asked whether they are sound, approves everything. That is
[huang2024selfcorrect] measured directly at step level.

> Reach is not how far back you look. It requires independence from the
> generator **and** task specialisation; neither alone suffices.

The number that matters most in §5 is not the 0.9033. It is the **0.0987** next
to it.

---

## 5. The assembled gate

500 Qwen solutions, 2,573 steps, split 175 dev / 143 calibration / 182 test by
hash of example id. Base risk on test steps is 0.1578, so only α < 0.1578 tests
anything at all.

### 5.1 The signal does not rank the risk

```
AUROC for detecting a globally-wrong step:
  0.5589   token-level (entropy, max surprisal, mean log-probability)
  0.5740   resampled semantic divergence
  0.5742   both
```

Combining them buys **0.0002**; the two correlate at r = +0.44 and are largely
the same information. Against the synthetic control at 0.8668 and noise at
0.4828, this is a property of the signal.

Semantic divergence is the signal the original design weighted most heavily, and
measuring it required deciding when two sampled continuations mean the same
thing. That choice is load-bearing: under exact string matching, AUROC falls
*below chance* to 0.4904, because measured disagreement then tracks notational
variety rather than doubt. Had the library default been used, this section would
have reported that semantic divergence is anti-predictive: a clean, believable,
wrong conclusion. We also implement the reference relation, bidirectional
entailment under an NLI model [farquhar2024semantic], and it does not rescue the
signal: 0.5625, 95% CI [0.512, 0.614]. The three relations nest on a single
permissiveness axis, calling 0.1%, 31.9% and 78.0% of pairs equal, and across
that entire range the measurement does not move.

### 5.2 Coverage holds; risk does not

| α | split conformal risk | verify % |
|---|---|---|
| 0.05 | **0.1491** | 4.6% |
| 0.10 | 0.1494 | 9.2% |
| 0.15 | 0.1516 | 14.1% |
| 0.30 | 0.1538 | 21.8% |

At α = 0.05 measured selective risk is **three times the target**. Risk barely
moves across the sweep (0.1491 → 0.1587) while verification climbs from 4.6% to
26.9%. The gate spends budget and buys nothing.

The mechanism is that split conformal fits the threshold so the acceptance
region *covers* 1 − α of correct steps. That is coverage. With an uninformative
score the accepted region contains wrong steps at approximately the base rate
wherever the threshold sits: the guarantee holds perfectly and the quantity of
interest is untouched.

> A conformal guarantee is a statement about the acceptance rule, not about the
> risk of what it accepts. With an uninformative score the two come apart
> completely, and **nothing in the procedure reports that.**

### 5.3 The oracle locates the fault

Projected final-answer accuracy against a no-gate baseline of 0.8022, with the
task PRM at its measured operating point:

| score | calls/q | first-bad recall | selective risk | PROJ accuracy |
|---|---|---|---|---|
| no gate | 0.00 | — | 0.1554 | 0.8022 |
| real, split conformal | 1.09 | 0.3590 | 0.1538 | 0.7912 |
| real, always verify | 1.89 | — | — | **0.7637** |
| probe (AUROC 0.6968) | 0.96 | 0.3077 | 0.1394 | 0.7802 |
| **oracle** | **0.37** | **0.4366** | **0.0885** | **0.9780** |

The highest-reach verifier available **loses 4 points of accuracy** at its
measured operating point, and more verification makes it worse. Behind a perfect
score the *same verifier at the same 9.87% false-alarm rate* gains **17.6
points**, on a third of the calls.

This is a base-rate effect. §4 reported `net = scope − FA = 0.8047`, measured on
a population conditioned on being inherited corruption, where every item was
wrong. In deployment the verifier sees whatever the gate selects, and behind a
near-chance score that is mostly correct steps.

> The figure of merit is not `scope − FA`. It is conditioned on what the gate
> selects: `scope × P(wrong | verified)` against `FA × P(correct | verified)`.
> The score sets that conditioning, so improving the score raises the verifier's
> net value without touching the verifier.

**The verifier was never the defect.** It is a good verifier aimed badly. That
correction collapses what an earlier analysis treated as three independent
failures into two.

But the oracle still misses α = 0.05 by 1.8×. At two calls per question over
a mean of 5.15 steps, most wrong steps go unverified however perfectly they are
ranked. The residual 0.0885 is the budget, not the score. Two bottlenecks, and
they are separable: the score is worth 0.154 → 0.089, and the budget is what
stands between 0.089 and 0.05.

### 5.4 A better signal exists, and is not enough

A logistic probe on the generator's own frozen hidden states, in the spirit of
[ni2025reprobe], reaches **AUROC 0.6968** against 0.5742 for everything else. So
the failure is not that step-level uncertainty is unreadable. It improves
selective risk at every α on fewer calls, and still misses α = 0.05 by 2.9×,
and leaves the task PRM net-negative at 0.7802.

Two disciplines are worth stating because they changed the result. The
selection-split AUROC is **0.8748** against 0.6968 on test, a 0.18 winner's
curse from choosing the best of 29 layers × 5 regularisation strengths on 287
selection steps. And an earlier draft called 0.6968 a floor on a learning curve
that had been scored on the selection split; measured properly at fixed layer
and regularisation, doubling the training data is worth **+0.0105, CI [−0.035,
+0.057]**.

---

## 6. How good does the score need to be, and is AUROC the right question?

Score quality can be made a dial. Under a binormal model a globally-wrong step
draws from N(d, 1) and a correct step from N(0, 1), so AUROC = Φ(d/√2) and the
separation inverts in closed form as d = √2·Φ⁻¹(AUROC). Everything else is held
fixed (same corpus, splits, calibrator, budget, and verifier at its measured
scope and false-alarm rate), so only the ranking moves. 13 grid points × 64
seeds.

| AUROC | sel. risk | 1st-bad recall | PROJ acc | |
|---|---|---|---|---|
| 0.550 | 0.1536 | 0.2853 | 0.7782 | net − |
| 0.625 | 0.1463 | 0.3786 | 0.7940 | net − |
| 0.650 | 0.1441 | 0.4111 | 0.8010 | ~same |
| **0.700** | 0.1405 | 0.4635 | **0.8119** | **net +** |
| 0.990 | 0.1258 | 0.7736 | 0.8712 | net + |

The crossing is at AUROC ≈ 0.65, below the probe that already exists.

And the threshold is made entirely of false alarms. Rerun with the
verifier's false-alarm rate switched off and there is no crossing to find: at
FA = 0 the same verifier is worth having at *every* score quality tested, down
to 0.55. Halving a verifier's false-alarm rate lowers the score quality you need
more than raising its detection rate does.

No score quality holds a binding α at this budget. The best any row manages
at α = 0.05 is 0.0956, at AUROC 0.99, still 1.9× the target. §5.3 showed that
at one point with a binary oracle; the sweep shows it across the whole range,
which rules out reading the oracle's failure as an artifact of its degenerate
score distribution. Score quality is not the binding constraint on risk control.
The budget is.

### 6.1 AUROC is not a sufficient description of a score

Pinning verification at 19.1% so that only the ranking differs:

| | AUROC | 1st-bad recall | PROJ acc |
|---|---|---|---|
| synthetic score | 0.6974 | **0.3413** | **0.8206** |
| probe, layer 25 | 0.6968 | **0.3077** | **0.7802** |

Two scores of near-identical AUROC, differing by **0.04 projected accuracy**.
The probe sits at the 0th percentile of 64 synthetic draws at the same AUROC. It
catches slightly *more* wrong steps and rescues *fewer* answers, and the
mechanism is measurable:

```
corr(normalised step position, probe score)      +0.1813
corr(normalised step position, composite score)  -0.2766
```

The probe flags late steps. Local error rises with position (corr = +0.950,
doubling from 11% at step 1 to 22% at step 8), so a score that chases positional
difficulty is rewarded on AUROC. But the projection only pays for the **first**
bad step, because everything downstream inherits corruption that repairing a
later step does not undo.

> Two scores with identical AUROC are worth different amounts. Under a
> propagation objective, what counts is which errors a score ranks highly, and
> the ones that count are the earliest. Rank by AUROC to compare with the
> literature; select on first-bad-step recall.

---

## 7. Related work

**Conformal gating.** [khosravi2026csa] controls selective risk with
anytime-pathwise validity, and its Appendix Thm E.1 gives sparse-verifier
validity via Bernoulli subsampling with importance weighting, the
censored-feedback machinery this work uses and cites rather than claims.
[kotte2026certify] supplies the impossibility bound we use as a feasibility
instrument. Both operate on **independent rounds**: one query, one answer, act
or abstain. In multi-step reasoning that independence is false by construction,
since accepting a wrong step changes the distribution of every step after it.
[barber2023beyond] relaxes exchangeability for *exogenous* drift; the violation
here is endogenous, so their weighted-quantile machinery is the closest
available starting point rather than a solution.

**Step scoring.** [lightman2023verify] established process over outcome
supervision; [ni2025reprobe] shows a sub-10M-parameter probe on frozen internal
states matching PRMs up to 810× larger, and reports that probe and PRM combine
better than either alone. [wen2026embedding] argues embedding perturbation
reflects intermediate-step uncertainty better than sampling-based agreement; we
measure the sampling-based half and find it near chance, which is their
conclusion reached from the other direction.

**Verifier placement.** [ro2025sherlock] selects verifiers per node on a known
workflow DAG using counterfactual fault injection and a learned cost model,
motivated by an observation we reach independently: verifier accuracy and cost
are not monotonically related. We do not claim that observation as new. What §4
adds is the decomposition. Measured on the population deterministic checking
cannot see, reach requires independence **and** specialisation, a
distinction a cost-versus-accuracy selector cannot make, because both properties are invisible
to it.

**Propagated errors.** [you2025probabilistic] states our §3 diagnosis
independently and gives a structural remedy: score each step solely against
*previously-verified* premises, reaching 90.3% F1 on propagated errors on their
own synthetic corpus. This is a result we must accommodate rather than dismiss.
It shows the inherited-corruption population is not intrinsically invisible. It
is invisible to a verifier reading a step against the generator's **unverified**
context, as every verifier in §4 does. It also assumes a verified prefix, which
is precisely what a budgeted gate cannot supply. Reconciling the two is the
clearest open direction this work points at.

---

## 8. Limitations

**Sample size.** 182 test questions and 925 test steps in the end-to-end run.
The headline gap (0.149 against a 0.05 target) is far outside sampling noise;
finer between-condition differences are not. The corpus holds 108 first-bad
steps, 40 in test, which is the binding constraint on §6.1.

**A "step" is not model-invariant.** Math-Shepherd's step is one calculator
operation; Qwen writes 5.15 steps per solution of which 59% are narration
asserting no arithmetic. Per-step rates across generators are rates over
different units, and no better extraction fixes that.

**Projected, not measured, accuracy.** The corpus is fixed, so gating cannot
change what the model wrote. Final-answer accuracy under gating is therefore
projected under an explicit repair model and labelled as such. With scope 0 and
no false alarms the projection reduces exactly to observed accuracy. It is
pessimistic about false alarms: the FA = 0 ablation isolates the direction, the
magnitude is not robust.

**Three generators, one band.** All are 7–8B instruction-tuned models on GSM8K.
The monotone relationships in §3.1 are measured across a narrow slice, and
nothing here establishes they continue to a 70B model or to a task that is not
arithmetic.

**Label semantics.** Math-Shepherd's labels are Monte-Carlo estimates of "leads
to a correct answer," not proofs, so measured global error is a lower bound.

---

## 9. Conclusion

Selective verification of LLM reasoning, assembled from components that each
work in isolation, does not control risk. The failures are locatable.

The signal does not rank the risk: neither token-level uncertainty (0.5589) nor
sampling-based semantic divergence (0.5740), and combining them buys 0.0002.
Conformal calibration then holds its coverage guarantee while missing selective
risk by 3×, and reports nothing amiss. The verifier with real reach is
net-negative at its measured operating point. But that third failure is *not*
independent: behind a perfect score the same verifier gains 17.6 points, and it
turns positive at AUROC ≈ 0.65, below a probe we already trained.

What survives as genuinely separate is the **signal** and the **budget**. A
perfect score still misses α = 0.05 by 1.8×, and no score quality up to 0.99
holds it. Better uncertainty estimation is necessary and not sufficient; better
calibration cannot help while the score does not rank; and the verifier was
never the problem.

Two results transfer beyond this setting. A deterministic verifier gets *less*
useful as generators improve, measured across three models with disjoint
intervals. And AUROC is the wrong figure of merit for a step-level score under
propagation: two scores of identical AUROC differ by 0.04 projected accuracy,
because only the first bad step can be repaired.

Each leg carries a measured number and a regression test, and the claim is
falsifiable in the only way that matters: a signal that ranks step error would
break it.

---

## Reproducibility

All code, data-fetching scripts, measurement outputs and GPU-run artifacts are
in a public repository. 356 tests, no GPU and no network required. Every refuted
claim that rests on a measurement carries a regression test, so a result cannot
silently stop reproducing.

---

## References

Author-year keys refer to `references.bib`; 21 entries, all verified against the
arXiv listing.

- angelopoulos2024crc — Conformal Risk Control. ICLR 2024. arXiv:2208.02814
- barber2023beyond — Conformal Prediction Beyond Exchangeability. *Ann. Statist.* 2023. arXiv:2202.13415
- cobbe2021gsm8k — Training Verifiers to Solve Math Word Problems. 2021. arXiv:2110.14168
- farquhar2024semantic — Detecting Hallucinations Using Semantic Entropy. *Nature* 2024
- huang2024selfcorrect — LLMs Cannot Self-Correct Reasoning Yet. ICLR 2024. arXiv:2310.01798
- khosravi2026csa — Conformal Selective Acting. 2026. arXiv:2605.20270
- kotte2026certify — When Can Conformal Risk Control Certify LLM Outputs? 2026. arXiv:2606.29054
- lightman2023verify — Let's Verify Step by Step. 2023. arXiv:2305.20050
- ni2025reprobe — ReProbe. 2025. arXiv:2511.06209
- ro2025sherlock — Sherlock: Reliable and Efficient Agentic Workflow Execution. 2025. arXiv:2511.00330
- vovk2005alrw — Algorithmic Learning in a Random World. Springer 2005
- wang2024mathshepherd — Math-Shepherd. ACL 2024. arXiv:2312.08935
- wen2026embedding — Embedding Perturbation... 2026. arXiv:2602.02427
- you2025probabilistic — Probabilistic Soundness Guarantees in LLM Reasoning Chains. 2025. arXiv:2507.12948
