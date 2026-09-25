# What this corpus can and cannot resolve

Every AUROC in this thesis is measured on one corpus: 500 Qwen2.5-7B solutions to
GSM8K test questions, 2,573 steps, of which 925 are in the test split. Until this
analysis none of them carried an error bar.

That is a problem specific to what the chapters argue. Most of the claims are
about a *difference* between two AUROCs measured on the same steps, and two of
the load-bearing ones assert that a difference is **absent**:

| chapter | claim | shape |
|---|---|---|
| 7.2 | "the signal does not rank the risk", AUROC 0.5589 | absence |
| 7.2 | "combining them adds nothing" (0.5740 → 0.5742) | absence |
| 7.2 | exact match "drives AUROC below chance", 0.4904 | difference |
| 7.2 | the equivalence relation "was load-bearing", 0.5488 vs 0.4904 | difference |
| 7.6 | the probe reaches 0.6968, "+0.12 over everything above" | difference |
| 7.3 | "adding the favoured signal changed AUROC by 0.015" | absence |

A small point estimate is equally consistent with a small effect and a small
corpus. Nothing in the results could tell those apart, so this document does.

Run: `python scripts/exp_significance.py` (CPU, ~3 min, writes
`runs/significance.json`). Machinery: `car.eval.inference`, tested in
`tests/test_inference.py`.

---

## Method

4,000 percentile bootstrap replicates. Every score is evaluated on the same
steps, so every comparison is **paired on the same resamples** — the corpus
variability two scores share cancels, which matters here because it is most of
the variability. The resampling unit is the **solution**.

### The reason for clustering is not the obvious one

The obvious argument is this thesis's own finding. Chapter 4 measures that 49.6%
of wrong steps inside wrong-answer solutions are locally valid and wrong only
because a premise was: one corrupted premise turns every step below it wrong at
once, so step labels within a solution are strongly dependent, so 2,573 steps are
not 2,573 draws.

Measured, that argument is wrong. On synthetic data where every step in a
solution shares one label — the extreme of propagated corruption — the design
effect is **0.94**. AUROC is a two-sample rank statistic; it barely notices the
class balance of a resample moving around.

What does inflate it is a per-solution shift in the **score**: one verbose
question whose every step draws a high divergence, one terse question whose every
step draws a low one. That alone gives **1.45**. The two mechanisms together give
**4.3**. Both are pinned in `tests/test_inference.py`, including a test that fails
if the claim is reversed — which is how the original wording was caught.

Measured on the corpus:

| score | deff (test) | deff (all steps) | SE inflation (test) |
|---|---|---|---|
| token-level composite | 2.13 | 2.13 | 1.46× |
| both signals | 2.21 | 2.27 | 1.49× |
| semantic divergence (numeric) | 1.80 | 2.16 | 1.34× |
| semantic divergence (exact match) | 1.86 | 3.27 | 1.36× |
| probe on hidden states | 1.83 | — | 1.35× |

Every naive standard error on this corpus would have been **1.3× to 1.8× too
small**. Two intervals below would have excluded chance on that arithmetic and do
not on this one.

### What the intervals cover

Corpus sampling variability, with the score held fixed. **Not** the cost of
fitting the score — the dev-split scaler, the probe's layer and regularisation,
the choice of which signal to report. That component is real and larger, and
chapter 7 already quotes it separately: the probe scores 0.8748 on the split used
to pick its layer and 0.6968 on test, so selection costs 0.18 AUROC. Reading
these intervals as covering it would hide the bigger of the two effects.

---

## Test split: 925 steps, 182 solutions, wrong-step rate 0.1578

| score | AUROC | 95% CI | clears chance? |
|---|---|---|---|
| token-level composite | 0.5589 | [0.4780, 0.6328] | no |
| both signals | 0.5742 | [0.4915, 0.6504] | no |
| semantic divergence (numeric) | 0.5740 | [0.5054, 0.6433] | **yes** |
| semantic divergence (exact match) | 0.5287 | [0.4586, 0.5950] | no |
| probe on hidden states | 0.6968 | [0.6302, 0.7569] | **yes** |

## All 2,573 steps

| score | AUROC | 95% CI | clears chance? |
|---|---|---|---|
| token-level composite | 0.5277 | [0.4838, 0.5760] | no |
| both signals | 0.5427 | [0.4983, 0.5920] | no |
| semantic divergence (numeric) | 0.5488 | [0.5081, 0.5910] | **yes** |
| semantic divergence (exact match) | 0.4904 | [0.4379, 0.5459] | no |

The probe is fitted on part of this set, so its all-steps AUROC (0.8367) is
partly in-sample and is not a row here. The gap to its 0.6968 on test is the
winner's curse §7.6 already reports, not a better measurement.

## Paired differences

| comparison | Δ AUROC | 95% CI | p | verdict |
|---|---|---|---|---|
| numeric vs exact match | +0.0453 | [−0.0138, +0.0949] | 0.124 | **not established** |
| both vs token-level | +0.0153 | [−0.0068, +0.0369] | 0.148 | **not established** |
| both vs divergence alone | +0.0003 | [−0.0390, +0.0445] | 0.955 | consistent with zero, ±0.04 |
| probe vs best measured signal | +0.1226 | [+0.0287, +0.2288] | 0.004 | **survives** |
| probe vs token-level | +0.1379 | [+0.0428, +0.2427] | 0.002 | **survives** |

---

## What changed in the thesis

**The probe's advantage survives, and it is the only comparison in chapter 7
that does.** +0.1226 [+0.0287, +0.2288], p = 0.004. Every conclusion that rests
on "internal states carry a signal the measured features do not" stands.

**"Exact match drives AUROC below chance" does not survive.** 0.4904 has a CI of
[0.4379, 0.5459], which covers 0.5. Exact-match clustering is *indistinguishable
from chance*, not below it. The chapter said "below", and the difference matters,
because "below chance" reads as *anti-predictive* — a signal actively pointing
the wrong way — and nothing here supports that.

**"The equivalence relation was load-bearing" does not survive, as stated.** The
relation is unquestionably load-bearing on the *score*: mean divergence 0.4239
against 0.6468, unanimous steps 1,006 against 524, the same value on only 61.8%
of steps. On the *answer* it is +0.0453 [−0.0138, +0.0949], p = 0.124 —
consistent with being worth nothing. §7.2 now says so.

There is a better lesson in the wreckage of that claim. The chapter's point was
that had the library default been used, it would have reported semantic
divergence as anti-predictive. That is still true — and what prevented it was not
choosing the better relation. A point estimate under *either* relation was going
to be over-read. It was the interval. The bootstrap, not the clustering, is what
saved the chapter, and the clustering deserves its care for a different reason: a
score that tracks verbosity is wrong in ways that surface elsewhere.

**"Combining buys 0.0002" is consistent with zero, to ±0.04 and no better.** The
arithmetic on the point estimates is right and the precision it implies is not
there. Stated as the interval from now on.

**"Generator uncertainty does not rank global step error" survives, but divergence
alone needs a footnote.** The token-level composite's interval covers chance, so
the claim holds for it. Semantic divergence's does not: 0.5740 [0.5054, 0.6433]
clears 0.5 on test and on all steps. It carries information — a little, far below
usable. The refutation this project delivers is that the signal is unusable, and
§7.4's crossing at ≈ 0.65 is what makes that quantitative. "Uninformative" would
be the wrong word and is not used.

---

## Two diagnostics

### Divergence's seven values are not the explanation

Semantic divergence over K = 5 samples is not a continuous score. Its value
depends only on the block sizes of the sample partition, so it takes exactly
seven values — the partitions of 5 — and on the test steps 36% sit at 0.000, 18%
at 1.000, 13% at 0.828, 12% at 0.311. So its AUROC could have been limited by
resolution rather than by signal, and the comparison against a continuous probe
would then be unfair in a way no amount of bootstrapping would reveal.

Quantising a score onto divergence's own grid — same value distribution, same
ordering — prices the resolution with the signal held fixed:

| score | continuous | on the grid | cost |
|---|---|---|---|
| probe on hidden states | 0.6968 | 0.6837 | 0.0131 |
| token-level composite | 0.5589 | 0.5663 | −0.0074 |

The probe loses 0.013; the token-level composite moves within noise. **Resolution
is not what holds divergence at 0.5740.** Sampling five continuations and counting
meaning classes is a coarse instrument and that is not why it fails here.

### What a fourth equivalence relation would have to do

The pending bidirectional-entailment run (`EntailmentEquivalence`,
`microsoft/deberta-large-mnli`) exists to answer one question: is chapter 7's
negative result a property of sampling-based step uncertainty, or of
`numeric_equivalence` standing in for the relation Kuhn et al. and Farquhar et al.
use? This analysis prices that question before it is asked.

The paired SE between two relations on these 925 steps is **0.0272**. A third
relation is measured against the same steps with the same pairing, so it gets the
same SE. At 80% power the smallest detectable difference is **0.076** — a new
relation registers only above **0.6503** AUROC.

Chapter 7's score-quality sweep, a separate experiment on synthetic scores, puts
the AUROC at which the verifier stops costing more than it recovers at **≈ 0.65**.
The agreement to within 0.001 is coincidence, and it is the single most useful
fact about the unrun experiment:

> On this corpus a clustering relation can only be **heard** if it is already good
> enough to be **useful**.

An entailment point estimate of 0.60 would be indistinguishable from 0.5740 and
would change nothing. One above 0.65 would not be a better stand-in for the
signal — it would *be* a usable signal, and would belong beside the probe in
§7.6. Either way the run is worth doing, because a null result at this power is
still the answer to the objection; what it cannot be is a *small* positive result,
and nobody should read one as vindication.

---

## Corrections to the record

- **"940 test steps"** appeared in §7.9, §9.2, `FINDINGS-PIPELINE.md` and
  `THESIS.md`, and matches nothing. The test split holds **925** steps; the
  end-to-end loop replays **914** of them, because `max_steps=16` truncates 11
  steps in solutions longer than 16. Both numbers are now stated, with which
  result uses which.
- **`scripts/exp_equivalence_relations.py`** claimed the entailment re-clustering
  takes "about half an hour" on CPU. Measured, it is ~28k NLI pairs at ~2/s on 16
  cores, i.e. about four hours. Corrected in the docstring.

## Limits of this analysis

- **One corpus.** These intervals describe resampling this corpus, not
  generalisation to another generator, another benchmark or another sampling
  temperature. Chapter 8's transfer run is the evidence on that, and it is a
  different question.
- **Percentile intervals.** BCa is better behaved for a skewed statistic. With
  AUROCs near 0.5 and n_pos in the hundreds the skew is mild, and percentile
  intervals have no tuning to get wrong. Worth revisiting if a claim ever turns
  on the third decimal, which none of these do.
- **The normal approximation in the power calculation.** Two significant figures
  is the most that should be read off 0.076, and the coincidence with 0.65 should
  not be read as a derivation.
- **Fitting variability is excluded** — see *What the intervals cover* above. For
  the probe it is the larger effect and it is quoted separately in §7.6.
