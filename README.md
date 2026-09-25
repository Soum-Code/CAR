# CAR — what step-level verification certifies, and what it misses

A measurement study of selective verification in multi-step LLM reasoning.

**The finding, in one line:**

> Conformal verification certifies that a reasoning step is *locally valid*.
> On GSM8K, **69.8% of globally-wrong steps are arithmetically perfect** — they
> are wrong only because a premise was. A calculator cannot see any of them.

Measured on 93,129 steps from 25,971 model-generated solutions, not simulated.

---

## Why this is the framing

The project began as a method proposal: calibrated uncertainty gating with
adaptive conformal calibration under censored feedback. Two rounds of
literature checking and stress-testing killed the method claims and left a
sharper empirical one. That history is kept in full — see
[docs/POSITIONING.md](docs/POSITIONING.md) and
[docs/FINDINGS-PROPAGATION.md](docs/FINDINGS-PROPAGATION.md) — because the
negative results are part of the contribution.

| original claim | status |
|---|---|
| Adaptive conformal under censored feedback is novel | **scooped** — [CSA](https://arxiv.org/abs/2605.20270) Thm E.1 publishes Bernoulli subsampling with 1/π importance weighting, under a stronger anytime guarantee |
| Composite uncertainty is the key signal | **crowded, then refuted** — a small probe on internal states matches far larger PRMs ([ReProbe](https://arxiv.org/abs/2511.06209)); sampling-based step uncertainty is [contested](https://arxiv.org/abs/2602.02427) and measured here at AUROC 0.5589, whose interval covers chance |
| H3: verify early beats verify late | **false** — refuted on chains, 5 synthetic DAG families, and real extracted graphs |
| Influence-weighted allocation | **false** — lost to plain uniform every time it was properly tested |
| StrategyQA as primary benchmark | **wrong choice** — 72.9% of its graphs are one hop deep |

What survived is not a method. It is a measurement, and it is one nobody has
made. The end-to-end run (C7) then refutes the original proposal outright, on
its own data — which is the contribution, not a setback.

---

## The claims, and the evidence for each

### C1 — The certified quantity is not the quantity of interest

A step fails two separably different ways:

```
global_correct(t) = local_valid(t) AND NOT premise_corrupt(t)
```

- **local invalidity** — `47 * 3 = 131`. Wrong in any context.
- **inherited corruption** — *Aristotle died in 1850, so he could have used a
  laptop.* Impeccable logic, false conclusion.

A verifier reports the first. Conformal machinery therefore calibrates the
first. Measured on GSM8K within wrong-answer solutions:

| | rate |
|---|---|
| local error | 0.1641 |
| global error | 0.7106 |
| **inherited corruption** | **0.4961** |

**69.8% of bad steps are arithmetically perfect.** Controlling local selective
risk at level α bounds nothing about the answer.

**And it is not an artifact of a weak generator — it gets worse.** Re-measured
on Qwen2.5-7B-Instruct (80% on GSM8K against Mistral-7B-SFT's ~45%), on 500
freshly generated test solutions:

| within wrong-answer solutions | Mistral-7B-SFT | Qwen2.5-7B |
|---|---|---|
| local error | 0.1708 | **0.0813** |
| globally-wrong steps that are locally valid | 0.7848 | **0.9040** |

The stronger model halves its arithmetic slips without halving its inherited
corruption, so more of what remains is invisible to a calculator. The intervals
do not overlap — under Wilson, and under a solution-clustered bootstrap that does
not assume steps are independent draws ([0.8430, 0.9531] on the Qwen side). See
[docs/FINDINGS-GENERATOR.md](docs/FINDINGS-GENERATOR.md).

> A deterministic verifier gets *less* useful as the generator improves.

### C2 — The gap does not close with more of the same verification

Entering corruption does not depend on the verifier's reach; escaping it does:

```
CLEAN     -> CORRUPT_1   e * (1 - v_t)
CORRUPT_k -> CLEAN       v_t * scope * decay^(k-1)
```

Simulated: local risk stays pinned near 0.15 across budgets while final error
spans 0.73 → 0.27. Measured: after the first bad step 95.9% of later steps stay
bad, and **0 of 25,971 solutions ever recovered**.

On Qwen2.5-7B the same measurement gives 66.4% persistence and 6 recoveries in
500 — still strongly absorbing, but *near*-absorbing was a Mistral property and
does not transfer unqualified.

### C3 — Verifier reach is the controlling design variable, and it is SEMANTIC

Measured across four verifiers on the arithmetic-blind population:

| verifier | independent? | task-trained? | scope | FA | net |
|---|---|---|---|---|---|
| arithmetic, step-local | yes | — | 0.0000 | — | 0.0000 |
| arithmetic, unbounded lookback | yes | — | 0.1999 | — | 0.1999 |
| same-model critic (the generator) | **no** | no | 0.0000 | 0.0000 | 0.0000 |
| independent judge (Qwen2.5-7B) | yes | no | 0.2283 | 0.0200 | 0.2083 |
| task PRM (Math-Shepherd-7B) | yes | **yes** | **0.9033** | 0.0987 | **0.8047** |

The generator judging its own work detects **zero** errors — it approves
everything, Huang et al. (ICLR 2024) measured directly. A general independent
judge barely beats arithmetic. Only a verifier that is *both* independent of the
generator *and* task-specialised closes the gap.

> Reach is not how far back you look. It requires a verifier independent of the
> generator and specialised for the task.

The original spec lists calculator / retrieval / sandbox as interchangeable
reliability mechanisms. They are not — they span 0.00 to 0.90 scope, and that
axis is absent from the spec entirely.

### C4 — "Verify early" is false

Refuted on linear chains, five synthetic DAG families, and real dependency
graphs extracted from both benchmarks. Front-loading is the *worst* allocation
shape at every verifier scope > 0.

The last escape — "maybe early steps are just harder" — is closed by
measurement: **corr(position, local error) = +0.950**. Error rate *doubles*
from 11% at step 1 to 22% at step 8. Later steps are harder, which favours
back-loading further.

The best structural signal is **ancestor count** (`depth`) on both benchmarks —
verify where the most upstream reasoning converges, not where the most
downstream damage could occur.

> An earlier version reported this as benchmark-dependent. That was an artifact
> of a dependency-extraction bug found by hand-validation; see
> [docs/FINDINGS-DEPGRAPH.md](docs/FINDINGS-DEPGRAPH.md).

### C5 — The risk target is constrained before any method is chosen

[Kotte](https://arxiv.org/abs/2606.29054) Prop. 3: when base risk μ > α, any
distribution-free method must verify or abstain on ≥ (μ−α)/(1−α) of steps.
With measured μ = 0.3908:

| α | 0.05 | 0.10 | 0.20 | 0.30 | 0.40 |
|---|---|---|---|---|---|
| floor | 35.9% | 32.3% | 23.8% | 13.0% | none |

α = 0.10 — the spec's value — charges a third of the budget as an entry fee.
`configs/default.yaml` now uses 0.30 and **enforces the floor at setup**.

The floor is a property of the generator, not of the task: on Qwen2.5-7B
μ = 0.1221 and α = 0.20 carries **no floor at all**. Any statement about
attainable α has to name the model it was measured on.

### C6 — StrategyQA cannot exercise the phenomenon it is used for

Extracted all 2272 annotated dependency graphs: mean depth 2.30, **72.9% exactly
one hop deep**, and only 11.2% of steps have any descendant other than the
answer. A step can only corrupt downstream reasoning if downstream reasoning
exists.

### C7 — Derived dependency graphs are 94.4% correct

Every GSM8K dependency edge here is derived, not annotated: line *i* links to
line *j* when an operand of *i* equals the result of *j*. Hand-validating 50
stratified graphs measured that at **5.6% edge error** — and found a systematic
bug on the way. The operand regex read each subtraction operator as a minus
sign, so **every subtraction in the corpus silently lost its dependency edge**.
Fixing it moved mean depth 2.54 → 2.79 and headroom 26.6% → 29.9%, and
overturned a published conclusion. See
[docs/FINDINGS-DEPGRAPH.md](docs/FINDINGS-DEPGRAPH.md).

### C8 — The assembled gate does not work

The whole loop, run on real output: 500 Qwen2.5-7B solutions, real uncertainty,
real conformal calibration, real budget. See
[docs/FINDINGS-PIPELINE.md](docs/FINDINGS-PIPELINE.md).

**The measured signals do not rank the risk.** Token entropy, surprisal and
log-prob combined give **AUROC 0.5589 [0.4780, 0.6328]**. Semantic divergence —
the signal the spec weighted most heavily, resampled at K=5 over all 2,573
steps — gives **0.5740 [0.5054, 0.6433]**, and combining them gives **0.5742**,
a paired **+0.0003 [−0.0390, +0.0445]** over divergence alone. The same pipeline
scores 0.8668 on synthetic features with real separation and 0.4828 on noise, so
the machinery works.

Intervals are 95% percentile bootstrap resampling *solutions*, not steps — the
design effect is 1.8 to 3.3, so the naive error bars would have been up to 1.8×
too small. Divergence is the one measured signal whose interval clears chance;
it is still far below the ≈ 0.65 the verifier needs. See
[docs/FINDINGS-SIGNIFICANCE.md](docs/FINDINGS-SIGNIFICANCE.md).

**A probe on internal states does rank it — and still is not enough.** A
logistic probe on the generator's own frozen hidden states reaches **0.6968
[0.6302, 0.7569]** — a paired **+0.1226 [+0.0287, +0.2288]**, p = 0.004, and the
only comparison in that chapter this corpus can resolve — and improves selective
risk at every α on fewer calls. It still misses α = 0.05
by 2.9×, and still leaves the task PRM net-negative. See
[docs/FINDINGS-PROBE.md](docs/FINDINGS-PROBE.md).

**And AUROC turns out to be the wrong target.** Sweeping a synthetic score of
controlled AUROC through the same gate puts the crossing — where the task PRM
stops costing more than it recovers — at **AUROC ≈ 0.65**, *below* the probe
that already exists, and shows that threshold is made entirely of false alarms
(at FA = 0 the same verifier pays for itself down to 0.55). It also refutes its
own metric: the probe converts AUROC 0.6968 into 0.7802 projected accuracy
where a synthetic score of identical AUROC reaches 0.8206, because only the
*first* bad step can be repaired and the probe ranks late ones
(corr with position +0.18, against −0.28 for the composite). See
[docs/FINDINGS-SCORE-QUALITY.md](docs/FINDINGS-SCORE-QUALITY.md).

> Measured under string-equality clustering the same samples give AUROC 0.4904 —
> and that reads as *below chance* only until an interval goes around it:
> [0.4379, 0.5459] covers 0.5, and the paired difference from numeric clustering
> is +0.0453 [−0.0138, +0.0949], p = 0.124. The relation changes the score a
> great deal (the two agree on 61.8% of steps) and the answer not measurably. An
> earlier draft claimed the relation was load-bearing on the result; it is
> withdrawn. What saved the conclusion was not picking the better relation — it
> was putting an interval on the number. See
> [docs/FINDINGS-SIGNIFICANCE.md](docs/FINDINGS-SIGNIFICANCE.md).

**So calibration certifies nothing useful.** Base risk is 0.1578, meaning any
α ≥ 0.20 is met by verifying nothing. At the α values that actually bind:

| α | 0.05 | 0.10 | 0.15 |
|---|---|---|---|
| measured selective risk | **0.1491** | **0.1494** | **0.1516** |
| with the probe score | 0.1432 | 0.1363 | — |
| with a perfect score | 0.0885 | 0.0885 | 0.0885 |
| verification rate | 4.6% | 9.2% | 14.1% |

Risk misses the target by 3× at α = 0.05 and barely moves across the sweep,
while verification climbs. Split conformal holds its *coverage* guarantee
throughout — coverage is a property of the acceptance rule, and with an
uninformative score it is simply not the risk of what gets accepted.

**And the verifier with reach is net-negative — but only behind a bad score.**
Projected accuracy against a 0.8022 baseline:

| score | verifier | calls/q | projected accuracy |
|---|---|---|---|
| real | task PRM (FA 0.0987) | 1.89 | **0.7637** |
| **oracle** | task PRM (FA 0.0987) | **0.37** | **0.9780** |

A false alarm can only fire on a step the gate chose to verify. C3's
`net = scope − FA` is the wrong figure of merit, and so is the population-level
`scope × P(wrong)` vs `FA × P(correct)`: the right one conditions on what the
gate selects. **The PRM is not a bad verifier being oversold — it is a good
verifier being aimed badly.**

An oracle score also locates the remaining bottleneck: it takes selective risk
0.154 → 0.089, but still misses α = 0.05 by 1.8×, because at two calls per
question most wrong steps go unverified however well they are ranked. Two
bottlenecks, the signal and the budget; the verifier is downstream of the
first.

---

## What remains to be measured

C3 was the centrepiece and is now measured (Kaggle P100, 7B PRM, 1,500
arithmetic-blind steps + 750 controls), and C1 has been reproduced on a second,
stronger generator. What is left is the end-to-end pipeline and cross-domain
breadth.

| # | experiment | status |
|---|---|---|
| 1 | Measure arithmetic verifier scope | **done** — 0.0000 at k=0, 0.1999 at k=∞ |
| 2 | Measure semantic verifier scope | **done** — 0.9033 at 9.9% FA (Kaggle P100) |
| 3 | Re-measure error rates on a second generator | **done** — Qwen2.5-7B, C1 rises 0.78 → 0.90 |
| 4 | Same-model + independent-judge scope arms | **done** — 0.0000 and 0.2283 |
| 5 | Full gate pipeline end-to-end on GSM8K | **done** — negative: AUROC 0.56, target missed 3x |
| 6 | Hand-validate ~50 GSM8K dependency graphs | **done** — found a systematic bug; corrected edge error 5.6% |
| 7 | Probe on frozen internal states | **done** — AUROC 0.6968; the signal exists and does not close the gap |
| 8 | Locate the score quality the verifier needs | **done** — AUROC ~0.65, and AUROC is the wrong metric |

---

## Install and verify

```bash
pip install -e ".[dev]"
```

```bash
python -m pytest
```

256 tests, no GPU, no network. Corpus tests skip if datasets are absent.

### Get the data

```bash
python scripts/download_data.py
```

Fetches StrategyQA, GSM8K and a Math-Shepherd sample into `data/raw/`
(gitignored, ~130 MB, idempotent). Check with `--verify`, refetch with
`--force`, or name one dataset to fetch just that.

> The Math-Shepherd fetch is strided across 48 range requests rather than being
> a plain download, and asserts the resulting class balance afterwards. That
> file is sorted into contiguous blocks by label, so any prefix read is
> effectively single-class — the first 80 MB is 100% wrong-answer GSM8K, and
> using it produces a plausible-looking table in which every error rate is
> wrong. It cost two debugging rounds to find. The script fails loudly rather
> than letting it recur.

Then reproduce the headline results:

```bash
python scripts/exp_measure_error_rate.py
```

```bash
python scripts/exp_propagation.py
```

```bash
python scripts/exp_generator_transfer.py
```

```bash
python scripts/exp_gate_pipeline.py
```

```bash
python scripts/exp_significance.py
```

```bash
python scripts/exp_rate_intervals.py
```

The last two are the ones to run before quoting any number from this project. It
puts a 95% interval on every headline number, pairs every comparison on the same
resamples, and reports which of them the corpus can actually support. Three
claims in earlier drafts did not survive it, and they are listed in
[docs/FINDINGS-SIGNIFICANCE.md](docs/FINDINGS-SIGNIFICANCE.md).

---

## Layout

```
src/car/
  types.py            ReasoningStep, StepRecord, Trajectory, Decision, Verdict
  propagation.py      local vs global correctness; the Markov model
  topology.py         reasoning DAGs and edge-following propagation
  backends/           LM interface; mock simulator, HF, disk cache
  generation/         JSON step schema, parsing, step generators
  uncertainty/        entropy, surprisal, semantic divergence, fusion
  conformal/          split · risk control · adaptive · feasibility floor
  control/            budget, structural allocation, the gate
  verification/       calculator, retrieval, oracle, simulated, same-model critic
  data/               GSM8K (primary), StrategyQA, Math-Shepherd, splits
                      generated.py: corpora from other generators
  eval/               AUROC, ECE, selective risk, coverage, false-safe
                      inference.py: clustered bootstrap intervals, paired
                      differences, design effect, minimum detectable effect
  agent/loop.py       the control loop
```

### The mock backend is not a toy

`MockBackend` draws a latent difficulty, makes correctness a Bernoulli draw
whose probability is a *known* function of it, then emits token scores as noisy
observations. If the conformal layer cannot recover the right threshold there,
the bug is ours. Set `signal_strength=0.0` for the null hypothesis: uncertainty
is pure noise, and CAR should then show no gain over a random gate.

### The GPU/CPU split

Generation needs a GPU and runs once, writing to a content-addressed cache.
Everything else — calibration, gating, baselines, ablations, metrics — reads
that cache on CPU. Rent a GPU for hours, not months.

> Do not casually quantise to 4-bit. This project measures token-level
> uncertainty derived from logits, and quantisation distorts exactly that
> distribution. If memory forces it, make precision an experimental variable.

---

## Discipline this repo enforces

- **Splits by hash of example id**, not shuffled index, so an example cannot
  migrate between dev/calibration/test as the dataset grows.
- **Coverage is not accuracy.** `1 − α` is a property of the acceptance rule,
  and it is *marginal* — coverage on a given slice can be far lower.
- **Feasibility is checked at setup.** A config whose budget sits below its own
  Kotte floor raises rather than producing a flat results table.
- **Verifier independence.** A second LLM with the same weights is a negative
  control, not a verifier.
- **Sampling traps are tested for.** Math-Shepherd is sorted into blocks by
  label; a prefix read is single-class. `load_solutions` takes a `stride` and a
  test asserts the prefix is more skewed.
- **Negative results are tests.** Every refuted claim has a test that keeps it
  refuted, so it cannot quietly stop reproducing.

---

## The thesis

The full draft is in [docs/thesis/](docs/thesis/README.md) — nine chapters,
every number reproducible from the scripts named at the head of each one.

| ch | | ch | |
|---|---|---|---|
| 1 | [Introduction](docs/thesis/01-introduction.md) | 6 | [Allocation](docs/thesis/06-allocation.md) |
| 2 | [Background](docs/thesis/02-background.md) | 7 | [The assembled gate](docs/thesis/07-the-assembled-gate.md) |
| 3 | [Framework](docs/thesis/03-framework.md) | 8 | [Feasibility and benchmarks](docs/thesis/08-feasibility-and-benchmarks.md) |
| 4 | [Measuring the gap](docs/thesis/04-measuring-the-gap.md) | 9 | [Limitations and conclusion](docs/thesis/09-conclusion.md) |
| 5 | [Verifier reach](docs/thesis/05-verifier-reach.md) | | |

---

## Licence

Dual-licensed under **MIT** ([LICENSE-MIT](LICENSE-MIT)) or **Apache 2.0**
([LICENSE-APACHE](LICENSE-APACHE)), at your option — MIT for the shortest
permissive terms, Apache 2.0 for its explicit patent grant.

The datasets are *not* redistributed here; `scripts/download_data.py` fetches
each from its own source, under its own terms. See [COPYRIGHT](COPYRIGHT).

---

## Key references

- Angelopoulos, Bates, Fisch, Lei & Schuster. *Conformal Risk Control.* ICLR 2024 — [arXiv:2208.02814](https://arxiv.org/abs/2208.02814)
- Khosravi & Huo. *Conformal Selective Acting.* 2026 — [arXiv:2605.20270](https://arxiv.org/abs/2605.20270). Thm E.1 is the censored-feedback result.
- Kotte. *When Can Conformal Risk Control Certify LLM Outputs?* 2026 — [arXiv:2606.29054](https://arxiv.org/abs/2606.29054). The impossibility bound.
- Wang et al. *Math-Shepherd.* ACL 2024 — [arXiv:2312.08935](https://arxiv.org/abs/2312.08935). The step-labelled data this study measures.
- Cobbe et al. *Training Verifiers to Solve Math Word Problems.* 2021 — [arXiv:2110.14168](https://arxiv.org/abs/2110.14168). GSM8K.
- Geva et al. *Did Aristotle Use a Laptop?* TACL 2021 — StrategyQA.
- Huang et al. *LLMs Cannot Self-Correct Reasoning Yet.* ICLR 2024 — [arXiv:2310.01798](https://arxiv.org/abs/2310.01798)
- Singh & Pawar. *The Hallucination Snowball.* 2026 — [arXiv:2608.14588](https://arxiv.org/abs/2608.14588). Escape probabilities 24.6/48.3/89.3%.
- Lakkaraju, Kleinberg, Leskovec, Ludwig & Mullainathan. *The Selective Labels Problem.* KDD 2017 — [pdf](https://cs.stanford.edu/~jure/pubs/contraction-kdd17.pdf)
