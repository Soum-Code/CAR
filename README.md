# CAR — Conformalized Agentic Reasoning

Calibrated selective verification for multi-step LLM reasoning.

A reasoning agent should not verify every step and should not trust every step.
CAR estimates uncertainty at intermediate steps, calibrates that signal against
held-out data, and uses it to decide when external verification is worth its
cost — and it verifies **before** an uncertain step is committed as trusted
state, not after the chain has already been built on top of it.

---

## The research problem

The headline contribution is not "use conformal prediction on reasoning steps".
That space is crowded. It is this:

> **You can only observe the outcome of the decisions you chose to check.**

The gate accepts a step when `score <= threshold` and verifies otherwise. So:

```
score >  threshold  ->  VERIFY    ->  label observed
score <= threshold  ->  CONTINUE  ->  label never observed
```

The risk CAR wants to control is the error rate among **accepted** steps. Those
are exactly the steps that produce no feedback. The estimand and the observable
set are disjoint by construction, and the censoring mechanism is the very
threshold being updated.

Standard adaptive conformal inference (Gibbs & Candès, NeurIPS 2021) assumes the
error is observed every round. Import it unchanged and it silently estimates
risk on the *reject* region and applies it to the *accept* region.

This is the **selective labels** problem (Lakkaraju, Kleinberg, Leskovec, Ludwig
& Mullainathan, KDD 2017) — a judge only learns the outcome for defendants they
released. Their fix exploits variation in leniency across many judges. CAR has
one policy, so it manufactures that variation with forced exploration.

### It is not a small effect

```
python scripts/demo_censored_feedback.py
```

Scores uniform on [0,1] with `P(wrong | score) = score`, so the accept-region
risk is `q/2` and the optimum is `q* = 2·alpha = 0.2` in closed form:

| update rule        | threshold | gap    | std    | verify rate |
|--------------------|-----------|--------|--------|-------------|
| naive              | 0.0105    | 0.1895 | 0.0007 | 98.7%       |
| exploration_only   | 0.2108    | 0.0108 | 0.0062 | 77.0%       |
| ipw                | 0.1892    | 0.0108 | 0.0325 | 82.2%       |

Naive ACI pins to the clip floor **at every horizon tested** and ends up
verifying 98.7% of steps — it has become always-verify at maximum cost. It is
being shown reject-region error (~0.60) against a target of 0.10, so it tightens
forever. More data does not fix it.

Both corrections recover `q*`. They differ in how: IPW scales each update by
`1/epsilon`, buying roughly `1/epsilon` times the convergence speed for roughly
`1/epsilon²` the variance.

> **Status:** the convergence rate this is built to test — selective risk
> approaching alpha at `O(1/sqrt(epsilon·T))` with budget overhead `epsilon·T` —
> is a **hypothesis**, not a theorem. This code is the instrument for testing it.
> Do not report it as a guarantee.

---

## Second contribution: verification value ≠ uncertainty

A step that is uncertain but that nothing depends on is cheap to get wrong. A
step that is mildly uncertain but carries five downstream conclusions is
expensive to get wrong.

```
verification_value  ~  P(step is wrong)  ×  downstream_influence
```

`dependency_ids` in the step schema makes this computable. It also means
**"verify early" is a prediction of the framework rather than a separate
empirical claim** — early steps have more descendants by construction, so they
score higher automatically.

Set `InfluenceWeighting(mode="none")` for the pure-uncertainty ablation.

---

## Install

```bash
pip install -e ".[dev]"
```

Add `".[model]"` only on a machine with a GPU. The whole control and calibration
stack runs on CPU.

```bash
python -m pytest
```

67 tests, a few seconds, no GPU, no network, no dataset.

---

## The GPU/CPU split

This is the workflow decision that keeps compute cost near zero.

**Phase 1 — generation (needs a GPU, run once).** Run the model over the
dataset, dump every step, token logprob and semantic sample to disk via
`CachedBackend`. A few hours total.

**Phase 2 — everything else (CPU).** Uncertainty combination, conformal
calibration, every gate policy, every baseline, every ablation, all metrics and
plots. Reads cached scores, never loads the model.

Almost all iteration is phase 2. Rent a GPU for hours, not months.

> **Do not casually quantise to 4-bit.** This project measures token-level
> uncertainty derived from logits, and quantisation distorts exactly that
> distribution. If memory forces it, make precision an explicit experimental
> variable and show calibration holds at both.

---

## Layout

```
src/car/
  types.py            ReasoningStep, StepRecord, Trajectory, Decision, Verdict
  backends/           LM interface; mock simulator, HF, disk cache
  generation/         JSON step schema, parsing, step generators
  uncertainty/        token entropy, max surprisal, semantic divergence, fusion
  conformal/          split · risk control · adaptive-with-exploration
  control/            budget, downstream influence, the CONTINUE/VERIFY/ABSTAIN gate
  verification/       calculator, retrieval, oracle, same-model critic
  baselines/          CoT, always-verify, quantile gate, random gate, oracle
  eval/               AUROC, ECE, selective risk, coverage, false-safe, Pareto
  data/               dev/cal/test splits with leakage assertions
  agent/loop.py       the control loop
tests/                67 tests
scripts/              runnable demos
```

### The mock backend is not a toy

`MockBackend` draws a latent difficulty per step, makes correctness a Bernoulli
draw whose probability is a **known** function of it, then emits token scores as
noisy observations. The true uncertainty/correctness relationship is therefore
something we control exactly.

If the conformal layer cannot recover the right threshold there, the bug is in
our code — not in the language model. Validate the statistics before spending
GPU hours.

Set `signal_strength=0.0` for the null hypothesis: uncertainty is pure noise.
CAR should then show no gain over a random gate. If it does, something leaks.

---

## Baselines worth not skipping

| Baseline | Why it belongs |
|---|---|
| **Random gate** at matched budget | Surprisingly strong. If CAR cannot beat random allocation at equal cost, the uncertainty signal is doing nothing. |
| **Oracle gate** | Upper bound. Separates "our gate is good" from "this task was easy". |
| **Same-model critic** | Negative control. Huang et al. (ICLR 2024) found intrinsic self-correction unreliable and sometimes harmful. Measure it here rather than citing it. |
| **Process reward model** | The elephant. PRMs are trained to do step-level error detection. A reviewer *will* ask why not just use one. |

---

## Discipline this repo enforces

- **Splits.** dev = engineering and weight fitting; calibration = threshold
  only; test = never touched for tuning. Assignment is by hash of example id,
  not shuffled index, so an example cannot migrate between splits as the dataset
  grows. `assert_no_leakage` and `check_calibration_size` fail loudly.
- **Coverage is not accuracy.** `1 - alpha` is a property of the acceptance
  rule. It is also *marginal* — coverage on any given slice can be far lower.
- **Traces.** Every score, threshold, influence, gate outcome, exploration flag
  and verdict goes to JSONL with a config hash. A results table you cannot trace
  back is not defensible in a viva.
- **Verifier independence.** The verifier must use evidence or a deterministic
  procedure the generator did not have. A second LLM with the same weights is a
  negative control, not a verifier.
- **Parse failures are counted, not repaired.** How often the generator breaks
  its own schema is a result.

---

## Key references

- Angelopoulos, Bates, Fisch, Lei & Schuster. *Conformal Risk Control.* ICLR 2024 — [arXiv:2208.02814](https://arxiv.org/abs/2208.02814). The right tool for controlling selective risk; plain split conformal controls coverage, which is a different quantity.
- Gibbs & Candès. *Adaptive Conformal Inference Under Distribution Shift.* NeurIPS 2021 — [arXiv:2106.00170](https://arxiv.org/abs/2106.00170).
- Farquhar, Kossen, Kuhn & Gal. *Detecting Hallucinations Using Semantic Entropy.* Nature 630, 2024 — [doi](https://doi.org/10.1038/s41586-024-07421-0).
- Huang et al. *Large Language Models Cannot Self-Correct Reasoning Yet.* ICLR 2024 — [arXiv:2310.01798](https://arxiv.org/abs/2310.01798).
- Lakkaraju, Kleinberg, Leskovec, Ludwig & Mullainathan. *The Selective Labels Problem.* KDD 2017 — [pdf](https://cs.stanford.edu/~jure/pubs/contraction-kdd17.pdf).
- Singh & Pawar. *The Hallucination Snowball.* 2026 — [arXiv:2608.14588](https://arxiv.org/abs/2608.14588). Error escape probabilities of 24.6% / 48.3% / 89.3% across pipeline boundaries.

Adjacent work to position against, not in the original spec:
[Uncertainty Heads](https://arxiv.org/html/2511.06209v2) (step uncertainty
matching PRMs 810× larger) · [ConfSpec](https://arxiv.org/pdf/2602.18447)
(confidence-gated verification) ·
[CCPO](https://arxiv.org/abs/2511.11828) (cost-aware policy + adaptive
threshold) · [When Can Conformal Risk Control Certify LLM
Outputs?](https://arxiv.org/pdf/2606.29054) (contains impossibility results —
read before attempting a proof).
