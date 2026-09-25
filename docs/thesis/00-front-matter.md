# What Step-Level Verification Certifies, and What It Misses

### A measurement study of selective verification in multi-step LLM reasoning

**P. Somnath Reddy**
M.Tech thesis, third semester
Draft — 2026-09-06

---

## Abstract

Language models that reason in steps are increasingly paired with verifiers:
calculators, retrievers, process reward models, or other models asked to check
the work. A natural design follows — score each step's uncertainty, calibrate a
threshold with conformal prediction, and spend a verification budget on the
steps the calibrated score flags. This thesis set out to build that system. It
does not work, and the reasons it does not work are measurable, separable, and
more informative than the system would have been.

We establish four results on GSM8K and StrategyQA, using 93,129 model-generated
reasoning steps from Math-Shepherd and a further 2,573 steps generated for this
work by Qwen2.5-7B-Instruct.

**First, the quantity a verifier reports is not the quantity of interest.** A
step can fail by being invalid on its own terms (`47 × 3 = 131`) or by being
impeccable reasoning from a corrupted premise. A verifier reports the first;
conformal machinery therefore calibrates the first. On Mistral-7B-SFT, 78.5% of
globally-wrong steps are arithmetically perfect. Re-measured on a generator
almost twice as accurate, the figure rises to **90.4%** — the gap is not an
artifact of a weak model, and a deterministic verifier becomes *less* useful as
the generator improves.

**Second, verifier reach is the controlling design variable, and it is
semantic.** Widening an arithmetic verifier's lookback window from one step to
unbounded raises its detection rate from 0.0000 to 0.1999 and saturates. A
model asked to check its own work detects zero errors. A general independent
judge reaches 0.2283. Only a task-specialised, generator-independent process
reward model reaches 0.9033. Reach is not window size; it requires independence
*and* task specialisation.

**Third, "verify early" is false.** Front-loading verification is the worst
allocation shape at every verifier scope above zero, on linear chains, five
synthetic DAG families, and dependency graphs extracted from both benchmarks.
The best structural signal is ancestor count, not descendant count. The
intuition that early errors are cheaper to catch is correct; the inference that
early steps should therefore be verified is not, because later steps are
measurably harder (corr(position, local error) = +0.950).

**Fourth, the assembled gate fails — and an oracle and a trained probe locate
why.** Generator uncertainty barely ranks step error (AUROC 0.5589 token-level,
0.5740 resampled semantic divergence, 0.5742 combined). Split conformal
consequently holds its *coverage* guarantee while missing its selective-risk
target by a factor of three, and the one verifier with real reach is
net-negative behind that score.

Two controls separate cause from consequence. An **oracle score** takes
selective risk 0.154 → 0.089 and projected accuracy 0.79 → 0.98 on a third of
the calls, so the verifier was never the problem — it was being aimed badly. A
**probe on the generator's own frozen hidden states** reaches AUROC 0.6968, so
the signal is not absent either — and doubling its training data leaves it at
0.6896, so that is what the instrument gives rather than a floor. Neither rescues the gate: the probe misses
α = 0.05 by 2.9× and the oracle by 1.8×, because at two verification calls per
question the budget binds regardless of ranking.

A sweep over synthetic scores of controlled AUROC then locates the threshold
the verifier needs. It is **≈ 0.65** — below the probe that already exists —
and it is made entirely of false alarms: at a 0% false-alarm rate the same
verifier pays for itself down to AUROC 0.55. The sweep also refutes the metric
it is built on. Two scores of identical AUROC are worth different amounts,
because error propagation means only the *first* bad step in a solution can be
repaired, and a score that ranks late steps highly earns AUROC it cannot
convert.

The contribution is therefore a characterisation of a design space rather than
a system: selective verification of LLM reasoning fails at the signal, at the
calibration, and at the verifier, and repairing any one of them is not
sufficient. Along the way we correct a systematic bug in a widely-used
dependency-extraction heuristic, show that a benchmark in common use for this
problem cannot exhibit the phenomenon it is used to study, and document three
measurement errors that each produced a confident and wrong result before being
caught.

---

## Declaration on what is measured and what is modelled

This thesis reports two kinds of number and never blends them.

**Measured** quantities come from real model output with real labels: per-step
error rates, verifier detection rates, selective risk on the accepted set,
score AUROC, dependency-graph error rates. Every one is reproducible from the
committed code and data.

**Modelled** quantities come from the propagation model in
`src/car/propagation.py`, whose closed form is validated against simulation to
within 0.015 over twelve configurations. Its assumptions — full repair on
detection, i.i.d. per-step error — are known to be wrong. Where a modelled
number appears it is labelled PROJECTED, and where the modelling assumption
affects a conclusion the direction of the resulting bias is stated.

The distinction matters most in Chapter 7, where final-answer accuracy under
gating cannot be measured on a fixed corpus and is therefore projected.

---

## Reproducibility

All code, data-fetching scripts, measurement outputs and the artifacts of every
GPU run are in the repository, which is public and dual-licensed under MIT or
Apache 2.0.

```bash
pip install -e ".[dev]"
python -m pytest                    # 329 tests, no GPU, no network
python scripts/download_data.py     # GSM8K, StrategyQA, Math-Shepherd sample
```

Each chapter names the script that reproduces its tables. Results that cost GPU
time are committed rather than regenerated: `runs/semantic_scope_*.json`,
`runs/generated_qwen25_7b.jsonl`, `runs/uncertainty_qwen25_7b*.jsonl`,
`runs/semantic_samples.jsonl`.

Every refuted claim in this thesis has a regression test that keeps it refuted,
so a result cannot silently stop reproducing.

---

## Contents

| ch | title |
|---|---|
| 1 | [Introduction](01-introduction.md) |
| 2 | [Background](02-background.md) |
| 3 | [Framework](03-framework.md) |
| 4 | [Measuring the gap](04-measuring-the-gap.md) |
| 5 | [Verifier reach](05-verifier-reach.md) |
| 6 | [Allocation](06-allocation.md) |
| 7 | [The assembled gate](07-the-assembled-gate.md) |
| 8 | [Feasibility and benchmarks](08-feasibility-and-benchmarks.md) |
| 9 | [Limitations, negative results, conclusion](09-conclusion.md) |
| — | [References](10-references.md) |
