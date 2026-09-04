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
| Composite uncertainty is the key signal | **crowded** — [UHeads](https://arxiv.org/html/2511.06209v2) matches PRMs 810× larger; semantic entropy is [contested at step level](https://arxiv.org/html/2602.02427) |
| H3: verify early beats verify late | **false** — refuted on chains, 5 synthetic DAG families, and real extracted graphs |
| Influence-weighted allocation | **false** — lost to plain uniform every time it was properly tested |
| StrategyQA as primary benchmark | **wrong choice** — 72.9% of its graphs are one hop deep |

What survived is not a method. It is a measurement, and it is one nobody has
made.

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

### C2 — The gap does not close with more of the same verification

Entering corruption does not depend on the verifier's reach; escaping it does:

```
CLEAN     -> CORRUPT_1   e * (1 - v_t)
CORRUPT_k -> CLEAN       v_t * scope * decay^(k-1)
```

Simulated: local risk stays pinned near 0.15 across budgets while final error
spans 0.73 → 0.27. Measured: after the first bad step 95.9% of later steps stay
bad, and **0 of 25,971 solutions ever recovered**.

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

### C6 — StrategyQA cannot exercise the phenomenon it is used for

Extracted all 2272 annotated dependency graphs: mean depth 2.30, **72.9% exactly
one hop deep**, and only 11.2% of steps have any descendant other than the
answer. A step can only corrupt downstream reasoning if downstream reasoning
exists.

---

## What remains to be measured

C3 was the centrepiece and is now measured (Kaggle P100, 7B PRM, 1,500
arithmetic-blind steps + 750 controls). What is left is breadth: the same
measurement for retrieval and same-model-critic verifiers, and a re-measurement
on the generator CAR actually uses.

| # | experiment | status |
|---|---|---|
| 1 | Measure arithmetic verifier scope | **done** — 0.0000 at k=0, 0.1999 at k=∞ |
| 2 | Measure semantic verifier scope | **done** — 0.9033 at 9.9% FA (Kaggle P100) |
| 3 | Re-measure error rates on Llama 3.1 8B | numbers currently from Mistral-7B-SFT |
| 4 | Same-model + independent-judge scope arms | **done** — 0.0000 and 0.2283 |
| 5 | Full gate pipeline end-to-end on GSM8K | scaffold ready, needs GPU generation |
| 6 | Hand-validate ~50 GSM8K dependency graphs | **done** — found a systematic bug; corrected edge error 5.6% |

---

## Install and verify

```bash
pip install -e ".[dev]"
```

```bash
python -m pytest
```

203 tests, no GPU, no network. Corpus tests skip if datasets are absent.

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
  eval/               AUROC, ECE, selective risk, coverage, false-safe
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
