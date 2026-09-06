# 3. Framework

## 3.1 The decomposition

Every measurement in this thesis rests on separating two failure modes of a
reasoning step:

```
global_correct(t)  =  local_valid(t)  AND  NOT premise_corrupt(t)
```

The two are separately observable on GSM8K, which is what makes the study
possible at all:

| quantity | how it is observed | instrument |
|---|---|---|
| `local_valid(t)` | does the step's own arithmetic hold? | deterministic evaluation of `<<expr=result>>` |
| `global_correct(t)` | does this prefix still lead to the right answer? | Math-Shepherd's `+`/`-` Monte-Carlo label |

The derived quantity that the thesis is about:

```
inherited_corruption(t)  =  local_valid(t)  AND  NOT global_correct(t)
```

A step that is arithmetically perfect and still wrong. No deterministic
verifier can see it, and none of the four verifier classes measured in
Chapter 5 can either — but §2.7 records one structural change that does lift
the ceiling: scoring each step against *previously-verified* premises rather
than against the generator's own unverified context. The population is
therefore invisible to a verifier reading generated context, which is a
narrower claim than "invisible", and the one this thesis supports.

### Two estimators for the headline ratio

The fraction of globally-wrong steps that are locally valid can be computed two
ways, and the difference only becomes visible when comparing generators.

```
C1_all        =  P(inherited) / P(global error)          over ALL steps
C1_checkable  =  P(local_valid | global error, checkable)
```

`C1_all` is the estimator behind the figure originally reported for this
project (0.6982). It is biased **downward** by the uncheckable rate: a step
with no arithmetic can never enter the numerator but always sits in the
denominator. On Math-Shepherd, where 91.4% of steps are checkable, the bias is
small. On a generator whose steps are 59% prose it collapses the estimate to
0.25 and says nothing about reasoning.

`C1_checkable` conditions on checkability in both numerator and denominator and
is the estimator to quote across generators. Chapter 4 reports both.

## 3.2 Verifier reach

A verifier is characterised by two numbers, both measurable:

- **scope** — `P(detect | step is globally wrong)`
- **false alarm** — `P(flag | step is globally correct)`

Scope is not a property of the verification *budget*; it is a property of the
verifier class. Chapter 5 measures it across four classes and finds it spans
0.0000 to 0.9033.

The modelling decision that carries the most weight appears in
`src/car/verification/scoped.py`: **a missed detection returns SUPPORTED, not
INSUFFICIENT.** A verifier that fails to see an error does not announce the
failure — it says the step looks fine. INSUFFICIENT would yield no label and
the calibrator would learn nothing; SUPPORTED yields a label of "no error here"
for a step that has one.

So a low-scope verifier does not merely help less. It feeds the calibrator
systematically wrong labels, and the calibrator converges on a threshold that
is confident about a risk it cannot see. That is a prediction of C3, and
Chapter 7 shows it rather than asserting it.

## 3.3 The propagation model

Entering corruption and escaping it are asymmetric, and the asymmetry drives
every simulated result. As an age-indexed Markov chain with `v_t` the
verification probability at step `t`, `e` the local error rate, and `scope`,
`decay` the verifier's reach and its fade with propagation distance:

```
CLEAN      ->  CORRUPT_1     e · (1 − v_t)                  <-- no scope term
CORRUPT_k  ->  CLEAN         v_t · scope · decay^(k−1)      <-- scope AND decay
```

**Entering corruption does not depend on the verifier's reach. Escaping it
does.** A verifier with scope 0 cannot reduce the corrupted-state occupancy at
any budget; it can only delay entry by catching local errors.

`decay ≈ 0.377` is fitted to the escape probabilities Singh & Pawar measured
across successive agent boundaries (24.6% / 48.3% / 89.3%).

The closed form is validated against simulation to within 0.015 over twelve
configurations (`tests/test_propagation.py`). Its assumptions are known to be
wrong in two ways, and the direction matters:

- **Full repair on detection.** A caught error is assumed fixed. Real repair is
  partial, so the model is optimistic about verification.
- **I.i.d. per-step error.** Measured, this is false: corr(position, local
  error) = +0.950. Later steps are harder, which makes the model *conservative*
  about the value of back-loading — the direction that makes Chapter 6's
  conclusion safer rather than weaker.

## 3.4 The measurement apparatus

### Splits by hash, not by index

`src/car/data/splits.py` assigns each example to dev / calibration / test by
`sha256(salt:example_id)`, not by shuffled index. An example therefore cannot
migrate between splits as the dataset grows or is re-filtered. The scaler for
the composite score is fitted on **dev only**; fitting it on calibration or
test would leak into the threshold and void the conformal guarantee.

### Feasibility checked at setup

`src/car/conformal/feasibility.py` raises at configuration time when the budget
sits below the Kotte floor for the configured α and measured μ, rather than
letting the run produce a flat results table that has to be diagnosed
afterwards.

### The mock backend is a null-hypothesis instrument

`MockBackend` draws a latent difficulty, makes correctness a Bernoulli draw
whose probability is a *known* function of it, and emits token scores as noisy
observations. If the conformal layer cannot recover the right threshold there,
the bug is in the implementation. Setting `signal_strength = 0.0` gives the
null hypothesis: uncertainty is pure noise and the gate should show no gain
over random allocation at matched budget.

This is used directly in Chapter 7. The same pipeline scores AUROC 0.8668 on
synthetic features with one-standard-deviation separation and 0.4828 on pure
noise, which is what licenses the interpretation of the measured 0.5589 as a
property of the signal rather than of the harness.

### One loop, components swapped

Every condition in Chapter 7 — plain chain-of-thought, always-verify, random
gate at matched budget, quantile gate, split conformal, adaptive conformal —
runs through the same `CARAgent` loop with one component replaced. Baselines
implemented as parallel code paths drift, and a Pareto plot comparing two
subtly different loops is not a comparison.

The two baselines most often skipped are the two that matter most:

- **Random gate at matched budget.** If the gate cannot beat random allocation
  at the same number of calls, the score is doing nothing and no amount of
  calibration will rescue it.
- **Oracle score.** The upper bound: the same calibrator, budget and verifier
  driven by a score that reads the label. It separates "the gate is bad" from
  "the task is hard at this budget", and Chapter 7 reports it as a row in every
  table.

### Replay, and what it forbids

Generation costs GPU time and runs once, writing a corpus; everything
downstream reads that corpus on CPU. `ReplayStepGenerator` satisfies the same
protocol as the live generator, so the gate, calibration and budget code paths
exercised in Chapter 7 are the real ones.

The cost is a hard limit on what can be claimed. The step text is fixed, so the
gate cannot change what the model writes, and a verified-and-corrected step
does not cause the following steps to be regenerated from the corrected
premise. Therefore:

| claim | status |
|---|---|
| selective risk on the accepted set | **measurable** |
| verification rate, calls, budget-blocked steps, recall, AUROC | **measurable** |
| final-answer accuracy under gating | **not measurable — projected** |

Reporting measured final-answer accuracy off a fixed corpus would be claiming
an intervention that never happened. Chapter 7 projects it under an explicitly
stated repair model and labels every such number PROJECTED.

## 3.5 Datasets

**GSM8K** (Cobbe et al., 2021) is the primary benchmark. Grade-school word
problems with worked solutions carrying inline `<<expr=result>>` calculator
annotations, which supply the deterministic local check. 7,473 train problems,
1,319 test.

**Math-Shepherd** supplies the global labels: 93,129 GSM8K steps from 25,971
Mistral-7B-SFT solutions, each step carrying `+`/`-`.

**StrategyQA** (Geva et al., TACL 2021) was the original primary benchmark and
is retained for the benchmark critique in Chapter 8. It ships annotated
decompositions, which is why it looked attractive; Chapter 8 shows those
decompositions are too shallow to exhibit propagation.

**A generated corpus** produced for this thesis: 500 GSM8K test problems solved
by Qwen2.5-7B-Instruct with global labels from 8,292 Monte-Carlo rollouts,
written in Math-Shepherd's own format so the same analysis code reads both.

### The sampling trap, stated up front

The Math-Shepherd file is sorted into contiguous blocks by label and by task.
**Any prefix read is close to single-class** — the first 80 MB is 100%
wrong-answer GSM8K, and using it produces a plausible-looking table in which
every error rate is wrong. It cost two debugging rounds to find, and a third
time when a `limit=` parameter reintroduced it.

`scripts/download_data.py` therefore fetches 48 strided range requests spread
across the whole file and asserts the resulting class balance, and
`load_solutions` takes an explicit `stride`. A test asserts that a prefix read
is more skewed than a strided one, so the trap cannot quietly return.
