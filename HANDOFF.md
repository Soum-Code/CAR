# Session handoff

**Branch:** `claude/gallant-hopper-xjwa5j` (pushed, in sync with origin)
**Base at session start:** `ff53d29` — *Add bidirectional entailment as a third meaning-equivalence relation*
**Head:** `6b35b61`, plus the commit adding this file
**Tests:** 333 passed, 10 skipped. `ruff` clean on every file touched.

---

## 1. Why this session did not do what it set out to do

The previous session ended with the bidirectional-entailment measurement **still
running**, and the commit message said `exp_equivalence_relations.py` would
compare all three relations "once it lands". It never landed — the container is
ephemeral, and `runs/entailment_cache.jsonl` was gone.

Re-running it here is **blocked**:

```
huggingface.co:443 — connect_rejected (egress proxy denied the CONNECT)
```

That host is denied by this environment's network policy, so
`microsoft/deberta-large-mnli` cannot be fetched. The same block also stops
`scripts/download_data.py math-shepherd` (48/48 chunks fail), which is why the
Mistral half of the C1 transfer claim could not be recomputed either.

**To unblock:** cloud environment menu in the session title bar → Edit → Network
access. Either raise the access level or add `huggingface.co` to the allowed
domains. Nothing else is needed; both runs are one command each (§4).

So the session did the thing the entailment run was *for*, from the other side:
it measured how much of the thesis its own corpus can actually support.

---

## 2. What was done, in five commits

### `31d3aa9` — Put an interval on every AUROC the thesis argues from

Every AUROC in the thesis was a bare point estimate. That is a problem specific
to what chapter 7 claims: four of its five headline sentences are about a
*difference*, and two assert a difference is **absent** — the case that needs an
interval most, because a small number can mean a small effect or a small corpus.

New: `src/car/eval/inference.py`, `tests/test_inference.py` (32 tests),
`scripts/exp_significance.py`, `runs/significance.json`.

Every point estimate reproduced exactly (0.5589 / 0.5740 / 0.5742 / 0.6968), so
the join is right. What is new is what sits around them.

### `2edd225` — Fold the intervals into the chapters, and withdraw two claims

Chapter 7 rewritten where the intervals contradict it. New §7.7 carries the
method, design effects, resolution diagnostic and power calculation; old §7.8
Limits renumbered to §7.9. Front matter, introduction, conclusion §9.1, README,
`THESIS.md`, `FINDINGS-PIPELINE.md` all updated.

### `398b705` — Draw the intervals

`fig13-intervals.png/pdf` — a forest plot: five AUROCs with intervals on the
left, the five paired differences on the right, each labelled with the sentence
it was written to support. Only `fig13` is committed; re-running
`make_figures.py` rewrites all thirteen and the other twelve came out
byte-different under this matplotlib with no change to what they show.

### `6bebd3c` — Carry the correction into FINDINGS-PIPELINE

The chapters withdrew two claims; the findings document they draw on still
asserted both.

### `6b35b61` — Check the rates too

`scripts/exp_rate_intervals.py`, `runs/rate_intervals.json`, plus clustered
proportion machinery in `inference.py`. `wilson()` moved out of
`exp_generator_transfer.py` into `car.eval.inference` so the two cannot drift.

---

## 3. The findings, so nobody has to re-derive them

### Chapter 7: one of five comparisons survives

| comparison | Δ AUROC | 95% CI | p | verdict |
|---|---|---|---|---|
| probe vs best measured signal | +0.1226 | [+0.0287, +0.2288] | 0.004 | **survives** |
| probe vs token-level | +0.1379 | [+0.0428, +0.2427] | 0.002 | **survives** |
| numeric vs exact match | +0.0453 | [−0.0138, +0.0949] | 0.124 | not established |
| both vs token-level | +0.0153 | [−0.0068, +0.0369] | 0.148 | not established |
| both vs divergence alone | +0.0003 | [−0.0390, +0.0445] | 0.955 | zero, to ±0.04 |

**Two claims withdrawn:**

- *"Exact match drives AUROC below chance."* 0.4904 [0.4379, 0.5459] covers 0.5.
  Indistinguishable from chance, not below it. "Below chance" reads as
  anti-predictive and nothing supports that.
- *"The equivalence relation was load-bearing."* Load-bearing on the **score**
  (mean divergence 0.4239 vs 0.6468, same value on only 61.8% of steps), not
  measurably on the answer.

There is a better lesson under the second. §7.2's point was that the library
default would have reported divergence as anti-predictive, and that stands —
but what prevented it was not picking the better relation, since a point
estimate under *either* relation was going to be over-read. It was the interval.

**One claim softened:** divergence at 0.5740 [0.5054, 0.6433] *does* clear
chance. The refutation is that it is unusable, not uninformative, and §7.4's
0.65 crossing makes that quantitative.

### Two diagnostics worth keeping

- **Resolution is not the excuse.** Divergence over K=5 takes exactly seven
  values (partitions of 5). Quantising the probe onto that grid costs it 0.0131
  AUROC, so coarseness is not why divergence fails.
- **The unrun experiment is priced.** Paired SE between two relations is 0.0272,
  so the minimum detectable difference at 80% power is 0.076 — a fourth relation
  registers only above **0.6503** AUROC. §7.4's independent sweep puts the
  verifier's break-even at **≈0.65**. Coincidence, but a clarifying one: *on
  this corpus a relation can only be heard if it is already good enough to be
  useful.* A small positive entailment result is noise, not vindication.

### Chapter 4: the rates, where the answer inverts

| rate | ρ | mean cluster | clustered vs Wilson |
|---|---|---|---|
| C1 (checkable) | 0.68 | 1.98 | 1.05× |
| local error | 0.37 | 2.68 | 1.04× |
| inherited corruption | 0.45 | 5.15 | 1.65× |
| global error | 0.38 | 7.71 | **2.31×** |

Rank by ρ and rank by cost and **the orderings are reversed**. A design effect
is ~1 + (m−1)ρ and needs both terms; C1's clusters hold two steps, so there is
nothing for ρ = 0.68 to act on. **Cluster size decides the cost, not how
dependent the data is.**

- **C1's published Wilson interval stands.** [0.8397, 0.9442] vs clustered
  [0.8430, 0.9531].
- **Global error's does not.** 0.5772 was quoted [0.5420, 0.6116], should be
  [0.5005, 0.6613]. Nothing turns on its third decimal; it is now right.
- **The Mistral side is bounded, not left open.** For the two transfer intervals
  to touch, Math-Shepherd's design effect would have to be **243** — 356
  globally-wrong checkable steps per solution against an overall mean near 3.6.
  The transfer claim holds whatever its cluster sizes are.

### The methodological pair

| statistic | the intuition | what was measured |
|---|---|---|
| AUROC | clustered labels widen it — premise propagation makes step labels dependent | they do not. A rank statistic barely feels them; inflation comes from per-solution shifts in the **score**. deff 1.8–3.3 |
| C1 | near-absorbing corruption widens it a great deal | it does not. ρ = 0.68 and the clusters hold two steps. deff 1.13 |

Both intuitions wrong, opposite directions, same corpus, different reasons.
Neither was settleable by argument. Both are pinned in tests, including one that
fails if the module docstring's claim is reversed — which is how the first was
caught.

### Corrections to the record found on the way

- **"940 test steps"** appeared in §7.9, §9.2, `FINDINGS-PIPELINE.md` and
  `THESIS.md` and matches nothing. The test split holds **925**; the loop
  replays **914** (`max_steps=16` truncates 11 steps in solutions longer than
  16). Verified by re-running `exp_gate_pipeline.py`, which reproduces every
  chapter 7 number exactly.
- **`exp_equivalence_relations.py`** claimed the entailment re-clustering takes
  "about half an hour" on CPU. It is ~28k NLI pairs at ~2/s — about four hours.
  Its verdict logic also tested against a round 0.60; it now tests against
  0.6503 and states that no ordering among the relations is established by point
  estimates alone.

---

## 4. Picking it up from here

### Environment setup (nothing is cached across sessions)

```bash
pip install -e ".[dev]"
python scripts/download_data.py gsm8k      # works
# python scripts/download_data.py math-shepherd   # BLOCKED: huggingface.co
```

`torch` must come from PyPI, not `download.pytorch.org` — that host is denied
too. `pip install "torch>=2.2"` pulls the CUDA build (~8 GB of nvidia wheels)
and works fine on CPU.

### Reproduce this session's results

```bash
python scripts/exp_significance.py      # CPU ~3 min  -> runs/significance.json
python scripts/exp_rate_intervals.py    # CPU ~1 min  -> runs/rate_intervals.json
python -m pytest                        # 333 passed, 10 skipped
```

### The blocked run, once `huggingface.co` is allowed

```bash
python scripts/gpu_semantic_divergence.py --cluster-only \
    --checkpoint runs/semantic_samples.jsonl \
    --equivalence entailment \
    --out runs/uncertainty_qwen25_7b_entail.jsonl      # CPU, ~4h, checkpointed

python scripts/exp_equivalence_relations.py            # compares all three
python scripts/exp_significance.py                     # picks the new relation up
```

The code path is ready and needs no changes. `exp_significance.py` already reads
`runs/uncertainty_qwen25_7b_entail.jsonl` if it exists, and the entailment cache
appends verdicts as they are produced, so an interruption costs minutes.

**Read `scripts/exp_equivalence_relations.py`'s docstring before spending the
four hours.** It states the only two possible outcomes and why a small positive
difference is not one of them.

### What I would do next, in order

1. **Unblock `huggingface.co` and run the entailment pass.** It answers a
   standing objection, and a null result is a real answer. Budget four hours.
2. **Extend the clustered intervals to chapter 5's verifier reach — this one is
   NOT blocked.** Scope 0.2283 and 0.9033 and the false-alarm rates 0.0200 /
   0.0987 are proportions over steps nested in solutions, with no intervals at
   all, and they feed chapter 7's "the best verifier is net-negative" verdict
   directly. The data is committed in `runs/semantic_scope_*.json` with
   `question` as the clustering key, so no download is needed:

   | file | split | rows | questions | mean cluster |
   |---|---|---|---|---|
   | `semantic_scope_prm` | positive | 1,500 | 691 | 2.17 |
   | `semantic_scope_prm` | control | 750 | 267 | 2.81 |
   | `semantic_scope_judge_independent` | positive | 600 | 272 | 2.21 |
   | `semantic_scope_judge_independent` | control | 300 | 102 | 2.94 |

   Cluster sizes of 2.2–2.9 are C1-like rather than global-error-like, so the
   correction is probably small — but that is exactly the prediction this
   session got wrong twice, and `proportion_ci` / `proportion_delta_ci` already
   exist. Half an hour of work. **Do this one first.**
3. **Chapter 6's allocation results** get the same treatment. Lower stakes: the
   influence-vs-uniform result was refuted five separate ways, so an interval is
   unlikely to move it.
4. **Selective risk vs α.** "Misses the target by 3×" (0.1491 against 0.05) has
   no interval. The gap is large enough that it is almost certainly safe, but
   "almost certainly" is what this session has twice found worth checking.

---

## 5. Files to know about

| path | what it is |
|---|---|
| `src/car/eval/inference.py` | all the interval machinery; module docstring explains why the resampling unit is the solution and why the obvious reason is wrong |
| `tests/test_inference.py` | 32 tests; the two load-bearing ones pin the design-effect mechanisms |
| `scripts/exp_significance.py` | every headline AUROC, paired differences, design effect, power, resolution |
| `scripts/exp_rate_intervals.py` | the same for rates, plus the Mistral sensitivity bound |
| `docs/FINDINGS-SIGNIFICANCE.md` | the lab notebook for all of the above, including limits |
| `docs/thesis/07-the-assembled-gate.md` §7.7 | the chapter section carrying the method |
| `runs/significance.json`, `runs/rate_intervals.json` | committed, because the chapters cite them |

### Limits stated in the work, not to be rediscovered

- The intervals cover **corpus sampling variability with the score held fixed**.
  They do not cover the cost of *fitting* the score. For the probe that is the
  larger effect (0.8748 on the selection split vs 0.6968 on test) and §7.6
  quotes it separately.
- Percentile intervals, not BCa. Fine at these skews; revisit if a claim ever
  turns on the third decimal, which none do.
- The power calculation uses a normal approximation. Two significant figures is
  all that should be read off 0.076, and the coincidence with 0.65 is not a
  derivation.
