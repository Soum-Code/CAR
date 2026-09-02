# Stress-test of the surviving contribution

Step 2 of the workplan. Tests whether POSITIONING.md §4.1 (risk control over
dependent steps) can carry a thesis, and whether §4.2 (influence-weighted
allocation) survives contact with numbers.

Reproduce: `python scripts/exp_propagation.py`, `exp_allocation.py`, `exp_cost.py`.
Model: `src/car/propagation.py`. Locked in by `tests/test_propagation.py` (26 tests).

**Summary: §4.1 survives and is stronger than expected. §4.2 is dead. So is the
spec's hypothesis H3.**

---

## The model

A step fails in two distinguishable ways:

```
global_correct(t) = local_valid(t) AND NOT premise_corrupt(t)
```

- **local invalidity** — the step doesn't follow from its premises (`47*3 = 131`)
- **inherited corruption** — the step follows perfectly, but a premise is false
  (*Aristotle died in 1850, so he could have used a laptop*)

Verifiers differ in which they can see, and that turns out to control everything:

| axis | meaning |
|---|---|
| `scope` | P(verifier detects inherited corruption). 0 = calculator, 1 = evidence retrieval |
| `scope_decay` | how that fades with propagation distance. Singh & Pawar ([2608.14588](https://arxiv.org/abs/2608.14588)) measured escape probabilities of 24.6% / 48.3% / 89.3% across successive boundaries → fitted decay ≈ 0.377 |
| `outcome` | does the answer depend on the last step (`terminal`) or on every step having been right when committed (`conjunctive`)? |

As an age-indexed Markov chain, with `v_t` the verification probability:

```
CLEAN     -> CORRUPT_1   e * (1 - v_t)                    <- no scope term
CORRUPT_k -> CLEAN       v_t * scope * decay^(k-1)        <- scope AND decay
```

**Entering corruption doesn't depend on the verifier's reach. Escaping it does.**
That asymmetry drives every result below. Closed form validated against
simulation to within 0.015 across 12 configurations.

---

## Result 1: local risk certifies the wrong quantity — CONFIRMED, strongly

Purely local verifier, chain length 8, local error 15%:

| policy | local risk | global risk | final error | calls |
|---|---|---|---|---|
| never verify | 0.1510 | 0.4884 | 0.7309 | 0.00 |
| uniform 25% | 0.1517 | 0.4086 | 0.6180 | 2.00 |
| uniform 50% | 0.1487 | 0.3145 | 0.4607 | 4.01 |
| uniform 75% | 0.1372 | 0.2037 | 0.2667 | 6.00 |

**Local selective risk is pinned at ~0.15 while final error spans 0.73 → 0.27.**
It carries essentially no information about the outcome.

This matters because local selective risk is *what a verifier reports*, and
therefore what conformal machinery can control. CSA controls exactly this
quantity. Under propagation, controlling it at level α bounds nothing about the
answer.

**This is the strongest result here and it is what §4.1 should be built on.** It
is not "steps are dependent, so exchangeability is awkward" — it is that the
certified quantity and the quantity of interest come apart, and the gap is
unbounded.

---

## Result 2: verifier scope is the controlling variable — CONFIRMED

Final-answer error vs budget:

| scope | v=0% | v=25% | v=50% | v=75% |
|---|---|---|---|---|
| 0.00 | 0.735 | 0.626 | 0.463 | 0.267 |
| 0.25 | 0.735 | 0.523 | 0.328 | 0.161 |
| 0.50 | 0.735 | 0.449 | 0.247 | 0.102 |
| 1.00 | 0.735 | 0.336 | 0.153 | 0.056 |

At v=25%, scope alone moves error by ~2×. **The spec's verifier table lists
calculator / retrieval / sandbox as interchangeable reliability mechanisms.
They are not** — they sit at different points on this axis, and the difference
dominates the choice of gating policy.

Scope is not a variable in the original spec at all. It should be.

---

## Result 3: "verify early" is FALSE — H3 does not survive

Final-answer error, budget 3 of 8 steps:

| scope | front-loaded | uniform | back-loaded | optimal | winner |
|---|---|---|---|---|---|
| 0.00 | 0.5480 | 0.5450 | 0.5468 | 0.5450 | uniform |
| 0.25 | 0.5086 | 0.4228 | 0.3565 | 0.2717 | **back** |
| 0.50 | 0.4767 | 0.3345 | 0.2331 | 0.1058 | **back** |
| 1.00 | 0.4293 | 0.2240 | 0.1065 | 0.0019 | **back** |

Front-loading is the **worst** shape at every scope > 0. Verified robust across:

- **repair success** 1.0 → 0.05 — back wins at every level
- **chain length** 4 → 32 — front-loading gets *worse* as chains lengthen
  (0.29 → 0.67), back-loading gets better (0.12 → 0.06)
- **decay** 1.0 and 0.377 — decay narrows the gap but never flips it
- **outcome model** terminal and conjunctive — under conjunctive everything is
  flat and scope becomes irrelevant

This contradicts spec hypothesis H3 ("early verification will produce a larger
reduction in propagated errors than end-only verification"), which is stated
unconditionally. It also kills POSITIONING.md §4.2, since influence weighting
*is* front-loading: in a chain, descendant count decreases monotonically with
position.

### The attempted rescue, and why it failed

Late verification is cheap in *calls* and expensive in *tokens* — catching a bad
premise at step 7 means steps 2-7 were built on it and must be redone. That is
the one honest argument for verifying early, and the model measures it:

| shape | final err | v-calls | regen steps | total cost | err × cost |
|---|---|---|---|---|---|
| front-loaded | 0.4172 | 3.00 | 0.31 | 11.31 | 4.72 |
| uniform | 0.2027 | 3.01 | 0.87 | 11.88 | 2.41 |
| back-loaded | 0.0869 | 3.01 | 1.24 | 12.25 | **1.07** |

The effect is real — front-loading does waste less work — but it is ~8% of total
cost against a ~5× accuracy gap. Not close.

### What the optimum actually looks like

| scope | s0 | s1 | s2 | s3 | s4 | s5 | s6 | s7 | shape |
|---|---|---|---|---|---|---|---|---|---|
| 0.00 | .38 | .38 | .38 | .38 | .38 | .38 | .38 | .38 | flat |
| 1.00, decay=1.0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | terminal block |
| 1.00, decay=0.377 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | **periodic** |

The decay row is the interesting one. When the verifier's reach fades, a single
late check can no longer see far enough back, and the optimum becomes
**periodic checkpointing** at roughly the interval over which detectability
survives. That is a concrete, testable design rule and it is not in any of the
compared work.

---

## Where this leaves the project

### Keep

**The certification gap (§4.1).** Existing conformal gating certifies local step
correctness. Under propagation that bounds nothing about the answer, and Result 1
measures the gap directly. This is sharper than the "dependence breaks
exchangeability" framing — it is not a technical inconvenience, it is a
mismatch between the certified and the desired quantity.

**Verifier reach as the design axis.** Allocation should be driven by how far
back the verifier can see, not by how uncertain a step is or how many
descendants it has. Nobody in the compared literature models this: CSA assumes
a deterministic verifier on independent items, Sherlock uses fan-in, PRMs score
locally.

### Drop

- **Influence-weighted allocation (§4.2).** Front-loading loses everywhere.
- **Hypothesis H3.** Not supported under any variant tested. If it is retained
  at all it must be restated conditionally, and the condition is not one this
  model produces.
- **"Verify before commitment" as a headline claim.** The model says verifying
  *later* is usually better when the verifier has reach.

### Honest limits

- Repair is modelled as fully restoring both local validity and global
  correctness. Real revision is messier. Tested down to `repair_success=0.05`
  and the ranking held, but the mechanism is still idealised.
- `terminal` is generous to late verification; `conjunctive` makes everything
  flat. Reality is between them and neither endpoint favours front-loading.
- Local error rate is i.i.d. across positions. If early steps were
  intrinsically harder, front-loading would gain — worth measuring on real
  StrategyQA traces.

---

# Addendum: the chain result could not see influence weighting

Everything above ran on **linear chains**, and on a chain

```
influence_schedule(v)  ==  front_schedule(v)      exactly
corr(position, influence) = -1.000
```

because descendant count is a strictly decreasing function of position. The
test rejected influence weighting without ever evaluating it as a distinct
policy. A chain also routes every error through the terminal node, handing a
structural win to back-loading that is a property of chains, not of reasoning.

Re-run on DAGs where the two signals decouple (`src/car/topology.py`,
`scripts/exp_topology.py`, 22 tests in `tests/test_topology.py`).

## Two artifacts found and fixed first

Both would have invalidated the rerun:

1. **Decay was inert.** Effective scope was computed from the distance to the
   *nearest incorrect ancestor*. Once corruption spreads, every intermediate
   node is itself incorrect, so that distance is always 1 and `decay**0 == 1`
   always. Every observed age was 1, and the decay=1.0 and decay=0.377 result
   tables were byte-identical. Now tracked as distance from the corruption
   **origin**; observed distribution is {0: 3731, 1: 1851, 2: 637, 3: 169}.

2. **The terminal node was verifiable.** Ancestor-weighted schedules put
   `p = 1.0` on it, and with `scope = 1.0` that is a perfect oracle on the final
   answer — driving measured error to exactly 0.0000 for three policies. A
   verifier that can definitively check the answer makes the reasoning chain
   unnecessary. The terminal is now excluded from the budget.

## Result: influence weighting is distinguishable, better than front, still loses

Final-answer error, matched budget 37.5% of nodes, global verifier:

| topology | corr(pos,infl) | uniform | front | back | influence | depth | winner |
|---|---|---|---|---|---|---|---|
| chain(10) | −1.000 | 0.3055 | 0.4240 | 0.2007 | **0.4240** | 0.2007 | back |
| parallel(3×3) | −0.533 | 0.4868 | 0.5310 | 0.4847 | 0.5333 | **0.3889** | depth |
| converging(d=3) | −0.881 | 0.5300 | 0.6945 | 0.3217 | 0.6191 | **0.1487** | depth |
| diamond(3×3) | −0.976 | 0.5333 | 0.6262 | 0.3755 | 0.6217 | **0.3459** | depth |
| bushy(13,s=0) | −0.652 | 0.5971 | 0.6689 | **0.5451** | 0.6654 | 0.5624 | back |
| bushy(13,s=3) | −0.776 | 0.5955 | 0.6714 | **0.5296** | 0.6610 | 0.5453 | back |

Note the chain row: influence and front are identical to four decimals, which
is the confound made visible.

Off-chain the two policies do separate, and influence is the better of the two
— by 0.075 on converging, 0.004–0.010 elsewhere, and it is marginally *worse*
on parallel. So the separation is real but small and not consistent in sign.

**Influence weighting still loses to plain uniform on every topology tested**
(+0.047 to +0.119). The §4.2 rejection stands — but now for a tested reason
rather than a confounded one. Holds under decay = 0.377 and under weak scope
(0.3).

## The replacement finding: ancestor count, not descendant count

`depth` — allocate proportional to |ancestors(v)| — wins on 3/6 topologies at
full scope and 4/6 at weak scope, and is never far off elsewhere. It beats
`influence` on every non-chain topology tested, and beats raw `back`-loading on
parallel(3×3) by 0.096, so it is more than position in disguise.

The intuition is the reach story again, in structural form:

> Verify where the most upstream reasoning **converges**, not where the most
> downstream damage **could occur**. A node deep in the graph is downstream of
> many premises, so one check there screens many potential errors at once. A
> node with many descendants is only worth checking if the errors it causes are
> not catchable later — and with a verifier that has reach, they are.

Descendant count answers "how much damage could this cause?". Ancestor count
answers "how much can I screen with one call?". Under a budget, the second is
the question that matters — and it is the opposite of what the spec and §4.2
proposed.

## Caveats on the addendum

- Margins are 0.05–0.15 absolute. Real, consistent, not dramatic.
- `depth` wins on the structured topologies; `back` wins on both random bushy
  graphs. The right structural signal is itself topology-dependent.
- Six topologies, one error rate, one budget level in the main table. The
  budget sweep covers 0.15–0.65 but omits `depth`, so it understates the
  ranking — read the main table, not the sweep.
- Real reasoning DAGs are not any of these shapes. Extracting actual dependency
  graphs from StrategyQA traces is the obvious next step and would make this
  argument empirical rather than synthetic. **Done below.**

---

# Addendum 2: real StrategyQA graphs — the benchmark is the problem

StrategyQA ships human-annotated decompositions in BREAK style, where steps
reference earlier steps by index:

```
[1] How many kids did Julius Caesar have?
[2] How many kids did Genghis Khan have?
[3] Is #2 greater than #1?
```

Those `#N` markers *are* a dependency graph, authored by annotators rather than
inferred from generated text. Extracted all 2272 of them
(`src/car/data/strategyqa.py`, `scripts/exp_strategyqa_topology.py`).

## The graphs are tiny and almost flat

| property | value |
|---|---|
| questions | 2272 |
| steps per question | mean **2.95** (2: 626, 3: 1219, 4: 342, 5: 85) |
| longest path (depth) | mean **2.30** (2: 1657, 3: 555, 4: 56, 5: 4) |
| shapes | converging 1351 (59%), chain 888 (39%), branching 33 (1%) |
| root premises | mean 1.63 |
| dead-end steps | **0** across the whole corpus |
| corr(position, influence) | **−0.917** |

Six structures cover most of the corpus, and two cover 71%:

```
980  n=3   [1]<-root ; [2]<-root ; [3]<-[1,2]     "look up two facts, compare"
626  n=2   [1]<-root ; [2]<-[1]
230  n=3   [1]<-root ; [2]<-[1] ; [3]<-[2]
136  n=4   [1]<-root ; [2]<-[1] ; [3]<-root ; [4]<-[2,3]
```

## Propagation has almost nowhere to happen

- **72.9%** of questions have longest path exactly 2 — root → answer, one hop
- only **2.6%** reach depth ≥ 4
- **11.2%** of all steps have any descendant other than the terminal

That last figure is the one that matters. A step can only corrupt *downstream
reasoning* if downstream reasoning exists. For 88.8% of StrategyQA steps, the
only thing below them is the answer itself. There is no chain to snowball down.

## Policy differences replicate, but are negligible on this corpus

Budget 37.5% of nodes, global verifier:

| policy | final error | vs uniform |
|---|---|---|
| depth | **0.2338** | −0.0085 |
| back | 0.2359 | −0.0064 |
| cut | 0.2389 | −0.0034 |
| uniform | 0.2423 | — |
| influence | 0.2466 | +0.0043 |
| front | 0.2469 | +0.0046 |

The synthetic ordering **replicates exactly**: `depth` best, `influence` and
`front` worst, `influence` ≈ `front` (they differ by 0.0003 here, because
corr(pos, influence) = −0.917 on real graphs too). But total spread across all
six policies is **0.0132** — about one percentage point.

Broken down by graph size, the reason is obvious:

| n steps | count | uniform | front | back | influence | depth | cut | spread |
|---|---|---|---|---|---|---|---|---|
| 2 | 626 | 0.1808 | 0.1808 | 0.1808 | 0.1808 | 0.1808 | 0.1808 | **0.0000** |
| 3 | 1219 | 0.2512 | 0.2541 | 0.2471 | 0.2538 | **0.2466** | 0.2495 | 0.0075 |
| 4 | 342 | 0.2954 | 0.3081 | 0.2756 | 0.3079 | **0.2703** | 0.2860 | 0.0377 |
| 5 | 85 | 0.3375 | 0.3634 | 0.3032 | 0.3641 | **0.2863** | 0.3143 | 0.0778 |

On 2-step graphs every policy is *identical* — with the terminal excluded there
is exactly one verifiable node, so no allocation decision exists. The spread
grows monotonically with size and reaches 0.078 at n=5, where `depth` beats
`influence` by 0.078. The effect is real; the corpus is simply dominated by
graphs too small for it to appear.

## The actionable conclusion: StrategyQA is the wrong primary benchmark

The spec makes StrategyQA the *primary* benchmark and the calibration source.
On this evidence that is a mistake — it has essentially no propagation
headroom, which is the phenomenon the entire project is about.

GSM8K, for comparison (7473 training solutions):

| | StrategyQA | GSM8K |
|---|---|---|
| steps per item | 2.95 | 3.58 lines / **3.17 calculator steps** |
| depth | 2.30 | ~3.17 (solutions are chains, each line feeds the next) |
| fraction ≥ 3 steps | 27.1% | **64.2%** |
| fraction ≥ 4 steps | 2.6% | **35.6%** |
| max observed | 5 | 9 |

GSM8K has roughly **3× the propagation headroom** and a real tail. But it is a
pure chain, so influence collapses back onto position and the allocation
question degenerates to front-vs-back.

Neither benchmark supports the full thesis on its own:

- **StrategyQA** — topological variety (59% converging), no depth
- **GSM8K** — genuine depth, no topological variety

Recommended: promote GSM8K to primary for the propagation and early-vs-late
claims, keep StrategyQA for the calibration and evidence-grounded verification
claims, and be explicit that the allocation result needs either deeper
multi-hop data (HotpotQA-style, or StrategyQA restricted to n ≥ 4) or
synthetic depth extension.

Extracted graphs are cached at `data/processed/strategyqa_dags.json`.

---

# Addendum 3: primary benchmark switched to GSM8K

Acted on the recommendation above. `configs/default.yaml` now has
`data.dataset: gsm8k`, with StrategyQA retained as secondary for calibration
and evidence-grounded verification.

## Deriving GSM8K dependency graphs

GSM8K has no annotated decomposition, but its inline calculator markers make
the graph derivable: line *i* depends on line *j* when an operand of *i* equals
the result of *j*. Operands matching no earlier result are givens, i.e. roots.

```
He eats 32 ... because 2 x 16 = <<2*16=32>>      roots: 2, 16
He eats 16 ... because 2 x 8  = <<2*8=16>>       roots: 2, 8
He eats 48 ... because 32 + 16 = <<32+16=48>>    depends on BOTH earlier lines
```

That is a *converging* structure, which matters: GSM8K is not merely a deeper
chain.

Extraction quality on 7473 training solutions:

| metric | value | reading |
|---|---|---|
| usable (≥2 calc steps) | 93.3% | 499 single-step solutions dropped |
| operand link rate | 30.7% | rest are givens from the problem statement — expected |
| **ambiguous links** | **9.6%** | operand matches an earlier result *and* a question number |
| orphan steps | 27.7% | non-first steps recomputing from givens — parallel branches, not failures |

The 9.6% ambiguity is the honest weak point. Linking to the earlier result is
right far more often than not, but roughly one link in ten could go either way.

## Structural comparison

| property | StrategyQA | GSM8K |
|---|---|---|
| questions | 2272 | **6974** |
| steps per item | 2.95 | 3.34 |
| longest path (mean) | 2.30 | **2.54** |
| max depth observed | 5 | **8** |
| steps w/ non-terminal descendant | 11.2% | **26.6%** |
| corr(position, influence) | −0.917 | **−0.771** |
| questions at depth ≥ 3 | 615 | **3030** |

Shapes:

| | StrategyQA | GSM8K |
|---|---|---|
| chain | 39.1% | 38.9% |
| converging | 59.5% | 25.7% |
| branching | 1.5% | **12.4%** |
| other | 0.0% | **22.9%** |

**Correction to Addendum 2.** It claimed GSM8K has "~3× the propagation
headroom", inferred from calculator-step counts. Measured on extracted graphs,
mean *depth* improves only 2.30 → 2.54 (10%). The 2.4× gain is on the headroom
metric specifically (11.2% → 26.6%), which is the right one but not what the
earlier sentence said.

## Policy choice matters more on GSM8K

Budget 37.5%, global verifier:

| policy | StrategyQA | GSM8K |
|---|---|---|
| uniform | 0.2418 | 0.2262 |
| front | 0.2460 | 0.2350 |
| back | 0.2353 | 0.2137 |
| influence | 0.2461 | 0.2285 |
| depth | **0.2335** | 0.2121 |
| cut | 0.2384 | **0.2071** |
| **spread** | **0.0126** | **0.0279** |

Spread doubles. Note the best policy differs — `depth` on StrategyQA, `cut`
(ancestors × descendants, i.e. bottlenecks) on GSM8K, which has 8× more
branching for bottlenecks to exist in. **The best structural signal is
benchmark-dependent**, so it should be selected on dev data rather than assumed.

Broken down by depth, which is what the whole propagation argument needs:

| depth | count | uniform | front | back | influence | depth | cut | spread |
|---|---|---|---|---|---|---|---|---|
| 1 | 496 | 0.1516 | 0.1518 | 0.1514 | 0.1516 | 0.1516 | 0.1516 | **0.0003** |
| 2 | 3418 | 0.1990 | 0.2010 | 0.1982 | 0.1942 | 0.2016 | **0.1856** | 0.0159 |
| 3 | 2141 | 0.2561 | 0.2694 | 0.2344 | 0.2607 | 0.2292 | **0.2254** | 0.0440 |
| 4 | 714 | 0.2890 | 0.3178 | 0.2495 | 0.3101 | **0.2387** | 0.2578 | 0.0791 |
| 5 | 175 | 0.3199 | 0.3640 | 0.2595 | 0.3595 | **0.2504** | 0.2948 | **0.1136** |

At depth ≥ 4 the gap between the best and worst policy reaches 0.08–0.11 —
an order of magnitude larger than anything visible on StrategyQA. `front` and
`influence` remain the two worst policies at every depth, which is now the
fourth independent replication of that result.

`configs/default.yaml` therefore also sets `data.min_depth: 3`, keeping the
3030 questions where allocation is a real decision.

## Remaining risks

- **9.6% ambiguous links.** A dependency-labelling error rate that no amount of
  simulation fixes. Worth hand-checking ~50 graphs before relying on it in the
  thesis.
- GSM8K is arithmetic only. The evidence-grounded verification story (retrieval,
  prompt injection, verifier scope < 1) has no home there — that is what
  StrategyQA is retained for, and the two claims now rest on different datasets.
- `local_error_rate = 0.15` is still assumed, not measured. Measuring the real
  per-step error rate of Llama 3.1 8B on GSM8K is the obvious next input, and it
  also feeds the Kotte feasibility test (`mu > alpha` ⟹ minimum abstention).

---

## Revised pitch

> Conformal gating for LLMs certifies the quantity a verifier reports: whether
> a step is locally valid. In multi-step reasoning that is not the quantity
> anyone cares about. A step can be locally valid, accepted under a certified
> risk bound, and still wrong because a premise three steps back was false —
> and we show the gap between the certified quantity and final-answer error is
> unbounded. What closes it is not more verification but verification with
> sufficient *reach*, and we characterise the optimal allocation as a function
> of how far back a verifier can see.

One measurable gap, one controlling variable, one design rule. The
censored-feedback machinery stays in the system and is cited to CSA.
