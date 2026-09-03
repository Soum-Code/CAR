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
- `local_error_rate = 0.15` is still assumed, not measured. **Done below.**

---

# Addendum 4: the per-step error rate, measured

Source: [Math-Shepherd](https://arxiv.org/abs/2312.08935) (Wang et al., ACL
2024) — GSM8K solutions generated by Mistral-7B-SFT with an automatic `+`/`-`
label on every step. 25,971 solutions, 93,129 steps.

Two independent signals per step, which is what makes this dataset decisive:

- **local** — does the inline `<<expr=result>>` arithmetic hold? Checked
  deterministically with `safe_eval`. No model, no judge.
- **global** — Math-Shepherd's `+`/`-`, i.e. does this prefix still lead to the
  right answer? Inherits corruption from a bad premise.

So the local/global gap that `car.propagation` simulated can be **measured**.

## A sampling trap, found twice

The Math-Shepherd file is sorted into contiguous blocks by label and task:

| file offset | task | final-answer accuracy |
|---|---|---|
| 0% | GSM8K | **0.0%** |
| 25% | GSM8K | **100.0%** |
| 50% | MATH | 100.0% |
| 75% | MATH | 0.2% |

The first run used an 80 MB prefix and reported "final answer wrong: 100.0%" —
every number came from the all-negative block. Fixed with a 48-chunk strided
download. The trap then reappeared inside the fixed file, because
`load_solutions(limit=N)` still takes a *prefix* (4.8% accuracy vs 26.7% for
the whole file); `load_solutions` now takes a `stride`, and a test asserts the
prefix read is more skewed than the strided one.

Second caveat, which no sampling fixes: **Math-Shepherd is a constructed PRM
training set.** It keeps a deliberate mix of good and bad completions, so its
raw `+`/`-` balance is a property of their construction, not the model's error
rate. All rates below are therefore **stratified by final-answer correctness**
and post-stratified to the model's reported accuracy (~41–52%).

## The measurement

| stratum | solutions | steps | local err | global err | inherited |
|---|---|---|---|---|---|
| final answer CORRECT | 6,940 | 21,505 | 0.0242 | 0.0000 | 0.0000 |
| final answer WRONG | 19,031 | 71,624 | 0.1641 | 0.7106 | **0.4961** |
| raw sample (biased) | 25,971 | 93,129 | 0.1305 | 0.5465 | 0.3816 |

Post-stratified:

| assumed model accuracy | local err | global err |
|---|---|---|
| 40% | 0.1081 | 0.4264 |
| **45%** | **0.1011** | **0.3908** |
| 52% | 0.0913 | 0.3411 |

**`local_error_rate ≈ 0.10`, not the 0.15 assumed throughout.** The estimate is
insensitive to the accuracy assumption (0.091–0.108); the global rate is not,
because global correctness is nearly definitional within each stratum.

## The central claim, measured rather than simulated

Within wrong-answer solutions: local error 0.1641, global error 0.7106,
inherited corruption **0.4961**.

> **69.8% of globally-wrong steps are arithmetically perfect.** They are wrong
> only because a premise was.

A calculator — a *sound, deterministic, non-parametric* verifier — cannot see
any of them. This is the local/global gap that Result 1 predicted, now measured
on 93k real steps rather than assumed in a simulator.

## Propagation is near-absorbing in practice

| quantity | value |
|---|---|
| steps after the first bad step still labelled `-` | **95.9%** |
| solutions that fully recovered | **0** of 25,971 |
| locally-valid steps downstream of a local error that are globally wrong | **89.6%** |

Zero recoveries out of 25,971. The `CORRUPT` state in the Markov model is not
merely sticky, it is effectively absorbing for this model.

## The last escape for front-loading is closed

| position | n | local err | global err |
|---|---|---|---|
| 1 | 23,444 | 0.1095 | 0.3077 |
| 2 | 23,480 | 0.1098 | 0.5159 |
| 3 | 17,229 | 0.1319 | 0.6475 |
| 4 | 9,773 | 0.1662 | 0.7250 |
| 5 | 4,845 | 0.1981 | 0.7963 |
| 6 | 2,186 | 0.2031 | 0.8569 |
| 8 | 417 | 0.2182 | 0.9124 |

**corr(position, local error rate) = +0.950.** Local error rate doubles from
11% at step 1 to 22% at step 8.

Earlier caveats repeatedly noted that if *early* steps were intrinsically
harder, front-loading would gain — the one untested escape for influence
weighting. The data says the opposite: **later steps are harder**, which
favours back-loading further. That escape is now closed, on measured data.

It also means the i.i.d. per-step error assumption in `car.propagation` is
wrong in a direction that makes the existing results *conservative* about how
much back-loading wins.

## Kotte feasibility, with a measured μ

Minimum fraction of steps any distribution-free method must verify or abstain
on, from Prop. 3 of [arXiv:2606.29054](https://arxiv.org/abs/2606.29054):

| assumed acc | μ | α=0.05 | α=0.10 | α=0.20 | α=0.30 | α=0.40 | α=0.50 |
|---|---|---|---|---|---|---|---|
| 40% | 0.4264 | 39.6% | 36.3% | 28.3% | 18.1% | 4.4% | ok |
| **45%** | **0.3908** | 35.9% | **32.3%** | 23.9% | 13.0% | ok | ok |
| 52% | 0.3411 | 30.6% | 26.8% | 17.6% | 5.9% | ok | ok |

**`alpha: 0.1` in `configs/default.yaml` forces verifying ≥32% of steps before
any method is even admissible.** The first α attainable without a floor is
**0.40**. This confirms the earlier suspicion and makes it quantitative: the
spec's α = 0.1 is not a modest target, it is one that costs a third of the
budget as an entry fee.

## Re-running the model with the measured rate

| budget | assumed 0.15 | measured 0.101 | delta |
|---|---|---|---|
| 0% | 0.4780 | 0.3471 | −0.1309 |
| 25% | 0.2591 | 0.1847 | −0.0744 |
| 50% | 0.1262 | 0.0881 | −0.0381 |
| 75% | 0.0475 | 0.0325 | −0.0150 |

Earlier simulations were pessimistic by ~13 points at zero budget. Rankings
between allocation policies are unaffected.

## Limits

- **Mistral-7B-SFT, not Llama 3.1 8B.** Re-measure before quoting for the
  model CAR actually uses. The method transfers; the number may not.
- Labels are automatic Monte-Carlo estimates of "leads to a correct answer",
  not proofs. A lucky wrong step can be labelled `+`.
- 12.3% of steps carry no arithmetic and are reported as uncheckable rather
  than assumed correct.
- Post-stratification leans on the reported ~41–52% accuracy; the local rate is
  robust to it, the global rate is not.

---

# Addendum 5: verifier scope, measured — reach is semantic, not structural

Chapter 5's central experiment. Two parts: an arithmetic verifier measured on
CPU, and a semantic verifier measured on a Kaggle P100.

## Part 1 — arithmetic reach is a window, and it tops out at 20%

Scope is not a property of a verifier *type*. It is a property of how much
context the verifier **re-examines**. A calculator checking only step *t*
cannot know its operands came from a wrong step *t−2*; the same calculator
re-checking *t−k…t* can.

Measured on 35,535 real inherited-corruption steps:

| window k | scope | cost (steps/call) | scope per step |
|---|---|---|---|
| 0 | **0.0000** | 1.00 | 0.0000 |
| 1 | 0.1197 | 1.85 | 0.0648 |
| 2 | 0.1690 | 2.42 | **0.0699** |
| 3 | 0.1880 | 2.73 | 0.0689 |
| ∞ | **0.1999** | 3.01 | 0.0663 |

The naive step-local verifier detects 1 case in 35,535 — literally zero reach,
confirming by measurement what the propagation model assumed by construction.
Detection is a clean step function in distance: window *k* catches corruption
originating within *k* hops and nothing beyond.

**Efficiency peaks at k=2**, so unbounded lookback is not the design point.

**The ceiling:** 80.0% of inherited-corruption steps have no upstream
arithmetic error at all. They are globally wrong with perfect arithmetic
everywhere above them, because the mistake is in the *setup* — wrong quantity,
wrong operation, misread problem. No arithmetic verifier at any window size can
see them.

## Part 2 — scope depends on the verifier, and only one kind works

Four verifiers on the arithmetic-blind population (locally valid, globally
wrong — the steps arithmetic provably cannot see), each against a control of
locally-valid **and** globally-correct steps. Scope is detection rate on the
population, false alarm is detection rate on the control.

| verifier | independent of generator? | task-specialised? | validation sep | scope | false alarm | **scope − FA** |
|---|---|---|---|---|---|---|
| arithmetic, step-local (k=0) | yes | n/a | n/a | 0.0000 | — | 0.0000 |
| arithmetic, unbounded lookback | yes | n/a | n/a | 0.1999 | — | 0.1999 |
| **same-model critic** (mistral-7b-sft) | **no** | no | **0.0000** | **0.0000** | 0.0000 | 0.0000 |
| independent judge (Qwen2.5-7B) | yes | no | 0.2133 | 0.2283 | 0.0200 | 0.2083 |
| **task PRM** (math-shepherd-7b) | yes | **yes** | 0.5788 | **0.9033** | 0.0987 | **0.8047** |

Read top to bottom, this is a cleaner result than "reach is semantic":

**The same-model critic approves everything.** mistral-7b-sft *wrote* these
solutions, and asked whether each step is sound it answers SOUND to all of them
— including the steps carrying its own `-` label. Validation separation is
exactly 0.0000 (mean goodness 1.00 on known-good, 1.00 on known-bad). It detects
zero errors. This is Huang et al. (ICLR 2024) measured directly: the parameters
that produced the error cannot catch it. Independence is not optional.

**Independence alone barely beats arithmetic.** Qwen2.5-7B is a different model
family, prompted zero-shot to check each step. It reaches scope 0.2283 — a
hair above the 0.1999 arithmetic ceiling. Its validation separation is weak
(0.21), so it is a poor step-checker in this format even on known labels. A
general instruct model is not enough.

**Only the task-specialised, independent verifier achieves high scope.** The
Math-Shepherd PRM — trained for exactly this judgement, and not the generator —
reaches 0.9033 at 9.9% false alarm.

So C3 sharpens to:

> Reach is not about how far back you look. It requires a verifier that is
> **both independent of the generator and specialised for the task**.
> Independence alone (a general judge) buys almost nothing over arithmetic; the
> generator judging itself buys literally nothing; only a task-matched,
> independent verifier closes the gap.

Result data: `runs/semantic_scope_prm.json`,
`runs/semantic_scope_judge_independent.json`,
`runs/semantic_scope_judge_same_model.json`.

Populations: PRM arm n=1,500/750; judge arms n=600/300 (generation is slower
than PRM scoring, so the judge arms use a smaller sample — scope precise to
roughly ±3.5%).

## The harness gate, and why the first answer was wrong

The first completed run reported the *opposite*: scope 0.9053 but false alarm
0.9493, i.e. **negative** net scope. It would have supported "semantic
verification cannot close the gap either" — the interesting negative result.

It was wrong, and the tell was that the PRM flagged 94.9% of the control group:
steps carrying Math-Shepherd's own `+` labels, the data this PRM was *trained*
on. A model that cannot recognise its own training labels is not measuring
anything.

`validate_prm` now runs before every measurement, scoring 400 known-label steps
and aborting if the good/bad separation is below 0.15. It caught three
successive broken configurations:

| run | separation | cause |
|---|---|---|
| v10 | — (not yet gated) | `ки` appended only to the final step, no blank lines |
| v11 | 0.0108 | prompt fixed, but candidate token ids wrong |
| v14 | 0.0570 | step tag resolved to 1107 `'ки'` instead of 12902 `'▁ки'` |
| v17 | **0.5788** | tokenizer pinned; passes |

Root cause of all three: **Kaggle's transformers tokenizes this SentencePiece
model differently from how it was trained.** The `▁` word-boundary marker is
not added, so `▁+` (648) became `+` (28806) and `▁ки` (12902) became `ки`
(1107). Both `legacy=True` and `use_fast=False` failed to restore it; only
pinning `transformers==4.44.2` did.

The lesson is worth carrying into the thesis: a verifier-scope number is only
as good as the evidence that the verifier works at all. Without the gate, this
project would have published a confident negative result produced entirely by a
tokenizer mismatch.

### A second harness bug, in the judge arms

The judge arms had their own version of the tokenizer trap. The first judge run
used **right padding** for batched generation; for decoder-only models that
inserts pad tokens between the prompt and the continuation, so the model
generates from padding and the output is corrupt. It surfaced as a
`right-padding was detected` warning and a Qwen scope of 0.1933 that could not
be trusted. Fixed to left padding, the number moved to 0.2283 — same
conclusion, but the point stands: batched-generation padding side is a silent
correctness bug, not a warning to ignore.

The same-model arm also needed a base-model path: mistral-7b-sft has no chat
template, so `apply_chat_template` raised and the arm was lost on the first
attempt. It now falls back to a plain `...\nAnswer:` completion.

Both fixes are guarded going forward: staged Kaggle scripts are now
syntax-checked before upload (a stale copy of a fixed file had already cost one
failed run), and unparseable judge verdicts score NaN rather than a silent
default.

## Limits

- Two generators involved: the verified solutions come from Mistral-7B-SFT
  (Math-Shepherd); one verifier arm (mistral-7b-sft) is that same generator,
  which is the point, but the PRM and Qwen arms are independent of it.
- The PRM was trained on Math-Shepherd labels and is evaluated against those
  same labels. Its 0.90 is therefore partly *in-distribution*: a ceiling on
  what a well-matched semantic verifier achieves, not evidence that any LLM
  judge would. The Qwen arm — a general judge, zero-shot — is the honest lower
  bound for "independent but not task-specialised", and it lands near the
  arithmetic ceiling.
- Control steps come from correct solutions and the population from wrong ones,
  so some separation may reflect problem difficulty rather than the specific
  step. A within-solution control would be tighter.
- PRM arm n=1,500/750 (±1.5%); judge arms n=600/300 (±3.5%).
- No retrieval arm. GSM8K premises are the problem statement, not an external
  corpus, so retrieval+entailment has no meaning here; it needs StrategyQA
  evidence, where propagation barely occurs (Addendum 2). That arm is
  genuinely not measurable on this benchmark.

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
