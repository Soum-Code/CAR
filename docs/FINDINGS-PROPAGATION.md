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
- Chains only. Diamond and tree topologies are not tested, and influence
  weighting could plausibly do better where a step has many *immediate*
  dependents rather than a linear tail.
- Local error rate is i.i.d. across positions. If early steps were
  intrinsically harder, front-loading would gain — that is worth measuring on
  real StrategyQA traces before discarding §4.2 permanently.

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
