# Hand-validation of derived GSM8K dependency graphs

Every GSM8K dependency edge in this project is **derived**, not annotated:
line *i* is linked to line *j* when an operand of *i* equals the result of *j*.
That derivation sits underneath every GSM8K topology and allocation number, and
it had never been checked against the source text.

50 graphs adjudicated, stratified 25/25 by whether the graph contains an
ambiguous link. Reproduce with:

```bash
python scripts/validate_dependency_graphs.py extract && python scripts/validate_dependency_graphs.py show
```

Judgements: `data/processed/depgraph_judgements.json`. Score with
`validate_dependency_graphs.py score`.

**Adjudicator: an LLM (Claude) reading each solution's prose against the derived
edges.** Not a human, and not blind — the derived edges were visible during
adjudication, so this measures *agreement with a careful reader*, not
independent reconstruction. For tracing which quantity an operand refers to,
that judgement is close to mechanical; it is still weaker evidence than blind
human double-annotation and should be described that way in the thesis.

---

## The validation immediately found a bug, and it was not the expected one

The caveat this exercise was meant to quantify was the 9.6% *ambiguous link*
rate. Instead the first packet exposed a systematic **recall** failure:

```
gsm8k_424
  L1: 10*9/10 = 9      [10<-given, 9<-given, 10<-given]
  L2: 10-9 = 1         [10<-given, -9<-?]      <- L1 produced 9. Edge (1,2) LOST.
  L5: 1000-250 = 750   [1000<-L3, -250<-?]     <- L4 produced 250. Edge (4,5) LOST.
```

The operand regex was `-?\d+(?:\.\d+)?`. The optional sign **absorbed the
subtraction operator**, so `"110-80"` produced `[110, -80]`; `-80` matched no
earlier result, and the dependency vanished. Every subtraction in the corpus
silently lost its edge. A second fault in the same regex dropped leading dots,
turning `.8` into `8`.

Fixed by parsing expressions with `ast` and collecting numeric literals, so an
operator character can never be absorbed into an operand
(`car.data.gsm8k.expression_operands`). Guarded by regression tests.

### Corpus effect of the fix

| metric | before | after |
|---|---|---|
| operand link rate | 30.7% | **35.4%** |
| orphan steps (no parents) | 27.7% | **19.7%** |
| mean longest path | 2.54 | **2.79** |
| max depth | 8 | 8 |
| steps with a non-terminal descendant | 26.6% | **29.9%** |
| graphs at depth ≥ 3 | 3,030 | **3,815** |
| shape: `other` | 22.9% | **5.9%** |

The `other` collapse is the strongest evidence the fix is right: those were
graphs fragmented into unclassifiable pieces by the missing edges, and they
resolve into ordinary chains and converging structures once the edges return.

**This strengthens the benchmark switch.** GSM8K's advantage over StrategyQA on
every propagation-relevant measure is larger than reported in Addendum 3, not
smaller.

---

## Measured error rate, after the fix

| stratum | graphs | edges | spurious | missing | edge error | graph error |
|---|---|---|---|---|---|---|
| ambiguous | 25 | 91 | 12 | 3 | **0.1648** | **0.4800** |
| clean | 25 | 48 | 2 | 0 | 0.0417 | 0.0400 |

Stratified to corpus weights (11.6% of graphs contain an ambiguous link):

> **corpus edge error rate 5.6%**, corpus graph error rate 9.1%

So roughly **1 edge in 18 is wrong**, and about **1 graph in 11** contains at
least one bad edge.

### The dominant failure is value collision

Almost every error is an operand that *coincidentally* equals an earlier
result while actually being a given. Three recurring shapes:

| pattern | example |
|---|---|
| a count reused | `gsm8k_7096`: L3's `3` is *3 coupons*, not L1's *$3 discount* |
| a fraction denominator | `gsm8k_2153`: L3's `4` is the `1/4` denominator, not L2's result |
| a unit/rate given | `gsm8k_5124`: L2's `400` is *gallons per acre*, not L1's *gallons per day* |

Three of the 15 errors are **missing** edges rather than spurious ones, all
from the same cause: two earlier lines produced the same value and the
"most recent producer wins" rule linked only the nearer one
(`gsm8k_377`, `gsm8k_4245`, `gsm8k_6735`).

### The ambiguity flag is a poor proxy for correctness

This matters for how the caveat should be stated:

- **False positives dominate.** The ambiguous stratum has 91 edges of which
  only 15 are wrong. Most flagged links are correct — the operand really is
  the earlier result.
- **False negatives exist.** `gsm8k_5251` was classified *clean* and still
  produced two spurious edges: the colliding value `2` is a "half → double"
  factor that never appears as a number in the question text, so the flag
  never fired.

The honest caveat is therefore **"~5.6% of derived edges are wrong"**, measured,
not "9.6% of links are ambiguous", which was a proxy that over-counts in one
direction and under-counts in the other.

---

## What changed in the conclusions

**Corrected.** Addendum 3 reported that the best structural allocation signal is
benchmark-dependent — `depth` on StrategyQA, `cut` on GSM8K. That was an
artifact of the missing edges. With the corrected graphs, **`depth` wins on
both**:

| policy | StrategyQA | GSM8K (fixed) |
|---|---|---|
| uniform | 0.2418 | 0.2447 |
| front | 0.2460 | 0.2556 |
| back | 0.2353 | 0.2264 |
| influence | 0.2461 | 0.2530 |
| **depth** | **0.2335** | **0.2219** |
| cut | 0.2384 | 0.2318 |
| spread | 0.0126 | **0.0338** |

The finding is simpler than before, not more complicated: ancestor count is the
best structural signal on both benchmarks, and the "pick it on dev data"
hedge is no longer needed on this evidence.

**Unchanged.** `front` and `influence` remain the two worst policies at every
depth — now a fifth independent replication, and the spread grows with depth as
before (0.0033 at depth 2 → 0.1279 at depth 6).

**Unaffected entirely.** C1 (69.8% of wrong steps locally valid), the per-step
error rate, and all of Chapter 5's verifier-scope results come from
Math-Shepherd step labels, not from derived GSM8K graphs. None of them touch
this code path.

---

## Limits

- 50 graphs, 139 edges. The stratified estimate carries roughly ±2% at this
  sample size.
- LLM adjudication, not blind human annotation (see the header).
- Only edges *between annotated calculator steps* are in scope. Solutions
  routinely contain unannotated reasoning lines (`gsm8k_4767`: "$100 - $60 =
  $40" has no `<<>>`), and a dependency running through one of those is
  invisible to any operand-matching scheme. Those are not counted as missing
  here, so **5.6% is a lower bound on total edge error**.
- The residual 5.6% is not fixable by better matching alone: distinguishing
  "3 coupons" from "$3 discount" requires reading the sentence, not the
  arithmetic.
