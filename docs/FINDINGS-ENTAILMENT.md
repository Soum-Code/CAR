# The reference relation does not rescue the signal

Chapter 7 reports that semantic divergence barely ranks step error, and that
the number *moves with the clustering relation*: over all steps, numeric
equivalence gives AUROC 0.5488 and re-clustering the same samples by string
equality gives 0.4904, below chance. That leaves an obvious objection to the
negative result. `numeric_equivalence` is a cheap stand-in for the relation the
literature actually uses, so the result might be a property of the stand-in
rather than of sampling-based step uncertainty.

This answers it by running the reference relation. Kuhn et al. and Farquhar et
al. cluster samples by **bidirectional entailment** under `microsoft/deberta-
large-mnli`; that is now `EntailmentEquivalence`.

**It does not rescue the signal.** Its AUROC is statistically indistinguishable
from the cheap relation's, and its 95% interval tops out at 0.614 — still far
from usable.

Reproduce:

```bash
python scripts/gpu_semantic_divergence.py --cluster-only \
    --checkpoint runs/semantic_samples.jsonl \
    --equivalence entailment \
    --out runs/uncertainty_qwen25_7b_entail.jsonl      # 19.6h CPU, resumable
python scripts/exp_equivalence_relations.py
```

All three relations see the **same 12,865 generations**, so nothing here is
confounded with a new draw from the generator.

---

## 1. Three relations, one answer

| relation | mean divergence | unanimous steps | AUROC all | AUROC test | composite |
|---|---|---|---|---|---|
| numeric equivalence | 0.4239 | 1,006 / 2,573 | 0.5488 | 0.5740 | 0.5742 |
| exact string match | 0.6468 | 524 / 2,573 | 0.4904 | 0.5287 | 0.5612 |
| **bidirectional entailment** | **0.1205** | **1,854 / 2,573** | 0.5174 | 0.5625 | 0.5805 |

On test, entailment is 0.0114 below numeric equivalence. **That gap is not a
ranking.** A solution-clustered bootstrap over the 182 test solutions puts it
at 95% CI **[−0.087, +0.059]**, with P(entailment actually worse) = 0.63 — a
coin flip.

An earlier draft of this document called entailment "slightly worse" while
dismissing its +0.0063 composite advantage as "well inside noise on 925 test
steps". That was a double standard, and in the direction that flattered the
conclusion: the gap being treated as a ranking is *wider* in CI terms than the
one being dismissed. Both are noise.

What survives the interval is the thing that matters:

| | |
|---|---|
| entailment AUROC on test | 0.5625, 95% CI **[0.512, 0.614]** |

Even the optimistic end of that interval is a signal nobody would gate on.

> Chapter 7's negative result is not an artifact of the equivalence function.
> Swapping in the relation the literature uses moves the measurement by less
> than its own sampling error, and leaves it weak.

## 2. How much corroboration three relations actually give

It is tempting to argue that three different clusterings landing in the same
place is a much stronger negative than one. That argument is weaker than it
looks, and the honest version is worth stating because the overclaim is easy.

The three relations are **not three independent probes**. They are three points
on a single permissiveness knob, and they nest almost perfectly:

| relation | pairs called equal (of 13,968) | contained in the next |
|---|---|---|
| exact string match | 13 (0.1%) | 100% are also numeric-equal |
| numeric equivalence | 4,462 (31.9%) | 93.0% are also mutually entailing |
| bidirectional entailment | 10,893 (78.0%) | — |

They also share every other degree of freedom: the same 12,865 generations,
K = 5, temperature 0.7, first-line truncation, the same greedy transitive
clustering, and the same normalised-entropy map — which for K = 5 admits only
seven distinct divergence values.

So what the three relations establish is narrower than "robust from three
directions". It is: **across the full usable range of cluster permissiveness —
from merging 0.1% of pairs to merging 78% — step-level sampling divergence does
not rank step error on this corpus.** That is still the conclusion the chapter
needs, and it is worth more than a single relation. It is not three independent
confirmations.

## 3. Entailment is the most permissive relation, not the strictest

This was the surprise, and the intuition runs the other way: bidirectional
entailment is a *two-sided* test, so it should be harder to satisfy than a
one-sided numeric check. It is not.

- **86.6%** of the 27,936 directed pairs are judged entailment (24,199).
- 10,893 of 13,968 unordered pairs are **mutually** entailing (78.0%).
- 1,854 of 2,573 steps come out unanimous — **72%**, against 39% under numeric
  equivalence.

Mean divergence collapses from 0.4239 to 0.1205. Under this relation the
generator looks almost entirely self-consistent.

### Where the permissiveness comes from

Of the 13,968 unordered pairs, **8,625 (61.7%)** have no extractable number on
at least one side, so `numeric_equivalence` falls back to string equality and
calls them distinct. Those are the narration and algebra-rearrangement steps —
the same population §9.2 flags as 59% of what Qwen writes. The relations differ
most exactly where neither is well defined.

On the **5,343 pairs where both sides assert a number** the two relations agree
**85.5%** of the time.

### But the disagreement is concentrated exactly where it matters

A pooled agreement rate conceals the important asymmetry, and this is the
diagnostic that actually bears on whether the NLI model is fit for this job:

| comparable pairs | count | entailment calls mutually entailing |
|---|---|---|
| numeric equivalence **agrees** | 4,453 | 4,141 — **93.0%** |
| numeric equivalence **disagrees** | 890 | 463 — **52.0%** |

**On the pairs that carry an arithmetic disagreement, the reference relation
erases half of it.** For example:

```
[32]  80% of $40 = 0.80 * 40 = $<<0.80*40=32>>32
[40]  \[ 40 + (40 \times 0.80) = 40 + 32 = 72 \]
```

One step asserts the increase, the other the total; the NLI model calls them
mutually entailing. (The bracketed values are `_asserted_value` outputs, and
the second is itself a misread — `arithmetic_claims` parses `40 + (40 × 0.80)`
and stops at 40 rather than the trailing 72. The heuristic and the NLI model
are both imperfect here, which is why the table above is the evidence and this
example is only an illustration.)

> An MNLI model is checking whether two sentences are about the same thing, not
> whether they compute the same quantity. On arithmetic steps those come apart.

### Does that leniency change the answer? No — measured, not argued

The obvious follow-up is whether entailment's AUROC is being propped up or
dragged down by those 463 mis-merges. Re-clustering with an **arithmetic veto**
that undoes exactly them (changing 134 of 2,573 steps) gives:

| | test AUROC | all-steps AUROC | composite |
|---|---|---|---|
| entailment | 0.5625 | 0.5174 | 0.5805 |
| entailment + arithmetic veto | 0.5577 | 0.5210 | 0.5776 |

Paired difference −0.0048, 95% CI [−0.018, +0.011]. **No measurable direction
either way.** An earlier draft asserted that the merges would bias the relation
to look *better* than it is; that was an unmeasured a-fortiori, and it also
contradicted the mechanism claimed two paragraphs earlier. The measurement
replaces both.

## 4. Cost, and why the artifacts are committed

27,936 NLI pairs on 16 CPU cores: **19.6 hours** (the run logs
`primed in 1174.6 min`). The sustained rate averaged ~0.4 pairs/s and fell
through the run, because pairs are sorted short-to-long and the padding cost
rises. Three things made that survivable, all in the code rather than in a
comment:

- **`prime_many` batches across steps.** One step has ~11 distinct ordered
  pairs — nowhere near a batch. Priming per step measured under 0.2 steps/s and
  would not have finished.
- **The cache is keyed on `(context, premise, hypothesis)`.** An earlier version
  keyed on the text pair alone. Two fragments can entail each other under one
  question and not another, so that would have silently reused wrong verdicts.
- **Verdicts append to `runs/entailment_cache.jsonl` as they are produced.** A
  20-hour run without a checkpoint is the failure that already cost this
  project a GPU session once.

The cache and both feature files are committed, so re-clustering under a
different criterion — like the arithmetic veto in §3 — costs no GPU and no
NLI pass.

## 5. What this closes

Chapter 9 listed bidirectional entailment as the cheapest untested alternative
and noted the raw samples were committed so it could be run without a GPU. It
has been run:

| | |
|---|---|
| Did the reference relation find signal the cheap one missed? | **No** — 0.5625, CI [0.512, 0.614] |
| Is it worse than numeric equivalence? | **Unknown, and the question is not interesting** — the gap's CI spans zero |
| Is the ch. 7 result an artifact of the equivalence function? | **No** — across a 0.1%→78% permissiveness range the answer does not move |
| Do three relations give three independent confirmations? | **No** — they nest; see §2 |
| Is the reference relation the right one here? | **Doubtful** — it merges 52% of the pairs that assert conflicting numbers |

The relation that would actually fit this task — sensitive to the asserted
quantity *and* able to handle the 59% of steps carrying no arithmetic — remains
unbuilt.

See [FINDINGS-PIPELINE.md](FINDINGS-PIPELINE.md) for the result this extends.
