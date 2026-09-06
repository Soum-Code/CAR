# 5. Verifier reach

> Reproduce: `python scripts/exp_verifier_scope.py`,
> `python scripts/gpu_semantic_scope.py --analyse runs/semantic_scope_prm.json`

Chapter 4 established that most globally-wrong steps are locally valid. The
design question that follows is whether a *better* verifier can see them, and
what "better" means. This chapter measures reach across four verifier classes
and finds it spans nearly the whole unit interval.

## 5.1 Reach is not a window

The first hypothesis is that a verifier fails on inherited corruption because
it looks at too little context. If so, widening the lookback should help.

`src/car/verification/lookback.py` reframes reach as a controllable window: a
`LookbackVerifier(k)` re-checks the arithmetic of the current step and the `k`
steps before it, and detects corruption when it finds a local error anywhere in
that window. On 35,535 real Math-Shepherd steps:

| lookback k | scope | steps detected | cost per call |
|---|---|---|---|
| 0 (step-local) | **0.0000** | 1 | 1.00 |
| 1 | 0.1197 | 4,255 | 1.85 |
| 2 | 0.1690 | 6,004 | 2.42 |
| 3 | 0.1880 | 6,682 | 2.73 |
| 5 | 0.1983 | 7,046 | 2.96 |
| ∞ (unbounded) | **0.1999** | 7,102 | 3.01 |

Three things are visible. Step-local arithmetic detects **nothing** — by
construction, since the population is steps whose own arithmetic is correct.
Widening the window has sharply diminishing returns: k = 1 buys 0.12, k = 2 to
∞ buys another 0.08 between them. And the cost triples while doing so.

**The ceiling is structural.** 28,434 of the 35,535 steps — 80.0% — have *no
upstream arithmetic error at all*. The mistake is in the setup, not the
calculation: a wrong quantity was used, or the wrong operation applied, and
every individual computation is correct. No amount of arithmetic checking, at
any window size, can see them.

The decay curve is sharp rather than gradual. Corruption originating `d` hops
upstream is detected with probability 1 by a window of `k ≥ d` and probability
0 otherwise, and the distance distribution falls off fast:

| distance from corruption origin | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| steps | 4,254 | 1,749 | 678 | 259 | 105 |

So the window is not really buying reach; it is buying coverage of a
short-tailed distance distribution, and the tail is not where the missing 80%
lives.

That defines the remaining question precisely:

> On the 28,433 steps arithmetic provably cannot see, what scope does a
> *semantic* verifier achieve?

## 5.2 The four arms

Three verifier classes were run on that arithmetic-blind population, plus a
control group of steps that are locally valid **and** globally correct — without
which detection rate alone is meaningless, since a verifier that flags
everything scores TPR 1.0 and is useless.

1,500 arithmetic-blind steps + 750 controls, Kaggle P100.

| verifier | independent of generator? | task-trained? | scope | false alarm | net |
|---|---|---|---|---|---|
| arithmetic, step-local | yes | — | 0.0000 | — | 0.0000 |
| arithmetic, unbounded lookback | yes | — | 0.1999 | — | 0.1999 |
| same-model critic (`peiyi9979/mistral-7b-sft`) | **no** | no | 0.0000 | 0.0000 | 0.0000 |
| independent judge (Qwen2.5-7B-Instruct) | yes | no | 0.2283 | 0.0200 | 0.2083 |
| task PRM (Math-Shepherd-7B) | yes | **yes** | **0.9033** | 0.0987 | **0.8047** |

**The same-model critic detects zero errors.** It is the model that wrote these
solutions, asked whether they are sound, and it approves everything — its
validation separation on known `+`/`-` steps is 0.0000. This is Huang et al.
(ICLR 2024) measured directly on step-level data: the parameters that produced
the error are being asked to catch it.

**A general independent judge barely beats arithmetic.** Qwen2.5-7B-Instruct is
a different model family with no exposure to this task's labels, and it reaches
0.2283 against arithmetic's 0.1999.

**Only the task-specialised, independent verifier closes the gap.** The
Math-Shepherd PRM reaches 0.9033 at a 9.87% false-alarm rate.

> Reach is not how far back you look. It requires a verifier that is
> independent of the generator **and** specialised for the task. Neither alone
> is sufficient — the same-model critic has specialisation without
> independence and scores zero; the general judge has independence without
> specialisation and scores 0.23.

The original specification listed calculator, retrieval and sandbox as
interchangeable reliability mechanisms. They are not. They span 0.00 to 0.90
scope, and that axis is absent from the specification entirely.

## 5.3 Why there is no retrieval arm

GSM8K's premises are the problem statement, not an external corpus, so
retrieval-plus-entailment has no meaning here. A genuine retrieval arm needs
StrategyQA's evidence paragraphs — but Chapter 8 shows StrategyQA is 72.9% one
hop deep, so the arithmetic-blind population barely exists there. The
measurement is not available on either benchmark, and saying so is more honest
than substituting a proxy.

## 5.4 Three measurement faults that each produced a confident wrong answer

This section is longer than the result it protects, deliberately. The first
completed run of §5.2 reported the **opposite** conclusion — negative net scope
— and it was entirely an artifact.

### The symptom

The PRM flagged **94.9% of the control group**: steps carrying Math-Shepherd's
own `+` label, i.e. the model's own training signal. A model that cannot
recognise its own training labels is not measuring anything.

### Fault 1 — wrong output logits

The `+`/`-` candidate token ids were derived with `tok.encode`. Locally
(transformers 5.8) that yields `▁+` = 648 and `▁-` = 387, the tokens the PRM
was trained on. On Kaggle the same call yielded `+` = 28806 and `-` = 28733 —
the same characters *without* the SentencePiece word-boundary marker, and
therefore entirely different embeddings. The PRM was being read at vocabulary
indices it had never been trained to emit at, and separated its own training
labels by 0.0108.

Fixed by hardcoding the documented ids and verifying they decode to the
expected pieces, so a tokenizer change fails loudly instead of silently
producing noise.

### Fault 2 — wrong input positions

The `ки` step-tag id is an *input* id, used to locate the positions at which to
read the score. It was hardcoded to the reference 12902; Kaggle's tokenizer
encoded it differently, the position mask matched nothing, and every score came
back NaN.

The distinction matters and is now enforced in code: output-side ids must match
*training*, input-side ids must match *whatever this tokenizer actually
produces*. `resolve_step_tag_id` resolves the tag in context from the live
tokenizer.

### Fault 3 — right padding in batched generation

The judge arms had their own instance of the same class of bug. Right padding
in batched decoder-only generation inserts pad tokens between the prompt and
the continuation, so the model generates from padding. It surfaced only as a
warning. Fixing it moved the Qwen judge result from an untrustworthy 0.1933 to
0.2283.

### The gate that catches this class of fault

A validation step now scores 400 steps with known Math-Shepherd labels *before*
the real measurement, and aborts below 0.15 separation. It caught three
successive broken configurations before the fourth passed at **0.5788**.

Without it, this project would have published a confident negative result
produced entirely by tokenizer mismatch.

> The general lesson, and it applies well beyond this chapter: **on a borrowed
> model, the harness must prove it can reproduce that model's known behaviour
> before any novel number from it is believed.** A validation gate on known
> labels costs one batch and is the cheapest insurance in the project.

The same-model critic's failure is the one case where a gate failure is the
*finding* rather than a bug — it fails the 0.15 separation test because it
genuinely cannot separate its own errors, which is what Huang et al. predict.
It is recorded as an uninformative verifier, not as a harness fault, and the
distinction is defensible only because the PRM passed the same gate on the same
data.

## 5.5 Scope decay with propagation distance

The propagation model assumes scope fades with distance from the corruption
origin. Measuring it required fixing an instrumentation bug first: the initial
implementation measured distance to the *nearest incorrect ancestor*, which is
always 1, making the decay term inert. Tracking the corruption **origin**
instead makes the quantity meaningful.

The measured decay is consistent with the 0.377 fitted from Singh & Pawar's
escape probabilities, but the sample thins quickly with distance and the
estimate is not precise enough to quote as a headline.

## 5.6 What this chapter establishes for the rest

Chapter 6 asks how to allocate a budget across steps; Chapter 7 runs the
assembled system. Both need scope as a *parameter*, not as whatever verifier
happens to be wired in. `ScopedVerifier.from_measured` turns this chapter's
table into a component, so the later chapters can be run at each measured
reach and the results attributed to it.

The number that turns out to matter most in Chapter 7 is not the 0.9033. It is
the **0.0987** next to it.
