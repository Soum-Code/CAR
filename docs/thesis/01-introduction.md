# 1. Introduction

## 1.1 The problem

A language model asked a multi-step question produces a chain of intermediate
claims and then an answer. Any one of those claims can be wrong, and a wrong
claim early in the chain is not merely a local defect: everything after it is
generated conditioned on it. The practical question is where to spend a limited
verification budget.

The obvious design, and the one this project set out to build, has three parts:

1. **A score.** Attach an uncertainty estimate to each step — token entropy,
   surprisal, disagreement across samples.
2. **A calibration.** Turn that score into a threshold with a distribution-free
   guarantee, so the system can claim a bound on the risk of what it lets
   through rather than a hand-tuned cutoff.
3. **A gate.** Verify the steps above the threshold, subject to a budget; let
   the rest through.

Each part has substantial literature behind it. Uncertainty estimation for LLMs
is an active area; conformal prediction supplies distribution-free guarantees;
process reward models supply step-level verifiers. Assembling them is a natural
thesis.

It does not work. This document is an account of why, and the reasons turn out
to be more interesting than the system.

## 1.2 The distinction the whole thesis rests on

A reasoning step can fail in two ways that are not usually separated, and
separating them is what makes everything else measurable:

```
global_correct(t)  =  local_valid(t)  AND  NOT premise_corrupt(t)
```

**Local invalidity** — the step does not follow from its own premises.
`47 × 3 = 131` is wrong in any context whatsoever, and a calculator catches it
every time.

**Inherited corruption** — the step follows perfectly from its premises, and a
premise is false. *Aristotle died in 1850, so he could have used a laptop* is
impeccable reasoning to a false conclusion. No amount of checking the step
itself will reveal anything.

A verifier reports the first. It is a function of the step and, at most, a
bounded window of context; it has no access to the truth of the premises unless
it can independently establish them. Conformal machinery calibrates whatever
the verifier reports, so it calibrates local validity.

The question this thesis asks first is: **how much of the failure does that
leave out?** The answer, measured on 93,129 real steps, is most of it.

## 1.3 What was proposed, and what happened to it

The project began as a method proposal: calibrated uncertainty gating with
adaptive conformal calibration under censored feedback, weighted by each step's
downstream influence. Two rounds of literature checking and stress-testing
removed the method claims one at a time.

| original claim | outcome |
|---|---|
| Adaptive conformal under censored feedback is novel | **scooped** — Conformal Selective Acting, Thm E.1, publishes Bernoulli subsampling with 1/π importance weighting under a stronger anytime guarantee |
| Composite uncertainty is the key signal | **crowded, then refuted** — a small probe on internal states matches far larger PRMs (Ni et al., ReProbe); measured here at AUROC 0.5589 |
| Semantic entropy at intermediate steps is the key signal | **refuted** — measured here at AUROC 0.5740 |
| Influence-weighted allocation | **refuted** — lost to plain uniform in five separate tests |
| "Verify early beats verify late" | **refuted** — front-loading is the worst shape at every scope > 0 |
| StrategyQA as the primary benchmark | **wrong choice** — 72.9% of its dependency graphs are one hop deep |
| α = 0.10 as a working risk target | **infeasible** — charges a 32% verification entry fee before any method is admissible |

What survived is not a method. It is a set of measurements, and they compose
into a claim about the design space that is stronger than the system would have
been.

This history is kept in full, in `docs/POSITIONING.md` and
`docs/FINDINGS-PROPAGATION.md`, because the negative results are the
contribution and a reviewer who finds the scoop independently will discount
everything else.

## 1.4 Contributions

**C1. The certified quantity is not the quantity of interest.** Measured on
Math-Shepherd's 93,129 labelled steps: within wrong-answer solutions, local
error is 0.1708 and global error 0.7106, so **78.5% of globally-wrong steps are
arithmetically perfect**. Controlling local selective risk at level α bounds
nothing about the answer. The claim is about verifiers reading the generator's
*unverified* context — §2.7 records one structural change, scoring against
previously-verified premises, that does lift the ceiling. (Chapter 4)

**C1b. The gap widens on a stronger generator.** Re-measured on
Qwen2.5-7B-Instruct at 80.0% GSM8K accuracy against Mistral-7B-SFT's ~45%, the
figure rises to **90.4%** with disjoint Wilson intervals. The stronger model
halves its arithmetic slips without halving its inherited corruption, so a
deterministic verifier gets *less* useful as generators improve. (Chapter 4)

**C2. Corruption is close to absorbing.** After the first globally-bad step,
95.9% of subsequent steps remain bad, and **0 of 25,971 Math-Shepherd solutions
ever recover**. On Qwen the figures are 66.4% and 6 of 500: still strongly
absorbing, but "near-absorbing" is a property of the generator and does not
transfer unqualified. (Chapter 4)

**C3. Verifier reach is the controlling variable, and it is semantic.**
Measured across four verifier classes on the population arithmetic provably
cannot see: step-local arithmetic 0.0000, unbounded arithmetic lookback 0.1999,
same-model critic 0.0000, independent general judge 0.2283, task-specialised
PRM **0.9033**. Reach requires independence from the generator *and* task
specialisation; neither alone suffices. (Chapter 5)

**C4. "Verify early" is false, and the useful structural signal is ancestor
count.** Front-loading is the worst allocation at every scope > 0, replicated
on chains, five synthetic DAG families and real graphs from two benchmarks. The
last defence — that early steps are intrinsically harder — is closed by
measurement: corr(position, local error) = **+0.950**, error rate doubling from
11% at step 1 to 22% at step 8. (Chapter 6)

**C5. The risk target is constrained before any method is chosen.** With
measured μ = 0.3908, Kotte's impossibility bound charges a 32.3% verification
entry fee at α = 0.10. The floor is a property of the generator, not the task:
on Qwen, μ = 0.1221 and α = 0.20 carries no floor at all. (Chapter 8)

**C6. StrategyQA cannot exhibit the phenomenon it is used to study.** All 2,272
annotated decompositions: mean depth 2.30, **72.9% exactly one hop**, and only
11.2% of steps have any non-terminal descendant. A step can only corrupt
downstream reasoning if downstream reasoning exists. (Chapter 8)

**C7. Derived dependency graphs are 94.4% correct, and hand-validating them
found a systematic bug.** An audit of 50 stratified GSM8K graphs exposed an
operand-extraction fault that silently deleted the dependency edge of *every
subtraction* in the corpus. Fixing it moved mean depth 2.54 → 2.79 and
overturned a published conclusion. (Chapter 6)

**C8. The assembled gate does not control risk, and the reason is the signal.**
Generator uncertainty does not rank global step error (AUROC 0.5589 token-level,
0.5740 semantic, 0.5742 combined, against 0.8668 on synthetic features with
genuine separation). Split conformal consequently holds coverage while missing
selective risk by 3× at every α that binds, and the highest-reach verifier is
net-negative at its measured operating point.

An oracle score separates cause from consequence: it takes selective risk
0.1554 → **0.0885** and projected accuracy 0.7912 → **0.9780** using a third of
the calls, so the verifier result is downstream of the score rather than a
separate defect. It also still misses α = 0.05 by 1.8×, which locates the
second and genuinely independent bottleneck: the budget. (Chapter 7)

**C9. A better signal exists, and is still not enough.** A logistic probe on the
generator's own frozen hidden states reaches **AUROC 0.6968** against 0.5742 for
everything else measured — so the failure is not that step-level uncertainty is
unreadable. It improves selective risk at every α on fewer calls, and still
misses α = 0.05 by 2.9× and leaves the task PRM net-negative. The probe is
trained on 670 steps with a learning curve that has not plateaued, so this is a
floor on the signal class, not a ceiling. (Chapter 7.6)

## 1.5 The thesis statement

> Conformal machinery applied to multi-step reasoning certifies whether a step
> is **locally valid** — that is what a verifier reports and therefore what can
> be calibrated. In multi-step reasoning that is not the quantity of interest,
> and the gap does not close by improving the generator, widening the
> verification window, or adding the uncertainty signal the design assumed
> would carry it. Selective verification of LLM reasoning fails at the signal,
> at the calibration, and at the verifier, and repairing any one of them is not
> sufficient.

## 1.6 Structure

Chapter 2 covers the background: conformal prediction and risk control, the
impossibility bound, process reward models, the selective-labels problem, and
prior work on error propagation. Chapter 3 sets out the framework — the
local/global decomposition, the propagation model, verifier reach as a
parameter, and the measurement apparatus.

Chapters 4 through 7 are the measurements, in the order the argument needs
them: the size of the gap, the reach of verifiers, the shape of the optimal
allocation, and the behaviour of the assembled system. Chapter 8 covers
feasibility and the benchmark critique. Chapter 9 states the limitations
honestly, collects the negative results, and concludes.

Three sections are worth reading even if the rest is skipped: §4.4 (the gap
widens on a better generator), §5.4 (three tokenizer faults that each produced
a confident wrong answer), and §7.4 (why the best verifier available loses
accuracy).
