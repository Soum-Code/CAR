# 3. Related work

Chapter 2 covered the machinery this thesis uses. This chapter covers the work it is positioned against: the result that scooped
its original contribution, the contested status of the signal the design assumed would carry it, and the prior work closest to its
two central claims.

## 3.1 The scoop: Conformal Selective Acting

Khosravi & Huo, *Conformal Selective Acting*
([arXiv:2605.20270](https://arxiv.org/abs/2605.20270)) controls selective risk
with anytime-pathwise validity via a Ville-type e-process per candidate
threshold.

The relevant part is Appendix Theorem E.1. Subsample verifier labels by a
predictable `B_t ~ Bernoulli(π_t)` with `π_t ≥ π_min > 0`, and use the
importance-weighted increment `X̃_t(q) := (B_t/π_t)·X_t(q)`. Then the
supermartingale property survives, all anytime-valid guarantees hold verbatim,
and expected certification delay inflates by at most `1/π_min`.

That is the forced-exploration-plus-inverse-propensity scheme this
project independently designed for censored feedback, under a *stronger*
guarantee than the long-run rate we had hypothesised. The machinery is
implemented here (`src/car/conformal/adaptive.py`) and used, but it is cited to
CSA rather than claimed.

CSA also characterises the fully-censored failure mode — run the verifier only
on accepted rounds and "the controller stalls at the current threshold but
stays valid" — the `naive` update mode in this codebase.

What CSA does *not* do is the opening this thesis originally aimed at. It
operates on independent rounds: one query, one answer, act or abstain. Round
`t`'s verifier outcome does not alter the correctness distribution of round
`t+1`. In multi-step reasoning that independence is false by construction, and
that observation is what the project was rebuilt around.

## 3.2 Uncertainty estimation and its contested status at step level

Token-level signals — predictive entropy, maximum surprisal, mean
log-probability — are cheap, available from any generation pass, and measure
how surprising the model finds its own output.

**Semantic entropy** (Farquhar et al., Nature 2024) clusters independently
sampled outputs into meaning-equivalence classes by bidirectional entailment
and takes entropy over the cluster distribution. It is the strongest of the
sampling-based signals at whole-answer level.

Its status at *intermediate step* level is contested. Wen et al.,
*Embedding Perturbation may Better Reflect Intermediate-Step Uncertainty in LLM
Reasoning* ([arXiv:2602.02427](https://arxiv.org/abs/2602.02427)), argue that
perturbing embeddings reflects step-level uncertainty better than
sampling-based agreement does.

The original specification for this project weighted semantic divergence most
heavily of all its features. Chapter 8 measures the sampling-based signal
directly and finds it near chance, which is Wen et al.'s conclusion reached
from the other direction — and it leaves embedding perturbation as the
untested alternative.

## 3.3 Propagation, verifier placement, and self-correction

**The Hallucination Snowball** ([arXiv:2608.14588](https://arxiv.org/abs/2608.14588))
models propagation across agent handoffs as a Markov chain and measures escape
probabilities of 24.6%, 48.3% and 89.3% across successive boundaries. Those
figures give the decay constant used in this thesis's propagation model
(≈ 0.377).

**Sherlock** (Ro et al., *Reliable and Efficient Agentic Workflow
Execution*, [arXiv:2511.00330](https://arxiv.org/abs/2511.00330)) is the closest
prior work on the *system* side — closer than an earlier draft of this chapter
admitted. It asks three questions: which nodes deserve verification, *which
verifier to attach to each*, and how to pay for it. Its three mechanisms:

1. **Vulnerability-guided placement.** Error-prone nodes are found by
   counterfactual fault injection — faults are injected and their effect on the
   final output measured — not by a structural heuristic such as fan-in.
2. **Cost-optimal verifier selection.** A learned selector chooses among
   verifiers per node, motivated by an observation this thesis arrives at
   independently: *"verifier behavior varies significantly across tasks, and
   their accuracy-cost relationship is highly non-linear: higher cost does not
   necessarily translate to higher accuracy."*
3. **Speculative verification.** Downstream nodes execute while verification
   runs in the background, rolling back to the last verified output on failure.

It reports +18.3% accuracy over a non-verifying baseline, 48.7% lower execution
time than non-speculative verification, and 26.0% lower verification cost than
Monte-Carlo-search placement.

**The honest position on the overlap.** Sherlock's second mechanism occupies the
same ground as Chapter 6: verifiers are not interchangeable, and choosing among
them is the design decision. This thesis does not claim that observation as new.
What Chapter 6 adds is a *measurement of why* — detection decomposed on the
population deterministic checking provably cannot see, separating independence
from task specialisation as two separately necessary properties — and Chapter 8
adds the base-rate correction that makes a high-detection verifier net-negative
in deployment. Sherlock selects verifiers empirically by cost and observed
accuracy, over the whole node population; it does not decompose where that
accuracy comes from, nor measure it on the propagated-error subset.

The remaining distinctions are real — a known workflow DAG against a chain built
as it is generated, a learned cost model against a calibrated risk target — but
they are smaller than an earlier draft of this chapter claimed.

**ARES** (You et al., *Probabilistic Soundness Guarantees in LLM Reasoning
Chains*, [arXiv:2507.12948](https://arxiv.org/abs/2507.12948)) is the closest
prior work on the *claim* side: it tests C1 directly, and it deserves more than
a citation.

Their diagnosis is C1 stated independently: *"current LLM-based error detection
methods often fail to detect propagated errors because earlier errors can
corrupt judgments of downstream reasoning."* That is the local/global gap seen
from the detector's side rather than the data's.

Their remedy is structural. Autoregressive Reasoning Entailment Stability scores
each step **solely against previously-verified premises**, inductively, and emits
a calibrated soundness score with statistical guarantees rather than a binary
label. Across four benchmarks it reaches 72.1% Macro-F1 (+8.2), and on long
synthetic chains it detects **propagated errors at 90.3% F1 (+27.6)**.

**This is a result the thesis has to accommodate rather than dismiss.** It shows
the inherited-corruption population is not intrinsically invisible. It is
invisible to a verifier reading a step against the generator's *unverified*
context, as every verifier measured in Chapter 6 does. The correct
reading of C1 is therefore narrower than "no verifier can see these steps": a
verifier conditioned on the generator's own uncorrected prefix cannot, and
restricting to verified premises is the structural change that lifts the
ceiling.

Two things stop the numbers being directly comparable. ARES's 90.3% is F1 on
their own synthetic ClaimTrees corpus; Chapter 6's 0.9033 is a detection rate on
Math-Shepherd's arithmetic-blind subset, and the numerical coincidence is
exactly that. And ARES assumes a verified prefix exists to condition on, which
is precisely what a budgeted gate cannot supply — it leaves most steps
unverified by construction. Reconciling the two is the most promising direction
this thesis can point at, and Chapter 10 does.

**Huang et al.**, *LLMs Cannot Self-Correct Reasoning Yet* (ICLR 2024,
[arXiv:2310.01798](https://arxiv.org/abs/2310.01798)) is the negative control
for Chapter 6: a model asked to check its own reasoning without external
feedback does not reliably improve and sometimes gets worse. A second LLM with
the same weights is not a verifier.

## 3.4 Where this thesis sits

| work | gates what | on what unit | trigger | guarantee | propagation |
|---|---|---|---|---|---|
| CSA | release vs abstain | whole output, independent rounds | calibrated score | anytime selective risk | no |
| Kotte | abstain | whole output | nonconformity | CRC + impossibility bound | no |
| ReProbe | nothing — scores only | reasoning step | <10M-param probe on frozen internal states | none | no |
| Sherlock | placement **and verifier choice** | workflow DAG node | counterfactual fault injection + learned cost model | none | yes — its motivation |
| PRMs (Lightman; Math-Shepherd) | nothing — scores only | reasoning step | trained reward model | none | no |
| Snowball | boundary gate at handoff | agent handoff | deterministic numeric match | none | yes — Markov model |
| ARES (You et al.) | nothing — scores only | step, given **verified** premises | entailment stability | certified soundness score | **yes — and detects it** |
| **this work** | measurement, not a gate | **dependent step within a trajectory** | — | — | **yes — the object of study** |

The last row is deliberately not a system. The original intention was to
occupy that row with a method; the measurements in Chapters 5–7 are the reason
it does not.
