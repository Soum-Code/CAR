# 2. Background

## 2.1 Conformal prediction and what it guarantees

Split conformal prediction takes scores on a held-out calibration set, computes
a finite-sample-corrected quantile, and uses it as a threshold. Under
exchangeability of calibration and test scores, the acceptance rule
`{s ≤ q̂}` has marginal coverage at least `1 − α`. The correction —
`⌈(n+1)(1−α)⌉/n` rather than a plain `(1−α)` quantile — is what makes the
guarantee hold at finite n instead of only asymptotically.

Three properties are routinely overstated, and each of them matters later:

**It is not accuracy.** `1 − α` is a property of the acceptance rule, not of
the model. A rule that accepts nothing has perfect coverage.

**It is marginal.** Coverage is averaged over the distribution. Coverage on any
particular slice can be far below `1 − α`, and nothing in the procedure reports
which slices.

**It is coverage, not risk.** Coverage says what fraction of *correct* items
fall inside the acceptance region. Selective risk says what fraction of
*accepted* items are wrong. When the score ranks well the two move together;
when it does not, they come apart completely. Chapter 7 measures a case where
coverage holds and selective risk misses its target by a factor of three.

**Conformal risk control** (Angelopoulos, Bates, Fisch, Lei & Schuster, ICLR
2024, [arXiv:2208.02814](https://arxiv.org/abs/2208.02814)) generalises the
guarantee from coverage of a set to the expectation of a bounded monotone loss,
which is the right shape for a verification gate: the loss is "a bad step was
accepted."

**Adaptive conformal inference** relaxes exchangeability by updating the
threshold online from observed errors, `q_{t+1} = q_t + γ(α − err_t)`. It buys
robustness to distribution shift at the cost of only long-run rather than
finite-sample validity.

**Conformal prediction beyond exchangeability** (Barber, Candès, Ramdas &
Tibshirani, *Annals of Statistics* 2023,
[arXiv:2202.13415](https://arxiv.org/abs/2202.13415)) is the general treatment.
It relaxes both assumptions conformal prediction rests on: **weighted quantiles**
give robustness to distribution drift, and a **randomization technique** admits
model-fitting algorithms that do not treat data points symmetrically — for
instance one weighting recent observations more heavily. The methods are
provably robust, losing substantially less coverage under drift while retaining
the standard guarantee when the data really are exchangeable.

It is worth being precise about what this does and does not reach, because
Chapter 9 names the gap as this thesis's open problem. Barber et al. handle
*drift* — the distribution moves, for reasons outside the procedure — and
*asymmetry* in the fitting algorithm. Neither is the situation here. In a
verification gate the exchangeability violation is **endogenous**: accepting a
wrong step at `t` changes the distribution of steps `t+1 … T`, because they are
generated conditioned on it. The procedure's own decisions move the
distribution. Their weighted-quantile machinery is the closest available
starting point and is not a solution.

## 2.2 The impossibility bound

Kotte, *When Can Conformal Risk Control Certify LLM Outputs?*
([arXiv:2606.29054](https://arxiv.org/abs/2606.29054)) proves the result that
should be run before any experiment rather than after.

**Proposition 3.** When the base risk `μ` exceeds the target `α`, any
distribution-free method must abstain on — for our purposes, verify — at least

```
(μ − α) / (1 − α)
```

of examples. It is a closed form, checkable in advance, and it depends on
nothing but two numbers.

For a verification gate this is a hard floor on the verification rate, and it
is not a small one. At a measured step error rate of μ = 0.3908, a target of
α = 0.10 charges 32.3% of all steps as an entry fee before any method is
admissible at all. This thesis uses the floor as an evaluation instrument: it
appears as a reference line on the feasibility tables in Chapter 8, and
`configs/default.yaml` refuses to start a run whose budget sits below it.

Kotte also reports that "hard NER/QA/CLS configurations are uncertifiable at
α = 0.10; relaxing to α = 0.30–0.40 unlocks practical certification," which is
the direct reason this project's α was changed from the originally specified
0.10.

## 2.3 The scoop: Conformal Selective Acting

Khosravi & Huo, *Conformal Selective Acting*
([arXiv:2605.20270](https://arxiv.org/abs/2605.20270)) controls selective risk
with anytime-pathwise validity via a Ville-type e-process per candidate
threshold.

The relevant part is Appendix Theorem E.1. Subsample verifier labels by a
predictable `B_t ~ Bernoulli(π_t)` with `π_t ≥ π_min > 0`, and use the
importance-weighted increment `X̃_t(q) := (B_t/π_t)·X_t(q)`. Then the
supermartingale property survives, all anytime-valid guarantees hold verbatim,
and expected certification delay inflates by at most `1/π_min`.

That is precisely the forced-exploration-plus-inverse-propensity scheme this
project independently designed for censored feedback, under a *stronger*
guarantee than the long-run rate we had hypothesised. The machinery is
implemented here (`src/car/conformal/adaptive.py`) and used, but it is cited to
CSA rather than claimed.

CSA also characterises the fully-censored failure mode — run the verifier only
on accepted rounds and "the controller stalls at the current threshold but
stays valid" — which is the `naive` update mode in this codebase.

What CSA does *not* do is the opening this thesis originally aimed at. It
operates on independent rounds: one query, one answer, act or abstain. Round
`t`'s verifier outcome does not alter the correctness distribution of round
`t+1`. In multi-step reasoning that independence is false by construction, and
that observation is what the project was rebuilt around.

## 2.4 Process reward models and step labels

Outcome reward models score a whole solution; process reward models score each
step. Lightman et al. established that process supervision outperforms outcome
supervision on mathematical reasoning.

**Math-Shepherd** (Wang et al., ACL 2024,
[arXiv:2312.08935](https://arxiv.org/abs/2312.08935)) is the dataset this
thesis measures on, and its construction is what makes the measurement
possible. It contains GSM8K and MATH solutions generated by Mistral-7B-SFT,
with each step carrying an automatic label:

- `+` — this step has the potential to lead to the correct answer
- `-` — it does not

The labels come from Monte-Carlo rollout ("hard estimation": `+` if any
completion from that prefix reaches the gold answer), not from human
annotation. That has two consequences used throughout. First, the label is
**global** correctness — a step inherits `-` from a corrupted premise even when
the step itself is impeccable, which is exactly the quantity this thesis needs
and cannot otherwise obtain at scale. Second, the labels are optimistic
estimates of "leads to a correct answer," not proofs: a lucky wrong step can be
labelled `+`.

Their own worked example is the clearest statement of the phenomenon:

```
Step 2: 13 x 8 / 13 = <<13*8/13=6>>6      -    arithmetically WRONG (= 8)
Step 3: 78 - 6 = <<78-6=72>>72            -    arithmetically CORRECT, inherits
Step 4: 72 - 20 = 52                      -    arithmetically CORRECT, inherits
```

Meanwhile GSM8K's inline `<<expr=result>>` calculator annotations let **local**
validity be checked deterministically, with no model and no judge. Having both
signals on the same steps is what turns the local/global gap from a thought
experiment into a measurement.

**ReProbe** (Ni et al., [arXiv:2511.06209](https://arxiv.org/abs/2511.06209))
is the efficiency result that reshapes the design space. A transformer probe of
fewer than 10M parameters, reading the frozen internal states of the generator
itself, *outperforms* PRMs up to 150x larger and stays competitive with PRMs up
to 810x larger. Its labels can come from a larger model or be generated
self-supervised by the original model, so it needs neither the human annotation
nor the Monte-Carlo consensus filtering PRMs are built on.

Two details matter for this thesis specifically. First, the probe's advantage is
largest **out of domain** — on planning and StrategyQA — while the strongest
PRMs *reach parity with it in domain, on MATH and GSM8K*. On the benchmark used
here, a probe would therefore be expected to land near the PRM rather than above
it. Second, Ni et al. report that combining probe and PRM beats either alone,
which suggests the two capture different aspects of step quality and points at
hybrid verifiers rather than a replacement.

Any proposal whose contribution is "a better step score" now has to beat a probe
costing 10M parameters, not a large model. This is also the signal class this
thesis did *not* test — see Chapter 9.

## 2.5 Uncertainty estimation and its contested status at step level

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
heavily of all its features. Chapter 7 measures the sampling-based signal
directly and finds it near chance, which is Wen et al.'s conclusion reached
from the other direction — and it leaves embedding perturbation as the
untested alternative.

## 2.6 The selective-labels problem

Lakkaraju, Kleinberg, Leskovec, Ludwig & Mullainathan, *The Selective Labels
Problem* (KDD 2017) formalises the difficulty that arises when outcomes are
observed only for items a policy acted on. A gate that verifies only the steps
it flags observes labels only where it flags, so the label distribution is
conditioned on the gate's own decisions. Naive updating from those labels does
not converge to the right threshold.

Forced exploration with inverse-propensity weighting is the standard remedy and
is what CSA's Theorem E.1 formalises for this setting. Two ordering constraints
follow and are enforced in the implementation:

1. The exploration coin must be drawn **before** the gate decision and
   independently of the score, or the propensity of observing a label is not
   known and the correction is invalid.
2. An INSUFFICIENT verdict must yield **no label**, not a label of "no error."
   Recording "I could not check" as "it was fine" is a lie the calibrator will
   act on.

The second constraint has a consequence Chapter 7 exploits: a verifier that
*misses* an error does not usually announce the miss, it reports the step as
sound. So a low-reach verifier does not merely help less — it feeds the
calibrator systematically wrong labels.

## 2.7 Propagation, verifier placement, and self-correction

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
same ground as Chapter 5: verifiers are not interchangeable, and choosing among
them is the design decision. This thesis does not claim that observation as new.
What Chapter 5 adds is a *measurement of why* — detection decomposed on the
population deterministic checking provably cannot see, separating independence
from task specialisation as two separately necessary properties — and Chapter 7
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
context — which is what every verifier measured in Chapter 5 does. The correct
reading of C1 is therefore narrower than "no verifier can see these steps": a
verifier conditioned on the generator's own uncorrected prefix cannot, and
restricting to verified premises is the structural change that lifts the
ceiling.

Two things stop the numbers being directly comparable. ARES's 90.3% is F1 on
their own synthetic ClaimTrees corpus; Chapter 5's 0.9033 is a detection rate on
Math-Shepherd's arithmetic-blind subset, and the numerical coincidence is
exactly that. And ARES assumes a verified prefix exists to condition on, which
is precisely what a budgeted gate cannot supply — it leaves most steps
unverified by construction. Reconciling the two is the most promising direction
this thesis can point at, and Chapter 9 does.

**Huang et al.**, *LLMs Cannot Self-Correct Reasoning Yet* (ICLR 2024,
[arXiv:2310.01798](https://arxiv.org/abs/2310.01798)) is the negative control
for Chapter 5: a model asked to check its own reasoning without external
feedback does not reliably improve and sometimes gets worse. A second LLM with
the same weights is not a verifier.

## 2.8 Where this thesis sits

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
occupy that row with a method; the measurements in Chapters 4–7 are the reason
it does not.
