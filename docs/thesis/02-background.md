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
is the efficiency result that reshapes the design space: a small probe reading
frozen internal states matches far larger process reward models. Any proposal
whose contribution is "a better step score" has to beat a cheap probe, not a
large model. It also names the signal class this thesis did *not* test — see
Chapter 9.

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

## 2.7 Error propagation in reasoning chains

**The Hallucination Snowball** ([arXiv:2608.14588](https://arxiv.org/abs/2608.14588))
models propagation across agent handoffs as a Markov chain and measures escape
probabilities of 24.6%, 48.3% and 89.3% across successive boundaries. Those
figures give the decay constant used in this thesis's propagation model
(≈ 0.377).

**Sherlock** (Ro et al., *Reliable and Efficient Agentic Workflow
Execution*, [arXiv:2511.00330](https://arxiv.org/abs/2511.00330)) treats
verifier placement on a workflow DAG as a topology-plus-budget problem, using
structure, offline, before execution. It is the closest prior work to the
allocation question in Chapter 6, and the distinction — offline on a known DAG
by pure structure, versus online on a chain built as it is generated by a
calibrated score — is real but has to be argued rather than assumed.

**You et al.**, *Probabilistic Soundness Guarantees in LLM Reasoning Chains*
([arXiv:2507.12948](https://arxiv.org/abs/2507.12948)) evaluate a step given
verified premises, which is the local/global decomposition of §3.1 approached
from the other side.

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
| ConfSpec | escalate draft → target | reasoning step | draft confidence vs hand-set γ | none | no |
| ReProbe | nothing — scores only | reasoning step | probe on frozen internal states | none | no |
| Sherlock | verifier placement | workflow DAG node | fan-in, static, offline | none | topology-aware |
| PRM line | nothing — scores only | reasoning step | trained reward model | none | no |
| Snowball | boundary gate at handoff | agent handoff | deterministic numeric match | none | yes — Markov model |
| You et al. | nothing — scores only | step, given verified premises | probabilistic soundness | none | yes |
| **this work** | measurement, not a gate | **dependent step within a trajectory** | — | — | **yes — the object of study** |

The last row is deliberately not a system. The original intention was to
occupy that row with a method; the measurements in Chapters 4–7 are the reason
it does not.
